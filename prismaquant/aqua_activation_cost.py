"""AQUA-AURA: price the ACTIVATION side and merge it into a weight cost.

WHY THIS IS A SEPARATE STAGE
----------------------------
The allocator's cost was weight-only. Choosing NVFP4 does not just round the
weights to 4 bits -- on a lane that serves it fused, it is W4A4 and commits the
layer's ACTIVATIONS to 4 bits at serve time. FP8 commits them to 8. BF16 leaves
them alone. A weight-only surrogate is structurally blind to that difference
(NVFP4 and NVFP4A16 render weights bit-identically), so on such a lane the DP
was buying 4-bit formats at a discount to their true cost.

"ON A LANE THAT SERVES IT FUSED" IS LOAD-BEARING AND WAS ONCE MISSING HERE.
Whether activations are quantized is a property of the RUNTIME, not of the
format, and one lane can answer differently per FAMILY. Gridbook's CB runtime:

  * ``FP8_CB_*`` is genuinely W8A8 -- ``linear.py`` feeds quantized ``xq`` with
    per-token dynamic scales into ``native_cutlass_scaled_mm``, and ``moe.py``
    declares ``_FP8_GROUPED_CONTRACT = "fp8_per_token_dynamic"``. Real A-side.
  * ``NVFP4_CB_*`` is NOT -- it decodes to BF16 and runs a BF16 GEMM ("the
    exact native BF16 bridge") unless a PROCESS-GLOBAL env selector picks a
    fused mode, and every CB gate and gold serve leaves those unset. A-side
    exactly zero.

Charging the NVFP4 family a phantom A4 is not conservative, it is wrong in a
direction that costs bytes: it made FP8 look relatively cheap and the DSv4-Flash
92 GB body bought it, going from 96.8% ``nvfp4_cb`` (K16 bulk) to 25.4%
``FP8_CB`` with the bulk rung crushed K16 -> K12 -- paying weight bits to escape
a cost it never incurs. Keeping the REAL FP8 term is what stops the correction
from overshooting the other way. ``activation_dloss_table`` therefore REQUIRES
the lane's ``served_activation_quantization.executes`` (glob patterns over
format names) and refuses to guess it.

This stage exists as its own step, rather than inside the cost stage, because
the A-side is genuinely separable ON ITS DEFAULT BASIS:

  * It needs NO render. ``activation_dloss`` reads the DENSE weight (as
    ``W[o,j]^2``), the card's ``g_sq_sum``, and the format's activation grid.
    The render basis never enters, so the number is identical whether the
    W-side was rendered with RTN or with the full GPTQ+JSO production recipe.
  * It therefore costs one streaming pass over the checkpoint and a row-blocked
    ``W^2 @ var`` per unit -- minutes, not the hours a render costs -- and it can
    be recomputed against any existing cost artifact without rebuilding it.

RENDER-CONDITIONED A-SIDE (OPT-IN, RESEARCH)
--------------------------------------------
``PRISMAQUANT_AQUA_ACT_WEIGHT_BASIS`` (or ``--act-weight-basis``) swaps ``W``
for the format's own rendered ``W_hat_f`` in that reduction. Unset it is a
byte-identical no-op and every bullet above still holds.

Why it is worth having despite costing the separability: the local error of a
quantized Linear decomposes as ``W_hat x_hat - W x = dW x + W_hat dx``, so the
activation term belongs on ``W_hat_f``, not on ``W``. Evaluating it there
absorbs the ``dW dx`` contribution. What stays dropped on BOTH bases: the
cross-correlation between ``dW x`` and ``W_hat dx``, every downstream nonlinear
interaction, and the routing interaction on a routed MoE (a perturbed input can
change which expert a token sees, and no second-moment term sees a route flip).

Two bases are offered and neither is ever a fallback for the other, because a
silent basis mix is the rendering confound the one-cache rule exists to prevent:

  * ``rtn`` -- the registry's own ``quantize_dequantize``, in process, no cache.
  * ``compensated`` -- the GPTQ+JSO tensors from a ``ProductionWeightCache``,
    i.e. the bytes the export will actually ship. Requires ``--production-cache``
    and refuses without it. A (unit, format) the cache does not hold is a HOLE,
    counted per format and reported; it is never quietly re-rendered on the
    other basis.

That separability is also the reason the term matters MORE than its size on an
RTN basis suggests: GPTQ and JSO shrink the W-side substantially and do nothing
whatever to the A-side. Measured on Qwen3.8-27B, production rendering cut
NVFP4's median W-side to 0.13x its RTN value while the A-side was unchanged, so
on the shipping render the activation term is several times the weight term for
the median Linear.

WHAT IT DOES NOT DO
-------------------
It does not choose formats and it does not rewrite an allocation. It writes one
number per (unit, format) into the cost rows and lets the DP do what it already
does. Hand-promoting the units it flags would be the post-allocator rewrite the
platform vetoes (principle 1): if the allocator picks something bad, the cost
model is what is wrong.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import pickle
import time

import numpy as np

from .allocator_candidates import ACT_DLOSS_KEY

#: Return the CUDA pool to the OS once it has reserved this much. On GB10's
#: UNIFIED memory a reserved CUDA block IS host RAM, so it competes with the
#: numpy side rather than living in a separate budget.
CUDA_RESERVED_DRAIN_GIB = 8.0


def log(m: str) -> None:
    print(f"[aqua-cost] {m}", flush=True)


def build_weight_resolver(weight_map: dict, profile=None) -> dict:
    """Map a card unit name to its checkpoint tensor key.

    These differ, and not cosmetically. A card's unit names come from the module
    tree the probe walked (``model.layers.7.mlp.gate_proj``), while a multimodal
    checkpoint nests the text tower one level deeper
    (``model.language_model.layers.7.mlp.gate_proj.weight``). Matching naively
    resolves a handful of units and silently prices almost nothing, which reads
    as "the menu is unavailable" rather than as a name mismatch.

    So index every ``.weight`` key under its own base name AND a de-nested
    alias. The alias is registered with ``setdefault`` so a real ``model.layers``
    key stays authoritative over one.

    THAT IS NOT ENOUGH IN GENERAL, and the de-nesting alias above is the tell:
    it is one architecture's rename hardcoded here. Some checkpoints rename the
    LEAVES too, not just the path. DSv4-Flash stores
    ``model.layers.0.mlp.experts.0.down_proj`` as
    ``layers.0.ffn.experts.0.w2.weight`` -- no ``model.`` prefix, ``mlp``
    renamed to ``ffn``, and the Mixtral ``w1/w2/w3`` leaf convention. No amount
    of path aliasing recovers that, and the failure is silent: resolution drops
    to 0/33325, every unit becomes unpriced, and because ``cost_entry_act_dloss``
    defaults to 0.0 the DP cannot distinguish "unmeasured" from "free".

    So when a ``profile`` is supplied, invert its own
    ``checkpoint_to_live_name`` over the weight map. That mapping is the
    architecture's declared, tested source-of-truth -- the same one the probe,
    the cost stage and the exporter resolve names through -- rather than
    another rename guessed here (principle 2: no heuristics when an explicit
    exists). Note the *forward* direction is not usable for this:
    ``source_tensor_name`` rewrites the path but not the leaf, so it yields
    ``layers.0.ffn.experts.0.down_proj``, which is not a key.
    """
    # A packed routed-expert parameter is a bare 3-D tensor with NO ``.weight``
    # suffix -- ``...mlp.experts.gate_up_proj``, not ``....gate_up_proj.weight``
    # -- because it is an ``nn.Parameter`` on the experts module rather than an
    # ``nn.Linear``. Indexing only ``.weight`` keys therefore drops every expert
    # tensor, which on an A3B is 94% of the parameters. They resolved to nothing
    # and were reported as "units without an act price" -- an accurate count of
    # a silent hole.
    def _bases(key: str):
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        yield base
        if ".language_model." in base:
            yield base.replace("model.language_model.", "model.")

    idx: dict[str, str] = {}
    for key in weight_map:
        own = next(iter(_bases(key)))
        idx[own] = key
    for key in weight_map:
        for alias in list(_bases(key))[1:]:
            idx.setdefault(alias, key)
    if profile is not None:
        mapped = 0
        for key in weight_map:
            if not key.endswith(".weight"):
                continue
            try:
                live = profile.checkpoint_to_live_name(key)
            except Exception:
                continue
            if live and live.endswith(".weight"):
                # setdefault: a key that already indexes under its own name
                # stays authoritative; this only ADDS reachable aliases.
                if idx.setdefault(live[: -len(".weight")], key) is key:
                    mapped += 1
        log(f"profile {type(profile).__name__} contributed {mapped} "
            f"checkpoint-name aliases")
    return idx


#: Dtypes a checkpoint tensor may have and simply BE the dense weight. Anything
#: else is an encoding that must be decoded (or refused), never cast.
_DENSE_FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")


def materialize_source_weight(name: str, weight, scale, fp8_map):
    """Dense fp32 ``(out, in)`` weight from whatever the checkpoint stores.

    ``activation_dloss`` needs the SOURCE weight as a dense float matrix, but a
    native-quantized checkpoint does not store one: DSv4-Flash keeps its routed
    experts as MXFP4 nibble-packs (int8 ``(out, in/2)`` + per-32 E8M0 scales)
    and everything else as block-FP8 (e4m3 + a 128x128 E8M0 scale grid).
    Reading those bytes as a weight is not approximately right, it is garbage
    -- the packed expert has half the columns and fp8 codes cast to float land
    in the code range, not the value range.

    DISPATCH IS DECLARATION-DRIVEN, NEVER SHAPE-INFERRED. ``fp8_map`` is the
    streaming loader's own ``_build_fp8_scale_inv_map`` product -- the same map
    the probe, the cost stage and the exporter loaded this model through -- so
    membership (and ``mxfp4_names``, and the dequant ``block``) come from the
    checkpoint's declarations, and this function's output agrees with the
    weights the loader installs. Decoders are REUSED, not re-derived:

      * MXFP4: ``mxfp4_widen.dequantize_mxfp4_source`` (documented mirror of
        ``layer_streaming._read_layer_to_device`` step 3b -- same LUT, same
        low-nibble-first order, same 0xFF NaN rule), in fp32. The loader's
        installed tensor is this value downcast to bf16.
      * block-FP8: ``layer_streaming._dequant_fp8_block_weight`` with the
        map's declared block. Torch decodes the ``float8_e8m0fnu`` scale in
        ``.to(bfloat16)`` (0xFF -> NaN included), and the aligned path is the
        same bf16 tile multiply as the loader's batched step 3, so the bf16
        result is bit-identical to the loader install; the fp32 upcast is
        lossless. Note the ``.scale`` sibling is a MULTIPLIER (the loader
        multiplies by it), despite the legacy ``scale_inv`` spelling.

    Every mismatch RAISES -- a unit this stage cannot materialize must never
    be silently skipped, because an unpriced A-side reads as 0.0 (= free) to
    the DP, which is the exact mispricing the stage exists to remove. And the
    encodings are uniform per tensor class, so the first bad unit means a
    systematic bug: dying on it costs seconds, not a wrong 33k-unit artifact.
    """
    import torch

    from .layer_streaming import (_FLOAT8_DTYPES, _check_mxfp4_packed_grid,
                                  _dequant_fp8_block_weight,
                                  _fp8_dequant_block)
    from .mxfp4_widen import dequantize_mxfp4_source

    entry = fp8_map.get(name + ".weight") if fp8_map else None
    is_dense_float = str(weight.dtype).split(".")[-1] in _DENSE_FLOAT_DTYPES
    if entry is None:
        if is_dense_float:
            return weight.to(torch.float32)
        raise RuntimeError(
            f"{name}: source tensor has dtype {weight.dtype} but no scale "
            f"entry in the streaming loader's fp8/mxfp4 map "
            f"(_build_fp8_scale_inv_map). Casting these bytes to float would "
            f"install code-range values, not weights -- the historical "
            f"fp8-range bug. Check the checkpoint's scale-sibling naming "
            f"against the model profile's fp8_scale_pairs.")
    if scale is None:
        raise RuntimeError(
            f"{name}: the scale map declares a sibling for this weight but "
            f"none was supplied to materialize_source_weight")
    if is_dense_float:
        raise RuntimeError(
            f"{name}: source tensor is already a dense float "
            f"({weight.dtype}) yet the scale map has an entry for it; "
            f"applying a block scale to a dense weight would corrupt it. "
            f"The map and the checkpoint disagree -- refusing to guess.")
    if name + ".weight" in getattr(fp8_map, "mxfp4_names", frozenset()):
        _check_mxfp4_packed_grid(name + ".weight", weight, scale)
        return dequantize_mxfp4_source(weight, scale, dtype=torch.float32)
    if weight.dtype not in _FLOAT8_DTYPES:
        raise RuntimeError(
            f"{name}: mapped as block-FP8 but the tensor dtype is "
            f"{weight.dtype}, not a float8 wire; the declaration and the "
            f"bytes disagree.")
    return _dequant_fp8_block_weight(
        weight, scale, block=_fp8_dequant_block(fp8_map),
        name=name).to(torch.float32)


def cached_act_path(act_dir: str, name: str) -> str:
    """Where the probe parked this Linear's real input rows."""
    return os.path.join(act_dir, name.replace(".", "__") + ".pt")


def measured_act_var(spec, x_cpu, device: str):
    """Per-input-channel error variance on the layer's REAL activations.

    The synthetic path this replaces samples independent per-channel Gaussians.
    That reproduces every channel's marginal exactly and destroys the joint --
    and the joint is what an NVFP4 block scale is a function of, since 16
    consecutive channels share one FP8 scale set by the largest magnitude among
    them in that token. Real rows carry the co-occurrence; a Gaussian batch
    cannot.

    LANDMINE: the cached tensors are CPU-resident. Without the explicit
    ``.to(device)`` the quantizer runs on CPU at full numerical fidelity and no
    speed, which reads as "slow GPU" rather than "wrong device".
    """
    import torch
    x = x_cpu.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        xq = spec.activation_quantize_dequantize(x)
        per_channel = ((x.float() - xq.float()) ** 2).mean(dim=0)
    return per_channel.double().cpu().numpy()


#: Env lever for the render-conditioned A-side. Unset (or empty) keeps the
#: source-weight basis, which is a byte-identical no-op.
ACT_WEIGHT_BASIS_ENV = "PRISMAQUANT_AQUA_ACT_WEIGHT_BASIS"

#: The default: evaluate the A-side on the SOURCE weight. Deliberately NOT a
#: ``RenderBasis`` member, because it is the absence of a render rather than a
#: kind of one -- and conflating the two is what would let a caller believe an
#: unrendered term had a basis.
ACT_WEIGHT_BASIS_SOURCE = "source"


def _act_weight_bases() -> tuple[str, ...]:
    """Accepted basis spellings.

    ``rtn``/``compensated`` are ``RenderBasis``'s own vocabulary, reused rather
    than respelled so a basis means the same thing here as on a card.
    """
    from .sensitivity_card import RenderBasis
    return (ACT_WEIGHT_BASIS_SOURCE, RenderBasis.RTN.value,
            RenderBasis.COMPENSATED.value)


def resolve_act_weight_basis(explicit: str | None = None) -> str:
    """Which weight the A-side reduction runs on. Default: the source weight.

    Off unless ``PRISMAQUANT_AQUA_ACT_WEIGHT_BASIS`` is set (or ``explicit`` is
    passed); when off this is a byte-identical no-op and returns
    ``"source"`` -- today's behaviour, the shipped default, and the only basis
    on which the A-side is render-independent.

    A malformed value is a hard error, never a silent fall back to the default:
    a typo that quietly reverted the lever would report a render-conditioned
    provenance for a source-weight number.
    """
    if explicit is None:
        raw = os.environ.get(ACT_WEIGHT_BASIS_ENV)
        if raw is None or not str(raw).strip():
            return ACT_WEIGHT_BASIS_SOURCE
        explicit = raw
    basis = str(explicit).strip().lower()
    bases = _act_weight_bases()
    if basis not in bases:
        raise SystemExit(
            f"REFUSE: {ACT_WEIGHT_BASIS_ENV} / --act-weight-basis must be one "
            f"of {list(bases)}, got {explicit!r}. "
            f"{ACT_WEIGHT_BASIS_SOURCE!r} evaluates the A-side on the source "
            f"weight (the default, render-independent); 'rtn' and "
            f"'compensated' evaluate it on the format's rendered W_hat and are "
            f"research levers.")
    if basis != ACT_WEIGHT_BASIS_SOURCE:
        log(f"render-conditioned A-side ACTIVE (research lever, "
            f"{ACT_WEIGHT_BASIS_ENV}={basis}): the activation term is "
            f"evaluated on W_hat_f, not on the source W. This absorbs the "
            f"dW.dx contribution; the dW x / W_hat dx cross-correlation, "
            f"downstream nonlinearities and routing interactions remain "
            f"dropped.")
    return basis


def rendered_weight_for_basis(basis: str, *, name: str, fmt: str, unit,
                              weight, plugin, render_cache):
    """``W_hat_f`` for one (unit, format) on ``basis``, or None when absent.

    Returns ``None`` ONLY for a ``compensated`` cache miss -- the caller turns
    that into a hole. It never substitutes one basis for the other: an
    unavailable production render is a reported gap, not a licence to price the
    RTN one under the production label.

    ``float32`` on the way out, matching the source path: the reduction squares
    it, and a bfloat16 array is not something the host fallback in
    ``_weighted_row_sum`` can even view as numpy.
    """
    import torch

    from .sensitivity_card import RenderBasis

    if basis == RenderBasis.RTN.value:
        n_e = unit.n_experts
        if n_e is None:
            return plugin.render(weight).to(torch.float32)
        # A packed [E, M, N] unit is ONE decision unit but E rendered matrices:
        # the exporter renders each expert as its own tensor (a per-tensor
        # scale is derived per expert, not once across the pack), so a single
        # collapsed render would be a different rendering from the shipped one.
        return torch.stack(
            [plugin.render(weight[e]).to(torch.float32) for e in range(n_e)],
            dim=0)

    if basis != RenderBasis.COMPENSATED.value:
        raise AssertionError(f"unhandled act weight basis {basis!r}")

    got = render_cache.get(name, fmt)
    if got is None:
        return None
    if tuple(got.shape) != tuple(weight.shape):
        raise RuntimeError(
            f"{name}/{fmt}: the production cache holds a tensor of shape "
            f"{tuple(got.shape)} but the source weight is "
            f"{tuple(weight.shape)}. Those are different tensors, and pricing "
            f"the A-side on the wrong one would be silent -- refusing.")
    return got.to(dtype=torch.float32)


def resolve_executed_activation_formats(*, lane_id: str | None,
                                        executes_all: bool = False):
    """The formats whose activation grid the SERVING LANE executes.

    One resolver so every A-side caller reaches the same authority. Returns
    either the string ``"all"`` or a ``frozenset`` suitable for
    ``activation_dloss_table(executed_activation_formats=...)``.

    Refuses rather than defaulting: "which formats does the runtime quantize
    activations for" has no safe default, and the unsafe one (trust the format
    registry) is the mispricing this exists to prevent.
    """
    if executes_all and lane_id:
        raise SystemExit(
            "REFUSE: --serving-lane and --lane-executes-all-activation-grids "
            "are mutually exclusive; the lane spec is the authority when a "
            "lane is named.")
    if executes_all:
        log("serving lane executes EVERY format's activation grid (asserted); "
            "pricing the full A-side")
        return "all"
    if not lane_id:
        raise SystemExit(
            "REFUSE: name the serving lane (--serving-lane) so the A-side is "
            "priced against the activation contract the runtime actually "
            "executes, or assert "
            "--lane-executes-all-activation-grids. There is no default: "
            "assuming the format registry's W4A4 claim is what priced a full "
            "A-side onto gridbook's BF16-bridge CB lane.")
    from .lane_spec import load_lane_spec
    spec = load_lane_spec(lane_id)
    contract = spec.served_activation_quantization
    if contract is None:
        raise SystemExit(
            f"REFUSE: lane {lane_id!r} does not declare "
            f"`served_activation_quantization`, so which formats it executes "
            f"activation quantization for is unknown. Declare it on "
            f"prismaquant/lane_specs/{lane_id}.json -- an empty `executes` "
            f"list is a valid and common answer.")
    log(f"serving lane {lane_id!r} executes activation quantization for: "
        f"{sorted(contract.executes) or '(nothing)'}")
    if contract.rationale:
        log(f"  rationale: {contract.rationale}")
    return frozenset(contract.executes)


def activation_dloss_table(card, model_path: str, formats: list[str], *,
                           device: str = "cpu", names=None,
                           act_dir: str | None = None,
                           profile=None,
                           executed_activation_formats=None,
                           act_weight_basis: str | None = None,
                           render_cache=None,
                           ) -> tuple[dict, dict, dict]:
    """``{unit: {format: act_dloss}}`` plus a report of what could not be priced.

    A format that does not quantize activations is simply absent from a unit's
    inner dict -- that is not a hole, it is the correct answer (BF16 costs
    nothing on the A-side). A format that DOES quantize activations but could
    not be priced is recorded in ``holes``, because an unpriced A-side read as
    zero is the exact mispricing this stage exists to remove.

    ``executed_activation_formats`` is REQUIRED and is the set of formats whose
    activation grid the SERVING LANE actually executes -- normally
    ``LaneSpec.served_activation_quantization.executes``. The format registry
    can only say whether a format *is* W4A4; it cannot know whether the runtime
    that will serve this artifact runs the fused kernel or decodes to BF16 and
    runs a BF16 GEMM. Those are different questions and this stage needs the
    second one. Passing ``"all"`` asserts that the lane executes every format's
    activation grid, and is the correct answer for a plain fused-W4A4 lane.

    It has no default on purpose. Defaulting to the registry's claim is what
    priced a full A-side onto the nvfp4_cb lane -- which serves on gridbook's
    exact BF16 bridge and quantizes no activations at all -- and cost the
    DSv4-Flash 92 GB body the majority of its codebook rung (K16 -> K12) buying
    FP8 promotions to escape a cost of zero.

    ``act_weight_basis`` selects the weight the reduction runs on -- ``None``
    reads ``PRISMAQUANT_AQUA_ACT_WEIGHT_BASIS`` and defaults to the source
    weight, which is a byte-identical no-op. ``render_cache`` is a
    ``ProductionWeightCache`` and is REQUIRED by (and only by) the
    ``compensated`` basis.
    """
    basis = resolve_act_weight_basis(act_weight_basis)
    from .sensitivity_card import RenderBasis
    if basis == RenderBasis.COMPENSATED.value and render_cache is None:
        raise SystemExit(
            "REFUSE: the 'compensated' A-side basis prices the activation term "
            "on the GPTQ+JSO tensors the export will ship, so it needs the "
            "ProductionWeightCache that holds them (--production-cache). "
            "Falling back to the in-process RTN render would report a "
            "production basis for a number that never saw the production "
            "recipe -- pass the cache, or ask for --act-weight-basis rtn.")
    if basis != RenderBasis.COMPENSATED.value and render_cache is not None:
        raise SystemExit(
            f"REFUSE: a production weight cache was supplied but the A-side "
            f"basis is {basis!r}, which never reads it. Silently ignoring it "
            f"would stamp a source/rtn number next to a production cache path "
            f"in provenance. Pass --act-weight-basis "
            f"{RenderBasis.COMPENSATED.value}, or drop the cache.")
    if executed_activation_formats is None:
        raise SystemExit(
            "REFUSE: executed_activation_formats is required. The A-side price "
            "depends on what the SERVING LANE executes, not on what the format "
            "registry declares the format to be -- gridbook's CB lane serves "
            "NVFP4_CB weights through the exact BF16 bridge and quantizes no "
            "activations, so its correct A-side is exactly zero. Pass the "
            "lane's `served_activation_quantization.executes` (an empty set is "
            "a valid, common answer), or the string \"all\" for a lane that "
            "genuinely serves every format's activation grid fused.")
    executes_all = executed_activation_formats == "all"
    # GLOB PATTERNS, not names: the answer is per FAMILY and rungs are
    # open-ended. `FP8_CB_*` must keep covering a rung added tomorrow, or the
    # A-side silently vanishes for it -- the same silent default this argument
    # exists to remove.
    patterns = (("*",) if executes_all
                else tuple(executed_activation_formats))
    if not executes_all and not patterns:
        raise SystemExit(
            "REFUSE: this lane executes NO format's activation quantization, "
            "so every A-side price is exactly zero and merging one would only "
            "overcharge the DP. Allocate from the weight-only cost instead of "
            "building an A-side that the served artifact never pays. "
            "(If a fused activation mode is being enabled for this artifact, "
            "declare the formats it executes on the lane spec first.)")
    import torch
    from safetensors import safe_open

    from .format_cost_protocol import price_activation_only
    from .format_cost_registry import RegistryFormatPlugin

    with open(os.path.join(model_path, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    resolver = build_weight_resolver(weight_map, profile=profile)

    wanted = list(names) if names is not None else [u.topology.name
                                                   for u in card.units()]
    resolvable = [n for n in wanted if n in resolver]
    log(f"weight-key resolution: {len(resolvable)}/{len(wanted)} card units "
        f"found in the checkpoint")
    # Refuse rather than write a no-op. "Nothing resolved" is never a valid
    # outcome for this stage, and the artifact it would otherwise produce is
    # indistinguishable from a real one -- same units, same formats, an A-side
    # that is absent and therefore read as 0.0 (free) by the DP. Only the
    # unambiguous case is a refusal; no coverage threshold is invented here,
    # because any such number would be a heuristic (principle 2). Partial
    # coverage is already reported per-format through `holes`.
    if wanted and not resolvable:
        raise SystemExit(
            f"REFUSE: 0 of {len(wanted)} card units resolve to a checkpoint "
            f"tensor, so there is nothing to price. This is a NAME-SPACE "
            f"mismatch, not an empty menu: card units come from the module "
            f"tree the probe walked, while the checkpoint may rename both the "
            f"path and the leaf. Pass a model profile (--model-path must be a "
            f"directory `detect_profile` recognises) so the architecture's own "
            f"`checkpoint_to_live_name` can supply the aliases. Sample card "
            f"name {wanted[0]!r}; sample checkpoint key "
            f"{next(iter(weight_map))!r}."
        )

    # The streaming loader's own scale map -- fp8/mxfp4 declarations, scale
    # sibling keys, and the checkpoint-declared dequant block. Empty for a
    # dense checkpoint, in which case every path below is byte-identical to
    # the pre-quantized-source behaviour (plain get_tensor + fp32 cast).
    from .layer_streaming import _build_fp8_scale_inv_map
    fp8_map = _build_fp8_scale_inv_map(model_path)
    if fp8_map:
        log(f"quantized-source checkpoint: {len(fp8_map)} scale-mapped "
            f"weights ({len(fp8_map.mxfp4_names)} declared MXFP4, dequant "
            f"block {fp8_map.block}); units will be materialized through "
            f"the streaming loader's decoders")

    # SHARD-AT-A-TIME. ``safe_open(device="cpu")`` mmaps the shard and
    # ``get_tensor`` faults its pages in; while the handle lives those pages stay
    # RESIDENT. Holding every handle grows RSS by the full bf16 size of every
    # tensor touched -- measured at 7.1 -> 48.2 GiB over 500 units on this model,
    # which is the body's weights almost exactly. Grouping by shard and closing
    # each handle bounds resident mmap to ONE shard. Order is irrelevant: each
    # unit is priced independently.
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name in resolvable:
        by_shard[weight_map[resolver[name]]].append(name)
    log(f"pricing the A-side of {len(resolvable)} units across "
        f"{len(by_shard)} shards, one shard resident at a time")

    table: dict[str, dict[str, float]] = {}
    holes: dict[str, list[str]] = collections.defaultdict(list)
    non_act: set[str] = set()
    not_executed: set[str] = set()
    t0 = time.time()
    done = 0
    var_source = collections.Counter()
    # Per-format render accounting for the opt-in bases. Hits and misses are
    # counted separately from `holes` because a miss has one specific cause
    # (the cache does not hold this (unit, format)) that the caller must be
    # able to see without parsing hole strings.
    render_hits: collections.Counter = collections.Counter()
    render_misses: collections.Counter = collections.Counter()
    for shard in sorted(by_shard):
        with safe_open(os.path.join(model_path, shard),
                       framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys()) if fp8_map else frozenset()
            for name in by_shard[shard]:
                unit = card[name]
                raw = handle.get_tensor(resolver[name])
                scale_t = None
                entry = fp8_map.get(name + ".weight") if fp8_map else None
                if entry is not None:
                    scale_shard, scale_key = entry
                    if scale_key in shard_keys:
                        scale_t = handle.get_tensor(scale_key)
                    else:
                        # Scale sibling serialized in a different shard than
                        # its weight -- rare, but the map records the shard
                        # so honour it rather than assume co-location.
                        with safe_open(scale_shard, framework="pt",
                                       device="cpu") as sh:
                            scale_t = sh.get_tensor(scale_key)
                w_t = materialize_source_weight(name, raw, scale_t, fp8_map)
                # A packed routed-expert unit is [E, out, in]: ONE decision
                # unit, E matrices. `out_features`/`in_features` describe one
                # expert's slice (that is what the probe recorded), so the
                # expected rank depends on whether the card carries per-expert
                # statistics for it.
                n_experts = unit.n_experts
                want = ((unit.out_features, unit.in_features)
                        if n_experts is None else
                        (n_experts, unit.out_features, unit.in_features))
                if tuple(w_t.shape) != want:
                    raise RuntimeError(
                        f"{name}: materialized weight has shape "
                        f"{tuple(w_t.shape)}, expected {want} from the "
                        f"probe stats (source dtype {raw.dtype}, "
                        f"{'scale-mapped' if entry else 'dense'}); refusing "
                        f"to price a wrong-shaped W")
                # Uploaded ONCE per unit and reused by every candidate format:
                # the A-side reduction is GPU work (principle 7), and pricing a
                # 3-format menu off a host array would either copy the tensor
                # three times or do 70 G float64 multiply-adds on the CPU. On a
                # packed expert tensor that is 2 GiB of copy per format.
                w_np = w_t.to(device=device, dtype=torch.float32) \
                    if torch.cuda.is_available() and device != "cpu" \
                    else w_t.to(torch.float32).numpy()
                del raw, scale_t, w_t
                # Real input rows if the probe cached them for this Linear.
                # Loaded once per unit and reused across formats -- the tensor
                # is the same batch, only the quantizer differs.
                x_cpu = None
                # Packed units deliberately do NOT take the measured path.
                # The probe caches the experts MODULE's input -- every token
                # before routing -- so a measured variance from it is one
                # population figure shared by all E experts, which is exactly
                # the routing structure this unit needs to keep. The per-expert
                # synthetic fit (source 2) is used instead: it loses the
                # cross-channel joint but keeps the per-expert distribution,
                # and routing is the larger effect on a 256-expert layer.
                if act_dir and n_experts is None:
                    p = cached_act_path(act_dir, name)
                    if os.path.exists(p):
                        blob_ = torch.load(p, map_location="cpu",
                                           weights_only=False)
                        cand = blob_.get("inputs")
                        # A shape mismatch means the cache is from a different
                        # model/shape; silently pricing on it would be worse
                        # than falling back, so require the match.
                        if (cand is not None
                                and cand.ndim == 2
                                and cand.shape[1] == unit.in_features):
                            x_cpu = cand
                row: dict[str, float] = {}
                for fmt in formats:
                    try:
                        plugin = RegistryFormatPlugin.build(
                            fmt, shape=(unit.out_features, unit.in_features),
                            device=device)
                    except Exception as exc:
                        holes[fmt].append(f"{name}: unbuildable ({exc})")
                        continue
                    if not plugin.descriptor.quantizes_activations:
                        non_act.add(fmt)
                        del plugin
                        continue
                    # The format quantizes activations; this lane may still not
                    # execute that. Same outcome (no A-side), different reason,
                    # so it is reported separately below rather than folded in.
                    if not executes_all and not any(
                            fnmatch.fnmatchcase(fmt, pat)
                            for pat in patterns):
                        not_executed.add(fmt)
                        del plugin
                        continue
                    # W_hat_f, when a render-conditioned basis was asked for.
                    # Acquired BEFORE the variance so a cache miss costs
                    # nothing: a miss is a hole either way.
                    w_hat = None
                    if basis != ACT_WEIGHT_BASIS_SOURCE:
                        w_hat = rendered_weight_for_basis(
                            basis, name=name, fmt=fmt, unit=unit,
                            weight=w_np, plugin=plugin,
                            render_cache=render_cache)
                        if w_hat is None:
                            render_misses[fmt] += 1
                            holes[fmt].append(
                                f"{name}: no {basis} render in the cache")
                            del plugin
                            continue
                        render_hits[fmt] += 1
                    v = None
                    if x_cpu is not None:
                        v = measured_act_var(plugin.spec, x_cpu, device)
                    if v is not None:
                        var_source["measured"] += 1
                    elif n_experts is None:
                        var_source["modelled"] += 1
                    else:
                        var_source["modelled_per_expert"] += 1
                    a = price_activation_only(unit, w_np, plugin, act_var=v,
                                              rendered_weight=w_hat)
                    if a is None:
                        holes[fmt].append(name)
                    else:
                        row[fmt] = float(a)
                    del plugin, w_hat
                if row:
                    table[name] = row
                del w_np, x_cpu
                done += 1
                if torch.cuda.is_available() and (
                        torch.cuda.memory_reserved() / (1 << 30)
                        >= CUDA_RESERVED_DRAIN_GIB):
                    torch.cuda.empty_cache()
                if done % 100 == 0:
                    log(f"  priced {done}/{len(resolvable)} "
                        f"({time.time() - t0:.0f}s)")
    log(f"A-side priced for {len(table)} units in {time.time() - t0:.0f}s")
    if var_source:
        log(f"act_var source: {dict(var_source)} (measured = real cached "
            f"activations; modelled = per-channel Gaussian fit; "
            f"modelled_per_expert = the same fit, one per routed expert)")
    if non_act:
        log(f"formats that leave activations alone (correctly unpriced): "
            f"{sorted(non_act)}")
    if not_executed:
        log(f"formats that quantize activations but THIS LANE DOES NOT EXECUTE "
            f"(correctly unpriced -- served A-side is exactly zero): "
            f"{sorted(not_executed)}")
    if basis != ACT_WEIGHT_BASIS_SOURCE:
        log(f"A-side weight basis {basis!r}: rendered {sum(render_hits.values())} "
            f"(unit, format) pairs, {sum(render_misses.values())} missing "
            f"({dict(render_misses)})")
        # Refuse rather than write a no-op, for the same reason 0/N resolution
        # refuses: an artifact whose act_dloss rows are simply absent is
        # indistinguishable from a real one and reads as 0.0 (free) to the DP.
        # Zero rendered pairs is the unambiguous case; no coverage threshold is
        # invented here (principle 2).
        if not sum(render_hits.values()):
            raise SystemExit(
                f"REFUSE: --act-weight-basis {basis} rendered 0 (unit, format) "
                f"pairs, so no A-side was priced on that basis at all. For "
                f"'compensated' this normally means the production cache holds "
                f"another assignment's formats (or another model's names); "
                f"check its coverage before re-running.")
    for fmt, names_ in sorted(holes.items()):
        log(f"HOLE: {fmt} quantizes activations but {len(names_)} units could "
            f"not be priced; those rows keep a weight-only cost. "
            f"e.g. {names_[:3]}")
    meta = {
        "act_var_source": dict(var_source),
        # The A-side has two independent approximations and both are stamped,
        # because "we replayed the real quantizer" and "we fitted a Gaussian"
        # are different claims and only one of them is a measurement.
        "act_var_paths": {
            "measured": "the format's own activation QDQ replayed on the "
                        "layer's real cached input rows (no distributional "
                        "assumption)",
            "modelled": "APPROXIMATION: the format's own QDQ over independent "
                        "per-channel Gaussians fitted to act_sq_sum/act_absmax "
                        "-- exact marginals, no joint, so it cannot represent "
                        "how outliers co-occur across the channels that share "
                        "one block scale, nor a clipping tail the second "
                        "moment averages away",
            "modelled_per_expert": "APPROXIMATION: the same fit, one per "
                                   "routed expert",
        },
        "act_weight_basis": basis,
        "act_weight_basis_is_render_conditioned":
            basis != ACT_WEIGHT_BASIS_SOURCE,
        "act_weight_basis_dropped_terms": (
            "the dW.x / W_hat.dx cross-correlation, downstream nonlinear "
            "interactions, and MoE routing interactions (a route flip is "
            "invisible to any second-moment term)"
            + ("" if basis != ACT_WEIGHT_BASIS_SOURCE
               else "; plus the dW.dx contribution, which only the "
                    "render-conditioned bases absorb")),
        "render_hits": dict(render_hits),
        "render_misses": dict(render_misses),
    }
    return (table, {k: v for k, v in holes.items()}, meta)


def merge_act_dloss(costs: dict, table: dict) -> dict:
    """Write ``act_dloss`` into the cost rows. Returns a merge report.

    Mutates ``costs`` in place. Rows with no priced A-side are left untouched
    rather than set to 0.0, so ``cost_entry_act_dloss``'s default and a genuine
    measured zero stay distinguishable in the artifact.
    """
    merged = 0
    unit_hits = 0
    missing_units = []
    for name, entry in costs.items():
        row = table.get(name)
        if not row:
            missing_units.append(name)
            continue
        unit_hits += 1
        for fmt, value in row.items():
            if fmt in entry and isinstance(entry[fmt], dict):
                entry[fmt][ACT_DLOSS_KEY] = float(value)
                merged += 1
    return {"units_in_cost": len(costs), "units_merged": unit_hits,
            "entries_merged": merged,
            "units_without_act_price": len(missing_units),
            "examples_without_act_price": missing_units[:5]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", required=True, help="sensitivity card .npz")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cost-in", required=True, help="weight-only cost pkl")
    ap.add_argument("--cost-out", required=True,
                    help="written; --cost-in is left untouched so the "
                         "weight-only allocation stays reproducible as an arm")
    ap.add_argument("--formats", default=None,
                    help="default: every format present in the cost artifact")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--serving-lane", default=None,
        help="lane id whose served_activation_quantization declares which "
             "formats' activation grid the runtime actually EXECUTES. The "
             "A-side is a property of the runtime, not of the format: a lane "
             "that decodes CB weights to BF16 and runs a BF16 GEMM pays no "
             "activation cost at all, and pricing one there makes the DP buy "
             "weight bits to escape zero. Required unless "
             "--lane-executes-all-activation-grids is given.")
    ap.add_argument(
        "--lane-executes-all-activation-grids", action="store_true",
        help="assert that the serving lane executes EVERY format's activation "
             "grid fused (the correct answer for a plain W4A4 lane). Mutually "
             "exclusive with --serving-lane.")
    ap.add_argument("--act-dir", default=None,
                    help="directory of cached real activations (the probe's "
                         "act/ dir). When given, act_var is MEASURED on each "
                         "Linear's real input rows instead of modelled from a "
                         "per-channel Gaussian fit; units with no cached rows "
                         "fall back to the model and are counted separately.")
    ap.add_argument(
        "--act-weight-basis", default=None,
        help="which weight the activation term is reduced over: 'source' "
             "(default, and what PRISMAQUANT_AQUA_ACT_WEIGHT_BASIS selects "
             "when unset -- render-independent), 'rtn' (the registry's own "
             "quantize_dequantize) or 'compensated' (the GPTQ+JSO tensors from "
             "--production-cache, i.e. the bytes the export ships). The "
             "rendered bases absorb the dW.dx term the source basis drops; "
             "both are RESEARCH levers with no served A/B.")
    ap.add_argument(
        "--production-cache", default=None,
        help="ProductionWeightCache pickle. Required by, and only read by, "
             "--act-weight-basis compensated.")
    ap.add_argument(
        "--production-cache-dir-override", default=None,
        help="relocate a disk-streamed production cache, as the exporter does.")
    ap.add_argument(
        "--production-cache-lru-gb", type=float, default=0.0,
        help="bound the cache's resident tensors (GiB); 0 disables the LRU.")
    args = ap.parse_args()

    from .sensitivity_card import SensitivityCard

    card = SensitivityCard.from_npz(args.card)
    card.validate()
    fingerprint = card.provenance.fingerprint()
    log(f"card: {len(card)} units, fingerprint {fingerprint}")

    with open(args.cost_in, "rb") as fh:
        blob = pickle.load(fh)
    costs = blob["costs"]
    formats = ([f.strip() for f in args.formats.split(",") if f.strip()]
               if args.formats
               else sorted({f for r in costs.values() for f in r}))
    log(f"cost artifact: {len(costs)} units, formats {formats}")

    # The architecture's declared name mapping. Optional by design: a model
    # whose checkpoint names match its module tree needs none, and a path that
    # no profile claims must not become a hard failure for those models. When
    # one IS detected it supplies the checkpoint aliases the generic index
    # cannot derive (see build_weight_resolver).
    profile = None
    try:
        from .model_profiles.registry import detect_profile
        profile = detect_profile(args.model_path)
    except Exception as exc:
        log(f"no model profile for {args.model_path} ({exc}); "
            f"falling back to generic name matching")
    if profile is not None:
        log(f"model profile: {type(profile).__name__}")

    executed = resolve_executed_activation_formats(
        lane_id=args.serving_lane,
        executes_all=args.lane_executes_all_activation_grids,
    )
    render_cache = None
    if args.production_cache:
        with open(args.production_cache, "rb") as fh:
            render_cache = pickle.load(fh)
        if args.production_cache_dir_override:
            render_cache.relocate(args.production_cache_dir_override)
        if args.production_cache_lru_gb and args.production_cache_lru_gb > 0:
            render_cache.enable_lru(
                int(float(args.production_cache_lru_gb) * 1024 ** 3))
        log(f"production weight cache: {len(render_cache)} entries from "
            f"{args.production_cache}")
    table, holes, meta = activation_dloss_table(
        card, args.model_path, formats, device=args.device,
        names=[n for n in costs], act_dir=args.act_dir, profile=profile,
        executed_activation_formats=executed,
        act_weight_basis=args.act_weight_basis, render_cache=render_cache)
    report = merge_act_dloss(costs, table)
    log(f"merge: {report}")
    # Belt and braces on the silent-no-op: resolution can succeed while every
    # price still comes back None (e.g. a scalar-only card with no g_sq_sum).
    if not report["entries_merged"]:
        raise SystemExit(
            "REFUSE: the merge wrote 0 entries, so --cost-out would be a "
            "byte-equivalent copy of --cost-in carrying the AQUA name. Most "
            "likely the card has no `g_sq_sum` (a scalar-only card built from "
            "a probe predating marginal emission); `activation_dloss` returns "
            "None for every unit in that case."
        )

    prov = dict(blob.get("provenance") or {})
    prov["aqua_activation_cost"] = {
        "card_fingerprint": fingerprint,
        "card_path": os.path.abspath(args.card),
        "formats_priced": formats,
        "holes": {k: len(v) for k, v in holes.items()},
        "merge_report": report,
        "act_dir": os.path.abspath(args.act_dir) if args.act_dir else None,
        "production_cache_path": (os.path.abspath(args.production_cache)
                                  if args.production_cache else None),
        **meta,
    }
    blob["provenance"] = prov
    with open(args.cost_out, "wb") as fh:
        pickle.dump(blob, fh)
    log(f"wrote {args.cost_out}")

    # A one-line readout of what the DP will now see differently. Not a result --
    # the served KL A/B is -- but enough to catch a merge that did nothing.
    ratios = []
    for name, entry in costs.items():
        n = entry.get("NVFP4")
        if isinstance(n, dict) and ACT_DLOSS_KEY in n and n.get(
                "predicted_dloss", 0.0) > 0:
            ratios.append(n[ACT_DLOSS_KEY] / n["predicted_dloss"])
    if ratios:
        r = np.array(ratios)
        log(f"NVFP4 A-side / W-side over {len(r)} units: "
            f"p10={np.percentile(r, 10):.2f} med={np.median(r):.2f} "
            f"p90={np.percentile(r, 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
