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
the A-side is genuinely separable:

  * It needs NO render. ``activation_dloss`` reads the DENSE weight (as
    ``W[o,j]^2``), the card's ``g_sq_sum``, and the format's activation grid.
    The render basis never enters, so the number is identical whether the
    W-side was rendered with RTN or with the full GPTQ+JSO production recipe.
  * It therefore costs one streaming pass over the checkpoint and a row-blocked
    ``W^2 @ var`` per unit -- minutes, not the hours a render costs -- and it can
    be recomputed against any existing cost artifact without rebuilding it.

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
import dataclasses
import fnmatch
import json
import os
import pickle
import re
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


def build_packed_expert_plan(weight_map: dict, card, names, profile) -> dict:
    """``{unit: {expert_idx: {projection: ckpt_key}}}`` for packed units the
    checkpoint stores PER EXPERT and unfused.

    A packed decision unit is ONE ``[E, out, in]`` live parameter, but many
    checkpoints serialize it as ``E * len(projections)`` separate 2-D tensors
    (``...experts.{e}.{proj}.weight``). :func:`build_weight_resolver` maps a
    unit to ONE checkpoint key, so it cannot express that fan-out and every
    such unit drops out at resolution -- SILENTLY, because an absent
    ``act_dloss`` reads as 0.0 (free) to the DP. On GLM-5.3-Flash that was the
    entire routed body: 84 of 288 units, and precisely the ones whose menu has
    a live NVFP4-vs-FP8 choice for the A-side to move. Assembling them is not
    an optimization; it is what stops the routed body being A-priced at zero.

    Every structural decision comes from the model profile
    (``packed_expert_param_names`` / ``packed_expert_projection_names``), never
    from a name pattern guessed here (principle 2). A unit whose expert set is
    incomplete on disk raises rather than assembling a short tensor: a packed
    weight missing experts would price a real A-side too low, which is the
    silent under-charge this whole stage exists to remove.
    """
    if profile is None:
        return {}
    try:
        parents = frozenset(profile.packed_expert_param_names())
    except Exception:
        return {}
    if not parents:
        return {}
    # Index the per-expert leaves once: experts_path -> {idx -> {proj -> key}}.
    by_path: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    pat = re.compile(r"^(?P<path>.+)\.(?P<idx>\d+)\.(?P<proj>[^.]+)\.weight$")
    for key in weight_map:
        m = pat.match(key)
        if m is None:
            continue
        by_path[m.group("path")][int(m.group("idx"))][m.group("proj")] = key

    plan: dict = {}
    for name in names:
        unit = None
        try:
            unit = card[name]
        except Exception:
            continue
        if unit is None or unit.n_experts is None:
            continue
        experts_path, _, parent = name.rpartition(".")
        if parent not in parents:
            continue
        per_expert = by_path.get(experts_path)
        if not per_expert:
            continue
        try:
            order = tuple(profile.packed_expert_projection_names(parent))
        except Exception:
            continue
        if not order:
            continue
        n_e = int(unit.n_experts)
        built: dict = {}
        for e in range(n_e):
            projs = per_expert.get(e)
            if projs is None or any(p not in projs for p in order):
                raise RuntimeError(
                    f"{name}: checkpoint stores routed experts per-expert but "
                    f"expert {e} is missing projection(s) {order}. Refusing to "
                    f"assemble a short packed weight -- it would under-price "
                    f"this unit's A-side rather than fail.")
            built[e] = {p: projs[p] for p in order}
        plan[name] = built
    return plan


def _warm_shard(path: str, block: int = 1 << 25) -> None:
    """Pull one shard into page cache with a single SEQUENTIAL pass.

    Measured, not assumed (principle 15). ``safetensors`` mmaps the shard and
    ``get_tensor`` faults pages in on touch. A packed routed-expert unit needs
    E * len(projections) small tensors scattered through a multi-GiB shard, and
    that random fault pattern ran at **92 MB/s against 648 MB/s sequential** on
    this NFS mount, with 21.6 GiB of block reads for 1.9 GiB of requested bytes
    -- 11x amplification, readahead pulling pages that are evicted before the
    walk reaches them. One sequential pass makes every later fault a RAM hit.

    This is a residency fix, not a numerical one: it changes when bytes arrive,
    never which bytes. Cheap to undo if a future layout makes it pointless.
    """
    buf = memoryview(bytearray(block))
    with open(path, "rb", buffering=0) as fh:
        while fh.readinto(buf):
            pass


def _slice_experts(arr, ids):
    """Expert-axis slice of a card's per-expert array (None stays None)."""
    if arr is None:
        return None
    return np.asarray(arr)[ids]


def _assemble_expert_slab(handles, plan_unit, expert_ids, order, fp8_map,
                          unit, name):
    """One ``[C, out, in]`` float32 slab for ``expert_ids``.

    Projections are concatenated on the OUTPUT axis in the profile's declared
    order (the packed-FusedMoE convention ``_pack_per_expert_into_packed``
    also uses), then experts are stacked on a new leading axis. The shape is
    checked against the card so a layout mismatch fails loud.
    """
    import torch
    slabs = []
    for e in expert_ids:
        parts = []
        for proj in order:
            key = plan_unit[e][proj]
            raw = handles(key).get_tensor(key)
            scale = None
            if fp8_map is not None and fp8_map.get(key) is not None:
                scale_shard, scale_key = fp8_map.get(key)
                scale = handles(scale_key, shard=scale_shard).get_tensor(scale_key)
            leaf = key[: -len(".weight")]
            parts.append(materialize_source_weight(leaf, raw, scale, fp8_map))
            del raw, scale
        slabs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
        del parts
    slab = torch.stack(slabs, dim=0).contiguous()
    del slabs
    want = (len(expert_ids), unit.out_features, unit.in_features)
    if tuple(slab.shape) != want:
        raise RuntimeError(
            f"{name}: assembled expert slab has shape {tuple(slab.shape)}, "
            f"expected {want} from the probe stats; refusing to price a "
            f"wrong-shaped W")
    return slab


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
                           experts_per_chunk: int = 16,
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
    """
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
    # Packed routed experts a 1:1 name map cannot reach: one live [E, out, in]
    # decision unit stored as E * len(projections) separate 2-D checkpoint
    # tensors. Without this they drop out silently and the DP reads their
    # A-side as 0.0 (free).
    packed_plan = build_packed_expert_plan(
        weight_map, card, [n for n in wanted if n not in resolver], profile)
    if packed_plan:
        log(f"per-expert packed assembly: {len(packed_plan)} additional units "
            f"reachable by stacking per-expert checkpoint leaves")
    # Refuse rather than write a no-op. "Nothing resolved" is never a valid
    # outcome for this stage, and the artifact it would otherwise produce is
    # indistinguishable from a real one -- same units, same formats, an A-side
    # that is absent and therefore read as 0.0 (free) by the DP. Only the
    # unambiguous case is a refusal; no coverage threshold is invented here,
    # because any such number would be a heuristic (principle 2). Partial
    # coverage is already reported per-format through `holes`.
    if wanted and not resolvable and not packed_plan:
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
                    v = None
                    if x_cpu is not None:
                        v = measured_act_var(plugin.spec, x_cpu, device)
                    if v is not None:
                        var_source["measured"] += 1
                    elif n_experts is None:
                        var_source["modelled"] += 1
                    else:
                        var_source["modelled_per_expert"] += 1
                    a = price_activation_only(unit, w_np, plugin, act_var=v)
                    if a is None:
                        holes[fmt].append(name)
                    else:
                        row[fmt] = float(a)
                    del plugin
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
    # ---- packed routed experts, assembled from per-expert checkpoint leaves.
    if packed_plan:
        log(f"assembling {len(packed_plan)} packed routed-expert units "
            f"({experts_per_chunk} experts per slab)")
        t1 = time.time()
        warmed: set = set()
        for pi, name in enumerate(sorted(packed_plan), 1):
            plan_unit = packed_plan[name]
            unit = card[name]
            need = sorted({weight_map[k] for by in plan_unit.values()
                           for k in by.values()})
            for sh in need:
                if sh not in warmed:
                    _warm_shard(os.path.join(model_path, sh))
            # Only the shards this unit used stay "warm"; anything else is
            # assumed evicted, so the next unit re-warms what it needs.
            warmed = set(need)
            order = tuple(profile.packed_expert_projection_names(
                name.rpartition(".")[2]))
            n_e = int(unit.n_experts)
            # One handle cache per unit. A layer's experts live in a handful of
            # shards; holding them open for the unit avoids reopening a shard
            # per expert while still bounding resident mmap to that layer.
            open_shards: dict = {}

            def handles(key, shard=None, _os=open_shards):
                sh = shard if shard is not None else weight_map[key]
                h = _os.get(sh)
                if h is None:
                    h = safe_open(os.path.join(model_path, sh),
                                  framework="pt", device="cpu")
                    h.__enter__()
                    _os[sh] = h
                return h

            row = {}
            try:
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
                    if not executes_all and not any(
                            fnmatch.fnmatchcase(fmt, pat) for pat in patterns):
                        not_executed.add(fmt)
                        del plugin
                        continue
                    # EXPERT-CHUNKED, and the chunking is exact rather than an
                    # approximation. `_activation_dloss_packed` sums over
                    # experts and then applies a normalization LINEAR in that
                    # sum (0.5 * total / n_tokens * gain), so the sum of
                    # per-chunk prices IS the whole-tensor price -- up to
                    # float64 addition order, and every term is a square times
                    # a variance, so there is no cancellation to amplify it.
                    # That identity is what lets a 19.3 GiB packed unit be
                    # priced in ~1 GiB slabs instead of not at all.
                    total = 0.0
                    for lo in range(0, n_e, experts_per_chunk):
                        ids = list(range(lo, min(lo + experts_per_chunk, n_e)))
                        slab = _assemble_expert_slab(
                            handles, plan_unit, ids, order, fp8_map, unit, name)
                        sub = dataclasses.replace(
                            unit,
                            expert_g_sq_sum=_slice_experts(
                                unit.expert_g_sq_sum, ids),
                            expert_act_sq_sum=_slice_experts(
                                unit.expert_act_sq_sum, ids),
                            expert_act_absmax=_slice_experts(
                                unit.expert_act_absmax, ids),
                            expert_tokens=_slice_experts(
                                unit.expert_tokens, ids))
                        w_slab = (slab.to(device=device, dtype=torch.float32)
                                  if torch.cuda.is_available()
                                  and device != "cpu"
                                  else slab.to(torch.float32).numpy())
                        del slab
                        # Name the estimator rather than letting
                        # price_activation_only fall through to the dense
                        # measure or the analytic model: chunks of one unit
                        # must all be priced by the SAME estimator or the sum
                        # is a mixture of two quantities.
                        measure = getattr(plugin,
                                          "expert_activation_error_variance",
                                          None)
                        var_e = measure(sub) if callable(measure) else None
                        part = (None if var_e is None else
                                price_activation_only(sub, w_slab, plugin,
                                                      act_var=var_e))
                        del w_slab, var_e
                        if part is None:
                            total = None
                            break
                        total += float(part)
                        if torch.cuda.is_available() and (
                                torch.cuda.memory_reserved() / (1 << 30)
                                >= CUDA_RESERVED_DRAIN_GIB):
                            torch.cuda.empty_cache()
                    if total is None:
                        holes[fmt].append(name)
                    else:
                        row[fmt] = float(total)
                        var_source["modelled_per_expert"] += 1
                    del plugin
            finally:
                for h in open_shards.values():
                    try:
                        h.__exit__(None, None, None)
                    except Exception:
                        pass
            if row:
                table[name] = row
            if pi % 5 == 0 or pi == 1 or pi == len(packed_plan):
                log(f"  packed {pi}/{len(packed_plan)} "
                    f"({time.time() - t1:.0f}s)")

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
    # A unit that never RESOLVED never reached the per-format loop above, so
    # nothing recorded it and `holes` read empty while whole classes of unit
    # were absent -- provenance that says "no holes" while the DP reads their
    # A-side as 0.0 (free) is the silent-zero shape this project keeps getting
    # burned by. Record them against every format that would otherwise have
    # been priced, so a consumer reading `holes` sees the real coverage.
    # ONLY units that never reached the loop. A unit that DID reach it and
    # could not be priced already appended itself with its real reason
    # ("unbuildable", or a bare name from a `price_activation_only` -> None),
    # and `table[name]` is only set when its row is non-empty -- so counting
    # `wanted - table` here would record those a second time under a reason
    # that is false for them. Materialization failures cannot hide in this
    # set: both the dense path and `_assemble_expert_slab` RAISE on a shape or
    # layout mismatch rather than skipping, so everything in
    # `resolvable | packed_plan` either lands in `table` or is already a hole.
    reached_pricing_loop = set(resolvable) | set(packed_plan)
    unpriceable = [n for n in wanted
                   if n not in table and n not in reached_pricing_loop]
    priceable_formats = [f for f in formats
                         if f not in non_act and f not in not_executed]
    if unpriceable and priceable_formats:
        log(f"{len(unpriceable)} card units never reached the pricing loop "
            f"(unresolved in the checkpoint); recording them as holes for "
            f"{priceable_formats}")
        for fmt in priceable_formats:
            holes[fmt].extend(f"{n}: unresolved in checkpoint"
                              for n in unpriceable)
    for fmt, names_ in sorted(holes.items()):
        log(f"HOLE: {fmt} quantizes activations but {len(names_)} units could "
            f"not be priced; those rows keep a weight-only cost. "
            f"e.g. {names_[:3]}")
    return (table, {k: v for k, v in holes.items()},
            {"act_var_source": dict(var_source)})


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
    ap.add_argument("--packed-expert-chunk", type=int, default=16,
                    help="experts per assembled slab for packed routed-expert "
                         "units whose checkpoint stores experts separately. "
                         "The per-chunk prices SUM to the whole-unit price "
                         "exactly (the normalization is linear in the "
                         "expert sum), so this bounds peak memory without "
                         "changing the number.")
    ap.add_argument("--act-dir", default=None,
                    help="directory of cached real activations (the probe's "
                         "act/ dir). When given, act_var is MEASURED on each "
                         "Linear's real input rows instead of modelled from a "
                         "per-channel Gaussian fit; units with no cached rows "
                         "fall back to the model and are counted separately.")
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
    table, holes, meta = activation_dloss_table(
        card, args.model_path, formats, device=args.device,
        names=[n for n in costs], act_dir=args.act_dir, profile=profile,
        executed_activation_formats=executed,
        experts_per_chunk=args.packed_expert_chunk)
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
        "packed_expert_chunk": args.packed_expert_chunk,
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
