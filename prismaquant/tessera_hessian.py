"""One Hessian, one identity, two callers.

The Tessera encoder's shipping default is *activation-aware*: given a unit's
``H = XᵀX`` it applies LDLQ (sigma 1.0, block 32) plus an exact full-Hessian
row-scale refit, and served KL on Qwen3-0.6B went 0.1512 -> 0.1046 at
byte-identical wire bpp.  Weights-only encodes stay byte-identical, so an H is
not a tuning knob: a rung priced without one is a price of *different bytes at
the same format name*.

Two PrismaQuant paths hand Tessera a Hessian -- the anchor campaign
(``tessera_campaign``) and the production render
(``tessera_render.render_tessera_production``, reached from
``ProductionWeightCache``).  Principle 8 says the surrogate, the KL validation
and the shipped bytes must be one rendering; if those two paths form H
differently, or from different calibration draws, they are two renderings with
one name and nothing raises.  So both call the functions here:

* :func:`hessian_from_rows` -- the only place ``XᵀX`` is formed, in one dtype
  and one accumulation order.
* :func:`calibration_identity` -- the only place the draw is named, and it
  emits exactly the triple Tessera's ``ActivationSource`` requires
  (``text_sha256``, ``fit_tokens``, ``fit_ids_sha256``).  Tessera refuses a
  provenance missing any of them, and the refusal is the point: an identity of
  ``None`` compares equal to another ``None``, which is how a merge guard goes
  vacuous.
* :func:`encoder_kwargs` -- the only place an ``ActivationSource`` is turned
  into encoder keywords, so the recipe (sigma, block, refit objective) is read
  from Tessera's own defaults rather than restated on either side.

The LDL factorisation is the expensive part and it is **rung-independent**: one
unit's H yields one ``ldl`` and one ``refit_metric`` that every rate on that
unit reuses.  Callers that sweep rungs compute the kwargs once per unit --
a twelve-anchor surface otherwise pays for twelve identical block-LDLs.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

__all__ = [
    "HESSIAN_IDENTITY_FIELDS",
    "activation_source",
    "calibration_identity",
    "encoder_kwargs",
    "encoder_recipe",
    "hessian_from_rows",
    "text_sha256",
    "token_ids_sha256",
    "unit_name_for",
]


def _identity_fields() -> tuple[str, ...]:
    """Tessera's own required provenance fields, read from Tessera."""
    from tessera.export import HESSIAN_IDENTITY

    return tuple(HESSIAN_IDENTITY)


#: The three fields that name *which* Hessian shaped the bytes.  Read from
#: ``tessera.export.HESSIAN_IDENTITY`` rather than typed: a second spelling of
#: a merge guard's key set is how the guard goes vacuous.
HESSIAN_IDENTITY_FIELDS = _identity_fields()


def hessian_from_rows(rows) -> "Any":
    """``XᵀX`` for one unit from its calibration activation rows.

    The **one** formation.  fp32 regardless of the rows' dtype (a bf16
    accumulation of a 512x4096 outer product loses the small eigenvalues LDLQ
    is there to see), flattened to ``[tokens, in_features]`` so a ``[batch,
    seq, in]`` capture and a pre-flattened one give the same matrix, and
    un-normalised because Tessera's ``compensate.regularize_hessian`` takes a
    count of its own -- pre-dividing here would be a second spelling of the
    encoder's normalisation.
    """
    import torch

    flat = rows.detach().to(dtype=torch.float32)
    flat = flat.reshape(-1, int(flat.shape[-1]))
    return flat.t() @ flat


def text_sha256(text: str) -> str:
    """The calibration corpus's identity."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def token_ids_sha256(batches: Sequence) -> str:
    """The identity of the token ids the forward pass actually saw.

    Over the *ids*, not the text: two tokenizers over one corpus are two
    different calibrations and a text sha alone would call them the same.
    Both travel, because the reverse is also true -- one tokenizer over two
    corpora can collide on neither.
    """
    import torch

    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.to(dtype=torch.int32).cpu().numpy().tobytes())
    return digest.hexdigest()


def calibration_identity(text: str, batches: Sequence, *, fit_tokens: int,
                         **extra: Any) -> dict:
    """The provenance dict an ``ActivationSource`` needs, plus anything extra.

    Exactly ``HESSIAN_IDENTITY_FIELDS`` are load-bearing; ``extra`` (model,
    seqlen, split role, seed) rides along unread by Tessera and is what makes
    the block legible in a receipt.
    """
    identity = {
        "text_sha256": text_sha256(text),
        "fit_tokens": int(fit_tokens),
        "fit_ids_sha256": token_ids_sha256(batches),
    }
    missing = [f for f in HESSIAN_IDENTITY_FIELDS if f not in identity]
    if missing:
        raise ValueError(
            f"tessera.export.HESSIAN_IDENTITY requires {missing}, which this "
            "function does not emit -- Tessera's required set changed and this "
            "is the one place that has to change with it"
        )
    return {**dict(extra), **identity}


def unit_name_for(qname: str) -> str:
    """Tessera's Hessian key for a PrismaQuant qname.

    ``ActivationSource`` keys on the tensor name minus one trailing
    ``.weight``, which is a PrismaQuant qname exactly.  Stated through
    Tessera's own helper so the two cannot drift apart.
    """
    from tessera.export import ActivationSource

    return ActivationSource.unit_name(f"{str(qname)}.weight")


def encoder_recipe() -> dict:
    """The activation-aware recipe an ``ActivationSource`` applies by default.

    Read from Tessera's own dataclass defaults, so a receipt quoting "sigma
    1.0, block 32, exact-H refit" is quoting the encoder rather than a comment.
    """
    from tessera.export import ActivationSource

    source = ActivationSource(
        hessians={}, provenance={f: "" if f.endswith("sha256") else 0
                                 for f in HESSIAN_IDENTITY_FIELDS})
    return {
        "ldlq_sigma": source.ldlq_sigma,
        "ldlq_block": source.ldlq_block,
        "refit_objective": source.refit_objective,
        "refit_reach_floor": source.refit_reach_floor,
    }


def activation_source(hessians: Mapping[str, Any], identity: Mapping[str, Any],
                      **overrides: Any):
    """Build Tessera's ``ActivationSource``.  **The one construction.**

    ``hessians`` is keyed by qname (``model.layers.0.mlp.up_proj``), which is
    already Tessera's unit-name convention.  ``identity`` must carry
    :data:`HESSIAN_IDENTITY_FIELDS`; Tessera refuses otherwise and this does
    not pre-empt that refusal with a friendlier one -- the message there names
    the capture tool.

    ``overrides`` reach the dataclass unchanged, so an ablation can move the
    sigma or the refit objective and the exported config records what it moved
    (``ActivationSource.config_block``).
    """
    from tessera.export import ActivationSource

    return ActivationSource(
        hessians=dict(hessians), provenance=dict(identity), **overrides)


def encoder_kwargs(source, qname: str, in_features: int, device="cpu", *,
                   scale_plane) -> dict:
    """One unit's encoder keywords from an ``ActivationSource``.

    ``scale_plane`` is **required and has no default**, and it must come from
    the recipe the encode itself resolved -- ``tessera_wire_recipe(family,
    rung).scale_plane``, threaded as an object, never a second lookup.  What
    it selects is the refit objective, and Tessera keys that by plane because
    the two answers were measured separately and disagree: ``hessian`` (the
    exact quadratic, closed-form on a CHANNEL row scale) against ``h^1.0`` (a
    diagonal power, because under the full H the LUT plane's coupled blocks
    converge to a worse point of their own quadratic).  A caller that does not
    name the plane cannot be served a default, so Tessera refuses -- and the
    refusal is right: pricing a unit under the other plane's objective prices
    an artifact the export does not ship (principle 8).

    **What is hoistable, and what is not.**  Two of the four keywords are a
    function of the Hessian alone -- ``ldl`` (the block-LDL factorisation,
    which is the expensive part) and ``ldl_block`` -- and the other two are a
    function of the Hessian *and the plane*.  The plane is a function of the
    grid, not of the rung: measured over every family PrismaQuant allocates
    (``realisable_rungs`` x ``tessera_wire_recipe``), the plane is constant
    across every rung of every family, and differs across families --
    ``channel`` on ``TESSERA_E4M3_K1``, ``lut16`` on both E2M1 families and
    every free grid.  ``TESSERA_E2M1_K2`` changes *body* at the coset cap
    (WINDOW below q896, TCQ at it) and keeps its plane through the change.

    So the hoist out of a rate sweep survives -- no rate reaches this call and
    the signature has nowhere to put one -- but it is keyed by ``(unit,
    plane)``, not by unit.  The bound is one factorisation per unit per
    distinct plane actually encoded, which on a single-family surface is one.

    A missing key or a wrong width raises inside Tessera, by design: both
    would otherwise be silent, and a wrong key encodes the unit weights-only,
    which raises nothing and quietly prices a different artifact.
    """
    return source.for_unit(f"{str(qname)}.weight", int(in_features), device,
                           scale_plane=scale_plane)
