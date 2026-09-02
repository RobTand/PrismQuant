#!/usr/bin/env python
"""Materialize the packed-expert imatrix entries a per-expert-Linear
checkpoint's harvest cannot contain, so a STACK-keyed routed book can be
burned, bundled and exported against one spelling of one tensor.

WHY THIS EXISTS
---------------
Campaign rule R1 burns routed learned books per ``(layer, stack, rung)``: one
book covers the fused ``gate_up_proj`` population.  Three consumers must then
agree, byte for byte, on the imatrix that weights the fused rows:

  * the BURN (`tools/dsv4_onlaw_book_burn.py --keying stack`) learns the book
    under it and stamps its digest into the shard's ``content_guard``;
  * the BUNDLE (`build_cb_learned_bundle._stack_col_weights`) reads the packed
    target's own ``col_weights`` entry, reshaped ``(E, 1, in)``, and refuses a
    pickle that has none -- it will not re-pool the per-member vectors into a
    second spelling;
  * the EXPORT (`export_nvfp4_cb_streaming._packed_expert_col_weights`) renders
    the fused weight with the packed entry when the pickle carries one, and
    otherwise derives it as the per-expert MEAN of the gate and up vectors
    (the fused stack has one input, so the two role vectors are two samples of
    the same per-column second moment).

DeepSeek-V4-Flash exposes its experts as per-expert ``nn.Linear``s, so the
pipeline's ``harvest_cb_col_weights`` writes per-expert entries only and the
``moe_imatrix`` packed synthesis (which needs a module-level activation entry)
never fires.  Its ``cb_col_weights.pkl`` therefore has NO packed entries: the
role-keyed burn never needed one, the stack-keyed bundle cannot start without
one.  This tool closes that gap with the EXPORT's own derivation -- the same
function, called on the same per-expert entries -- so the entry written here
is exactly the tensor the export would have derived for the fused render.
The per-expert entries are left untouched: they remain the role-keyed burn's
identity and the export's per-member validation input.

The output is a NEW pickle next to the input (never in place: the input's
sha256 is stamped on every artifact of the per-role campaign) plus a
provenance sidecar that carries the input's cold-expert provenance verbatim
and records this augmentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Mapping

import torch

# The export's own pooling rule, imported on purpose (not re-derived): a
# producer and a consumer that each spell the same mean drift apart.
from prismaquant.export_nvfp4_cb_streaming import _packed_expert_col_weights
from prismaquant.routed_moe_codebooks import (
    ROUTED_ROLE_PROJECTIONS,
    ROUTED_STACK_KEYS,
)

AUGMENTATION_SCHEMA = "prismaquant.dsv4_packed_expert_col_weights.v1"
AUGMENTATION_RULE = (
    "export_nvfp4_cb_streaming._packed_expert_col_weights:"
    "per_expert_mean_of_fused_role_vectors"
)

_PER_EXPERT = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\.(?P<projection>[A-Za-z0-9_]+)$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expert_stack_members(
    col_weights: Mapping[str, object],
    profile,
) -> dict[str, dict[tuple[str, int], str]]:
    """``{packed_qname: {(projection, expert): member_qname}}`` from the
    per-expert entries in *col_weights*, keyed the way the export keys its
    ``expert_stack_members`` (packed parent per the model profile)."""

    members: dict[str, dict[tuple[str, int], str]] = {}
    for qname in col_weights:
        match = _PER_EXPERT.match(str(qname))
        if match is None:
            continue
        projection = match.group("projection")
        if projection not in ROUTED_ROLE_PROJECTIONS:
            continue
        packed_parent = str(
            profile.packed_expert_parent_for_projection(projection)
        )
        if packed_parent not in ROUTED_STACK_KEYS:
            raise ValueError(
                f"{qname}: profile maps {projection} onto packed parent "
                f"{packed_parent!r}, which is not a routed stack key "
                f"{list(ROUTED_STACK_KEYS)}"
            )
        packed_qname = f"{match.group('prefix')}.{packed_parent}"
        members.setdefault(packed_qname, {})[
            (projection, int(match.group("expert")))
        ] = str(qname)
    for packed_qname, by_member in members.items():
        experts = sorted({expert for _projection, expert in by_member})
        if experts != list(range(len(experts))):
            raise ValueError(
                f"{packed_qname}: per-expert entries are not contiguous "
                f"0..E-1 (first ids {experts[:8]})"
            )
        projections = sorted({projection for projection, _expert in by_member})
        for projection in projections:
            missing = [
                expert for expert in experts
                if (projection, expert) not in by_member
            ]
            if missing:
                raise ValueError(
                    f"{packed_qname}: {projection} has no entry for experts "
                    f"{missing[:8]}"
                )
    return members


def augment_packed_expert_col_weights(
    col_weights: Mapping[str, object],
    profile,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return ``(augmented, added)``: *col_weights* plus one ``(E, 1, in)``
    entry per packed expert target, derived by the export's own rule.

    An input entry that already exists is kept only if it equals the
    derivation exactly; a different spelling is refused rather than silently
    preferred, because every downstream consumer reads the entry as-is.
    """

    base = {str(name): torch.as_tensor(value) for name, value in col_weights.items()}
    members = expert_stack_members(base, profile)
    if not members:
        raise ValueError("no per-expert routed imatrix entries to pool")
    # Derive on a copy that has NO packed entries so the export's
    # skip-if-present branch cannot mask a disagreeing input entry.
    per_expert_only = {
        name: value for name, value in base.items() if name not in members
    }
    derived = _packed_expert_col_weights(per_expert_only, members, profile)
    added: list[str] = []
    augmented = dict(base)
    for packed_qname in sorted(members):
        value = derived[packed_qname].to(torch.float32).contiguous()
        existing = base.get(packed_qname)
        if existing is not None:
            existing = torch.as_tensor(existing)
            if (
                tuple(existing.shape) != tuple(value.shape)
                or not torch.equal(existing.to(torch.float32), value)
            ):
                raise ValueError(
                    f"{packed_qname}: input already carries a packed entry of "
                    f"shape {tuple(existing.shape)} that differs from the "
                    "export's per-expert-mean derivation; refusing to choose "
                    "between two spellings of one imatrix"
                )
            continue
        augmented[packed_qname] = value
        added.append(packed_qname)
    return augmented, added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--col-weights", required=True, help="per-expert imatrix pickle")
    parser.add_argument("--out", required=True, help="augmented pickle to write (must not exist)")
    parser.add_argument(
        "--model-dir", required=True,
        help="source checkpoint dir; the profile names the packed parents",
    )
    args = parser.parse_args()

    source = Path(args.col_weights)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"REFUSE: {out} exists; an augmented imatrix is never overwritten")
    if out.resolve() == source.resolve():
        raise SystemExit("REFUSE: --out must differ from --col-weights (never in place)")

    from prismaquant.model_profiles import detect_profile
    profile = detect_profile(str(args.model_dir))

    with source.open("rb") as handle:
        col_weights = pickle.load(handle)
    augmented, added = augment_packed_expert_col_weights(col_weights, profile)
    if not added:
        raise SystemExit(
            "REFUSE: every packed entry already present and equal to the "
            "derivation; nothing to write"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(out)

    source_sidecar = Path(f"{source}.provenance.json")
    sidecar_payload: dict = {}
    if source_sidecar.is_file():
        sidecar_payload = json.loads(source_sidecar.read_text())
        if not isinstance(sidecar_payload, dict):
            raise SystemExit(f"REFUSE: {source_sidecar} is not a JSON object")
    sidecar_payload["packed_expert_augmentation"] = {
        "schema": AUGMENTATION_SCHEMA,
        "rule": AUGMENTATION_RULE,
        "source_col_weights": str(source.resolve()),
        "source_col_weights_sha256": _sha256_file(source),
        "source_provenance_sha256": (
            _sha256_file(source_sidecar) if source_sidecar.is_file() else None
        ),
        "added": added,
        "added_count": len(added),
        "entry_shape": "(experts, 1, in_features) float32",
        "output_sha256": _sha256_file(out),
    }
    Path(f"{out}.provenance.json").write_text(
        json.dumps(sidecar_payload, indent=1, sort_keys=True) + "\n"
    )
    print(
        f"[packed-col-weights] wrote {out}: {len(augmented)} entries "
        f"({len(added)} packed entries added; input had {len(col_weights)})"
    )
    print(f"[packed-col-weights] sha256 {_sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
