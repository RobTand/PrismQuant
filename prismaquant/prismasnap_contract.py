"""Small lane-admission helpers for PrismaSnap-prepared source checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
from functools import wraps
import inspect


PRISMASNAP_PROVENANCE_JSON = "prismasnap_provenance.json"


def require_verified_prismasnap_if_present(model_dir: str | Path) -> None:
    """Fail before expensive pipeline work on unverified/corrupt Snap input."""
    marker = Path(model_dir) / PRISMASNAP_PROVENANCE_JSON
    if not os.path.lexists(marker):
        return
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"PrismaSnap provenance is not a regular file: {marker}")
    from .prismasnap_validation import validate_prismasnap_checkpoint

    validate_prismasnap_checkpoint(model_dir, require_verified=True)


def refuse_prismasnap_for_unvalidated_lane(
    model_dir: str | Path,
    *,
    lane: str,
) -> None:
    """Fail closed when a snapped source enters a non-native exporter.

    The measured production treatment is the native compressed-tensors
    NVFP4/FP8/BF16 pipeline.  GGUF and Gridbook/codebook transforms have not
    cleared the same A/B; fixed-book CB was materially harmful in the pilot.
    No research/force flag bypasses this release boundary.
    """
    marker = Path(model_dir) / PRISMASNAP_PROVENANCE_JSON
    if os.path.lexists(marker):
        raise RuntimeError(
            f"PrismaSnap source is not admitted to the {lane} lane: {marker}. "
            "Use the native compressed-tensors pipeline or an unsnapped source."
        )


def refuse_prismasnap_lane_before_output(*, lane: str):
    """Decorate programmatic exporters so refusal precedes their transaction."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            model_dir = bound.arguments.get("model_dir")
            if model_dir is None:
                raise RuntimeError("PrismaSnap lane gate could not bind model_dir")
            refuse_prismasnap_for_unvalidated_lane(model_dir, lane=lane)
            return function(*args, **kwargs)

        return wrapped

    return decorate


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    require_verified_prismasnap_if_present(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
