"""Read-only original-artifact acceptance; execute through PrismaBuild on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismaquant.joint_aura_source_transition import (
    _CONTRACT, _canonical, _sha, create_transition, load_transition,
    require_verified_transition, source_proof,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bindings = {
        "plan": {"path": str(args.plan), "sha256": _CONTRACT["plan_sha256"]},
        "prepared": {"path": str(args.run_root / "prepare/prepared.json"), "sha256": _CONTRACT["prepared_sha256"]},
        "inspection": {"path": str(args.run_root / "resume-inspection-01.json"), "sha256": _CONTRACT["inspection_sha256"]},
    }
    checkpoint = args.run_root / "checkpoints"
    receipt = create_transition(bindings=bindings, checkpoint_dir=checkpoint,
                                output=args.output_dir / "transition.json")
    config = json.loads(args.plan.read_bytes())
    verified = load_transition(receipt, config=config, plan_sha256=bindings["plan"]["sha256"],
                               prepared=bindings["prepared"], checkpoint_dir=checkpoint)
    require_verified_transition(verified, checkpoint_dir=checkpoint, resume=True, joint_activation=True)
    original = json.loads((checkpoint / "manifest.json").read_bytes())["identity"]
    actual = dict(original, git_commit=verified.execution_provenance["execution"]["git_commit"],
                  producer_source_sha256=verified.execution_provenance["execution"]["producer_source_sha256"])
    assert verified.measurement_identity(actual) == original
    changed = dict(actual, seed_base=actual["seed_base"] + 1)
    try:
        verified.measurement_identity(changed)
    except ValueError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("changed seed was admitted")
    proof = source_proof()
    result = {"schema": "prismaquant.joint_source_transition.artifact_gate.v1", "passed": True,
              "device": "cpu", "gpu_work": False, "receipt": receipt, "proof": proof,
              "execution": verified.execution_provenance, "preserved_units": _CONTRACT["preserved_units"],
              "total_units": _CONTRACT["total_units"], "changed_seed_rejected": rejected,
              "manifest_sha256_after": _sha(checkpoint / "manifest.json"),
              "prepared_sha256_after": _sha(bindings["prepared"]["path"])}
    (args.output_dir / "result.json").write_bytes(_canonical(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
