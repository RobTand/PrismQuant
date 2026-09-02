#!/usr/bin/env python3
"""Arm-B runner: the canonical gold environment restored to 0.8.9 DEFAULTS.

The a24fce2 gold contract is CLOSED by design: `exact_llm_contract` pops every
allowlisted `PRISMAQUANT_*` variable and installs the canonical gold state —
which pins PRISMAQUANT_CB_MOE_PERSISTENT_B=0 and PRISMAQUANT_CB_GEMV=inherited
(the 0.8.8-era defaults its numbers were taken on).  This shim measures the
candidate DEFAULT state instead: gridbook 0.8.9's unset-means-auto semantics.

  1. run the tool's own contract activation UNCHANGED (pop + canonical
     install + attest);
  2. THEN apply the declared deviations — PERSISTENT_B=auto, CB_GEMV=auto —
     the exact meaning 0.8.9 gives an UNSET flag, applied before the serving
     runtime is imported so the lane latches read them at model load.
     PRISMAQUANT_CB_FP8_GEMV_V2 needs no deviation: it is absent from the
     gold allowlist, so the canonical install never touches it and the
     candidate's unset-means-auto default applies on its own;
  3. rewrite the in-process contract receipt to the OBSERVED truth, with a
     `pb_ab_declared_deviations` record naming every deviation (including
     the allowlist-invisible FP8 GEMV selector, for the reader).

Everything else about the serve stays the gold contract's, byte for byte.

Usage:  pb_default_state_shim.py <tool.py> [tool args...]
"""
from __future__ import annotations

import importlib.util
import os
import sys

DEVIATIONS = {
    "PRISMAQUANT_CB_MOE_PERSISTENT_B": "auto",
    "PRISMAQUANT_CB_GEMV": "auto",
}
REASON = ("gridbook 0.8.9 default-state validation: unset means auto for "
          "the persistent-B lane and both GEMV selectors")

# vLLM's EngineCore child (spawn start method) re-imports the parent's
# __main__ via runpy with __name__ == '__mp_main__' and multiprocessing's
# own argv.  Unguarded, this file then read sys.argv[1] as a tool path and
# died building a spec from bootstrap garbage (observed run 4, 2026-08-17).
# The guard makes the child's re-run a no-op; the deviated flags still reach
# the child via inherited os.environ.
if __name__ == "__main__":

    tool_path = sys.argv[1]
    sys.argv = [tool_path] + sys.argv[2:]

    spec = importlib.util.spec_from_file_location(
        "pb_default_state_tool", tool_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)   # __name__ != "__main__": main() not run

    _orig_activate = mod._activate_dsv4_gridbook_contract

    def _activate_with_default_state(args):
        kwargs = _orig_activate(args)
        if not getattr(args, "dsv4_gridbook_contract", False):
            return kwargs
        os.environ.update(DEVIATIONS)
        receipt = getattr(mod, "_DSV4_GRIDBOOK_CONTRACT", None)
        if receipt is not None and "pb_ab_declared_deviations" not in receipt:
            env = dict(receipt.get("environment", {}))
            env.update(DEVIATIONS)
            receipt = dict(receipt)
            receipt["environment"] = env
            devs = {
                flag: {"canonical": "0" if "PERSISTENT" in flag
                       else "inherited",
                       "applied": value, "reason": REASON}
                for flag, value in DEVIATIONS.items()
            }
            devs["PRISMAQUANT_CB_FP8_GEMV_V2"] = {
                "canonical": "absent from the gold allowlist (never popped, "
                             "never set)",
                "applied": "unset -> auto (the candidate's 0.8.9 default)",
                "reason": REASON,
            }
            receipt["pb_ab_declared_deviations"] = devs
            mod._DSV4_GRIDBOOK_CONTRACT = receipt
        return kwargs

    mod._activate_dsv4_gridbook_contract = _activate_with_default_state

    # Second declared deviation: the CANDIDATE OVERLAY. The serve fingerprint
    # verifies every installed gridbook source file against the pinned wheel's
    # RECORD, and this arm deliberately overlays the candidate worktree over
    # the installed package — so the canonical provenance refuses with
    # "differs from RECORD", which is TRUE and must be recorded, not forged.
    # On exactly that refusal, substitute a candidate-overlay attestation:
    # the wheel pin attestation as passed, the candidate commit, and a
    # sha256 manifest of the overlaid package sources, so the receipt names
    # the code that actually served.
    import hashlib
    import json as _json
    import pathlib as _pathlib

    _sf = sys.modules.get("serve_fingerprint")
    if _sf is None:
        raise SystemExit(
            "pb_default_state_shim: serve_fingerprint was not imported by "
            "the tool; cannot install the declared-overlay provenance wrap")
    _orig_provenance = _sf.gridbook_distribution_provenance

    def _overlay_provenance(attestation):
        try:
            return _orig_provenance(attestation)
        except ValueError as exc:
            if "differs from RECORD" not in str(exc):
                raise
            import gridbook
            root = _pathlib.Path(gridbook.__file__).parent
            manifest = {}
            for path in sorted(root.rglob("*")):
                if (path.is_file() and "__pycache__" not in path.parts
                        and path.suffix in
                        {".py", ".cu", ".cuh", ".h", ".json"}):
                    rel = path.relative_to(root.parent).as_posix()
                    manifest[rel] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
            digest = hashlib.sha256(_json.dumps(
                manifest, sort_keys=True).encode()).hexdigest()
            return {
                "schema": "pb_ab.candidate_overlay_provenance/1",
                "declared_deviation": {
                    "kind": "candidate source overlay over the pinned wheel",
                    "candidate_gridbook_commit": os.environ.get(
                        "GB_CANDIDATE_COMMIT", "(env GB_CANDIDATE_COMMIT "
                        "unset)"),
                    "reason": REASON,
                    "canonical_refusal_it_replaces": str(exc),
                },
                "wheel_pin_attestation_as_passed": attestation,
                "overlaid_source_files": len(manifest),
                "overlaid_source_manifest_sha256": digest,
            }

    _sf.gridbook_distribution_provenance = _overlay_provenance

    raise SystemExit(mod.main())
