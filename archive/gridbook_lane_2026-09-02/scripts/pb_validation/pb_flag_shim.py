#!/usr/bin/env python3
"""Arm-B runner: the canonical gold environment plus ONE DECLARED deviation.

The a24fce2 gold contract is CLOSED by design: `exact_llm_contract` pops every
allowlisted `PRISMAQUANT_*` variable and installs the canonical gold state, so
a container-level `-e PRISMAQUANT_CB_MOE_PERSISTENT_B=1` is silently reverted
to the canonical "0" before the model loads (observed: run-3 arm B's route
probe reported flag='0' and served the bridge on all 32 FP4-CB layers).  That
closure is correct for gold slots — and it means a lane A/B cannot ride the
gold tools unmodified.

This shim is the honest instrument for the B arm:

  1. run the tool's own contract activation UNCHANGED (pop + canonical
     install + attest — the attest sees and blesses the canonical state,
     which at that instant is exactly what holds);
  2. THEN apply the single declared deviation (PRISMAQUANT_CB_MOE_PERSISTENT_B
     =1), still before the serving runtime is imported, so the lane latch
     reads it at model load;
  3. rewrite the in-process contract receipt to the OBSERVED truth: the
     environment map carries the deviated value and a
     `pb_ab_declared_deviations` record names it, so the result JSON's
     serve fingerprint says what actually ran.  The two arms' contract
     hashes therefore differ — in exactly and only the declared flag —
     instead of arm B forging arm A's fingerprint.

Anything else about the serve (kwargs, util, eager, batched tokens, kv cache,
max_logprobs) stays the gold contract's, byte for byte.

Usage:  pb_flag_shim.py <tool.py> [tool args...]
"""
from __future__ import annotations

import importlib.util
import os
import sys

FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B"

# vLLM's EngineCore child (spawn start method) re-imports the parent's
# __main__ via runpy with __name__ == '__mp_main__' and multiprocessing's
# own argv. Unguarded, this file then read sys.argv[1] as a tool path and
# died building a spec from bootstrap garbage, killing the engine child at
# startup (observed run 4, 2026-08-17). The guard makes the child's re-run
# a no-op; the deviated flag still reaches the child via inherited
# os.environ, which is how the route probe already proved dispatch.
if __name__ == "__main__":

    tool_path = sys.argv[1]
    sys.argv = [tool_path] + sys.argv[2:]

    spec = importlib.util.spec_from_file_location("pb_ab_deviated_tool", tool_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)   # __name__ != "__main__": main() does not run

    _orig_activate = mod._activate_dsv4_gridbook_contract


    def _activate_with_declared_deviation(args):
        kwargs = _orig_activate(args)
        if not getattr(args, "dsv4_gridbook_contract", False):
            return kwargs
        os.environ[FLAG] = "1"
        receipt = getattr(mod, "_DSV4_GRIDBOOK_CONTRACT", None)
        if receipt is not None and "pb_ab_declared_deviations" not in receipt:
            env = dict(receipt.get("environment", {}))
            env[FLAG] = "1"
            receipt = dict(receipt)
            receipt["environment"] = env
            receipt["pb_ab_declared_deviations"] = {
                FLAG: {
                    "canonical": "0",
                    "applied": "1",
                    "reason": "persistent-B requalification A/B, arm B "
                              "(NATIVE-PARITY single-arm measurement)",
                },
            }
            mod._DSV4_GRIDBOOK_CONTRACT = receipt
        return kwargs


    mod._activate_dsv4_gridbook_contract = _activate_with_declared_deviation

    raise SystemExit(mod.main())
