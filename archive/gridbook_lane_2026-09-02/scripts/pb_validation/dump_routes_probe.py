#!/usr/bin/env python3
"""Dispatch-proof probe: which route served the routed-MoE prefill?

Gridbook records every dispatch on the layer (`emit_route`, twelve setattrs)
but NOTHING consumes `read_route` -- provenance nothing reads is a confession
log, not a gate. This probe is the consumer.

Serve-config identity is BORROWED, not imitated: the probe imports the gold
KL tool (`tools/measure_vllm_full_kl.py` from the reviewed snapshot) and boots
through its `_load_llm` + `_activate_dsv4_gridbook_contract`, so the engine
the routes are read from is configured exactly like the KL arms
(gpu_memory_utilization=0.84, eager, max_num_batched_tokens=512, fp8 KV,
max_model_len=seqlen+16). A first version of this probe used LLM() defaults;
0.92 utilization on a 121 GiB unified box drove MemAvailable under the 4 GiB
watchdog floor and the watchdog correctly killed it.

vLLM V1 runs EngineCore in a subprocess where the layer objects are invisible;
VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps it in-process (probe-only setting).

Run once per arm:
    (default)                          -> expects policy=bf16_grouped_bridge
    PRISMAQUANT_CB_MOE_PERSISTENT_B=1  -> expects symbol=cb_moe_persistent_b_prefill
                                          on every FP4-CB routed layer; the 11
                                          FP8-CB routed layers keep the bridge.

Exit codes: 0 = routes observed and printed; 2 = no route records found
(a probe that observed nothing must not pass).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ["PB_PROBE_MODEL"]
OUT = sys.argv[2] if len(sys.argv) > 2 else ""
ROOT = os.environ["PQ_RUNTIME_PRISMAQUANT_ROOT"]
SEQLEN = 512
# Captured BEFORE the KL tool's contract activation: the a24fce2 gold
# contract is CLOSED (exact_llm_contract pops every allowlisted PRISMAQUANT_*
# var and installs the canonical state, PERSISTENT_B=0 included), so the
# container-level -e is the INTENT channel only.  Run-3 arm B proved this the
# hard way: the flag reached the container and the probe still measured the
# bridge, because activation popped it before model load.  When the intent is
# set, the probe applies the same declared-deviation wrap pb_flag_shim.py
# gives the measurement tools: canonical install first, then the one flag,
# then the receipt rewritten to the observed truth.
WANT_PB = os.environ.get("PRISMAQUANT_CB_MOE_PERSISTENT_B", "") == "1"
FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B"


def load_kl_tool():
    path = os.path.join(ROOT, "tools", "measure_vllm_full_kl.py")
    spec = importlib.util.spec_from_file_location("measure_vllm_full_kl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap_with_declared_deviation(kl) -> None:
    orig = kl._activate_dsv4_gridbook_contract

    def wrapped(args):
        kwargs = orig(args)
        if not getattr(args, "dsv4_gridbook_contract", False):
            return kwargs
        os.environ[FLAG] = "1"
        receipt = getattr(kl, "_DSV4_GRIDBOOK_CONTRACT", None)
        if receipt is not None and "pb_ab_declared_deviations" not in receipt:
            env = dict(receipt.get("environment", {}))
            env[FLAG] = "1"
            receipt = dict(receipt)
            receipt["environment"] = env
            receipt["pb_ab_declared_deviations"] = {
                FLAG: {"canonical": "0", "applied": "1",
                       "reason": "persistent-B requalification A/B, arm B"}}
            kl._DSV4_GRIDBOOK_CONTRACT = receipt
        return kwargs

    kl._activate_dsv4_gridbook_contract = wrapped


def main() -> int:
    kl = load_kl_tool()
    if WANT_PB:
        wrap_with_declared_deviation(kl)
    args = SimpleNamespace(
        model=MODEL, dsv4_gridbook_contract=True, quantization=None,
        dtype=None, gpu_memory_utilization=None, enforce_eager=None,
        max_num_batched_tokens=None, max_logprobs=None,
    )
    llm = kl._load_llm(args, max_model_len=SEQLEN + 16)

    from vllm import SamplingParams
    # One prefill above every batch threshold (routed T > 16, dense M > 128)
    # inside the contract's 528-token window; max_tokens=1 so the last forward
    # IS the prefill and the route records reflect it.
    prompt = "the quick brown fox jumps over the lazy dog " * 44   # ~400 tok
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))

    def find_model(root):
        import torch.nn as nn
        seen: set[int] = set()
        stack = [root]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, nn.Module) and any(
                    hasattr(m, "_cb_route_state") for m in obj.modules()):
                return obj
            for attr in ("model_executor", "driver_worker", "worker",
                         "model_runner", "model", "engine_core", "engine",
                         "llm_engine", "core"):
                child = getattr(obj, attr, None)
                if child is not None and not isinstance(
                        child, (str, int, float, bool)):
                    stack.append(child)
        return None

    from gridbook.nvfp4_activation_contract import ROUTE_FIELDS, read_route

    model = find_model(llm)
    if model is None:
        print("PROBE-ERROR: could not reach the torch model via V1 internals",
              file=sys.stderr)
        return 2

    hist: collections.Counter = collections.Counter()
    per_layer = {}
    for name, mod_ in model.named_modules():
        rec = read_route(mod_)
        if rec is None:
            continue
        key = (rec["kind"], rec["policy"], rec["symbol"], rec["state"])
        hist[key] += 1
        per_layer[name] = {f: rec[f] for f in ROUTE_FIELDS}

    if not hist:
        print("PROBE-ERROR: zero route records observed -- a probe that saw "
              "nothing must not pass", file=sys.stderr)
        return 2

    flag = os.environ.get("PRISMAQUANT_CB_MOE_PERSISTENT_B", "")
    print(f"\n=== ROUTE HISTOGRAM (PRISMAQUANT_CB_MOE_PERSISTENT_B={flag!r}) ===")
    for (kind, policy, symbol, state), n in sorted(hist.items()):
        print(f"{n:5d}  kind={kind:8s} policy={policy:28s} "
              f"symbol={symbol:36s} state={state}")
    if OUT:
        with open(OUT, "w") as f:
            json.dump({"flag": flag, "histogram": [
                {"kind": k, "policy": p, "symbol": s, "state": st, "count": n}
                for (k, p, s, st), n in sorted(hist.items())],
                "per_layer": per_layer}, f, indent=1)
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
