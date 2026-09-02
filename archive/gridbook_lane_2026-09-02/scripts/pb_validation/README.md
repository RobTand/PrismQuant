# Persistent-B / GEMV served-validation instruments (2026-08-14 .. 2026-08-18)

Committed verbatim from their run locations (`/home/rob/dq-runs/dsv4-flash-0731/pb-validation/`
and `/home/rob/dq-runs/gridbook-fp8-qual/`) so the instruments that produced the
gridbook 0.8.9 promotion evidence are tracked, not host-only. Paths inside are
host-bound by design — these are evidence records of what ran, not portable tools.

- `run_pb_ab_92gb.sh` + `pb_flag_shim.py` + `dump_routes_probe.py` — the FP4
  persistent-B same-session served A/B on the DSv4 92 GB body (kl_mean −0.051 %,
  PPL −0.30 %).
- `run_pb_ab_clean_fp8.sh` — the flag=1 legs on the shipped clean 87 GB body
  (r1–r4): died at load on the 11 per-role FP8-CB layers, the discovery that made
  `auto` the only defaultable semantics.
- `run_pb_ab_clean_default.sh` + `pb_default_state_shim.py` +
  `dump_routes_probe_default_state.py` — the 0.8.9 default-state served leg
  (r5–r7): env fully unset, 32 FP4-CB layers on the lane, 11 announced bridge
  fallbacks; kl_mean +0.17 % / kl_p99 −0.03 % / PPL −0.06 % vs the gold record.
  The shim carries the two declared deviations (canonical-env pin of the old
  defaults; candidate-overlay provenance for the worktree gridbook) — see the
  gold-contract documentation-doctrine notes in the shipcard/serve_fingerprint
  history.
- `run_089_release_qual.sh` — the gridbook 0.8.9 release qual (VERDICT-089 PASS
  at 23a3955). Phase 5 pins PYTHONPATH to this repo because
  `test_cb_gemv_v2_cuda` importorskips `prismaquant.nvfp4_cb_formats`, and
  refuses an all-skip: a green-looking zero-passed file is a FAIL.
