# Calibration row counts without Hessians (#255)

The shared `_collect_activations` returned zero `token_counts` whenever
`want_hessian=False`, although it had observed inputs and calibrated maxima.
The increment was inside the Hessian branch. This blocked the mixed-family
sanity preparation's truthful six-input scale receipt with `max_rows=0`.
That preparation remains tracked separately in #253.

PB red `c69b6445f40408be4d672c5778f2363fc2c6681da2dd506c0b7f1649a6ec18d0`
ran a genuine small CPU torch dense model with six observed input rows. Both
no-H subcases (`max_rows=0` and `2`) returned zero instead of six; the full-Gram
H-mode control passed. This was the intended regression, not a dependency or
collection failure. Overall exit 1, no successful CAS claimed.

The count now increments for every nonempty input before either the H branch
or scoring-row cap. H-mode Gram computation is unchanged. No-H mode still
returns no Hessians; its returned counts and campaign calibration provenance
now report actual observed rows. Export-input writing only serializes Hessian
counts when real Hessians exist. No inferred expert rows are introduced.

PB green `862727591e5ccd98238eb82c0041418d698b21bd841e5d888bb99c822b42a038`
passed both real-tensor tests, zero skips, on the qualified GB10 CPU interpreter
with GPU visibility disabled, one CPU/4 GiB and native threads bounded to one.
Terminal exit zero and resource cleanup complete. The actual 458-byte CAS
payload was read and rehashed to
`e981fd25570c1542cbbd27c90e5be910bc1d29b87eb6e187d40c8dae56feeb69`;
receipt `3733e49559350c19e47bf4a23cd2b9bb186a8e45cbab77bf131f55c4b6e71c54`.
Full private receipts are retained under
`/home/rob/tessera-runs/mixed-lfm-237-2026-09-06/preparation-evidence/`.
The command was the qualified Python interpreter running
`-m unittest discover -s tests -p test_tessera_capture_row_counts.py -v`
through the published `pbrun.py` client, with a `gb10` environment dependency
and PB selecting the eligible worker. No GPU measurement was performed here.
