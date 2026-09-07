# Campaign capture finalization

The collector retained every scoring chunk while concatenating the complete
scoring-row output, then retained every GPU Hessian while copying the complete
Hessian output to CPU. On shared-memory GB10, both copies consume physical DRAM.
The finalizer now releases each consumed chunk list and each transferred GPU
Hessian. It returns freed CUDA allocator blocks after each 256 MiB of transfers
and after the remainder. The forward, accumulation, row order, dtype, counts
and activation maxima are unchanged.

Static sizing from the frozen 512 × 512 first-model census is 15.773 GiB of
checkpoint files, 31.242 GiB of Hessians and 7.749 GiB of scoring rows. The old
finalization tensor floor was 93.756 GiB, excluding runtime, allocator and
workspace overhead. This is source-derived sizing, not a measured process peak.
The sizing artifact is
`/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/baseline-sizing.json`.

The live-reference regression failed on the original collector in PrismaBuild
`69ef257b14aa398dffeaf804e19dd3d769290cab566eabf9aca0a325918ff48c`:
the next concatenation still observed the preceding unit's source chunks.
After the change, CPU tests for finalization, campaign and packed capture passed
63 tests, with 5 CUDA-required skips, in 176.40 seconds on DL380g10 (4 workers,
10 GiB reservation, native threads 1). The regression also checks exact prefix
rows, uncapped Hessians, full counts and maxima.

PrismaBuild action:
`053484ff72926578c1efcb48b6c560eb4c0e569f815ab117f201912aafb0d5c2`.
Verified action exit status 0; CAS payload:
`/mnt/shared/prismabuild-fleet/cas/blobs/47/4780f69596394c81f56d4ba8ebb4ff12dd225894ecf50329141eb500582d66c1`.

Full-model GPU before/after profiling and both-host Netdata validation are
pending; no runtime speed or measured peak-memory improvement is claimed here.
