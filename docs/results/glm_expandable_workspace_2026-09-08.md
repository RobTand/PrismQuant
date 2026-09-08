# GLM layer workspace with expandable CUDA segments — 2026-09-08

The bounded GLM-5.3-Flash BF16 layer-4 workspace completed all **512 original
512-token, batch-1 forwards**, returned **867 independent CPU outputs**, audited
every H/X tensor for finite values and recorded its hash, and found the next
source layer still resident (`delivery: hot`). All 867 row counts equal the
observations retained by the earlier failed run; the minimum is 1,772 rows.
This qualifies this one workspace under the recorded environment. It is not a
production capture, full-model source qualification, probe or served result.

The run selected `PYTORCH_ALLOC_CONF=expandable_segments:True` in the original
pinned producer container. No production defaults or gates changed. The shared
collector and completed-source-release options remain explicit experiments.

## Measured memory and attribution

The original run refused during output return after 508 completed groups and
760 qnames. The new run completed all 579 groups and 867 qnames, returning
43,637,538,816 Hessian bytes and 6,060,769,280 scoring-input bytes. It retained
the original 104 GiB PB reservation, 92 GiB GPU subset cap, 102 GiB conservative
cgroup-plus-CUDA guard and 8 GiB host-available floor. Its continuous sampled
conservative maximum was **101,422,903,296 bytes (94.46 GiB)**; minimum host
available memory was **27,823,779,840 bytes (25.91 GiB)**. No guard fired.
PB's separate cgroup peak was 68,102,283,264 bytes; these are different accounting
planes and must not be added to peaks measured at another time.

There are 1,017 matching group boundaries. At the last one,
`before_output_group:model.language_model.layers.4.mlp.experts.67.gate_proj`,
both runs had completed 508 groups / 760 qnames and had exactly
30,584,587,264 live CUDA bytes. Reserved bytes changed from **48,754,589,696 to
36,096,180,224**, a reduction of **12,658,409,472 bytes (11.79 GiB)**.
Inactive split bytes changed from 18,170,002,432 to zero. After full CPU return,
the new run held 27,132,675,072 live and 32,782,680,064 reserved CUDA bytes,
including its resident successor. The old run has no completed output hashes,
so full-output byte equality against that failed run is not claimed.

**The host NFS module changed between runs.** The kernel release, GPU UUID,
Torch/CUDA, source snapshot and calibration binding match, and the actual sealed
commands differ only by the allocator variable and output path. However, the
baseline nfsv4 source version was `939D67D3912BE82E3030657`, watermark `0`;
the new version was `25D7A3033B8656AA181BEFD`, watermark `5000`. Both module
build-ID notes are in the results. The matching live-allocation boundary makes
the CUDA reservation delta directly checkable, but this is not a fully isolated
allocator A/B and does not attribute the entire physical-memory result to one
variable. No repeat was run merely to obtain a speed claim.

## Profiles and host evidence

Both runs retain Torch CPU/CUDA Chrome traces and text profiles for the first
two target batches and batches 32–33. Recorded kernel durations were
1,037,933 / 1,298,732 microseconds before and 995,666 / 1,262,505 microseconds
after, respectively. The later window has 18,948 kernel events in each run;
these are bounded profile windows, not full-run GPU time. The source-forward
and output-return scopes remain distinct. Total new workspace wall time was
568.172 seconds, including its completed output audit; the failed baseline
stopped at 541.227 seconds. Those durations cover different completed work and
are not a throughput comparison.

Live Netdata covered both Sparks: 526 samples per host before and 553 after,
with zero telemetry errors. Mean whole-host CPU non-idle was 10.07% / 8.75%
on Sparky and 1.68% / 2.04% on Sparklina. Sparky mean GPU power was
41.80 W / 40.65 W against the 140 W device envelope; Sparklina was about
3.97 W. PB independently recorded 39.82 W mean and 47.71 W peak across its
larger action window, 642.938 scope CPU-seconds and complete cleanup.
No GPU saturation, energy-efficiency improvement or served-quality claim is made.

## Reproduction and verification

Baseline PB action:
`9b534c59db1b013ff0dc5fb0ccde646818246f3a26d2ddfb9fb5a517acc5bf36`.
Its exact source snapshot is `f71b9b71713faf1efcdd4c302c90e9a31b64c713`,
based on reviewed source `5a9b8e525d50694a95a0f23e2a46672fd5af409d`.
New full workspace PB:
`0f27ee249a4e54e5c11d9f3c2e59e7668a41d877b8cb94d3e22a0248907e20a0`.
Its snapshot differs only by PB closure metadata. The GPU action ran on Sparky,
six preferred CPUs with native threads bounded to one, Torch 2.13.0+cu130 /
CUDA 13.0, in image
`sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9`.
The original input binding remains
`0d52fb4d17cd81e037eb66572a03616df3f53bd1848787fa1a9ff350a1045593`;
source binding was unchanged at exit.

The tiny support check, `experiments/cuda_expandable_segments_gate.py`,
executed in the same pinned GPU container as PB
`705dda3c79d780777dda9bd7ab0743bd239a6d7a802ee098430e8843f4557aeb`.
It verified 2,097,152 exact GPU round-trip integer values, an actual expandable
segment in the allocator snapshot, and zero allocated/reserved CUDA bytes
after release. The first portable attempt could not find that literal Docker
image ID on Sparklina and did not execute the GPU check; the successful attempt
retained the original host/image dependency.

CPU trace/telemetry analysis and compile checks passed on DL380 as PB
`4a274c76f3029bb867a1c65e7d8d318a6a344cbffb5dda208427c26c8ef36393`.
The initial analysis action `0c9c04146071…` failed because the analysis helper
mistook the integer unit count for a list; the helper was fixed and rerun, with
no GPU repetition. An earlier compile-only action `16beea8a4b64…` also passed.
Root verified four successful canonical CAS receipts, actual payload hashes,
source bundles, support-check/analysis source equality and all 18 retained
before/after output artifacts. No test skips or missing runtime tools are
hidden by these checks; this PR changes experiment helpers and this report.

Evidence root:
`/mnt/shared/tessera-measurements/glm-streaming-source-20260907/`.
The exact new command and source binding are in
`workspace-prefix-07-submission.json`; full results, traces, hashes and telemetry
are under `workspace-prefix-07-expandable/`, compared with
`workspace-prefix-06-source-release/`. Analysis and root verification are
`expandable-analysis-01.json` and `expandable-root-audit-01.json`.
Re-run the checked-in analysis helper through PB with those `--before` and
`--after` directories and a fresh `--out` path. Full GPU reproduction uses the
sealed baseline source and submission command, a fresh output directory and the
recorded external runtime; current HEAD is not substituted for that source.

The next integration work is to expose the qualified collector/source lifecycle
through explicit streamed-capture configuration and validate the complete
source route and calibration census. This result supplies no completed
production capture and does not admit a shipping GLM quant.
