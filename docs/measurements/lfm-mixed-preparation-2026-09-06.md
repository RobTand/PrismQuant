# LFM mixed-family sanity preparation (#253)

This opt-in preparation creates inputs for a PrismaBuild-owned export campaign.
It does not launch the export or select partitions. Its explicit fixed layout
quantizes all 2,178 eligible body matrices: 22 routed stacks / 2,112 E4M3 q1024
projections, first two dense FF blocks / six E2M1x2 q896 matrices, and remaining
60 dense BF16-grid q1792 matrices. Real BF16-grid trellis differs from plain
BF16 passthrough. The producer owns classification, source layout, expert
projection and 52 dense fusion owners; all retained source tensors are named.

The producer is sealed Tessera 6faa5ce314cadeee8a190cbeadcf6cde3a333efb.
Preparation uses the previously qualified immutable 47dd producer image and its
offline WikiText cache. Existing PQ `_calibration_tokens(32, 512, 0)` supplies the
same text and draws as issue 183. `_collect_activations` retains zero scoring
rows, computes no H, and observes every dense input row and its maximum.
`_static_input_scales` receives the producer fusion roster through its profile
interface, preserving one conservative scale per fused owner. Six F32 scales
travel with their policy, exact corpus/token/tokenizer identities and 16,384
observed rows per unit. No expert row coverage is claimed. The full model must
be GPU resident throughout this one coherent capture.

The preparation seals the plan, producer projection, source identity, retained
population, scale tensor and calibration receipt, host PB snapshot/image/source
identity and cProfile. Outputs must be fresh. The source manifest covers every
external Tessera file, including producer scripts; neither container Git nor an
unverified worker path supplies source authority. Host observation preserves
PB CPU affinity, image Config/RootFS identity, both-host Netdata, power, process
scope and exact owned-container cleanup. Full terminals remain private; any
publication copy must redact scope tokens before Git staging.

`python experiments/lfm_mixed_preparation.py host --mode preflight ...` runs
producer classification and source checks with GPU visibility disabled (4 GiB).
The same explicit arguments with `--mode calibrate` use a four-CPU, 64-GiB GPU
action with bounded native threads. Required arguments are `--model`, `--out`,
`--tessera-repo`, `--tessera-source-manifest`, its SHA via
`--tessera-source-manifest-sha256`, and two `--netdata-url` values. The CLI is
executed only inside a PB action. Final `preparation-seal.json` is the handoff;
a preflight seal cannot substitute for a `mode=calibrate` seal with scales.

The fixed weights-only plan is a sanity experiment, not per-Linear empirical
selection or an optimized production recipe. EUGR dense4/16 qualification and a
mixed dense/routed census gate are prerequisites owned separately by the
integration task. Existing official-image dense cells do not qualify EUGR.
No runtime pin or production menu changes are included here.
