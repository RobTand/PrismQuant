# Tessera packed campaign capture — CPU evidence, 2026-09-04

Scope: PrismaQuant #183, branch `codex/pq183-packed-campaign`, based on
`fe7182175bd77d2119b91f4a086c41a36612148a` plus explicit serving-context
dependencies `86360de3` / `f229b205`. This is capture/refusal evidence, not a
GPU encoder, streaming-equivalence, exported-artifact, quality, or throughput
measurement. The campaign's main-entry packed-population refusal remains closed.

All actions used the deployed PrismaBuild runtime generation `aa6d3cfa2f77`,
Sparky, one CPU, 4 GiB, and `CUDA_VISIBLE_DEVICES=''`. Interpreter:
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`. Each command was bounded
by `/usr/bin/timeout -k 30s 240s`, with PB `--timeout-s 300 --wait-s 900`.

## Population refusal

Pre-fix action `5ef3e4795ca7b8da4b794fa36aaf8a0a4778edca582df41fbc9b70d764c14af2`
ran the real main entry on a mixed dense/packed synthetic LFM model with layer
strides 1 and 2. Both arms logged one target Linear and attempted calibration,
raising `AssertionError: started calibration with a dense-only campaign population`
at `tests/test_tessera_campaign_packed.py:42` (then-current test lines).

After guard `cac126a2`, action
`16e66f2ffd4e301da733fbfaae5721321a8c0679220e88e8db5cec580a90c8b1`
ran the packed and existing campaign test files: 33 passed, 5 skipped.
Receipt: `707c865b41cfafbddd18fef567a6faa062016a03d2a4112f47855c2513e3793d`.
All five skips are the campaign's `Tessera encodes need CUDA` cases.

## Per-expert capture

Pre-fix action `f8977f5909db4c7bbc8b1dcdec6dd47239396731dce9f45321ab00846b268d7e`
ran `python -m pytest -q -rs tests/test_tessera_campaign_packed.py` before the
capture implementation. Result: 4 failed, 2 passed, no skips. Failure line:

```text
prismaquant/tessera_campaign.py:538: KeyError: 'model.layers.2.feed_forward.experts.0.w1'
```

The same missing-module lookup failed expert-1 targets in both unobserved-expert
cases (`want_hessian=False` and `True`). The new tests execute the actual small
MoE forward and compare captured inputs to inputs recorded by each expert's
live forward; the expected Hessian does not call the capture implementation.
Each observed expert sees four rows while the scoring cap is one. Gate/up
projection views preserve the live source storage and declared split; down
inputs are the actual post-SwiGLU rows. A selected subset must retain identical
rows/Hessians. An unobserved selected expert must refuse, with hooks removed.

Green action `ac21275a8d358d9f9924d777f88ac6ea20592f98cbedda62a541dd5cbbeba082`
ran:

```text
python -m pytest -q -rs tests/test_tessera_campaign_packed.py tests/test_tessera_campaign.py tests/test_packed_expert_hook_scope.py tests/test_streaming_production_cache.py
```

Result: 41 passed, 7 skipped. Receipt:
`20bc55bf30a0cd2fd3cd5fecb8441983a7d26f63309e8cbc532bab2291061dea`.
Five skip reasons, verbatim: `Tessera encodes need CUDA`.
Two skip reasons, verbatim: `streaming production render is GPU-or-bust`.
The packed reservoir tests passed; the GPU streaming-equivalence test did not run.

Final capture/shared-helper/doc action
`adc2ee3fedc8a015f70b3546f11b49bd49ef024c02df2c659fa5bea049abb8f6`
ran:

```text
python -m pytest -q -rs tests/test_tessera_campaign_packed.py tests/test_routed_expert_namespace_aliasing.py tests/test_measure_quant_cost_packed_experts.py tests/test_architecture_doc.py tests/test_docs_staleness.py
```

Result: 41 passed, no skips. Receipt:
`f61d9c30c945e9b00d0a69195645887b29704f0c6f9986729b193a6bb294d93d`.
This additionally checked the shared routed namespace/router behavior and the
architecture/staleness assertions. No uncollected module was reported in either
green run. These are overlapping targeted populations, not an 82-test suite.

## Remaining gate

The producer must publish its explicit source-to-stack projection and validate
source/calibration/recipe/encoder identities and exact unit coverage before
packing the cached bytes unchanged. Then an actual packed-model campaign,
export, and matched-byte served measurement must establish full population
coverage and priced == written == served. No capture helper pass substitutes
for those measurements, and no production admission was opened here.

## On-path fix: respect existing profile pins

Reviewing the future packed campaign wiring found the dense target walk did
not consult `ModelProfile.is_pinned_name`. This was fixed separately, without
adding or changing any pin: the campaign now honors the existing profile
declaration before activation capture and menu construction.

Red action `43339f3343aa73919eb1ad463c474f31719b2a62130f064252621e77e394955e`
ran `tests/test_tessera_campaign_pins.py`: 2 failed, no skips. Both failed at
line 49 with `AssertionError: campaign attempted to price profile-pinned Linears`.
The LFM arm included its pinned `conv.in_proj`, `conv.out_proj`, and
`feed_forward.gate`; a changed explicit pin declaration also failed, proving
that merely hardcoding LFM's current roster would not fix the tested rule.

Green action `9bbd92484c598dfd25024597b12fd5c7388fd925134574e3952c265270369707`
ran the pin, packed-capture, and existing campaign test files: 39 passed,
5 skipped. All five skips say `Tessera encodes need CUDA`. Receipt:
`42f3a7d6c81c5dff4f5091aa14e1a9af1e4abe7ed2aa029ea36d952a0609954c`.
The same CPU-only resource envelope and interpreter described above applied.

## Capture dtype follow-up

The live-forward capture comparison also runs with original BF16 weights and
inputs, while its expected Hessian is accumulated in FP32. This checks that
the full-row consumer keeps the model's live dtype through routing/SwiGLU,
rather than replaying FP32 reservoir rows against BF16 weights.

For a genuine pre-capture replay, the PB snapshot fetched exact commit
`4682dcf9a7a91dc48956afa83e9aaf8369c7f38b` and restored only its
`prismaquant/tessera_campaign.py` into the disposable worker checkout, retaining
the new test. Both FP32 and BF16 cases failed at the old line 538 with the
same missing expert-target `KeyError`; no master baseline suite was run.

Green action `87145c9cfd5613f2ec42d33dd085441385ff6d8074a433e22231eb9cb96ca4ba`
ran the packed capture, pin, architecture, and docs-staleness test files:
28 passed, no skips. Receipt:
`3efcb00dcd0f8c45f89f0b43024c9dfd0701d38c587acea076ae5dc13e964609`.
Both dtypes ran on CPU; this remains no GPU claim.
