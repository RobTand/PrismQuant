# Receipt: the Gridbook codebook lane, retired 2026-09-02

**Kill order.** Robert, 2026-09-02: *"put Tessera in PrismaQuant and remove
Gridbook."* Not a measurement verdict — a decision to carry **one**
non-vLLM-native wire. Sanctioned containers are now three: `compressed-tensors`
on vanilla vLLM, GGUF, and the Tessera wire on Tessera's own vLLM plugin.

**Last commit where the lane was live:** `d263f54` (`tessera/decouple-gridbook`).
**Branch:** `tessera/remove-gridbook`.
**Archive:** `archive/gridbook_lane_2026-09-02/` (its `README.md` is the
narrative; this file is the inventory and the test accounting).

---

## 1. What was archived, by area

`git mv` throughout, so `git log --follow` reaches every file.

| Area | Count | What |
|---|---|---|
| `prismaquant/` modules and assets | 45 | The pin directory `gridbook_runtime/` (pin JSON, materialized contracts 0.8.10 / 0.8.11 / 0.9.1, contract index, shell helpers); `gridbook_runtime_pin.py`, `gridbook_serving_runtime_pin.py`, `gridbook_assignment.py`, `gridbook_environment.py`, `gridbook_execution_contract.py`, `gridbook_format_contract.py`, `cb_route_status_gate.py`; the exporter (`export_nvfp4_cb.py`, `export_nvfp4_cb_streaming.py`, `cb_export_config.py`, `build_cb_learned_bundle.py`); the validators (`validate_cb_endpoint.py`, `validate_cb_performance.py`, `native_baseline_feasibility.py`); the `dspark_*` matched-performance stack (5); the `rtx4090_*` stack (6) plus `validate_rtx4090_fp8_cb*.py` (2); `dsv4_aura_cb_reprice.py`, `dsv4_w8a16_export_handoff.py`, `dense_anchored_cb.py`, `d03_exact_rate.py`; `lane_specs/nvfp4_cb.json`; four `serving_profile_specs/*.json`; one serve-dispatch example table |
| `tools/` | 8 | `dsv4_gridbook_contract.py`, `aura_cb_reprice_preflight.py`, `run_aura_cb_reprice.sh`, `cb_encode_exclusive_bench.sh`, `dsv4_onlaw_book_burn.py`, `certify_native_baseline_feasibility.py`, `dsv4_packed_col_weights.py`, `verify_dsv4_w8a16_export_handoff.py` |
| `scripts/` | 40 | All of `scripts/pb_validation/`; the CB serve/export/validation shells; and the five CB production drivers plus their serve-smoke, A/B and encode-tier companions (`run_27b_prod_nvfp4cb.sh`, `run_35b_prod_nvfp4cb.sh`, `run_hy3_prod_nvfp4cb.sh`, `run_hy3_prod_joint.sh`, `run_laguna_s21_prod.sh`, `serve_hy3_*.sh`, `serve_laguna_smoke.sh`, `serve_qwen27b_smoke.sh`, `measure_27b_ab.sh`, `measure_35b_ab.sh`, `canary_ladder.sh`, `encode_hy3_mtp_cb.sh`, `build_hy3_mtp_cb_inputs.py`, `measure_encode_tiers.py`) |
| `tests/` | 75 files (73 `test_*.py` modules + `cb_synthetic_target.py` + one `fixtures/*.json`) | Selected **by import dependency**, not by name prefix: the objective list is every file `pytest --collect-only` can no longer collect once the modules above are gone. Includes `conftest.py`'s `synthetic_cb_target` fixture helper and its two consumers, and the `fixtures/gridbook_runtime_contract.v12.*.json` |
| `docs/` | 29 | `docs/lanes/nvfp4-cb/` entire (27 files, every served measurement the lane produced), plus `docs/design/gridbook_lane_eligibility_contract.md` and `docs/design/rtx4090_fp8_gridbook_policy.md` (each carries an ARCHIVED banner) |

**Not archived, deleted outright** (no measured result attached): the
`gridbook-contract:` CI job (31 lines of `.github/workflows/ci.yml`), two
package-data globs in `pyproject.toml`, two glob patterns in
`.github/scripts/check_dist.py`, and the packaged-Gridbook-helper check in
`.github/scripts/check_installed.py`. Each site carries a dated comment.

---

## 2. What was renamed rather than removed

`prismaquant/gridbook_lane_eligibility.py` → **`prismaquant/lane_eligibility.py`**.
Despite the old name it is the **generic** closed-world lane-eligibility engine
and it is what admits the **Tessera** lane today. Changes:

- `GridbookLaneEligibilityError` → `LaneEligibilityError`.
- Dropped the publisher-specific schema (`gridbook.lane-eligibility.v3`), the
  contract-index schema, the packaged asset directory, `load_contract_index()`
  and `materialized_contract_path()`.
- `LANE_ELIGIBILITY_SCHEMAS = frozenset({LANE_ELIGIBILITY_SCHEMA_TESSERA})`.
- `ROUTE_ATTESTATION_SCHEMA`: `prismaquant.cb_route_attestation.v2` →
  `prismaquant.lane_route_attestation.v3`.
- Provenance keys `gridbook_serving_version` / `gridbook_serving_commit` →
  `serving_runtime_version` / `serving_runtime_commit` (same rename in
  `serving_profiles.py`, `allocator_candidates.py` and their tests).
- **`load_eligibility_table()` / `load_published_formats()` now require an
  explicit `contract_path=`.** With none they return
  `EligibilityTable(present=False, …)`, every unit resolves UNATTESTED, and
  export fails closed. That is the correct default for a repository that no
  longer ships a contract of its own.

Also moved rather than deleted: `_packed_expert_col_weights` from the archived
`export_nvfp4_cb_streaming` into `prismaquant/routed_experts.py` as
`packed_expert_col_weights()` (its live consumer is `expert_empirical_cost.py`,
which is **not** a CB module), and `CODEBOOK_TENSOR_PREFIX` inlined into
`read_traffic.py` with a dated comment.

---

## 3. Capability the removal actually costs

Stated plainly, because a removal that quietly drops a capability is a lie of
omission.

1. **`FP8_BLOCK_UE8M0_SOURCE` → `ROUTE_STATUS_BLOCKED`.** Its only serve route
   was the plugin's `Fp8SourceW8A16LinearMethod`. The verdict changed because
   the runtime left, not because anyone reweighed the evidence — which is kept
   verbatim and scoped to the runtime it was measured on. Per principle 1 the
   rung stays **priced** (an allocator that wants it is reporting a serving gap,
   and that signal is the point); per principle 9 export fails closed on it
   without an explicit override.
2. **`MXFP4_SOURCE` is orphaned.** It keeps `ROUTE_STATUS_BACKED` — its route is
   stock vLLM Marlin MoE, which was never Gridbook's to give — but the
   `nvfp4_cb` container was its only **writer** and the `nvfp4_cb` profile its
   only **offer**, so the producer cannot reach it end to end. Exporter gap, not
   route gap; the two are deliberately not conflated
   (`tests/test_source_passthrough_family.py` §3 says so in the file).
3. **`embed_tokens` is no longer quantizable on any lane.** It was quantizable
   only through the Gridbook lane's `quantized_embedding`
   (`docs/design/runtime_flags.md` §`ALLOW_PINNED`).
4. **Three shipped public CB artifacts can no longer be re-verified or
   re-published from this repository** (the AQUA 20 GB 27B, the DSv4 92 GB
   build, the RTX 4090 FP8-CB validation artifact). They are **not** withdrawn.
   `shipcard.open_cb_export_shipcard`, `CB_REQUIRED_SLOTS`,
   `RTX4090_REQUIRED_SLOTS` and the Gridbook distribution-identity /
   native-record / performance-record verifiers went with the lane; re-verifying
   one means reading it out of the archive.
5. **The CB ship gate is gone rather than generalised.** `shipcard`'s
   `_verify_ship_gate_record` ran only behind the lane's `is_gridbook_cb` flag.
   Removing the lane leaves two readings of that branch and only one of them is
   a *removal*: running it on every lane instead would be a **new** refusal that
   no current native card passes. The comment at its former call site records
   this. (The *fill-time gold replay* in `shipcard_cli.py` **was** generalised,
   because there the CB-only scoping was the accident — the comment above it
   always demanded what `verify()` demands, and `tests/test_publish_artifact.py`
   confirms `score_positions=all` is required on every lane.)
6. **`docs/design/constrained_pareto_allocation.md` is orphaned as policy.** Its
   mechanism is live; the normative served-parity policy it defers to
   (`format-speed-policy.md` §1) is in the archive.

---

## 4. What was deliberately NOT removed — the remainder

The CB **format / cost / render plumbing** is still in the tree, recorded as
debt **D34** in `docs/ARCHITECTURE.md` §12:

`cb_layout.py`, `nvfp4_cb_formats.py`, `nvfp4_cb_footprint.py`, `cb_ldlq.py`,
`cb_ldlq_atoms.py`, `cb_minchain.py`, `cb_warm_state.py`, `cb_banked_books.py`,
`cb_learned_promotion.py`, `cb_learned_bundle.py`, `cb_anchored_cost.py`,
`cb_ladder_cross_family.py`, `cb_imatrix.py`, `routed_moe_codebooks.py`,
`mxfp4_widen.py`, `activation_fair_pricing.py`, `aqua_activation_cost.py`,
`source_class_format_plan.py`, `trellis_*.py`, plus CB branches inside
`production_weight_cache.py`, `allocator.py`, `format_registry.py`,
`export_native_compressed.py`, `layer_config.py`, `lane_spec.py`,
`serve_constraints.py`, `decision_units.py` and `model_profiles/*` — and roughly
60 tests that exercise them.

**Why:** the excision is several hundred diffuse edits concentrated in exactly
the files the `tessera/continuous-menu` branch is rewriting
(`production_weight_cache.py` alone has ~173 CB clusters, `shipcard.py` 93,
`allocator.py` 65), and merging that against a live branch is more dangerous
than the debt. **What it means:** removing the lane made those rungs
unexportable and unservable — the property principle 9 cares about — but a
`FORMATS` menu can still name a `*_CB_*` rung and the DP can still price it.
What stops it is a *refusal* (the exporter, and the `production-render-score`
pairing guard whose `FORMATS` limb survives), not an *absence*.

**Consequently the acceptance grep is NOT met**, and here is its magnitude,
measured on the commit rather than asserted:

```
grep -rn -i gridbook --include='*.py' --include='*.sh' --include='*.json' \
     --include='*.md' --include='*.toml' --include='*.yml' . \
  | sed 's|^\./||' \
  | grep -vE '^(archive/|docs/handovers/|docs/measurements/|docs/results/|docs/audits/|CHANGELOG\.md)'
```

| scope | hits | files |
|---|---:|---:|
| whole tree, every extension | 4047 | — |
| minus `archive/`, `docs/handovers/`, `docs/measurements/`, `docs/results/`, `docs/audits/`, `CHANGELOG.md` | **892** | **147** |
| of those, **dated** — a `2026-09-02` / "retired" / "archive/gridbook_lane" / "superseded" within ±2 lines | 343 | — |
| of those, **undated** | **549** | **95** |
| undated, restricted to code (`prismaquant/`, `tests/`, `tools/`, `.github/`) | **169** | **66** |

Two clarifications the raw number needs. First, `docs/ARCHITECTURE.md` alone
carries 247 of the 892 — but 102 of those are in the **append-only provenance
stamp log** (lines 1–1251), which is history by construction in exactly the way
a handover is, and the newest stamp on it is this retirement. The body carries
145, and they are the D34 plumbing plus the DSv4 gold-contract and shipcard
sections that describe measured runs. Second, the largest undated code
residues are `allocator_candidates.py` (11), `trellis_footprint.py` (8),
`format_registry.py` (8), `cb_layout.py` (8), `routed_moe_codebooks.py` (6),
`export_native_compressed.py` (6) — i.e. D34's list, unchanged.

**No hit is in a lane, a pin, a serving profile, an exporter or a gate.**
The four things the criterion actually protects are all absent: no
`gridbook_*` module outside the archive, no pin, no `lane_specs/nvfp4_cb.json`,
no `EXPORT_CONTAINER=nvfp4_cb` path that does anything but `exit 2`.

### Four capability losses, recorded because they are losses and not debt

1. **`FP8_BLOCK_UE8M0_SOURCE` is now `ROUTE_STATUS_BLOCKED`.** Its only route
   was the plugin's `Fp8SourceW8A16LinearMethod`. The evidence strings on the
   verdict are kept verbatim and re-scoped rather than rewritten.
2. **`MXFP4_SOURCE` and `MXFP8_UE8M0_G32` are the lane's orphans.** It keeps a genuinely backed route
   (stock vLLM Marlin — never Gridbook's), but the CB container was its only
   writer and the CB serving profile its only offer, so no exporter writes it
   and no profile proposes it. `export_native_compressed._quantize_2d` ends in
   `raise ValueError(f"unsupported format: {fmt}")`, so the path fails closed.
   `MXFP8_UE8M0_G32` is the same shape with a different history: it was never
   a compressed-tensors scheme, the CB *streaming* exporter wrote its planes
   itself, and that exporter is archived. Both keep a live `FormatSpec` and a
   working render; neither has a writer.
3. **The `serving_lanes` block of a serving-profile spec has zero live
   declarations.** Checked against `d263f54`: of the ten specs that existed,
   `nvfp4_cb.json` was the *only* one that ever declared `serving_lanes` (or
   `producer_policy`). The structured per-lane `route_status` /
   `activation_contract` / `fused_mid_m` table that principle 9 reads is now a
   parser with nothing to parse — the native lane's route status has always
   come from the source-passthrough contracts instead. The parser is kept
   because it is the shape the Tessera lane will have to declare in. This is
   also what removed 17 of the 18 `test_model_profile_conformance` parametrize
   IDs and the 5 `test_serve_constraints` cases: their subject was that one
   spec.
4. **The sample-parallel incremental probe is unavailable.** Both its
   `prepare-run-contract` minter and its per-worker source-census
   revalidation were built on `prismaquant/rtx4090_artifact_census.py` — "this
   module is deliberately architecture-specific… the qualified Qwen3.8-27B
   layout is closed here" — the strict-Ada FP8-CB campaign's census, which
   imports `cb_layout`. Nothing can mint a contract now, and a contract left
   on disk from before the retirement would be admitted with one leg of its
   identity replay missing — the exact failure the check existed to refuse.
   So `incremental_probe.py --global-calibration-tensor` refuses up front with
   a dated `SystemExit` naming the archive, and
   `docs/design/sample_parallel_probe.md` carries a status banner. Reviving
   sample parallelism means giving the census a lane-independent source of
   truth, not deleting the gate.

Two further items are *kept-but-unbacked* rather than lost, and are named here
so they are not mistaken for oversights:

- `shipcard.py`'s `SAFETENSORS_CONTENT_RECEIPT_SCHEMA` /
  `validate_safetensors_content_receipt` /
  `safetensors_content_receipt_manifest` have no live caller since the strict
  RTX4090 FP8-CB publication gate retired. They are lane-neutral, and the
  schema string is stamped into receipts that exist on disk, so deleting them
  would strand those receipts unreadable rather than merely unused. A dated
  comment sits on the constant.
- `ROLE_COMPOSITE_FUSED_SOURCE_EXEMPT` still exempts `DeepseekV4Profile` from
  declaring a fused-sibling source. Its entire justification was that the
  Gridbook consumer could construct a merged Linear as independent role
  decoders. Discharging the exemption means deciding what DeepSeek-V4 should
  declare on the native lane — a producer-behaviour change, not a removal — so
  the entry is kept and re-documented as unbacked.

### Two production observations surfaced by the removal, not fixed here

- **`check_serving_shape` fails OPEN on an unknown profile id.** It catches
  `FileNotFoundError` and silently resolves to `research`, which permits every
  shape, while `serving_lane_route` / `serving_lane_catalog` fail *closed*
  (`None` / `{}`). Found while re-pointing `tests/test_serving_lane_metadata.py`:
  ten of that file's CB load-gate cases were *passing* after the profile was
  archived — not because the gate held, but because the unknown id resolved to
  a profile with no gate at all. They were deleted rather than left green.
  Nothing in production was changed: the asymmetry predates this work and
  fixing it is a refusal-semantics decision, not a removal.
- **`activation_pricing_branches["unrecorded"]`** — the stamping for expanded
  members of aggregated super items that have no candidate of their own
  (`prismaquant/allocator_candidates.py:2153`) — is now tested nowhere. It is
  profile-independent and works under `research`; its only coverage rode a
  deleted CB test and was not smuggled into a renamed one.

### One flag was added, not removed

`--serve-image` / `PQ_SERVE_IMAGE` on `tools/measure_vllm_full_kl.py` and
`tools/measure_vllm_wikitext_ppl.py`. The serving container image used to come
from the Gridbook pin; with the pin archived it has no attested source, so both
tools now fail closed immediately after `parse_args` — before any model load —
when neither the flag nor the env var is set. Defaulting to a tag would be
exactly the unattested runtime claim principle 14 forbids: the image is what
the measurement is *of*. Documented in `docs/design/runtime_flags.md` §6.

### Two fingerprint behaviours change for future runs

- Removing `gridbook_runtime_pin` from the serve manifest and shrinking
  `SERVER_ENV_ALLOWLIST` changes `fingerprint()` and `gold_producer_identity()`
  for every manifest minted from now on: a historical `serve_manifest.json`
  will not match a fresh re-serve of the same stack. That is correct — the
  manifest attests a stack that no longer includes a Gridbook wheel — but it
  means old and new manifests are not comparable by fingerprint.
- `EXTENSION_PATTERN` in `tools/serve_fingerprint.py` no longer names any
  plugin `.so`. Until the Tessera lane's basenames are added, a residency scan
  will fingerprint Tessera as "nothing resident".

---

## 5. Pipeline surface changes

- `run-pipeline.sh`: 2849 → 2374 lines. `EXPORT_CONTAINER=nvfp4_cb` is now the
  **twelfth `exit 2` gate**, in the same shape as the archived cost modes and
  pointing at `archive/gridbook_lane_2026-09-02/README.md`. All eight top-level
  `nvfp4_cb` blocks are gone (gguf-shared conditions kept their gguf half).
  `bash -n` passes.
- **Shell knobs removed** (each was read only inside an `EXPORT_CONTAINER=nvfp4_cb`
  block): `CB_EXPERT_EMPIRICAL`, `CB_SCALE_CODING`, `CB_CODEBOOK_SOURCE*`,
  `CB_CODEBOOK_BUNDLE`, `CB_CODEBOOK_ITERS/SEED`, `CB_SCALE_SWEEP*`,
  `CB_ACTIVATION_SCOPE`, `CB_LEARNED_TRAINER_VERSION`, `CB_OUT`.
  With no shell default, `allocator.py:2377` now refuses a CB menu that reaches
  it without an explicit `--cb-scale-coding` — fail-closed, which is right.
- **Shell knobs deliberately KEPT**: `CB_EXPERT_NSAMPLES` / `_SEQLEN` /
  `_SAMPLE`, `CB_LADDER_INTERP`, `CB_COL_WEIGHTS`. The GGUF lane and the generic
  imatrix harvest (`harvest_cb_col_weights`, three live call sites) read them.
- `shipcard.py`: 4608 → 2218 lines.
- **Lane declarations removed from the model profiles**: `EXPORT_LANES` in
  `model_profiles/structure.py` is now `("compressed-tensors", "gguf")`, so a
  spec that still names `nvfp4_cb` fails to load loudly. Tessera is
  deliberately **not** added: this tuple is the `EXPORT_CONTAINER` vocabulary
  and `run-pipeline.sh` has no tessera container arm (debt D33 — no exporter
  codec), so adding it here would be admission, not removal. Six
  specs dropped it from `supported_lanes` and `laguna.json` lost both its
  `preferred_lane` and its `default_serving_profile`. `glm5_next.json`'s
  re-enable rationale is rewritten as superseded, keeping its durable lesson
  (a lane the runtime does not name for the architecture serves *coherent
  garbage*, commit `9a79963`).

---

## 6. Test accounting

Both runs: `CUDA_VISIBLE_DEVICES=""`,
`PYTHONPATH=/home/rob/tessera/src:<worktree>`, `TMPDIR=/home/rob/tmp`,
`TRITON_CACHE_DIR=/home/rob/.triton-cache`, `-p no:cacheprovider`, host only
(no serves, no docker).

**Before** — a pristine detached worktree at `d263f54` (`/home/rob/pq-wt/gb-baseline`),
so the baseline could not be contaminated by the removal in progress:

```
5719 passed, 144 skipped, 3 xfailed, 22 warnings, 174 subtests passed in 1288.28s
```

**After** — `tessera/remove-gridbook`:

```
3879 passed, 115 skipped, 3 xfailed, 21 warnings, 174 subtests passed in 379.43s
```

**The counts reconcile exactly.** Collected node IDs, base vs branch:

```
base   (d263f54)                5865 collected
branch (tessera/remove-gridbook) 3996 collected
  removed                       1882
  added                           13   (renames + re-parametrisations)
  5865 - 1882 + 13 = 3996             ✓
```

and the 1882 removals split cleanly by mechanism:

| where | node IDs | what
|---|---:|---|
| the **73 archived test modules** | 1691 | selected by import/fixture dependency on an archived artifact, not by name |
| **30 files that stay** | 191 | individual tests deleted or re-pointed, each with a dated in-file note |
| | **1882** | |

No test was removed by a skip, an `importorskip`, or a collection error: the
branch has **zero** collection errors, and the archived-file total was measured
by collecting the archived paths in the pristine baseline worktree
(`/home/rob/pq-wt/gb-baseline`, d263f54): **1691 node IDs across 73
`test_*.py` modules**, of which 1686 were archived during the sweep and 5 are
`test_dsv4_serve_source_snapshot.py`, archived afterwards. The archive holds **75 files** under
`tests/` because `cb_synthetic_target.py` (the `synthetic_cb_target` fixture
helper) and `fixtures/gridbook_runtime_contract.v12.30287aa.json` travel with
them and collect nothing. Counted from git rather than from the sweep's own
bookkeeping: `git diff d263f54..HEAD --name-status -M` shows 75 paths leaving
`tests/`, none deleted outright, and 197 renames into the archive in total
(45 `prismaquant/`, 75 `tests/`, 40 `scripts/`, 29 `docs/`, 8 `tools/`) plus
the archive `README.md` = the 198 files on disk.

Tests deleted from files that **stay** (each with a dated in-file note saying
what it asserted and why the assertion no longer has a subject):

| File | Deleted | Why |
|---|---|---|
| `test_shipcard.py` | 17 | Gridbook / CB / rtx4090 / dspark card slots |
| `test_serving_profiles.py` | 2 + 1 rewritten | CB rung scope, W4A16 denial; the pin test became `test_serving_runtime_version_backs_nothing` |
| `test_serve_constraints.py` | 5 | `nvfp4_cb`-profile constraint cases |
| `test_source_class_format_plan.py` | 4 | `serving_backed_*` on CB routes |
| `test_source_passthrough_family.py` | 6 | bound to `target_profile="nvfp4_cb"`; deleted rather than re-pointed at `vllm_packed_moe`, which denies the rung and would have turned a capability loss into a green assertion |
| `test_run_pipeline_gates_execute.py` | 12 | gates behind a container that now `exit 2`s before they are reached |
| `test_run_pipeline_defaults.py` | 3 | CB export gate, CB activation scope, learned-bundle ordering |
| `test_architecture_doc.py` | 1 | `test_cb_defaults_match_the_shipped_drivers`: both sides gone (the defaults and the four drivers) |
| `test_profile_export_lanes.py` | 1 + 5 re-pointed | the lane vocabulary lost `nvfp4_cb`; the preflight cases moved to the lanes that remain, because the *mechanism* (an architecture declares its lanes, the preflight refuses an undeclared one) is what the retirement leaves standing. `test_narrow_4090_profile_inherits_cb_serializer_by_lane_identity` deleted: its serving profile was archived |
| `test_cb_layout.py` | 2 | the archived CB launch scripts and the archived `serving_profile_specs/nvfp4_cb.json`; the layout ladders themselves still run |
| `test_model_profile_conformance.py` | 18 (parametrize IDs) + 1 narrowed | see §4 — 17 of the 18 are serving-spec keys that only the CB spec ever declared |
| `test_deepseek_v4_profile.py` | 1 renamed | `test_gridbook_cb_export_lane_is_declared` → `test_export_lane_declaration_is_native_only` |
| `test_docs_staleness.py` | 1 renamed | `..._without_mirroring_gridbook_flags` → `..._without_mirroring_plugin_flags`; the pin it requires the doc to name is now Tessera's |
| `test_publish_artifact.py`, `test_serve_fingerprint_descendants.py`, `test_kl_ab.py`, `test_sample_parallel_probe*.py`, `test_container_runtime_identity.py` | 15 + 9 + 5 + 9 + 0 | the `tools/` half, done by a delegated worker |
| `test_serving_lane_metadata.py` | 34 | the whole `serving_lanes` mechanism — CB load gates, backed-rung tables, the fused-mid-M lane, selection route provenance. **Ten of these were passing, not failing**, and were deleted anyway: `check_serving_shape` fails open on an unknown profile id (it resolves silently to `research`, which permits every shape), so they had become tautologies wearing a gate's docstring |
| `test_mxfp8_ue8m0.py` | 7 deleted + 4 re-pointed | the CB container's dense-vs-packed menu split, the archived cost driver, the serving-lane declaration, and the two exporter-codec tests; the menu-mechanics tests moved to `research` after checking every reason code matched the original expectations unchanged |
| `test_runtime_snapshot_judge_split.py` | 2 | one fixture path per class (a runtime path and a judge-only path) re-pointed at live files; `test_the_snapshot_tool_is_the_one_entry_the_container_also_runs` deleted — its JUDGE_ONLY_PATHS entry is kept but now has no live launcher to witness it |
| `test_artifact_completeness_namespaces.py` | 4 | the DSpark CB sidecar "fifth namespace"; the other four namespaces are untouched |
| `test_lane_spec_and_gguf_kl.py` | 4 | the lane vocabulary lost `nvfp4_cb`; the endpoint-agnostic ship-gate test re-pointed to `tessera` — the finding is about lane *shape*, not about which plugin |
| `test_make_uniform_assignment.py`, `test_activation_fair_pricing.py`, `test_interpolated_output_mse_pricing.py`, `test_aqua_activation_cost.py` | 0 deleted, all re-pointed but one | see the note on `research` below |
| `test_wave3_selection_and_provenance.py`, `test_moe_imatrix.py`, `test_prismasnap.py`, `test_cb_serialization_contract.py`, `test_shipcard_git_provenance.py`, `test_qwen*_profile.py`, `test_prismaquant_export_native_compressed.py` | 1–2 each | archived exporter modules, archived drivers, or a `supported_export_lanes` tuple |

**`research` is the honest re-point target for the pricing tests, and it was
checked rather than assumed.** `serving_profile_specs/research.json`
deliberately declares no export lane and no format-menu restriction — its own
rationale says it exists "to keep research rungs with no served path measurable
end-to-end in the emulation harness". That is exactly the status the CB pricing
plumbing now has: priceable and renderable, servable nowhere. Verified:

```
check_format_applicability((2048,2048), "FP8_CB_K36", target_profile="research")
  -> legal=True
... target_profile="vllm_packed_moe"
  -> legal=False, reason="exporter_cannot_emit"
```

Re-pointing at `vllm_packed_moe` would have turned a capability loss into a
green assertion, which is why it was not done anywhere. One test could not move:
`test_make_uniform_assignment.py` turned on a CB **shape** rule (FP8_CB_K36's
group-256 gate) that only the archived profile declared, so the file was
re-keyed onto a live rule of the same shape — MXFP8_E4M3's 32-value MX block
under `research`, which separates the fixture's 272-column `up_proj` from its
512-column siblings exactly as the group-256 gate did. The mechanism under test
(unit atomicity, passthrough-only BF16 fallback, legality round-trip) is
unchanged; only the rung and the rule are live ones.

Narrowed rather than deleted: `test_run_pipeline_defaults.py::test_cb_unlicensed_guard_actually_fires`
keeps executing the guard's real predicate on its surviving `FORMATS` limb, and
now **also** executes the retired-container gate's own predicate — so the test
proves *which* gate refuses `nvfp4_cb` today rather than asserting a string.

---

## 7. Docs updated in the same commit (principle 13 / AGENTS.md 12)

`docs/ARCHITECTURE.md` (new provenance stamp; §1.1 container table; §3.5 the
twelfth `exit 2` gate; §3.3 defaults block; §3.2 stage rows `2d-CB` and `4/4 E-cb`
struck; §8.4 conformance matrix; §9 intro and **DIAGRAM-2 re-drawn**; §9.2
replaced by a dated retirement section; new debt **D34**), `AGENTS.md`
principle 5 (four containers → three, plus a do-not-re-add clause),
`docs/design/runtime_flags.md` (§5 banner, §6 gains `PQ_SERVE_IMAGE`, §7
rewritten around the rule that outlives the lane, shell-knob list marked
removed), `docs/design/sample_parallel_probe.md` (status banner), `docs/README.md` (lane
section, index rows, satellite table), `README.md` (container list and the
Codebook-lane section), `prismaquant/README.md`, `docs/RELEASING.md`, and
`CLAUDE.md` principles 8/9/14 plus a **§9 graveyard row** for the lane itself
(retired by decision not defect; what it won, what it cost, and the durable
lesson that a lane is priced by the runtime it obliges you to maintain).

**Not updated, and outside this worktree:** `/home/rob/CLAUDE.md` — the user's
global instructions file, which is not in the repository and cannot be
committed here. It still describes the sanctioned lanes as including Gridbook
(§7) and does not carry the graveyard row. The in-repo `CLAUDE.md` copy in this
worktree is a different file and *is* updated.

---

## 9. The merge constraints, verified mechanically

The coordinator merges this branch against `tessera/continuous-menu`, which is
being written concurrently in `/home/rob/pq-wt/tessera-continuous`. Two
constraints were given, and both are checkable rather than assertable:

**(a) Do not touch the Tessera pin.**

```
$ git diff d263f54..HEAD --stat -- prismaquant/tessera_runtime \
      prismaquant/tessera_serving_runtime_pin.py prismaquant/tessera_render.py
 prismaquant/tessera_render.py | 8 ++++----
```

`tessera_runtime/` and `tessera_serving_runtime_pin.py` are byte-identical to
`d263f54`. `tessera_render.py`'s four changed lines are **not pin logic**: they
are the import and exception-symbol renames forced by
`gridbook_lane_eligibility.py` → `lane_eligibility.py`
(`from .gridbook_lane_eligibility import` → `from .lane_eligibility import`,
`GridbookLaneEligibilityError` → `LaneEligibilityError`). `tessera_serving_contract_path`,
`require_exact_tessera_runtime_release` and `tessera_lane_attested`'s logic are
untouched, so the lane still answers `False` for every rung by the PENDING
sentinels in the pin.

**(b) Keep edits minimal in the concurrently-edited files.**

```
$ git diff d263f54..HEAD --stat -- 'prismaquant/tessera_*.py' \
      prismaquant/production_weight_cache.py prismaquant/weight_session.py \
      prismaquant/perturbed_x_cache.py prismaquant/aura_cost.py \
      prismaquant/format_registry.py prismaquant/allocator.py
 prismaquant/tessera_render.py | 8 ++++----
```

`production_weight_cache.py`, `weight_session.py`, `perturbed_x_cache.py`,
`aura_cost.py`, `format_registry.py` and `allocator.py` are **byte-identical to
`d263f54`** — zero edits, not merely small ones. This is the direct cost of
leaving D34 standing, and it is the reason the acceptance grep is unmet: those
six files hold most of the residue.

`docs/ARCHITECTURE.md` §4 is untouched: the diff's hunks jump from new-file
line 2249 to 6289, and §4.9 (the last §4 subsection on this branch) begins at
3580. §4.10 does not exist here; it is the other branch's addition, so there is
nothing to conflict with.
