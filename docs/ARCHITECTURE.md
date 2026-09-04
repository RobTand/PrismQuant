# PrismaQuant Architecture

Re-stamped (2026-09-04, `codex/pq87-physical-ab`) for the opt-in bounded
physical boundary controller. `experiments/pq87_physical_ab.py` freezes and
hashes native BF16 source before starting one read-only-model server, then
requires the client to observe that same content. It journals every completed
HTTP response and cap step, uses explicit one-sequence/4096-context/1-GiB-KV
limits, and bounds the whole physical action to 1200 seconds plus cleanup.
Timeout or refusal remains inconclusive; cleanup verifies only its uniquely
named container. CPU controller tests are not a served result, and no shipping
policy changes. Gate: `tests/test_pq87_physical_ab.py`.

Re-stamped (2026-09-04, `codex/pq-87-control-relative`) for the paired
instrument's **exact native weight identity** (§7.2). Each boundary binding
carries the existing model-metadata hash plus the shared safetensors content
manifest; its compound `artifact_id` is recomputed during replay. The client
hashes before and after measurement and uses the existing weight-stat
attestation around each hash and across the interval to refuse mutation,
including same-size replacement or change-and-restore. Generic shipcard
identity is unchanged. These observations attest source content during the
measurement, not what a pre-existing server loaded earlier: the physical
campaign must pin/freeze the same artifact before launch. Gate:
`tests/test_measure_boundary_control_identity.py`.

Re-stamped (2026-09-04, `codex/pq-87-control-relative`) for shared serve
process discovery. `tools/serve_fingerprint.py` now identifies the vLLM
executable, Python module/script or worker process title rather than searching
arbitrary argv payloads. A measurement client's `--image spark-vllm@...` and
a shell's quoted launch command no longer join the server process census or
change its session identity. Gate: `tests/test_serve_fingerprint_descendants.py`.

Re-stamped (2026-09-04, `codex/pq-87-control-relative`) for the **opt-in
paired boundary instrument** (§7.2). `prismaquant/boundary_control.py`
records a frozen prompt/seed schedule and replays every raw scored outcome.
The BF16 budget grows until no sampled completion is censored, bounded by
the live model context minus chat-tokenized input lengths; exhausting that
bound or the explicit iteration backstop is inconclusive. A candidate uses
the identical final cap and schedule, with common campaign/host/stack and
tokenization bindings. `tools/measure_boundary_control.py` captures live
process/residency manifests before and after, and can also run the historical
raw-request arm in the same control session. Per-stratum defect deltas are
**advisory only**: this adds no pass threshold, fills no shipcard slot, and
does not replace the historical mandatory 64/zero policy. No physical served
result is claimed by implementation or CPU tests.

Review follow-up on the same branch binds the actual chat token-ID arrays,
not merely their lengths, and requires the same producer-source closure digest
across arms. The BF16 label also refuses an observed non-BF16 dtype or an
explicit quantization override even when the source config says bfloat16.

Re-stamped (2026-09-04, `codex/pq-87-control-relative`) for the boundary
probe's input refusal (§7.2). Empty prompt/repetition populations, non-finite
or nonpositive sampling temperatures, nonpositive token caps and malformed
defect bounds now fail before HTTP, rather than producing an empty pass or
asking the endpoint to validate the measurement contract. This does not
change any shipping threshold. Gate: `tests/test_ship_boundary_behavior.py`.

As of: 2026-09-04 · `codex/pq-ship-integration`. Stamps
follow, newest first, each recording its own branch and date.

Re-stamped (2026-09-04, `codex/pq-175-ci-tessera`) for the **pinned Tessera
CI dependency** (§8.6; RobTand/prismaquant#175). Both CPU jobs resolve the
single `TESSERA_DEV_PIN_COMMIT` literal with a stdlib-only source parser before
PrismaQuant can be imported, check out that exact commit with ordinary checkout
access without persisting credentials, and install its bytes before
PrismaQuant. Tessera remains private until its fixes and audit are complete;
checkout failure is expected until Rob publishes it. No private-repository
secret is required by this workflow. A failed
pin resolver stops the step before publishing any checkout output; its exit
status is preserved by a separate shell assignment, so an empty ref cannot
fall back to Tessera's default branch. Gate:
`tests/test_ci_tessera_install.py` (pre-fix: the workflow contained no pinned
Tessera checkout).
Re-stamped (2026-09-04, `codex/pq-tessera-v4-consumer`) for **development
admission's predicate refusal** (§5.7). The shared parser retained cell
predicates, but the shape-free development reader discarded them. It now
refuses non-empty predicates before constructing an unconditional menu; the
export resolver continues to evaluate predicates against unit facts. The
reviewed contract has no predicated cells, so its current menu is unchanged.
Test: `test_tessera_dev_predicates.py`.

Re-stamped (2026-09-04, `codex/pq-tessera-v4-consumer`) for **Tessera lane
schema v4 consumption** (§5.7, §9.4). Development and export readers now share
the cell parser. Executed `(symbol, decoder)` pairs and explicit residency
scope are retained; malformed launches, missing plugin/residency declarations
and overlapping scopes refuse. Per-unit route resolution requires a residency
for v4 and records the selected launches. The development answer is reviewed
against Tessera `1221d2a`, including its window-GEMV native extension, while
the exact release pin remains PENDING and no routed-MoE attestation is added.
Tests: `test_lane_eligibility_v4.py`, `test_tessera_contract_v4.py`.

Re-stamped (2026-09-03, `muse/pq-172-173-append`) for the **append-identity
follow-ups** (§5.4; RobTand/prismaquant#172, #173). Two configurations the
#170 guard could not describe now have a section each. Streaming packed
appends (`module_acts_override`, one materialized module per call) record a
`packed_expert_streaming_append` union keyed by the experts-module qname
each call saw — shared budgets/mode plus each layer's activation-snapshot
hash — so consecutive layers merge while the same module under a different
budget or snapshot refuses. The resident `packed_expert_append` section binds
the fit corpus per pair (`pair_fit_calibration_hashes`, `"qname|FMT"` to
`calibration_data_hash`): disjoint pairs under a different calib merge with
each pair keeping its own hash — the sanctioned M4 lazy gap-fill renders
disjoint FP8 pairs on the render-split calib — while the same pair under a
different hash refuses, since the bytes on disk are the first render's.
Deliberately still unbound: the render narrowing on both paths. Gates:
`tests/test_packed_append_streaming_and_pair_identity.py` (8, each shown
failing before the fix).
Re-stamped (2026-09-03, `muse/pq-146-stamp`) for the **pre-guard
trust-admission record** (§5.4; RobTand/prismaquant#146, remaining leg). The
guard admitted a directory holding shards but no sidecar on trust with a
WARNING, then wrote the current identity as if guarded since birth — from that
moment the directory was indistinguishable from one guarded throughout, and a
downstream artifact could not say part of it rested on unverified shards. The
first write now records `pre_guard_admission` in the sidecar itself (the
trusted shard count plus their sorted `.pt` names; a fresh directory carries
no such record), and every later sidecar write carries it forward: the append
merge preserves the stored file, the base adoption re-validates the record
through the same validator, and the match path rewrites nothing — so the guard
cannot launder its own confession on the next run. The record is history,
never render input: it is excluded from the comparison, so a guarded directory
carrying it still resumes cleanly. Warn-and-admit stays, since refusing would
strand every existing cache. No downstream consumer is wired: nothing outside
`production_weight_cache.py` reads `render_identity.json`, so no shipcard or
receipt echoes the admission; the sidecar is the record. Gates:
`tests/test_production_cache_render_identity.py` (7 new: a hand-built v1
sidecar resumes, pre-guard records while fresh does not, the append merge and
the base adoption preserve, no false mismatch, malformed refuses).
Re-stamped (2026-09-03, `muse/pq-170-append`) for the **production cache
append-identity guard** (§5.4; RobTand/prismaquant#170). #146's sidecar
covered the dense fill only, so the MTP and packed-expert appends — which
stream further shards into the same `--cache-dir` after that fill returns —
could land two configurations in one directory under a still-matching
sidecar. Each append now compare-or-writes its own `mtp_append` /
`packed_expert_append` section of `render_identity.json` (budgets, gate
corpus, MTP activation source, hooked enumeration) before any capture or
render, and refuses naming the field on mismatch; the base fill compares
its base projection only and preserves both sections. Deliberately unbound:
the render narrowing (MTP stripes replace scope; the M4 lazy gap-fill
renders disjoint subsets with identical bytes) and the packed fit corpus
(existing shards are never re-rendered, and the sanctioned gap-fill runs on
the render-split calib). Gates:
`tests/test_production_cache_append_identity.py` (8, each refusal shown
failing before the fix).

Re-stamped (2026-09-03, `muse/pq-146-cache`) for the **production cache
directory's render-identity guard** (§5.4; RobTand/prismaquant#146). Resume
admitted a unit by file presence alone, so the same `--cache-dir` resumed
under a different `--render-scope`, include-file, lever string, `max_act_rows`
or calibration mixed units rendered against different row sets into one
directory, and the settings were stamped into `cache.metadata` only at the
END of a fill and never read back at the start. `build_production_cache_
render_identity` now canonicalises every value-bearing render input — scope,
requested formats, resolved levers, mechanism order, calibration hash, the
hooked-enumeration digest and count, the exact rendered `qname|FMT` pairs, and
`max_act_rows` — into a `render_identity.json` sidecar written atomically on
the first fill and compared on every later one, immediately after the
`cache_dir` mkdir and before any shard is read or written. A mismatch names
the exact field and says to rebuild rather than resume; an unreadable sidecar
refuses fail-closed. **A directory with no sidecar (every cache built before
this) is admitted on trust with a WARNING** — refusing them would strand every
existing 90 GB cache, and that trade is recorded here rather than hidden in the
code. Gates: `tests/test_production_cache_render_identity.py` (9),
`tests/test_cb_cache_pair_resume_identity.py` (updated for the earlier, louder
refusal).

Re-stamped (2026-09-03, `muse/pq-140-licence`) for the **licensed per-member
promotion** (RobTand/prismaquant#140 -- closing the second prose copy the
#132 stamp below left tracked). Serving-unit promotion moved a unit onto one
Tessera family with a rate per member on the strength of its own docstring,
and fired for packed-expert groups the contract does not cover. The groups are
now tagged by kind where they are built, and the family branch runs only for
fused-kind components under the pinned contract's `fused_module` word for the
rate field (`q256: per_member`, read through `tessera_menu.fused_module_licence`;
`None` with no pin is the absence of a licence, not a permissive default). A
packed-expert component takes the uniform path -- the contract's
`expert_parallel` is a closed world with no unit in it -- a re-tightened
`q256` collapses fused groups to one rung and re-stales the dev pin with the
field named, and a component unioning a fused group with a packed one refuses
rather than picking a side. Gates:
`tests/test_allocator_family_licence.py` (packed lands on one rung, fused keeps
its rates, shared/absent licence collapses both, mixed refuses, each shown
failing before the fix); the pin-level refusal is #132's
(`tests/test_tessera_menu.py::test_a_withdrawn_fused_module_licence_refuses_at_the_pin`).

Re-stamped (2026-09-03, `muse/pq-162-slots`) for the **derived lane-slot
vocabulary** (§7.1; RobTand/prismaquant#162 closed). `shipcard.ALL_SLOTS` and
`shipcard.LANE_SCOPED_SLOTS` are no longer two enumerated tuples: the
vocabulary is every `shipcard_slot` any `lane_specs/<lane>.json` declares,
union the base set, and each derived slot names its replay in
`shipcard.LANE_SLOT_VERIFIERS`. `route.census`'s entry is the #136
priced-vs-served replay, dispatched through the registry. A lane that declares
a slot with no verifier is REFUSED at parse time, the same way
`LaneGate.from_dict` refuses a null slot with no `unrecorded_reason` -- so a
fourth lane with a novel gate is one spec file plus one verifier, and neither
roster exists to be forgotten.


Re-stamped (2026-09-04, `codex/pq-87-boundary-chat`) for the **sampled
boundary-behavior request contract** (§7.2; issue #87, still open). The first
physical run showed that the original implementation posted bare prompts to
`/v1/completions`; vLLM applies no chat template there, so healthy DSV4 and
Qwen references both failed 30/30 before artifact behavior was exercised. The
check now posts a user `messages` body to `/v1/chat/completions`, explicitly
enables thinking and reasoning output, and recovers the raw boundary semantics
from either unparsed `message.content` or vLLM's structured `reasoning` /
`reasoning_content` split. The endpoint plus request/response schema are filed
in the check metrics and replayed by `shipcard.verify`, so a copied zero count
from the old route cannot certify an artifact. **This fixes only the request
contract.** The same run disproved the universal 64-token/zero-defect policy
(healthy DSV4: 7/30 at 64; stock Qwen3-8B: 10/15 even at 600), so those values
remain fail-closed historical defaults pending a same-session control-derived
cap and control-relative verdict; no artifact is promoted and #87 is not
closed on this commit. Gate: `tests/test_ship_boundary_behavior.py`.

Re-stamped (2026-09-03, `pq132-fused-licence`) for the **fused module's
per-member rung licence, read from the contract** (§4.10; PrismaQuant #132
consuming RobTand/tessera#37). The group knapsack gave each member of a fused
group its own rung on the strength of its own docstring -- "they can disagree
about the rate" -- which is a claim about what the serving runtime's loader
accepts, carried in a field no gate can read. Tessera publishes the same fact
machine-readably (contract v6's `fused_module.fields`, checked on the Tessera
side against `tessera.serving.scheme.FUSED_MODULE_FIELDS`, the dict the loader
itself gates on), and the fold now reads it. The read runs **both** ways, and
the second direction is where the defect was: holding only the FAMILY fixed
does not honour the block's `shared` fields, because `wire_recipe` is a
function of `(grid, q256)`. Measured on the parent commit, a family-only fold
put a WINDOW-body rung (`TESSERA_E2M1_K2_R850`) and a TCQ one (`R896`) in one
module, on the frontier at 30 bytes -- two `body` values where the contract
publishes one
(`tests/test_allocator_sibling_aggregation.py::test_one_family_is_not_one_decoder_so_two_bodies_never_share_a_module`).
The fold now partitions each member's menu by the shared fields a rung decides
and sums only within a class; a `shared` field it cannot evaluate refuses
outright rather than being skipped; and `fused_module` is inside
`contract_answer`, so a contract that re-tightens `q256` re-stales the pin with
a named field instead of silently changing what may be allocated.
**Default behaviour changed**: with no contract pinned -- production today,
since no Tessera RELEASE tag exists -- the fold returns no options at all where
it previously folded, and stamps `__licence__` saying so. The stamp is written
for every group with a Tessera rung on a member's menu; a group whose members
carry only stock formats never asks the licence question and its super item is
byte-identical to before. `mixed_rung_receipt`
is still `false`: the relaxation is proven by a decode identity, not by a
serve. The solver's second, still-prose copy of the same licence
(`allocator_solver._resolve_family_group`) is out of scope here and tracked as
PrismaQuant #140.

Re-stamped (2026-09-03, `muse/pq-147-145-hooks`) for the **activation hook
scope gates** (§5.4; issues #145, #147). The §5.4 stamp below closed with the
hook-scope field as provenance nothing read; the three consumers now exist.
`union_production_cache` refuses a bundle whose shards' hook digests disagree
— equal digests is the rule, since each stripe hooks the whole enumeration
and renders its slice, and the merged scope sums the slices so a
full-coverage union reads exactly like the unstriped build it reproduces.
`synthesize_production_render_cost_payload` stamps the priced contract with
the hook enumeration it was priced from and refuses a baseline priced from a
different rendering. `_production_cache_fingerprint` records the enumeration
the shipped bytes were rendered against, which reaches the artifact through
the render levers the shipcard echoes. The same session closed the packed
path the dense fix did not reach: `_PackedExpertActivationCollector` built
its hook set render-narrowed while `force_format` hooked every module, so a
frontier build and an assignment-scoped export build sampled different rows
(measured 2/2 layer-1 expert tensors diverging on a frozen two-layer MoE);
the collector now hooks the full visible module set with `store_qnames`
narrowing only storage, stamps `activation_hook_scope_packed`, and the union
merge requires agreeing packed digests. What this does NOT do: guard the
cache directory itself — that is #146, closed by the stamp below. Gates:
`tests/test_activation_hook_scope_gates.py` (5),
`tests/test_packed_expert_hook_scope.py` (4).

Re-stamped (2026-09-03, `muse/pq-156-158-replay`) for the **three receipt
blocks the producers wrote and nothing replayed** (§7.1, §7.2;
RobTand/prismaquant#156, #157, #158). Every `ship_gate` receipt carried its
threshold contract, check ledger, token evidence and endpoint binding, every
`native_export.*` receipt its arm identity and generation evidence, and every
card a ten-key build forensic block — and `verify` read one key of the third
block and none of the other two, so a wrong-server receipt, a mislabeled arm,
a lowered threshold, an empty ledger, or an unvalidated Fisher correction all
verified clean. `verify` now replays all three: the `ship_gate` threshold
contract against the fixed catastrophic bounds, the exact four-check ledger,
positive scored-token evidence and the endpoint binding's presence-and-shape;
the `native_export` arm, residency and generation evidence against the slot;
and the build block's Fisher policy bit (refusal) plus the forensic hashes'
shapes. Staged deliberately: the replays read what the producers already
stamp, so previously-verifying minimal fixtures now fail and the suites carry
producer-shaped records instead. The one honest limit is stated in §7.2: an
offline `verify` cannot bind a well-formed receipt to the wrong server, only
refuse one that names none. Gates: `tests/test_shipcard.py::test_ship_gate_*`,
`::test_native_export_*`, `::test_build_*`, shown failing before the wiring.

Re-stamped (2026-09-03, `muse/pq-156-158-replay`) for the **ship-gate receipt
replay** (§7.2; RobTand/prismaquant#156). `verify` closed the one universal slot on a bare
`passed` flag while the producer filed a threshold contract, a check ledger, token evidence
and an endpoint binding nobody read. The replay now runs on every lane — tool identity and
producer commit, exact catastrophic bounds against the filed thresholds, a complete passed
ledger, perplexity numerics re-cleared with a positive scored-token count and known
non-spec-decode state, and the endpoint binding present and shaped — with fixtures derived
from the producer's own `DEFAULT_*` bounds and check names rather than restated. Gates:
`tests/test_shipcard.py` (refusals for lowered bounds, missing binding, incomplete ledger,
unscored perplexity).

Re-stamped (2026-09-03, `claude/pq-tessera-lane`) for the **lane-aware ship
record** (§7.1, §9.2, §9.4, D33; RobTand/prismaquant#119 in part, #162 filed).
An earlier stamp below records the Tessera lane's addition. The gate set that lane
declared was enforced by nothing: `lane_specs/tessera.json` gave `route.census`
-- principle 12's second leg on this lane -- `shipcard_slot: null`, and the arm
exited about 130 lines above the driver's shipcard block, so no card existed to
carry that gate even had it named a slot. Four links hold now.
`lane_spec.LaneGate.from_dict` refuses a null `shipcard_slot` that carries no
`unrecorded_reason`, so "advisory by construction" is a value rather than a
silence. `route.census` names its slot and the derived vocabulary knows
it (`shipcard.lane_scoped_slots`, #162 closed). `python -m prismaquant.lane_shipcard open --lane tessera` opens a record
whose slots are the lane's own gates, and the tessera arm runs it before
`exit 0` (`run-pipeline.sh:2452-2458`). And `shipcard.required_slots` UNIONS the
lane's slots with the base set, so a lane can add a requirement and never
subtract one; a declared slot with no registered verifier RAISES rather than
being filtered away (`shipcard.lane_gate_slots`, #162 closed: the vocabulary
is derived from every lane spec and each derived slot names its replay). Two rosters moved into
the declaration they belong to: `wired_architectures`, from two sets in
`tests/test_profile_export_lanes.py`, and `producer_tools`, from a hardcoded
`for` loop in the driver into preflight gate 4
(`tessera_export_lane.require_producer_tools`) with each tool's `stability` and
`tracking_issue`. `model_profiles.structure.RETIRED_EXPORT_LANES` names the wall
`nvfp4_cb`'s code went behind, so the vocabulary refusal says where to look.
What this does NOT do: run a gate. Each Tessera gate needs a fresh vLLM
container with the pinned plugin editable-installed, which is R16's open half
and stays with #119, and both producer tools remain
`unsupported_experiments` until the release-tag commit promotes them. Gates:
`tests/test_lane_gate_recording.py` (34), `tests/test_profile_export_lanes.py`.

Re-stamped (2026-09-03, `muse/pq-136-141-142-receipts`, rebased onto the
#137/#138/#139 fingerprint and #119 lane-record work) for the **three
Tessera receipt fields nothing consumed** (§7.1, §7.4; PrismaQuant #136,
#142 — #141 filed against work that has not landed here, so it is recorded
as open, not closed). The serve manifest recorded only the `.so` basenames
it *found*, so a resident serve that kept serving on the named substitute
decoder fingerprinted differently with nothing naming what ran; the pin now
transcribes the contract's `when_unavailable` block beside prefix/glob/match
under the same contract→pin refusal, and the fingerprint tool reads that
block from the transported pin JSON (no carried constant survives) and
records per-row expected-vs-found status on every manifest -- excluded from
both fingerprints, so no recorded fingerprint moves. `kl_ab` replays the
block and names the substitute in the §7.4 refusal instead of implying the
drift band. And the route `decoder` Tessera stamps on every route record
finally has a consumer: the priced-vs-served comparison
(`prismaquant/tessera_route_receipt.py`) refuses a census run on a known
substitute, without a decoder, or disagreeing with the priced routes either
way, and the receipt closes the lane's required `route.census` slot --
required through the lane union, not a rate-axis special case -- replayed
from the carried records at publication. Gate:
`tests/test_tessera_substitute_decoder.py`,
`tests/test_tessera_route_receipt.py`, each shown failing before the fix.

Re-stamped (2026-09-03, `claude/pq-120-tp-fail-closed`) for the **fourth
fail-open seam of prismaquant#120**. The stamp below says `check_serving_shape`
now fails closed "like its three siblings"; a fourth seam had not been counted.
`allocator_candidates._tensor_parallel_applicability` — the Tessera
shard-granularity gate (§4.10) — caught the same `FileNotFoundError` and
loaded `research`, so a typo'd `--target-profile` priced every Tessera rung
under the research world size and left the export to refuse what the
allocator had already chosen. It now returns the `profile_mismatch` refusal
`check_serving_format` returns, naming the id in `detail` and `provenance`;
`profile_id=None` still resolves to `research`, the declared default. Zero
`load_serving_profile("research")` fallbacks remain in the package. Gate:
`tests/test_tessera_menu.py::test_tp_gate_refuses_an_unknown_profile_instead_of_becoming_research`,
shown failing on the parent commit before the change.
Re-stamped (2026-09-03, `pq133-native-extensions`) for the **serve
fingerprint's Tessera extensions, now derived** (§7.4, §9.4; PrismaQuant #133
consuming RobTand/tessera#28). The pin's `serving_extension_basenames` was a
hand-written claim about which CUDA libraries the Tessera plugin loads, with a
refusal only at the last link (pin → tool) — the link that was already sound —
and the hand-written value was wrong by one character. The tool matched it by
substring over the whole mapped path, which is not the predicate the runtime
publishes. Contract v7's `native_extensions` is now the source: the reader
refuses a contract that does not publish the table, the table is part of
`contract_answer`, `require_pin_native_extensions_match_contract` refuses a pin
that is not the contract's table in either direction (under the dev pin at
runtime, and in the test suite always — the release admission path does not call
it), and the tool applies the `match` rule the table names and refuses a rule it
does not implement. The
`when_unavailable` half — a resident serve that substitutes
`torch_materialize_stock`, and `TESSERA_SERVE_MODE` being absent from the
manifest entirely — is recorded as open, not closed (#142, #143).

Re-stamped (2026-09-03, `claude/pq-121-uniform-gate`) for the **byte-matched
uniform control on the shipcard** (§7.1; PrismaQuant #121, the closure of the
served 2.00x loss in #117 / RobTand/tessera#1). A rate-axis artifact (one whose
`quantization_config.quant_method` is `tessera`) now carries a required
`uniform_control` slot: the verdict of the allocated arm against the
byte-matched uniform arm, measured on the gold lane. `verify` REFUSES a card
whose allocation lost to its control, and refuses the ways a loss could be
dressed up -- a control that is not byte-matched, a self-widened tolerance, a
control whose `model_sha` is the candidate's own, a `passed` flag that
disagrees with the KL it sits beside, a control that never served. The hatch
is `shipcard_cli override-control`: basename re-typed at stamp time (the
`--force-unverified` ceremony), stamped with the forgiven ratio and bound to
the card's `model_sha`; `verify` re-checks the binding, never the directory
name, because the publisher verifies a snapshot copy under a randomised
basename. `publish_artifact` prints the verdict beside the bpp claim. The
control's KL comes from outside the card, as a gold-shaped record replayed
through the same verifier the candidate's is (`fill-control`). What the gate
does not do: fix the allocator -- the oracle ceiling for this loss is 0.941x,
so the closure is a gate, not a better cost. Gate:
`tests/test_shipcard_uniform_control.py` (27).
Re-stamped (2026-09-03, `claude/pq-126-plane-charges`) for the **TCQ forest** in the
byte accountant (§8.4). `wire_overhead_q256` had four terms and no forest, so
the ALPHABET and DESCENDANT planes a trellis body writes per unit were priced
at zero and `tessera_footprint` handed the plane builder an empty descendant
blob; the cap rung of E2M1x2 was 512 B light per unit and the arity-1 E2M1
rungs 24-44 B (RobTand/prismaquant#126). The forest is now sized by
`tessera.grammar.forest_plane_bytes`, which makes every TCQ recipe
shape-dependent: `recipe_is_shape_free` and `artifact_q256_bounds` are deleted
rather than made to return a floor nothing may price bytes with.
Amended (2026-09-03, `claude/pq-126-plane-charges`, at merge): two things the
forest term exposed. (i) `expand_tessera_menu`'s **W(n<=A)** rule compared
`bits_per_param` -- planes and forest included -- against the route's
activation width, so the day the forest was charged the attested E2M1x2 cap
rung (R896, 3.5 b/wt) left the A4 menu and every family's cap rung would have
left its own route; the rule is about the weight's coding rate and now
compares `rung / 256`. (ii) The accountant asked Tessera's grammar whether a
`(rung, columns)` pair is realisable only on the TCQ body (to size the
forest); a WINDOW rung the encoder refuses at a width (257/256 over 320
columns needs 5/4 columns at rate 2) was priced and offered. `_schedule_rates`
now walks `bresenham_rate_schedule` for every body, refusing in
`TesseraFormatError`, and the sweep in `tests/test_tessera_forest_bytes.py`
asserts the two accountants agree on which rungs EXIST as well as what they
cost, with a bound derived from 256 | 768, 1024 rather than typed.
Re-stamped (2026-09-03, `claude/pq-130-stripe-rows`) for **what the striped
cache-union identity does not bind** (§5.4; issue #130). The campaign identity
binds coverage, source, calibration, code, settings, levers, scope and formats,
and the section read as a guarantee of equivalence with an unsharded render. It
is not one. Measured by rendering, not by reading the diff: a stripe narrows the
collector's *hook* set, which the shared priority generator makes every later
Linear's row sample depend on, and 4/4 units diverge in rows and rendered bytes
— for a non-prefix stripe, for a *prefix* stripe once `NSAMPLES >= 2`, and for
the shipping `--render-scope assignment` default against a `format-menu` build
of the same recipe. Hooking the full enumeration and narrowing only the render
reproduces the unstriped bytes exactly. The claim is recorded where the
mechanism is described, not only where the defect was found.
Re-stamped (2026-09-03, `claude/pq-130-stripe-rows`) for the **activation hook
set** (§5.4; issues #130, #135). A narrowed render used to keep different
calibration rows than a whole one, so a stripe — and the shipping
`--render-scope assignment` default — rendered different bytes than a
`format-menu` build of the same recipe. Measured by rendering, not by reading
the diff. The collector now hooks the caller's whole `qnames` enumeration and
narrowing arrives as `render_assignment`/`render_qnames`, which makes the bytes
of a `(qname, fmt)` pair a function of the enumeration and the calibration and
not of which subset a call renders. Caches stamp the hooked enumeration
(`activation_hook_scope`) so a reader can tell two renderings apart, which
`render_scope` alone could not. The claim is recorded where the mechanism is
described, not only where the defect was found.
Re-stamped (2026-09-03, `claude/pq134-memo-key-space`) for the **rung-keyed
memo bounds** (§4.10; RobTand/prismaquant#134). The two memos next to the ones
below were sized by round numbers against the wrong space:
`tessera_render.tessera_rung_is_serialisable` at 4096 and
`tessera_formats._recipe_for` at 512, both keyed on a rung of *any* family a
name can name -- 13,068 over the twelve `enumerate_grid_space()` builds, not
the 6916 of the four the menu admits. Measured before: the serialisable memo
took **0 hits against 6916 misses on every pass**, and `_recipe_for` hit only
within a rung (34,580 hits, 6,920 misses on its *second* pass over one shape).
Both are now sized by `grid_space_rung_keys()`, counted off the grid space at
the grammar's cap so the sizing cannot reenter the memo it sizes; after, a
second pass and a second *shape* both recompute nothing. The mechanism moved to
the leaf module (`tessera_formats.lazily_sized_cache`) so there is one spelling
of it. tessera#46's own two memos were re-checked against the wider space and
are correctly bounded: a non-serialisable family is refused at the first menu
gate, so a 13,068-rung grid-space pass at tp=2 fills the geometry memo with
exactly 6916 entries and no more.

Re-stamped (2026-09-03, `claude/pq-menu-cache-bound`) for the **menu memo
bound** (§4.10; RobTand/tessera#46). Admitting the 16-bit family took one
2048x1024 unit's research menu from 3055 rungs to 6764 and left both per-rung
memos at `maxsize=4096`, which is smaller than one shape's menu: a pass evicted
its own entries and a repeat pass hit nothing (cold 10.06 s, warm 9.958 s). The
bound is now a product of two named factors — the widest menu one shape can
produce, computed from `menu_families()`, times a stated shape retention — so
it moves with the family roster and cannot fall under one shape. The two memos
are sized apart on measured entry cost (351 B flat vs 8.7–131.6 KB, O(cols)),
and the memory each commits is stated where it is chosen.

Re-stamped (2026-09-03, `tessera/decouple-gridbook`) for the **family roster
in §8.4's plane bullet**, which counted eleven families because it was measured
on a worktree predating the day `ANCHOR_BUDGET_BITS` became the TCQ body's
refusal rather than the grid's. Re-measured on the
merged tree: twelve, `TESSERA_BF16_K1` on WINDOW/CHANNEL climbing L=14→15→16 to
its 4096 terminal with its plane unmoved, so the `(unit, plane)` memo key still
holds and E4M3 and BF16 correctly share entries. The test that attests the same
fact now derives its roster from `menu_families()` instead of naming five
formats. *A doc re-stamped in lockstep with a stale measurement is current and
wrong.*

Re-stamped (2026-09-03, `claude/tessera-plane-drift`) for the **activation
seam's plane axis** (§1.1, §8.4). Tessera's 2026-09-02 release made the refit
objective a map keyed by scale plane and deleted the two CHANNEL-only guards
this document quoted. PrismaQuant read half that change: `for_unit` refused
every caller (the symptom that raised) while `rung_accepts_hessian` went on
pricing both E2M1 families weights-only (the one that did not). The plane is
now threaded from the recipe the encode itself resolved, the predicate is
derived from what the pinned `ActivationSource` emits rather than restated, and
the campaign's per-unit memo is keyed `(unit, plane)`. The sections that read
"CHANNEL-plane-only" are retracted in place. *Currency is not truth*: this file
was re-stamped twice in lockstep with that false belief.

Re-stamped (2026-09-03, `tessera/decouple-gridbook`) for the **Tessera dev
pin's move from identity to answer, and the third attested rung** (§4.10,
§10.x; RobTand/tessera#38, #14). The pin compared the environment's value
against an exact commit and the contract file's sha256 against a recorded one.
Both legs fired on *identity*, and the thing they name is an editable checkout
on the same box, so every Tessera commit that touched a `detail` string, a
changelog paragraph or `contract_version` turned the attested path off and
seven tests red in a repo nobody was editing — while the rungs' meaning had not
moved. That is P14 read backwards: prose explains and is never a value a gate
reads, so a prose edit is not a thing to re-review. `contract_answer()` now
draws the line mechanically and `TESSERA_DEV_PIN_ANSWER` is the reviewed
answer as a literal whose git diff *is* the review; a commit that moves no
answer passes silently, one that moves any answer refuses naming the field —
**including a family or cell that is merely added**, which is additive for a
reader and not for an admission gate. The commit and sha survive as the record
of the review and travel into provenance beside the bytes this run read.
Second fix in the same read: the parser required `candidate_rungs_q256`, the
name Tessera retired at contract v2 and keeps only as a deprecated alias it
says it drops at schema v2; it reads `attested_rungs_q256` first now. With
`pq-allocator-menu` merged in the same commit range, the attested menu is
**three** rungs: contract v5's BF16 receipt and PrismaQuant's body-aware anchor
budget arrived independently and met.

Re-stamped (2026-09-03, `pq-tessera-lane`) for the **addition half of the
lane decision** (§1.1, §3.3, §8.4, §9.4, D33, D34; issues #116 and
RobTand/tessera#20). Robert's 2026-09-02 decision made the sanctioned lanes
compressed-tensors, GGUF and Tessera; the driver implemented the retirement
half the same day and the addition half never landed, so `EXPORT_LANES` was a
2-tuple with a test pinning it and no Tessera artifact could be planned,
exported or gated from here regardless of what Tessera could produce. Four
things change. `tessera` joins the `EXPORT_CONTAINER` vocabulary and `qwen3`
— the only architecture with served evidence and matching fused-group wiring —
declares it. `run-pipeline.sh` gains a real `EXPORT_CONTAINER=tessera` arm
that plans and encodes through **Tessera's own** tools under `TESSERA_REPO`
rather than growing a second codec here, behind three fail-closed preflight
gates (`tessera_export_lane.py`): the checkpoint's structure against the
contract's declared `structures`, the lane spec's `executes` against the
contract's `formats[]` rows (principle 14), and the release pin — which is the
only one that refuses today, and does so naming RobTand/tessera#17. The serve
fingerprint learns the Tessera decoder's `.so`, so a serve running Tessera's
own native decode no longer reports "nothing resident" (the names are the
pin's; deriving them from the runtime's own table is RobTand/tessera#28). And
D34's two deferred observations are closed: `check_serving_shape` now fails
**closed** on an unknown profile id like its siblings instead of resolving
silently to `research` (a fourth seam, `_tensor_parallel_applicability`, was
found and closed by the stamp above), and `activation_pricing_branches`'s
`unrecorded` bucket is tested again.
Re-stamped (2026-09-02, `tessera/decouple-gridbook`) for the **archival of the
Gridbook trellis rate surface** (§4.9). The five `prismaquant/trellis_*.py`
modules, `serving_profile_specs/trellis_research_sm121.json` and the four
`tests/test_trellis_*.py` files are archived whole at `archive/trellis_wire_2026-09-02/`, tracked as
RobTand/prismaquant#118. They priced `gridbook.trellis.wire.v1`, and the lane
that read those bytes was retired the same day by the stamp below; Tessera's
wire is `prismaquant.tessera.v1`, a different plane set and deliberately not a
port, so there was nothing to migrate. Both seams the surface reached through
are cut: the exporter's `parse_trellis_format_name` branch is gone and a
`TCQ_*_R256` name now takes the generic `FORMAT_SCHEME` refusal, which still
fails closed and still names the format; `allocator_candidates` no longer
imports `trellis_menu`. **`PRISMAQUANT_TRELLIS_SURFACE` still refuses** — from
`allocator_candidates.refuse_retired_trellis_surface`, called as the first
statement of `build_candidates`, naming the retired wire and #118. Dropping the
variable would have handed a stale driver a *different* allocation with no
diagnostic, which is the fail-open class of prismaquant#120. Gate:
`tests/test_retired_trellis_surface_refusal.py`.
Re-stamped (2026-09-02, `pq-allocator-menu`) for the **16-bit Tessera family**
(§4.10): `ANCHOR_BUDGET_BITS` is now the TCQ body's refusal rather than the
grid's, so `TESSERA_BF16_K1` is namable, enumerable and priceable and
`menu_families()` returns four. The wall is asked of
`tessera.export.recipe_table` — does this grid reach the TCQ body at any rung —
so `TESSERA_E4M3_K2` stays refused for the reason it always was and nothing
here caps the window's own width. Both byte accountants now charge a WINDOW
table at `PayloadGrid.code_bytes * 2^L`, which is a 2x correction on BF16 and a
no-op on every other family, pinned exactly against
`tessera.calculator.terminal_rate`. Nothing about attestation moved *at the
time this landed*: the family was namable and priceable, and the pinned
contract attested no rung of it. **It does now** — Tessera contract v5 gave
`TESSERA_BF16_K1` `attested_rungs_q256: [1792]` and two dense sm_121 cells on
the 2026-09-02 BF16 route receipt, so `route_admission('TESSERA_BF16_K1_R1792')`
answers `backed_with_serve_flag` and the attested menu is **three** rungs. The
2026-09-02 campaign table still reports `2 of 2423`, because it was priced
before the family existed and holds no BF16 column: that count is a fact about
that table's vintage, not about the menu. Also in this stamp: the Tessera menu path gains its first
end-to-end test against a real campaign cost table
(`tests/test_tessera_menu_real_table.py`, Tessera issue #19) — the class of
failure that hid the `TESSERA`-token expansion hole was that every test asked
one function about one hand-written list.
Re-stamped (2026-09-02, `pq-gridbook-debt`) for **the attestation source the
Tessera menu stamps** (§5.7). `RouteAdmission.source` read
`gridbook_serving_runtime_pin:lane_eligibility` on every unit built without a
development override — a module archived that morning
(`archive/gridbook_lane_2026-09-02/`) whose pin stopped governing Tessera
admission when Gridbook withdrew its Tessera lane. The table that actually
answered was Tessera's OWN packaged `runtime_contract.json` (v0.1.0, publishing
`TESSERA_E2M1_K2` and `TESSERA_E4M3_K1`), so the verdict was right and its
provenance named a runtime that no longer speaks — the unattested assertion
principle 14 refuses, in the field a gate reads. It now reads
`tessera_packaged_contract:lane_eligibility:<version>`. The `detail` was wrong
in the same place and for the same reason: it said the release "publishes no
cell covering this family and rate" when the contract publishes exactly that
rung, and what refuses is `_release_pin_satisfied()` — no Tessera release tag.
`tessera_render.tessera_lane_admission` now returns the refusing conjunct
beside the verdict and four honest strings replace the one wrong one; the
VERDICT is still `tessera_lane_attested`, so patching it cannot invent a
rationale the contract never gave. Two smaller residues in the same pass: the
sample-parallel probe's retirement refusal moved to parse time (it sat below
`load_num_hidden_layers`, so a retired mode's refusal needed a live checkpoint
to reach, and was untestable without one), and the dead
`ROUTE_GRIDBOOK_MXFP8_DENSE` constant — no reader anywhere in the tree — was
deleted. Three more residues in the same pass, all of them a value or a
string a reader would ACT on rather than prose they would merely read.
(a) `serving_profile_specs/tessera_research_sm121.json` declared
`"runtime": "gridbook_plugin"` and derived its `world_size: 1` from
`gridbook_runtime_contract.0.9.1.json`'s tensor_parallel table — a runtime name
and a contract file that both left the tree that morning. It now declares
`vllm+tessera_plugin`, the same string `lane_specs/tessera.json` uses, and
re-derives the ceiling from Tessera's own packaged contract, whose
tensor_parallel table pins `TESSERA_E2M1_K2` and `TESSERA_E4M3_K1` at
`max_world_size` 1 — same conclusion, checkable derivation. Its
`_emulation_only_rationale` said the pinned Gridbook release publishes no cell
naming any Tessera family; the packaged Tessera contract publishes both
families, so what refuses is the pin, and the rationale now says which.
(b) `allocator.py`'s `--serve-dispatch-table` help offered
`prismaquant/serve_dispatch_tables/gridbook_gb10_2026-08-01.example.json`,
which went to the archive with the lane while `example_table_path()` was
correctly updated to return `None` — a CLI pointing at a file the tree does not
ship. (c) `allocator.py`'s `--allow-quantized-pinned` help said `embed_tokens`
is servable "on the GRIDBOOK lane only, via the quantized_embedding
declaration". That was the only route, so with the lane retired there is now
**no sanctioned route for a quantized token embedding at all** — a fifth
capability loss alongside the four Tessera #21 names, recorded on that issue.
Three smaller present-tense claims were re-scoped to the past in the same pass
(`mxfp4_widen`'s `MXFP8_GROUPED_ROUTE_EVIDENCE`, which is a constant a gate
reads and a test pins, and two `aqua_activation_cost` refusal strings that cite
the CB lane's zero A-side as live). Guarded by
`tests/test_serving_profiles.py::test_no_live_serving_profile_spec_names_the_retired_gridbook_runtime`,
which requires every remaining mention of Gridbook in a live serving-profile
spec to carry its retirement date — the machine-checkable half of "record the
scope or do not record the claim". Three decisions were surfaced rather than
taken: the `trellis_*` stack (#118), `check_serving_shape`'s fail-open (#120),
and the 111 undated Gridbook mentions still in module docstrings and comments (#123). D34
otherwise unchanged.
Previously stamped (2026-09-02,
`tessera/remove-gridbook`) for the **retirement of the Gridbook codebook
lane** (§1.1, §3.5, §9, §9.2, D34). Robert, 2026-09-02: *"put Tessera in
PrismaQuant and remove Gridbook."* PrismaQuant carries **one** non-vLLM-native
wire, and it is Tessera's. The producer-side lane is archived whole at
`archive/gridbook_lane_2026-09-02/` — pins, exporter, lane spec, four serving
profiles, ship-gate slots, 73 test modules (1,691 node IDs) and the lane's 27 documents including every
served measurement it produced — and `EXPORT_CONTAINER=nvfp4_cb` is now the
twelfth `exit 2` gate. The sanctioned containers are three:
`compressed-tensors`, GGUF, Tessera. Three things the removal really does, and
none of them is an opinion: `FP8_BLOCK_UE8M0_SOURCE` becomes
`ROUTE_STATUS_BLOCKED` because its only route was the plugin (the rung stays
priced, per principle 1; export fails closed, per principle 9);
`MXFP4_SOURCE` keeps its backed stock-Marlin route but loses its only writer
and its only serving profile; and `gridbook_lane_eligibility.py` — the
**generic** engine that admits Tessera — is renamed `lane_eligibility.py`, its
publisher-specific schema and asset lookup removed, and now demands an explicit
`contract_path=` so that with none, every unit resolves UNATTESTED. The CB
format/cost/render plumbing is deliberately **not** excised and is recorded as
debt **D34**. Previously stamped (2026-09-02,
`tessera/decouple-gridbook`) for **Tessera's own vLLM plugin** (§5.7, §9,
§9.2, §9.4, §5.1's format table, D33): Tessera now serves ITSELF. The
serving half moved into the Tessera repository as the package
`tessera.serving` — a `vllm.general_plugins` entry point
`tessera = "tessera.serving:register"`, registering `quant_method = "tessera"`,
selected by the checkpoint with no enable flag and one operator knob
`TESSERA_SERVE_MODE=resident|streamed` — and it packages its own
`runtime_contract.json` (`tessera.runtime-contract.v1`, lane table
`tessera.lane-eligibility.v3`) whose every cell carries a new machine-readable
field `requires_plugin: "tessera"`: stock vLLM has no reader for these bytes,
so the route is plugin-gated, not merely flag-gated. **Gridbook's Tessera lane
is withdrawn** — its contract v14, described in the stamp below and in §9.2,
was never released, so nothing that shipped is broken, and the Gridbook pin no
longer governs Tessera admission; read that stamp and that passage as history.
Producer side, three things are new and one is rewired.
`gridbook_lane_eligibility.py` becomes a two-vendor parser by three ADDITIVE
widenings — an accepted-schema set, a `tessera_wire` format kind that is
rate-addressed exactly as `tcq_trellis` is, and `requires_plugin` as an
OPTIONAL cell key propagated as `requires_plugins` through `RegimeRoute` /
`UnitRoute` / the provenance payload — and every pre-existing Gridbook
eligibility test passes unchanged. `prismaquant/tessera_runtime/` +
`tessera_serving_runtime_pin.py` are the new boundary object, modelled on the
Gridbook serving pin minus the wheel digest (Tessera publishes no wheel; it is
installed from a source checkout, and asserting a digest for an archive that
does not exist is the hand-assertion principle 14 refuses).
`tessera_lane_attested` now reads Tessera's packaged contract through
`importlib.resources`, RAISES on a cell that claims a route without naming its
plugin, and ANDs the release pin into its answer — **and the answer is False
today BY THE PIN**: there is no Tessera release tag (cutting one is Rob's
call), so the pin carries `PENDING_TESSERA_RELEASE_COMMIT` /
`PENDING_TESSERA_RELEASE_VERSION`, `version_is_release: false`, and
`require_exact_tessera_runtime_release` refuses them. Without that conjunct
admission would flip to True the moment the `tessera` package became
importable, which it already is as a producer-side render dependency — a
producer-side import is not a serving release. A fourth lane spec
`lane_specs/tessera.json` declares the bar (gates ADVISORY, dense-only, TP=1,
no routed MoE — no served measurement covers experts, so the contract carries
no `routed_moe` cell), and its `served_activation_quantization.executes` is
DERIVED from the packaged table by a test rather than typed. No exporter, no
stage, no menu default changed: `FORMATS` stays `NVFP4,FP8_DYNAMIC,BF16`.
Re-stamped (2026-09-02,
`tessera/continuous-menu`) for the **Tessera dev pin, the H seam and the group
fold's measured cost** (§4.10): PrismaQuant now reads Tessera's own packaged
`runtime_contract.json` through a named development override
(`PRISMAQUANT_TESSERA_DEV_PIN`, checked on the contract's *answer* rather than
its bytes since issue #38, refusing rather than degrading to "unattested"), and that table publishes **one rung per family and
`max_world_size: 1`**, so the attested menu is three points (one per family
since contract v5) and empty above tp=1
while the research menu is thousands of rungs per unit (6764 on a 2048x1024
unit, 3709 of them the 16-bit family's). Tensor-parallel legality now
asks `tessera.layout.shard_granularity` rather than deriving a period locally;
a mixed Bresenham schedule takes the column period to the 256-column
superblock, which collapses a row-parallel `[3072,1024]` menu from 3060 rungs
to 17 at tp=8. Activation-aware encoding goes through
`tessera.export.ActivationSource`, formed in one place
(`prismaquant/tessera_hessian.py`) for both the anchor campaign and the
production render, and since Tessera's 2026-09-02 release it reaches **every
plane**: E4M3 (CHANNEL) and both E2M1 families (LUT16) all take an H, and
`for_unit` requires the unit's `scale_plane` because the refit objective is a
map keyed by plane. The former "CHANNEL-plane-only" reading is **retracted** —
it restated two Tessera guards that release deleted, and under it the W4A4
route was priced against bytes the export does not write: Tessera measures
served KL 0.6404 weights-only against the H-aware wire's 0.5310 at identical
bytes. The one-rung fused-group constraint is measured on a
per-group-anchored table at **1.237x / 1.558x / 1.113x**. The `TESSERA` menu
token now expands to priced-**and-attested** rungs and prints the narrowing:
a cost table is priced under the campaign's menu mode while attestation is a
property of the runtime, so a research-priced table read back on the default
path used to make `require_producer_formats` refuse the entire run and the
backed rungs never reached the DP. An explicitly named unattested rung still
refuses. Measured on the default path with the dev pin: 2 of 2423 priced rungs
attested, floor 4.000 bpp (7 x `E2M1_K2_R896`), ceiling 4.042 bpp, and
`--target-bits 3.0` refuses as infeasible. **And the menu now carries a
measured requirement**: served on Qwen3-0.6B against a byte-matched uniform
arm, this menu's own allocations lose **2.00x** in KL-vs-BF16 at 4.0 bpp
(2.33x / 2.88x at 3.0 / 5.0) with bytes exact to the bit, and the loss is in
the cost model on the units it priced (1.93x on those seven Linears, where the
surrogate scores them 1.13x better). `SELECTION_MODE=validated-surrogate`
**plus a byte-matched uniform control served beside the candidate** is
therefore a requirement of this menu, not a suggestion; the allocator warns and
stamps `tessera_menu.selection_caveat` into provenance. Previously re-stamped
(2026-09-02,
`tessera/continuous-menu`) for the **Tessera continuous menu** (§4.10): the
production allocator can now allocate across Tessera's whole realisable rate
set — four families, thousands of legal rungs per unit — through the `FORMATS=TESSERA`
token, an anchor campaign that measures a few rungs per (unit, family) and
refuses to extrapolate past them, dominance-only menu reduction (never a hull),
and serving-unit promotion relaxed to share a FAMILY with rates free per member.
Tensor-parallel degree and direction are per-unit legality inputs read through
one function and stamped into provenance. Two facts the brief assumed otherwise
are recorded rather than worked around: Tessera's embedded axis is a decode-time
completion axis and **does not exist on the wire**, so every rung costs its own
encode; and no pinned release attests a Tessera route, so the default menu is
empty and research mode stamps `unattested`. No export leg, no served KL.
Previously re-stamped (2026-09-02,
`tessera/decouple-gridbook`) for **Gridbook contract v14** (§5.7, §9.2, D33):
the Tessera lane serves two families — `TESSERA_E2M1_K2` (W4A4) and
`TESSERA_E4M3_K1` (W8A8) — behind one flag pair, and one checkpoint carrying
both was served in one process; the pin is unchanged (0.9.1 / v12), so
admission still answers False by the pin, and `tests/test_tessera_formats.py`
gains the v14-shaped admission case. Re-stamped (2026-09-02,
`tessera/decouple-gridbook`) for the **Tessera seam** (§5.7): a Tessera
rung is a synthesized `FormatSpec`, not a registry row — `format_registry.get_format`
falls through to `tessera_render.synthesize_tessera_spec` for any
`TESSERA_<grid>_K<arity>_R<q256>` name, `FormatSpec.bits_for_shape_fn` prices
it exactly per shape or refuses to quote a scalar, and the render goes through
the `tessera` package with no numeric constant of its own. **Producer
eligibility is a lookup, not a constant** (`tessera_lane_attested`): it reads
the pinned SERVING release's eligibility table and admits a rung only when a
`device_qualified` cell on a native route names its family and rate. The pin
is still 0.9.1 / contract v12, which publishes no Tessera row, so every
Tessera rung is producer-ineligible today **by the pin, not by an edit**;
Gridbook contract v13 (unreleased) carries the first row from a served
receipt. No exporter, lane spec, stage or menu default changed: `FORMATS`
stays `NVFP4,FP8_DYNAMIC,BF16`. This stamp closes a principle-13 gap: the
branch's fourteen Tessera commits reached this document only now. Re-stamped
(2026-08-30, `claude/gridbook-v12-pin`) for the **Gridbook 0.9.1 / contract-v12 pin
advance**, which is the step the stamp below deliberately did not take.
Gridbook 0.9.1 was released from `227420f` (tag `v0.9.1`, PyPI wheel
`cb4d7ad6…`, digest confirmed against `pypi.org/pypi/gridbook/0.9.1/json`),
its packaged contract materialized byte-verbatim as
`gridbook_runtime_contract.0.9.1.json` (`836b7831…`), and **both** pins moved
together with `GRIDBOOK_RUNTIME_CONTRACT_SCHEMA` → `gridbook.runtime-contract.v12`
— `parse_gridbook_runtime_pin` refuses a pin whose schema differs from that
constant, so the three are one change. **This changes gate behaviour, in the
direction the pin was supposed to change it and one direction it was not.**
Route status on the CB lane stops being `unattested`-by-absence and becomes a
resolution against a published table: `route_status_source` now reads
`gridbook_runtime_contract:0.9.1:no_cell` where it read `…:0.8.11:absent`.
**And the v12 table names no CB cell on `sm_121` at all** — its twelve CB cells
are `compile_only` on `sm_89`/`sm_120`, its four `sm_121` cells are
`device_qualified` trellis — so with `serving_profile_specs/nvfp4_cb.json` now
declaring `target_platform: sm_121` (a v3 cell is platform-scoped; no platform
named is no route resolvable), **a CB export targeting GB10 fails closed**
unless the artifact declares a non-native target or carries a per-run override,
either of which is stamped on the shipcard. That is the table reporting a real
serving gap, not a gate defect, and no cell was invented to paper over it: the
only device-qualified CB-on-sm_121 receipt that exists is Gridbook **0.4.0**
(2026-07-31, `/home/rob/dq-runs/evidence/gridbook-0.4.0-jason-gb10/`), four
months and the whole v2/fused/persistent-B kernel line before this release, and
whether it is still evidence is a judgment for Rob, not a gate for an agent.
The blast radius is worth stating plainly, because it measures how much of the
CB surface was resting on the absence of a table: **ten test modules, 90
tests**, build synthetic CB bodies and now have to declare
`PQ_CB_NON_NATIVE_TARGET` to build at all (one opt-in `pytestmark` per module,
`tests/cb_synthetic_target.py`; not autouse, so nothing is silenced by
default, and the real `sm_121` refusal stays asserted against the real pin in
`tests/test_cb_route_status_gate.py`). None of those bodies is ever served —
they are CPU fixtures for packing and decode geometry — but the count is the
honest measure of how many CB exports in this tree have no backed route on the
platform the lane targets. The count is measured, not asserted: stripping the
declaration one module at a time reproduces the failures per module
(`test_nvfp4_cb_streaming` 51, `test_cb_lane_sharding` 10,
`test_nvfp4_cb_formats` 9, `test_resident_routed_moe_cbl` 6,
`test_nvfp4_cb_pipeline` 6, `test_nvfp4_activation_contract` 3,
`test_dsv4_packed_expert_export_bridge` 2, `test_dspark_cb_streaming_sidecar`
1, `test_per_expert_cb_export` 1, `test_cb_export_pipeline` 1 — plus 5 more
in `test_per_expert_cb_export` reached only by declaring inside its
module-scoped fixture, because pytest sets higher-scoped fixtures up before a
function-scoped one). Nine modules declare module-wide; the tenth,
`test_cb_export_pipeline`, declares on the single test that exports, because
the other three drive the pipeline without one and must keep running with the
gate armed. Two further modules
were marked on a first pass and the mark then removed: the two
`test_validate_cb_endpoint_*` modules pass with the declaration stripped, so
their failures were the pin-literal and receipt-digest classes below, not the
route gate. An inert declaration is false documentation about where the
serving gap falls, so it does not stay.
Second, quieter consequence, recorded because a build will meet it: 0.9.1
publishes `formats[FP8_CB_K].producer_rungs = [40, 44, 48]` and
`formats[NVFP4_CB_K].producer_rungs = [12..24]`, a narrower ladder than the
decode/reader surface, and the shipped DSv4 routed experts are built on
`FP8_CB_K28`/`K32`. `gridbook_format_contract.py` is **deliberately left bound
to v4/v11** so that narrowing does not silently re-scope what PrismaQuant may
build inside a pin commit; its docstring records the open decision, and a v12
contract handed to that reader raises rather than degrades. That narrowing was
not merely latent: `test_gridbook_runtime_contract.py` asserted
`producer_rungs == list_producer_formats(family)` on a branch dormant under v4,
and v12 fires it on both families, so the pin commit splits the equality —
keeping `producer_rungs ⊆ producer menu` and moving the reverse direction (a
menu ban, vetoed by principle 1) to principle 9's per-artifact export gate. `NVFP4_CB_S`
disappears from the published formats, which is consistent — PrismaQuant
deleted that family on 2026-08-17. Verified against the release wheel on
GB10/sm121: RELEASING §2.2 compile check (including the trellis extension),
fused-FP4 SASS 58 passed / 0 skipped, kernel-adjacent 444 passed,
`test_trellis_r256_cuda` 49 passed / 0 skipped. Re-stamped (2026-08-30,
`claude/lane-eligibility-v3`) for the **lane-eligibility v3 parser and the
scoped route-status refusal** (§9.2.1). `gridbook_lane_eligibility.py` and
`ServingLaneSpec.route_status_for` read
`gridbook.lane-eligibility.v1` (`{schema, regimes, lanes}`); Gridbook publishes
v3 at `30287aa` (`{schema, platforms, regimes, structures, cells}`), so the
consumer would have refused the very table it was written to consume. The
parser now reads v3 — platform-scoped cells, explicit `rungs` / `rungs_q256`
dispatched on each family's `formats[].kind`, `activation_contract`,
`qualification` — and `UnitStructuralFacts` gained `rate_q256`. **This changes
gate behaviour**, which is why it is stamped here: because a v3 table has no
`unbacked` cell, absence is the runtime's only way to say no, so an uncovered
unit resolves `unattested` and the export gate now **fails closed on it** for
any family the contract publishes, while units outside those families are
counted and reported rather than judged. The previous claim in §9.2.1 that
flipping the index to `present` would need "no PrismaQuant code changes" is
**retracted** — it was written about a table nobody had yet seen. The pins are
**not** bumped: that step is release-keyed and outward-facing (see the end of
§9.2.1). Re-stamped (2026-08-29,
`rescue/grouped-woa`) for
**pricing the grouped operand `wo_a`** (§8.9): DSv4's grouped-BMM attention
output projection — 17.9% of decode read traffic — was never an allocator
decision because the probe skipped its class. The grouped Fisher accumulator
(`prismaquant/sensitivity_probe.py`: exact per-group bmm reductions, flat-plane
marginals, the ONE global-token normalization) landed FIRST; then
`DeepseekV4GroupedLinear` moved from the spec's `probe_skip_module_class_names`
to the new `probe.grouped_module_class_names`, which routes the probe to the
grouped accumulator and lets the walk claim `wo_a` as an ordinary `decide`.
Cost cells flow from probe keys with no new plumbing; the joint-output-MSE
screen ships honestly unmeasured (`output_mse_measured=False`) for grouped
units because the dense screen mis-models the contraction. No shipped artifact
changes: the DSpark sidecar contract keeps all three `wo_a` bases on
source-FP8 W8A16 (`dspark_cb_expected_physical_targets`), CB export still
refuses grouped operands, and and the W8A16 handoff's frozen source closure
drifted on three files (`model_profiles/base.py`,
`model_profiles/deepseek_v4.py`, `specs/deepseek_v4.json`), which are
re-frozen in this stack with the review recorded in
`dsv4_w8a16_export_handoff.py` (spec data-only, `deepseek_v4.py`
docstring-only by whole-module AST, `base.py` one added accessor with zero
existing bodies changed; no exporter, completeness, decode-source or
footprint consumer reads either declaration). `wo_a` becoming an allocator
decision means the 92 GB budget split must be re-checked on the next DSv4
export rather than inherited.
Re-stamped (2026-08-29, `rescue/name-projection`) for
**the cost consumer migrating onto the shared name-projection layer** (§8.8.1):
`_scan_source_dtype_manifest` lost its private checkpoint→live→recipe builders
(`_strip_weight_suffix`, `_to_recipe_name`, `_packed_to_recipe_name`,
`_per_expert_packed_recipe_name`) and now projects every source-kind row
through `NameProjection.checkpoint_to_live` / `recipe_unit` /
`packed_parent_of_expert_param` (`prismaquant/allocator_candidates.py`);
`decision_units._recipe_name` is retired for
`name_projection.strip_weight_leaf`, as is the inline leaf surgery in
`production_render_cost.canonical_cost_name` (whose umbrella-infix half stays
a total normalizer — render/cost payloads key costed MTP rows physically,
which a declining projection must never drop) and `measure_quant_cost`'s
act-cache candidate builder. Emitted values are UNCHANGED for existing
profiles and artifacts — the manifest's profile=None convention now builds
over the repo's declared generic baseline (`DefaultProfile`, the substitution
`resolve_cost_target_name` always made) instead of inlined string surgery.
One deliberate behavior change, fail-closed only: a profile accessor that
RAISES now propagates `NameProjectionError` instead of silently skipping the
row (the wo_a shape); MTP rows stay recipe-native verbatim by an explicit,
commented short-circuit pending a profile declaration of recipe-native
checkpoint prefixes. Probe, footprint, and read-traffic remain unmigrated.
Previously re-stamped (2026-08-22, `walker/consumer-probe`) for **the probe
consumer migrating onto the shared name-projection layer** (§8.8.1): `FisherAccumulator`
now builds one `NameProjection`, and the probe holds no private name mapping — the
packed-expert shard-scope filter reads its block id through `NameProjection.block_id`
(previously a positional `[:3]` qname slice of its own; value-identical on every qname
shape reachable at that site, pinned by `tests/test_consumer_probe_name_projection.py`),
and the Fisher skip set keys its embedding clause on the profile's declared
`ModelProfile.embedding_name()` (previously a hardcoded `"model.embed_tokens"` substring
test). Probe stats keys, inventories, and shard regexes are untouched, and the regression
baseline is byte-identical; the walk-edge-list migration itself stays open for all four
consumers.
Previously re-stamped (2026-08-22, `walker/consumer-footprint`) for **the
second R5 consumer migration**: `footprint.py` holds no private name mapping anymore — its
private `.weight` strips and its packed-expert parser (`packed_expert_alias` + the legacy
parent fallback, now defined in `name_projection.py` and re-exported) are the shared
layer's; `source_tensor_bytes_manifest` / `floor_bytes_for_model` accept a prebuilt
`NameProjection` (keyword-only, mutually exclusive with the raw accessor kwargs), map
checkpoint keys through `checkpoint_to_live`, keep the profile's DECLARED drops as raw-key
floor entries (data, not exceptions), and propagate layer refusals instead of swallowing
them. Emitted bytes are unchanged — projection/accessor parity is pinned test-side.
Previously re-stamped (2026-08-22, `walker/consumer-readtraffic`) for
**the first name-projection consumer migration: read-traffic**
(`prismaquant/read_traffic.py`). Both entry points build one
`NameProjection(profile)` and route every checkpoint→live question through
it: the classifier's decline-to-map rule branches on
`ProjectedName.outcome == declared_out_of_graph` (`excluded_non_text_graph`),
the private `_strip_weight` leaf helper is deleted in favor of
`strip_weight_leaf`, and a profile accessor that fails now refuses as
`NameProjectionError` instead of silently passing the raw key through
(previously it would have re-priced unmappable tensors at p=1). No emitted
number moves; the class table, byte authorities, and refusals are pinned
byte-identical by `tests/test_read_traffic.py`. Previously re-stamped
(2026-08-22, `walker/name-projection`) for
**the shared name-projection layer** (§8.8.1, `prismaquant/name_projection.py`):
one profile-routed projection between the live/recipe/checkpoint/export/vLLM
namespaces, fail-closed with structured refusal codes, explicit
many→one/one→many shapes for fused siblings and packed experts, round-trip
identity pinned where the profile's rules are total (Qwen3, Qwen3.5
multimodal, DSv4 — including the surfaced `hc_head` inverse gap), and no
rank/shard/degree anywhere in the API (the TP seam stays in
`model_walk.per_device_bytes`). Consumers are not migrated; the module ships
with its conformance tests only.
Re-stamped (2026-08-29, `rescue/walker-export-gate`) for
**the discovery walker as a fail-closed export gate** (§8.8,
`prismaquant/model_walk.py`): the R5 design contract's remaining open half —
wiring the walk as an export gate — is closed; the probe/cost/footprint/
read-traffic migration onto its edge list stays a separate workstream.
`run-pipeline.sh` stage **[3d]** (`python3 -m prismaquant.model_walk`) runs
before EVERY export lane (Tessera, GGUF, compressed-tensors) and refuses (`exit 2`)
on an unclaimed matmul-fed node, an unresolved floating multiplicand, an
unknown walk-failure kind, or a decided-but-unpriced contradiction, deciding
from STRUCTURED fields only (`artifacts/model_walk.json`, gate schema
`prismaquant.model_walk_gate.v1`). The explicit override
(`PRISMAQUANT_WALK_GATE_OVERRIDE=<reason>`) excuses trace incompleteness ONLY
(DSv4's data-dependent scalar aborts the fake trace) and is stamped; claim
failures have no override — they are fixed by pinning with reasons in
`ModelProfile.walk_claim_rules()`. Same commit: the walk became a cacheable
artifact (`SCHEMA prismaquant.model_walk.v1`; envelope `{schema, provenance,
result}`, atomic write, fail-closed reload on foreign schema / execution
mismatch / claim-rule digest mismatch), carrying trace-time provenance
(model+config identity, versions, input contract) and the serialized applied
rule list; byte semantics are pinned as LOGICAL-TOTAL with the
`per_device_bytes(total, tp_degree, policy)` seam for Tensor-Parallel
(decision unit: whole logical tensor; dispositions are TP-invariant; the
future group/shard misalignment refusal lands as an unknown-until-implemented
walk-failure kind that already refuses). The R5 profile sweep's findings land
here too: universal base rules now pin MoE router gates and decide packed
expert stacks (six/seven profiles were unclaimed; gemma4's router was
decided-but-never-priced — wrong polarity inside the claim table itself, now
pinned like every other router), and the gate surfaces that contradiction
class structurally.

Re-stamped (2026-08-29,
`docs/architecture-currency-20260829`) for a **§8.4 conformance-matrix
correction on `glm5_next`**, found by a principle-13 sweep of `origin/main`
rather than by a test — the two doc gates were green across both drifts. (i) The
row claimed the CB lane. `58eb69d` withdrew it: the pinned Gridbook 0.8.11
`producer_profiles.supported_ids` does not name `glm5_next` at any release
through 0.9.0, so declaring it was the over-declaration that serves
uninitialised expert memory rather than crashing, and the spec now carries the
exact re-enable condition. That commit changed a serving-lane declaration
without touching this file — a § P13 trigger missed, and `d2b4614` later
updated this file and still did not catch it. (ii) The row named
`vllm_packed_moe` as the default serving profile; the spec has said
`vllm_glm5_next_packed_moe` since `b35ed53`, and that profile — which denies
quantized formats outside the three module families PR #53906 wires a
`quant_config` into — appeared nowhere in this document. Drift (ii) predates the
`b4a8846` refresh, so it is the § P13 "currency is not truth" case: this file
was re-stamped in lockstep while carrying a false statement about a serving
default. Nothing else in the window crossed the trigger; the reasoning is in the
commit message.

Earlier stamp (2026-08-29, `claude/trellis-continuous-surface`) for the **trellis seam correction**
(§4.9): the seam that landed earlier the same day is now **fail-closed** when
enabled, and this document's claim that it made trellis rungs "pass the same
legality, aggregation and byte accounting every other candidate does" is
**retracted** — it was false on all three counts, written in the same commit as
the code it described (§ P13, "currency is not truth"). `trellis_menu.
UNWIRED_LINKS` is the authoritative eight-entry ledger of what is missing and
is the text of the refusal; `build_trellis_menu` still builds a correctly
priced menu for research, now using the repo's own `_shape_from_stats` and
`_stats_indicates_packed_expert` (the hand-rolled 2-tuple underpriced a
128-expert row **128×**, silently, and the guard meant to catch it read a stats
key nothing writes). Two of the seam's tests asserted on source text and now
assert behaviour. Earlier same-day stamp
(`claude/trellis-continuous-surface`) for the **opt-in continuous trellis rate
surface** (§4.9): `prismaquant/trellis_menu.py` is now the one seam through
which the previously islanded trellis modules reach
`allocator_candidates.build_candidates`, behind
`PRISMAQUANT_TRELLIS_SURFACE=<manifest.json>` whose unset default returns the
menu object unchanged. Rungs are named by the shape-free closed
`TCQ_{E2M1,E4M3}_R<q256>` vocabulary so fused-sibling and packed-expert
promotion — which intersects member menus by format name — still finds a common
format; the per-tensor recipe digest rides `serialized_identity`. New profile
`serving_profile_specs/trellis_research_sm121.json` declares
`target_platform: sm_121` for the one reason it exists: `_capability_gate`
returns legal without comparing anything when a profile declares no platform,
and six of ten specs declare none. Export refuses a `TCQ_*` assignment with a
pointed message — there is no render mechanism and no runtime attestation, so
this is allocation-time reach only.

Re-stamped (2026-08-28,
`claude/trellis-continuous-surface`) for the **PrismaSnap source-dtype
contract** (§3.0): the BF16-realization lane now accepts a native-FP8 source as
well as BF16, dequantizing through the checkpoint's declared block grid and
folding in BF16. The prior refusal was guarding a real hazard, not being
conservative — `apply_diagonal_transform` restores the *source* dtype by
default, so a naive widening would have rounded every fold back into
`float8_e4m3fn` against a stale block scale. Uplifted operands drop their
scale siblings from shard and index. FP8 **materialization** is not yet
reachable — every liftable FP8 source refuses, up front, at
`_preflight_uplift_publishability`; the dense lane only, as the MoE plan
lane remains BF16-gated. Re-stamped (2026-08-28,
`claude/trellis-continuous-surface`) for the **packed-expert cache render
identity (SB-1)**: `fill_packed_expert_cache_entries` appended packed-MoE keys
to `cache.weights` without touching `requested_entries`, `render_scores` or
`render_gates`, so the exact cache union refused at its first subcommand on
every packed-MoE model (35B, 122B, 284B, GLM-5.3). Every packed expert is now
scored honestly — per-expert render scores summed into the one record the cache
key owns — each packed entry carries a truthful empty-trace gate record naming
the mechanism that actually ran, the counters are recomputed from the cache so
the M4 lazy gap-fill loop stays exact, and no exclusion or subtraction is added
to the union arithmetic (§ exact multi-host striping;
`tests/test_packed_expert_union_identity.py`). Earlier same-day stamp
(`claude/trellis-continuous-surface`) for the **PrismaSnap dead-gate repair**:
`merge_checkpoint_parts` now takes `production` and runs
`_require_attested_container()` first, so the CLI's `merge-checkpoint-parts`
(which now passes `production=True`) is held to the same attested-container
contract as `materialize-part`. It deliberately takes **no** `device`: the
union executes on no device, so the CUDA leg of `_require_production_execution`
would have been satisfied by a caller-supplied string with no referent, and
that leg is now split out for the paths that do execute -- each of which
asserts `torch.cuda.is_available()` immediately after the gate;
`_producer_identity` additionally binds `prismasnap_moe.py`,
`prismasnap_moe_checkpoint.py` and `prismasnap_contract.py`, so an edit to the
MoE plan stack or the lane-admission contract invalidates a producer receipt;
and `_layer_source_graph`'s `source` parameter now binds every planned MoE
operand to the live checkpoint header (`_Checkpoint.metadata`) instead of
trusting the tensor census alone, matching the dense twin. Earlier stamp
(2026-08-27,
`claude/trellis-continuous-surface`) for the **glm5_next export bring-up**: the
streamed exporter now builds the multimodal skeleton when the profile declares
`requires_multimodal_skeleton()` (stage_multimodal + declared-arch class, weight
map `multimodal=True`), the tensor-sink rename composes `live_to_recipe_name`
before `export_tensor_name`, the 3e layer passthrough ships concat-merge SOURCE
tensors verbatim instead of the merged live key, per-layer progress lines are
unconditional with sweep rate/ETA, `fill_packed_expert_cache_entries` probes its
device from the first non-meta parameter, live↔recipe probe/cost rekeying exists
as `glm53_stock_reprice rekey-recipe`, and `specs/glm5_next.json` declares the
MLP gate/up fused groups attested from the pinned PR image (see §Model-profile
plugin, concat_merges / multimodal-forced export). Earlier same-day stamp
(`claude/trellis-continuous-surface`) for the **GLM stock anchored campaign +
streamed non-CB checkpointing**: the streamed AURA durable-checkpoint guard now
accepts the production-anchor renderer's exact identity as the value-bearing
render identity, so an anchored run on a CB-free menu checkpoints on the anchor
identity alone (the fail-closed refusal remains for non-anchored non-CB;
`tests/test_streamed_cost_checkpoints.py`), `run_streamed_production_anchor_aura`
exposes `allow_packed_expert_omission` for MoE models whose routed experts are
priced empirically, the streamed capture/reverse passes emit per-layer rate/ETA
progress lines, and a new manual campaign row (§ stage table: **GLM stock
anchored**) wires `prismaquant.glm53_stock_harvest` (GPU batched adjoint over
the dense plan) → `prismaquant.glm53_stock_reprice campaign` (CPU three-
provenance merge) for the GLM-5.3-Flash teacher artifact. Re-stamped
(2026-08-26, `claude/trellis-continuous-surface`) for the **GLM-5.3-Flash /
Qwen3.8-Flash-Next plugin-contract change**: two new registered profiles (`qwen4_exp` 200,
`glm5_next` 210 — §8.4), and a new spec key + profile accessor,
`concat_merges` / `ModelProfile.concat_merge_groups()`, backed by a generic N→1
source-concatenation bridge in the streaming loader
(`layer_streaming._merge_concat_sources`, wired into the probe/cost read path and
the exporter). It closes glm5_next's one hard blocker — the KDA short
convolution's `{q,k,v}_conv1d → conv1d` `Concatenate(dim=0)`, which a
1:1-or-drop `checkpoint_to_live_name` cannot express and which would otherwise
have loaded uninitialised. Both new profiles are enablement scaffolds: no vLLM
class is importable for either, so neither is export-ready (§8.4). Re-stamped
(2026-08-26, `claude/snapquant-productionalization`) for the **model-calibrated PrismaSnap fold
threshold**: the served fold gate is no longer the fixed `5e-4` alone but
`max(5e-4, 2.0 × measured null perturbation floor)` when the attestor receives a
saturation-licensed null-floor receipt, content-hashed into
`fold_fidelity.threshold_derivation` (the fixed constant, calibrated at 0.6B, sits
below the measured 27B BF16 perturbation floor; evidence and decision in
`docs/results/qwen38_prismasnap_fold_gate_floor_2026-08-26.md`). The realized v2
machinery is merged with the MoE plan family on one mainline. Re-stamped (2026-08-25,
`codex/prismasnap-bf16-v2-20260825`) for the **cast-aware BF16-realized PrismaSnap v2
plan transition**. The nominal fast `stage,polish` v1 search remains the immutable parent,
but it is no longer executed directly by the repaired arm. `realize-bf16` accepts only one
complete merged v1 plan plus its content-verified original BF16 source and emits a distinct
v2 schema/algorithm. For each input/post-norm seam it preserves the nominal float64 scale,
stores the selected projected BF16 norm bytes, and derives a separate positive consumer
inverse from projected/source effective gamma. Zero or collapsed gamma, non-finite values,
sign disagreement, or a non-bit-exact quotient replay executes identity for that channel;
materialization writes the projected BF16 norm directly rather than recreating it from the
quotient. The transition re-scores exact `BF16(W / s_real)` under the static-6 NVFP4 full
render objective with `importance * s_real**2`, falling an entire seam back to identity on no
strict improvement. Dense up/down is identity in v2. Parent plan/scales/producer, source,
nominal vectors, projected norm payloads, executed vectors, reason counts, and realized
objectives are cryptographically bound; provenance v2 carries that derivation through parts,
collation, served fold attestation, native export admission, and the unchanged downstream
pipeline. Legacy v1 plans/provenance remain readable, but v2 cannot re-enter worker-plan merge.
The reason for the transition is measured and preserved at
`/home/rob/dq-runs/prismasnap-qwen38-27b-20gb-20260825/fold-fidelity/`: direct v1 snapped-BF16
KL was `0.002956950685081059` while source-vs-source was exactly zero, so the v1 checkpoint
failed the `5e-4` prerequisite and remains unverified. No repaired 27B fold result, quantized
A/B result, or release promotion is claimed by this stamp. Previously re-stamped (2026-08-25,
`claude/snapquant-productionalization`) for the **candidate PrismaSnap
additive BF16 source-preparation lane**. PrismaSnap is outside the four-stage quantization
pipeline: an unmarked source takes the historical path unchanged, while a marked source must
carry a content-replayed `VERIFIED` receipt before the existing probe, AURA, per-Linear
allocator, `ProductionWeightCache`, recache, native compressed-tensors exporter, and validation
stack may consume it. Production planning fixes the measured-fast `stage,polish` search,
requires CUDA plus an attested container and complete source/probe/producer identity, and
materializes ordinary BF16 safetensors without a serving transform. The version-2 cluster
campaign runner can close an explicit local/SSH dependency graph without an agent: two Spark
workers build disjoint layer plans and shard parts, exact-union merges reject overlaps or gaps,
and the low-space campaign requires same-filesystem hardlink collation after the remote part is
content-verified on the coordinator. Materialization alone is not admission; original-BF16 to
snapped-BF16 served all-position KL must be at most the attested fold threshold — the plan's
`5e-4` by default, or `max(5e-4, 2.0 × the model's measured null perturbation floor)` when the
attestor is given a saturation-licensed null-floor receipt (2026-08-26: the fixed `5e-4`,
calibrated at 0.6B, sits below the 27B BF16 perturbation floor of ~5–6.6e-4, so the threshold
is now derived per model; see `docs/results/qwen38_prismasnap_fold_gate_floor_2026-08-26.md`).
The receipt requires at least two independent matched-mass null arms agreeing within 3×, is
content-hashed into `fold_fidelity.threshold_derivation`, and is re-validated on every
provenance read. Source, calibration, evaluator, serve-stack, index, and shard identities are
replayed before the atomic `VERIFIED` transition.
The pipeline GGUF branch and direct-GGUF API, plus both Gridbook/codebook exporters, reject any
PrismaSnap marker; the native exporter replays
the verified source immediately before copying its receipt under the unambiguous
`source_prismasnap_provenance.json` name. The first production gate is the still-running,
text-only/no-vision Qwen3.8-27B strict-decimal-20-GB A/B; its requested 20% KL improvement is an
acceptance threshold, **not a result**. The MoE planner and release-layout gate are second and
remain experimental; Qwen3.5-family native/wrapper namespace resolution and explicit RMSNorm
parameter-offset metadata are prerequisites, not a Qwen3.8-125B-A3B promotion. No completed
27B result, MoE result, release readiness, or serving-performance claim follows from this stamp,
and the final integration commit is deliberately not predicted. Previously re-stamped
(2026-08-24, `codex/rtx5060-fp4-gridbook-pilot-20260824`) for the **maintained SM120/RTX50
W8A16 exclusion**. The shared registry retains W8A16 assignment compatibility, while generic
source-model container profiles retain source-FP8 wire compatibility for already-published
artifacts; neither is target-performance eligibility. `qwen38_sm120_cb_validation_only` now
carries an explicit deny for `format_registry.W8A16_COMPAT_FORMAT_NAMES` on top of its closed
allow-list, and the dense
Qwen3.8 campaign refuses any non-BF16 body source before plan construction, allocation, or GPU
work. Both CB materializers read the allocator's stamped target profile and reject a denied
assignment before opening the destination transaction, so a hand-edited W8A16 recipe cannot
bypass chooser legality. New SM120 candidates therefore remain exactly both public CB ladders
plus native NVFP4/FP8_E4M3/BF16; source-FP8 and other W8A16 formats cannot enter its chooser or future
release decision. The generic DSv4/MiniMax compatibility producers and readers are deliberately
unchanged. No device or served performance claim follows from this policy. Previously re-stamped
(2026-08-24, `codex/rtx5060-fp4-gridbook-pilot-20260824`) for the **validation-only SM120
candidate registration and the dense full-ladder AQUA scaffold**. The new
`qwen38_sm120_cb_validation_only` profile extends `nvfp4_cb`, binds exact platform `sm_120`,
and closes admission to NVFP4-CB K1..K25, FP8-CB K4..K48 step 4, plus deliberate native
NVFP4/FP8_E4M3/BF16 terminals. It has no `producer_policy`: candidate v11 is compile-only and
unpinned, so this is neither a device-qualified claim nor a release identity. The dense
anchored driver defaults to that explicit profile (with a CLI override), stamps it into plan
and allocator provenance, and exposes both complete public producer ladders. Its FP8 panel now
spans K4/K28/K48 around a K24 anchor and validates off-panel at K8/K20/K36/K44; the campaign is
lattice-only and AQUA sees both executed activation contracts with no manual family preference.
The immutable release pin plus exact device-qualified route contract remain the fail-closed
shipping gate. No new device or served measurement is claimed, and the final integration
commit is deliberately not predicted. Previously re-stamped
(2026-08-24, `codex/rtx4090-fp8-gridbook-pilot-20260824`) for the
**CPU-tested exact sample-axis probe reducer and four-render-per-Linear RTX 4090 FP8
campaign**. The opt-in probe lane partitions one immutable calibration tensor by complete
samples; its two-stage barrier first reduces raw shifted-token CE, then every worker reruns
phase 1 with the same global mean and executes the complete dense text qname schedule. The
strict reducer adds raw Fisher/marginal sufficient statistics and finalizes once at the global
token denominator; it also refuses raw Fisher marginals whose row/column sums do not reproduce
the raw trace. Its activation side independently replays the complete global priority domain,
exact top-R order/cardinality, and fused-sibling row alignment through the existing
activation-cache path. Bundle validation consumes the same no-follow probe bytes it hashes;
the burn passes that captured payload forward, while lazy activation loads are checked against
the captured committed tensor identities through a pinned directory descriptor. The documented
launcher executes its verifier from the exact pinned Git object, makes and checks the candidate
snapshot non-writable, starts the host-verified immutable registry RepoDigest with canonical `/model`,
`/dataset`, and `/run` paths plus read-only source/model/data mounts, and enters only through
Python's safe-path snapshot bootstrap. A closed common execution attestation joins the sample
execution identity to that live snapshot closure/commit/tree and launcher-owned RepoDigest before
either burn entry point can touch CUDA. The RTX4090 cost lane stripes whole layers across
two hosts and physically renders only lattice `FP8_CB_K4`, `FP8_CB_K16`, `FP8_CB_K48`, and
delegated `FP8_E4M3` per body Linear. It preserves those measured AURA rows, imputes only the
nine remaining K%4 CB rungs inside the lattice segment, and adds BF16 solely as the exact
unrendered source terminal. Producer scoring is fail-closed full-graph Inductor under one
explicit strict switch: every live helper return proves exactly one compiled backend dispatch,
while resume-restored units inherit the same settings/producer/source identity through AURA's
already validated unit envelopes. The shard receipt distinguishes those covers and marks LDLQ
atom execution not applicable. No NVFP4 format is admitted. Its streamed source contract is exactly
two existing `LayerCache` slots, effective lookahead one, and
`require_prefetched_residency=true`: both forward capture and reverse AURA require each current
layer to be resident or already prefetched, while reverse schedules the next lower layer before
the current render/backward work. A missed or rejected prefetch refuses before synchronous
source I/O. Both mechanisms are implemented and
CPU-tested, but the two-host GPU campaign, merged-vs-monolithic numeric gate, validation-only
GB10 artifact, and physical RTX 4090 correctness/performance/full-graph qualification remain
unrun. Existing Gridbook graph requirements are unchanged. The final integration commit is
deliberately not predicted in this provenance stamp. Previously re-stamped
(2026-08-24, `audit/math-reunderwrite-2026-08-21`) for the **final implemented, still
unqualified strict RTX 4090 FP8-Gridbook candidate and its categorically unreleasable GB10
structural-validation producer**. FP8-CB has two different authorities:
`cb_layout.FP8_ACCEPTED_RUNGS` is the backward-compatible reader domain
(K4/K8/K12/K16/K20/K24 plus every historical integer K28 through K48), while
`cb_layout.FP8_PRODUCT_RUNGS` is exactly K4..K48 step 4 and is the only new-producer ladder.
The opt-in `qwen38_rtx4090_fp8_cb` profile admits that producer ladder, delegated
`FP8_E4M3`, and BF16 only. It fixes `lm_head` and MTP to BF16, sets
`CB_ACTIVATION_SCOPE=none`, forbids native and codebook NVFP4 at every manifest/census
boundary, and remains lattice-only until raw learned-v2 promotion evidence is carried into the
strict artifact attestation (`rtx4090_qwen38_policy.py`). Its architecture-specific source
census closes the official wrapper at exactly 1,199 source tensors and 615 Linears, requires
an assignment for every Linear, then replays exact config-group ownership, `ignore`, finalized
tensor keys/dtypes/shapes, and codebook-sidecar references (`rtx4090_artifact_census.py`). The
AURA build streams from one absolute checkpoint directory; its cached streamed-model identity
is the source-census identity, while cost, frontier KL, and export share the probe's full-corpus
imatrix and its value digest. The complete frozen publication tree, including documentation
and evidence, may not exceed exactly 18,000,000,000 bytes, and neither that invariant nor any
strict evidence/replay failure is forceable by `--force-unverified` or `--confirm-name`.
Strict publication folds container, upload-block, and raw-tensor hashes into the existing
held-descriptor freeze traversal, then replays the frozen manifest/ledger/index from that receipt
without a second weight scan (`tools/publish_artifact.py`). The physical serve workload is 32K and one live smoke request
with `n=1`, TP=1, and exactly 4 GiB of FP8 KV; `max_num_seqs=64` exists solely so vLLM FULL
capture admits `[1,2,4,8,16,32,64]`, not as a 64-concurrent-context claim. Graph evidence
requires compilation mode 3, explicit Inductor, `FULL_AND_PIECEWISE`, **7/7 PIECEWISE and
7/7 FULL** capture completion, and the installed wrapper's direct
`torch.compile(..., fullgraph=True, dynamic=False)` call, with any compiler/graph fallback
refusing (`rtx4090_graph_contract.py`, `validate_rtx4090_fp8_cb.py`). The strict launcher
additionally requires a closed vLLM runtime pin naming only the official upstream Git URL,
one exact commit and version, and the installed `RECORD` digest; server collection binds
matching PEP 610 VCS metadata plus RECORD-bound `direct_url.json` and wrapper bytes, and
manifest/shipcard replay require the same pin. No candidate vLLM pin is supplied, so an
immutable image digest alone cannot qualify the lane. The last pre-vLLM operation inside the
serving container scans every individually read-only-bound safetensors shard once and writes a
stat-bound no-clobber receipt; post-serve census and endpoint/shipcard replay reuse it without a
second weight-payload pass (`tools/serve_fingerprint.py`). Gridbook contract v11
separates reader `rungs` from `producer_rungs` and adds closed-world
`gridbook.lane-eligibility.v2` (superseded by the published v3 table, §9.2.1); a candidate runtime must device-qualify exact sm89 dense decode
and batch routes for the **entire twelve-rung producer ladder**, even if one artifact selects a
legal subset. Its v0.9.0 tensor-parallel and expert-parallel contract subtrees and runtime
behavior remain preserved. The available sm89 evidence is deliberately only `compile_only`.
The separate `qwen38_rtx4090_fp8_cb_validation_only` profile may build the same FP8-only wire
artifact on exactly one DGX Spark GB10 through the ordinary GPU-bound pipeline, but stamps it
`UNRELEASABLE_VALIDATION_ONLY` and validates it only through
`validate_rtx4090_fp8_cb_validation_only.py`. Shipcard verification, publication (including
force flags), and the physical strict validator categorically refuse that identity. The
production profile and its device-qualified resolver are unchanged: no physical RTX 4090
correctness, graph, memory, or performance gate has run, no v11 Gridbook release is claimed or
pinned, and strict release remains fail-closed. The final
integration commit is deliberately not predicted in this provenance stamp. Previously
re-stamped (2026-08-23,
`audit/math-reunderwrite-2026-08-21`) for the **fixed output-head policy,
profile-synthesized MTP production renders, and exact striped cache union**. `LM_HEAD_FORMAT`
now defaults to historical BF16; an explicit non-BF16 head is a measured fixed auxiliary
assignment outside the body DP/bpp and inside exact artifact bytes, while
`ALLOW_PINNED=lm_head` remains the mutually exclusive research/DP path. Every persisted
cost/cache stage binds the resolved head policy. `build_production_cache` now appends
profile-built `mtp.*` Linears from the probe activation rows into the one
`ProductionWeightCache`, with exact source/module/activation coverage and qname-allowlist
filtering. `production_cache_stripes.py` emits deterministic layer-local qname stripes, and
`union_production_cache.py` publishes, verifies, and atomically unions portable disjoint
cache bundles under exact source/calibration/code/settings/render coverage; this native-only
set union is not the archived surrogate-driven smart union. The final integration commit is
deliberately not predicted in this provenance stamp. Earlier same-day stamp (2026-08-29, `review/pr86`, merged to main as #89) for the
**codebook lane's namespace contract** (§6.2): delegated (stock-CT) and
source-passthrough group targets now ship in the CANONICAL namespace rather
than the full live-tree spelling, mirroring the pinned consumer's
`_canonical_prefix`/`_candidate_bases`; live-spelled expert-group keys are
normalized to the recipe spelling and collisions refuse; packed expert stacks
are named by their group's checkpoint prefix. Same commit fulfils the FP8
serialized-scale contract in the packed-expert imatrix replay
(`moe_imatrix._load_tensors`), reading the dequant grid through the streaming
loader's `_declared_weight_block_size` so the two paths cannot disagree about
one checkpoint. Reported and implemented by smb209 (PR #86).
Previously re-stamped (2026-08-21,
`merge/proven-rescues`) for the
**Gridbook 0.8.11 PRODUCER pin**: the producer pin
(`gridbook_runtime_pin.json`) advanced 0.8.5/v3 → 0.8.11/v4 at commit
`187c721`, in lockstep with the serving pin, and a new test
(`test_gridbook_runtime_boundary.py::test_producer_and_serving_pins_name_the_same_gridbook_release`)
now makes divergence between the two a failure. The two pins had drifted three
releases: builds, exports and the gold measurement environment resolved through
0.8.5 while every route-status gate, certificate and shipped artifact resolved
through 0.8.11. CI surfaced it sideways — the `gridbook-contract` job installs
the PRODUCER commit, and only 0.8.10/0.8.11 have indexed materialized
contracts. Consequences carried in this revision: the closed gold measurement
environment grew 29 → 31 names (`PRISMAQUANT_CB_FP8_GEMV_V2`,
`PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R`, both canonical `"0"`), execution 19 →
21; the 29-name historical projection still hashes to its original digest, so
the 0.8.5-era gold environment is proven unchanged rather than re-frozen; and
the FP8-CB fused mid-M backed set is untouched (the `rungs_by_runtime_version`
table already carried `0.8.11`). Previously re-stamped (2026-08-21,
`merge/proven-rescues`) for the
**producer half of campaign rule R1** (§"The learned-codebook selector"): the
routed book burn and selector (`tools/dsv4_onlaw_book_burn.py`,
`tools/dsv4_onlaw_book_select.py`) take `--keying stack|role`, defaulting to
`stack` like the bundle builder, and a stack cell is the bundle's own fused
population — every expert's gate rows then its up rows in the profile's
declared order, weighted by the packed target's imatrix entry through the
builder's `_stack_col_weights`. A per-expert-Linear checkpoint's harvest has
no packed entry, so `tools/dsv4_packed_col_weights.py` writes one with the
export's own `_packed_expert_col_weights` derivation (the per-expert mean of
the gate and up vectors) into a sibling pickle, never in place. Verified on
DSv4 layer 0: the production builder under stack keying binds the new
`gate_up_proj` shards and the unchanged `down_proj` role shards with its own
digests. Previously re-stamped (2026-08-21, `merge/proven-rescues`) for the
**Gridbook 0.8.11 serving pin**: 0.8.11 is 0.8.10 plus two CUDA-graph capture
fixes, both reported by smb209 and neither changing a route, a codec, or a
default. gridbook#46 pre-warms the MXFP8 dense lane's swizzled-plane A-side
offsets at load (an unpinned first-use copy had aborted `FULL_DECODE_ONLY`
capture). gridbook#47: `_padded_route` read a routing-**dependent** trim count
on the host — and, for the BF16 grouped bridge, the per-expert block offsets —
so vLLM 0.27's default `FULL_AND_PIECEWISE` capture of prefill sizes above the
16-token GEMV band died at engine start; that abort was protective, because a
captured graph would have replayed one routing's tile count on every later
routing. Under capture the fused lanes launch the static-capacity layout
(`P // tile_m + E`, the `PRISMAQUANT_CB_GROUPED_TRIM=0` arm); only the opt-in
sm12x bridge (`PRISMAQUANT_CB_BF16_SM120=1`) refuses capture — the default
bridge and persistent-B never host-read. Eager and decode-band dispatch are
byte-identical to 0.8.10; measured on the shipped DSv4 body, the card command
decodes 20.53–20.61 tok/s (0.8.10: 20.54–20.63) and vLLM's default
`FULL_AND_PIECEWISE` with capture sizes up to 64 now starts and serves. Pin 0.8.11/v4, commit `187c721`, wheel digest read
from `gridbook:0.8.11-clean-187c721`'s PEP 610 record (PyPI archive
byte-identical to a tag rebuild, 60/60 members); the packaged runtime contract
is byte-identical to 0.8.10's, materialized as
`gridbook_runtime_contract.0.8.11.json` and still `lane_eligibility: absent`;
`fp8_cb_fused_mid_m` carries its backed set forward unchanged. Previously
re-stamped (2026-08-21, `feat/cb-route-status`)
for **structured route status on the CB lane** (§9.2.1,
`prismaquant/gridbook_lane_eligibility.py` +
`prismaquant/cb_route_status_gate.py`): serving-lane eligibility becomes a
fail-closed export gate instead of a serve-time discovery. Every CB serving
lane gains a `route_status_source` naming which structural class of the
runtime's eligibility table it consults, and `ResolvedServingLane` carries
`route_status` / `requires_serve_flags` / `route_status_source` resolved from
the **serving** pin — never declared in a spec file. Measured and recorded
here rather than worked around: Gridbook 0.8.10's packaged
`runtime_contract.json` (unchanged in 0.8.11) publishes `formats`, `packing`, `layout`,
`abi_features`, `quant_method` and `producer_profiles`, and **no
lane-eligibility table at all** — its own validator enforces that exact key
set — so the honest verdict for this pin is `unattested`, derived from a
byte-verbatim copy of the packaged contract now materialized in
`prismaquant/gridbook_runtime/`. Transcribing the runtime's persistent-B
role-split refusal into local constants would have been an assertion, which
principle 14 refuses. The export gate runs in both CB exporters before any
byte is written, because `role_split` — the fact that made 11 shipped DSv4
layers take an announced fallback with nothing consuming it — exists only
once the per-`(qname, format)` codebook cells resolve. Its payload shape is
load-bearing: under an absent attestation it carries `units_unattested` and
**no** backed/fallback counters at all, so the
`units_on_fallback_route = 0` defect is unrepresentable rather than merely
discouraged. `archive/gridbook_lane_2026-09-02/docs/design/gridbook_lane_eligibility_contract.md` (archived 2026-09-02) is the
proposal that asked Gridbook for a table; it is superseded as a specification
by the v3 shape Gridbook actually published (below) and is retained only as the
record of the ask. Re-stamped (2026-08-21, `feat/pooled-stack-books`)
for **pooled routed codebooks and the split-book ship gate** (campaign rule
R1, §"The learned-codebook selector"). Routed learned books are now burned per
`(layer, STACK, rung)` — gate and up pooled into one book — and the exporters
emit one codebook target for a fused weight whose bundle cell says `stack`.
The pre-R1 per-`(layer, projection, rung)` form survives behind
`build_cb_learned_bundle --routed-book-keying role` for the campaign's A/B
arm, and its artifact now needs `--allow-per-role-books` at export, stamped on
the shipcard. The defect this closes: the shipped 87 GB DSv4 body's 11 routed
FP8-CB layers agree on format — gate = up = down = `FP8_CB_K28`, union-find
promotion ran and bound — yet ship distinct `codebook_sha256` per role,
because codebook identity was in no promotion's domain and the exporter
emitted per-role unconditionally. Nothing refused it. The allocator is
untouched: no post-allocator rewrite, no format ban, and the new refusal is a
structural producer-side count of the codebooks one fused weight names, not a
claim about a runtime. Not covered here: the GPU burn that produces pooled
books, and any quality comparison between the two arms — that is the
campaign's B3 real-KL A/B. Re-stamped (2026-08-21, `feat/discovery-walker`)
for **the discovery walker** (§8.8, `prismaquant/model_walk.py`): every
weight-bearing computation is discovered by traversal — module tree plus one
matmul-intercepted forward traced under `FakeTensorMode` on a meta load —
and every discovered node must carry a claim (`decide` / `pin(reason)` /
`exclude(reason)`) from `ModelProfile.walk_claim_rules()`, or the walk fails
with the parameter named and the op cited. This is the `wo_a` enablement
generalized: the pipeline's decision universe stops being defined by the
pipeline's own enumeration. Walking the real DSv4 modeling code immediately
surfaced three further unclaimed matmul-fed families (router gates, mHC
mixers, hyper head), now pinned with reasons in
`DeepseekV4Profile.walk_claim_rules()`. API only — no consumer is wired yet;
migration of probe/cost/footprint/read-traffic onto the walker's edge list
is separate work. Re-stamped (2026-08-21,
`feat/decode-read-bytes-stat`) for **per-token decode read bytes as a
first-class exported stat** (§7, `read_gb_per_token`). An artifact has two
rates, and we only ever reported one: bpp prices the disk, while decode
throughput is set by the bytes streamed per generated token — and on a sparse
MoE those diverge by ~40x on the dense-vs-expert margin, because a dense weight
is read every token while an expert stack is read `topk/E` of the time. On the
shipped DSv4-Flash 87 GB artifact the dense path is **8.3% of the checkpoint and
76.8% of the read**; on Ornith-1.5-35B-A3B's export, **10.4% and 81.5%**. The
allocator's byte budget cannot see this, so it overspends decode bandwidth on
the dense path. Principle 1 says that is a measurement gap: `read_traffic.py`
measures it exactly (no estimates — assigned units priced by the footprint
primitives, everything else by the checkpoint's own spans, the whole ledger
reconciled against `assignment_artifact_bytes` to the byte before any
probability is applied), and it is stamped beside `achieved_bpp` on the
exporter card, the CB card, and every `validate_assignments_kl` result.
Embedding, MTP sidecar, and non-text-graph tensors are **excluded but
itemized**, never silently dropped; CB codebooks are reported as resident
bytes. **No allocator change** — pricing the axis inside the DP is a separate
decision. Re-stamped (2026-08-21,
`feat/cb-lane-sharding`) for **the CB lane joining the ~1 GiB shard standard**
(§"Output packaging"). Both CB exporters take `--shard-bytes` with the native
lane's 1 GiB default, `run-pipeline.sh` passes `EXPORT_SHARD_BYTES` on the CB
branch, and the partition rule is now one shared implementation
(`prismaquant/shard_layout.py`) rather than two writers that agreed by
inspection. One resulting shard still publishes `model.safetensors` with no
index, so a budget at least as large as the artifact reproduces every
pre-2026-08-21 CB layout; more than one publishes
`model-XXXXX-of-YYYYY.safetensors` plus `model.safetensors.index.json`, which
is what a stock HF/vLLM loader already reads. Motivation is a real user: the
shipped 87 GB single-container DSv4 CB artifact stalled the default HF loader
on a 128 GB unified-memory GB10 and the reporter resharded it by hand
(RobTand/gridbook#47). **`model_sha` is unchanged and deliberately so** — it
binds weight-container filenames and sizes (`shipcard.py:270-279`), so a
different shard budget is a different artifact identity, which is correct for
a freshly exported artifact and is not a reshard-in-place story. What the
change adds instead is a layout-INVARIANT payload digest,
`provenance.tensor_payload_identity` (`shard_layout.tensor_payload_identity`,
stamped by **both** CB exporters), hashed in the pass that already touches the
bytes: two exports of identical tensors at different budgets agree there and
differ in `model_sha`, so a resharded export stays recognisable as the same
model without changing what identity means. Previously re-stamped (2026-08-20,
`merge/proven-rescues`) for **the A-side reaching routed MoE experts, on both
the surrogate and the gate** (§"AQUA-AURA"). AQUA priced only `nn.Linear`
units, because per-channel marginals come from a backward hook and a packed
`[E, M, N]` expert is an `nn.Parameter` on a fused module: on Ornith-1.5-35B-A3B
that left **94.5% of the quantizable parameters** weight-only. The site already
existed — `install_packed_expert_hooks` swaps `F.linear` for the experts
module's whole forward and therefore sees every expert slice, **`down_proj`
included, whose input is the post-SwiGLU intermediate** — and nothing was
reading its `(x, gy)` for the activation side. Card **schema 1.1** carries
`expert_g_sq_sum`/`expert_act_sq_sum`/`expert_act_absmax`/`expert_tokens` per
expert, because routing makes `g` and the activation distribution functions of
`e` and the aggregate does not factor. Two normalizations on purpose: `g` by the
**global** token count (PR #14 — a rare expert should price low), the variance
fit by that expert's **routed** count (it is a per-token magnitude, not a
frequency); one denominator for both would discount a rare expert twice.
`PerturbedActivationCache` installs the same interception at eval, so the
selecting gate measures what the surrogate prices — vLLM registers both
`w13_input_global_scale` and `w2_input_global_scale`, so this closes a
principle-8 execution-identity gap and **frontier KL is not comparable to
earlier runs**. Coverage 5.5% → 99.89% of quantizable mass; NVFP4 A/W median
**0.67 on packed experts vs 5.77 on dense**. Corollary: `expert_empirical_cost`
must stay activation-blind or the two paths double-count. Principle 7 with it —
the A-side reduction moved to the GPU (host float64 matvec; agrees to 7.4e-9)
and the per-expert synthetic draw became one shared common-random-numbers
sample instead of ~65 s of CPU `randn`. Previously re-stamped (2026-08-20,
`main`) for **the default lane's own activation contract, and ~1 GiB output
shards**. (a) `compressed_tensors` had
never declared `served_activation_quantization`, so AQUA-AURA *refused* on the
one lane every flagship ships through — the term had only ever been priced on
CB. It is declared now, and **derived rather than asserted**: vanilla vLLM
packages no runtime contract, but on this lane what the runtime executes is a
function of the artifact we write, since vLLM's compressed-tensors dispatcher
selects the scheme from the checkpoint's own
`config_groups[*].input_activations` that `export_native_compressed` emits. Both
ends are readable, so the list is their intersection. `executes` =
`["NVFP4", "FP8_E4M3", "MXFP8_E4M3", "MXFP8_E5M2", "FP8_SOURCE"]`, verified
against the image the lane's `serve` block names and the current serving pin
alike (byte-identical vLLM `0.26.1rc1.dev693+g7f7a32cfe`), dense scheme **and**
fused-MoE method checked separately per family. **`MXFP4` is excluded on
purpose**: its scheme declares no `input_activations` key at all, so vLLM serves
it W4A16 while the format registry's descriptor calls it W4A4 — pricing it off
the registry would charge a phantom A4, the exact shape of the DSv4
mispricing. One recorded hole: `FP8_SOURCE` is executed by the runtime but
priced at zero, because its descriptor says otherwise and AQUA skips
non-activation formats before consulting the lane; no shipped artifact has
combined an FP8_SOURCE rung with AQUA, and the first that does must close it.
(b) Published artifacts are packaged in **~1 GiB safetensors shards**
(`EXPORT_SHARD_BYTES`, and `export_native_compressed`'s own `--shard-bytes`
default, both down from 5 GiB) on Robert's standing instruction; the CB lane
was out of scope then and is covered as of 2026-08-21 (stamp above).
Previously re-stamped
(2026-08-18, `main`) for
the **CB-lane A-side correction — both families execute** (§"AQUA-AURA"):
`lane_specs/nvfp4_cb.json` `served_activation_quantization.executes` is now
`["NVFP4_CB_*", "FP8_CB_*"]`. The 2026-08-17 entry (`["FP8_CB_*"]`) rested on
reading gridbook's "exact native BF16 bridge" as activations-left-exact; the
runtime QDQs NVFP4_CB activations to E2M1 group-16 on every served route
(`linear.py` `fp4_act_qdq_or_codec`, moe.py's three routed sites, `codec.py`
`fp4_group16_act_qdq`), so the bridge names a GEMM schedule, not an activation
precision — retracted the same day; spec, tests, and this doc caught up here.
Measured on the Qwen3.8-27B card: the corrected merge re-allocates identically
to the shipped artifact, while the stale entry moves 337/496 body units (272
FP8_CB → NVFP4_CB family flips). Derivation debt stands per principle 14: the
packaged Gridbook runtime contract publishes no activation table yet, so the
spec cites runtime call sites instead of asserting preflight equality.
Previously re-stamped (2026-08-18, `main`) for the **Gridbook 0.8.10 serving pin**: 0.8.10 is 0.8.9 plus a fix
for a load regression 0.8.9's own suite could not see — the tri-state
refactor renamed a `moe_gemv_select` symbol that `gridbook/moe_mixed.py`
still imported, so any artifact declaring `per_expert_format_groups` (a
split-bank mixed expert stack) died with an ImportError at `config.py`'s
dispatch, while uniform stacks — every published artifact — were unaffected.
The break shipped green because `tests/test_moe_mixed.py` importorskip'd
gridbook's own module (the importorskip all-skip trap, second occurrence);
0.8.10 skips only on missing vLLM, adds a static import-name guard, and the
serving image build hard-gates `import gridbook.moe_mixed`. No route, codec,
or default changes for uniform stacks, so the pin supersedes 0.8.9 with zero
serving-behaviour delta on everything shipped today: pin 0.8.10/v4, commit
`f4b3274`, wheel digest read from `gridbook:0.8.10-clean-f4b3274`'s PEP 610
record (PyPI archive byte-identical to a tag rebuild, 60/60 members);
`fp8_cb_fused_mid_m` gains the 0.8.10 backed-set key carried forward
unchanged (packaged runtime contract byte-unchanged since 0.8.6).
Previously re-stamped (2026-08-18,
`fix/aqua-profile-aware-resolver`) for
the **Gridbook 0.8.9 serving pin — the qualified CB kernels default on**:
three selectors became tri-state with unset → auto in Gridbook 0.8.9
(`PRISMAQUANT_CB_MOE_PERSISTENT_B` — both persistent-B payload families
engage where the load-time predicate and the extension attest, and per-role
FP8-CB split books keep the expand+grouped bridge, announced per layer — plus
`PRISMAQUANT_CB_GEMV` and `PRISMAQUANT_CB_FP8_GEMV_V2` in the routed M ≤ 16
band). Every EXPLICIT spelling keeps its exact 0.8.8 semantics, so the
canonical gold environment (`PERSISTENT_B=0`, `CB_GEMV=inherited`) replays
gold routes unchanged; `PRISMAQUANT_CB_FP8_GEMV_V2` stays outside the closed
Gridbook-0.8.5 measurement registry by design (that profile is release
evidence and its scan refuses namespace changes) **[superseded 2026-08-21 by
the producer-pin advance: the registry now describes 0.8.11 and pins that
selector to `0`; see the top stamp]**, and the 0.8.9 default-state
served leg on the shipped clean 87 GB body measured kl_mean +0.17 % /
PPL −0.06 % vs its gold record, inside the ±0.7 % cross-session KL envelope.
The serving pin moves to 0.8.9/v4 with the wheel digest read from
`gridbook:0.8.9-clean-23a3955`'s PEP 610 record (the published PyPI archive
is byte-identical to a rebuild from tag v0.8.9, 60/60 members);
`fp8_cb_fused_mid_m` gains its omitted 0.8.8 backed-set key plus 0.8.9 — the
packaged runtime contract is byte-unchanged since 0.8.6 so both mirror the
attested table — and the fp4-CB lane's detail now records that its
"MoE T > 16 expands to BF16 + Ampere grouped GEMM" sentence describes
runtimes BEFORE 0.8.9 and the announced fallbacks after it. Recorded evidence
is untouched: gold drivers bind their pin from the runtime snapshot that
carried the artifact, so shipped-era replays stay on 0.8.8 by construction.
Previously re-stamped, the same day, for
the **card-figure `model_sha` exclusion** (`CARD_FIGURE_FILENAMES`, §9.3 — the
README doctrine now also covers `allocation-map.png`/`byte-budget.png`, with a
legacy fallback) and, previously the same day, for
the **body-only parity scoping** (§9.3): `perf.matched_budget_parity`'s
verifier census walks every construction unit of the displaced container, and
a namespace-excluded (body-only) export moves the excluded units into a
separate sidecar artifact by construction (DSv4: `mtp.` lives in the 4.597 GB
DSpark draft) — a gate no correct body-only artifact can pass. `required_slots`
now scopes the demand out for such artifacts, keyed on `quant_config.json`
`provenance.excluded_namespaces` (bound by `compute_model_sha`, so not
flippable without breaking identity); a non-null parity claim on any card is
still replayed. Previously re-stamped for
the **served activation contract governing the A-side** (§"AQUA-AURA"): whether
a format quantizes activations is a property of the RUNTIME, not of the format
registry. Gridbook's CB lane decodes to BF16 and runs a BF16 GEMM unless a
process-global selector picks a fused mode, and every CB gate and gold serve
leaves it unset — so the A-side there is exactly zero. Pricing one anyway drove
DSv4-Flash at 87.403 GB from 96.8% `nvfp4_cb` (K16 bulk) to 25.4% FP8_CB with
the bulk rung crushed K16 → K12, paying weight bits to escape a cost the served
artifact never incurs. Lanes now declare
`served_activation_quantization.executes`, the pricing requires it, and a lane
that executes nothing refuses the merge. Previously re-stamped for
the **streamed rope-axis fix** (§4.1): `_call_layer` bypasses `Model.forward`,
and where DSv4-Flash's rotary keys (*rope axes* `main`/`compress`) failed to
match a layer's reported *attention schedule*, the streamed driver silently
substituted `main` — rotating 41 of 46 layers on base 10000 with YaRN off
instead of 160000 with YaRN. That is the defect behind the perplexity-262
teacher, it is a reintroduction of the bug PATCH 06 had already fixed inside
the vendored forward, and it reached the teacher, the incremental probe and the
sensitivity probe alike. The mapping now has one definition and the silent
fallback raises. Re-stamped with it for
the **teacher forward-fidelity gate** (§7.3): the DSv4 gold lane's first
execution built a BF16 teacher whose *own* teacher-forced perplexity was 262 —
far worse than the 2.34-bpp student it was grading — and every existing gate
passed it, because top-K coverage cannot distinguish a confidently wrong
distribution from a correct one. The new gate enforces **context-monotonicity**
on the teacher's own NLL, recovered from the payload alone, and refuses at
build, `teacher_meta` and `load_teacher_evidence` alike. Re-stamped with it for
the **DSv4 gold lane's first execution**, which found two further defects that
made it unrunnable as written (§7.3). Its teacher support widens from top-1024 to
**top-8192** — measured, not chosen: at 1024 the BF16 teacher misses the 0.90
per-position coverage floor on 34 of 4,088 positions, so the gate refused every
teacher the lane could build; 8192 is the smallest swept K that clears it, and
the floor itself is unchanged. And the lane now installs the **0.8.8/v4 serving
pin** rather than the 0.8.5/v3 producer pin, because both gold measurement tools
attest the serving pin against the live runtime and would refuse a producer-pin
container after the teacher pass had already been paid for.
Previously re-stamped for
the **judge/runtime split in the serve gate**, and with it the
**split-release decode topology and the per-role cover** (§9.3). The gate bound
the *judge* to the artifact's build commit as well as the served stack, which
made a validator bug incurable for bytes already on disk; the container still
runs the artifact's own build commit, but the host-side verdict now runs at
HEAD, guarded by a descendant check and a `judge-divergence` proof that every
differing closure path is judge-only. The fixes that exposed it: a DSv4 release
may now ship as two artifacts, with the body declaring itself by *recording* the
`mtp.` omission; and a routed packed expert stack is covered per **role** rather
than per tensor, because the CB ABI binds one codebook per role.
Previously re-stamped for the
**published-bpp cross-check** — `shipcard.verify` now refuses a card whose
`build.achieved_bpp` contradicts the per-unit serialized bytes the recipe it
names declares (§7). Precedence alone could not catch a right field describing
the wrong point: DSv4-Flash `artifact-aura-cb-112p69` published 4.3065 bpp from
a sibling `pareto.knees.json` for a recipe that prices to 2.7385, 57% wide.
Previously re-stamped (2026-08-17, `merge/proven-rescues`) for the
**deletion of the signed `NVFP4_CB_S*` family and the per-regime correction to
native-execution reporting** (§9.2). The signed family is gone from the registry,
encoder, exporter, footprint and serving profile: every native Gridbook FP4 route
tests the unsigned two-tier product layout exactly
(`n_sub == 2 and type_size == 4*k + 9`), so an `n_sub = 1` rung could only ever
ride a fallback, and it had already lost 78.48% of matched weight-MSE comparisons.
`subtable_bit_widths(..., "signed", ...)` and a recipe carrying `cb_mode: "signed"`
now **refuse** rather than silently resolve to the product rung of the same `k` —
a silent fall-through would hand a stale caller a different subtable geometry
under the same name. No allocation on disk referenced a signed rung, so no
assignment changed. Recorded with it: **`fused_mid_m` is one lane for one
batch-size regime, not a whole-unit verdict.** NVFP4-CB *decode* is native and
default-on on both the routed (`T <= 16`) and dense (`M <= 8`) paths; only the
batch regime rides the Sm80 BF16 bridge, and the mid-M fused lane is dense-only
and can never cover a routed expert. The DSv4 92 GB body's "73.7% unbacked" was
that misreading — its decode path is 100% native — and native execution is now
reported per regime. Previously re-stamped for the
**Gridbook 0.8.8 serving pin, the gold-contract lane scope, and the
`README.md` exclusion from `model_sha`** (0.8.7 shipped
the quantized-embedding method and its dispatch branch, but vLLM dispatches
quantized modules from inside the layer constructor and `qwen3_5.py` builds its
embedding with neither a `quant_config` nor a `prefix`, so neither was ever
reached — §3.2. Recorded with it: `shipcard.verify` was holding *every* CB
artifact to the **DSv4-Flash** gold contract, which pins
`tokenizer_mode=deepseek_v4`; the first CB artifact off that lane therefore had
`gold.kl`/`gold.ppl` slots that no measurement could fill. The flag is now the
lane, read fail-closed off the identity-bound `config.json`, and
`serve_manifest.json` — the fingerprint of a *serve*, not artifact content — no
longer moves `model_sha`, so validating an artifact stops invalidating its own
card; §9.3. **`README.md` is now excluded from `model_sha` for the same
reason**: on the Hub the README *is* the model card, so hashing it made
documenting an artifact invalidate its own shipcard — and made an artifact
structurally unable to quote its own gold numbers, since every gate record binds
`model_sha` but the gold measurements only exist after the gates. Re-running the
gates does not escape it (KL is only bit-identical *within* a docker session).
Legacy cards stamped while the README was hashed still verify, through
`compute_model_sha(..., legacy_readme_hashed=True)` behind a single
`accepted_model_shas()` helper that `verify` and `reattest` share so the two
cannot drift on which identities they accept; §9.3. **The card figures
(`CARD_FIGURE_FILENAMES`: `allocation-map.png`, `byte-budget.png`) joined that
exclusion 2026-08-18** by the same argument — they are rendered *from* the
attested `quant_config.json` after the gates by construction, decoded by no
runtime, and hashing them made illustrating an artifact invalidate the records
that measured it; exact filenames, not a category, with a
`legacy_figures_hashed=True` fallback in `accepted_model_shas()` and the same
exclusion honored by the gold inventory replay. The **whole CB ship-gate stack turned out to be DSv4-shaped**, not
just the gold contract: `validate_cb_endpoint.py` pinned the served-model
brand, `--tokenizer-mode deepseek_v4`, the eugr image and `model_type ==
"deepseek_v4"` itself, and `perf.matched_budget_parity` structurally requires a
displaced container a net-new size class does not have — nor can a *body-only*
(namespace-excluded) DSv4 artifact satisfy its census, whose walk demands the
construction units living in the separate DSpark draft; both scopings are keyed
on identity-bound artifact provenance, and a volunteered parity claim is still
replayed wherever it appears. No CB artifact on this
box — DSv4's included — had ever had a single slot filled, so the stack had
never run end-to-end. The three genuinely lane-varying constants now come from
one table, `CB_SERVING_LANE_SPECS`, that the launcher *reads* rather than
restates; §9.3.) Previously re-stamped for
**architecture-conditional naming** (a profile now receives the `model_type`/
`architectures` the checkpoint declares, and a structure spec may carry
`naming_variants` so a family's multimodal wrapper and its text-only carve-out
each get the namespace vLLM will actually build them under — §"Plug-in a new
architecture". A delegated `config_groups` target is the one string in an
artifact that must match vLLM's module tree literally; spelled for the wrong
class it matches nothing, the unit loads unquantized, and its orphaned scale
kills the load. That is how a finished 12.98 GB Qwen3.8-27B CB artifact failed
its first serve.) Previously re-stamped for the
**Gridbook 0.8.7 serving pin** (advanced from 0.8.6 so the CB lane can serve a
`quantized_embedding`, which no released runtime could; the build pin stays at
0.8.5/v3, the serving contract stays v4 byte-for-byte, and the fused mid-M
backed set is carried forward under a new key rather than widened — §3.2. The
0.8.7 image was built from the *published* wheel, so the served-image digest
rule and `pip download` now name the same archive and the 2026-08-14 cache
collision cannot recur at this version.) Previously re-stamped for
`prismaquant.dense_anchored_cb` (a **dense** sibling driver for the
platform-agnostic anchored-cost mechanism, and the first campaign to apply
**AQUA on a CB menu** — the anchored branch adds the A-side rather than
multiplying it by P5a's transfer, so AURA's activation blindness stops hiding
the `NVFP4_CB`/`FP8_CB` A4↔A8 boundary; §4.3. It renders 22.5% of a full-menu
campaign, transiently, where the stock `COST_MODE=aura` path would retain
~865 GB. AQUA-on-CB remains a candidate until its served A/B lands.) Recorded
with it: a validation rung that is also the segment's **anchor** measures
nothing — its dex is 0.0000 by construction — and the **shared planner**
`plan_cb_panel_and_validation` now refuses that configuration for every CB
driver, after the first campaign shipped 48 such vacuous cells; §4.3. The
guard originally landed in `dense_anchored_cb._panel_policy`, which has **no
importers** — it is a `__main__` — so it protected nothing on the DSv4 lane
that builds its own policy literal. A validation rung the *panel* also
contains is a weaker but genuinely different case and is **not** refused: it
still tests transfer to a held-out unit, and on a two-rung ladder whose second
rung is the anchor (DSv4 routed experts: K28, K32=anchor) it is the only
validation that can structurally exist. Every validation cell is now tagged
`held_out_axes` (`["unit","rung"]` vs `["unit"]`) and the factorisation report
breaks its counts out per axis, so a single-axis hold-out can no longer read
as a two-axis one; §4.3. Previously re-stamped at `a56c6f8`:
(`run-pipeline.sh` now forwards **`ALLOW_PINNED`** to the allocator, so a
profile-pinned `lm_head` can enter the DP budget instead of shipping at source
dtype — 2.543 GB, 20% of a 13.0 GB card budget, on Qwen3.8-27B; §4.5. Recorded
with it: a **CB rung on `lm_head` exports cleanly and does not load**, because
no gridbook method claims `ParallelLMHead` — shape legality is not servability,
and the head rides the delegated stock-CT path instead; §9.2. Previously
re-stamped at `e586670`: the CB lane can ship a **quantized token embedding**, a
weight-only `quantized_embedding` wire contract that rides neither
`config_groups` nor `ignore`, with its gridbook serving mechanism proven under
eager *and* CUDA-graph capture — §6.2 and §9.2. Previously re-stamped at `1dbf146`:
AQUA-AURA's activation term now reaches a production allocation, and its
`act_var` can be measured on real cached activations rather than modelled — see
the AQUA-AURA note below and §"Three cost tiers"). Previously verified against
implementation baseline commit `ed4f2e0` (v0.12.3) merged with the Sensitivity
Card contract branch at `c8f3cfd`. That baseline carries the cherry-picked Gridbook
wheel-attestation repair described by the named schemas and symbols below, the
Gridbook pin promotion, the serve-environment census readability fix (`SPT_NOENV`,
§"Parent + EngineCore live attestation", verified end-to-end on a live server
2026-08-14), and the serving-wheel cache integrity guard (§9.2). The final
integration commit is
deliberately not predicted in this provenance stamp. This revision includes the Spark
BF16 AURA-anchor residency and streamed-reverse lifetime corrections, the activation-safe
AURA terminal/replay policy, the endpoint live-session and matched-budget
execution-route identity contracts, the immutable AURA producer-image and mounted-source
resume identity contract, the content-addressed campaign/release source-snapshot closure,
the exact Gridbook VCS-or-wheel installed-import-origin closure, the strict Gridbook runtime-contract-v3
feature boundary, and the bundle-authoritative
per-rung learned/lattice source-map contract, the routed-MoE learned-codebook
producer contract, the DeepSeek DSpark source-overlay contract, the experimental
DSpark hybrid CB/source-FP8 sidecar producer and physical/construction namespace
contract, the finalized-assignment/render/source exact-scope closure, stable
pre/post DSpark serving-stack fingerprint equality, claim-aware optional-slot
verification, canonical DSpark group/layout/assignment and logical-codebook
payload closure, canonical frozen-byte/full-file-set publication authority with
remote parent-commit CAS, the separate current-consumer Gridbook-0.8.7 serving
profile and paired DSpark runtime/route census, strict
matched no-MTP/K12-MTP throughput-and-headroom release gate with a source-closed,
lifecycle-owned arm collector, reference-only
Entrpi long-prefill/concurrency comparison contract, the streamed CB
cached-menu render/consume contract, and the profile-declared routed-expert
AURA/empirical hybrid key-space contract, the offline value-closed DSv4
WikiText gold-input contract, plus the platform-agnostic anchored-cost
mechanism, CB mapping plugin, DSv4 one-shot acceptance-driver contract, the
anchored-AURA allocator admission branch (P0, closed 2026-08-11), the
one-purpose CPU-only W8A16 readmission plus the deliberately restamped v2
pre-export handoff gate, and the shareable Sensitivity Card contract with its
format-cost plugin seam and the default-ON probe per-channel Fisher marginal
emission that backs it (§4.8, §8.7, §12 D30).

The current handoff approval binds the 15-file
`prismaquant.dsv4_w8a16.export_source_closure.v1` semantic closure at identity
`7517d87cfc231e5cfa60faa74728505a272540d33d8bc15c2196c54e62fa7fca`;
the v2 handoff implementation source is
`ba1a7ae844a2187abe24f059037ceb22d3da73aecf2136ead39230b5cb8320de`
and its reviewed streaming-exporter member is
`f740ebd9b90e586fc10ffa975f2f624ee8f7f85b02bb927701e946989bc3319f`.
older v1 handoff receipts remain dated history and are not current export
authority.

The Sensitivity Card item changes exactly one shipping default, and it is
probe-side: `incremental_probe.py` now emits five per-channel marginal vectors
into `probe.pkl` unless told not to (§3.3, §4.1). The card and its three cost
tiers remain additive modules with no `run-pipeline.sh` call site and no
`COST_MODE` value.

**AQUA-AURA is no longer one of them.** As of 2026-08-14 its activation term
reaches a production allocation: `aqua_activation_cost.py` prices the A-side off
a card and merges it into an existing weight-only cost pkl as `act_dloss`, and
`allocator_candidates.cost_entry_predicted_dloss` adds it on the weight-only
branches only (`78ec816`, `3c14209`). The term rides **inside** P5a's per-family
multiply — `(base + act) · penalty`, never `base · penalty + act`, since the
fitted constants run to ×8103 and adding outside dilutes an A-side worth 6× the
W-side to 0.07%, which produced a Pareto byte-identical to the weight-only one.
`act_var` is measured on the probe's real cached activations when
`--act-dir` is given and modelled from a per-channel Gaussian fit otherwise
(`1dbf146`; the two differ by 3.23% of Linears at 5.0–5.5 bpp). It is **still
research** — there is no served KL/PPL A/B against the weight-only arm at
matched bpp, and that A/B is the promotion gate.

**It covers routed MoE experts as of 2026-08-20, not just the dense trunk.**
Until then the A-side reached only `nn.Linear` units, because per-channel
marginals come from a `register_full_backward_hook` and a packed `[E, M, N]`
expert is an `nn.Parameter` on a fused module — on an A3B that is 94.5% of the
quantizable parameters priced weight-only. `sensitivity_probe`'s existing
`F.linear` interception (`install_packed_expert_hooks`) is the equivalent site
and already holds `(x, gy)` per expert slice, **including `down_proj`'s, whose
input is the post-SwiGLU intermediate**; it now also accumulates
`expert_g_sq_sum [E, M]`, `expert_act_sq_sum [E, N]`, `expert_act_absmax
[E, N]` and `expert_tokens [E]` into card **schema 1.1**. Per expert, not
aggregated: routing makes both `g` and the activation distribution functions of
`e`, and `Σ_e W²·g·var ≠ (Σ_e W²)(Σ_e g)(Σ_e var)`.

Two normalizations, and they differ on purpose. `expert_g_sq_sum` divides by
the **global** calib token count (the PR #14 convention — a rarely-routed
expert *should* price low, that is its share of the objective);
`expert_act_sq_sum` divides by **`expert_tokens[e]`**, because it fits a
per-token noise magnitude rather than a frequency. Using the global count for
both discounts a rare expert twice — PR #14's inverted importance weighting in
mirror image. Corollary for the empirical branch: `expert_empirical_cost.py`
must stay **activation-blind**, or the two paths double-count the expert
A-side.

**The selecting gate measures the same term (2026-08-20).**
`PerturbedActivationCache` emulated activation quantization with a module
`forward_pre_hook`, which on a packed-experts module sees only the module input
— gate_up's. It now installs the same `F.linear` interception at eval, so each
projection is quantized with its own calibrated scale, and the pre-hook stands
down for packed plans so gate_up is not quantized twice. vLLM's
`CompressedTensorsW4A4Nvfp4MoEMethod` registers **both**
`w13_input_global_scale` and `w2_input_global_scale`, so the served runtime
quantizes both: this closes a principle-8 execution-identity gap, and frontier
KL values are **not comparable to pre-2026-08-20 runs**. Measured on
Ornith-1.5-35B-A3B: card AQUA coverage 5.5% → 99.89% of quantizable mass, merge
310→390 units (the 11 remaining are BF16 MTP sidecar), and the NVFP4 A/W ratio
is median **0.67 on packed experts vs 5.77 on dense** — experts do not behave
like the trunk.

**The A-side belongs to the SERVING LANE, not to the format registry — and on
this lane BOTH CB families execute their activation grids (corrected
2026-08-18).** Whether choosing a format actually quantizes activations is a
property of the runtime that will serve the artifact, so
`activation_dloss_table` refuses the format registry's word for it and resolves
the executed set from the lane spec. On `nvfp4_cb` gridbook executes both:
`FP8_CB_*` is W8A8 — `gridbook/linear.py` hands quantized `xq` with per-token
dynamic scales to `native_cutlass_scaled_mm`, and `moe.py` declares
`_FP8_GROUPED_CONTRACT = "fp8_per_token_dynamic"` — and `NVFP4_CB_*` rounds
activations onto the E2M1 group-16 grid before every GEMM (`linear.py`
`fp4_act_qdq_or_codec`; `moe.py`'s three routed `fp4_act_qdq` sites;
`codec.py` `fp4_group16_act_qdq`, which mirrors
`format_registry._make_rtn("fp4_e2m1", 16)`). Gridbook's "exact native BF16
bridge" is a **GEMM-schedule** statement, not an activation-precision one: the
BF16 operand has already been rounded onto the E2M1 grid, and
`nvfp4_activation_contract.bridge_contract` documents that every quality route
consumes the exact native QDQ the decode path uses, differing only in GEMM
schedule. A 2026-08-17 revision read "exact" as "activations left exact" and
scoped the lane's executes to `["FP8_CB_*"]`; it was retracted the same day —
the runtime was never ambiguous — and the spec, this section, and the tests
were corrected on 2026-08-18. The shipped DSv4 92 GB body and the shipped
Qwen3.8 13 GB artifact were priced with both A-sides and are correct as
shipped. The stakes are measured, not argued: re-allocating the Qwen3.8 card
under the stale entry moves **337 of 496 body units** (272 of them FP8_CB →
NVFP4_CB family flips — the DP floods into the family whose real activation
error the cost stopped seeing, reporting predicted_dloss 0.0486 vs the shipped
0.0735 only because it prices less of reality), while the corrected entry
re-allocates to the shipped `layer_config.json` **byte-for-byte**. Fresh cost cards consume this list **silently**
(`dense_anchored_cb` resolves `lane_id="nvfp4_cb"` with no flag; any non-empty
set means no refusal), which is why the value carries runtime call-site
citations, and — per principle 14 — should graduate to a preflight equality
assert once gridbook packages its per-scheme activation contracts in
`runtime_contract.json` (it computes them today; it does not package them).
Lanes declare `served_activation_quantization.executes` as **glob patterns
over format names** (`lane_spec.LaneActivationContract`,
`lane_specs/nvfp4_cb.json` — `["NVFP4_CB_*", "FP8_CB_*"]`), because rungs
within a family are open-ended; enumerating rungs would under-declare silently
the day one is added. `activation_dloss_table` **requires** the pattern set
and refuses to fall back on the registry's W4A4 claim; a lane that declares an
empty set refuses the merge outright rather than writing an all-zero A-side
that is indistinguishable later from a real one. Pass
`--lane-executes-all-activation-grids` for a lane that genuinely serves every
activation grid.
Name resolution goes through the **model profile**: the stage inverts
`checkpoint_to_live_name` over the safetensors index, because a card's unit
names come from the module tree the probe walked while a checkpoint may rename
both the path and the leaf (DSv4-Flash stores
`model.layers.N.mlp.experts.E.down_proj` as `layers.N.ffn.experts.E.w2`, which
resolved **0 of 33,325** units under the previous path-only aliasing). The
forward direction is not usable — `source_tensor_name` rewrites the path but
not the leaf. Two failures that were previously silent now **refuse**: zero
name resolution, and a merge that writes zero entries (the signature of a
scalar-only card with no `g_sq_sum`). Both would otherwise emit an artifact
with the right units and formats and an absent A-side, which
`cost_entry_act_dloss` reads as 0.0 — i.e. free — making "unmeasured"
indistinguishable from "costless" to the DP.
On a **quantized-source** checkpoint the resolved tensor is not a dense weight
(DSv4-Flash: MXFP4 nibble-packed routed experts, block-FP8 everything else,
both with E8M0 `.scale` siblings), so `materialize_source_weight` dispatches on
the streaming loader's own `_build_fp8_scale_inv_map` — declaration-driven,
never shape-inferred — and reuses the loader's decoders
(`dequantize_mxfp4_source`, `_dequant_fp8_block_weight`); the materialized W is
bit-identical to the weights the loader installs for the probe. Dense
checkpoints hit the map-empty passthrough and behave exactly as before; any
tensor that cannot be materialized to the card's `(out, in)` **raises** rather
than pricing a wrong-shaped or code-range W.
The external runtime record pins Gridbook **0.9.1** at exact commit
`227420f9821bab7089632ee914f0ba050f82b817`, with
`gridbook.runtime-contract.v12` and the exact required feature map
`routed_moe_per_role_codebook_lut=1`, `source_fp8_block128_w8a16=1` plus
`dspark_construction_physical_bridge=1`, and `version_is_release=true`. It
advanced 0.8.5/v3 → 0.8.11/v4 on 2026-08-21 and 0.8.11/v4 → 0.9.1/v12 on
2026-08-30, and is held in lockstep with the serving pin by a test. The
feature map is unchanged across the v12 advance: the 0.8.11 and 0.9.1 packaged
contracts have **equal** `abi_features`, `quant_method`, `packing`, `layout`
and `producer_profiles` maps, and differ only in `contract_version`, the
`formats` rows (two trellis families added, `NVFP4_CB_S` removed,
`producer_rungs`/`kind` added to the two CB families) and the three new
`lane_eligibility` / `tensor_parallel` / `expert_parallel` keys. The 0.8.5 wheel — the release that first attested
`source_fp8_block128_w8a16` — passed its GB10/sm121 GPU gate (91 passed, 0
skipped), including
raw-source W8A16 residency, native decode/prefill dispatch, and JIT extension
identity/capability. That closes the route-existence and export gates; exact
full-artifact eager/graph generation, performance parity, and served quality
remain post-export shipcard gates. The measured command, immutable inputs, and
raw evidence paths are recorded in
`docs/results/gridbook_0p8p5_w8a16_gate_2026-08-12.md`; that gate has not been
re-run on 0.8.11 or 0.9.1, both of which carry the same attestation forward in
their packaged contracts; the claim's scope is still the 0.8.5 measurement. Gridbook 0.8.4 introduced the explicit
routed per-role LUT feature; capability
decisions now read the closed feature map rather than infer from a numeric
version. The on-law K28/K32/K36/K40/K44/K48 FP8-CB set is unchanged. This
branch also preserves the dated 2026-08-01 DeepSeek-V4-Flash-0731
92 GB study record (§9.2) as historical candidate-era evidence. It is not promotion
evidence for the current runtime: the release path is the separately gated
112.690 GB AURA artifact authorized only by the exact W8A16 handoff.

That producer record is not silently repointed when the serving plugin
changes: it moves only by a reviewed edit, and since 2026-08-21 a test refuses
a producer/serving pin that name different releases, so drift is loud in
either direction. Serving has a
second, current-consumer pin in
`gridbook_runtime/gridbook_serving_runtime_pin.json`. It names Gridbook **0.9.1**
(0.8.6 → 0.8.7 on 2026-08-15, → 0.8.9, → 0.8.11 on 2026-08-21, → 0.9.1 on
2026-08-30), runtime-contract v12, the
routed-MoE/source-FP8 features above, and
`dspark_construction_physical_bridge=1`. **That pin is now RESOLVED** (release
commit `98916b09`, `version_is_release=true`), superseding this section's former
description of it as `PENDING_...` sentinels;
`test_packaged_serving_pin_is_resolved_and_loads_in_shell` asserts the resolved
state from both the Python and shell readers.
`gridbook_serving_runtime.sh` downloads or accepts only the exact pinned wheel,
validates its archive paths and METADATA, installs it from the read-only local
file, and checks PEP 610 plus the feature closure at the pinned schema. A wheel that fails that
digest check is **never published into the digest-named cache** — the cache's
first branch trusts the directory name, so caching a rejected wheel is permanent
and bricks the lane (2026-08-14 incident; guarded and regression-tested in
`tests/test_gridbook_serving_runtime_pin.py`).

**Which wheel the digest names — the tension is now dissolved in practice, not
resolved in principle.** This section formerly required the digest "reported by
the published PyPI file", on the ground that a local pre-tag wheel is not
byte-reproducible evidence. The shipped pin does the opposite, and
`prismaquant/gridbook_serving_runtime_pin.py` states that rule normatively: the
digest is read out of the **served image** — the runtime the accepted numbers
were taken on — and "a locally rebuilt wheel is a DIFFERENT archive and must not
be substituted here". Both cannot be the rule.

At 0.8.6 the two rules genuinely disagreed on which archive to name. Measured
2026-08-14: the two 0.8.6 wheels were **content-identical** (all 58 archive
members byte-for-byte equal; they differ only in zip container metadata), and
the PyPI wheel was built by the release run from exactly the pinned commit
`dde15e04` — so the rules selected the same *code* and differed only in which
archive's digest was asserted. The consequence was operational: a plain
`pip download gridbook==0.8.6` did **not** satisfy that pin, and the lane had to
be given `GRIDBOOK_SERVING_RUNTIME_WHEEL=<wheel extracted from the served image>`
(or a correctly populated cache).

**0.8.7 makes them agree.** `gridbook:0.8.7-clean-98916b0` was built by
installing the *published* `gridbook==0.8.7` archive from a local file, so the
image's PEP 610 `archive_info.hashes.sha256` **is** the PyPI wheel's digest
(`c58daf5d…`). The served-image rule still governs — the pin is still read out
of the image — but it now selects the published archive, so `pip download` and
the cache path both satisfy it and the 2026-08-14 poisoning collision cannot
recur at this version. Verified before use: all 59 members of the PyPI wheel are
byte-identical to a local rebuild from the tag. Which rule is normative when
they *don't* coincide is still Robert's call; building the image from the
published wheel is the cheap way to keep the question moot.

**0.8.8: the mechanism 0.8.7 shipped was unreachable on this architecture.**
vLLM dispatches a quantized module from inside the layer's own constructor, and
`qwen3_5.py` builds `VocabParallelEmbedding(vocab_size, hidden_size)` with
neither a `quant_config` nor a `prefix` — so `get_quant_method` is never called
for the lookup table and both the method and the dispatch branch 0.8.7 added
were dead on Qwen3.5/3.6. It fails silently at construction and surfaces later
as vLLM refusing a parameter nothing claims (`embed_tokens.weight_global_scale`).
`gridbook/embedding_construction.py` supplies the two arguments the model's own
`__init__` already holds, inert unless the config is ours, declares that exact
prefix, and the embedding was built bare. This is load-bearing for the
Qwen3.8-27B "A" artifact: its embedding is 248320x5120 = 1.27 B parameters,
1.83 GB of 12.98 GB, and serving it BF16 instead takes the artifact from
12.09 GiB to 13.79 GiB — which is the 16 GiB card. `gridbook:0.8.8-clean-064a4cb`
is built the same way (from the published wheel), so its
`archive_info.hashes.sha256` is again the PyPI digest (`a982e884...`), verified
60/60 members byte-identical to a local rebuild from the tag.

**What 0.8.7 changed, and what it did not.** It adds the `quantized_embedding`
serving mechanism (`model.embed_tokens` as a Gridbook-claimed, weight-only unit
— vLLM's compressed-tensors embedding path raises for FP4/FP8) and re-audits the
vLLM 0.26 dense-NVFP4 preflight, which had no entry for the
`FlashInferCutlassNvFp4LinearKernel` that 0.26's ladder actually selects and so
failed closed on every delegated dense W4A4 group. Its three new routed-MoE
kernel lanes are all opt-in and default-off, and `gridbook/runtime_contract.py`
is byte-identical to 0.8.6 — contract v4 and the three-feature ABI closure carry
over, so the **build** pin (0.8.5/v3) is untouched and the fused mid-M backed set
`{28,32,36,40,44,48}` is carried forward under a new `0.8.7` key rather than
widened. The DSpark serving profile shares this pin; its closed source census
was re-run against both tags rather than assumed, and gains exactly two
identifiers (`PRISMAQUANT_CB_FP8_GEMV_V2`, `PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R`),
both declared and pinned **off** because they name lanes that did not exist in
the runtime its accepted numbers were measured on.

**Which pin a launcher crosses is a property of the artifact.** Until
2026-08-15 `scripts/serve_dsv4_cb_validate.sh` was the only script sourcing
`gridbook_serving_runtime.sh`; every other launcher sourced the build-pin
helper and installed the build pin from source into `vllm-node:latest` (0.8.5
at the time; 0.8.11 since the 2026-08-21 producer-pin advance — those
launchers' artifacts have not been re-served on it). A CB
artifact whose recipe assigns `model.embed_tokens` to NVFP4 cannot load that
way — the quantized embedding mechanism does not exist before 0.8.7 — so the
Qwen3.8-27B CB "A" build serves through
`scripts/serve_qwen38_cb_a_smoke.sh`, the second serving-pin launcher, against
`gridbook:0.8.8-clean-064a4cb` — 0.8.8 and not 0.8.7, because on 0.8.7 the
declared NVFP4 embedding never reaches our dispatch and the artifact does not
load at all. It still runs `install-container`, which here
is a reinstall-and-verify no-op whose value is the assertion, not the install:
it proves the running interpreter imports the exact reviewed archive. The
membership set lives in `tests/test_gridbook_runtime_boundary.py`
(`SERVING_PIN_SCRIPTS`). The two pins remain independent *mechanisms* — only
the serving pin binds a wheel digest — but since 2026-08-21 they may not name
different releases: `test_producer_and_serving_pins_name_the_same_gridbook_release`
fails on drift. Choosing the wrong helper is still a load-time failure rather
than a warning; what the lockstep test removes is the silent three-release gap
that made the wrong helper *plausible*.

This revision retains the four 2026-07-30 architecture re-vet waves documented in
`docs/audits/architecture_re-vet_2026-07-30.md` and closes the runtime-ownership debt: the
vendored Gridbook tree and sync path are gone, producer ABI/menu/config facts have one owner,
and required CI checks the independent producer and consumer at one immutable commit. The
Gridbook boundary (0.8.5 then, 0.8.11 now) carries forward the closed
measurement environment — 29 variables then, 31 since the 0.8.11 advance — with
`GRIDBOOK_MXFP8_DENSE` now affirmatively absent, exact installed-distribution provenance,
artifact-derived native-extension requirements, and a dedicated raw-source W8A16 kernel
family. These harden evidence and admission and back the source W8A16 route for export; they do
not by themselves promote an unmeasured full artifact. The four behavioural facts a
returning reader must know are that **`COST_MODE` defaults to `aura`** (§3.3), Gridbook serving
is native CUDA/CUTLASS-only and fails closed (§9.2), and fused native-NVFP4 remains default-off
after its teacher-backed quality gate (§9.2); direct group-32 MXFP8 remains W8A8 while the
block-128 checkpoint source route is W8A16.

**Prime directive:** the code is the authority. Where this document and the tree disagree, the
document is wrong — fix it, or record the divergence in §12; never propagate it.

---

## Contents

[0 Maintenance contract](#0-maintenance-contract) ·
[1 What PrismaQuant is](#1-what-prismaquant-is) ·
[2 Methodological spine](#2-methodological-spine) ·
[3 The quantization pipeline](#3-the-quantization-pipeline) ·
[4 Cost models & allocation](#4-cost-models--allocation) ·
[5 Formats & render](#5-formats--render) ·
[6 Export & serving invariants](#6-export--serving-invariants) ·
[7 Validation & ship gates](#7-validation--ship-gates) ·
[8 Model support: the plugin architecture](#8-model-support-the-plugin-architecture) ·
[9 Serving lanes](#9-serving-lanes) ·
[10 Hardware & environment](#10-hardware--environment) ·
[11 History](#11-history--what-was-tried-and-rejected) ·
[12 Known gaps and debt register](#12-known-gaps-and-debt-register)

## 0. Maintenance contract

This file is the master document. `docs/README.md` is the index and carries a status tag
(CURRENT / HISTORICAL / ARCHIVED) per doc; everything else is a rule set this file points at,
a lane record, or history.

**The rule.** A commit that changes any of the following must update this file in the same
commit: (1) a `prismaquant/run-pipeline.sh` default, gate, or stage order (§3); (2) the format
menu, a scale rule, or a render lever (§5); (3) an export codec, a `config_groups` emission
rule, or a serving invariant (§6); (4) a ship-gate threshold or what the pipeline runs versus
echoes (§7); (5) the plugin contract — profile accessors, registry order, serving-profile
schema, gridbook per-arch wiring (§8); (6) a serving-lane default or a promoted/reverted kernel
lever (§9). If topology changed, the affected mermaid diagram changes with it. The provenance
block at the top must be re-stamped (date, commit, branch) on every substantive edit.

**Corollaries.** Handovers (`docs/handovers/`, gitignored) and dated results (`docs/results/`,
`docs/lanes/*/`) are append-only history: they record what was true on a date and never
substitute for updating this file. Every normative claim here carries a `file:line` or a commit
hash — a claim without one is a lead, not a fact. Staleness discovered in this document is a
bug: fix it, or if the fix is larger than the edit, add it to §12 with a severity. Do not
silently leave a wrong line here for the next reader.

## 1. What PrismaQuant is

Mixed-precision LLM quantization that chooses a serving format **per Linear**, selects the
assignment on **real end-to-end KL-vs-BF16**, and ships the result as an artifact a stock or
plugin-extended vLLM serves. No forked runtime on the native lane. Allocation is a
multiple-choice knapsack over a per-(Linear, format) cost (§4); the winning candidate is decided
by measurement, not by the cost model (§2).

### 1.1 The three artifact containers

| Lane | Container | Runtime | Formats | Status |
|---|---|---|---|---|
| Native | `compressed-tensors` | vanilla vLLM, Blackwell CUTLASS | maintained: NVFP4, FP8_DYNAMIC/E4M3, BF16; FP8_SOURCE remains a source-artifact compatibility codec | production default for the maintained W4A4/W8A8/BF16 menu; W8A16/source FP8 is not SM120 performance eligibility |
| **Tessera** | `tessera` wire (`quant_method = "tessera"`) | vLLM + **Tessera's own** plugin (`tessera.serving`), installed into the stock image from an immutable pin; PrismaQuant consumes only the packaged `runtime_contract.json` | the Tessera trellis wire, synthesized and priced by name (§5.7) | **declared, with a container arm, and refused by exactly one thing**: `EXPORT_CONTAINER=tessera` is a lane the driver can act on since 2026-09-02 (§9.4), and the pin's PENDING sentinels are the only refusal left — `tessera_lane_attested` answers False for every rung by the pin, not by an edit. Dense-only at TP=1; the contract carries no `routed_moe` cell |
| GGUF | single `.gguf` | llama.cpp; vLLM via `vllm-gguf-plugin` | Q2_K…Q8_0 k-quants + IQ family + BF16 | enabled end-to-end; the only 2–3 bpw path |

A fourth container — the **codebook (CB) / gridbook** lane — was **retired 2026-09-02** on Robert's decision (*"put Tessera in PrismaQuant and remove Gridbook"*) and archived whole at `archive/gridbook_lane_2026-09-02/`, its served measurements included. `EXPORT_CONTAINER=nvfp4_cb` now `exit 2`s. See §9.2 for what went with it.

Lane detail, defaults and proven results: §9. Export codecs: §6. Pipeline defaults: §3.3.

### 1.2 Shipped artifact family

bpp is over **quantizable body** parameters only (excludes `lm_head`, MTP/visual sidecars,
and profile-pinned Linears) and labels are **not** comparable across accounting eras (§12).
A fixed quantized `lm_head` stays outside that body denominator and the body's predicted
Δloss, but its measured auxiliary Δloss and exact serialized payload are carried separately and
its bytes count toward every whole-artifact budget (`allocator.py` `aux_fixed_*` fields).
conf-KL =
confident-position KL-vs-BF16; ALL-KL = all positions. Comparative lane deltas belong to §9;
the numbers below are each artifact's own readout.

| Artifact | Lane | bpp | Quality readout | Provenance |
|---|---|---|---|---|
| Qwen3.6-27B `prismaquant-cb-5.5bit-vllm` | CB | 5.501 | ALL-KL **0.0134** / conf-KL 0.0113; PPL 9.166 vs BF16 9.123 | `docs/lanes/nvfp4-cb/prod_27b_results.md` |
| Qwen3.6-27B `PrismaAURA-5.5bit` | native | 5.5 | ALL-KL 0.0321 / conf-KL 0.0241; TEB 91 (BF16 86) | same A/B table; TEB from memory, unverified vs code |
| Qwen3.6-27B PrismaSCOUT 5.31 (DOI `10.57967/hf/8656`) | native | 5.31 (≈4.76 under current accounting) | held-out KL 0.0151, 20.17 GB | superseded by the two rows above |
| Ornith-1.0-35B (CB) | CB | 4.758 | conf-KL **0.01706** / ALL-KL 0.0278; PPL 9.542 (+1.1%) | `docs/lanes/nvfp4-cb/prod_35b_results.md` |
| Ornith-1.0-35B PrismaAURA | native | 4.748 | conf-KL 0.03625 re-measured; the older **0.0143** figure is a different protocol and is *not* comparable | `prod_35b_results.md` |
| Hy3-295B-A21B `prismaquant-cb-2.9bit-vllm` | CB | 2.902 | no quality claim possible (no 295B BF16 teacher on one box); TEB 87/100; serves on one Spark | `docs/lanes/nvfp4-cb/prod_hy3_results.md` |
| Hy3-295B-A21B `PrismaQuant-2.8bit-gguf-vllm` | GGUF | 2.799 (103.686 GB) | TEB 87/100 (IQ) vs 86 (k-quant) | `docs/lanes/gguf.md` |
| Hy3-295B-A21B `PrismaQuant-5.3bit-2xSpark-vllm` | GGUF | 5.3 (190 GB) | two-Spark target | memory, unverified vs code |
| Laguna-S-2.1 117B | CB | 6.0 (84 GB) | no BF16 teacher at 117B; serves 256k ctx | memory `laguna_s21_lane`, unverified vs code |
| Gemma4-31B-IT | native | 6.0 | −24% conf-KL vs the shipped 5.5, +5.9 pp top-1 | memory, unverified vs code |
| LFM2.5-8B-A1B | native | ~6.58 (labelled 6.5) | ToolEvalBench = BF16 parity | memory, unverified vs code |
| Qwen3.5-122B-A10B · Mistral-Medium-3.5-128B · Qwen3.6-35B-A3B | native | 4.75 | — (the 35B-A3B predates 4 allocator/export fixes; do not re-export without an orthogonal reason) | memory, unverified vs code |
| MiniMax-M2.7 | native | 3.2 | — | memory, unverified vs code |

The two CB rows carry the load-bearing result: at matched body bytes, codebook formats buy
materially more quality. The magnitudes, and the speed side of the trade, are §9.2.

Author: Robert Tand, independent researcher; public attribution uses
`robert.tand@icloud.com`. Paper: `paper/main.tex` (AURA spine; the PrismaSCOUT spine was
retired 2026-06-05 and archived at `paper/archive/prismascout_paper_2026-06-05.tex`).

## 2. Methodological spine

### 2.1 Two axes

- **Local** — *given a fixed format, how do you round this Linear best?* Well studied: GPTQ,
  AutoRound, rotations, scale rules. The render toolkit; it runs *under* whatever format is
  chosen (§5).
- **Global** — *how many bits does each Linear get, and in which hardware format?* Allocation,
  and the contribution (§4). Sensitivity is wildly unequal across a transformer's Linears, so a
  heterogeneous assignment extracts quality no single-format method structurally can.

### 2.2 Surrogates generate, real KL selects

The governing sentence: *an allocator does not need a perfect cost model if every candidate it
proposes can be cheaply re-scored end-to-end on a held-out split.* Cross-layer interaction
therefore stops being a quantity you must **model** and becomes one you **observe**.

The modelling branch of the literature (CLADO's pairwise IQP, HAWQ-V3's second-order ILP,
CoopQ's Shapley allocator) is not reproduced here: measured pairwise interaction is noise at
the bit-widths that ship (3/1180 pairs significant; pair-term ρ = −0.10), and the apparent
non-additivity is largely a bf16 KL-differencing floor — in fp32 the per-Linear unary KLs are
near-additive (`paper/main.tex` §additivity; §11).

**There is one cost level, not three.** The three-level cascade (L1 additive Fisher → L2
perturbed-X fixed point → L3 propagated end-KL) was **retired from the spine on 2026-07-30**
and its code walled at `archive/l3_propagated_2026-07-30/` (re-vet R4). What ships is *one
faithful unary cost* — **AURA by default since 2026-07-30** (re-vet R2),
`production-render-score` as the explicit/legacy spelling, plus measured empirical unit-KL for
profile-declared routed experts in either packed or per-expert-Linear form — and *real held-out KL* to select among
the candidates that cost proposes. The retiring evidence is measured, not argued: the L2
fixed point beat additive L1 by **−1.5%** while AURA beat L1 by **−38.5%** on the same
baseline (`aura_cascade_headtohead`); a better single cost was worth 25× more than another
level. Status, citations and what survives: §4.4; wall and lesson: §11.

### 2.3 Metric authority

Highest first. A claim is worth exactly the rung it was measured on.

| # | Metric | Contract | Where |
|---|---|---|---|
| 1 | Served-artifact vLLM KL-vs-BF16 at matched bpp: exact full vocabulary where feasible; DSv4Flash all-position top-8192 support plus one tail bucket | n=8 × seqlen=512 | `tools/measure_vllm_full_kl.py`; DSv4 source builder `tools/build_streamed_full_kl_teacher.py` with the offline input from `tools/prepare_dsv4_wikitext_inputs.py` — invoked **manually**, never by the pipeline |
| 2 | Direct WikiText PPL on the served artifact | pinned WikiText test revision; 8,192-token prefix in 16 non-overlapping 512-token windows; 8,176 scored positions | `tools/measure_vllm_wikitext_ppl.py` with that same offline input, contract `prismaquant.wikitext_ppl_calibration/1` — manual |
| 3 | Mean NLL alongside PPL; KL-vs-BF16 (`/home/rob/dq-runs/kl_tool.py`) for IT/BOS-sensitive models where raw PPL is meaningless | — | §7.5 |
| 4 | Downstream suite on materialized artifacts: GSM8K, IFEval, MMLU, **ToolEvalBench** (`--no-think --hardmode --parallel 1`) | — | tool-use fidelity is the deep reason KL matters: a small probability shift at a decision point flips a tool call |
| 5 | Cheap last-token "hook KL" screens | — | **triage only**; never a selection or promotion metric |

Rung 2 can veto a rung-1 win — a lower *mean* KL can hide a heavier tail. A candidate that
improves calibration KL but regresses held-out PPL/NLL or a downstream task stays
research-only unless Robert explicitly accepts the trade. (The selector has no tail term
today; §12 D1.)

**Held-out discipline.** The selection split must be disjoint from the text that generated any
cost — an audit found "validation" KL had been in-sample; the house rule and the
token-disjoint construction are documented at `validate_assignments_kl.py:513,581`. Small-scale
levers are validated on Qwen3-0.6B *and* 4B with `--calib-repeats ≥ 4`; single-seed n=8/T=512
is dangerously noisy (+10% can flip to −5.2% across repeats).

**Reproducibility is a gate.** Git commit, calibration hash, assignment hash and cache
hit/miss/RTN-fallback counts are baked into output JSON; an irreproducible number is
quarantined, not trusted. KL is bit-identical within a docker session and drifts across them —
mechanism, magnitude and the resulting A/B rule are §7.4.

### 2.4 Promotion ladder

| Stage | Bar |
|---|---|
| Research | opt-in, documented, excluded from defaults |
| Candidate | small-model GPU + vLLM smokes, plus a measurement plan on a real target |
| Production recipe | wins or preserves KL/bpp/runtime on the target stack; serving suite green; tests |
| Default-on | cleared on the target **and** one more representative model/shape |

Regression or inconclusive → demote back to Research. The numeric ship gate that guards
materialization is separate, automated and thresholded (§7.2).

### 2.5 Honest accounting

Retraction is routine and is itself a deliverable. The grouped-KL surrogate's "−3.52% PPL win"
was a local/HF screen that **inverted** on the vLLM A/B; the "17 promotions / 0.0056 KL" polish
headline and the "4× lower KL" framing were withdrawn the moment the comparisons were found
non-rigorous; the staged-render last-token-KL win regressed direct PPL; `current_only`
extrapolation won its hook screen and lost full-vocab KL; the damp sweep's "+137.5% if
disabled" was a hook screen that inverted on the gold lane. Hence the rule: **never sell a
screen as a result.** Expect most pipeline "improvements" to be <5% deltas — the cost surrogate
is itself mis-ranked against PPL at the margin (5.5 bpp beats 6.0 bpp on Qwen3-4B WikiText
PPL). Negative results are recorded with the durable lesson (§11); the paper publishes the
graveyard.

## 3. The quantization pipeline

The orchestrator is `prismaquant/run-pipeline.sh` — **not** the repo root; several older docs
imply a root-level copy and there is none. One bash script, four numbered phases (probe → cost
→ allocate → cache+export), each phase file-artifact-coupled and skip-if-exists.
`prismaquant/pipeline.py` is a *declarative* spec layer invoked once at the top; it executes
nothing (§3.6).

### 3.0 Optional PrismaSnap BF16 source preparation

**PrismaSnap is purely additive to this graph.** It is an offline BF16 checkpoint-to-checkpoint
pre-pass, not a format, allocator, cache, export layout, kernel, or runtime adapter. An original
source with no `prismasnap_provenance.json` enters the four numbered phases exactly as it did
before this feature. A snapped source is still an ordinary HF BF16 checkpoint; once admitted,
the same probe, AURA objective, per-Linear allocator, `ProductionWeightCache`, recache, native
exporter, and serving validators run with no PrismaSnap scheduler or serve-time operation
(`prismasnap.py` module contract; `prismasnap_contract.require_verified_prismasnap_if_present`).

**The source may be BF16 or native FP8; the fold's output is always BF16.**
`SUPPORTED_SOURCE_DTYPES = {BF16, F8_E4M3}` (`prismasnap_checkpoint.py`). A fold is a
per-**channel** diagonal, while a native-FP8 checkpoint carries a per-**block**
`weight_scale_inv`, so the fold is not constant inside a block and cannot be absorbed into the
scale. An FP8 operand is therefore dequantized once through the checkpoint's own declared grid
(`_Checkpoint.load_bf16`, delegating to the streaming path's `_declared_weight_block_size` /
`_apply_fp8_dequant_inplace` — one dequant mechanism, per principle 8) and folded in BF16, which
is the dtype `attest_fold_fidelity` serves anyway. The materializer forces `output_dtype` to
BF16 for those operands: letting it default to the source dtype would round each fold back into
`float8_e4m3fn` while the stored block scale stayed put — wrong values with no error. A BF16
source takes a byte-identical no-op path. `I8`, per-tensor-scale FP8, and any dtype PrismaSnap
cannot lift exactly are refused by name; a checkpoint that pairs FP8 weights with scales but
declares no `weight_block_size` fails closed rather than assuming a 128x128 grid.

An uplifted operand's `weight_scale_inv` no longer describes it, so it is dropped from the
shard and from the output index, and `_verify_output_census` states that derived census exactly
rather than relaxing its comparison.

**Completion boundary — FP8 sources support plan build and fold measurement, not
materialization.** This is not a corner case: it is *every* liftable FP8 source, so no FP8
checkpoint can currently be materialized. Uplift leaves the output **mixed** — folded operands
BF16 with no scale, untouched operands still block-scaled FP8 — and two consumers read that mix
without a way to be told about it. (a) `config.json` declares one `quantization_config` for the
whole checkpoint, and the block-scaled FP8 schemas in play
(`{quant_method, fmt, weight_block_size, scale_fmt}`) carry no per-operand exclusion key, so
there is no attested way to name the uplifted operands (principle 14). (b) A profile that
overrides `fp8_scale_pairs` supplies the scale map from the *profile*, not the index
(`layer_streaming.py:301`), so dropping a scale from the index never reaches it — it would
dequant a BF16 tensor against a scale the shard no longer holds. Both are owed design work.
`_preflight_uplift_publishability` refuses at plan validation, **before any bytes are written**,
rather than at the downstream config/census gates that are only reached after the whole body has
been streamed — on a 284B FP8 source the late refusal costs hours and a full checkpoint of disk.

The widening covers the **dense** lane. `prismasnap_moe_checkpoint.py` still gates every
operand, router and norm on `dtype == "BF16"`, so an FP8 MoE source is refused at plan
discovery.

The candidate's nominal search is the measured-fast `stage,polish` algorithm, not Fable's
original sequential greedy implementation. `PrismaSnapSearchConfig` is versioned data and
production accepts only its canonical value: group size 16; alpha candidates
`[0,.125,.25,.375,.5]`; at most four fixed-global rounds; round-one staging of the top half of
predicted gains; then true-render polish over the top-eight-gain and near-maximum groups, capped
at 16 groups per seam. Each logical tensor keeps its own static-6 NVFP4 global, candidate folds
are rounded to fp32 for the nominal objective. A full true-render no-op comparison is the hard
upper bound. Fused-sibling globals, one-final-cast materialization, a different codec objective,
and the slower greedy search are different algorithms, not aliases for nominal v1
(`prismasnap.PrismaSnapSearchConfig.as_dict`, `search_diagonal_scale`,
`prismasnap_checkpoint.plan_dense_checkpoint`).

Before materialization, `realize-bf16` derives the executable v2 plan. Offset-one norm bytes are
projected once to BF16 and stored directly; consumer columns use a separate positive realized
inverse, unsafe channels execute identity, and the exact executed-BF16 static-6 objective must
strictly beat the original seam or the seam becomes identity. Up/down is identity. The v2
validator reconstructs every copied parent field and requires its canonical digest to equal the
bound merged-v1 plan digest; it separately hashes every nominal, executed, and projected vector
(`prismasnap_checkpoint.realize_bf16_plan`, `_validate_bf16_realized_plan_semantics`). Rejected
norm seams and all up/down seams omit identity transforms; changed-tensor coverage and v2 seam
summaries therefore report executed movement/objective gain while the parent records retain the
explicitly nominal search statistics.

The application lifecycle is
`PREPARED → PLANNED-v1 → REALIZED-v2 → MATERIALIZED → VERIFIED → COMMITTED`:

- **PREPARED** validates the original config/index/shard closure, complete tensor census and
  BF16 transform domain, profile, probe bytes and calibration/source binding, producer source
  closure, CUDA device, and attested container rootfs before a production plan may start.
- **PLANNED** is an atomic, self-digested plan plus float64 scale vectors. It binds three exact
  dense seams per claimed layer, the complete safetensors tensor metadata, transform order,
  search semantics, and the fp64 algebra gate (`≤1e-10`).
- **REALIZED** is the derived executable plan. It binds its complete merged-v1 parent and
  original source, preserves nominal statistics as nominal, stores direct BF16 norm payloads
  plus separate realized consumer inverses, records per-channel refusal reasons, and gates each
  norm seam on its exact executed-BF16 render objective. Production derives it on CUDA with
  `tools/prismasnap.py realize-bf16 --resume`.
- **MATERIALIZED** is the atomically published BF16 checkpoint. Per-shard receipts prove the
  source/output bytes, tensor shapes/dtypes, changed-tensor count, exact source-shard cover,
  checkpoint weight-map digest, index digest, and whole shard-content identity. It is explicitly
  not a numerical verdict.
- **VERIFIED** is the only state admitted downstream. `attest-fold-fidelity` replays the original
  source identity, teacher payload and calibration windows, BF16 teacher/student launch
  contracts, both serve fingerprints, all-position metric coherence, and current checkpoint
  bytes; original-BF16 → snapped-BF16 forward KL must be at most the attested threshold:
  the plan's `5e-4`, or `max(5e-4, 2.0 × measured null floor)` under a saturation-licensed
  `--null-floor-receipt` whose content hash is stamped into the verified provenance.
- **COMMITTED** is the fail-closed handoff of that verified source to the unchanged native
  pipeline/export transaction. `PLANNED`, `REALIZED`, `MATERIALIZED`, and `VERIFIED` are
  serialized receipt states; `PREPARED` and `COMMITTED` name validated transition boundaries rather than additional
  mutable JSON states (`prismasnap_checkpoint.py`, `prismasnap_validation.attest_fold_fidelity`).

Multi-Spark operation is deterministic application code, not an agent protocol.
`cluster_campaign.manifest.v2` declares local/SSH hosts, exact argv arrays, dependencies,
working directories, closed environments, expected receipt hashes, timeouts, retry bounds, and
maximum parallelism. Workers execute with `shell=False`; host-local locks, PID/start-time/owner
checks, fsynced self-hashed compare-and-swap state, sealed-stage receipts, and dependency barriers
make restart decisions mechanical and ambiguous ownership a refusal
(`cluster_campaign.run_campaign_v2`). For the two-Spark dense campaign, workers plan disjoint
layer sets, `merge-plans --resume` accepts only their exact layer union, workers materialize
disjoint original shard sets, and `merge-checkpoint-parts --resume` accepts only an exact
non-overlapping source-shard union. The remote part is transferred as an explicit
content-addressed campaign stage. `merge-checkpoint-parts` is the last writer of the tree the
two `materialize-part` workers staged, so the CLI runs it under the same `production` execution
gate they use (`_require_production_execution`: CUDA-typed device plus an attested container
rootfs); an unattested merge would otherwise launder both workers' receipts. The strict
low-space collation arm passes
`--require-hardlinks`: every admitted part shard must already be on the coordinator filesystem,
and the staging and committed trees must retain the same device/inode binding or fail. This
avoids another full-checkpoint copy; the general-purpose CLI's durable-copy mode is not the
27B low-space campaign contract (`prismasnap_checkpoint.merge_plans`,
`merge_checkpoint_parts`; `tools/prismasnap.py`).

Planning and transform materialization are CUDA-only in production and bind the attested Docker
rootfs plus producer commit/source bytes; source identity binds semantic config, sorted weight
map, every shard size/hash, and a portable content digest; the probe receipt binds its own bytes,
calibration hash/dataset/sample count/sequence length/modality, original model path, BF16 dtype,
streaming-GPU device map, and execution device. Plan construction streams only the tensors for
one layer and materialization streams one shard at a time. Part transfer and hardlink collation
are explicit offline checkpoint-lifecycle I/O, separate from the GPU-bound search/transform and
the ordinary GPU-bound quantization hot path (`prismasnap_checkpoint._require_production_execution`,
`_validate_source_identity`, `_validate_probe_source_contract`).

Admission is deliberately narrow. `run-pipeline.sh` validates any marker before work and then
admits it only when `EXPORT_CONTAINER=compressed-tensors`; the native exporter re-hashes the
verified BF16 input before opening its transaction and again before preserving the source receipt
as `source_prismasnap_provenance.json`. The pipeline GGUF branch and direct-GGUF API, plus both
Gridbook/codebook exporters, reject the mere presence of a PrismaSnap marker before output. The measured candidate
menu is therefore native `{NVFP4, FP8_DYNAMIC, BF16}` only
(`run-pipeline.sh:92-99`; `export_native_compressed.main`;
`prismasnap_contract.refuse_prismasnap_for_unvalidated_lane`).

Promotion remains ordered and evidence-gated. First is the Qwen3.8-27B text-only/no-vision,
strict decimal 20,000,000,000-byte arm against the existing unsnapped control, with the same
16×512 diverse-v1 probe, assignment/accounting semantics, production cache behavior, and gold
8×512 served protocol. The requested ≥20% all-position and confident-position KL reduction is
still a threshold, not a measured result (`docs/results/qwen38_prismasnap_20gb_ab_2026-08-25.md`,
status **RUNNING**). `plan_dense_checkpoint` currently refuses a non-dense profile. MoE is the
second experimental gate and must separately prove router/shared-gate compensation, packed
expert axes and per-expert importance, route stability, partial-source disk lifecycle, and the
released tensor layout. The Qwen3.5-family profile's native-causal versus wrapper source census
and explicit RMSNorm stored-parameter offset are prerequisites for that work, not evidence that
Qwen3.8-125B-A3B is admitted (`model_profiles/qwen3_5.py`,
`ModelProfile.rms_norm_parameter_offset`; `docs/design/prismasnap.md`).

**DIAGRAM-1 — Pipeline dataflow:** source checkpoint to three artifact containers, with the
four `COST_MODE`s, the opt-in validated-frontier loop, and the manual (echoed-only) ship gate.

```mermaid
flowchart TD
  ORIG["original BF16 source checkpoint<br/>HF safetensors"]
  SRC["ordinary pipeline source<br/>unmarked BF16, or COMMITTED verified PrismaSnap BF16"]
  ORIG -->|"no PrismaSnap marker: historical path"| SRC

  subgraph PSNAP["optional additive PrismaSnap source preparation -- before [1/4]"]
    PSPREP["PREPARED<br/>source config/index/shards + full tensor headers<br/>probe/calibration + producer/container identity"]
    PSPLAN0["Sparky plan-dense<br/>disjoint layers; CUDA measured-fast stage,polish"]
    PSPLAN1["Sparklina plan-dense<br/>disjoint layers; CUDA measured-fast stage,polish"]
    PSMERGE["PLANNED-v1<br/>merge-plans --resume<br/>exact layer union + 3 seams/layer"]
    PSREAL["REALIZED-v2<br/>realize-bf16 --resume<br/>direct BF16 norms + realized inverses<br/>executed objective / identity fallback"]
    PSPART0["Sparky materialize-part<br/>disjoint original shards"]
    PSPART1["Sparklina materialize-part<br/>disjoint original shards"]
    PSTRANSFER["content-verified remote-part transfer<br/>explicit campaign stage"]
    PSCOLLATE["MATERIALIZED<br/>merge-checkpoint-parts --resume --require-hardlinks<br/>exact shard union; same-filesystem/inode proof"]
    PSKL["served original-BF16 → snapped-BF16<br/>all-position fold KL ≤ max(5e-4, 2× null floor)"]
    PSVER["VERIFIED<br/>source/calibration/serve/index/shard replay"]
    PSCOMMIT["COMMITTED<br/>atomic native-pipeline handoff"]
    PSPREP --> PSPLAN0
    PSPREP --> PSPLAN1
    PSPLAN0 --> PSMERGE
    PSPLAN1 --> PSMERGE
    PSMERGE --> PSREAL
    PSREAL --> PSPART0
    PSREAL --> PSPART1
    PSPART0 --> PSCOLLATE
    PSPART1 --> PSTRANSFER --> PSCOLLATE
    PSCOLLATE --> PSKL --> PSVER --> PSCOMMIT
  end

  ORIG -. "candidate pre-pass" .-> PSPREP
  PSCOMMIT --> SRC
  PROBE["[1/4] incremental_probe -- run-pipeline.sh:544-560<br/>per-Linear empirical Fisher h_trace<br/>artifacts/probe.pkl"]
  ACT["activation cache<br/>WORK_DIR/act"]
  CBL["learned-CB pre-render gate (scope fp8)<br/>immutable value-bearing CB_CODEBOOK_BUNDLE<br/>trained once before cost/cache/KL/export"]
  BASE["[2/4] incremental_measure_quant_cost -- :645-658<br/>RTN per-Linear-per-format error<br/>cost.pkl (local) or cost_baseline.pkl"]

  subgraph SPROBE["optional exact sample-axis probe map/reduce"]
    SPCAL["immutable ids[N,T] + closed source/run contract"]
    SPCE["N stage-1 workers<br/>raw shifted-token CE sum/count"]
    SPBAR["global CE barrier<br/>exact cover + one global mean"]
    SPWORK["N full-model stage-2 workers<br/>complete qname census per sample stripe"]
    SPMERGE["strict raw-stat + activation merger<br/>ordinary probe.pkl + act/"]
    SPCAL --> SPCE --> SPBAR --> SPWORK --> SPMERGE
  end

  SRC --> PROBE
  PROBE --> ACT
  SRC -. "source/config/header/content identity" .-> SPCAL
  SPMERGE --> ACT
  SPMERGE --> BASE
  SRC --> CBL
  ACT --> CBL
  PROBE --> BASE
  ACT --> BASE
  CBL --> BASE

  subgraph COST["cost stage -- one of three COST_MODEs, dispatched in the COST_MODE case"]
    PRS["production-render-score -- explicit/legacy<br/>build_production_cache --render-scope format-menu<br/>CB streaming: render -> acknowledged score checkpoint -> discard<br/>then production_render_cost -> cost.pkl"]
    LOC["local<br/>the RTN base cost IS the allocator cost<br/>the CB/GGUF lanes shipping recipe"]
    AUR["aura -- DEFAULT since 2026-07-30<br/>aura_cost excludes profile-declared routed experts -> cost_aura.pkl<br/>then expert_empirical_cost --merge-base -> cost.pkl<br/>then the [3c] additivity report"]
    CBH["CB sub-stage (:966-1035)<br/>cb_col_weights.pkl imatrix harvest, then<br/>expert_empirical_cost --replace-experts"]
  end

  subgraph DSVA["platform-agnostic anchored-cost mechanism -- DSv4 is the CB acceptance driver"]
    DAP0["evaluate<br/>format-blind streamed checkpointed KL-adjoint -> gW_i<br/>global Fisher; profile-declared routed experts"]
    DAMAP["map plugin<br/>format_registry family + model-profile role<br/>ladder/rate, transfer-equivalence partition,<br/>renderer + anchor policy + provenance"]
    DAP1["price<br/>one production anchor per legal unit/segment<br/>render -> fp32 AURA scalar -> discard<br/>within-equivalence fit + hull + exposure"]
    DAP3["allocate<br/>one exact-byte DP under the driver budget<br/>no iteration; blind export assignment"]
    DAART["driver-specific exportable artifacts<br/>DSv4 CB: layer_config.json + selection.json<br/>+ pareto.knees.json + render-input cb_col_weights.pkl"]
    DAP0 --> DAP1 --> DAP3 --> DAART
    DAMAP --> DAP1
  end

  BASE --> PRS
  BASE --> LOC
  BASE --> AUR
  LOC --> CBH
  SRC --> DAP0
  ACT --> DAP0
  CBL --> DAMAP

  ALLOC["[3/4] allocator + allocator_solver -- :1076-1090<br/>multi-choice knapsack DP over Linear x format<br/>union-find serving-unit promotion<br/>artifacts/layer_config.json + pareto.csv"]

  PRS --> ALLOC
  LOC --> ALLOC
  AUR --> ALLOC
  CBH --> ALLOC

  subgraph VS["SELECTION_MODE=validated-surrogate -- OPT-IN; default is surrogate (:250)"]
    FR["A. build_production_cache --render-packed-experts<br/>production_weight_cache_frontier_raw.pkl"]
    VAK["B. validate_assignments_kl -- :1243-1277<br/>measured held-out KL per Pareto point<br/>validated_frontier_kl.json"]
    SVF["C. select_validated_frontier -- :1281-1288<br/>kneedle -> rewrites layer_config.json"]
  end

  ALLOC --> FR
  FR --> VAK
  VAK --> SVF

  PCACHE["[4/4] D. build_production_cache / production_recache<br/>ProductionWeightCache -- the one rendered-weight store<br/>body + profile-synthesized MTP; levers: gptq, static_act_order, joint_scale_opt"]

  subgraph XUNION["optional exact multi-host cache build -- native materialized renders only"]
    SPLAN["production_cache_stripes<br/>probe-shaped LPT over whole decoder layers<br/>plus indivisible lm_head / MTP groups"]
    SCACHE["independent build_production_cache workers<br/>one --include-qnames-file and cache bundle per stripe"]
    SUNION["union_production_cache<br/>manifest -> verify-shard -> union -> verify<br/>exact disjoint set union; no candidate selection"]
    SPLAN --> SCACHE --> SUNION
  end

  SVF --> PCACHE
  ALLOC -->|"SELECTION_MODE=surrogate"| PCACHE
  PROBE -. "qname shapes" .-> SPLAN
  ALLOC -. "assignment coverage" .-> SCACHE
  SRC -. "same checkpoint identity" .-> SCACHE
  ACT -. "MTP activation rows" .-> PCACHE
  ACT -. "same calibration rows" .-> SCACHE
  CBL -. "exact learned tensors" .-> PCACHE

  EXPCT["export_native_compressed -- :1665-1699"]
  EXPCB["export_nvfp4_cb or export_nvfp4_cb_streaming<br/>auto-switch above 80 GB source (:1585-1641)<br/>optional read → ordered encode → bounded ordered write"]
  EXPGG["convert_hf_to_gguf.py skeleton -> export_gguf<br/>(:1461-1493)"]

  PCACHE --> EXPCT
  SUNION --> EXPCT
  ALLOC -->|"EXPORT_CONTAINER=nvfp4_cb, PRODUCTION_CACHE=0"| EXPCB
  ALLOC -->|"EXPORT_CONTAINER=gguf, PRODUCTION_CACHE=0"| EXPGG
  CBL -. "same tensors; no retraining" .-> VAK
  CBL -. "same tensors; emit once" .-> EXPCB
  DAART -. "selected assignment; same render inputs" .-> EXPCB

  OUTCT["compressed-tensors checkpoint<br/>WORK_DIR/exported"]
  OUTCB["CB checkpoint + quant_config.json + cb_codebooks.pqcb<br/>WORK_DIR/exported_nvfp4_cb"]
  OUTGG["single-file GGUF<br/>WORK_DIR/exported.gguf"]

  EXPCT --> OUTCT
  EXPCB --> OUTCB
  EXPGG --> OUTGG

  GGSMOKE["llama-completion greedy smoke<br/>in-lane, :1500-1516"]
  NOSMOKE["no in-lane serving smoke<br/>gate set declared in lane_specs/nvfp4_cb.json (R16)"]
  OUTGG --> GGSMOKE
  OUTCB --> NOSMOKE

  subgraph GATE["ship gate -- NOT executed by the pipeline"]
    VNE["validate_native_export<br/>vLLM eager+graph load + greedy smoke<br/>echoed at :1704-1705"]
    VQM["validate_quantized_model<br/>PPL 25 / mean-NLL 3 / worst-NLL 6 / MTP p0 0.60<br/>+ sampled boundary 0 defects (5 prompts x 6 reps, temp 1.0)<br/>validate_quantized_model.py -- never echoed"]
    GOLD["gold lane, invoked by hand<br/>measure_vllm_full_kl.py -- n=8 x 512<br/>DSv4 all-position topK-1024 + tail KL<br/>measure_vllm_wikitext_ppl.py -- 8192-token PPL"]
  end

  OUTCT --> VNE
  VNE --> VQM
  VQM --> GOLD
  NOSMOKE --> VQM
  GGSMOKE --> GOLD

  classDef optin stroke:#c07800,stroke-width:2px,stroke-dasharray:4
  classDef manual stroke:#c0392b,stroke-width:2px
  class PSPREP,PSPLAN0,PSPLAN1,PSMERGE,PSPART0,PSPART1,PSTRANSFER,PSCOLLATE,PSKL,PSVER,PSCOMMIT,AUR,CBH,FR,VAK,SVF,DAP0,DAP1,DAP3,DAART,SPLAN,SCACHE,SUNION optin
  class VNE,VQM,GOLD,NOSMOKE manual
```

### 3.1 Pre-flight gates

In current order, all failing closed: required `MODEL_PATH`/`WORK_DIR` (`48-49`); optional
PrismaSnap marker replay plus native-only admission (`93-99`); architecture lane and
serving-profile resolution (`357-388`); GGUF and CB lane consistency (`508-551`);
GPU-or-bust — both `DEVICE` and `EXPORT_DEVICE` must match `cuda*` and inline Python asserts
`torch.cuda.is_available()` (`554-565`); the archived-lever gates of §3.5 (`737-1077`);
`COST_MODE` dispatch and unknown-mode refusal (`391-507`, `972-1013`); work-dir creation
(`1081`); `SELECTION_MODE` legality (`1083-1089`); and declarative spec write/validate
(`1128-1147`). The PrismaSnap check is deliberately before profile resolution, GPU setup,
work-dir creation, cache work, or export: a corrupt, merely `MATERIALIZED`, or non-native marked
source cannot consume an expensive stage before refusal
(`prismasnap_contract.require_verified_prismasnap_if_present`).

The two lane gates encode one contract, and since re-vet **R3** they say so directly (§4.7):
the GGUF and CB exporters requantize the bf16 skeleton with **imatrix-weighted** renders, so
the render that produces the allocator's cost must be imatrix-weighted too — a
**render-faithfulness assertion**, keyed off the format family, not a `COST_MODE` whitelist.
`PRODUCTION_CACHE=0` + the matching `TARGET_PROFILE` remain mandatory for their own reasons
(those exporters never read the export cache; the exporter hard-fails on out-of-profile
formats). The lanes themselves are §9.

### 3.2 Stage table (execution order)

Line refs are `run-pipeline.sh` unless stated. Artifact paths are relative to `$WORK_DIR`.
**The line numbers in this table are indicative, not a contract** — the four 2026-07-30 re-vet
waves moved them repeatedly, and §3.3 already dropped them for exactly that reason. The
non-decaying anchor is the bracketed **stage label the script echoes** (`grep '\[2d-CB\]'`),
which is what the rows touched since are keyed on.

| # | Stage | Script | Artifact(s) | Reuse guard | Mode/lane gate |
|---|---|---|---|---|---|
| **PREPARED → PLANNED (optional, pre-pipeline; ×2)** | Validate the original BF16/source/probe/producer/container closure, discover exactly three dense seams per claimed layer, run the canonical CUDA `stage,polish` search, then exact-union the two disjoint layer plans | `tools/prismasnap.py plan-dense --resume`; `merge-plans --resume`; autonomous dependency/retry/barrier execution by `tools/run_cluster_campaign.py` / `prismaquant.cluster_campaign` | one self-digested plan directory per Spark; merged `plan.json` + `scales.safetensors`; campaign manifest/state/logs/sealed receipts | exact argv and producer closure; full source identity + tensor-header census; probe bytes/calibration/execution identity; canonical search; host locks and receipt hashes; merged layer cover must be disjoint, complete, and carry exactly three semantically valid seams per layer | opt-in candidate source-prep only; production requires CUDA and attested container; current planner is dense-only and excludes vision/MTP seams |
| **PLANNED → MATERIALIZED (optional, pre-pipeline; ×2)** | Stream disjoint original shard sets through the merged transform program, transfer the remote part by explicit receipt, then exact-union on the coordinator. The strict 27B low-space arm requires hardlinks and proves device/inode identity in staging and after commit | `tools/prismasnap.py materialize-part --resume`; `merge-checkpoint-parts --resume --require-hardlinks` | two `part.json` directories with per-shard receipts; atomically committed BF16 HF checkpoint + `prismasnap_provenance.json` state `MATERIALIZED` | plan/source/header identity; exact disjoint source-shard cover; source/output hash, size, tensor shape/dtype and transform census per shard; same-filesystem hardlink proof; producer/container attestation on the merge as well as on each part; fsynced no-clobber staging and committed-tree replay | still outside `run-pipeline.sh`; `MATERIALIZED` is not downstream admission |
| **MATERIALIZED → VERIFIED → COMMITTED (optional, pre-pipeline)** | Serve snapped BF16 against the original BF16 teacher, attest all-position fold KL, atomically replace provenance with the verified receipt, then hand the ordinary checkpoint to the unchanged native pipeline | `tools/measure_vllm_full_kl.py`; `tools/prismasnap.py attest-fold-fidelity`; admission by `python -m prismaquant.prismasnap_contract --model "$MODEL_PATH"` | student result + teacher metadata/payload; `prismasnap_provenance.json` state `VERIFIED`; native output copy `source_prismasnap_provenance.json` | BF16/no-quantization/no-spec-decode launch contracts; teacher/source/calibration windows and corpus; both serve fingerprints; coherent metrics; current index/weight-map/every-shard identity; forward KL `≤5e-4` | native compressed-tensors `{NVFP4, FP8_DYNAMIC, BF16}` only; GGUF and Gridbook/codebook refuse any marker; an absent marker is the historical no-op path |
| **1/4** | Sensitivity probe — per-Linear empirical Fisher `h_trace`, body + MTP in one pass; tied heads materialized and excluded, KV-sharing cotangents grafted (§7.5) | `prismaquant.incremental_probe` (`544-560`) | `artifacts/probe.pkl`; activations → `act/`; shards → `work/`; `logs/probe.log` | settings-hash `probe` (`703`); reuse also re-checks stored `calibration_modality` | — |
| **1/4×N (manual)** | Exact sample-axis probe map/reduce. Every worker processes the complete dense text qname census over one canonical contiguous sample partition. Stage 1 publishes raw shifted-token CE; a global scalar barrier closes the cover; stage 2 reruns phase 1 with that global mean, then the strict reducer finalizes raw Fisher/marginals once and publishes the deterministic dense-body activation union (§4.1) | `prismaquant.sample_parallel_probe` + `prismaquant.incremental_probe` + `prismaquant.sample_parallel_probe_merge` | immutable calibration/run contract and digest-bound sample cover; per-worker CE/probe/cache shards; global CE receipt; ordinary merged `probe.pkl` + `act/` | committed read-only source snapshot + host-verified immutable registry RepoDigest; source/config/header/content identity; exact sample cover replayed before GPU setup; duplicate-key-free trusted JSON; marginal-to-trace and independently replayed global top-R/fused-row checks; no-follow same-byte probe and committed-identity lazy activation consumption; no-clobber output | opt-in operator workflow; no `torch.distributed`, qname/layer partitioning, alternate cache, serving-runtime change, h-detail, visual, routed, or packed path |
| **2/4** | Baseline per-(Linear,format) RTN cost. The measured menu is derived `COST_FORMATS`; `lm_head` is included only for a fixed non-BF16 or DP-unpinned head | `prismaquant.incremental_measure_quant_cost --[no-]include-lm-head` | `artifacts/cost.pkl` (`COST_MODE=local`) or `artifacts/cost_baseline.pkl`; `logs/cost.log` | settings-hash `base-cost`, including all three resolved head-policy axes and `COST_FORMATS` | — |
| **2a-CB** | imatrix column-weight harvest | `harvest_cb_col_weights` — ONE shell function, four call sites (`[2/4] pre-cost`, `[2b/4] cost-cache`, `[2d-CB]`, `[4/4]`) → `export_gguf.build_imatrix_from_act_cache` + `moe_imatrix.synthesize_packed_expert_col_weights` | `artifacts/cb_col_weights.pkl` | settings-hash `cb-col-weights` | CB lane; called by whichever stage needs the vector first |
| **pre-2-CBL** | Train/verify the immutable value-bearing codebook bundle. Historical trainer v1 keeps its measured rung policy. Trainer v2 derives the existing probe imatrix and emits lattice for every FP8 producer rung unless an independently validated per-rung promotion receipt authorizes the exact learned candidate | `ensure_cb_learned_bundle` → `prismaquant.build_cb_learned_bundle` → streaming source reader; v2 additionally uses `cb_imatrix` + `cb_learned_promotion` | `artifacts/cb_learned_bundle.pqcb`; optional external `prismaquant.fp8_cbl_promotion_receipt.v1` input is embedded and digest-bound | settings-hash `cb-learned-bundle` includes the calibration/imatrix input identity; bundle load revalidates complete source, role, candidate-table, and receipt bindings | CB lane, learned scope only; runs before the first cost/cache/KL render. V2 without a valid receipt is all-lattice, never a guessed learned/lattice crossover |
| **2b/4** | Format-menu production render for allocator cost. Materialized mode retains render shards; streamed CB mode synchronously renders each full-menu pair, checkpoints the consumer acknowledgement, then discards the tensor (§5.4). The probe activation directory enables profile-synthesized MTP append; remaining profile pins are skipped | `build_production_cache --render-scope format-menu --activation-cache-dir act/` using derived `COST_FORMATS`; transient lifetime is implemented by `streaming_production_cache.py` through the existing `ProductionWeightCache` | `artifacts/production_render_score_cache.pkl`; `…_weight_cache/` contains tensors only for materialized mode, while transient CB pairs retain identity/digest/consumer sidecars but no rendered-weight shard | settings-hash `render-cost-cache`, including the resolved head policy | `production-render-score`; transient mode is CB-only and must cover the complete requested menu |
| **2c/4** | Synthesize allocator cost from render scores | `prismaquant.production_render_cost` (`704-711`) | `artifacts/cost.pkl` | settings-hash `render-cost` (`858`) + cost-mode provenance (`859`) | `production-render-score` |
| **2b/4** | Format-menu render for AURA dW. A materialized cache exposes dW later; the streamed CB lifetime exposes each canonical render to the synchronous cost consumer and discards it only after that row is durably acknowledged (§5.4). It uses the same activation-cache/MTP and remaining-pin policy as the cost cache | `build_production_cache … --render-scope format-menu --activation-cache-dir act/` | frontier cache under validated-surrogate, else `production_render_score_cache.pkl`; transient CB mode retains pair attestations rather than loser weight shards | settings-hash `aura-dw-cache`, including the resolved head policy | `aura`; `exit 2` if the menu is BF16-only; every requested candidate must be consumed |
| **2c/4** | AURA downstream-KL-adjoint cost. A fixed head remains auxiliary and is omitted; only `ALLOW_PINNED=lm_head` asks AURA to price the head inside the research DP | `prismaquant.aura_cost` with `--include-lm-head` only in DP-unpinned mode | `artifacts/cost_aura.pkl` | settings-hash `aura-cost`, including the resolved head policy | `aura` |
| **2d/4** | Hybrid finalize: empirical profile-declared routed-expert unit-KL + sidecar backfill | `prismaquant.expert_empirical_cost --merge-base --backfill-base` (with the shared `--col-weights` on weighted cached-menu lanes) or inline backfill (`run-pipeline.sh`, AURA `[2d]`) | `artifacts/cost.pkl` | settings-hash `aura-hybrid-cost` + cost-mode provenance | `aura` |
| ~~**2d-CB**~~ (RETIRED 2026-09-02 with the lane, §9.2) | CB hybrid: replace routed-expert rows with empirical unit-KL | `harvest_cb_col_weights "[2d-CB]"` → `expert_empirical_cost --replace-experts --col-weights` | `artifacts/cost_local_raw.pkl`, `artifacts/cost.pkl`, `cb_col_weights.pkl` | settings-hash `cb-hybrid-cost` + the in-payload merge probe; col-weights `cb-col-weights` | CB lane, `CB_EXPERT_EMPIRICAL=1` |
| **2b/4 cw** | Cost-cache col-weights (weighted lanes only) | `harvest_cb_col_weights "[2b/4] cost-cache"` → `build_production_cache --col-weights` | `artifacts/cb_col_weights.pkl` | settings-hash `cb-col-weights` | `COST_RENDER=cached-menu` on a CB/GGUF lane (§4.7) |
| **P0–P3** | Platform-agnostic anchored-AURA mechanism: format-blind streamed adjoint; plugin-mapped production anchors per legal `(unit,family,equivalence_class)`; within-segment shape fit; recomputed hull; one byte-budget DP (§4.3) | frozen DSv4 shim `tools/run_aura_cb_reprice.sh` → `prismaquant.dsv4_aura_cb_reprice`; generic mechanism `prismaquant.anchored_cost`; CB mapping plugin on `format_registry` + `model_profiles` | identity-bound scalar checkpoints; the driver atomically emits an exportable artifacts directory containing the AURA-stamped `layer_config.json`, matching `selection.json`, allocator `pareto.knees.json`, and the exact platform render inputs (`cb_col_weights.pkl` on CB) | qname-keyed atomic resume bound to model/menu/arm/plugin/calibration/format-plan/render-input identity | generic evaluate/price/allocate mechanism with a machine-specific map plugin; DSv4 remains the acceptance vehicle; never a full-menu render campaign |
| **RTX4090 sparse P1–P3 ×2 (manual)** | Strict 18 GB specialization of anchored AURA: derive the exact bundle-captured probe imatrix, attest one snapshot/image execution, split whole layers across two hosts, render only lattice K4/K16/K48 plus native FP8 per body Linear, reconstruct one exact global payload, fit the lattice rung law, byte-preserve direct rows, impute nine CB rungs, add the BF16 source terminal, and solve once (§4.3) | `prismaquant.rtx4090_fp8_burn` + `prismaquant.cb_anchored_cost` | durable no-clobber FP32 column weights; immutable execution attestation and two-stripe plan; 496×4 measured AURA cells; one mixed measured/imputed/source-terminal allocator table | complete disjoint qname/format/purpose/renderer/source/imatrix/snapshot/image/arm identity; worker chunk counters remain explicitly unverified planning metadata | Qwen3.8 strict FP8-only validation-artifact campaign; no NVFP4 and no twelve-rung physical render sweep |
| **GLM stock anchored (manual)** | Stock-vLLM single-rung specialization of anchored AURA for a non-CB menu: one batched streamed KL-adjoint over the dense plan (135 units × NVFP4, production render gptq+static_act_order+JSO), priced/merged CPU-side with measured packed-expert unit-KL and exact source terminals. Unblocked 2026-08-27: the streamed checkpoint guard accepts the production-anchor renderer identity as the value-bearing render identity on a CB-free menu (`tests/test_streamed_cost_checkpoints.py`), and the anchored renderer's transient per-layer render makes the retained format-menu dW cache moot | `tools/run_glm53_stock_harvest.sh` → `prismaquant.glm53_stock_harvest` (GPU) → `prismaquant.glm53_stock_reprice campaign` (CPU) with the `prismaquant.stock_anchored_cost` map plugin | harvest wrapper pkl (arm identity + exact plan + AURA payload) → `cost_stock_anchored.pkl` (135 anchored + 84 empirical + 69 pinned = 288 units; packed-expert FP8_SOURCE terminals refused `profile_mismatch` and recorded as serving-gap findings) | per-unit identity-bound AURA shards; arm identity binds render levers + costed format + probe calibration (a CB-free renderer identity binds no levers by itself); campaign refuses filtered harvests, plan drift, RTN-fallback rows, foreign calibration | GLM-5.3-Flash teacher-artifact campaign; single costed rung per unit so no shape transfer is expressible; routed experts stay on the empirical unit-KL currency |
| **3/4** | Allocator — multi-choice knapsack over per-Linear body formats plus explicit fixed auxiliary head/MTP/visual assignments (§4) | `prismaquant.allocator --lm-head-format "$LM_HEAD_FORMAT_CANONICAL"` | `artifacts/layer_config.json`, `artifacts/pareto.csv`, `artifacts/pareto_assignments/` (validated-surrogate only); `logs/allocator.log` | **none — always runs** | fixed non-BF16 head and `ALLOW_PINNED=lm_head` are mutually exclusive |
| **4/4 A** | Frontier format-menu cache, including profile-synthesized MTP when the menu requests it | `build_production_cache … --render-scope format-menu --render-packed-experts --activation-cache-dir act/` | `artifacts/production_weight_cache_frontier_raw.pkl` + `…_frontier/` | settings-hash `frontier-cache`, including the resolved head policy | validated-surrogate; `exit 2` if `PRODUCTION_CACHE=0` |
| **4/4 B** | Measured held-out KL per Pareto point | `prismaquant.validate_assignments_kl` (`1243-1248` per-point, `1272-1277` batched) | `artifacts/validated_frontier_kl.json` + `…_parts/*.json` (merged `1250-1269`) | settings-hash `frontier-kl-point` per point (`1294`) | validated-surrogate |
| **4/4 C** | Frontier point selection | `prismaquant.select_validated_frontier` (`1281-1288`) | overwrites `artifacts/layer_config.json`; `layer_config_validated_assignment.json`; `validated_frontier_selection.json` | none | validated-surrogate |
| **4/4 D** | Production cache build / recache for the selected assignment, including exact non-BF16 `mtp.*` renders synthesized through the profile from `act/` | `production_recache` or `build_production_cache --recache-layer-config --activation-cache-dir act/`; all direct builder calls receive the remaining profile pins | `production_weight_cache_frontier_<digest>_recached.pkl`, `production_weight_cache_recached.pkl` / `…_raw.pkl` and their shard directories | settings-hash `production-cache-recached`, `frontier-recache`, `production-cache-raw`, including the resolved head policy | `PRODUCTION_CACHE=1`; explicit quantized MTP fails closed without exact source/module/activation coverage |
| **4/4 D×N (manual)** | Exact multi-host cache striping and set union. Plan whole-layer/auxiliary qname groups, run independent allowlisted cache builds, manifest and verify each portable bundle, union, then verify again after transfer | `prismaquant.production_cache_stripes`; `build_production_cache --include-qnames-file`; `prismaquant.union_production_cache {manifest,verify-shard,union,verify}` | `stripe-plan.json`, `stripe-NN.qnames.txt`, per-worker cache bundles, final `production_weight_cache.pkl` + content-addressed `weights/` + `union_manifest.json` | campaign identity binds source, calibration, full producer code, settings, render semantics, and assignment or stripe-plan coverage | operator workflow, not a `run-pipeline.sh` default; native materialized formats only (§5.4) |
| **3c** | AURA additivity report — `residual = measured_end_KL − Σ predicted_dloss`, stamped into `cost.pkl` `provenance["additivity"]` (§4.3) | `prismaquant.aura_additivity_gate` (+ optional `validate_assignments_kl` under `AURA_ADDITIVITY_GATE=measure`) | `artifacts/aura_additivity.json`, `aura_additivity_kl.json` (measure only); `logs/aura_additivity*.log` | none — non-blocking report, skip-if-exists on the KL half | `COST_MODE=aura`, `AURA_ADDITIVITY_GATE≠0` |
| **3d** | Discovery-walker export gate (§8.8) — meta-load + one fake forward against the profile's claim rules; refuses the run before ANY export lane on an unclaimed matmul-fed node / unresolved floating multiplicand / unknown walk-failure kind / decided-but-unpriced contradiction, decided from structured fields only. Override env excuses trace incompleteness ONLY (stamped); claims have no override | `prismaquant.model_walk` (`python3 -m prismaquant.model_walk`) | `artifacts/model_walk.json`; `logs/model_walk.log` | none — always runs; it is the gate | every lane; knobs `WALK_GATE_SEQLEN=8`, `WALK_GATE_EXECUTION=fake`, `PRISMAQUANT_WALK_GATE_OVERRIDE=<reason>` |
| **4/4 E-gguf** | GGUF skeleton + export + llama.cpp smoke | `convert_hf_to_gguf.py` (`1461-1464`), `prismaquant.export_gguf` (`1469-1493`), `llama-completion` (`1500-1516`) | `artifacts/skeleton.gguf`, `exported.gguf` | settings-hash `gguf-skeleton` (`1488`); export always runs | GGUF lane; **exits 0** |
| ~~**4/4 E-cb**~~ (RETIRED 2026-09-02 with the lane, §9.2) | CB col-weights + codebook export | `harvest_cb_col_weights "[4/4]"`, `export_nvfp4_cb[_streaming]` | `exported_nvfp4_cb/` in **~1 GiB safetensors shards** (`EXPORT_SHARD_BYTES`, default `1073741824`) + `.pqcb` codebook sidecar | settings-hash `cb-col-weights`; export always runs | CB lane; no in-lane serving smoke; **exits 0** |
| **RTX4090 strict build (operator campaign)** | Run the ordinary per-Linear pipeline under the dense Qwen3.8 context-first policy, prove the exact 1,199-tensor/615-Linear source census, then replay config ownership and finalized artifact headers/sidecar before publication | `scripts/run_qwen38_rtx4090_fp8_cb_18gb.sh` → `run-pipeline.sh` + `rtx4090_qwen38_policy` + `rtx4090_artifact_census` | strict top-level `format: fp8_cb` lattice artifact, exact policy/runtime-contract/source-identity stamps, complete tensor-format assignment, and strict `rtx4090.fp8_cb` shipcard slot | no special cache: streamed AURA, the existing activation cache, probe-derived full-corpus imatrix, `ProductionWeightCache`, prefetch, allocator, and CB exporter remain authoritative and identity-bound | opt-in `TARGET_PROFILE=qwen38_rtx4090_fp8_cb`; exact menu FP8-CB K4..K48 step 4 + `FP8_E4M3` + BF16, `| **RTX4090 validation-only GB10 build** | Exercise the exact strict serializer, source/artifact census, top-level `fp8_cb` manifest, and complete FP8-only assignment under Gridbook v11 compile-only SM89 structural cells | `scripts/run_qwen38_rtx4090_fp8_cb_validation_only_gb10.sh` → the same launcher/pipeline with `RTX4090_BUILD_DISPOSITION=validation_only` → `validate_rtx4090_fp8_cb_validation_only` | artifact stamped `UNRELEASABLE_VALIDATION_ONLY`, `runtime_qualification_ceiling=compile_only`, `build_host=dgx_spark_gb10`; no serving receipt | exactly the strict build's streamed AURA, activation cache, imatrix, `ProductionWeightCache`, required resident prefetch, allocator, and CB exporter; no parallel cache | launcher requires one visible GB10 at CC 12.1; assignments/config groups/tensor formats admit only FP8-CB K4..K48 step 4, delegated `FP8_E4M3`, and BF16. Shipcard, publisher, and strict RTX4090 validator always refuse it regardless of filled slots or force flags |
| **RTX4090 validation-only direct export** | Consume a completed allocator assignment plus its exact value-bound column weights and original source; rerender only selected FP8-CB units, copy required source/BF16 tensors, then structurally validate | `scripts/export_qwen38_rtx4090_fp8_cb_validation_only.sh` → `prismaquant.rtx4090_validation_export` → existing `export_nvfp4_cb_streaming` → `validate_rtx4090_fp8_cb_validation_only` | the same top-level `fp8_cb`, `UNRELEASABLE_VALIDATION_ONLY` artifact; no retained-menu cache or serving receipt | no cache build: exporter streams the source and invokes the existing weighted renderer only for the selected assignment; `cb_render_identity` binds the exact supplied col-weights values and complete decoded-source identity | requires the allocator's assignment-bound whole-artifact budget stamp at a positive ceiling no greater than 18,000,000,000 bytes; refuses source namespace exclusions and every non-policy/NVFP4 format. It never calls `run-pipeline.sh`, and release/graph gates remain unchanged |
| **4/4 E** | compressed-tensors export (§6) | `prismaquant.export_native_compressed` (`1665-1699`) | `exported/` in **~1 GiB safetensors shards** (`EXPORT_SHARD_BYTES`, default `1073741824`); `logs/export.log` | **none — always runs** | default lane |

The strict launcher keeps validated-surrogate selection on the production
render path (`PRODUCTION_CACHE=1`, required resident prefetch) and deliberately
sets `PRODUCTION_RECACHE=0`: the CB serializer re-encodes the selected
assignment with the same bound codec and cannot consume a selected-assignment
recache. `AURA_COST_STREAMING=1` requires one absolute checkpoint directory,
and `PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE` points into that directory so
the AURA resume identity is also the source-census identity. `CB_IMATRIX_SOURCE=probe`
makes cost, frontier KL, and export use one digest-stamped full-corpus imatrix
rather than a row-capped replay estimate. The official conditional-generation
wrapper is accepted only after its nested dense text identity, exact schedule,
1,199 source tensors, and 615 Linears validate; established text-only staging
then presents the flattened causal-LM body to the ordinary pipeline. The
strict serializer accepts only canonical lattice codebooks until a future
artifact contract can bind the raw learned-v2 promotion ledger, imatrix, and
source closure. Publication replays the strict slot after freezing the upload
tree; neither an invalid/unfilled slot nor frozen replay failure can be waived
or stamped with the generic force flags.

**Output packaging — ~1 GiB shards (2026-08-20; CB lane 2026-08-21).** Published artifacts are
packaged in ~1 GiB safetensors shards, down from 5 GiB, on Robert's standing instruction
("package all models using 1gb file sizes"). The value lives in
`prismaquant/shard_layout.py` (`DEFAULT_SHARD_BYTES`), which every lane's `--shard-bytes`
default reads (so a hand-run export gets it too), and in `run-pipeline.sh`'s
`EXPORT_SHARD_BYTES`, passed through on **both** the compressed-tensors branch and the CB
branch and echoed in the settings banner. The partition rule is one implementation
(`shard_layout.plan_shards`): accumulate in emit order, close the current shard when the next
tensor would push it past the budget, and give a single oversized tensor its own shard whole
rather than splitting it — so a wide `lm_head`/`embed_tokens` row may still exceed the target.
That is the writer's contract, not a violation of it. **There is no zero sentinel** in either
lane: the legacy single-container layout is what a budget at least as large as the finished
artifact already produces, and the writers publish `model.safetensors` with no index whenever
exactly one shard results.

**Scope now covers the CB lane.** `export_nvfp4_cb` (resident) and
`export_nvfp4_cb_streaming` (the 200-300B-class path) both take `--shard-bytes`; the streaming
writer partitions its recorded entry sequence up front, so every shard's name and header are
known before a byte is written, streams each container to a temporary file, runs the
source-coverage `before_publish` assertion once after the last producer, and only then renames
every container and writes the index — an abort at any point still leaves no partial artifact.
The CB refusal contract follows the layout rather than being told it: a single container must
carry no index and a `model-XXXXX-of-YYYYY` run must carry one naming exactly the planned
containers (`nvfp4_cb_footprint.cb_export_artifact_inventory`), and the shipcard's
`weight_content_manifest` lists every published shard with the digest the writer accumulated
in-stream (no second read of a 100 GB container). Motivation: the shipped 87 GB single-container
DSv4 CB artifact stalled the default HF loader on a 128 GB unified-memory GB10 and the reporter
resharded it by hand (RobTand/gridbook#47); `scripts/reshard_safetensors.py` is that repair
after the fact. **Serving evidence has that scope and no more** — the CB lane serves only in the
pinned Gridbook runtime, which this repo may not import (`AGENTS.md`), so what is attested here
is that the layout is the standard HF one and that the reporter's own reshard served; no CB
serve of a PrismaQuant-sharded artifact has been run.

The strict RTX4090 lane additionally closes value identity without multiplying full-artifact
I/O. `shipcard.verify_safetensors_content_once` walks each finalized shard forward exactly once
through `O_NOFOLLOW` descriptors and computes the full-container SHA-256 and every raw-tensor
SHA-256 in that same traversal. It binds the result to device, inode, size, mtime, and ctime.
The resident strict exporter pays that one read after `save_file`; the streaming strict writer
already hashes both scopes while emitting bytes, so it captures a zero-reread stat receipt.
Final export replay accepts either receipt only while the namespace and every bound stat are
unchanged; the operator build launcher's final check is metadata-only and does not add another
weight-content pass. A copied artifact or independent serve validator has no process-local receipt and
therefore performs exactly one fresh content traversal. The strict serve launcher performs that
traversal inside the actual container namespace immediately before `exec vllm`, with each shard
overlaid by its own read-only bind mount so a host pathname replacement cannot retarget the
verified inode. Its no-clobber `/run` receipt is reused by the post-serve finalized census through
stat and ledger validation, then rechecked against the same underlying bind-source stats by the
host endpoint gate, never a second payload pass, and finally bound into shipcard evidence.
In-place mutation changes the receipt stat and refuses. The small FP16 codebook sidecar
keeps its separate exact digest replay. Generic artifacts retain their historical manifest
behavior. This cross-process handoff is intentionally fail-closed, not atomic: per-shard mounts
eliminate pathname replacement, while a host in-place writer can still race vLLM's open but
cannot yield accepted evidence because post-serve ctime/mtime replay then fails. Unforked vLLM
does not accept inherited verified shard descriptors; copying would require a second content pass.

**Nothing in the pipeline validates the artifact** — that is a physical lane boundary (`vllm`
is not importable in the build venv), not laziness. What the script now does instead of
echoing a suggested command is **print the open ship record**: the closing block runs
`python -m prismaquant.shipcard_cli show <exported>/shipcard.json` and names every slot still UNFILLED
(re-vet R13 + the deferred wave-1 item). The build lane opens the record; the serve lane must
close it. Both the numeric ship gate and the gold-lane KL/PPL contracts remain manual; §7
owns that.

### 3.3 Defaults at HEAD (+ the 2026-07-30 re-vet waves)

This table is the single source of truth for pipeline defaults; other sections reference it
rather than restate it. `tests/test_architecture_doc.py` pins the enumerable half against
`run-pipeline.sh`, so a default change that skips this table fails the suite. (Line numbers
were dropped from this block: the re-vet waves shifted them, and a stale `file:line` is worse
than none — `grep ': "${NAME:='` is exact and never decays. The heading's HEAD hash was dropped
2026-08-09 by the same argument: it had decayed to `8f14400`, a commit not in this branch's
history at all. The provenance stamp at the top of this file is the one place a commit id is
maintained.)

```
FORMATS=NVFP4,FP8_DYNAMIC,BF16   TARGET_BITS=4.75
PARETO_TARGETS=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25
NSAMPLES=32  SEQLEN=1024  DATASET=…/calibration/diverse-v1.jsonl
EXPERT_GATE_DATASET=…/calibration/xdom-gate-v1.jsonl (cross-domain)
ACTIVATION_ROWS_LIMIT=1024 on the GGUF/CB lanes else 256
COST_MODE=aura (R2 flip 2026-07-30; = COST_RENDER=cached-menu x
                COST_OBJECTIVE=aura-adjoint) PRODUCTION_CACHE_PREFETCH=require
PRODUCTION_RENDER_COST_SCORE_FIELD=weight_mse (M6, §4.2)
TARGET_DISK_GB=<unset>  EXPORT_CONTAINER=compressed-tensors
TARGET_PROFILE=<unset, spec-resolved>  TARGET_PROFILE_DEFAULT=vllm_packed_moe
ALLOW_PINNED=<none>  (forwards allocator --allow-pinned; see §4.5)
LM_HEAD_FORMAT=BF16  (fixed auxiliary policy; FP8_E4M3 is the approved native
                     compressed-tensors quantized-head rung, §4.5)
COST_FORMATS=<canonical FORMATS plus LM_HEAD_FORMAT when that fixed value is
                     non-BF16; no duplicate when the rung is already present>
SELECTION_MODE=surrogate, or validated-surrogate under a TARGET_DISK_GB card
EXPORT_PRODUCTION_CACHE_PREFETCH=require (native lane, D8)
MTP_FORMAT=BF16  PRODUCTION_CACHE=1  PRODUCTION_RECACHE=1
PRODUCTION_CACHE_LEVERS=gptq,static_act_order,joint_scale_opt
PRODUCTION_CACHE_RENDER_SCOPE=assignment  …_CACHE_PREFETCH=require
VALIDATED_SOURCE_PREFETCH=require   VALIDATED_FRONTIER_PICK=kneedle,
                                    or `budget` under a TARGET_DISK_GB card
VALIDATED_FRONTIER_SKIP_CALIB=$NSAMPLES (held-out disjointness, ON)
CB_EXPERT_EMPIRICAL / CB_SCALE_CODING / CB_CODEBOOK_* / CB_SCALE_SWEEP*  —
                     lost their shell DEFAULTS on 2026-09-02 with the Gridbook
                     codebook lane (archive/gridbook_lane_2026-09-02/). Every
                     default was set only inside an `EXPORT_CONTAINER=nvfp4_cb`
                     block, and that container now `exit 2`s. They survive as
                     settings-hash entries (`${VAR:-}`) because the CB render
                     plumbing still reads them (debt D34), so a persisted
                     cost/render artifact must still invalidate when an
                     operator sets one. With no shell default,
                     `allocator.py:2377` refuses a CB menu that reaches it
                     without an explicit --cb-scale-coding.
ACTIVATION_FAIR_PRICING=1  (exported as PRISMAQUANT_ACTIVATION_FAIR_PRICING)
PRISMAQUANT_CB_LDLQ / _SCOPE / _GATE, PRISMAQUANT_CB_MINCHAIN* — no shell
                     default since 2026-09-02; they survive only as
                     settings-hash entries (`${VAR:-}`). Their semantics are
                     unchanged and their code is live (debt D34), but nothing in
                     the pipeline sets them now that the CB export path is gone.
AURA_ADDITIVITY_GATE=measure
PRISMAQUANT_GGUF_IMATRIX=1  DEVICE=cuda  EXPORT_DEVICE=cuda
```

**The output-head policy has three resolved stage-identity axes.**
`LM_HEAD_FORMAT` is canonicalized once; `LM_HEAD_RENDER_ACTIVE=1` when either a fixed
non-BF16 head or the research DP-unpin needs head rows; and `LM_HEAD_DP_UNPINNED=1` only when
`ALLOW_PINNED` structurally names the profile's output head. `run-pipeline.sh` adds those
three values to `STAGE_SETTINGS_ENV`; `pipeline.py:_HEAD_SETTINGS` includes them in
`base-cost`, `render-cost-cache`, `render-cost`, `aura-dw-cache`, `aura-cost`,
`aura-hybrid-cost`, `frontier-cache`, `frontier-recache`, `production-cache-recached`, and
`production-cache-raw`. Cost-bearing stages use derived `COST_FORMATS`, so changing a fixed
head rung cannot reuse a body-only cost/cache artifact merely because `FORMATS` stayed the
same. `CACHE_FORMATS` and `AURA_CACHE_FORMATS` are derived from that expanded menu where
their stages need it. These are identity inputs, not new independent operator knobs.

BF16 preserves the old pin exactly. A non-BF16 `LM_HEAD_FORMAT` and an
`ALLOW_PINNED` token that lifts the head are a hard conflict: the first fixes a measured
auxiliary rung outside the body DP/bpp, while the second asks the research DP to choose one.
The pipeline includes the head in the baseline cost and render census whenever either mode is
active, but AURA includes it only for the DP-unpinned mode; fixed-head predicted loss is
reported as `aux_fixed_predicted_dloss` / `total_predicted_dloss_with_aux`, not folded into
the body objective. The fixed path requires an untied head plus exact probe and cost rows and
a candidate legal for the selected serving profile. The exporter lifts every alias of that
one structural head from profile pins while retaining unrelated pins; explicit `--ignore`
still wins. This policy quantizes no embedding by implication. The native production rung is
`FP8_E4M3`; CB head formats remain unservable (§9.2). A materialized format-menu cache may
render additional legal menu rungs for the active head even though the fixed assignment uses
only one; that is bounded cache/render overhead, not assignment or accounting drift
(`fixed_head.py`, `allocator.py`, `run-pipeline.sh`, `export_native_compressed.py`).

**One default in this section is not a shell default.** `--emit-marginals` /
`PRISMAQUANT_PROBE_MARGINALS` is **ON by default in the probe module itself**
(`incremental_probe._marginals_enabled`, `_env_flag(..., default=True)`), and
`run-pipeline.sh` sets nothing — `grep -c MARGINAL prismaquant/run-pipeline.sh`
→ 0. It is listed here anyway because it changes what every pipeline run writes
into `probe.pkl` (§4.1, §4.8): five per-channel vectors per Linear, costing
`(2·out + 3·in)·4` bytes. `--no-emit-marginals` restores byte-identical legacy
output. The flag is *not* a grouping axis but a key-set axis, so it joins the
precompute fingerprint, the per-shard reuse meta (`_expected_probe_shard_meta`)
and the LPS-invariant linear cache's `_CONTENT_META_KEYS` (`5639a8b`): a
flag-off shard is refused rather than pooled into a flag-on run, where it would
otherwise contribute marginal-less entries that nothing downstream checks for.

**`PRISMAQUANT_CB_LDLQ_SCOPE` is the authoritative LDLQ selector**; the older boolean
`PRISMAQUANT_CB_LDLQ` survives only as its degenerate spelling. `cb_serialization_context_from_env`
(`nvfp4_cb_footprint.py:660`, scope read `:683`) validates the scope against
`{none, nvfp4, all}` (`:717-720`) and, when the scope is set, requires the legacy bool to
agree with `scope != "none"` (`:726-737`) — with exactly one back-compat exemption, legacy
`true` paired with `scope=nvfp4`, because the bool cannot express a mixed per-family scope.
With the scope unset the bool decides and the scope is *derived* from it: `all` when true,
`none` when false or absent (`:739-745`). Under `require_explicit` at least one of the two
must be present (`:691-701`) — the CB producer settings are never defaulted silently. Neither
name has a `run-pipeline.sh` shell default; the CB drivers export them directly.

**The learned-codebook selector is versioned build-time intent; the bundle is
render authority.** Historical trainer v1 preserves the measured policy that
uses learned FP8 books at K28–K46 and lattice at K47/K48, with
`CB_CODEBOOK_SOURCE_SCOPE=none|fp8|all` selecting the family and
`CBL_RUNG_POLICY[k]["enabled"]` selecting its cells. Once a value-bearing bundle
is present, `CBLearnedBundle.codebook_source_by_format` freezes its complete
per-rung map and `cb_fields_for_context` checks the exact `(qname, format)` cell
before it decides whether calling strict `codebook_for()` is legal. Changing a
process-global policy after context creation cannot reinterpret that artifact;
`all` still warns because learned NVFP4-CB is measured NO-GO
(`cb_learned_bundle.py`, `nvfp4_cb_footprint.py`).

**Learned-v2 has no assumed crossover.** Its domain is the exact twelve-rung
FP8 producer ladder K4..K48 step 4, and every rung starts on the committed
lattice. `cb_imatrix.py` derives dense `act_sq_sum / n_tokens_seen` and routed
`expert_act_sq_sum / expert_tokens` values from the existing sensitivity probe;
it creates no calibration cache. A rung becomes learned only when
`prismaquant.fp8_cbl_promotion_receipt.v1` passes both held-out result gates and
binds the complete source checkpoint/tensor map, training calibration and
imatrix values, role census, exact candidate FP16 table digests, and all rung
decisions. Missing calibration, a density shortfall, a tie, or a missing,
tampered, substituted, or incomplete receipt keeps the rung lattice or refuses;
one rung's result is never transferred to another (`cb_imatrix.py`,
`cb_learned_promotion.py`, `build_cb_learned_bundle.py`). Trainer v1 remains the
compatibility default; choosing v2 changes no production default by itself.

**Routed learned books are keyed per (layer, stack, rung) — campaign rule R1.**
A routed book covers the fused `w13` stack, gate and up pooled, and `down_proj`
is a one-projection stack that keeps its own book. `build_cb_learned_bundle
--routed-book-keying` selects the rule: `stack` (the default) or `role`, the
pre-R1 book per `(layer, projection, rung)` that the campaign's A/B arm still
burns. The keying reaches everything that identifies a book — the burn
selection and bank shard name the population in their `projection` member
(`gate_up_proj` under stack keying, so a per-role bank can never satisfy a
pooled request), the pooled book is trained against the fused rank-3 stack and
the packed target's own imatrix entry, and every routed learned cell records
`routed_book_keying` in the bundle manifest. An absent field reads as `role`:
pre-R1 books are per role by construction. The producer side speaks the same
keying: `tools/dsv4_onlaw_book_burn.py --keying stack` burns one `gate_up_proj`
cell per `(layer, rung)` over the fused population (`load_population`, the
builder's `provide_weight` row order and `_stack_col_weights` reshape), and
`tools/dsv4_onlaw_book_select.py --keying stack` authors the selection over
`(layer, gate_up_proj | down_proj)`. The packed imatrix entry a stack cell is
burned against is the export's own derivation, materialized by
`tools/dsv4_packed_col_weights.py` for checkpoints whose experts are per-expert
Linears (DSv4) — the burn refuses a pickle without it, or one whose entry is a
different spelling of the per-expert vectors, so burn, bundle and export weight
the fused rows by one tensor. `down_proj` is a one-projection stack: its role
cell and its stack cell are the same tensors under the same name, so a role
bank's `down_proj` shards satisfy the stack request unchanged.

Both CB exporters branch on the bundle's own record, never on a flag. A
stack-keyed cell emits ONE codebook target for the fused weight, under the
packed spelling a lattice layer already uses; a role-keyed cell keeps the
per-half declaration Gridbook 0.8.4+ resolves independently. **A fused routed
weight whose scheme would name more than one codebook refuses at export**
(`routed_moe_codebooks.fused_targets_with_split_books`,
`export_nvfp4_cb_streaming.py`, `export_nvfp4_cb.py`), unless
`--allow-per-role-books` is passed, which stamps
`build.routed_codebook_books.per_role_books_override` onto the shipcard beside
the fused targets it covers. The gate's predicate is structural and
producer-side — it counts distinct codebook references the artifact is about to
write — so it asserts nothing about a runtime; the runtime consequence (split
books cannot attest the persistent-B FP8 fast lane) appears only in the
human-facing message. What made this a rule: the shipped 87 GB DSv4 body's 11
routed FP8-CB layers all agree on format, gate = up = down = `FP8_CB_K28`, and
still split because book identity was in no promotion's domain
(`tests/test_cb_banked_bundle_builder.py`,
`tests/test_resident_routed_moe_cbl.py`).

**Scale search remains family-scoped producer identity.**
`CB_SCALE_SWEEP_SCOPE=none|nvfp4|fp8|all` resolves the scale-search arm through
`scale_sweep_for_format`; with the scope unset, the old `CB_SCALE_SWEEP` boolean
still means all/none (`:205-216,525-547,921-1031`). Production two-tier FP4
requires the NVFP4 family to sweep, so a mixed artifact's measured one-shot-FP8
arm is `nvfp4`, while sweep-matched CBL is `all` (or the legacy unset+true
spelling; `:265-274`). Source-bearing cost stamps enumerate the exact
`codebook_source_by_format` map, round-trip it, and compare the complete key/value
map at the cost/render gate; the compact serialized-payload context copied into
`quant_config.json` carries the complete frozen bundle map too. The legacy
scalar is always the ANY of the stamped map, learned-content digests are required
iff that map contains a learned rung, and an explicit non-`none` build scope is
retained when the scalar alone cannot represent it (including an all-lattice
K47/K48 policy menu). A missing K43 entry, learned→lattice flip, or contradictory
scalar is a refusal (`nvfp4_cb_footprint.py`;
`tests/test_per_rung_codebook_source.py`). The render stamp writes a sweep scope
only for a genuinely mixed `nvfp4|fp8` choice; homogeneous scale-search choices
retain the legacy shape. Therefore the unset/default all-lattice source and
legacy all-family sweep retain the old stamp shape and rendered bytes, pinned
against baseline `76666bd` by
`tests/test_cbl_scope_identity.py:67-128`.

`EXPORT_CONTAINER` ∈ {`compressed-tensors`, `gguf`, `tessera`} selects the lane — the
sanctioned three (Robert, 2026-09-02); `nvfp4_cb` was the fourth until the Gridbook lane
retired that day and now `exit 2`s (§9.2), and `tessera` was added on 2026-09-03 together
with the driver arm that can act on it (§9.4). The
preflight **refuses a lane the architecture has not declared** (`supported_lanes`,
re-vet R6) — an undeclared lane does not fail at serve time, it serves uninitialised expert
memory and generates coherent-looking garbage. Being in the vocabulary is therefore not
permission to build: `tessera` is declared by `qwen3` alone, and its own preflight
(`prismaquant/tessera_export_lane.py`) still refuses every run on the PENDING release pin.

**`COST_MODE=aura` is the default since 2026-07-30 (re-vet R2).** Both flagship artifacts
(regen-27B, 35B arm-E) were produced with it and its served margin over the previous default
is −38%/−39.5% confident-KL at the 4B knee across two calibrations and −17.9% at 27B (§4.3).
`production-render-score` remains fully supported **on non-CB menus** and is the
explicit/legacy spelling — historical artifacts reproduce by setting it. It is **unlicensed on
any CB/CBL-containing menu** and fails `exit 2` there (§3.5): its score field is `weight_mse`,
and the per-unit factorization `mse(e,K) ≈ s_e·g(K)` fails in weight currency across a
codebook-basis change (CV monotone in rung, 0.088 at K28 → 0.224 at K48; 8 of 10 rung-pairs
breach the 0.10 bar, while lattice→lattice on the same planes passes at 0.067/0.056). The flip was gated on the two preconditions R2
named, both landed: the `provenance["cost_mode"]` stamp (§3.4), so a `WORK_DIR` built under
the old default **rebuilds its cost table loudly** instead of silently allocating on the other
estimator; and the wired additivity report (§4.3, stage `[3c]`), so every AURA artifact carries
its own trust-region number. The three CB/GGUF lanes resolve their own render and objective
through the render-faithfulness assertion (§4.7) rather than through this default.

**`TARGET_PROFILE` is deliberately unset** (re-vet R11 / D4). `resolve_target_profile` gives an
explicit request precedence, so a shell default silently beat every architecture's
`spec.default_serving_profile` — measured cost 2026-07-11: 226 dense FP8 Linears coerced to
BF16 on the Hy3 export. Unset, the spec wins; `TARGET_PROFILE_DEFAULT=vllm_packed_moe` is the
fallback for architectures that declare nothing (never `research`, whose menu is unbounded);
an explicit `TARGET_PROFILE` still wins, so every in-tree launch script is bit-identical. The
resolved profile is stamped into `layer_config.json`'s reserved `__prismaquant__` block and
read back by the exporter, so allocator and export cannot disagree.

**`TARGET_DISK_GB` makes the card the constraint** (re-vet R1 / D12). When set it overrides
`TARGET_BITS` (the allocator re-emits at the bpp whose exact footprint fits), narrows the
Pareto sweep to the ~3 rungs that can ship, flips `SELECTION_MODE` to `validated-surrogate`
and `VALIDATED_FRONTIER_PICK` to `budget` — min measured KL among the allocations that fit.
An explicit `SELECTION_MODE`/`VALIDATED_FRONTIER_PICK` still wins. §4.6 owns the selection
semantics; §4 owns the cost-mode semantics; §5 owns the lever semantics.

### 3.4 Reuse guards and the silent-reuse class

**The key set is `pipeline.py`'s job; the values are the shell's.** `STAGE_SETTINGS_KEYS`
(`pipeline.py`) declares, per artifact, which settings that artifact's identity depends on.
`run-pipeline.sh` passes every value once (`STAGE_SETTINGS_ENV`, `596`+), `pipeline.py
--write-stage-settings` projects them onto each declared key set and emits
`artifacts/stage_settings.json`, and `require_stage_settings <artifact> <stage> [LATE=v …]`
(`684`) reads that projection instead of re-deciding the key set at the call site. Late-computed
values (`AURA_CACHE_FORMATS`, `CACHE_FORMATS`, `ASSIGNMENT_DIGEST`) are passed as overrides;
a declared key nobody supplies is a hard stop, not a silent gap. This is re-vet **R5**, and it
closes **D6** by mechanism rather than by enumeration — the twelfth stage cannot arrive without
a guard, because adding one is a table entry, not a bespoke argument list.

Contract:

| state | outcome |
|---|---|
| artifact absent | record this stage's projection, build |
| recorded projection matches | reuse |
| recorded projection differs | **`exit 2`**, naming every differing key and the stale file |
| no record for this stage (pre-guard artifact) | **WARN**, record, guard from then on |

The manifest is `<artifact>.settings.json`, keyed by stage, so two stages can legitimately own
one path — under `COST_MODE=aura` + `validated-surrogate` the AURA dW cache and the frontier
cache **are the same file** (principle 8's one-render identity), and both key sets coexist.
Pre-R5 flat manifests are read as a `legacy` block and still guard the stage whose key set they
match, so no live `WORK_DIR` is invalidated by the upgrade.

**Coverage is now every skip-if-exists artifact** — **16 call sites over 16 declared
artifacts**: `probe`, `base-cost`, `render-cost-cache`, `render-cost`, `aura-dw-cache`,
`aura-cost`, `aura-hybrid-cost`, `cb-col-weights`, `cb-hybrid-cost`, `frontier-cache`,
`frontier-kl-point`, `frontier-recache`, `production-cache-recached`, `production-cache-raw`,
`gguf-skeleton`, and `cb-learned-bundle`. The last is keyed on the source model,
format menu, learned scope, bundle path, and the exact col-weight file digest
(`pipeline.py:158-167`; `run-pipeline.sh:1053-1081`). (Wave 3 reported 16 sites
because `cb-col-weights` was guarded at three
near-copies of the harvest; wave 4's `harvest_cb_col_weights` collapsed those into one function
with four callers, so the guard is now stated once and the artifact-to-site map is 1:1.) Render-affecting env is captured in `RENDER_ENV_SETTINGS` (`585`:
`PRISMAQUANT_NVFP4_SCALE_RULE`, `PRISMAQUANT_GPTQ_DAMP_SWEEP` default `0`,
`PRISMAQUANT_GPTQ_DAMP`, `PRISMAQUANT_ACT_CLIP_QUANTILE` default `0.999`,
`PRODUCTION_CACHE_LEVERS`, `PRODUCTION_CACHE_DISABLE_LEVERS`) and spliced into every artifact
that stores rendered weights.

**Over-keying is the risk the table is written against.** Declaring a key an artifact does not
depend on forces a spurious rebuild, and some of these are 90 GB. The rule applied: key an
artifact on the inputs that change its *bytes*, and key expensive artifacts conservatively (the
probe is keyed on model/corpus/windows/modality and **not** on `FORMATS` — it is format-blind;
`cb_col_weights.pkl` is keyed generously because it rebuilds in minutes). Historical manifest
key names (`NS`, `SL`, `SEED`) are preserved where they existed, so artifacts built before R5
compare equal instead of rebuilding.

**Cost tables get a second, orthogonal gate.** `cost.pkl` is the same path under *every*
`COST_MODE`, so a settings match is not sufficient — the file could be the previous mode's
estimator. Every producer (`incremental_measure_quant_cost`, `production_render_cost`,
`aura_cost`, `expert_empirical_cost`, and the inline sidecar-backfill finalize) now stamps
`provenance["cost_mode"]` from `--cost-mode`, and `cost_table_reusable()` (`669`) makes reuse of
the *allocator's* table conditional on it matching. A mismatch **rebuilds** with a loud line
naming both modes; an unstamped (pre-R2) table warns and is reused, never invalidated. Under
`COST_MODE=local` the baseline *is* the allocator table so it carries the same gate; under the
other modes `cost_baseline.pkl` is mode-agnostic on purpose and is shared across mode changes.
This is re-vet **R2 precondition (i)** — the prerequisite to flipping the `COST_MODE` default,
which is *not* done here.

**Sanctioned study-grade assembly (opt-in, 2026-08-03).** The production guards above and the
CB shard/serialized-payload/lattice/render-scope gates remain fail-closed. For the explicitly
user-accepted DSv4 learning experiment, `allocator --accept-research-cost-table` may assemble a
complete `layer_*.pkl` store over a production base when both `--research-cost-base` and
`--research-cost-segments-dir` are supplied. `research_cost_acceptance.py` verifies layer ids,
row keying/counts, source hashes and overlap precedence, then stamps
`cost_provenance="research_assembled_segments_user_accepted_2026-08-03"` plus its manifest.
The stamp travels into `selection.json` and the reserved layer-config metadata. A stamped table
is refused without the allocator flag; an unstamped table cannot be blessed by the flag.

### 3.5 Archived modes — the twelve `exit 2` gates

| Trigger | Lines | Archive |
|---|---|---|
| `COST_MODE=grouped-kl` | `308-312` | `archive/grouped_kl_2026-05-28` |
| `COST_MODE=production-render-staged` \| `-tail` | `318-322` | `archive/production_render_staged_2026-07-30` |
| `FISHER_WEIGHTED_GPTQ` truthy | `205-213` | `archive/fisher_2026-05-15` |
| `FISHER_OUTPUT_MSE_ALLOCATOR` truthy | `205-213` | `archive/fisher_2026-05-15` |
| `PRODUCTION_CACHE_LEVERS` ∋ `fisher_gptq` | `215-220` | `archive/fisher_2026-05-15` |
| `HADAMARD_DUQUANT` truthy | `354-360` | `archive/hdq_2026-05-14` |
| `PRODUCTION_CACHE_LEVERS` ∋ `hadamard_duquant` | `361-366` | same |
| `MULTI_SHOT_PASSES` ∉ {unset, 1} | `367-373` | `archive/multi_shot_2026-05-19` |
| `ALLOC_PROPAGATED_SENSITIVITY_REPORT` non-empty | `374-380` | `archive/l3_propagated_2026-07-30` |
| `PRODUCTION_CACHE_UNION` truthy (the archived surrogate-driven **smart-union candidate selector**, not the exact set union in §5.4) | `381-387` | `archive/union_cache_2026-07-30` |
| `MSE_PROMOTION` truthy | `388-394` | `archive/mse_promotion_2026-07-30` |
| `EXPORT_CONTAINER=nvfp4_cb` (the retired Gridbook codebook lane, 2026-09-02) | at the `EXPORT_CONTAINER` validity check | `archive/gridbook_lane_2026-09-02` |

**A thirteenth `exit 2` gate that is *not* an archived mode.** `COST_MODE=production-render-score`
(or `production-render`) refuses when the run targets a CB/CBL menu — detected as
`EXPORT_CONTAINER=nvfp4_cb` **or** a `FORMATS` entry matching `*_CB_*`. Since 2026-09-02 the
first of those two signals is subsumed: the `nvfp4_cb` container gate fires first and refuses
outright. The `FORMATS` limb still fires, and still matters, because the CB format/cost
plumbing survives the lane (§9.2, debt D34) and can still be named in a menu. The mode itself is
alive and correct off CB; what is unlicensed is the *pairing*, because the mode scores on
`weight_mse` and that currency does not survive a codebook-basis change (§4.3 wording above;
measured CV 0.088 → 0.224 across K28→K48). Activation currency holds where weight currency
does not, which is why `aura` is unaffected. The guard's predicate is executed — not merely
asserted to exist — by `tests/test_run_pipeline_defaults.py::test_cb_unlicensed_guard_actually_fires`,
which trips it on each CB signal independently and confirms both non-CB controls pass.

Four of these landed with the 2026-07-30 re-vet (R17, R4, R18 ×2). Each error string carries
**the measurement that killed the lever**, so the refusal teaches rather than merely blocks.
The archive directory names are load-bearing for the orchestrator: moving or renaming one
breaks its gate. Lessons: §11.

The word *union* is overloaded historically, but the mechanisms are not. The archived shell
flag selected which candidates deserved rendering from an `output_mse` percentile. The live
`prismaquant.union_production_cache` CLI makes no quality decision: it accepts only already
rendered, disjoint keys under exact expected coverage and performs their verified set union.
It is therefore a manual cache-build transport/reconciliation tool, not a restored pipeline
mode (`union_production_cache.py`; §5.4).

A fourteenth refusal lives outside `run-pipeline.sh`: `export_native_compressed.main()` calls
`_refuse_archived_block_output_match()`, which `SystemExit`s if
`PRISMAQUANT_BLOCK_OUTPUT_MATCH` is set truthy (`archive/block_output_match_2026-07-30`,
re-vet R25). It is a `SystemExit` rather than a shell gate because the lever was an exporter
env var no pipeline stage ever set.

### 3.6 `pipeline.py` — what the contract layer actually is

**It has exactly one load-bearing job, and §3.4 is it.** `pipeline.py` owns
`STAGE_SETTINGS_KEYS` — the per-artifact declaration of which settings each build artifact's
identity is keyed on — and the `--check-stage-settings` guard the orchestrator calls at every
skip-if-exists site. That is the one thing the shell provably got wrong (ten artifacts with no
guard at all, six more each holding their own opinion of their key set), and it is the one
thing `pipeline.py` was already positioned to fix: it receives the settings and it is the only
place where "what does this artifact depend on?" can be reviewed as a table rather than
rediscovered per call site. Re-vet **R5**, adjudicated in favour of Lens 2's narrow promotion.

`_HEAD_SETTINGS = (LM_HEAD_FORMAT, LM_HEAD_RENDER_ACTIVE, LM_HEAD_DP_UNPINNED)` is part of that
load-bearing table, not descriptive spec metadata. Those axes cover every persisted cost,
AURA, frontier, and production-cache artifact that can contain or depend on `lm_head`; the
cost-stage `FORMATS` key is sourced from derived `COST_FORMATS`. A head-policy change therefore
invalidates reuse before any pickle is opened (`pipeline.py:STAGE_SETTINGS_KEYS`, §3.3).

Everything else in the file remains **descriptive**: it also writes and `--validate`s a spec
JSON declaring 14 artifacts, 3 gates and 9 base stages plus render-mechanism stages generated
from `render_score.resolve_render_mechanism_order`. Nothing downstream reads that JSON back,
and its `validate()` is tautological in the production path — the spec it validates is the one
`default_production_pipeline_spec()` just generated from its own hardcoded `ResourceContract`s,
and `run-pipeline.sh` never passes `--input`. Treat the *spec* half as documentation with a
linter. Coverage stays partial in both directions by choice (re-vet: modelling the ten
executed-but-unmodelled stages would be fiction-surface without teeth): `validate.vllm_smoke` is
always stripped and `validate.kl` is stripped whenever `SELECTION_MODE=surrogate`.

`APPROVED_RESOURCE_OWNERS` is now honest (D10): `rendered_weights → ProductionWeightCache`,
`perturbed_activations → PerturbedActivationCache`, `streaming_model_weights → LayerCache`
(`layer_streaming.py`). The two placeholder names that existed nowhere in the tree
(`StreamingActivationCache`, `StreamingModelPrefetch`) are deleted, and a test asserts every
approved owner has a class behind it. `kl_measurement.QuantWeightCache`, the other candidate
owner, went to the archive wall with L3 (§4.4) and is no longer a live holder.

The one-cache rule (§5.4) is still enforced by the runtime strict-cache gates, not by this file.

### 3.7 `WORK_DIR` layout

Created at `408`:

```
artifacts/  probe.pkl, cost*.pkl, layer_config*.json, pareto.csv,
            pareto_assignments/, production_*_cache.pkl + shard dirs,
            validated_frontier_kl*.json, cb_col_weights.pkl,
            cb_learned_bundle.pqcb, skeleton.gguf,
            pipeline_spec.json, stage_settings.json, *.settings.json
act/        probe activation cache        work/  streaming layer shards
logs/       probe|cost|allocator|export   exported/  compressed-tensors ckpt
```

Plus `exported.gguf` (GGUF lane) and `exported_nvfp4_cb/` (CB lane) directly under `$WORK_DIR`.
Sizing discipline — a 27B cache is ~90 GB — is §10.

## 4. Cost models & allocation

The allocator needs one number per `(Linear, format)`: `predicted_dloss`, the estimated
end-loss damage of that rendering. Below, the machinery that produces it and spends a bit
budget against it. Paths are repo-root-relative; the orchestrator is
`prismaquant/run-pipeline.sh`. One lane exception to "one cost run, one cost table": on the CB
lane under an LDLQ scope the same run also emits a raw (no-LDLQ) render sidecar, so a second
allocator-consumable cost table falls out of it for free — §6.5.2.

### 4.1 Stages that always run

| Stage | Module | Produces |
|---|---|---|
| L1 Fisher probe | `incremental_probe.py` (`run-pipeline.sh:539-560`) | `artifacts/probe.pkl` (per-Linear `h_trace`, `n_params`, shapes; since 2026-08-14 also the five per-channel marginal vectors, default-ON — §3.3, §4.8) **and** `WORK_DIR/act`, the activation cache every later stage reads |
| Base RTN cost | `incremental_measure_quant_cost.py` (`:606-661`) | per-`(Linear, format)` measured RTN error; under `aura` demoted to sidecar-backfill source (`:928-952`) |

The probe is streamed shard-by-shard through `layer_streaming` — head resident, body paged,
MTP a built-in shard kind (`incremental_probe.py:2-17`); a modality guard aborts on
probe/`CALIBRATION_MODALITY` mismatch (`:562-599`).

**Exact sample-axis probe map/reduce is an opt-in execution of that same probe, not a new
estimator or cache.** `sample_parallel_probe.py` prepares one immutable `ids[N,T]` tensor,
derives the complete qname/disposition manifests from the strict Qwen3.8 configuration and
safetensors headers, binds the source checkpoint's value-bearing content identity, and emits a
closed run contract. That identity also binds the immutable PrismaQuant runtime-snapshot
closure/commit/tree and a trusted launcher's host-inspected `sha256:` container-image digest.
Each worker re-hashes the mounted read-only snapshot and derives its own source census through a
host-local `PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE` before any CE/precompute/shard reuse; it
compares only the stable config/tensor/weight-map/model-content/qname projection across hosts and
repeats the local source fingerprint/census and snapshot checks immediately before publication.
The host-local streamed identity v1 and its original path-bearing `content_sha256` remain unchanged
for compatibility. Cross-host joins use an additive portable-content projection: recursively
canonicalized config semantics exclude only `_name_or_path` and `transformers_version`, while
checkpoint maps and unique shard basename/size/SHA-256 records remain value-bearing. Local cache
validation independently reconstructs the live text-only semantic config, so path neutrality does
not permit a rope/config drift. Run-contract creation requires a coordinator-local complete cache
and records that upstream portable digest beside the path-independent source-census digest. The
RTX4090 burn revalidates every host cache against its live checkpoint and requires both portable
joins before GPU-bearing measurement; a direct shard-hash-only contract is refused.
Python validates the snapshot but does not claim to inspect its own Docker image: the immutable
image digest is an explicit launcher-attestation trust boundary. Partitions are canonical
contiguous, disjoint sample ranges whose union is exactly `0..N-1`; every worker processes the
complete model schedule with `--unified-sweep --no-include-visual`. Partial layer schedules,
user-authored qname lists, h-detail, visual modules, and routed or packed statistics/caches fail
closed.

The operator runbook is strict about that boundary: it executes the snapshot verifier from the
exact pinned Git object rather than the candidate it judges, materializes a clean committed
snapshot, removes and checks all write bits, verifies the exact pinned registry RepoDigest on the
host, and launches by that digest-qualified reference. Docker's backend-dependent local image ID
is not a cross-host authority. Both hosts mount their possibly different absolute host roots at
the same canonical `/model`, `/dataset`, `/run`, and `/worker-state` paths; snapshot/model/dataset
are read-only. The container runs as the invoking non-root host UID/GID, with writable home,
temporary, and compiler caches confined to the private worker-state mount; the two reviewed Spark
hosts expose their NVIDIA device nodes to that user. The same container re-verifies the snapshot,
removes `PYTHONPATH`, enables safe-path/no-user-site/no-bytecode modes, sets both required CB
compilation switches, and invokes the producer only through the snapshot-owned
`prismaquant_source_bootstrap.py run-module`. Bare live-tree `python -m` commands are not a valid
campaign launch.

The digest-bound sample cover is a required worker input, not merely merge-side metadata.
Before model/GPU readiness, each worker strictly decodes the cover and run contract (duplicate
JSON object members refuse at any depth), replays their closed schemas/digests, and requires
the cover's execution identity and qname census to equal the run contract. The selected
partition loaded from the current calibration tensor must equal its exact cover member; that
comparison transitively binds the calibration artifact SHA-256 plus the local/global tensor
hashes, so a post-cover calibration replacement fails in CPU preflight.

The ordinary versioned `precomputed.pt` remains the sole phase-1/2 resume cache. Sample-v1 reuse
requires the exact dense-Qwen resident cover (`lm_head` only), closed raw-stat types and source
geometry, `N_i·(T-1)` tokens, FP32 marginals/`h_full`/per-token gradient squares with consistent
traces, finite activations at every layer boundary, and empty resident-activation, routed, and
shared-state maps. Writes are atomic; any readable-but-invalid payload is a cache miss and
recomputes through the existing streamed GPU path.

Importance weighting has a real two-stage scalar barrier. Stage 1 emits each partition's raw
shifted-token CE sum/count. `merge-importance` proves the exact sample cover and exactly
`N·(T-1)` observations before publishing one digest-bound global CE mean. Both local and global
v2 receipts bind the complete execution identity, stable source projection, producer snapshot,
dtype, and image digest, so none can cross a producer/source/runtime boundary. Stage 2
deliberately reruns phase 1, applies that identical global mean before body backward, and emits
raw Fisher and marginal sufficient statistics. `sample_parallel_probe_merge.py` adds
scalar/vector sums, takes an elementwise maximum only for activation absolute maxima, and calls
the ordinary Fisher finalizer once at the global `N·T` denominator. Per-worker coverage is
disposition specific: dense body rows see `N_i·T`, `lm_head` sees `N_i·(T-1)`, and MTP sees
`N_i·(T-2)`; the actual head Linear call excludes the unscored final token, so its raw count and
activation marginals have that same `T-1` cover. All Fisher rows still normalize by global
`N·T`. Before and after addition, `sum(fisher_row)` and `sum(fisher_col)` must each reproduce
`h_trace_raw` within the same `rtol=1e-3` numerical qualification used by `SensitivityCard`;
finite but miswired marginals therefore cannot enter the imatrix/rank-1 Fisher path. “Exact”
here means one calibration set, objective, sufficient-stat reduction, and final
normalization. It does not promise bitwise equality with a monolithic run because partitioning
changes GEMM shapes and therefore low-order floating-point accumulation.

Only dense body Linears publish FP32 activation rows; terminal BF16 `lm_head` and MTP retain
probe statistics but no activation-cache entry. The exact priority law is the collision-free
uint32 PRP `blake2b64-keyed-fmix32x2-prp-global-row-fused-group-v2`: BLAKE2b derives little-
endian `k0,k1` from the calibration hash and fused-group UTF-8 bytes, then priority is
`fmix32(fmix32(global_row XOR k0) XOR k1)` using overflow-free int64 limb multiplication.
`0 < N·T <= 2^32` is a hard contract. Global rows, priorities, and top-R reservoirs remain on
the input device during hooks; one cached plan is shared by q/k/v or gate/up siblings and the
device-to-host transfer occurs only at layer flush. The reducer independently reconstructs the
law with scalar/NumPy code, and exact local-top-R/global-top-R associativity produces the merged
cache without a tie convention. Public validation does not trust the manifest's claims: it
recomputes priorities over the full global row domain, exact order/cardinality, row uniqueness
and range, and identical fused-group rows. The campaign uses R=1024. Final publication requires one
`--output-bundle` directory containing fixed `probe.pkl`, `activation_cache/`, and `commit.json`:
the complete pair is staged/fsynced below one parent and atomically no-clobber renamed, so a
mid-publication crash exposes no half-pair and an exact retry succeeds. All model execution
remains on the streamed model, `LayerCache`, prefetch, and existing activation-cache path; there
is no `torch.distributed` process group, parameter tensor parallelism, secondary residency
layer, or serving/graph-compilation change (`sample_parallel_probe_contract.py`,
`sample_parallel_probe.py`, `sample_parallel_probe_merge.py`, `incremental_probe.py`).
Consumers validate the bundle as one object: exact topology and commit bytes, the complete closed
probe census, activation-manifest identity, and every activation tensor are replayed before use.
Probe hashing and deserialization consume one regular no-follow descriptor and the burn uses that
captured payload rather than reopening a pathname. `ActivationIndex` keeps an opened directory
descriptor for the validated activation cache and verifies each lazy blob's stable file identity,
tensor hashes, qname, and merge stamp against the captured manifest at consumption; an
`os.replace` between validation and use therefore refuses rather than feeding different rows.

**The streamed driver must reproduce the model's own forward, and where it
cannot observe a rule it must not guess one.** `_call_layer` drives one decoder
layer at a time and therefore bypasses `Model.forward` entirely, so everything
that forward does *between* layers has to be re-supplied: the profile's
`expand_hidden_for_layers`/`collapse_hidden_after_layers` (DSv4's `hc_mult=4`
streams), `extra_layer_kwargs` (DSv4 hash-routed layers' `input_ids`),
`new_forward_pass_state` (Gemma4 shared K/V), the mask, and the rope. Rope is
the subtle one, because two namespaces collide: a multi-rope model's rotary is
keyed by *rope axis*, while a layer can only report its *attention schedule*.
On Gemma3/Gemma4 those coincide. On DSv4-Flash they never do — axes are
`main`/`compress`, schedules are `sliding_attention` /
`compressed_sparse_attention` / `heavily_compressed_attention` — so the lookup
missed every layer, and the code answered the miss by substituting `main`.
That silently rotated 41 of V4-Flash's 46 layers on base 10000 with YaRN
disabled where `DeepseekV4Model.forward` uses `compress_rope_theta` 160000 with
YaRN, an angle error that grows with position. The BF16 teacher built through
that path scored perplexity **262** — worse than the 2.34-bpp student it
existed to grade — while the served artifact was healthy, and it was used as a
gold KL reference. The bug is notable twice over: PATCH 06 had already found
and fixed exactly it *inside* the vendored forward
(`modeling_deepseek_v4.py:1514-1521`), and the streamed path reintroduced it.
The mapping now has a single definition,
`DeepseekV4RotaryEmbedding.rope_axis_for_layer_type`, which the model's forward
and `ModelProfile.rope_axis_for_layer_type` both resolve through;
`_compute_position_embeddings` re-keys the rope dict by attention layer type so
every `_call_layer` caller is fixed at once; and the `main` fallback is gone —
an unresolved layer type now raises. Guarded by
`tests/test_multilayer_rope_forward.py` and
`tests/test_deepseek_v4_profile.py::test_rope_axis_mapping_matches_the_vendored_definition`.
See also the teacher forward-fidelity gate (§7.3), which is what makes a
teacher this broken impossible to ship silently.

`h_trace` is the empirical CE Fisher diagonal trace. Additive model:
`0.5 · h_trace · weight_mse · gain` (`allocator_solver.py:60-63`, derivation
`allocator.py:13-52`).

**One denominator: the global calibration token count** (PR #14, `f53945f`). Every row —
dense trunk and per-expert alike — is `h_trace_raw / (nsamples × seqlen)`
(`finalize_fisher_stats`, `sensitivity_probe.py:496-534`; the incremental backend calls the
same function, `incremental_probe.py:2644`, stamping `meta["fisher_norm_tokens"]` at `:2754`).
Both backends share it, and `h_detail` blobs use the identical count (`h_detail_version: 4`,
`sensitivity_probe.py:488`).

This **reverses** the earlier per-routed-token convention that this document and `CLAUDE.md`
previously described. Dividing an expert row by its own routed count inflates it by
(global / routed) — the same `1/p_e` overweighting audit M4 set out to remove, merely implicit,
and exactly inverted importance weighting (the least-used experts look the most sensitive).
Typical inflation is ~`n_experts/top_k` (≈32× on a 256-expert top-8 model); the degenerate
1-routed-token case reaches ~33,000× at a 32k-token calibration. `n_tokens_seen` and
`route_prob` both survive as metadata only.

**Legacy probes hard-fail.** `renormalize_probe_fisher` (`allocator.py:1066-1163`, called
`:1455`) recomputes every row from the stored raw accumulators. Per-row
`h_trace_norm_tokens` stamps win over the meta count — a merged multimodal visual pass was
finalized at its own token count, so honouring the stamp keeps the recompute idempotent. A row
that carries raw accumulators but has neither a stamp nor a usable meta count is a `SystemExit`
naming the remedy (re-probe); `--allow-legacy-fisher-norm` (`:1190-1195`) downgrades it to a
warning for reproducing historical allocations (`612fc38`). Re-solves of the shipped
Qwen3.6-27B and Qwen3.5-35B-A3B probe/cost pairs at `TARGET_BITS` were unchanged by the fix.

### 4.2 `production-render-score` — the explicit/legacy cost mode

The default until 2026-07-30, now the explicit spelling that reproduces every pre-flip
artifact (§3.3; the flip is re-vet R2). It builds a format-menu render-score cache and derives cost from the scores
the render itself recorded (`run-pipeline.sh:665-715`; staged/tail variant `:717-823`).
Contract at `production_render_cost.py:1-16`: the rendered score is the damage of the weights
export will actually ship, so rows set `output_mse_measured=False` and the allocator consumes
`predicted_dloss` directly instead of re-applying the Fisher proxy.

The streamed CB spelling changes **render lifetime only**, never menu membership or score
semantics. `streaming_production_cache.py` visits every eligible `(Linear, requested CB rung)`,
hands the cache-canonical tensor to the scalar consumer synchronously, and requires an
acknowledged pair checkpoint before releasing it. `production_render_cost.py` may consume a
score-only CB pair only through that attestation; an unattested scalar with no retained shard
is stale, not a cache hit. The complete pair count is a close condition, so disk pressure can
bound residency but can never prune the allocator's candidate set (§5.4; contrast the archived
`PRODUCTION_CACHE_UNION`, §11).

**M6 — the score field is `weight_mse`, not `h_trace × output_mse`.** The legacy product
carries activation energy `E‖x‖²` twice, since `h_trace` is already a weight-space Fisher
trace. Served A/B at matched 4.75 bpp: Qwen3-4B KL −50.8% / PPL −15.1%; Qwen3-0.6B KL −58.5% /
PPL −24.4% (`run-pipeline.sh:191-200`). 27B-class confirmation is ladder debt.

The stratified per-expert subsample (`PRISMAQUANT_EXPERT_COST_SAMPLE`) is applied on this path
too since `79964de` — `_measure_production_render_dense` had the `resolve_cost_target_name` fix
but not the subsample, so under the *default* cost mode the lever silently did nothing on
exactly the models that need it (DSv4: 256 experts × 3 projections × 43 layers). Split at
`incremental_measure_quant_cost.py:291`, extrapolated at `:421`, filled before the `render_path`
stamp so extrapolated rows still carry production provenance. Export still quantizes every
expert.

### 4.3 AURA — the default cost mode (`COST_MODE=aura`)

`aura_cost.py`. Cost is the KL-adjoint inner product with the production-rendered weight error
(`:5-14`, impl `:695-725`):

```
predicted_dloss[i,f] = 0.5 · mean_k ( <gW_i^(k), dW_{i,f}> )²
gW_i^(k) = ∂/∂W_i fisher_probe_scalar(logits; seed=k)   # KL/GN Fisher, rademacher
dW_{i,f} = Q_f(W_i) − W_i                               # production-rendered
```

The probe is `kl_fisher.fisher_probe_scalar` (`kl_fisher.py:77-131`). `dW` provenance is
recorded per row as `rendered` vs `rtn` (`aura_cost.py:195-234`) — immaterial at fp4, decisive
at fp8 (+36% served KL under RTN dW); `--require-production-cache` makes a missing rendered row
fatal and the pipeline always passes it (`run-pipeline.sh:886`). Passthroughs are zero-cost by
construction (`aura_cost.py:_ZERO_COST_FORMATS`). Every routed expert declared by the resolved
`ModelProfile` is hard-excluded, independent of whether its physical weight is a packed 3-D
Parameter or a per-expert 2-D `nn.Linear`
(`routed_experts.py:ProfileRoutedExpertClassifier`,
`aura_cost.py:_guard_packed_expert_coverage/_target_linears`). The classifier routes the
decision through `packed_expert_format_group()` after the profile's name mappings and validates
the projection vocabulary through `packed_expert_projection_names()`,
`unpacked_expert_projection_names()`, and
`vllm_fused_moe_scheme_projection_names()`; missing, malformed, conflicting, or throwing
profile answers are fatal rather than an empty success. The pipeline passes
`--allow-packed-expert-omission` and covers the omitted routed rows in `[2d]`. Three sub-stages:
`[2b]` format-menu cache for dW (`run-pipeline.sh`, AURA `[2b]` — under
`validated-surrogate` this *is* the frontier cache, per the one-cache principle), `[2c]`
`aura_cost`, `[2d]` hybrid finalize.

On the streamed CB path, `[2b]` and the scalar consumer are coupled by the synchronous
`ProductionWeightCache` consumption protocol in §5.4: AURA reduces the live canonical render
to its row before the pair can be discarded. The resulting cost table must cover the same
full format menu as a materialized run. The later assignment/export pass remains
assignment-scoped and retained; it re-renders only the selected pairs and must match each
scored pair's canonical tensor digest before the bytes are accepted.

**Empirical expert costs** (`expert_empirical_cost.py`) exist because AURA's smooth cost is
route-flip-blind on routed experts (Spearman 0.45→0.35 under faithful dW; predicted NVFP4/FP8
ratios 2–49× vs measured 1.1–1.5×, module preamble). The unit is every tensor in one
profile-declared serving-format group (vLLM FusedMoE must share one format); unit cost is
end-to-end mean-token `KL(BF16 ‖ unit-quantized)` split across allocator rows ∝ `n_params`
(`measure_expert_unit_costs`). Packed models retain their existing direct-stack renderer.
For live per-expert Linears, the empirical path validates contiguous expert/projection coverage,
virtual-packs the rows in `packed_expert_projection_names()` order, quantizes the same full
stack spelling export uses, scatters the rendered slices into the live model for KL, and restores
the originals exactly (`_unpacked_expert_units`, `_virtual_packed_module`,
`_unpacked_unit_kl`). CB col weights use the exporter's `_packed_expert_col_weights` pooling
rule; a missing member vector is fatal, and the AURA `[2d]` invocation receives the same
`--col-weights` argument as its cached-menu render. FP8 stays in the expert menu by standing
decision — no hardcoded ban; the DP plus real KL rejects it. CB families render the whole stack
in one qdq call, with opt-in holdout-gated RD-law ladder interpolation
`D(k)=C·2^(−k/4)`.

#### Platform-agnostic anchored cost; DSv4 CB acceptance driver

The dependency direction is **evaluate → price → allocate**, with a machine-specific
**map plugin** supplying the vocabulary that pricing needs. `prismaquant.anchored_cost` owns the
format-blind mechanism: scalar production anchors, identity-bound resume, shape fitting,
same-segment extrapolation, hull computation, and exposure reporting. A mapping plugin obtains
format family from `format_registry`, consumes exact source-gated payload rates from the shared
allocator legality path, obtains role/unit structure from the active `model_profile`, and declares five facts: the source-gated candidate ladder, the
shape-transfer equivalence partition, the production renderer hook and arm identity, the anchor
rung policy, and extra provenance identity fields. The core does not import or enumerate CB,
GGUF, NV, MX, or FP vocabulary.

The shape-transfer equivalence class is load-bearing. The generic segment key is
`(family,role,equivalence_class)`, and the core refuses to fit or apply one curve across two
classes declared by the plugin. CB maps the class to its bundle-authoritative codebook basis;
ordinary single-basis families can declare one trivial class, while a future platform can
declare a different partition without changing the mechanism. This turns the learned/lattice
seam defect from a campaign convention into a checked plugin contract.

`prismaquant.dsv4_aura_cb_reprice`, launched by the frozen
`tools/run_aura_cb_reprice.sh`, is the acceptance driver wiring the DSv4 profile, the CB mapping plugin,
and the **112.690 GB exact-byte budget** into that mechanism. It remains a one-shot campaign
rather than a four-phase `run-pipeline.sh` cost mode: rank the weights, solve once, and export
the resulting assignment blind. There is no contested set, certificate, or cost-driven
iteration. The only quality gate is the served artifact: all-position top-8192-plus-tail-bucket
vLLM KL-vs-BF16 plus direct WikiText PPL against `artifact-112p69-raw` at matched bpp. Qwen3.8-27B can reuse the
same generic mechanism and CB plugin while supplying its own model profile, source-gated unit
classes, budget, and acceptance driver.

On a single Spark, this driver hard-caps the streamed source `LayerCache` at one decoder layer
and disables lookahead prefetch. The worst routed layer subtracts each production anchor from
its source weight in FP32, stores the resulting `dW` in BF16, and upcasts that `dW` for the
load-bearing FP32 gradient projection (`aura_cost.run_streamed_production_anchor_aura`). The
production renderer's identity-bound transient-consumer seam hands one canonical CPU anchor at
a time directly to that subtraction; it never materializes a complete layer anchor mapping.
The complete BF16 `dW` plane remains resident for all probes. Post-accumulate hooks project and
clear each fully accumulated parameter gradient immediately instead of retaining a second
source-sized gradient plane, each probe's outgoing boundary cotangent replaces its consumed
incoming tensor in place, and dead activation boundaries and CUDA blocks are released
progressively. The normal CUDA caching allocator remains enabled inside each layer's VQ
render loop so temporary matmul buffers are reused; after the final transient anchor, the
driver synchronizes and calls `empty_cache()` before backward. The base campaign image still
carries a historical `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, so the DSv4 launcher explicitly
overrides it to `0` and asserts that value inside the container rather than turning every inner-loop allocation into a driver
`cudaMalloc`. A 2026-08-12 diagnostic attempt with caching disabled produced no durable layer
after 21 minutes and averaged only about 26% GPU utilization; a privileged stack sample placed
the active thread in `nvfp4_cb_formats._vq_assign` below `cudaMalloc`. An operator override
cannot reintroduce a multi-layer source cache into this campaign.

The producer environment is also a resumable input, not a local tag convention.
`tools/run_aura_cb_reprice.sh` defaults to the immutable
`gridbook@sha256:f7dad9260fea6f4207bd894acc9ebc034d91c599a70489a89ab1938a75db9c47`
campaign image, rejects every mutable tag, resolves the reference to a full Docker image ID
once, and launches by that ID. Before the first container of a future campaign,
`tools/prismaquant_runtime_snapshot.py` uses `git archive` to materialize the exact clean,
reviewed HEAD under a commit/tree-addressed local cache. Its manifest inventories and hashes
every tracked regular file and symlink, not only the importable package. The cache publisher is
atomic and serialized; an existing entry is always re-hashed before reuse. The launcher verifies
the complete closure on the host, mounts that standalone snapshot at `/pq:ro`, and passes its
commit, tree, closure hash, and PrismaQuant package-source hash into the container. There the
snapshot helper replays the complete closure check and `tools/container_runtime_identity.py`
proves both the package hash and Python import origin, with user-site/current-directory import
fallbacks disabled, before the same shell process immediately execs the DSv4 producer. The
dense path repeats that complete boundary immediately before each of its two producers and
execs the terminal one. Thus neither a changing live worktree nor an old site-package install
can enter the multi-hour measurement window.

The existing resumable identity semantics remain unchanged:
`tools/container_runtime_identity.py` atomically binds the checkpoint tree to the image
reference and ID, reviewed PrismaQuant commit and complete package-source hash, and external
implementation-receipt hash. A nonempty legacy checkpoint tree with no identity is refused;
an existing identity must match exactly. Replay, export, and gold measurement reuse the same
content-addressed snapshot boundary. Gold leaves `PYTHONPATH` absent: the tracked
`tools/prismaquant_source_bootstrap.py` accepts the already-verified snapshot root as a
transport assertion, requires Python safe-path mode, proves that the bootstrap itself and
`prismaquant.__init__` share that exact root, and only then adds it to `sys.path`. The same
bootstrap runs the shipcard module, so neither GPU measurement nor receipt filling can fall
back to an image-installed PrismaQuant package. The already-running 2026-08-12 acceptance campaign
predates this generic identity file and remains bound by its external commit/image receipt; it
is deliberately not retroactively migrated.

The first FP32-storage launch reached 474 MiB `MemAvailable` and the host 3-GiB safety guardian
killed its container at 2026-08-12 11:44:30 EDT (`/var/log/gpu-guardian.log`); BF16 delta storage
removes about 25.8 GB from that exact live set. A subsequent no-cache launch still entered
backward with a 25.62-GiB `dW` plane, could accumulate a 12.18-GiB parameter-gradient plane,
and retained incoming cotangents while growing outgoing cotangents; it fell to 6.9 GiB available
before its controlled stop at 13:28 EDT. Immediate gradient harvest plus in-place cotangent
rollover remove about 20 GiB from that reverse peak. These changes alter storage and lifetime,
not the FP32 subtraction or gradient/`dW` dot. Gradient-harvest, cotangent-rollover, boundary-
release, and transient-consumer identities are bound into the restart journal, so an older
journal cannot silently resume under the new scheduler.

**AURA is the campaign's one cost currency.** Weight MSE and activation/output MSE are
degenerate projections of the same weight error, not parallel allocator terms: `gW` already
contains the input activation and downstream backpropagated sensitivity. Consequently the
campaign never adds `cw_m2`, imatrix dispersion, `weight_mse`, or activation MSE to
`predicted_dloss`. `cb_col_weights.pkl` still matters, but only as the production renderer's
imatrix input, part of render identity, and an input copied into the exportable artifact. A
panel may use weight-MSE *ratios* to diagnose or fit ladder shape as described below; no bare
weight-MSE value can price a unit. Likewise, `predicted_dloss` already contains the KL Fisher,
so extrapolation must not multiply by `h_trace` or any second sensitivity term.

For unit `i`, its production-rendered anchor `K_hat` supplies the measured level and the
within-segment ladder supplies only a ratio:

```
cost(i,K) = predicted_dloss(i,K_hat) * g[family,role,equivalence_class](K)
                               / g[family,role,equivalence_class](K_hat)
```

The anchor key is every legal **`(unit,family,equivalence_class)`**, while fitting, application,
and provenance use the stricter generic segment key
**`(family,role,equivalence_class)`**. In the CB plugin the equivalence-class values are the
authoritative `learned` and `lattice` basis labels. The DSv4 census is 33,325
NVFP4-lattice anchors (K12–K18 is legal for every unit), 33,325 FP8-learned anchors (experts
K28–K33; nonexperts K28–K46), and 301 FP8-lattice anchors (nonexpert K47/K48 only): **66,951
production renders before panel and validation renders**. Experts stop at K33 under the exact
source-rate ceiling and therefore have no FP8-lattice segment. Every unit retains its exact
source terminal in `UnitSpec` and render identity, but retention is not allocator admission:
only a terminal whose registered activation path is identity may receive the constructed
zero-cost row described below.

No `g` fit or application may cross a family or a plugin-declared equivalence boundary. In the
CB mapping, that means no transfer across the learned/lattice seam. In particular,
FP8-learned and FP8-lattice are separate vertical levels even though both spell `FP8_CB`; the
DP compares their separately rendered per-unit anchors. `codebook_source_by_format` in the
immutable bundle is the rendering authority for that split. A family-level segment that joins
K28–K48 is invalid, as is using a learned anchor to normalize K47/K48. This is a structural
response to measured cross-basis direction rotation, not a tolerance around it.

P0 streams the checkpointed KL-adjoint with global Fisher normalization. P1 fuses the fixed
production arm into the same one-layer reverse window: render the layer's legal anchors,
form each `dW` with FP32 subtraction and BF16 storage, reduce it with an FP32 gradient/`dW`
dot product to its FP32 AURA scalar, durably acknowledge it, and discard the tensor before the
layer unloads. RTN is not an anchor substitute. Per-unit checkpoints are
SHA-256/qname keyed, atomic, and identity-bound to the model, complete legal menu and format
plan, production arm, learned/lattice bundle map, calibration/probe contract, renderer and
`cb_col_weights` input. Resume trusts names and identities rather than list position and
refuses any mismatch. For routed learned cells, every bundle bank origin must cover exactly the
43×3×6 DSv4 `(layer,projection,K28..K33)` coordinates and carry the SHA of the supplied routed
selection; stamping an unrelated coverage-valid selection beside different bundle books is a
preflight and driver refusal.

The measured-output scope and extrapolation-input scope are distinct in provenance. The former
contains only the sparse anchor/panel/holdout cells that actually produced `dW`. The latter binds
the exact source tensors, imatrix, codebooks, and production arm for the complete legal ladder so
the allocator/exporter can reproduce whichever extrapolated rung the DP selects; it explicitly
states that those outputs were not materialized. Global renderer and transient-consumer
identity is hashed once per streamed payload and nested in the per-unit scalar journal. Because
the production-anchor consumer publishes no durable pair sidecar, it deliberately skips the
otherwise-required full canonical-tensor receipt hash; hashing a throwaway 10-GiB-class tensor
would add CPU and UMA bandwidth without creating evidence that survives the call.

The pinned DSv4 calibration has 51 projection units belonging to 17 never-routed experts.
Those units are not inferred merely from absent activation files: the driver requires exact
equality among the activation-cache misses, profile-declared routed-expert names whose probe
records have `n_tokens_seen == 0`, and the names in the imatrix provenance sidecar. The rule,
name set, and sidecar SHA are part of the production-arm identity. The existing cold-expert
renderer branch then emits the same imatrix-weighted production render used by export when no
activation rows exist; it is neither RTN nor a borrowed sibling cost level
(`dsv4_aura_cb_reprice._validated_cold_expert_provenance`;
`streaming_production_cache.StreamedProductionAnchorRenderer`).

P2 fits shape on a bounded, rank-identifiable panel independently within each
`(family,role,equivalence_class)` segment. Existing p7 weight-MSE rungs may supply lattice-only shape;
learned segments require the fresh production-arm panel. On the panel, fitting the ratios once
in weight-MSE currency and once in AURA currency is a direction-stability diagnostic, and the
disagreement is reported. A disjoint holdout renders at least two rungs inside the learned
basis and reports predicted-vs-measured AURA dex error against the 0.05 reference bar. These
reports do not gate or rewrite the allocation, and a bad result does not trigger an automatic
cross-basis substitution or full-menu fallback.

The DSv4 policy now instantiates 32 fitting units per each of seven roles at four
NV-lattice and four FP8-learned rungs: `7×32×(4+4) = 1,792` logical panel
cells. FP8-lattice has only one legal on-law rung (K48), so it is priced from
its own anchor rather than pretending a one-coordinate segment has a fit. The
disjoint learned-basis holdout remains `7×4×2 = 56` cells. Panel/anchor overlap
removes `7×32×2 = 448` duplicate physical renders, so the complete bounded
union is `66,951 + 1,792 + 56 - 448 = 68,351` production renders, versus
334,454 legal allocator cells on the source-rate-restricted on-law menu.
`dsv4_aura_cb_reprice.render_economics_report` is the numeric authority: it
scales each physical cell by exact probe `n_params` in 2048×4096 equivalents,
uses measured timing where available and explicitly labelled next-rung-up
proxies for untimed K32/K40/K44, and reports the measured-phase 32-probe P0
projection. The campaign has not completed, so no fixed GPU-hour total is
claimed here. Its output `campaign_report.json:economics` records the current
projection and limitations. The scalar-checkpoint/cost/export layout persists
no rendered weights; its disk projection charges one filesystem block per
physical scalar, legal cost cell, and source-plan unit plus one imatrix copy,
while explicitly declining to call variable pickle/JSON and Pareto payloads a
proven upper bound (`dsv4_aura_cb_reprice.render_economics_report`).

P3 recomputes each segment's lower convex hull from that run's fitted `g`; no Track-A hull is
hardcoded. Hull removal is the only authorized candidate exclusion because an interior
`(bits,g)` point cannot be optimal under the anchored positive-level factorization. Render
budgets never truncate the legal menu. The campaign then runs one exact-byte DP and emits a
**new**, atomically identity-bound directly exportable artifacts directory containing the
AURA-stamped `layer_config.json`, the same render-input `cb_col_weights.pkl`, the allocator's
`pareto.knees.json` bpp-accounting sidecar, and matching `selection.json` with at least
`feasible`, `chosen_achieved_bits`, `predicted_dloss`, and `budget_bytes`; it never overwrites
the Track-A comparison artifact. The DSv4 export driver consumes this publication, not the raw
allocator directory, and verifies all four output digests before taking the GPU lock. Its
route-pending pre-check unions the selected assignment with the exact header-discovered DSpark
construction overlay, whose fixed units do not appear in the allocator keyspace.

The completed streamed pass can be hardened without a second model load or GPU measurement.
Replay admission is not inferred from a checkpoint count or an inactive systemd unit.
`tools/wait_dsv4_aura_campaign.py wait` first re-executes from a complete
content-addressed snapshot of one clean release commit, subscribes to the already-active
`pq-aura-dsv4-streamed-cached.service`, and binds its `MainPID`, `/proc` start time, and
`InvocationID` without starting, stopping, or restarting it. It requires that same non-restarting
invocation to terminate with systemd `Result=success`, `ExecMainCode=CLD_EXITED`, and
`ExecMainStatus=0`. Only then does it audit the exact 33,325-file manifest closure, monolithic
payload scope, and exact payload-byte equality for all 775 units in each historical layer
42 through 38. A no-clobber, canonical self-hashed
`artifacts/campaign_completion_receipt.json` binds that closure to the waiter snapshot. The
receipt lives in the campaign's operator-writable artifacts directory; the root-owned,
read-only checkpoint journal remains untouched.
The activation-safe replay requires the receipt's producer commit to equal its own immutable
runtime commit. Before receipt admission, the replay module also requires Python safe-path mode
with no `PYTHONPATH` or bytecode writes, proves its own `__file__` is inside the selected
runtime-source snapshot, and re-hashes that snapshot's exact commit, tree, and full tracked-file
closure. Those three identities must also equal the completion receipt's producer snapshot; a
caller-supplied commit environment value alone is therefore not replay authority.
Replay then cross-checks its independent deep reconstruction against the receipt before the CPU
tail may run (`dsv4_campaign_completion`, `dsv4_aura_cb_reprice._release_runtime_commit`).
`--replay-streamed-payload` accepts only this work directory's completed
`artifacts/streamed_anchor_aura.pkl`, then independently reconstructs every measured scalar
from the SHA-256-bound per-unit AURA journal. It verifies the manifest and campaign identity,
complete unit/chunk scope, shapes, source-weight identities, calibration, format/purpose plan,
renderer arm, and payload/envelope digests; missing, extra, changed, or cross-campaign state
fails closed. Historical synthetic terminal-zero rows are admitted only in their exact legacy
shape and are **quarantined**, never copied into the new cost table. The CPU tail refits,
reprices, runs the one exact-byte DP, and publishes separately under
`artifacts/replay-activation-safe`, `allocator-aura-activation-safe`, and
`artifacts/exportable-aura-activation-safe`, stamping `measurement_invoked=false`, the source
payload and journal identities, quarantine counts, and `no_gpu_measurement_or_render=true`.
The original streamed payload and pre-hardening publication are not overwritten
(`dsv4_aura_cb_reprice._load_and_audit_completed_streamed_payload`,
`run_dsv4_anchor_replay`).

**Anchored-AURA allocator admission — CLOSED 2026-08-11.**
`allocator_candidates.cost_entry_is_anchored_aura_supersurrogate` identifies an anchored row by
three independent stamps — `cost_currency = aura_predicted_dloss`,
`cost_source = production_arm_render`, `fisher_application_count == 1` — in the forgery-refusing
style of `cost_entry_is_source_passthrough`, and
`AURA_SUPERSURROGATE_ALLOCATOR_SEMANTICS = True` declares the branch exists. Four behaviours:
the value is read directly (no P5a transfer), the row is kept out of the activation-calibration
sample, a measured zero is retained instead of being removed as `activation_cost_unmeasured`,
and the row is stamped `anchored_aura_extrapolation`. No epsilon floor, fabricated `output_mse`,
or rewritten probe `h_trace` was used; the zero-guard bypass is scoped to matching rows, so every
other cost table keeps it at full strength.

**The word "activation-inclusive" was wrong and has been retired here.** An earlier draft of this
section claimed anchored AURA is an activation-*inclusive* supersurrogate. It is not.
`aura_cost.py` runs its KL-adjoint on **unquantized** boundary activations and `dW` is a weight
delta, so no activation-quantization error enters the number. AURA is activation-**weighted**
(`gW` carries `X` — the alignment term its measured win lives in) and
activation-quantization-**blind**. "Supersurrogate" remains correct as a **currency** claim: one
projection replaced the two-factor magnitude score (`h_trace × output_mse`, `h_trace × cw_m2`)
that preceded it. It is not an error-model claim.

**Activation-safe terminal admission is gated.** DSv4's routed-expert terminal
`MXFP4_SOURCE` preserves both the source weight and activation contract. Gridbook's
dedicated `Fp8SourceW8A16LinearMethod` (added in 0.8.5, carried by the pinned 0.8.11)
gives `FP8_BLOCK_UE8M0_SOURCE` the same numerical
property: its block-128 E4M3 weight plane and one-byte UE8M0 scale plane remain resident and
byte-verbatim while BF16 activations pass unchanged. The registry therefore declares
`act_bits=None`, identity activation QDQ, and the distinct
`gridbook_fp8_source_w8a16` route. The 301 source-eligible nonexpert terminals are honest
constructed-zero candidates; the approved 112.690 GB allocation selects 120. The direct
`MXFP8_UE8M0_G32` re-encode remains W8A8 (`act_bits=8`, group 32,
`GRIDBOOK_MXFP8_DENSE=1`) and is not a substitute for, or an alias of, that source route.

The numerical re-admission and the serving promotion are independent gates. The fixed
`--w8a16-readmission` path is CPU-only, fresh/no-clobber, consumes only the exact allowlisted
completed producer/receipt/journal, permits only the audited 0.8.4→pinned-release format-plan
semantic delta (0.8.5 when written, 0.8.11 since 2026-08-21 — the current side reads the pin
constant, the recorded historical side stays 0.8.4),
reconstructs all measured rows from unit receipts, and re-admits only the historical
block-source terminal zeros. It reruns the DP and requires full qname→format equality plus the
canonical assignment digest and exact selection metrics before atomically publishing a new
AURA-stamped directory. The allocator subprocess re-enters through the same verified source
bootstrap with `PYTHONPATH` removed, so it cannot fall through to the image's older installed
PrismaQuant; ordinary installed-wheel runs retain their normal module entrypoint. Generic replay
remains same-snapshot and activation-safe. The exact
Gridbook pin (0.8.11 since 2026-08-21) plus the 91-test installed-wheel GPU gate measured on
0.8.5, the release that first attested the feature, back the source W8A16 route for
export without a route-pending acknowledgement. The tracked pre-export handoff still refuses
an unresolved/unreleased pin, any pending route, changed source/bundle/publication bytes,
changed bytes in the reviewed exporter/source semantic closure, or an existing output. The
new `prismaquant.dsv4_w8a16.export_handoff.v2` receipt carries the closure schema, exact
per-file SHA-256 map, and canonical closure identity above. It supersedes v1 only as current
handoff authority; it does not rewrite or reinterpret any dated historical receipt.
Readmission does not itself imply GPU measurement or export; full-artifact native parity
remains a post-export shipcard gate
(`dsv4_aura_cb_reprice`, `dsv4_w8a16_export_handoff`).

**Residual CB-family activation blindness (reported; terminal shortcut gated).** Every rung of `nvfp4_cb` and
`fp8_cb` has `act_quant_changes_input = True`, and an anchored table carries no measured
`output_mse`, so P5a has no calibration sample and `penalty_for` already returns exactly 1.0 —
skipping it is a provenance statement, not a number change. Two facts bound the exposure. The
activation path is **constant across K within each CB family**, so the blindness cannot reorder
rungs *inside* a family; it can only shift the `nvfp4_cb`-vs-`fp8_cb` family-choice margin. And
AURA's validated wins (−38%/−39.5% @4B, −17.9% @27B on served KL) were measured **against**
`h_trace × output_mse` — a baseline that *did* carry the A side, since `measure_quant_cost`
applies `activation_quantize_dequantize(X)` — on menus already mixing W4A4 NVFP4, W8A8 FP8 and
BF16, i.e. the same family-choice margin. This is carried exactly like the route-flip limitation
below: named, reported, with the served A/B as the arbiter.

**Standing routed-expert limitation.** AURA measures smooth local weight damage but is blind
to route flips (`expert_empirical_cost.py` module contract). Routed-unit discovery/classification
is owned by the active model profile; that model-profile axis is unchanged by the mapping-plugin
refactor. On this CB lane the empirical
alternative remains default-off (`CB_EXPERT_EMPIRICAL=0` in `run-pipeline.sh`), because
empirical unit-KL was refuted at CB fidelity by the BF16 chaos floor. Anchored AURA therefore
improves the local MSE previously used for routed experts, but it does **not** model routing
discontinuities; activation weighting inside `gW` must not be presented as route-flip coverage.

**UCB — two of them, both default-off, neither set by the pipeline.** Cost-side
`PRISMAQUANT_COST_UCB_Z` adds `z·stderr` before the DP (`allocator_candidates.py:357-370`);
`z=0` is bit-identical to no-UCB and it only bites on the `predicted_dloss` branch (AURA /
expert-empirical), not `output_mse`/`weight_mse`. Selection-side `--kl-ucb-z` yields
`kl_ucb = mean + z·stderr` over calib repeats (`validate_assignments_kl.py:640-660`), consumed
by `select_validated_frontier --metric ucb`. Both are **research-only** as of R28
(`docs/design/runtime_flags.md` §1): the one measured win (`z=2`, −8.0% on the 27B old-vs-new
AURA A/B) is a thin-calibration result, and at production calibration the hedge moves 6/252
rows to served parity — hence default-off, no driver, and the standing decision to keep `z=0`.

**The additivity report — stage `[3c]`, wired 2026-07-30** (re-vet R2 precondition (ii); the
R2-vs-R19 disagreement was resolved as *wire it*, not *wall it*). AURA's one structural
assumption is that per-Linear KL contributions add. `aura_additivity_gate.py` records
`residual = measured_end_KL(assignment) − Σᵢ predicted_dlossᵢ` with an honest stderr — exact
per-probe when the rows carry `x2_per_probe` (probe-aligned raw samples, so the correlated sum
is computed rather than approximated), else the independence lower bound, and it says which.
It is a **report**: never blocking, never touching an allocation. It runs after the assignment
is final (that is why it is `[3c]` and not `[2d]` — no assignment exists at `[2d]`), and its
result is stamped into `cost.pkl`'s `provenance["additivity"]`, so every artifact derived from
that table carries the trust-region number. **`AURA_ADDITIVITY_GATE=measure` is the default
since 2026-07-30** (Robert's ruling on the R2 residue): it reports from a measured KL the run
already produced when there is one (validated-surrogate's frontier JSON, free) and otherwise
runs **one bounded KL eval** of the final assignment against the same format-menu dW cache AURA
costed on — so every AURA-default run performs the measurement and **every artifact carries a
real residual**. The wiring's one weak spot was that under `SELECTION_MODE=surrogate` an
artifact carried a prediction and no residual, leaving AURA's structural assumption a two-model
memory instead of a per-artifact number; the ruling closes it, at the price of one bounded GPU
eval per run. `auto` (the pre-ruling behaviour) stays selectable and is zero-added-GPU: it
reports only from a measurement the run already made, otherwise recording the predicted sum with
`measured_kl: null` and a status naming the reason. `0` disables. Either way it is a report —
non-blocking, never touching an allocation.

#### RTX4090 four-render specialization of anchored AURA

`prismaquant.rtx4090_fp8_burn` specializes the same anchored-cost mechanism for the strict
18,000,000,000-byte Qwen3.8-27B Ada artifact. One immutable campaign plan binds the probe,
full-corpus imatrix/column-weight values, source checkpoint identity, producer snapshot,
execution attestation, dataset, and complete 496-Linear body census. The balanced two-host plan
keeps whole decoder layers intact: layers 0–31 and 32–63, 248 Linears per host. That gives
exactly 992 physical renders per host and 1,984 globally—not a twelve-rung sweep.

The burn's `derive-col-weights` entry point first validates the committed sample merge bundle
as one object, consumes the exact captured in-memory probe that was hashed, derives the ordinary
raw FP32 imatrix map, requires its exact calibration identity, and publishes durably without
clobbering an existing or racing output. `attest-execution` then closes the sample execution
identity over the live producer snapshot and the launcher-owned registry RepoDigest. Both `prepare` and
each GPU-bearing `measure` replay that attestation, host-local streamed source identity, portable
source census, merge bundle, imatrix, and snapshot before requesting CUDA.

The plan, checkpoint identity, and runner all bind the same streamed-source settings: the
existing `LayerCache` has two slots, the effective prefetch lookahead is one, and
`require_prefetched_residency` is true. That final field is an explicit runner policy, not an
inference from nonzero lookahead; generic streamed runners retain the backward-compatible false
default. It fail-closes both forward boundary capture and AURA's reverse traversal to resident
or completed-prefetch data. Forward capture leaves the tail window resident; immediately after
each reverse install, the runner schedules the next lower layer through `StreamingContext` so
its shard read overlaps the current layer's render and backward work. If an entry is absent
after its required prefetch (including a refused insertion), the campaign raises before the
synchronous cold-load path. No campaign-specific cache or residency mechanism exists.

Every body Linear renders only lattice `FP8_CB_K4`, `FP8_CB_K16`, `FP8_CB_K48`, and delegated
native `FP8_E4M3`; BF16 is an unrendered exact source-passthrough terminal. K16 is the production
anchor, while K4/K16/K48 are the fitting panel. The global shard merger reconstructs its
receipt from the exact disjoint qname cover; validates the full sparse format plan, purposes,
renderer, source weights, imatrix, serialized-payload context, and arm identity; and labels
worker chunk counts as unverified planning metadata rather than execution evidence. No native
or codebook NVFP4 name is legal at any campaign boundary.

The final allocator table byte-preserves all four direct AURA rows. The anchored lattice fit
imputes only `FP8_CB_K8`, `K12`, `K20`, `K24`, `K28`, `K32`, `K36`, `K40`, and `K44`; it never
relabels an imputed row as measured. Weight MSE exists exactly on the three CB panel cells as a
fit diagnostic and is never an allocator currency. The fitted hull report describes the
pre-overlay imputation law, not the final mixed direct/imputed/source-terminal table. Exact
candidate bytes come from the production serialized-footprint primitive, including format
payloads and sidecars, before one solve targets the complete-publication ceiling.
`PRISMAQUANT_CB_ENCODE_COMPILE=1`, `PRISMAQUANT_CB_ATOM_COMPILE=1`, and
`PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1` are mandatory, identity-bound producer inputs. Live
scorer calls use full-graph Inductor with compiler suppression disabled and refuse unless every
call enters exactly one compiled backend dispatch. Resume-restored units remain admissible only
through AURA's already checksummed/deserialized unit envelopes whose manifest identity binds the
same compile settings, producer arm/snapshot, and streamed source; each shard receipt separates
live and restored coverage. Because serialization fixes `ldlq=false`/`ldlq_scope=none`, the atom
route is stamped `not_applicable` rather than falsely claimed as executed. These producer proofs
do not weaken or replace the existing physical RTX 4090
eager and mandatory full-graph serving gates; validation on GB10 remains explicitly
`UNRELEASABLE_VALIDATION_ONLY` (`cb_compile_contract.py`,
`rtx4090_cb_compile_proof.py`, `cb_anchored_cost.py`, `rtx4090_fp8_burn.py`, §7.1, §9.2).

#### Dense CB sibling driver, and where AQUA enters the anchored lane

`prismaquant.dense_anchored_cb` is the second caller of the same mechanism, for a
**dense** model on the CB lane (first use: the Qwen3.8-27B 16 GiB gridbook artifact).
It is a sibling of `dsv4_aura_cb_reprice`, never a refactor of it — that module is
frozen so a shipped campaign replays byte-for-byte, and its routed/dense expert split,
learned FP8-CB basis, two-rung routed ladder and W8A16 readmission lane have no dense
analogue. The dense driver supplies only census, ladders, panel/validation policy,
anchors and budget; all pricing stays in `anchored_cost` / `cb_anchored_cost`.

Three dense-specific differences are structural rather than cosmetic:

* **One source-payload class, so no `SourceClassFormatPlan`.** That planner splits the
  menu into lower- and higher-rate classes and requires the expert menu to be a *strict
  subset* of the nonexpert one; a dense model has nothing to split, and declaring a fake
  expert menu would put a menu in the artifact no unit can take. `DensePlan` derives both
  ladders from `format_registry` narrowed by the explicitly selected serving profile's
  production format rule
  (which used to drop the research-only signed `NVFP4_CB_S13..S16`; that family was
  **deleted 2026-08-17**, so there is no signed rung left to drop). It does **not** narrow
  either family by `_serving_backed_family`: fused mid-M is a performance route for one
  batch regime, not format admission, and its historical K28..K48 subset had incorrectly
  hidden public FP8 producer K4..K24 from AQUA. The selected profile and module constants
  must agree exactly or plan derivation refuses. `--target-profile` defaults to
  `qwen38_sm120_cb_validation_only`, and that identity plus exact `target_platform=sm_120`
  travel in `DensePlan`, the render arm, plan report, and allocator invocation. This proves
  structural candidate registration only; it does not replace the external release/device
  gate (§9.2).
* **One basis.** Learned codebooks are a measured null on Qwen dense (holdout ~1.00 across
  K28–K43), so the campaign runs `CB_CODEBOOK_SOURCE_SCOPE=none` and there is no learned
  segment. `_normalize_source_map` refuses "a global source scalar or K-range inference",
  which is correct when a bundle is in play; scope `none` is that refusal's exact
  complement — a declaration that *no* format takes a learned book, which entails the
  per-format map rather than guessing it. The driver builds the map from the scope
  explicitly and refuses every other bundle-less combination.
* **AQUA is applied, and the anchored branch is why it can be.** AURA is
  activation-quant-blind, so a pure anchored table prices `NVFP4_CB` (act_bits 4, group 16)
  and `FP8_CB` (act_bits 8) identically on the A-side — the whole decision a CB menu is
  made of. Since 2026-08-14 `allocator_candidates.cost_entry_predicted_dloss` reads the
  anchored branch as `base + cost_entry_act_dloss(cost_entry)`: the A-side is **added**,
  never multiplied by P5a's per-family constant, because that multiply is an estimator
  *transfer* from weight space to output space and an anchored projection is already in
  the right currency. So `PRISMAQUANT_ACTIVATION_FAIR_PRICING` does not gate AQUA here at
  all. (The DSv4 driver still passes that kill switch; it landed 2026-08-11, three days
  before AQUA-AURA, when P5a priced only the weight-only branches. Revisit it when DSv4 is
  re-run with AQUA.)

  The A-side reaches the rows through the separate `aqua_activation_cost` stage, merged
  between pricing and the DP. It needs **no render** — `activation_dloss` reads the dense
  `W[o,j]^2`, the card's `g_sq_sum` and the format's activation grid — so it layers onto a
  production-rendered W-side with no rendering confound, and `--act-dir` makes its
  `act_var` measured rather than modelled. The weight-only anchored payload is written and
  kept, so the pre-AQUA allocation stays reproducible as the A/B arm; **AQUA-on-CB is a
  candidate until that served KL/PPL A/B at matched bpp lands, not a result.**

The current menu is 37 rungs (NVFP4 K1..K25 plus FP8 K4..K48 step 4). On the known
496-body-unit Qwen3.8 census that is **18,352** full-menu cells and 992 per-unit family
anchors. Panel and validation counts are derived from the live role census rather than copied
from the earlier six-rung FP8 campaign: per role the driver measures 10 units × (7 NVFP4 + 3
FP8 panel rungs) and four disjoint units × (5 NVFP4 + 4 FP8 validation rungs). The plan report
records the exact resulting union and fraction before GPU work. All renders remain transient.
The stock `COST_MODE=aura` path would retain roughly 45.5 GB per rung on this model — about
1.68 TB for the public menu — for a cache the exporter never reads.

A validation cell holds out up to **two** axes, and only one of them is ever vacuous. At
the segment's **anchor** the fit reproduces the measurement by construction — the
prediction is `anchor × ratio(anchor, anchor)` = `anchor × 1.0` — so every such cell
reports dex exactly 0.0000 however bad the fit is. The first Qwen3.8-27B campaign
(2026-08-15) listed `FP8_CB_K36`, its own fp8 anchor, as a validation rung, and 48 of its
192 validation cells were therefore vacuous — visible in the plan report as
`deduplicated_multi_purpose_cells: 48`. **`cb_anchored_cost.plan_cb_panel_and_validation`
refuses that configuration for every CB driver.** The guard first landed in
`dense_anchored_cb._panel_policy`, but that module is a `__main__` with no importers, so
it did not cover the DSv4 lane, which builds its own policy literal and calls the shared
planner directly; keying the check off `plugin.anchor_formats` inside the planner covers
both paths and any future one by construction.

A validation rung the **panel** also contains is a different, weaker case and is *not*
refused. The shape at that rung was fitted from panel units, but the cell applies it to a
held-out unit's own anchor, so it genuinely tests cross-unit transfer — and on a two-rung
ladder whose second rung is the anchor (DSv4 routed experts: K28, K32=anchor) it is the
only validation that can structurally exist, so a hard refusal would ban validating routed
experts at all. Instead each cell carries `held_out_axes` (`["unit","rung"]` or
`["unit"]`) and `heldout_validation_report` reports `n_cells` / `n_above_bar` /
`max_abs_dex` **per axis** alongside the pooled figures, plus `off_panel_rung_evidence`.
A pooled statistic cannot distinguish a fit validated off the panel from one validated
only on it; that is the difference the split makes visible.

Current tables. Dense campaign v3: NVFP4-CB panel K1/K2/K11/K12/K23/K24/K25,
with validation at K3/K10/K14/K22 and K25 on held-out units. Its shape basis is
`(rung, parity, below-K12 hinge, above-K24 shoulder)`, so the historical
K12..K24 line is not silently extrapolated to either endpoint. K25 is the only
public rung above K24, so its term is a measured one-point shoulder, not a
claimed high-band slope. FP8-CB uses a K24 anchor, a K4/K28/K48 panel spanning the full
producer ladder, and off-panel validation at K8/K20/K36/K44. Its shape basis is only `rung`
because every producer rung is divisible by four; the four held-out cells challenge the
straight-line proposal on both sides without laundering the anchor into validation. DSv4:
NVFP4-CB at **K13/K17** — it previously had **no** validation at
all, leaving 233,275 of 334,454 legal DP cells (**69.7%**) priced by a fit nothing
checked; the total is the inventory's own census, 233,275 NVFP4 + 66,048 routed FP8 +
1,806 dense FP8 + 33,325 exact source terminals — and
FP8-CB learned at **K36/K40**, where K36 is the sole non-anchor rung outside the panel
(K48 is lattice, not learned) and K40 rides along on the unit axis, replacing a K28/K44
pair that was entirely panel-trained without saying so.

**Known hole, measured not re-architected.** A cohort is drawn only from units legal at
every panel *and* validation rung of their role, so on DSv4 the fp8-learned cohorts are
dense-only: the fit is trained on ≤32 and validated on 4 dense units, then applied to
~33,153 routed experts that are structurally excluded from both (they stop at K33). A
routed expert's whole exposure to the fit is one scalar — its own measured K32 anchor ×
the shared K28/K32 ratio — so ratio error is common-mode across all experts in a role and
shifts the K28/K32 boundary rather than the per-expert ranking, which the project's own
churn precedent (cost CV 23% → 3% churn → 0σ served) bounds. The DSv4 card must state how
routed experts were validated, or that they were spot-checked with N and dex quoted.

### 4.4 L2 and L3 — retired 2026-07-30 (`archive/l3_propagated_2026-07-30/`)

Both levels of the old cascade are **walled**, not merely off. Re-vet **R4**; the wall's
README carries the full lesson.

**L2 perturbed-X was never a cost stage.** No `COST_MODE` ever ran a re-measure/re-solve
loop; the accepted set is `local | production-render-score | aura` (`run-pipeline.sh`
`COST_MODE` case). `perturbed_x_cache.py` **stays live** — it is the activation-cache /
model-loading utility that `validate_assignments_kl`, `kl_measurement` and
`production_recache` depend on, and it always was.

**L3 propagated end-KL is walled.** `kl_sensitivity_probe.py` (3,678 L),
`propagated_sensitivity_costs.py`, `sensitivity_response.py`, the five
`--propagated-sensitivity-*` allocator arguments and the L3 half of `kl_measurement.py`
(97 top-level symbols, ~4.3k lines, now
`archive/l3_propagated_2026-07-30/prismaquant/kl_measurement_l3.py` — still importable
against the live tree) all moved. Setting `ALLOC_PROPAGATED_SENSITIVITY_REPORT` is `exit 2`
(§3.5).

**Why, in three measurements** — none of them an argument:

| Evidence | Result |
|---|---|
| `aura_cascade_headtohead` | L2 fixed point beats additive L1 by **−1.5%**; AURA beats L1 by **−38.5%** |
| `xlayer_sensitivity_2026_06_09` + `cross_layer_additivity_fp32` | pairwise residual **+5–12% and diffuse**, **3/1180** pairs significant; the apparent non-additivity is a **bf16 differencing artifact** — per-Linear KLs add in fp32 |
| §11 / `prismaclade_l3_non_additivity` | L3 costs measured under an L2 context do **not** sum when many flip at once, so L3's expensive measurement could not be composed anyway |

L3's only consumer (DP/coordinate-descent polish) was already archived
(`archive/polish_2026-05-15/`), and `kl_sensitivity_probe` had zero references in
`run-pipeline.sh` at wall time.

**What survives in `kl_measurement.py`** (1,206 lines, down from 5,731): whole-assignment
`measure_assignment_kl`, the per-sequence tail machinery (`sequence_token_nll`,
`summarize_per_sequence_kl`, `return_per_sequence`), `assignment_bit_total` /
`assignment_hash`, `l2_cost_value`, and the `CUDAGraphRegistry`. `validate_assignments_kl`
and `validation_harness` are unchanged.

### 4.5 Solver

**Profile pins, fixed `lm_head`, and `ALLOW_PINNED` (2026-08-23).**
`ModelProfile.is_pinned_name` historically force-excludes names like `lm_head` from the DP
budget and ships them at source dtype. That BF16 behavior remains the default. There are now
two explicit ways to lift only the structural output-head aliases while leaving every other
profile pin intact (`fixed_head.py`):

- `allocator --lm-head-format FP8_E4M3` is the production fixed-auxiliary policy. The
  allocator requires an untied head, a probe row, the corresponding measured cost row, and a
  candidate legal for the target serving profile. It fixes that candidate before the body
  solve; the head is excluded from body bpp, activation-fair pricing, body
  `predicted_dloss`, and the DP decision, but its exact payload and measured loss remain in
  `aux_fixed_assignment_payload_bits_total`, `aux_fixed_params`,
  `aux_fixed_predicted_dloss`, and `total_predicted_dloss_with_aux`. Whole-artifact byte
  selection therefore cannot hide it.
- `allocator --allow-pinned lm_head` remains the independent research mode: the head enters
  the ordinary multi-choice DP and is priced by budget-value. It too requires real probe/cost
  rows. A non-BF16 fixed head and this DP-unpin are a hard conflict; an output head that is a
  tied embedding cannot be fixed quantized because there is no independent source tensor and
  the embedding-side perturbation is not measured by the head cost (§7.5).

The allocator stamps `lm_head_format` and `lm_head_mode = profile_pinned_bf16|fixed|dp` in
its metadata. `run-pipeline.sh` uses the same resolved policy to expand `COST_FORMATS`, choose
the head probe/cost census, filter production-cache profile pins, and pass the exporter's
remaining `--ignore` names. `export_native_compressed` trusts that allocator metadata for
only the head aliases; an explicit operator `--ignore` remains highest precedence. It matters
at card scale: on Qwen3.8-27B a BF16 `lm_head` spans 2.543 GB, **20% of a 13.0 GB
whole-artifact budget**. The approved quantized head is `FP8_E4M3` on the native
compressed-tensors lane; no token embedding is quantized by this policy. A CB rung on
`lm_head` still exports cleanly and then **fails at load** — no Gridbook method claims
`ParallelLMHead` — so a CB artifact must delegate a quantized head to the stock CT path
(§6.2, §9.2; `allocator.py`, `run-pipeline.sh`, `export_native_compressed.py`).

`allocator_solver.py`. Multi-choice knapsack DP over average-bits-per-parameter bins,
numpy-vectorized (`solve_allocation :427-520`); the baseline per Linear is its cheapest
candidate, bins = `(target − min_bits)/bit_precision + 2`, and backtrack mirrors the forward
charge exactly. `_charged_bins` (`:409-424`) charges any strictly positive Δbits at least one
bin, so sub-half-bin upgrades are never free.

**The DP unit is the serving-atomic unit, not the Linear** (#17, `f719d93`). A packed-MoE
serving group is atomic at serve time — vLLM's FusedMoE loads every projection of every routed
expert in a layer under ONE scheme — so "upgrade one expert row" is not a real option, and
pricing it per-row while `promote_serving_units` charges the whole group is a ~1000× price
mismatch: mispriced expert rows top the per-bin ranking, feasibility tightening over-corrects,
and cheap dense rows starve. `aggregate_packed_serving_groups`
(`allocator_candidates.py:993-1174`) pre-aggregates each group into one multi-choice item whose
per-format cost and byte cost are the exact sums of its members, over the **intersection** of
member-legal formats; `expand_packed_group_assignment` (`:1176-1189`) broadcasts the decision
back for emission. Post-DP MoE promotion becomes a validated no-op. A group with no common
legal format falls back to individual rows and is then **not allocatable** — `compute_achieved`
raises rather than score the unpriced member at zero Δloss (which would make the illegal state
look cheapest to the min-Δloss ratchet). `--no-packed-aggregation` (`allocator.py:1281-1288`)
restores per-row pricing for back-compat experiments only.

**Serving-unit promotion is union-find, and legality-aware** (#28, `9b4347f`).
`promote_serving_units` (`:302-327`) unions fused-sibling and packed-MoE groups in one
order-independent pass; `_promote_group_components` (`:234-299`) chooses the component's
format via `_choose_group_format` (`:192-231`) from per-row legal sets derived from the
candidate lists (`legal_formats_from_candidates :103-116`) — the **cheapest legal-for-all**
format at or above the max rank, falling back to the highest legal-for-all only when nothing
above is common, and raising with every member's legal set when the intersection is empty
(`_serving_group_menu_error :152-189`). Before this, promotion took only `assignment`,
`format_rank` and `groups`, so it wrote the max-rank format blind to whether the rest of the
unit could carry it — members do not share a shape, and often it could not. The legality
argument is optional by design: omit it and the legacy two lines run verbatim, so hand-built
and auxiliary MTP/visual assignments cannot acquire a new failure. `promote_fused` (`:362-406`)
still hard-asserts post-promotion coherence. Non-regression: re-solving the shipped 27B and 35B
at `TARGET_BITS` changed 0 of 614 and 0 of 500 assignments.

**Termination is feasible-only, and solves are memoized** (#16, `8d3d0dc`).
`solve_with_promotion` (`:606-851`) contracts that the returned assignment is always feasible
(`achieved ≤ target + overshoot_tolerance`) and, among feasible iterates, the one with
**minimum total predicted Δloss** (ties → larger achieved bits). Δloss is the objective; density
is not a proxy for it — 5.5 bpp has beaten 6.0 bpp on served PPL. Three silent fallbacks are
gone: a `solve_allocation` returning None no longer yields the previous over-target iterate,
an arbitrarily deep undershoot is no longer accepted, and the stall exit no longer returns an
iterate above target. When no iterate is feasible within `max_iters=40` the rung is INFEASIBLE
and `(None, nan)` is returned so callers drop it from the Pareto curve. Search is damped
descent to the first feasible iterate, then bracket bisection with a min-Δloss ratchet
(promotion is a coarse step function, so `achieved(tightened)` is locally non-monotone). A
`diagnostics` dict is filled in place on every return path — `min_bits`, `evals`,
`closest_achieved_bits`, `floor_achieved_bits` — which is the only thing that makes an
INFEASIBLE verdict actionable. `PRISMAQUANT_SOLVER_TRACE` (`:37`) prints per-eval timing.
`allocator.py` memoizes the solve per target (`:1959-1982`): it is a pure function of the
target given fixed stats/candidates, and the byte-budget grid plus ratchet bisection re-visit
targets the Pareto sweep already solved. Callers get a **copy** of the assignment dict —
fused-sibling expansion mutates it — and per-target diagnostics are kept beside the memo so a
cache hit never loses them.

**Bit-exact re-encodes price at zero, but only on an identity activation path** (#20,
`5028fff`). `cost_entry_is_bit_exact` (`allocator_candidates.py:233-286`) short-circuits a
measured `weight_mse == 0.0` to `predicted_dloss = 0.0` — genuinely optimal when the format
stores the source weights verbatim (MXFP8 over an FP8 128-block source; MXFP4/6/8 over an
MXFP4-packed QAT source). But `W' == W` silences only the *weight* side: for W·A· formats the
cost pipeline quantizes activations before measuring, so a weight-lossless MXFP4 re-encode of
an MXFP4 source would price at dloss 0.0 — the unbeatable global minimum at any budget — while
serving 4-bit activations. The gate is therefore the dtype-level predicate
`FormatSpec.act_quant_changes_input` (`format_registry.py:75-106`: `act_bits` absent or ≥ 16),
not a heuristic; unregistered formats never short-circuit, and an entry declaring an explicit
`cost_source` (the production-render pipeline, whose `weight_mse` is a placeholder) is never
treated as bit-exact.

**A candidate may not be larger than the source representation it replaces.** This is a
default-on allocator legality invariant, not a cost-model preference and not an environment
toggle: for every unit with an exact shape and a source census, the integer comparison is
`candidate_payload_bytes <= source_payload_bytes` (equality is legal). The source kind resolves
through `SOURCE_PASSTHROUGH_CONTRACTS` to the registered format that owns a scaled physical
layout; ordinary FP16/FP32/etc. tensors retain their safetensors dtype and resolve through
`footprint.py`'s existing dtype-width authority. An explicit unknown, unreadable, or heterogeneous
owner is rejected rather than assigned a guessed scalar bpp (`source_footprint_owner_for_kind`,
`allocator_candidates.py`). Candidate and source bytes share `footprint.py`'s exact tensor
payload helpers, so registered formats use `memory_bytes_for_shape`, plain dtypes use their
serialized element width, and CB formats use their versioned serialization context, including
row scales/layout rather than the nominal `FormatSpec` approximation. The
per-unit comparison deliberately excludes shared/deduplicated sidecars, which remain charged
once by whole-assignment accounting. `_source_bpp_applicability`
(`allocator_candidates.py:392-469`) performs the exact-byte test before the candidate reaches
the DP; a legacy/offline call with no source census is explicitly `not_evaluated`, while a
present but unknown source kind aborts candidate construction before a unit can disappear.

Every elimination is auditable in `format_applicability.json`, not just counted in a console
line. Its `source_bpp_legality` provenance carries schema
`prismaquant.source_bpp_legality.v1`, the comparison and derivation rules, the no-census and
unknown-kind policies, an explicit evaluated/not-evaluated status, `eliminated_count`, and the complete sorted `eliminated_candidates`
records. Each `candidate_exceeds_source_bpp` record names the qname, shape, source kind/format,
candidate format, exact source/candidate payload bytes, floating bpp readouts, and their integer
bit numerators plus common parameter denominator (`allocator_candidates.py:442-467,1754-1842`;
report emission `allocator.py:2994-3033`).

**Opt-in gate_up/down role split** (#21, `237a029`). `--packed-role-split`
(`allocator.py:1289-1300`) keys each packed expert group as two per-layer serving units
(gate+up, down) by wrapping the profile view (`packed_role_split_profile`,
`allocator_candidates.py:1243+`), so DP aggregation and serving promotion stay consistent. It
**hard-errors** unless the resolved serving profile declares
`supports_per_role_expert_schemes` (`serving_profiles.py:399-405`, gate
`require_per_role_expert_scheme_support :636-674`). GGUF declares it — expert tensors are
stacked per projection, each carrying its own ggml type. vLLM's compressed-tensors packed-MoE
path does not: `CompressedTensorsMoEMethod` selects one scheme per FusedMoE layer, so a
role-split checkpoint is unloadable. Default off.

Candidate legality, passthrough integrity, cost-source precedence and fused-sibling aggregation
also live in `allocator_candidates.py`; the invariants they enforce are §6.4's.

### 4.6 Selection

`SELECTION_MODE` defaults to `surrogate` (§3.3): `layer_config.json` straight from the DP at
`--target-bits`, no real KL. The knee is the post-cliff log-error kneedle
(`allocator.py:212-220`); raw-linear and global-log knees are diagnostics.
`_rd_curve_diagnostic` (`:285-338`) fits `log10(Δloss)` vs bpp and at `R² ≥ 0.99` prints that
there is no intrinsic knee and ship bpp should come from a byte budget or measured saturation.

`SELECTION_MODE=validated-surrogate` (`run-pipeline.sh:1056-1288`, requires
`PRODUCTION_CACHE=1`) is the real-KL path: Pareto assignments → one format-menu frontier cache
→ `validate_assignments_kl` per point (`--calib-skip-first $NSAMPLES` is the held-out
mechanism; `--kl-scope full_sequence` since M26) → `select_validated_frontier` → optional
`MSE_PROMOTION` rewrite → `production_recache` re-fits activation scales for the selected
assignment.

A rate-axis pick is not a selection until a byte-matched uniform arm corroborates it
(#117): `select_validated_frontier` re-ranks only the allocator's own Pareto rows and
carries no uniform arm, so on a non-uniform Tessera assignment it cannot see uniform
beating the pick at matched bytes. It therefore writes the recipe files the control loop
needs, stamps `uniform_control.status: outstanding` (selected label, measured KL, bytes,
and what would pass) into the layer config and the selection summary, and exits nonzero
(`RATE_AXIS_UNCERTIFIED_EXIT`) rather than certifying. The refusal is a gate, not a wall:
`--acknowledge-outstanding-uniform-control RUN_ID` (or
`PRISMAQUANT_ACKNOWLEDGE_OUTSTANDING_UNIFORM_CONTROL`; an explicit flag wins) stamps
`status: acknowledged` with the run id and returns 0 so the pipeline can build the
candidate the control is built from -- building it is a precondition of the comparison,
not a way around it. A pick that is itself uniform (one distinct format over every
assigned unit) is its own control and exits 0 with `status: not_applicable`. The wall
sits at publication: `verify`/publish (#121) refuse until the served control closes the
slot.

`select_validated_frontier.py` builds an **η-dominance** envelope: rows sorted by (bpp, kl), a
point enters only if it beats the running best by more than `--kl-noise-floor`
(`_frontier_from_rows`). Picks: `kneedle` (default without a card), `budget` (**the default
under `TARGET_DISK_GB`**), `best-kl`, `lowest-bpp`, `practical-knee`, `saturation`.
Diagnostics emitted with the pick: surrogate-vs-KL Spearman, `worst_rank_inversion`,
leave-one-out kneedle stability.

**Tail veto** (D1, 2026-07-30) — **DEFAULT-ON, contract statistic `kl_max`** (ruled by Robert
2026-07-30). §2.3 rule: KL is a *screening* metric and a lower mean can hide a heavier tail —
the shipped 27B PrismaSCOUT has a worse max-prompt NLL than the artifact it beat on mean KL.
`--tail-veto {none,kl_p99,kl_max,nll_p99}` adds a second admission condition to the same single
pass: a row that improves mean KL enters only when
`row[tail] <= incumbent[tail] * (1 + --tail-eta)`, the incumbent being the last *admitted*
frontier point. Columns come from `validate_assignments_kl`'s per-sequence emission (§7.1) and
carry the gold lane's key names, so a selection row and a served row read the same.

- **The contract statistic is `kl_max`** — the worst sequence. It is the statistic that would
  have caught the broken 27B that passed on the *mean* while 80% of its prompts were bad, which
  is the same reason §7.2's ship gate guards p99 per-prompt NLL. `nll_p99` continues to be
  recorded on every row (both are free), so switching the contract later is a flag, not a
  re-measurement.
- **Default-on is safe because the failure is one-sided.** A spurious veto only makes the pick
  **more conservative** — it refuses a lower-bpp/lower-mean point and keeps a higher-bpp one
  with a smaller tail — and it is never silent: every refusal is printed and retained in the
  summary under `vetoed_rows` with its `veto_reason` (`tail_regression`, or `tail_missing` when
  a row predates the emission). The `--tail-veto` help text says exactly this.
- **`--tail-eta` defaults to `auto`: the slack is derived, not chosen** (house rule 2). `auto` =
  the incumbent row's **relative stderr of the tail statistic across calibration repeats**
  (`std/√n ÷ mean` over `<column>_repeats`, floored at 0) — i.e. how much that tail moves when
  nothing about the assignment changes, so a candidate inside its own measurement noise is
  admitted and one outside it is a real regression. `validate_assignments_kl` emits the
  per-repeat tails from the same forwards the mean already paid for. With a **single** repeat
  there is no spread: `auto` degrades to a strict `0` **and prints a warning**, because
  single-seed tails are noisy (§2.5: a +10% reading has flipped to −5.2% across repeats) — run
  validation with `--calib-repeats ≥ 4` to get a real slack. An explicit numeric `--tail-eta`
  always wins. Derivation documented at `select_validated_frontier.tail_eta_auto`.
- **A pre-R9 validation JSON carries no tail column at all.** Vetoing every row would turn a
  stale input into an empty frontier, so the veto goes **inert** with a loud warning and the
  run reproduces the mean-only envelope (`tail_veto_inert_reason`, recorded in the summary).

`--tail-veto none` restores the pre-R9 envelope byte-for-byte (pinned by a regression test).
The veto also applies inside the leave-one-out rebuild, so the stability diagnostic reflects
the same envelope the pick came from. There is **no second eval pass** — the tail was already
being computed and discarded.

**Byte budget = constraint, measured KL = objective** (re-vet **R1**, closes D12). Two disjoint
ship selectors used to exist: the allocator's `--target-disk-gb` picked by *predicted* Δloss
among the allocations that fit, and `select_validated_frontier` picked by *measured* KL but was
byte-blind (`grep -c bytes` → 0). The selector that owned the ship decision ran on the
surrogate; the one that measured could not see the card — §2.2 inverted exactly where it is
load-bearing, and the surrogate-knee failure is on record (27B: surrogate 5.857/0.056 vs
validated 5.31/0.015). They are now one stage:

* `TARGET_DISK_GB` is plumbed through `run-pipeline.sh` into the allocator. When set it
  **overrides `TARGET_BITS`** (the allocator re-emits at the chosen bpp) — the CLI semantics,
  now the pipeline's.
* The allocator prices **every Pareto candidate** with `footprint.assignment_artifact_bytes`
  — the same accounting its own byte-budget selector uses, so the two can never disagree — and
  stamps `artifact_bytes` into each Pareto assignment payload and the manifest.
* Under a card it then **narrows the Pareto set to the bracket around the byte-feasible bpp**
  (the largest fitting rung ±1, ~3 of 11), with a log line naming the largest fitting rung. A
  computed narrowing, not a hardcoded rung count; skipped loudly if any candidate is unpriced.
  This is what makes byte-budget selection ~3 KL evals rather than 11, and it is why
  `validated-surrogate` defaults **on** under a card and stays opt-in without one.
* `select_validated_frontier --mode budget --target-disk-gb` picks **min measured KL among the
  rows whose exact footprint fits**. `measured_rows` gains the `artifact_bytes` column, read
  from the row or from the allocator payload at `row["path"]`. Bytes are monotone in bpp and
  the frontier is the KL lower envelope, so the min-KL fitting frontier row is the min-KL
  fitting row overall. Unpriced rows or an infeasible card are hard errors, never a silent
  fallback to another pick.

`footprint.py` reproduces real `metadata.total_size` to 0.00% on three 27B artifacts (`GB = 1e9`);
`saturation_select.select_under_byte_budget` grids the rungs and the ratchet bisects the
memoized DP for an exact fit. Kneedle stays available and stays what `_rd_curve_diagnostic`
already calls it: a diagnostic on a log-linear RD curve.

**The floor is a per-tensor manifest, and it cannot go negative** (#15, `bb974a0`; `0a9dc00`).
The identity is `artifact_bytes = floor + Σ_reencoded memory_bytes_for_shape`, with
`floor = source_total − Σ_reencoded source_bytes`. Charging that second term at a *regime-wide*
per-param rate breaks on a mixed source — a DSv4-Flash checkpoint (I8-nibble MXFP4 experts +
E8M0 scales, F8 attention, BF16 floor) charged at the FP8_SOURCE 1 B/param layout removed more
bytes than the checkpoint holds and drove the floor to −113 GB, letting an artifact more than
twice the budget "fit". `source_tensor_bytes_manifest` (`footprint.py:229-329`) now sums each
re-encoded Linear's **actual safetensors header byte span** (weight + scale siblings), resolving
per-expert-on-disk names to the packed live names the allocator uses via the profile's
`packed_expert_parent_for_projection` — the same bridge layer-streaming uses — and keeping
tensors the live-name map declines rather than dropping them (`0a9dc00`). Three failure modes
are **hard errors raised before any selection number is consumed**
(`resolve_reencoded_source_bytes :331-423`, `check_floor_non_negative :462-495`): a re-encoded
name the manifest cannot resolve (its source bytes stay in the floor while its quantized bytes
are still added — on a packed-MoE model that is the whole expert mass, after which every rung
reads "below the floor"), two names resolving to the same source span (bytes removed twice, so
an over-budget artifact reads as fitting), and a negative floor, which is reported with a
per-tensor-class byte breakdown so the offending class is named rather than rationalized. The
byte-budget selector calls the shared `assignment_artifact_bytes` rather than an inlined copy
(`allocator.py:2309-2345`) — the inlined copy was the one path the exactness tests never
covered.

### 4.7 The two cost axes, and what the lane gates actually assert (R3)

`COST_MODE` silently decided **two independent things**: which *render* produces the
per-`(Linear, format)` error, and which *objective* maps it to `predicted_dloss`. Since
2026-07-30 they are named — `COST_RENDER ∈ {inline, cached-menu}` × `COST_OBJECTIVE ∈
{weight-recon, render-score, aura-adjoint}` — and `COST_MODE` is the documented **spelling**
over them, with its three values unchanged in meaning:

| `COST_MODE` | `COST_RENDER` | `COST_OBJECTIVE` |
|---|---|---|
| `local` | `inline` | `weight-recon` |
| `production-render-score` | `cached-menu` | `render-score` |
| `aura` (default) | `cached-menu` | `aura-adjoint` |

Setting the axes directly is equivalent; setting both spellings at once is `exit 2`, and the
two unimplemented pairs stop with the reason (`inline × aura-adjoint`: the adjoint consumes
production-rendered dW from a format-menu cache by definition).

**Why the split mattered.** The CB and GGUF lanes hard-`exit 2`'d unless `COST_MODE=local`,
justified as "production-render-score scores UNWEIGHTED registry renders" — a property of the
**render**, not of the objective, which is why an objective change was being blocked by a
render argument. The right key already existed: `measure_quant_cost._cost_render_uses_imatrix`
decides per **format family** (CB always weighted, `gguf` tracking `PRISMAQUANT_GGUF_IMATRIX`),
which is why `local` was never one thing either — under it the CB lane already measured the
exporter-faithful weighted encode while the compressed-tensors lane measured unweighted RTN.

The gates are now one **render-faithfulness assertion**: the render that produces the
allocator's cost must be the render the exporter ships. `inline` satisfies it by construction;
`cached-menu` satisfies it iff the `ProductionWeightCache` render applies the same imatrix,
which **CB Milestone C** made possible — `render_production_weight` /
`build_production_cache` take `col_weights`, applied to the weighted families only, with every
other format's bytes bit-identical (`tests/test_col_weights_render_identity.py`). The pipeline
harvests the vector once (`harvest_cb_col_weights`, one definition, four call sites) and passes
`--col-weights` to the cost cache. `PRODUCTION_CACHE=0` stays required on those lanes for the
unchanged reason: their **exporters** requantize the bf16 skeleton and never read a production
cache, so building the *export* cache burns hours on bytes that never ship.

**What this deliberately does NOT do.** The CB lane can now run `render-score` or
`aura-adjoint`; neither is its default and neither is recommended. AURA's −38%/−17.9% margins
are native-lane results, and CB's error surface (VQ quantization plus the expert route-flip
floor) is a different animal. The pipeline prints that the combination is opt-in when it is
selected. The A/B that would justify a CB default has not been run — and is **deferred by
Robert (2026-07-30)** behind the NVFP4-vs-CB-FP8@4.5 criteria work; CB stays `weight-recon`
until it runs.

**The trust-region readout that rides with the default objective.** `AURA_ADDITIVITY_GATE`
defaults to **`measure`** (ruled 2026-07-30), so every `COST_MODE=aura` run performs one
bounded end-KL eval and stamps a real `residual` into `cost.pkl`'s `provenance["additivity"]`
— stage `[3c]`, §4.3. `auto` (report only from a measurement the run already made) and `0`
remain selectable.

### 4.8 The Sensitivity Card — probe once, price an arbitrary menu (2026-08-14)

Landed on `feat/aura-sensitivity-contract`. Read the status line first, because
everything below is additive and unpromoted: **none of these modules is a
pipeline stage**, `run-pipeline.sh` calls none of them, `COST_MODE` gains no
value, and no shipped allocation goes through them. The one shipping default the
work changes is probe-side (§3.3), which is why the section exists at all under
§0's rule. Full design, including what is proven and what is not:
`docs/design/sensitivity_card_contract.md`.

**Why.** Probing a model is the expensive, model-specific half; choosing formats
is the cheap, platform-specific half. Today they are fused — `probe.pkl` carries
scalars that only become a cost next to a **rendered menu cache** built for one
particular format list (§4.2), so a new menu means a new render, a probe cannot
be shared because it is not sufficient on its own to price anything, and W4A4
and W4A8 are literally the same candidate because the cost is weight-space only.

**What the probe can now store.** The per-element diagonal weight Fisher
`H[o,i] = Σ_t g[t,o]²·x[t,i]²` is already materialized per chunk as `chunk_h` at
every accumulation site, and has never been storable: the unified-sweep path
documents it as *"47k × 17 MB = 800 GB CPU, doesn't fit"*, which is why only
`h_trace` and `h_w2_sum` survived. Its two **marginals** are storable, and are
what a format-independent cost actually needs:

```
fisher_row[o] = Σ_i H[o,i]        fisher_col[i] = Σ_o H[o,i]
```

`out + in` floats instead of `out · in`, read off the `chunk_h` each site
already forms, so no extra matmul (`incremental_probe._marginal_chunk:97-123`).
Three further vectors ride along because they are **not** recoverable from the
weight-Fisher marginals: `act_sq_sum` (the imatrix, `diag(XᵀX)`), `g_sq_sum`
(the **output**-space Fisher diagonal), and `act_absmax`. The merge rules differ
per key and getting that wrong is the likely bug — sums add elementwise,
`act_absmax` is a bound and merges by elementwise **maximum**
(`_MARGINAL_MAX_KEYS:90`) — so one `merge_marginals` (`:143-161`) serves both the
per-layer host flush and the cross-shard partial-stats merge and the two cannot
drift apart on that rule. Sync discipline is preserved rather than re-litigated:
the five vectors accumulate device-resident in fp32 and drain through one flat
concatenation and one `.cpu()` per layer (`_marginal_accumulate:126-140`,
`_marginal_flush:164-186`). A `.cpu()` inside the hook would put back the ~94k
per-Linear syncs that `PRISMAQUANT_DEFERRED_FISHER_SYNC` removed.

`sum(fisher_row) == sum(fisher_col) == h_trace_raw` is the wiring check, and it
is free: a transposed axis, a dropped chunk or a wrong-shaped accumulator all
break it. `SensitivityUnit.validate` (`sensitivity_card.py:245-288`) enforces it
at `rtol=1e-3`, which catches the whole class of bug where one accumulator is
normalized and another is not. Its *exactness* is site-dependent — §12 D30.

**The card.** `sensitivity_card.py` — one compressed `.npz` with a JSON header,
loaded `allow_pickle=False` (`:463-470`), because a shareable artifact must load
without executing arbitrary objects. Per unit it carries shape, `n_params`, the
**global** calibration token count (the one denominator of §4.1 — a per-expert
routed count inverts importance weighting), the raw un-normalized scalars, and
whichever of the five vectors exist. Three refusals are structural, not
advisory:

- **Calibration is identity.** `assert_compatible` (`:385-406`) refuses a
  cross-calibration merge or comparison outright — the same rule a CB book gets
  by hashing its imatrix into its book key (§9.2). Rebase the card, do not
  compare across calibrations.
- **Render basis is stamped, not assumed.** A shareable card is necessarily RTN:
  compensated renders need per-Linear Hessians (~100 MB/Linear at 27B scale),
  which are not shippable. That is load-bearing, not a footnote — RTN-vs-rendered
  `dW` is immaterial at fp4 and decisive at fp8 (+36% served KL, §4.3), so a card
  priced on one basis mis-ranks the other's 8-bit rungs, and mismatched bases
  refuse to compare.
- **Structure travels; policy does not.** The card records that q/k/v *are*
  siblings, which experts are packed together, and each unit's source dtype
  (`UnitTopology`, `:141-166`). It deliberately does **not** record that a
  runtime *requires* siblings to share a format, or that packed experts need
  vLLM canonical scheme names: those are §6.4 serving invariants, derived
  downstream from whichever profile the author names (§8.1). Baking vLLM's
  packing into a shareable file makes it wrong for llama.cpp, and wrong the day
  vLLM changes.

Card footprint against the full-`H` footprint of the same probe rows: 27B dense
104.2 GB → 75.2 MB, 35B-A3B 8.2 GB → 17.4 MB, MiniMax-M2.7 at per-expert
granularity 0.9 TB → 2.3 GB (`docs/design/sensitivity_card_contract.md` §3,
which flags the per-expert case as the one where the card is not yet small —
size tracks probe *rows*, not parameters).

**Three cost tiers, one seam** (`format_cost_protocol.py`):

| tier | weight term | status |
|---|---|---|
| `SCALAR` | `0.5 · h_trace · weight_mse · gain` | **exactly today's model** — same formula, same inputs |
| `MARGINAL` | `0.5 · (row @ dW² @ col) / h_trace_raw / n_tokens` | rank-1 reconstruction of `H`; `SCALAR` is its rank-0 collapse |
| `AQUA` | `MARGINAL` + an activation term | W4A4 and W4A8 stop being one candidate |

`SCALAR` is not *equivalent to* `allocator_solver.predicted_dloss` (§4.1) — it is
the identical expression, kept as a named one-liner (`weight_dloss_scalar:184-192`)
so any divergence is a one-line diff rather than a hunt, and pinned by
`tests/test_sensitivity_card.py::test_scalar_model_reproduces_allocator_solver`.
A card carrying no vectors falls back to it (`weight_dloss_marginal:209-210`),
which is what lets an existing `probe.pkl` convert to a card with no re-probe:
degraded, not broken. `MARGINAL` is **exact** when `H` is genuinely rank-1
(`test_marginal_model_matches_exact_fisher_on_rank1`, `rtol=1e-10`), which is the
sharpest available check on the quadratic form and its normalization.

**The solver seam is unchanged.** `sensitivity_card_allocate.candidates_from_card`
(`:47-84`) emits exactly the `dict[name, list[Candidate]]` that
`allocator_solver.solve_allocation` already consumes —
`Candidate(fmt, bits_per_param, memory_bytes, predicted_dloss)`, §4.5 — so an
arbitrary format menu enters as an arbitrary list of plugins rather than as a
change to the DP. Only `DELTA_LOSS` may leave the module toward the solver
(`CostComponents.assert_currency:172-178`), because only loss is additive across
units. A unit with no legal format is dropped from **both** `stats` and
`candidates` rather than handed over as an empty option list, which the DP would
otherwise read as a free choice — the same failure shape as the unpriced packed
member of §4.5.

**AQUA-AURA, and why it needs its own sensitivity.** `h_trace` is a *weight-space*
curvature. An activation-quantization error is an *input-side* perturbation
`x → x + dx` that reaches the loss as `dy = W dx`. Multiplying an input-side
error by a weight-space sensitivity is the currency error
`activation_fair_pricing.py` exists to correct (§3.3 `ACTIVATION_FAIR_PRICING`),
so `activation_dloss` (`:227-254`) routes through `g_sq_sum` — the output-space
Fisher, the one term a weight-space `h_trace` structurally cannot supply — and
never through `h_trace`. Three rules hold the line, each of them principle 2 in
practice: an unmeasured A-side returns **`None`, never `0.0`**, so a missing
measurement can never read as a free one; the `1/12` step variance is a property
of a uniform grid, not a tuned constant (`uniform_act_quant_variance:257-281`);
and there is **no speed/quality scalarization constant** — `speed_index` and
`predicted_dloss` come back as separate axes, because choosing between them is a
frontier selection (§4.6), and inventing a weighting would be exactly the
heuristic-where-an-explicit-exists that is banned. `activation_fair_pricing.py`
is left running and untouched: superseding it is a promotion decision on served
evidence, not a drive-by refactor.

**Status.** The tier that reproduces today's behaviour is exact and unit-tested;
the two that do not are **screening surrogates with no served A/B**, and §2.5
says a screen is never sold as a result. Do not describe `MARGINAL` or
AQUA-AURA as validated, promoted, or default. What promotion would require is
the ordinary §2.4 ladder: rank agreement against measured `output_mse` on a
small model, allocation churn against a shipped `cost.pkl` inside the known ~3%
noise, and — for AQUA — a served W4A4-vs-W4A8 A/B. §12 D30 carries the honest
gaps.

### 4.9 The continuous trellis rate surface — ARCHIVED 2026-09-02

> **Archived (2026-09-02, `tessera/decouple-gridbook`, #118).** The five
> `prismaquant/trellis_*.py` modules, the `trellis_research_sm121` profile and
> the four `test_trellis_*.py` files now live at
> `archive/trellis_wire_2026-09-02/`, with that directory's README as their
> obituary. They priced **`gridbook.trellis.wire.v1`**; Rob retired the
> Gridbook lane on 2026-09-02, so no sanctioned runtime reads those bytes any
> more. The byte model is **not** a port of Tessera's
> `prismaquant.tessera.v1` — a different plane set, said so in
> `tessera_footprint.py`'s own header — so nothing migrated; the pricing
> *machinery* had already been renamed onto Tessera
> (`tessera_{formats,footprint,allocator,rate_surface,menu}.py`), and Tessera's
> continuous rungs need no flag at all: `reduce_continuous_menu` folds them
> into every menu unconditionally.
>
> **What changed in the live tree.** `export_native_compressed` no longer
> imports `parse_trellis_format_name` and no longer carries a `TCQ_*`-specific
> branch; such a name now takes the generic "absent from `FORMAT_SCHEME`"
> refusal, which fails closed and names the format
> (verified by execution on `TCQ_E2M1_R1000`).
> `allocator_candidates` no longer imports `trellis_menu`, and its unused
> `trellis_provenance=` kwarg is gone. **The flag is still refused**, by
> `allocator_candidates.refuse_retired_trellis_surface` as the first statement
> of `build_candidates`: a set `PRISMAQUANT_TRELLIS_SURFACE` raises, naming
> `gridbook.trellis.wire.v1`, the archive path and #118. Dropping the variable
> instead would have let a stale driver keep exporting it and get a
> *different* allocation with no diagnostic — a gate that fails open, the
> failure class of prismaquant#120. Gate:
> `tests/test_retired_trellis_surface_refusal.py`.
>
> Everything below is the section as it stood on 2026-08-29 and is kept as
> history — the design, the eight unwired links and the refusal reasoning are
> the durable part. Read every present tense in it as past tense; the
> file:line citations point into `archive/trellis_wire_2026-09-02/`.

#### History, as the section stood on 2026-08-29: the continuous trellis rate surface — opt-in, allocation-time only

`prismaquant/trellis_{formats,footprint,allocator,rate_surface}.py` (3701 LoC,
69 tests) address and exactly price the Gridbook rate-256 tail-biting trellis
families. Until 2026-08-29 nothing in the pipeline imported them: no
`run-pipeline.sh` stage, no format menu, no exporter. `prismaquant/trellis_menu.py`
is now the ONE seam that does, and it is off by default.

**The flag, and what it does today.** `PRISMAQUANT_TRELLIS_SURFACE=<manifest.json>`.
Unset, `augment_candidates` returns its input object unchanged, so a run
without the flag executes exactly the path it executed before the seam existed
— the `PRISMAQUANT_FISHER_CAP_MULTIPLIER` precedent (§ P6). **Set, it refuses.**

> **Correction (2026-08-29, same day).** The first version of this section said
> the seam's placement inside `allocator_candidates.build_candidates` meant
> "trellis rungs pass the same legality, aggregation and byte accounting every
> other candidate does." That was **false on all three counts**, and it is the
> § P13 "currency is not truth" case in its own doc: the paragraph was written
> in the same commit as the code and was wrong about it. The seam appends
> AFTER the per-spec legality loop, so it never runs
> `check_stats_format_applicability` nor the `_memory_bytes_by_format` write at
> `allocator_candidates.py:1950`; aggregation drops every rung; byte accounting
> `KeyError`s. `trellis_menu.UNWIRED_LINKS` is now the authoritative list —
> eight entries, each with a file:line — and it is the text of the refusal.

The eight, in the order a run would hit them: no TCQ `FormatSpec`
(`format_registry.py:1267-1272`); the exact assignment-payload filter falling
through to `fr.get_format` because nothing writes `_memory_bytes_by_format` for
a TCQ row, which kills the allocator inside the Pareto sweep **before**
`layer_config.json` and makes the pointed refusals in `layer_config` and the
exporter unreachable (`allocator.py:3369-3386`); fused-sibling aggregation
building super-item menus by iterating `FormatSpec` objects
(`allocator_candidates.py:2464`); the identical packed-expert construction
(`:2701`); `promote_serving_units`' `format_rank` lookup, which does not crash
today only because aggregation guarantees a TCQ unit is a lone ungrouped
Linear (`allocator_solver.py:340-342`); the byte-budget path's own registry
lookup (`footprint.py:1183`); `build_candidates` being called with neither
`cost_mode=` nor `trellis_provenance=`, so the currency gate compares against
`os.environ.get("COST_MODE","aura")` — a variable `run-pipeline.sh` sets with
`:=` and never exports — and the manifest identity and anchor contract are
discarded rather than travelling with the assignment (`allocator.py:2756`,
§§ P12/P14); and the anchors' currency being weighted SSE under an activation
second moment, an output-MSE proxy and **not** the AURA KL-adjoint the DP ranks
in (`trellis_rate_surface.py:43-52`).

**Why refuse as a whole rather than wire it halfway.** The eight do not fail
alike. The registry gaps crash **loudly**; the aggregation gaps are **silent** —
they drop every rung from every fused and packed group and hand back a
plausible frontier in which only `o_proj` and `down_proj` could carry one. A
partial fix that removed the crashes would trade the loud failure for the
silent one, and the packed-expert case was worse still: a hand-rolled 2-tuple
shape priced a 128-expert row as one expert (a **128×** underprice, reported as
"0 unit(s) skipped"), while the guard meant to catch it read a stats key
nothing writes. Both are fixed — the seam now uses
`allocator_solver._shape_from_stats` and
`allocator_candidates._stats_indicates_packed_expert`, and refuses a packed row
with a counted reason — but the fix is inside `build_trellis_menu`, which is
research-reachable and cannot reach an artifact.

**So: `build_trellis_menu` builds a correctly priced menu; `augment_candidates`
refuses.** Enabling the surface means landing the eight links with tests that
exercise behaviour and then deleting the refusal — not passing a flag.

**Why a manifest, not a `FORMATS` enum entry.** A trellis rung is
`(family, body_rate_q256, layout, schedule, alphabets)`, and the wire carries
one 4-bit rate code per input column shared across rows, so the rate resolution
is `SUPERBLOCK_WEIGHTS/columns` q256 — 0.25 q256 on a 1024-column Linear,
effectively continuous. What makes a rung *cost* something is a measured
anchor, and anchors are per-campaign data. The manifest names them; the module
fits a monotone piecewise-linear surface in `(q256, log2 dloss)` between
BRACKETING anchors only and densifies it. Every densified rung gets a real
per-column schedule and therefore an EXACT byte footprint: the rate is
interpolated, the bytes never are.

**The name the DP sees is shape-free, and that is load-bearing.** Candidates
carry `fmt = TCQ_{E2M1,E4M3}_R<q256>` (the closed 2546-name vocabulary
`trellis_formats.ALL_LEGAL_TRELLIS_FORMAT_NAMES`, round-tripped by
`parse_trellis_format_name`). `TrellisAllocatorCandidate.allocator_key` is NOT
used: it embeds the pre-render recipe digest, which hashes the SHAPE, so q_proj
and k_proj would share no format at any rung and `aggregate_fused_siblings` /
`aggregate_packed_serving_groups` — which intersect member menus BY FORMAT NAME
— would silently collapse every fused and packed group back to individual rows.
The per-tensor recipe digest still travels, on `Candidate.serialized_identity`,
which is where per-member layout identity belongs.

**Three refusals, each closing a specific hole.**

1. *A profile that declares no `target_platform`.* `trellis_allocator._capability_gate`
   (`:578-586`) returns **legal** when the platform is `None` — deliberately,
   because admission is then the experiment's responsibility. Six of the ten
   serving-profile specs take that branch, `research` among them. Allocating
   against a gate that cannot compare anything is worse than no gate: it looks
   like a check. So the manifest must name a profile declaring an exact
   platform. `serving_profile_specs/trellis_research_sm121.json` is that
   profile — `target_platform: sm_121`, `emulation_only: true`, no export lane,
   and deliberately **no `format_rules`**: naming 2546 TCQ rungs in an allow
   list would assert they are `format_registry` entries, and they are not (a
   trellis rung has no `FormatSpec` and no RTN `quantize_dequantize`, because
   nothing renders one).
2. *An objective the run is not pricing in.* The manifest declares `cost_mode`
   and `currency`; a mismatch with the run refuses. One DP prices in one
   currency.
3. *An unstated activation contract.* The manifest must declare the contract
   its dloss numbers were measured under, and it is stamped on the provenance
   payload. The hull anchors were priced **W\*A16** while both families' native
   `_scaled_mm` routes are **A=W** (W8A8 for E4M3, W4A4 for E2M1). Rendering
   identity without execution identity priced a real A-side at zero once
   already (NVFP4_CB, 2026-08-17); the stamp is what stops a future A=W lane
   inheriting a W\*A16 loss.

**Export fails closed, and says why.** `export_native_compressed` refuses any
`TCQ_*` assignment with a pointed message rather than the generic
"absent from FORMAT_SCHEME" one, because the generic message blames the serving
profile's export lane and that is the wrong diagnosis. No lane bound is
missing: `ProductionWeightCache` renders no trellis wire, so there are no bytes
to pack, and the producer Gridbook pin publishes no executed-activation-contract
table for these families, so an exported artifact could not state its own
activation contract (§ P14). **This is allocation-time reach only** — the
surface lets the DP see the continuum, report where bytes would go, and price
the choice in exact serialized bytes. Promoting it to an artifact needs a
render mechanism and a runtime attestation first; both are open (§12).

Gate: `tests/test_trellis_menu.py` (15), on top of the surface's own 69.
Two of the original 13 asserted on **source text** — that the string
`trellis_menu.augment_candidates` appeared in `build_candidates` — which passes
whether or not the call does anything, and did pass while the enabled path
could not produce an assignment at all. They now assert observed behaviour: the
seam raises `TrellisSeamUnwiredError` naming every entry of `UNWIRED_LINKS`, a
128-expert row is skipped-with-reason rather than underpriced, and the two
cheapest ledger entries are re-checked against the code so a stale entry fails
the suite.

**2026-09-02 (a).** The trio has a Tessera twin — `tessera_{formats,footprint,allocator,rate_surface}.py`
— that points the same pricing machinery at Tessera families instead of the
Gridbook TCQ ones. It is imported by nothing in the pipeline; the live Tessera
seam is the registry fallback in §5.7, and this section's refusal is unchanged.

**2026-09-02 (b), and this is where the history above stops.** The Gridbook
modules are archived (#118) and the seam is gone; see the banner at the top of
this section for what the live tree does now. The Tessera twin is no longer a
twin of anything live — it is the only one of the two left, and §4.10 is its
seam.

### 4.10 The Tessera continuous menu — `FORMATS=TESSERA` (2026-09-02)

Every other entry on the format menu is a *point*: `NVFP4` is one rate, `FP8_E4M3`
is one rate, and a menu is the handful of them a launcher lists. A Tessera family
is an *axis* — a rate addressable at a 1/256-bpp quantum, realised as a per-column
Bresenham schedule — so "which rungs are on the menu" is a question with a
per-unit answer in the thousands, not a comma-separated string. This section is
the seam that lets the production allocator ask it.

**The four families, and why not five.** `tessera_menu.menu_families()` asks
Tessera, and gets `TESSERA_BF16_K1`, `TESSERA_E2M1_K1`, `TESSERA_E2M1_K2`,
`TESSERA_E4M3_K1`. E2M1 arity 2 exists because it fills the rungs between arity
1's and is what makes 4.0 bpp addressable at all.

**`TESSERA_E4M3_K2` is refused by Tessera itself** — 2^16 anchors scored per
trellis step, at the wall the encoder already declines — and that refusal is
pinned by a test rather than reproduced as a local exclusion list, because it is
a cost refusal that would come back the day the anchor budget moves.

**`TESSERA_BF16_K1` is the 16-bit family, and the anchor budget is the body's,
not the grid's.** `ANCHOR_BUDGET_BITS` used to refuse `payload_bits >= 16` flat,
at family construction, which reads the wall as a property of the code space. It
is a property of the **body**: a TCQ step scores `2^payload_bits` anchors, while
a WINDOW step scores `2^window_bits` states (16 384 at the default L=14) and has
no forest at all — `tessera.export._plan_for` returns the grid in the forests'
place — so the 65 536-code BF16 grid is touched once per unit, to snap its
table. Tessera says exactly this about its own serialisable set, and admits
BF16 for that reason while keeping `E4M3^2` out. Read flatly, the budget made
the whole 16-bit half of the rate axis unallocatable: the family could not be
named, so `menu_families()` never saw it and no cost table could hold a column
for it. `TesseraFamily.__post_init__` now asks
`tessera.export.recipe_table(grid)` whether the TCQ body is reachable at **any**
rung and refuses only then, so a family whose body varies with the rung
(E2M1x2's does) is answered per rung rather than by assumption, and nothing here
caps the window's own width — `export._window_bits_for` widens L to the rate
rather than refusing it, and a wall Tessera does not have would be this module
deciding the DP's candidate set (P1). A grid Tessera declines to build at all
(`tuple_grid` above 2^16 codes, e.g. `BF16^2`) is refused with Tessera's own
message re-raised as a `TesseraFormatError`.

**One map, not two.** Admitting the base to `tessera_formats._HARDWARE_BASES`
made the family namable, priceable and enumerable, and left it *unallocatable*:
`tessera_render._grid_for` held a second base→grid map, still listing E2M1 and
E4M3 only, so it called BF16 a free Lloyd-Max base —
`tessera_rung_is_serialisable` went False, `_producer_eligible` went False, and
the menu dropped every rung of the family in **both** modes while nothing
raised. `_grid_for` now reads `_HARDWARE_BASES` and returns the family's own
`payload_grid()`, and the family's allocatability test asks the question in the
allocator's vocabulary — does `require_producer_formats` accept the name, does
`expand_tessera_menu` carry rungs of it, does each of those resolve back
through `get_format` — because a test that asks only `tessera_formats` cannot
see a second map living in another module.

The family arrived here to be **priced**, not to be served, and it is now
both — the two states one contract version apart, which is worth keeping on the
page because the producer-side code did not change between them.

*As it landed* (Tessera contract v3/v4): the pinned contract published the
pattern and attested no rung of it. Since tessera `f6bdb42` ("Publish the rungs
the decoder reads, and refuse the ones it does not") a `TESSERA_BF16_K1_R{k}`
row existed in `formats[]` with an **empty** `attested_rungs_q256` and no
`lane_eligibility` cell. The lane spec's `executes` list carried the third glob
anyway — deriving a list from a table means copying the table, not filtering it
to the rows we like (P14) — while `route_admission` refused every rung of the
family for want of a cell. The published-but-unrung row was the honest shape of
"the decoder reads these bytes, the runtime serves none of them yet".

*As of Tessera contract v5*: the family has `attested_rungs_q256: [1792]` and
two `sm_121` dense cells (`..._decode_mm_w16a16`, `..._batch_mm_w16a16`,
`bf16_unquantized`, `backed_with_serve_flag`) on the 2026-09-02 BF16 route
receipt — four route censuses, 112/112 modules, served KL in both residency
modes against the image-matched teacher. So `route_admission` answers
`backed_with_serve_flag` at R1792 and `unattested` at every other rung, and the
attested menu is three rungs, measured on `[2048, 1024]`:

| rung | bpp | A side |
|---|---|---|
| `TESSERA_E2M1_K2_R896` | 4.0000 | `w4a4-nvfp4-e2m1-group16-ue4m3` |
| `TESSERA_E4M3_K1_R1024` | 4.0781 | `w8a8-dynamic-e4m3-channel` |
| `TESSERA_BF16_K1_R1792` | 7.1406 | `w16a16-bf16-channel` |

Every other BF16 rung is still reachable only under
`PRISMAQUANT_TESSERA_MENU=research`, stamped `unattested`, for the export gate
to fail closed on. **The rule did not bend to admit the family; the runtime
published a receipt and the rule read it** — which is the whole shape of P14,
and is why the producer side needed no edit for the status to change. Its layout claim — `tessera_serving_route` answers
`w16a16-bf16-channel`, terminal `BF16` — is a statement about what the decode
lands in, in Tessera's own words ("a plain BF16 tensor (W16A16)",
`export.wire_recipe`), never an attestation (P14).

**A wire commitment is asked of the grid, once.** Widening the widest grid by
256x turned a cost that had always been redundant into the dominant one:
`tessera_rung_is_serialisable` hashed the family's whole grid per *rung*, behind
an `lru_cache` sized 64 against menus of thousands of names, so the cache
thrashed and one 2048x1024 unit's **attested** menu -- which then drops every
BF16 rung as unattested -- went from 0.19 s to 52 s. A profile put 234.0 of
234.5 s in `grid_digest`: 6916 calls at 34 ms, computing an answer that cannot
vary with the rate. `tessera_render.family_grid_is_serialisable` now asks it of
the family, and `menu_families` reads the same predicate instead of digesting
again; the attested menu costs 0.082 s, below where it started. The guard counts
digest calls rather than seconds, because the defect is a quantity of work and a
clock would only measure the box.

**A menu memo is sized in shapes, not in a round number.** The two neighbouring
caches (`tessera_menu._shard_geometry`,
`tessera_footprint._exact_bits_for_shape`) were sized 4096 against the same
6764-rung menu, so one pass over one shape evicted its own entries and the next
recomputed every one of them: research-mode cold 10.06 s and warm 9.958 s, no
memo hit at all (RobTand/tessera#46). What was wrong was the bound's *unit*.
`menu_cache_bound()` is now `menu_rungs_per_shape() * shapes`, and the first
factor is asked of `menu_families()` — 6916 today, 3055 before the 16-bit
family — rather than written down, so it moves when the roster does. Because
the bound is a whole number of shapes and refuses to be fewer than one, no
setting of `PRISMAQUANT_TESSERA_MENU_CACHE_SHAPES` can put a shape where it
evicts itself; that is the property
`tests/test_tessera_menu.py::test_every_menu_memo_survives_one_shapes_widest_menu`
pins, reading `maxsize` off the live memo instead of restating a number.

**A rung-keyed memo is sized over the grid space, which is not the menu.** Two
more memos on the same call graph are keyed on a *rung* rather than on a
(rung, shape): `tessera_render.tessera_rung_is_serialisable` on the full format
name, and `tessera_formats._recipe_for` on `(base, base_size, arity, rung)`.
Both carried round numbers -- 4096 and 512 -- and the count that bounds them is
**not** `menu_rungs_per_shape()`. A name is admitted by
`parse_tessera_format_name`, which takes any family `enumerate_grid_space()`
can build: twelve today, the eight `TESSERA_LM*` ones included, **13,068
rungs** against the menu's 6916 (measured, prismaquant#134). Sizing either off
the menu count would leave it 2x short the day something prices an `LM*`
family, which is the same drift one level up. So the bound is
`tessera_formats.grid_space_rung_keys()`, summed over the grid space at the
*grammar's* cap (`payload_bits` per code) rather than at `family_rate_cap`'s:
the recipe-dependent cap reads `tessera_wire_recipe`, so a bound stated against
it would call `_recipe_for` while sizing `_recipe_for`, and `tessera_wire_recipe`
never validates its rung, so every rung up to the grammar's cap is a reachable
key whatever the recipe says. 14,988 keys against 13,068 reachable, and the
over-count is free: an `lru_cache` does not preallocate. `_recipe_for` adds one
`rung=None` entry per family (15,000).

What the old bounds cost, measured on the memos rather than on a clock: over
one 6916-rung research pass the serialisable memo took **0 hits and 6916
misses, on every pass** -- the "0 hits / 13,832 misses" in the issue is two
such passes -- and `_recipe_for` took 34,580 hits and 6,920 misses on its
*second* pass over the *same* shape, every hit intra-rung (five lookups of one
key while that rung is priced) and not one across rungs. After: a second pass
takes 6916 hits / 0 misses, and so does a pass over a *different* shape, which
is the reuse these two exist for -- their answers are shape-independent and
`tessera_campaign` expands a menu once per Linear, so a model with 37,861 units
over 25 distinct shapes asks the same 6916 names ~37,861 times.

The lazy-sizing mechanism now lives once, in the leaf module
(`tessera_formats.lazily_sized_cache`); `menu_scaled_cache` is the *bound*,
which is the part only the menu knows. A bound that asks the memo it is sizing
is refused rather than answered uncached, because that is a design error in the
bound and a fallback would hide it.

tessera#46's own two memos are correct under this wider standard, checked
rather than assumed: `RouteAdmission.admits` returns False for a
non-serialisable family in *both* menu modes, so no `LM*` rung reaches the
shape-keyed memos. A 13,068-rung grid-space pass at tp=2 asks the serialisable
memo 13,068 times and fills `_shard_geometry` with exactly
`menu_rungs_per_shape()` = 6916 entries, which is its bound.

The two memos are sized apart because their entries differ by 25x–375x
depending on the column count, measured
with `tracemalloc` around a saturating pass and differenced against the same
pass with the memo cleared. A byte total is a `Fraction` and a tuple key: **351
B an entry at every shape tried** ((256,256), (2048,1024), (4096,16384)), so one
shape is 2.32 MiB and the default 25 shapes — the *counted* distinct 2-D Linear
shape roster of GLM-5.3-Flash, 25 shapes across 37,861 tensors — commits 57.9
MiB. A `_ShardGeometry` carries Tessera's whole Bresenham column schedule, one
integer per column: **8,735 B an entry at 1024 columns and 131,615 B at
16,384**, i.e. 58 MiB and 868 MiB for a single shape's rungs. Retaining 25
shapes there would be 1.4 GiB on GLM's narrowest expert and 21 GiB on its
widest, so it retains one — which is the whole self-eviction requirement, and
it is the memo that fills only at tp>1 (measured: 0 fills in a 6764-rung tp=1
research pass). Back-to-back on one box state under `cProfile`, the research
pass over a 2048x1024 unit goes from cold 22.05 s / warm 22.19 s (0.99x, 0
hits) to cold 21.72 s / warm 1.92 s (11.3x, 6916 hits against 6916 misses —
every rung priced exactly once); attested is untouched at 0.32 s on both sides,
as the issue said it would be.

**A window table is charged at the grid's own code width.** `PayloadGrid.
code_bytes` is one byte up to 256 codes and two on BF16, whose code *is* a bf16
word, so its ALPHABET plane is `2 * 2^L` bytes. Both accountants —
`tessera_formats.wire_overhead_q256` and
`tessera_footprint.tessera_tensor_payload_breakdown` — read it from the grid;
charging it at one byte under-priced the 16-bit route by half its table (0.0625
bpp on a 2048x1024 unit at L=14), and the byte budget is spent in this currency.
Pinned as exact `Fraction`s against `tessera.calculator.terminal_rate`, which
takes `code_bytes` from the same place.

**A TCQ body's forest is charged too, and it is why no Tessera rung has a rate
without a shape.** A trellis body writes an ALPHABET plane of `2^(R+1)` anchor
codes and a DESCENDANT plane of `2^(cap+1)` bytes *per distinct rate in the
schedule*, once per unit -- one byte per anchor whatever `code_bytes` is, and
independent of how many positions the unit has. PrismaQuant priced both at zero
until 2026-09-03, so every arity-1 E2M1 unit came out 24-44 B light and every
E2M1x2 unit at the coset cap 512 B light (RobTand/prismaquant#126); the
exporter's bytes were always right, only the accountant was not. The size is
taken from `tessera.grammar.forest_plane_bytes(rates, cap)` and never
re-derived here (P14) -- `wire_overhead_q256` has five terms now, and its TCQ
branch takes a `rung` because the schedule, hence the set of distinct rates,
hence the forest, is a function of the rung and the column count.

The consequence is a contract change, not a constant: `wire_overhead_q256`
requires `shape` on every recipe, `recipe_is_shape_free` is *gone* (its answer
became "no" universally, and a predicate with one answer is a comment), and
`TesseraFamily.artifact_q256_bounds` is gone with it (its premise -- that the
per-rung interval is the position-domain one shifted by a constant -- is false
in both shape and rung). Every synthesized Tessera `FormatSpec` therefore
carries `bits_for_shape_fn` and no `exact_bits_per_param`, so `effective_bits`
raises rather than quoting a rate that is not the artifact's. `R896` is not
4.0 bpp: it is 4.0556 on a 96x768 unit, 4.1333 on 96x320 and 4.0013 on
1024x3072. Three-way exactness -- `artifact_bpp` x
`tessera.control.unit_wire_bits` x `encode_linear(...).exact_bytes` -- is pinned
with no tolerance in `tests/test_tessera_forest_bytes.py`, and
`tessera_footprint.tessera_tensor_payload_breakdown` now hands the plane
builder the descendant blob it had been passing empty.

Stock `BF16` is unaffected and stays what it was, the passthrough rung of the
stock registry (§P11): a Tessera BF16 rung is named `TESSERA_BF16_K1_R<q>` and
is a lossy 1-to-16-bit trellis code over a bf16 alphabet, not a passthrough, so
`PASSTHROUGH_SOURCE_REQUIREMENTS` (which keys on the format NAME) never sees it.

**Three gates, and only three.** A rung is a candidate when all of them agree, and
each is a separate function so a refusal names which one:

| Gate | Function | Question |
|---|---|---|
| wire | `tessera_render.tessera_rung_is_serialisable` | can these bytes be written at all? |
| shape + TP | `tessera_menu.tessera_shape_legal` / `tessera_tp_legal` | does the rung tile the shard each rank will hold? |
| route | `tessera_menu.route_admission` | does a pinned runtime execute it? |

**`route_admission` is the one seam** (principle 14). It is the single function in
the production path that reads a serving contract on Tessera's behalf; the menu,
the campaign, the candidate gate, the lane stamp and the provenance all consume
its `RouteAdmission`. It is **deliberately not memoised**: a cache over a
runtime-contract read must not outlive the contract, and the key that would be
needed is the identity of the contract itself, which the cache cannot state. A
name-keyed `lru_cache` demonstrably inverted a pinned-release test that ran
after one which patched the attestation; putting the callables in the key fixed
that case and not the next (a test patching `_pinned_serving_table` changes the
contract without changing any function object, and the stale verdict came
straight back). The whole lookup is ~0.1 ms and `expand_tessera_menu` is
memoised above it.

**The default menu is three rungs under the dev pin, and empty without it — and
the reason is the PIN, not the table.** With no development contract pinned,
`route_admission` reads Tessera's **own** packaged `runtime_contract.json`
through `tessera_render._pinned_serving_table`, which publishes
`TESSERA_E2M1_K2`, `TESSERA_E4M3_K1` and (contract v5) `TESSERA_BF16_K1`, and
carries `device_qualified` native cells for the receipted rungs. What withholds them is
`_release_pin_satisfied()` — no reviewed Tessera release tag exists — so every
rung answers `unattested` (*absence of a claim*, not `unbacked`) and the default
`attested` menu holds none of them. `tessera_lane_admission` returns that reason
beside the verdict and `RouteAdmission.detail` carries it verbatim, because a
rung refused by the pin and a rung refused by an absent cell need different
fixes and must not read the same. `RouteAdmission.source` names the table that
answered — `tessera_packaged_contract:lane_eligibility:<version>` in production,
`tessera_dev_pin:runtime_contract:<commit>` under the override. It named
`gridbook_serving_runtime_pin:lane_eligibility` until 2026-09-02; that pin is
archived and stopped governing Tessera admission when Gridbook withdrew its
Tessera lane, so the string had become a claim about a runtime that no longer
answers — the unattested assertion principle 14 refuses, in the field a gate
reads (`tests/test_tessera_lane_admission.py`).

Tessera now packages its own contract (`tessera/serving/runtime_contract.json`,
schema `tessera.runtime-contract.v1`), read by `tessera_runtime_contract.py`. No
Tessera RELEASE tag has been cut, so production admission stays fail-closed; the
development override `PRISMAQUANT_TESSERA_DEV_PIN` opts in (any non-empty
value, recorded verbatim in provenance), and the installed contract's **answer**
— every value the *admission* gate reads, canonicalised by `contract_answer()`
— must equal `TESSERA_DEV_PIN_ANSWER` or the read raises with a field-level
diff naming what moved. It never degrades to `unattested`, because a stale pin that silently
empties the menu is exactly what the pin exists to prevent.

**The pin gates on the answer, not on identity** (issue #38, fixed 2026-09-02).
It used to compare the environment's value against an exact commit *and* the
file's sha256 against a recorded one. Both legs fired on identity, and the
thing they name is an editable checkout on the same box, so every Tessera
commit that touched a `detail` string, a changelog paragraph or
`contract_version` turned PrismaQuant's attested path off and seven tests red
in a repo nobody was editing — while the rungs' meaning had not moved. That is
principle 14 read backwards: prose explains and is never a value a gate reads,
so a prose edit is not a thing to re-review. `contract_answer()` draws that
line mechanically — families, reader rate ranges, attested rungs, world-size
ceilings, every cell field the route gate reads, and the canonical
`quant_method`; **not** `contract_version`, `plugin_version`, `attested_on`,
`detail` or `rationale`, which are identity or prose and travel into
provenance instead. A commit that moves no answer passes silently; one that
moves any answer — including *adding* a family or a cell, which Tessera's
changelog correctly calls additive for a reader and which is emphatically not
additive for an admission gate — refuses and says which field.
`TESSERA_DEV_PIN_COMMIT` and `TESSERA_DEV_PIN_CONTRACT_SHA256` survive as the
record of the review: the build and the bytes a human read when the answer was
accepted. Provenance carries both, plus the bytes *this* run read and a
`bytes_are_the_reviewed_bytes` flag, so prose-only drift is visible without
being fatal. This is deliberately weaker than a release pin and says so; a
Tessera RELEASE tag is what retires the override.

**Its scope is admission, and the split is deliberate.** `contract_answer()`
covers what the *allocator menu* is made of. The export lane reads three other
values out of the same file — `lane_eligibility.structures`, `platforms` and
`regimes`, through `lane_eligibility.load_eligibility_table` in
`tessera_export_lane.require_declared_structure` — and those are the RELEASE
pin's to gate: an exact commit and sha, fail-closed today on `PENDING`
sentinels. Folding them into the dev pin would make an export-lane edit
re-stale the allocator's menu, which is #38's own failure mode wearing a
different hat. Each pin covers the values its own gate reads.

Under that pin the attested menu is **three rungs, not a range**:

| rung | bpp on `[2048, 1024]` | route status | serve flags | max TP world |
|---|---|---|---|---|
| `TESSERA_E2M1_K2_R896` | 4.0000 | `backed_with_serve_flag` | `TESSERA_SERVE_MODE=resident\|streamed` | 1 |
| `TESSERA_E4M3_K1_R1024` | 4.0781 | `backed_with_serve_flag` | `TESSERA_SERVE_MODE=resident\|streamed` | 1 |
| `TESSERA_BF16_K1_R1792` | 7.1406 | `backed_with_serve_flag` | `TESSERA_SERVE_MODE=resident\|streamed` | 1 |

Each is one point of its family — the first two the family's native terminal
rate, 0.078 bpp apart; the third the one rung the BF16 route's receipt covered
— so **the attested menu has no rate axis at all** — the continuous axis is priced and allocatable
only under `PRISMAQUANT_TESSERA_MENU=research`, and widening it is a change to
Tessera's published `attested_rungs_q256`, not to anything here.

**Three since contract v5, and the third arrived by two independent changes
meeting.** The runtime published `TESSERA_BF16_K1 @ q256 1792` with two dense
`sm_121` cells; PrismaQuant learned to *construct* the family when
`ANCHOR_BUDGET_BITS` became the TCQ body's refusal rather than the grid's
(Tessera issue #14, §4.10). Either alone would have left the rung invisible —
a receipt for a family the producer cannot name, or a namable family the
runtime attests nothing of — and neither side was edited to accommodate the
other. The bpp figures are in §4.10's table; the row is
`TESSERA_BF16_K1_R1792` at 7.1406 bpp on `[2048, 1024]`, `w16a16-bf16-channel`. Research mode
admits every serialisable rung and stamps `route_status=unattested` on each so
the export gate fails closed on it (principles 1 and 9: an honestly priced rung
is never removed from the menu; the *export* is where an unbacked route is
refused).

Route status and serve flags are read off the attesting cell rather than typed:
these cells say `backed_with_serve_flag`, and collapsing that to `backed` would
both overstate the claim and drop the `TESSERA_SERVE_MODE` the serve needs.

The priced A side is compared against the executed one before any of that.
`route_admission` prices `tessera_serving_route`'s layout fact while the serve
executes the attesting cells' `activation_contract`, and the two vocabularies
do not match character for character, so `check_tessera_activation_agreement`
compares their `(act_bits, act_group_size)` projection
(`tessera_runtime_contract.cell_activation_projection`). Cells that disagree
with each other, a cell vocabulary the projection does not transcribe, or a
priced triple the cells do not execute all raise rather than admitting: pricing
one A side while the serve executes another is a currency error
(`tests/test_tessera_lane_admission.py`).

**REQUIREMENT: this menu may not be selected on the surrogate alone.**
Measured 2026-09-02, served
(`docs/measurements/tessera-allocated-served-2026-09-02.md`): three of this
menu's own allocations were exported and served on Qwen3-0.6B against a
**byte-matched uniform** arm, and lost at every budget — KL-vs-BF16 **0.3485 vs
0.1746** at 4.0 bpp (2.00x), 2.33x at 3.0, 2.88x at 5.0, with bytes exact to
the bit and 112/112 modules on the declared route. The loss is *in the cost
model, on the units the cost model priced*: a separator pair serving only the
seven measured Linears reads 0.02517 allocated vs 0.01306 uniform (1.93x, 95%
of the whole-body gap in log terms) while the same seven units score 1.13x
**better** in the allocator's own `h_trace × output_mse`. It is a
rate-distortion **slope** error, not an ordering error: the surrogate charged
the deep cut (`down_proj` R1006→R749) at 3.69x, its largest penalty and the
right sign, and scored the six remaining moves as a 1.30x net win where serving
says 1.19x loss. So:

* `SELECTION_MODE=validated-surrogate` is a **requirement** of this menu, not a
  suggestion. A Tessera `layer_config.json` produced under
  `SELECTION_MODE=surrogate` is a *candidate*, and the allocator now prints a
  warning saying so and stamps `tessera_menu.selection_caveat` into provenance
  (`tessera_menu.surrogate_selection_caveat()`), so the status is a property of
  the artifact and not of a terminal.
* **Necessary but not sufficient**, and this is the second requirement: the
  validated frontier re-scores the *allocator's own* Pareto points, all of them
  surrogate-allocated (`select_validated_frontier.py` and
  `validate_assignments_kl.py` contain no uniform/baseline/control arm in 3471
  lines), so it can rank 4.0 against 5.0 but cannot see that *uniform at 4.0*
  beats *allocated at 4.0*. The gate that caught this is a
  **byte-matched uniform arm served beside the candidate** — two serves, and it
  inverted the answer. Principle 3 applied to allocation itself.
* `COST_MODE=aura` is deliberately **not** named as the fix. No measurement of
  AURA on a Tessera rung exists in this tree's receipts (`docs/measurements/`
  holds none; the two auto-memory notes that mention both say "in principle,
  not yet run"), and naming an unmeasured estimator as the repair would be
  exactly the assertion principle 14 forbids. It is a candidate for the repair,
  to be decided by measurement.

The repair is not in this seam. Every construction downstream of the cost —
surface interpolation, dominance pruning, bin collapse, the group Minkowski
fold, the DP — is exact *given* the cost and adds no independent check on it.
The one structural guard that exists is ordinal: `TesseraRateSurface` refuses a
non-monotone anchor set rather than laundering it into a cost (2 of 21 surfaces
refused on the 0.6B group table). A slope error inside a monotone surface is
invisible to it, which is precisely the measured failure.

**The `TESSERA` token expands to priced-and-attested, and says how much it
narrowed.** A cost table is priced under whatever menu mode the *campaign* ran;
the attested set is a property of the *runtime*. A research-priced table read
back on the default path therefore holds thousands of columns the pinned
contract does not publish, and `expand_menu_tokens` used to hand every one of
them to `require_producer_formats`, which refused the **whole menu** — so the
default path could not allocate at all and the two backed rungs never reached
the DP. The token now intersects the priced columns with
`format_is_producer_eligible`, the same predicate the guard refuses on, so it
can never expand to something the guard then rejects (one rule, two uses). The
narrowing is printed — `[alloc] Tessera menu: 2 of 2423 priced rungs are
attested by the pinned runtime` — because an allocation over two rungs and one
over three thousand must not look the same in a log (P12); an **explicitly
named** unattested rung still refuses, so only the token narrows; and an empty
intersection refuses with the reason and both fixes named. Measured on the 0.6B
group table under the dev pin: floor 4.000 bpp (7 × `E2M1_K2_R896`), ceiling
4.042 bpp (4 × `E4M3_K1_R1024` + the `qkv` group on E2M1, because q/k priced no
E4M3 rung at R1024), and `--target-bits 3.0` refuses as infeasible. **`2 of
2423` is that table's number, not the menu's width**: the table was priced
before `TESSERA_BF16_K1` was constructible and holds no BF16 column, so the
third attested rung has nothing there to match. Re-verified on the same
`cost.pkl` after the family landed — still `2 of 2423`, which is the correct
answer for a table of that vintage and the wrong one to quote as the menu. Note which of the two attested
rungs is measured: `E2M1_K2_R896` is the family cap and is a **measured
anchor** on all seven units, while `E4M3_K1_R1024` is an **interpolated**
column of the rate surface on the five units that have one — so the attested
menu's E4M3 leg rests on interpolation in the currency impeached above.

**Tensor parallelism has two legs, and both bind.** The *attestation* leg is the
contract's `tensor_parallel` block, whose semantics are `closed_world`: it lists
both families at `max_world_size: 1`, so **no Tessera rung is attested at TP > 1
at any shape**, and `tessera_tp_world_attested` refuses in the attested menu
before geometry is consulted. The research menu passes that leg by construction
(it prices unattested rungs deliberately and stamps every one). The *geometry*
leg is below, and a refusal names which leg answered.

**Tensor parallelism is a per-unit legality input.** A Tessera rate is a schedule
over the reduce dimension, so what a rank can encode is a function of *its* column
count. Column-parallel Linears (q/k/v/gate/up) shard `out_features` and keep every
column — their realisable set is unchanged at any TP degree; row-parallel ones
(o_proj/down_proj) shard `in_features`, and at TP=8 a 1024-column unit becomes a
128-column one with a **far** smaller realisable set: 3060 → **17** rungs on
`[3072, 1024]` (3060 at TP≤4). Packed experts are whole units under expert
parallelism. The degree and the kind come from the serving profile's
`tensor_parallel` block (`serving_profiles.TensorParallelSpec`, enumerated regex
rules — no name heuristics), the granularity from **one** function,
`tessera_menu.tessera_shard_granularity`, which asks
`tessera.layout.shard_granularity` (since Tessera `f3e7d0a`) by handing it the
unit-shaped geometry the rung implies at that shape. The degree the candidate
was gated for is stamped into its provenance.

That delegation changed the answer, and the change is the reason principle 14
exists. This module used to derive the column period from the scale plane alone
— 32 for E8M0 groups, 16 for LUT half-blocks, **1 for a CHANNEL plane** — and
that last number is wrong: Tessera raises the column period to the 256-column
superblock for any rung whose Bresenham schedule is *mixed*, because the quota
`sum(rates) == root × columns` only closes on a whole superblock. Every rung
but the handful at integer rates is mixed, so the honest TP constraint on a
row-parallel Linear is `cols / tp % 256 == 0`, which is what collapses 3060 to
17 at TP=8 and what the local derivation would have admitted silently.

**One function owns the encoder call, and it is H-gated**
(`tessera_render.encode_tessera_unit`). The Tessera encoder's shipping default
is *H-aware*: given a unit's `H = XᵀX` it applies LDLQ (sigma 1.0, block 32)
plus an exact full-H row-scale refit, and weights-only encodes stay
byte-identical. So a rung priced without an H is a price of different bytes at
the same format name, and the seam refuses rather than downgrade silently:

* **The seam is `tessera.export.ActivationSource`**, not a kwarg name.
  `ActivationSource.for_unit(name, in_features, device, scale_plane=...)`
  turns one unit's H into the encoder keywords it implies — `ldl`,
  `ldl_block`, `refit_metric`, `refit_reach_floor` — and PrismaQuant forwards
  that object's output and
  nothing else: a keyword it did not emit is refused, so this seam cannot
  become a second place where encode settings are chosen. Capability is
  *probed*, never assumed (principle 14): `_encoder_accepts_hessian` asks a
  throwaway source which keywords it emits and
  `inspect.signature(encode_linear_planes)` which it takes, and the seam
  refuses when they disagree. Deliberately *not* "call and catch `TypeError`",
  which would swallow every unrelated argument error.
* **Which rungs can take an H is a fact about the WIRE, and it is DERIVED —
  never restated.** The paragraph that stood here quoted two Tessera guards
  verbatim ("LDLQ is implemented for the CHANNEL scale plane", "read only by
  the CHANNEL plane's refit") and concluded that the W4A4 route had no H-aware
  encode at all. Tessera's 2026-09-02 release deleted both — its own receipt
  says why, *"That reason does not survive reading the loop it guards"*: the
  plane is read once per pass, before the block loop, whatever its column
  granularity. `refit_metric` is refused only on S6b now and `ldl` on no plane
  at all. The restatement went stale where nothing could see it, which is
  exactly what principle 14 forbids a producer-side claim about another runtime
  from being able to do. So `tessera_render.rung_accepts_hessian` no longer
  states a condition: it **asks the pinned `ActivationSource` what it emits for
  this rung's plane** (`_encoder_kwargs_for_plane`) and answers on whether an
  H-bearing keyword (`ldl`, `refit_metric`) is in that set. A Tessera release
  that moves a keyword moves the predicate with it. The seam's forwarded-key
  whitelist (`_encoder_accepts_hessian`) takes the **union over planes**,
  because s6b emits no `refit_metric` and probing one plane would narrow the
  whitelist until it refused another plane's legitimate keyword as unknown.
  `hessian_applied` is stamped **per cost row**, not per table, because a
  mixed-family campaign is legitimately half H-aware.
* **The plane is a property of the family, and the memo is keyed by it.**
  Enumerated over every family PrismaQuant allocates (`enumerate_grid_space` ×
  `realisable_rungs` × `tessera_wire_recipe`, **12 families**, every rung): E4M3_K1 is WINDOW/CHANNEL
  L=14 at every rung (256–2048); BF16_K1 is WINDOW/CHANNEL too and climbs
  L=14→15→16 as its rate reaches its 4096 terminal, plane unmoved — it is the
  twelfth family, constructible only since `ANCHOR_BUDGET_BITS` became the TCQ
  body's refusal rather than the grid's (§4.10 — the constant is still 16; what
  moved is who is asked), and a plane table measured before that merge counts
  eleven; E2M1_K2 is LUT16
  at every rung and changes *body* at the q896 coset cap (WINDOW below it, TCQ
  at it) while keeping its plane through the change; E2M1_K1 and all eight LM
  families are TCQ/LUT16. So the plane is
  constant across rungs and differs across families — `encoder_kwargs` stays
  hoisted out of the rate sweep (no rate reaches it, and the signature has
  nowhere to put one) but its per-unit memo is keyed `(unit, plane)`, since the
  campaign prices one unit on every family in its menu and a unit-only key
  would price the second family under the first family's objective, silently.
  `scale_plane` is keyword-only with **no default**: a default would be one
  plane's measured answer applied to another plane's wire. The expensive half
  is unaffected — the block-LDL is a function of the Hessian alone, asserted in
  the test, because if it moved with the plane the hoist would be worthless.
* **H comes from PrismaQuant's own calibration activations, never the held-out
  / KL split**, accumulated as raw `XᵀX` over **every** calibration row the
  Linear sees — *before* the render score's `max_act_rows` cap, since a 256-row
  Hessian on a 3072-column `down_proj` is rank-deficient by twelve. It is
  passed un-normalised, because Tessera's `regularize_hessian` takes a count of
  its own and a second normalisation here would be a second spelling of the
  encoder's.
* **One formation, one identity, two callers** (`tessera_hessian.py`). The
  anchor campaign and `render_tessera_production` (under
  `ProductionWeightCache`) both call `hessian_from_rows` (fp32, one
  accumulation order, `[b, s, in]` and `[t, in]` giving one matrix) and
  `calibration_identity`, which emits exactly the triple Tessera requires —
  `text_sha256`, `fit_tokens`, `fit_ids_sha256`. Both shas travel: two
  tokenizers over one corpus are two calibrations a text sha would call the
  same, and one tokenizer over two corpora is two an id sha separates only by
  luck. The production render **requires** the triple in
  `levers['tessera_hessian_identity']` and refuses without it — bytes shaped by
  an H are not reproducible from the weights, so the capture has to be named.
* **The LDL is hoisted out of the rate sweep.** `ldl` and `refit_metric` are
  functions of the unit's H alone, so the campaign memoises the kwargs per unit
  and a twelve-anchor surface factorises one `[in, in]` matrix once instead of
  twelve times.
* **A missing qname is a hard failure** (`HessianContractError`), not a
  fall-through — this tree has already shipped a render whose activation lookup
  missed, silently rendered RTN and raised nothing. The error has its own class
  so the campaign's `except Exception: continue` per-anchor guard cannot absorb
  a contract refusal into a skipped anchor.
* **`--hessian {require,off}`**, default `require`, mirroring
  `render_production_weight`'s `ldlq_missing_activation_ok=False`. `off` prices
  weights-only *deliberately* and stamps `hessian.supplied=false` on every row
  and on the payload, with the identity (`text_sha256`, `fit_tokens`,
  `fit_ids_sha256`, source, split role, seed) beside it.
* **The provenance has a consumer.** `tessera_menu.assert_uniform_hessian_
  identity` refuses a cost table whose Tessera rows carry two different Hessian
  identities, and the allocator calls it at load and stamps the result into
  `__prismaquant__.tessera_hessian`. Unstamped rows are counted as
  *unstamped*, not assumed to match. Provenance that nothing consumes is a
  confession log; this one is a gate.

* **`render_production_weight` routes Tessera here, ahead of its format
  cascade** (`production_weight_cache.py`, `render_tessera_production`). Left to
  the cascade, a `TESSERA_*` unit fell through to the registry's synthesized
  `quantize_dequantize` — a *weights-only reconstruction*, not the decoded wire
  — so an allocator-chosen Tessera unit would have been cached, KL-scored and
  exported from bytes nobody encoded, silently on both counts. The interception
  forms `H = XᵀX` from `activations[qname]` (calibration rows, `.to(device)`
  explicitly — the CPU-residency landmine), refuses a missing key and refuses a
  column count that does not match `weight.shape[1]`. The deliberate opt-out is
  the `tessera_weights_only` lever, which whoever sets must stamp.

* **The predicate does not import Tessera.** All four sites below are on the
  hot path of every *non*-Tessera format, and `tessera_formats` /
  `tessera_render` both require the `tessera` package at import. So the
  question "is this mine?" is `format_registry.is_tessera_format_name` — the
  family's name grammar anchored at the start, the same line `get_format`
  already drew — and `tessera_render` is imported inside the Tessera branch
  only. Pinned by a subprocess test that blocks the `tessera` import.
* **All three cache-miss RTN fallbacks refuse Tessera** —
  `weight_session._format_weight` and `perturbed_x_cache` (both gated by
  `PRISMAQUANT_STRICT_PRODUCTION_CACHE`, default refuse) and `aura_cost`'s
  `dW` fallback (gated by `require_production_cache`, **default off**). Each
  falls back to the registry `quantize_dequantize`, which for Tessera is the
  weights-only reconstruction again — the same silence, one function further
  on. They now raise, as they already did for CB. `aura_cost` is the one that
  mattered most: it is the default `COST_MODE` and its guard is off by
  default. The orchestrator cannot reach it with a Tessera name today
  (`run-pipeline.sh:69` refuses `FORMATS` containing the `TESSERA` token
  before the AURA cost stage), but a direct call can.

`render_tessera_weight` remains the **weights-only reference** render. It is
reached only through the registry's synthesized `quantize_dequantize` — public
API, so it cannot be made unreachable. The four paths by which a *pipeline*
consumer could reach it are closed by name: `render_production_weight`
intercepts, and the three fallbacks above refuse. Anything else calling
`spec.quantize_dequantize` directly still gets it, and gets what it asked for.
It does not and will not take an H.

**A fused group is one family, not one rate** (`allocator_candidates.
tessera_group_composites`). A Tessera rung is a family and a rate glued into
one name, and only the family is what the runtime dispatches on. The
aggregation therefore gives each fused/packed group, per shared family, the
**Minkowski sum** of its members' `(bytes, cost)` menus -- the group's own
exact multi-choice knapsack -- kept as a Pareto set under **dominance**
pruning, never a hull (the budget is discrete; a point inside the hull can be
the unique optimum at one capacity). The outer DP chooses among those option
sets plus the stock per-NAME options; `expand_fused_sibling_assignment` reads
the per-member rung map back out, and `_promote_group_components` (already
family-aware through `format_promotion_class`) is a verified no-op on the
result. Intersecting by format NAME instead -- the previous behaviour -- forced
a shared **rate** that no runtime asks for, and on a continuous axis collapsed
a group's menu to whatever single rung its members happened to share: on
Qwen3-0.6B layer 0 it cost **1.247x / 1.294x / 1.122x** in Δloss at 3/4/5 bpp,
while the family constraint itself costs 1.008x / 1.000x / 1.000x against an
unconstrained bound. Exactness is enforced, not claimed: the fold is pinned
against brute force on a menu with a non-convex pocket, a uniform-rung option
must price identically through both constructions or the aggregation refuses,
and the fold refuses outright at `PRISMAQUANT_COST_UCB_Z > 0` because
`z*sqrt(sum stderr^2)` is not additive. Measured again on a cost table whose
anchors were placed per group (the correct placement) the one-rung constraint
costs **1.237x / 1.558x / 1.113x**, and at 4.0 the fold is shown to *contain*
the unconstrained arm's own rung triple, so the family constraint there is free
and the 0.003% residual is the outer DP's charged-bin quantisation.
`PRISMAQUANT_TESSERA_GROUP_KNAPSACK=0` disables the fold as a debug ablation --
that is the arm those ratios are measured against, and it stamps `__ablation__`
into the group report so a run cannot be mistaken for a default one.

**The licence is the contract's, and it narrows the fold as well as opening
it** (PrismaQuant #132, `allocator_candidates.tessera_group_composites`,
`tessera_menu.fused_module_licence`, `tessera_formats.fused_shared_signature`).
What one vLLM-fused module's roles may disagree about is a fact about the
serving runtime, so it is read from Tessera's `runtime_contract.json`
`fused_module.fields` block -- checked on the Tessera side against
`tessera.serving.scheme.FUSED_MODULE_FIELDS`, the dict the loader itself gates
on -- and never asserted here. It publishes `family`, `structure`, `grid`,
`body`, `plane`, `columns` as **shared** and `q256`, `rows` as **per_member**
(contract v6, RobTand/tessera#37). The read is one read per run, through
`tessera_menu.tessera_runtime_contract`, so the licence a group folds under and
the route admission that priced its rungs cannot come from two Tessera builds
inside one run. Four consequences, all of them enforced rather than described:

* `q256: per_member` is the licence for the fold at all. Withdraw it and the
  fold enumerates nothing, so the group keeps the per-NAME intersection, which
  asserts nothing per member. **With no contract pinned -- production today,
  since no Tessera RELEASE tag exists -- there is no licence to read and the
  fold likewise declines, where before this change it folded.** Either way it
  stamps `__licence__` into the group report, so a receipt can tell "the
  runtime says one rung" from "nothing was asked" -- but only where the
  question arises: a group with no Tessera rung on any member's menu returns
  before the stamp, so the per-NAME path's super items are untouched
  (`tests/test_allocator_sibling_aggregation.py::test_a_stock_menu_gains_no_group_options_at_all`).
* **holding the family fixed is not enough to honour the shared fields.**
  `tessera.export.wire_recipe` is a function of `(grid, q256)`, not of the
  family: on `E2M1x2` it writes a WINDOW body over LUT16 below the TCQ cap and
  a TCQ body at 896. Measured on the parent commit, a family-only fold offered
  `q_proj` at R896 (TCQ) beside `k_proj` at R850 (WINDOW) -- two `body` values
  in one module, on the frontier at 30 bytes, and `body` is `shared`. The fold
  now runs per **coherence class**, keyed by the shared fields a rung decides
  (`family`, `grid`, `body`, `plane`), so only rungs that agree on all of them
  are summed, and one family may now span more than one class.
* a `shared` field the fold cannot evaluate is a **refusal**, not a skip: the
  vocabulary is closed by `tessera_formats.FUSED_MODULE_RUNG_FIELDS` (a rung
  decides it) and `FUSED_MODULE_SHAPE_FIELDS` (it is fixed before the allocator
  chooses anything), and a `shared` field in neither raises. `columns` is the
  one shared shape field that a *profile* could violate, so it is checked where
  it can differ -- on the group the profile built, in
  `aggregate_fused_siblings` -- and a group whose members read different input
  widths declines the fold with the widths named in its receipt rather than
  failing the run.
* the block is inside `contract_answer`, so a Tessera commit that re-tightens
  it re-stales the pin with a field-level diff instead of silently changing
  what this producer may allocate.

**Serving premise attested for the LOADER, not for a serve**: the contract's
`mixed_rung_receipt` is `false` -- the relaxation is proven by a decode
identity (a real Qwen3-0.6B q/k/v at three rungs decoding element-for-element
to what the three roles decode alone), not by a vLLM serve. Principle 9's
export gate, not this, decides what ships. The remaining prose licence is
`allocator_solver._resolve_family_group`, which states the same relaxation in
its docstring and applies it to packed-MoE groups the contract does not cover
(`expert_parallel.units` is `[]`); tracked as PrismaQuant #140.

**Anchors are placed per fused GROUP** (`tessera_campaign`). The
`--max-artifact-bpp` cap is a WIRE bpp and the wire->body map is
shape-dependent (the CHANNEL plane amortises over rows), so solving each
member's top anchor independently put siblings on different body grids even
when they share `in_features`: `q_proj` topped out at R1388 where `k/v_proj`
topped out at R1372, and every bisected anchor below inherited the offset. One
grid per group, from the intersection of its members' realisable sets, and
every member measures the same rungs. The adaptive loop then keeps splitting
the worst-predicted interval on any surface still failing `--loo-gate` until
it closes or `--anchor-budget` is spent; `--max-rounds` defaults to 0 (off)
because a round adds ONE anchor per failing surface, so capping rounds capped
how far the worst surface could be improved -- the opposite of the loop's
purpose. Each surface records its anchors, encode seconds and which of the two
stopped it.

**Cost is an anchor campaign, not an enumeration** (`prismaquant/tessera_campaign.py`).

> **Premise correction.** This work was briefed on the belief that Tessera's
> "embedded rate axis" would let one deep encode per unit yield exact decodes of
> every lower rate. It does not, on the wire: the embedded axis is a decode-time
> **completion** axis, `encode_linear` writes `completion=0`, `build_unit_artifact`
> writes exactly one terminal, and a WINDOW body has no completion axis at all.
> Every rung of every family costs its own encode. That fact is what makes a
> campaign necessary rather than merely convenient, and it is pinned by
> `test_no_embedded_axis_on_the_wire`.

So the stage measures a few anchors per `(unit, family)`, fits a monotone
piecewise-linear surface in `(q256, log2 dloss)`, and **refuses to extrapolate**:
a rung outside the measured envelope is omitted from the cost table, not invented.
A non-monotone anchor set is recorded in `non_interpolable` and that family is
priced only where it was measured -- and a refused surface is reported as a
refusal, never as a perfect fit: `_surface_loo` returns `None` for its
leave-one-out error and `gate_closed: False`, and the adaptive loop skips
refused members when taking the worst-LOO maximum. (Reading the absent error
with a `0.0` default did the opposite, and closed the gate on exactly the
surfaces the loop exists to keep spending on.) Anchor placement is adaptive — a second round
splits the interval whose interior anchor its own leave-one-anchor-out check
predicts worst — and gated on a measured LOO error, not a taste constant.

Two properties make the numbers comparable with the rest of the menu:

* **Priced as served.** The render is scored under the route's own activation
  contract — NVFP4's W4A4 quantiser on E2M1 rungs, per-token FP8 on E4M3 — taken
  *by reference* from the serving format's registry row, so the A leg is not a
  second implementation of it. The same weight rate therefore costs differently on
  the two routes, which is asserted directly by
  `test_the_same_weight_rate_costs_differently_on_the_two_routes`.
* **Rendering identity (§P8).** The tensor that is priced *is* the decoded wire:
  `_encode_and_render` returns `read_unit_artifact(unit.blob)`, and the blob is
  stored beside the render in the cache. Not two code paths that agree — one
  object.

Rows are written in the codebase's own currency: `output_mse` and **no**
`predicted_dloss` field, because in this tree `output_mse` is a raw MSE while
`predicted_dloss` is already the `½·h_trace·mse` product. Writing the MSE into
both fields — which the first version of this stage did — prices a Tessera rung
`h_trace/2` times low against every other format on the menu; a test now pins it.
Interpolated rows join `cost_entry_is_band_interpolated`'s branch, so they are
priced from their own output-space fit and inherit
`drop_interpolated_candidates_dominated_by_measured`.

**Reducing the menu for the DP, and reporting which reduction did it**
(`allocator_candidates.reduce_continuous_menu`). Two reductions, kept separate
because they license different claims:

* **dominance** (`tessera_menu.prune_dominated`) drops a rung only when another is
  no larger in bytes **and** no larger in cost. Exact for any knapsack.
  **Explicitly not a convex hull** — the budget is discrete, so a point strictly
  inside the hull can be the optimum at one particular remaining capacity, and
  hull pruning drops exactly those points. Refused here rather than offered behind
  a flag.
* **bin collapse** (`tessera_menu.collapse_to_dp_bins`) drops a rung only when
  another lands in the same charged bin of *this* DP (`_charged_bins` at
  `--bit-precision`, against the unit's own cheapest candidate — the baseline
  `solve_allocation` actually uses). Exact for this solver at this precision, and
  no stronger.

Both counts, per unit, are stamped into `layer_config.json`'s
`__prismaquant__.tessera_menu`. Without them a coarse-looking set of selected
rates cannot be attributed: a campaign that priced few rungs and a bin width that
swallowed many look identical in the output. Non-Tessera candidates are
partitioned out before either reduction, so a run with no Tessera rung on the menu
is byte-identical to one built before the function existed.

**Promotion shares a family, not a rate** (`allocator_solver._resolve_family_group`).
Serving-unit promotion exists because fused siblings and packed experts are one
tensor to the runtime, so members cannot disagree about what the runtime
dispatches on. For a Tessera rung that is the **family** — the grid and arity the
decoder is compiled for — not the rung. `tessera_formats.format_promotion_class`
returns the family for a Tessera name and *the name itself* for everything else,
so the relaxation is unreachable on a stock menu by construction. Each member then
takes the cheapest legal rung of the shared family at or above its own rank (the
same non-degrading contract the uniform path has), and if any member has no legal
rung in the family the whole thing falls back to uniform promotion — the
relaxation can only widen what promotion accepts, never add a way for it to fail.
This matters at 0.6B already: `q_proj` is `[2048, 1024]` and `k/v_proj` are
`[1024, 1024]`, three different tensors fused into one `qkv_proj`, and the uniform
rule would force a shared *rate* where only a shared *route* is required.

**The per-member rates are licensed, and fused-only** (RobTand/prismaquant#140).
The paragraph above used to assert the serving constraint in prose, at a second
site, and for every serving-unit kind: `promote_serving_units` flattens fused
groups and packed-expert groups into one list, and the family branch fired for
any component whose max-rank format had a coarser promotion class. The groups
are now tagged by kind where they are built, and the branch runs only for
fused-kind components under the pinned contract's `fused_module` word for the
rate field -- `q256: per_member` since contract v6, read through
`tessera_menu.fused_module_licence`, the menu's declared one read, so the
licence and the route admission cannot come from two Tessera builds in one run.
A packed-expert component takes the uniform path even under a per-member word:
`expert_parallel` publishes a closed world with no unit in it, so per-expert
rungs under one family is a claim nothing attests. With no contract pinned --
production, while no Tessera release tag exists -- there is no table to derive
the rate's freedom from and every component takes the uniform path; a contract
that re-tightens `q256` to `shared` collapses fused groups the same way and
re-stales the dev pin with the field named, because the word is in the pin's
answer. A component that unions a fused group with a packed one is ambiguous
under both scopes and refuses rather than picking a side; on a stock menu the
uniform path treats every kind identically, so overlapping stock groups behave
exactly as before. `None` is the absence of a licence, never a permissive
default, and a `q256` word the block does not name raises instead of guessing.

**...but the default path never asks it to** (measured 2026-09-02). Fused
siblings are aggregated into a single DP item *before* the solve
(`allocator.py` → `allocator_candidates.aggregate_fused_siblings`), and that
aggregation builds one super-item candidate **per format name** (`for spec in
formats: … Candidate(fmt=spec.name, …)`). The DP therefore returns exactly one
rung for the whole group, and `_promote_group_components` — relaxation included
— is a no-op on it. The call site says so in as many words: *"The DP can't pick
mixed-sibling solutions because there's only one item per group."* The
relaxation is reached only when the siblings stay separate DP units, i.e. under
`--no-fused-aggregation`. Both arms are measured in the receipt.

This is also the honest reading of the **wire**: `bresenham_rate_schedule(root,
n_columns)` is a per-COLUMN quota shared by every row of a unit
(`tessera/grammar.py`). Siblings concatenated along ROWS into one unit *cannot*
carry different rates — the schedule is indexed by the axis they share. Free
per-sibling rates presuppose that the runtime decodes q, k and v as separate
units and concatenates after, which is a claim about a runtime and is therefore
not made here (principle 14). `tests/test_tessera_menu.py::
test_pre_aggregation_forces_one_rung_on_a_fused_group` pins the fact so a change
that makes the DP mixed-rung capable has to come here and say so.

**The reduction is re-applied after aggregation.** Super items are built from
`specs_sorted` directly and never pass through the per-Linear reduction, so
`allocator.py` runs `reduce_continuous_menu` a second time on the
post-aggregation dict (after `filter_candidates_for_profile`). Without it a
fused group carries the entire unreduced Tessera axis into the DP. Both passes
are reported, in `layer_config.json` under
`__prismaquant__.tessera_menu = {per_linear, aggregated}` — written on every
run, not only on the byte-budget path, which is where the first version of this
stamp landed and therefore never appeared in a `--target-bits` run.

> **Attestation, read rather than assumed.** Whether a runtime can serve a
> fused group whose members hold different rungs of one family is a fact about
> that runtime, and Tessera publishes it: `fused_module.fields.q256` is
> `per_member` since contract v6 (RobTand/tessera#37), and the promotion above
> reads that block instead of restating it. Two things the block still does not
> say: `mixed_rung_receipt` is `false`, so no *serve* has covered such a
> module -- the relaxation is proven by a decode identity -- and nothing in it
> covers per-expert rungs, which is why packed components stay uniform. This
> section describes what the *allocator* may consider; §9's export gate
> decides what ships, and under the default attested menu there are no Tessera
> candidates for the relaxation to act on. The export lane additionally refuses
> routed-MoE models outright (it reads the contract's `structures: ["dense"]`
> and the absent `routed_moe` cell), so no mixed-rung expert group can reach
> an artifact even where the allocator states one.

**The route travels with the choice.** `serving_lane_route` falls through to
`tessera_menu.tessera_resolved_serving_lane` for a `TESSERA_*` name when no profile
declares a lane for it — which is always, since no spec can enumerate a
continuous rate axis.
Without that fallback `selection_serving_lane_provenance` reports
`no_declared_lane` (absence of a declaration) where the truth is a declaration of
absence (`unattested`); principle 9 wants the second one countable, and it is now.
The fallback is deliberately *after* the profile lookup, so a profile that one day
declares a real Tessera lane overrides it like any other format.

**Wiring.** `FORMATS` accepts the token `TESSERA`;
`tessera_menu.expand_menu_tokens` replaces it with exactly the Tessera columns the
run's own cost table holds — the widest menu the DP could honestly consider, and a
set that needs no second copy of the campaign's legality decisions. `run-pipeline.sh`
refuses the token with a pointer to the campaign command until that stage is wired
into the orchestrator, rather than emitting a cost table with no Tessera column and
an allocation that silently contains none.

**What gates the interpolation, and what does not.** The rate surface is gated
by the campaign's own leave-one-anchor-out error and by a menu-density arm
(solve the same targets against a menu of measured anchors only, and compare).
That arm must hold **aggregation** fixed to mean anything: `aggregate_fused_
siblings` intersects the members' menus by format *name*, a Tessera name
carries its *body* rate, and the campaign's top anchor is a *wire* cap
(`--max-artifact-bpp`) whose body equivalent is shape-dependent through the
plane's row amortisation — so `q_proj` tops out at `R1388` where `k/v_proj` top
out at `R1372`, every bisected anchor inherits the offset, and on a
measured-only table the qkv E4M3 menus intersect to a single rung. Comparing an
aggregated sparse arm against an aggregated dense one measures that collapse,
not the interpolation. (The collapse is a property of per-unit anchor
placement, not of discrete menus: anchoring a fused group on one shared body
grid would dissolve it without interpolation. Not implemented.) The
pre-existing
`allocator_candidates.drop_interpolated_candidates_dominated_by_measured` is
**not** in that path: it is referenced by tests and docstrings and has **no live
call site** in this tree, so it fires on nothing, Tessera or otherwise. Stated
here rather than quietly relied on.

**DP wall time is stamped, not re-timed by hand.** `allocator.py` accumulates
`solve_with_promotion` wall time across the tightening retries a target needs
and writes `solver_seconds` / `solver_calls` per target into
`layer_config.json` `__prismaquant__.solve_diagnostics` (and into
`selection.json` on the byte-budget path). `_solve_diagnostics` was previously
read only by the infeasibility message, so what a solve cost was invisible to
everything downstream.

**`pipeline.py` needs no entry for the token.** The declarative contract names
`FORMATS` only as a *cache-key ingredient* (`"FORMATS<-COST_FORMATS"` and
friends); it does not enumerate legal format names anywhere, so `TESSERA` is
carried by `run-pipeline.sh` and `allocator.py` alone. Checked, not assumed.

**Gate:** `tests/test_tessera_menu.py` (34) + `tests/test_tessera_campaign.py`
(35; several are CUDA-marked),
on top of `tessera_{formats,footprint}`'s own suites.

**Measured on Qwen3-0.6B layer 0, 2026-09-02**
(`docs/measurements/tessera-continuous-menu-2026-09-02.md`). Menu 3039-3063
rungs per unit; 102 anchors expand to 16893 priced rungs (166x). The DP lands
within 3e-4 bpp of a 3.0 / 4.0 / 5.0 target. Against measured anchors only, at
matched aggregation, it costs **1.09-1.41x** in Delta-loss and is 0.052 /
0.095 bpp short of the 4.0 / 5.0 budgets — and *with* aggregation it cannot
reach the budget at all (4.639 at a 5.0 target), because the sibling
intersection leaves E4M3 with one rung. Pre-DP fused aggregation costs 11-23%
in Delta-loss at matched bpp. Trusted-cost error at the chosen rungs (17 fresh
encodes) is 0.999 / 0.824 / 0.968 true-over-predicted, conservative at all
three, while per-rung it spans 0.68-1.27 and the E4M3 LOO gate does **not**
close on 4 of 7 units. Rung labels are *body* rates: the 3.0 assignment's
param-weighted body is 2.929 bpp and its charged wire 3.000 bpp, and one rung
(`R814`) costs 3.258 bpp on `q_proj` but 3.320 on `k_proj` because the plane
amortises over fewer rows. DP time: 11 solves, 2.32 s, over a 4315-rung menu.

**Not done, and not claimed:** no export leg, so nothing here has been served and
no KL — screen or gold — was measured on a Tessera allocation. **And no anchor
was priced with a Hessian**: the seam above is built, gated and tested, but the
H-aware encoder branch is not in the pinned Tessera, so every measurement cited
here is a `--hessian off` weights-only price. See
`docs/measurements/tessera-continuous-menu-2026-09-02.md` for the receipt and the
open list.

## 5. Formats & render

### 5.1 The menu

`format_registry.py`. For non-CB formats, `FormatSpec` byte accounting is
shape-exact rather than a nominal scalar:
`scale_count_for_shape` (`:123-155`) handles `scale_block_shape`, per-channel `group_size==0`,
and 3-D packed-expert stacks; `memory_bytes_for_shape` / `effective_bits_for_shape`
(`:157-168`) are what the DP, `footprint.py`, and the Pareto table consume for
those formats. Aliases:
`FP8`/`FP8_DYNAMIC` → `FP8_E4M3`, `MXFP8` → `MXFP8_E4M3` (`:170-188`).
`act_quant_changes_input` (`:75-106`) is the **single** predicate for "does the serving kernel
consume quantized activations" (`act_bits` absent or ≥ 16 ⇒ no): the allocator's bit-exact
short-circuit (§4.5), the KL validator's activation-quant assignment, `layer_state_cache` and
`perturbed_x_cache` all key off it, so a format's activation semantics cannot drift between
pricing and emulation. Registry-vs-callable consistency is pinned by
`tests/test_bit_exact_cost_pricing.py`.

| Format | line | w-bits / group / scale | eff. bpp (2-D) | Status |
|---|---|---|---|---|
| `NVFP4` | `:667-677` | 4 / 16 / fp8 e4m3, A4 g16 | 4.5 | Production, in the default menu (§3.3) |
| `FP8_E4M3` | `:750-762` | 8 / per-channel / fp32, A8 per-token | 8 + 32/in_f | Production (`FP8_DYNAMIC`), in the default menu |
| `BF16` | `:798-808` | 16 / — | 16 | Production, **passthrough only**, in the default menu |
| `FP8_SOURCE` | `:825-837` | 8 / block (128,128) / fp32 | 8.00195 | Source-artifact compatibility, **passthrough only**, verbatim copy; excluded from maintained SM120 production/chooser profiles |
| `MXFP8_E4M3` | `:720-728` | 8 / 32 / e8m0 | 8.25 | Registered, profile-allowed, **de-menued** |
| `NVFP4A16`, `MXFP4`, `MXFP8A16`, `MXFP8_E5M2`, `FP8_E5M2`, `INT8_W8A16`, `INT4_W4A16_g128` | `:678-795` | — | — | Research / registry-only |
| GGUF k-quants + IQ | `:884-902` | `_make_gguf_spec :864` | 2.0625–8.5 | GGUF lane (§9.3) |
| `TESSERA_<grid>_K<arity>_R<q256>` (e.g. `TESSERA_E2M1_K2_R896`) | synthesized, `tessera_render.py` via `format_registry.py:1322-1338` | E2M1 (4-bit) or E4M3 (8-bit) base, arity 1–2, body rate in 1/256-bpp steps, scale plane from the family's wire recipe; route priced (W4A4 / W8A8 / kernel) | exact per shape (`bits_for_shape`); 4.00 at `E2M1_K2_R896` on (2048, 4096) | **Research** (§5.7): priced and rendered by name; producer-eligible only through `tessera_lane_attested`, which reads Tessera's own packaged contract and is **False under the PENDING Tessera serving pin** (no release tag exists); lane spec declared (§9.4), no exporter in this repo |
| `NVFP4_CB_K1..K25`; FP8-CB reader K4/K8/K12/K16/K20/K24 + every K28..K48; FP8-CB producer K4..K48 step 4 | `cb_layout.py`, registry construction in `format_registry.py` | product-VQ codebook, g256 | NVFP4 v2 body `k/8 + 0.28125` bpw; FP8 index stream `k/8` bpw plus per-row FP32 scales; both add a deduplicated codebook sidecar | NVFP4 reader/producer domains are both every integer K1..K25; K26..K32 exist only in direct lattice/codec research and have no public format id. New endpoint artifacts remain Gridbook-v11 route-attestation and measurement gated. New FP8-CB producers emit exactly K4/K8/K12/K16/K20/K24/K28/K32/K36/K40/K44/K48; reader-only off-law K29–K47 artifacts remain inspectable but cannot enter a new menu, assignment, bundle, or export. Strict Ada profile allows only this FP8 ladder + `FP8_E4M3` + BF16 (§9.2) |
| ~~`NVFP4_CB_S13..S16`~~ | — | signed codebook, g256 | — | **DELETED 2026-08-17** — every native Gridbook FP4 route requires the unsigned two-tier product layout (`n_sub == 2 and type_size == 4*k + 9`), so a signed rung could only ever ride a fallback; it had already lost 78.48% of matched weight-MSE comparisons. Registry, encoder, exporter and footprint branches removed; `cb_layout.subtable_bit_widths(..., "signed", ...)` and a recipe carrying `cb_mode: "signed"` now **refuse** rather than resolve to the product rung of the same `k` (§9.2) |

MXFP8 is de-menued rather than denied — `vllm_packed_moe` still allows `MXFP8_E4M3` — because
its E8M0 pow2 scale wastes ~√2 of a binade and exact-scale FP8 Pareto-dominates it; offered
both, the allocator never picks it.

`FP8_SOURCE`'s `quantize_dequantize` is identity (`:835`): the bf16 view *is* the lossless
dequant of the source E4M3, so cost is exactly zero. Legal on dense Linears under
the generic `vllm_packed_moe` compatibility profile, **illegal on packed experts** (absent
from the expert allow-list — §6.4), and explicitly denied by the maintained SM120 profile.
Registry/read compatibility is intentionally broader than target performance eligibility.
CB candidates deliberately use a different byte authority: the versioned
`CBSerializationContext` in `nvfp4_cb_footprint.py` prices the exact FP4 layout,
FP8 per-row FP32 scales, and deduplicated sidecar identity. Candidate
construction, assignment accounting, reports, and exporter assertions share
that context; exact post-export inventory remains the final artifact check.

NVFP4 endpoint geometry is explicit rather than inferred. K1 splits `(1,0)`
and carries a one-row zero-bit second subtable; public K25 splits `(13,12)`.
Production v2 type sizes are 13 and 109 bytes per superblock respectively. The
FP16 sidecar is `8·(2^ceil(k/2)+2^floor(k/2))` bytes: 24 bytes at K1 and
98,304 bytes at K25. Every d4 width 0..16 is materialized in the pinned asset,
but widths 14..16 and the direct `(16,16)` uint32 codec boundary are explicitly
research-only; public production lookup K1..K25 refuses a missing key.
Historical width-6..12 tensors remain unchanged;
`fp4-d4-nested-e2m1-v3` makes widths 0..5 nested subsets of width 6 and widths
13..16 nested supersets of width 12, so widening the new bands cannot worsen
nearest-codeword distortion solely through table replacement. Width 16
necessarily has duplicate physical rows because the 15-value E2M1 grid
contains only 15^4 distinct d4 vectors.

The FP8 reader/producer split is deliberate compatibility, not rounding. The
reader accepts the six low step-four rungs K4..K24 and every historical integer
rung K28..K48. `list_producer_formats()` and every strict explicit-menu check
accept only the twelve `k % 4 == 0` rungs from K4 through K48; K29, K43, K47,
and every other off-law legacy wire id remain readable but are never silently
mapped to a neighboring rung. `FormatSpec.min_capability_sm=89` says the format
can be considered on Ada; production legality still requires the exact
platform/regime `device_qualified` cell from the external Gridbook v11
lane-eligibility contract (§9.2), so registry admission alone makes no 4090
support claim (`cb_layout.py`, `gridbook_format_contract.py`,
`gridbook_execution_contract.py`).

NVFP4 *weight* RTN routes through the export codec (`_nvfp4_export_aligned_rtn` `:636-663`) so
emulation and shipped bytes share one rendering; *activations* do not — per-group dynamic RTN,
because the codec's per-tensor global scale would make emulation batch-dependent while serving
uses a static `input_global_scale` (`:674-676`). The `torch.compile` RTN hot path is
MSE-identical but not bit-identical to eager (~0.036% of elements flip at codebook midpoints,
`:445-458`).

### 5.2 Scale rules and JSO

NVFP4 scale rules live in the *exporter*, not the registry
(`export_native_compressed.py:111-132`): `static_6` (default), `four_over_six_mse`,
`joint_mse`. `joint_scale_opt` / `joint_scale_optimization` / `codebook_mse` are all aliases of
`joint_mse` — three names, one rule.

- **JSO = `joint_mse` evaluated inside the GPTQ loop**, per-group levels default `(6.0, 4.0)`
  (`_parse_joint_scale_levels :292-310`). The full 7-level grid collapses to {6,4} for 99.998%
  of groups at +0.009% aggregate weight-MSE, and the trim is monotone: a genuinely hurt Linear
  can only be *promoted* to FP8/BF16, never silently degraded. Override
  `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`.
- `static_6` is the `PRISMAQUANT_NVFP4_SCALE_RULE` env default, governing non-JSO RTN renders
  only; `four_over_six_mse` is a separate, non-JSO rule. Do not conflate the three. Single
  selection point for RTN / GPTQ / scale-sweep / packed / export: `_select_nvfp4_group_scales`
  (`:316-349`).

### 5.3 GPTQ damp

Sweep **OFF** since 2026-06-12 (`gptq_damp_sweep_enabled` `export_native_compressed.py:2543`,
env default `"0"` at `:2555`), fixed damp **1.0** (`_resolve_gptq_fixed_damp :2558-2577`). The
sweep's evaluator was in-sample; the V1 served A/B had fixed damp winning every gold-lane
readout across calibration draws at ~4.4× less render time.
`PRISMAQUANT_GPTQ_DAMP_SWEEP=1` reproduces historical artifacts, `PRISMAQUANT_GPTQ_DAMP`
overrides the constant, per-role overrides at `:2586-2640`. The second reader that used to
default the same variable to `"1"` (`kl_sensitivity_probe.py`) was a forked copy of the lever
defaulting; it now delegates to `production_weight_cache._resolve_production_render_levers`
(`kl_sensitivity_probe.py:272-285`), so there is one default. §12 D5, FIXED 2026-07-30.

### 5.4 The single rendered-weight store

`ProductionWeightCache` (`production_weight_cache.py:137`) is the only store for rendered
weights and `render_production_weight` (`:1785`) the only producer. Not tidiness: the
surrogate, the KL validation, and the exported bytes must be the *same* rendering, or every A/B
carries a rendering confound. Levers are recorded on the cache (`:165`, `:835-858`), which is
what makes M19 (§6.1) possible.

**Profile-synthesized MTP is an append scope of that same cache.** Transformers does not
instantiate the Qwen3.5/3.6 MTP sidecar, so the ordinary resident body walk cannot discover
`mtp.*`. `mtp_production_cache.fill_profile_mtp_production_cache` asks the resolved
`ModelProfile` to build the module, reads the exact `mtp_source_prefix()` tensors, loads them
through the profile's packed-expert-aware loader, and reuses
`streaming_production_cache._render_dense_layer`; it does not introduce another renderer or
residency mechanism. The probe's existing activation rows supply GPTQ/JSO inputs. Profile,
source tensor, module-name, activation-row/width, requested-pair, and resulting cache coverage
are all checked exactly before the append can succeed. An explicit non-BF16 MTP assignment
therefore requires `--activation-cache-dir` and fails closed rather than degrading to RTN or
BF16.

Compatibility is deliberate. Assignment scope appends only its concrete non-BF16 `mtp.*`
pairs. Format-menu scope without an activation-cache directory retains the historical
body-only behavior; with that directory it appends every live MTP Linear for every requested
non-BF16 menu format. `--include-qnames-file` filters both the body walk and the synthesized
append: a stripe with no `mtp.*` names creates no MTP entries, and the MTP-owning stripe
creates only the listed names. The cache updates the ordinary `requested_entries`,
`render_scores`, failures, activation maxima, and backing shards, plus an exact
`prismaquant.production_weight_cache.mtp_render.v1` receipt binding qnames, formats by qname,
source prefix/count, activation row counts, and row limit. CB MTP renders are refused until
their identity-bound CB pair contract can be merged; the supported production-cache append
is the native materialized lane (`mtp_production_cache.py`, `build_production_cache.py`).

**The packed-MoE expert append carries the same render-identity contract.**
`fill_packed_expert_cache_entries` adds 3-D `(experts_qname.pn, fmt)` keys to that same cache,
and it updates the same metadata the union reads: `requested_entries` **recomputed** from the
cache rather than incremented (the M4 lazy expert gap-fill calls the append in a loop over
frontier points with overlapping assignments, and an increment drifts on every rerun), one
render-score record per packed key, one render-gate record per packed key, and
`packed_expert_coverage`. Records are replaced by key inside one exact packed scope, never
appended. The score is **measured, not declared**: every expert in the stack is scored by the
same scorer the dense path uses, against its own routed rows, and the per-expert records are
summed into the one record the cache key owns — experts the router never reached are counted
in `experts_without_activations` rather than averaged into an output metric they have no
evidence for. The gate record is truthful about a path that runs no progressive gate: an
**empty trace**, the mechanism that actually executed (batched fixed-damp GPTQ, or the
untraced per-expert render stack), and the per-expert GPTQ-vs-RTN do-no-harm counts the render
did measure. No declared exclusion or subtraction is added to the union arithmetic. The
streaming builder merges rather than overwrites these fields, because its own
`render_formats_by_qname` plan is dense-only (`production_weight_cache.py`
`_finalize_packed_expert_cache_metadata`, `streaming_production_cache.py`,
`tests/test_packed_expert_union_identity.py`).

**A streamed CB format menu is a lifetime mode of that same cache, not a second cache.**
`build_production_cache --streaming --render-scope format-menu` routes the complete requested
CB menu through `streaming_production_cache.py`; there is no disk-budget candidate filter.
For each `(Linear, rung)`, the producer canonicalizes the render exactly as a persisted cache
shard would, makes that tensor available to one synchronous consumer, and waits for the
consumer's durable acknowledgement. The pair sidecar then binds the full CB render identity,
the canonical tensor SHA-256, render-score digest, and consumer acknowledgement. Only after
that checkpoint is accepted may the rendered tensor be evicted and its weight shard omitted.
The manifest refuses to close if any eligible requested pair is missing or unacknowledged.

The transient path is **CB-only**. Its correctness rests on measured bit determinism under a
pinned render context; non-CB/GPTQ menus remain materialized until they have their own direct
repeat-render proof. The CB context remains fail-closed and includes scale coding, codebook
source (and immutable bundle/value identity when learned), scale-sweep choice, LDLQ scope,
min-chain mode/version, and encoder tier, plus the exact source-weight, imatrix/`col_weights`,
calibration, layout, and renderer identities (`production_weight_cache.py` CB pair identity
and `tests/test_col_weights_render_identity.py`). None of the allocator's incomplete-context or
`col_weights`-identity gates is relaxed.

Selected bytes are verified, not assumed. The cost-stage sidecar's canonical tensor digest is
the commitment to the render that was scored. The later assignment-scoped cache/export pass
re-renders only the winning rung, retains it as before, and compares its canonical digest with
that commitment; a mismatch is a hard failure. Thus streamed menu scoring bounds transient
disk residency while the surrogate, selected-assignment KL, and export still share one
bit-identical rendering. Assignment-scoped cache builds and materialized format-menu/frontier
builds retain their existing shard semantics.

**Exact multi-host striping is a plan plus a fail-closed set union, not tensor
parallelism.** `production_cache_stripes.py` reads positive-size qnames from `probe.pkl`, keeps
each decoder layer whole, makes `mtp.*` and `lm_head` separate indivisible auxiliary groups,
and asserts that the profile's fused-sibling and packed-expert serving units cannot cross a
group boundary. It balances those groups deterministically with longest-processing-time
binning over `n_params × max(in_features, 1)`, preserving layer-local source prefetch. The
output schema `prismaquant.production_cache_stripe_plan.v1` contains the probe digest,
model/profile, requested formats, work/parameter counts, and SHA-256-bound
`stripe-NN.qnames.txt` files. Workers run ordinary, independent
`build_production_cache --include-qnames-file ...` calls into distinct disk-backed cache
bundles; no worker shares a cache directory or tensor file.

`union_production_cache.py` owns the portable reconciliation workflow:

```
python -m prismaquant.union_production_cache manifest ... (--assignment A | --stripe-plan P)
python -m prismaquant.union_production_cache verify-shard ... (--assignment A | --stripe-plan P)
python -m prismaquant.union_production_cache union ... (--assignment A | --stripe-plan P)
python -m prismaquant.union_production_cache verify ... (--assignment A | --stripe-plan P)
```

Every subcommand requires exactly one coverage authority. `--assignment` means the exact
non-BF16 pairs in the selected layer config. `--stripe-plan` SHA-verifies every qname file and
means the exact Cartesian product of all planned qnames and formats; its semantic plan digest
is part of coverage. A worker manifest may cover only its disjoint subset, but the final union
must equal the complete expected key set with no extras. Each shard manifest SHA-binds its
pickle and every backing tensor and binds a path-independent source-checkpoint identity,
calibration hash, full producer commit and source hash, explicit settings payload, render
levers/scope/retention/streaming state, requested formats, mechanism order, and assignment or
stripe-plan identity. Missing/tampered/out-of-bundle/symlinked/shared files, in-memory or
non-materialized caches, render failures, overlaps, activation-max overlap, identity drift,
and unknown differing metadata all refuse. Resident caches and streaming caches are both
eligible only when their rendered weights are materialized; transient consumed-and-evicted CB
menus are not.

The merge has explicit rules for `requested_entries`, render scores, packed-expert coverage,
activation maxima, and the MTP receipt. It deliberately refuses all CB keys because their pair
identity sidecars do not yet have a union rule. The publisher copies verified tensors into a
temporary bundle under content/key-addressed names, fsyncs, verifies complete coverage, and
atomically publishes `production_weight_cache.pkl`, `weights/`, and `union_manifest.json`
without overwriting an existing output. `verify` must be run again after transfer. This is an
operator-selected native cache-build path and is not wired as a `run-pipeline.sh` default
(`production_cache_stripes.py`, `union_production_cache.py`,
`tests/test_production_cache_stripes.py`, `tests/test_union_production_cache.py`).

**What that identity binds, and what it does not (issues #130, #135).** The campaign identity
binds coverage, source, calibration, producer code, settings, levers, render scope/retention,
formats and mechanism order (`union_production_cache.py:486-553`). It does **not** bind the
qname enumeration the activation collector hooked, and that enumeration is what the rendered
bytes are a function of. One `torch.Generator` feeds every hooked Linear's priority reservoir
(`production_weight_cache.py:921-922`), so the slice of the stream a Linear receives depends
on how many rows every earlier hook consumed; rows feed the GPTQ Hessian and the Hessian feeds
the bytes. `56c765d` draws for every *hooked* Linear, which makes `--resume` reproduce a fresh
build because resume narrows only the *store* set. Two narrowings reached the **hook** set and
were therefore out of its reach: a stripe, via `--include-qnames-file` shortening the `qnames`
list before the fill call (#130), and `--render-scope assignment` — the `run-pipeline.sh:590`
default — because `qname_set` was both the render set and the collector's hook set (#135).
Measured on a frozen 4-layer model with GPTQ on and the reservoir under selection pressure:
4/4 units diverged in rows *and* rendered bytes for a non-prefix stripe, for a prefix stripe
at `NSAMPLES >= 2` (so the cause was the shared stream, never `plan_stripes`'s binning), and
for an assignment-scoped build against a format-menu build of the same recipe. A single BF16
unit last in the enumeration moved all 7 remaining units' bytes.

**Closed by holding the hook set at the caller's enumeration.** `fill_production_weight_cache`
hooks `eligible_qnames` and narrowing arrives as `render_assignment` or the `render_qnames`
parameter; `--include-qnames-file` sets the latter instead of shortening `qnames`. Both
narrowed arms then reproduce the unstriped bytes exactly
(`experiments/stripe_row_identity_byte_baseline.py`,
`tests/test_striped_render_row_identity.py`, 9 passing). Shortening `qnames` itself remains
byte-moving by construction — it *is* the hook set — and is pinned as a caller error rather
than left to be re-learned on a 90 GB cache. Every cache now stamps
`metadata["activation_hook_scope"]` with the hooked enumeration's sha256, its size, the
rendered count and a `render_narrowed` flag, so equal hook digests plus an equal `calib_hash`
is a readable claim that two caches were rendered against the same rows. **The stamp is
now read, not just written (#147).** `union_production_cache` refuses shards whose hook
digests disagree — equal digests is the pass condition, never exact equality of the whole
scope dict, which would refuse every striped union for rendering disjoint slices; the merged
scope keeps the shared digest and sums the slices
(`tests/test_activation_hook_scope_gates.py`). The render-cost payload carries the hook
scope it was priced from in its provenance, and synthesis refuses a baseline priced from a
different rendering; the export fingerprint records the enumeration the shipped bytes saw,
so a shipped artifact names its rendering. Packed experts stamp the hook-only
`metadata["activation_hook_scope_packed"]` (`fill_packed_expert_cache_entries` hooks the
full visible module set and `store_qnames` narrows only storage, #145) and the union merge
requires agreeing packed digests while tolerating dense-only shards. **One caveat remains.**
The
cache directory has no render-identity guard at all — resume is file-presence only — so a
directory whose units were rendered under a narrower hook set will keep those bytes and
silently mix them with newly rendered ones. Rebuild an assignment-scoped cache directory
rather than resuming it across this change.

Render mechanisms are a registry with declared ordering semantics, not a lever string parsed in
spelling order (`render_score.py:188-260`): each `RenderMechanismSpec` declares `operation`,
`scope`, `phase`, `gate_metric` and optional `before`/`after`, and
`resolve_render_mechanism_order` resolves them topologically. Built-ins (`:322-380`):
`four_over_six` (40); then `joint_scale_opt` → `static_act_order` → `gptq` — both levers sit
at phase 50 with `before=("gptq",)` and no relation to each other, and
`resolve_render_mechanism_order` resolves that to `[joint_scale_opt, static_act_order, gptq]`
(matching `pipeline.py`'s own stage list); `fisher_gptq` (50, archived); `scale_sweep` (60,
after gptq). The production lever set is §3.3.

### 5.5 Named invariants

| Name | One line | Detail |
|---|---|---|
| **M6** | Allocator cost scores `weight_mse`, not `h_trace × output_mse` | §4.2 |
| **M19** | Export re-derives NVFP4 codes under the render's *recorded* scale rule, not the export-entry `static_6`. Default ON | §6.1 |
| **M26** | Frontier KL is scored `full_sequence`, not last-token | §4.6, §7.1 |

### 5.6 `input_global_scale` is a free post-export knob

The NVFP4 activation global scale can be patched in place after export and re-measured — no
re-render. `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` selects the compressed-tensors
`generate_gparam` convention `FP8_MAX·FP4_MAX/amax` over the legacy `FP4_MAX/amax`
(`export_native_compressed.py:874-910`); it rescues blocks far below calibration amax from FP8
subnormals at the cost of clipping any serve block above it. Served A/Bs 2026-07-02, weights
byte-identical: 35B-A3B MoE frontier −14.1% KL (win), LFM2.5 +5.8% (loss), 27B regen dense
+37.5% (loss). Strongly artifact-dependent, so the default stays legacy (`0`) and any change
requires a per-artifact served A/B.

### 5.7 Tessera — a synthesized family, priced and rendered by name, admitted only by attestation (2026-09-02)

**What it is.** Tessera is the trellis-coded weight format developed in its own
repository (`/home/rob/tessera`, package `tessera`: encoder, wire, decoder,
kernels; design record `docs/design/embedded_native_weight_coding_2026-08-31.md`).
Its grammar has five axes — base grid × body × body rate × scale plane ×
serving route — and PrismaQuant's seam to it is two modules and one registry
fallback:

- `tessera_formats.py` — grammar and pricing. `parse_tessera_format_name`
  (`:1160`) splits `TESSERA_E2M1_K2_R896` into a family and a rung; a family is
  a (base grid, arity) pair (`tessera_family :779`); the rate cap is a property
  of the body the family's wire recipe names (`family_rate_cap :818`, never a
  subtraction); the rate axis is continuous at a 1/256-bpp quantum and
  `validate_body_rate_q256` (`:850`) / `realisable_rungs` (`:866`) say which
  rungs encode; `tessera_wire_recipe` (`:197`) is the one source of body and
  scale plane per grid and rung; `artifact_bpp` (`:883`) and
  `wire_overhead_q256` (`:376`) are the byte accountant, and **both require a
  shape** (below); `tessera_serving_route` (`:1049`) is the fifth axis.
- `tessera_render.py` — the render adapter. `render_tessera_weight` (`:408`)
  encodes and decodes through the `tessera` package and holds no numeric
  constant of its own (`TESSERA_CONV_MEMORY`, `TESSERA_GROUP`, `TESSERA_HALF`
  are read from `tessera.export`, because a second copy of a rate constant is a
  drift bug waiting for a rate to change). `synthesize_tessera_spec` (`:538`)
  builds a `FormatSpec` on demand -- always a `bits_for_shape_fn`, never a
  scalar rate, because no Tessera rung has one; `tessera_rung_is_serialisable`
  (`:328`) asks
  the wire whether a grid is committed (a rung that renders but is not
  committed would otherwise die in `alphabet_plane()` at export, after the whole
  production cache is built); `tessera_lane_attested` (`:157`) is the admission
  gate.
- `format_registry.get_format` (`:1322-1338`) falls through to
  `synthesize_tessera_spec` for anything Tessera-shaped and to the ordinary
  `KeyError` for anything else. **Rungs are family parameters, not `REGISTRY`
  rows**: one family addresses ~9500 rungs, and freezing them into the registry
  would turn a continuous rate axis into a menu someone maintains. Every
  consumer that resolves a format by name — candidates, the production-cache
  render, the KL validator — works unchanged, and nothing else in the pipeline
  imports the seam.

**Exact pricing, or refusal.** `FormatSpec.bits_for_shape_fn` (added on this
branch) is a format's own accountant. A Tessera spec has no scalar bpp:
`weight_bits` is a label quoted at `_LABEL_SHAPE = (2048, 4096)` so the
registry's integer field has a stated meaning, and `bits_for_shape(shape)` /
`effective_bits_for_shape(shape)` are the price. Asking a shape-dependent
recipe for a scalar raises rather than flooring (`def11bd`), and
`effective_bits_for_shape` — what the DP, `footprint.py` and the Pareto table
consume (§5.1) — honours the accountant, so the surrogate, the byte gate and
the wire agree about one artifact (the earlier `FormatSpec` special case
overcharged Tessera by 6.25% on both).

**The route is priced.** `TesseraServingRoute` (`:831`) is a joint property of
the base grid and the scale plane: E2M1 over a per-16 block plane (S6b or
LUT16) decodes to the stock NVFP4 tile and executes W4A4
(`w4a4-nvfp4-e2m1-group16-ue4m3`, `tessera.decode.materialize_nvfp4`); E4M3
over a CHANNEL plane at arity 1 decodes to the stock `strategy: channel` FP8
pair and executes W8A8 (`w8a8-dynamic-e4m3-channel`); everything else — E4M3
over a block plane, E2M1 over CHANNEL, any free grid — is the kernel route,
weight-only decode inside the GEMV, and no runtime serves it. Pricing the route
is what stops the allocator comparing a W4A4 rung against a W8A8 one as if the
activation side were free — the NVFP4_CB lesson of 2026-08-17 (§4 P8).

**Admission is a lookup, not a constant — and Tessera's own table is what it
reads.** Principle 9 makes "a runtime executes this rung natively" a measured
platform fact and principle 14 says the fact is read from the runtime's own
table. Since 2026-09-02 that runtime is **Tessera's own vLLM plugin** (§9.4):
package `tessera.serving`, entry point `tessera = "tessera.serving:register"`
under `vllm.general_plugins`, registering `quant_method = "tessera"`, selected
by the checkpoint with no enable flag and one operator knob
`TESSERA_SERVE_MODE=resident|streamed`. `tessera_lane_attested(name)` resolves
the name to its payload family and rate through
`lane_eligibility.resolve_payload_rung` against the contract that
plugin PACKAGES — `tessera/serving/runtime_contract.json`, reached by
`importlib.resources.files("tessera.serving")`, never repo-root arithmetic —
and answers True only on the AND of three conjuncts, each of which fails closed
alone:

1. **The table admits the rung.** The contract publishes the family and carries
   a `device_qualified` cell on a `backed` or `backed_with_serve_flag` route
   whose `rungs_q256` names that rate. No table, an unpublished family, a rate
   no cell names, a `compile_only` cell or a `fallback` route all answer False.
2. **Every matching cell names its plugin.** `requires_plugin: "tessera"` is a
   machine-readable cell field, not prose, because stock vLLM has no reader for
   these bytes: the route is plugin-gated, not merely flag-gated, and an export
   gate must be able to refuse an artifact whose serve command would not install
   the runtime. A cell claiming a route with no plugin requirement is a
   CONTRACT DEFECT and this function **raises** rather than admitting it. The
   requirement travels into provenance through `resolve_unit_route`, which
   aggregates it onto every `UnitRoute` / `RegimeRoute` as `requires_plugins`
   and into `EligibilityTable.provenance()` as `required_plugins`.
3. **The pinned runtime is an exact reviewed release.**
   `require_exact_tessera_runtime_release(load_tessera_serving_runtime_pin())`.
   Without this conjunct the answer would flip to True the moment the `tessera`
   package became importable — which it already is, as the render adapter's own
   dependency — and a producer-side import is not a serving release.

**It is False today by the PIN, not by an edit and not by an absent table.**
Tessera's packaged contract publishes its attested families with `device_qualified`
native cells and the package imports fine here; what withholds them is
`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`, whose `commit`
and `version` are the conspicuous sentinels `PENDING_TESSERA_RELEASE_COMMIT` /
`PENDING_TESSERA_RELEASE_VERSION` with `version_is_release: false`. There is no
Tessera release tag and cutting one is Rob's decision; when it happens, the pin
file and the reader's two release constants
(`TESSERA_SERVING_RUNTIME_RELEASE_{VERSION,COMMIT}`) resolve in ONE reviewed
commit, because the reader requires them to be equal and neither half admits
anything alone. `tests/test_tessera_lane_admission.py` proves both directions:
the tracked pin is refused with "PENDING", and under a released-pin fixture on
the REAL packaged contract `TESSERA_E2M1_K2_R896` and `TESSERA_E4M3_K1_R1024`
are admitted while `TESSERA_E2M1_K2_R512` is not.

Until 2026-09-02 this was a module constant (`False`), then briefly a lookup
against GRIDBOOK's serving pin, whose unreleased contract v13/v14 carried the
Tessera rows from the served receipts recorded in §9.2. **Gridbook's Tessera
lane is withdrawn and the Gridbook pin no longer governs Tessera admission**;
read §9.2's Tessera passage as history. The per-artifact question (this
platform, this unit's regimes) still stays with `resolve_unit_route` at export.

**One cell parser, with explicit version semantics.** `lane_eligibility.py`
reads Tessera's v4 table for export and for `tessera_runtime_contract.py`'s
development pin. A v4 cell requires `requires_plugin: "tessera"`, non-empty
`executes` pairs and exactly one residency selector bounded by its family's
`residency_modes`. Two cells may not claim the same platform/family/structure/
regime/residency scope, even at different rungs. `resolve_unit_route` takes an
explicit `residency`; omission is unattested, and a resolved `RegimeRoute`
carries both the residency and launches. The reviewed development answer
includes those values, so changing a decoder cannot retain the old answer.
Explicit legacy Tessera v3 tables retain their original grammar and carry no
fabricated launches. Gridbook schemas remain refused. Release admission is
still gated by the exact PENDING release pin.
The development reader also refuses non-empty cell predicates: its family/rate
menu lookup has no unit facts with which to evaluate a shape constraint.
Export's `resolve_unit_route` has those facts and evaluates the predicates.

**The boundary has an export arm, not a second codec.**
`prismaquant/run-pipeline.sh` selects `EXPORT_CONTAINER=tessera`, preflights
the lane and calls Tessera's own plan and encoding tools under `TESSERA_REPO`.
`export_native_compressed.py` deliberately has no Tessera codec. The arm opens
a lane-gated shipcard; `prismaquant/lane_specs/tessera.json` (§9.4) declares the
serve, endpoint, gates, KL evaluator and executed activation contracts.
Allocation uses `tessera_menu` and its reviewed development contract; that
research menu does not bypass the exact release pin required for export.
The pending release and supported producer-tool boundary remain D33, rather
than an absence of pipeline integration. Tests include
`test_tessera_formats.py`, `test_tessera_footprint.py`,
`test_tessera_shape_dependent_recipe.py`, `test_tessera_lane_admission.py`,
`test_tessera_menu.py` and `test_tessera_export_lane.py`.

## 6. Export & serving invariants

`prismaquant/export_native_compressed.py` (9,130 lines) turns a `layer_config.json` recipe plus
(normally) a `ProductionWeightCache` into a `compressed-tensors` checkpoint. §5 owns the
render; this section owns the bytes and the metadata that make vLLM accept them. Bare `:N`
refs are that file.

### 6.1 Codec map

`_quantize_2d` `:4740-5225`, dispatching on `_canonical_export_format` `:676-680`.

| Format | codec | emitted tensors |
|---|---|---|
| NVFP4 | `quantize_dequantize_nvfp4` `:3147`, packer `pack_fp4_indices` `:851` | `weight_packed`, `weight_scale` (fp8 e4m3, g16), `weight_global_scale`, `input_global_scale` (`:4971`) |
| MXFP4 | `quantize_dequantize_mxfp4` `:3476` | packed fp4 + uint8 E8M0 scales (g32) |
| MXFP8_E4M3 / _E5M2 | `quantize_dequantize_mxfp8` `:3549` | fp8 weight + uint8 E8M0 scales (g32) |
| FP8_E4M3 / _E5M2 | `quantize_dequantize_fp8_dynamic` `:3696` | fp8 weight + per-row fp32 `weight_scale` |
| BF16 | `_passthrough_tensor` `:5727` | verbatim |
| FP8_SOURCE | verbatim copy (§6.3) | source `weight` + `weight_scale_inv` |
| 3-D packed experts | `_quantize_3d_packed` `:5228` + `_split_packed_expert_tensor` `:4540` | per-expert per-projection tensors |

Activation-aware passes compose inside `_quantize_2d` (`:4818-4832`): `gptq`, `scale_sweep`,
`static_act_order`, `joint_scale_opt`, the latter two forced to require `gptq`
(`:4830-4831`). `input_global_scale` follows the compressed-tensors `generate_gparam`
convention `FP8_MAX·FP4_MAX/max_abs` (`_nvfp4_input_global_scale_from_max_abs :895-910`).

**Export refuses what it cannot emit** (#27, `29f3cff`). `EXPORTABLE_FORMATS` `:7517` is
*derived* from `FORMAT_SCHEME` plus the container passthrough, never hand-listed, and the vLLM
lane spec reads its menu from that constant. A format with no emit path used to be rewritten
to BF16 behind a `print` — a Linear allocated at ~4.25 bpp shipped at 16, blowing the byte
budget it was selected under and leaving the artifact's real bpp disagreeing with its own
`layer_config.json`. It is now a hard error naming the Linear, the format and the resolved
profile (`:1548`, `:1574-1589`), with the wrong-container cases (`nvfp4_cb`, GGUF) called out
by name. The *legitimate* coercion is deliberately kept: a format the exporter can emit but
which is shape-illegal or profile-denied still falls back to BF16 and is still audited.

**Research-cost ship refusal.** Every exporter reads the reserved layer-config metadata and
refuses a selection carrying the sanctioned study-grade cost stamp unless the operator passes
the separate `--allow-research-cost-selection` acknowledgement. This is independent of the
allocator-side acceptance: allocating a learning experiment does not silently authorize
shipping it. CB artifacts record the manifest and acknowledgement in `quant_config.json`;
GGUF records it in metadata. Unstamped production selections follow the existing gates
unchanged (`research_cost_acceptance.enforce_research_export_acknowledgement`).

**M19 — export honours the render's scale rule.** `_export_match_render_scale_rule`
`:2130-2147` reads the cache's `levers["nvfp4_scale_rule"]` and re-derives NVFP4 codes under
*that* rule rather than the entry default `static_6`, making the re-quant of the cache's bf16
dequant near-idempotent; `PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE` **defaults ON**
(`:2143`). Packed companion `_packed_expert_render_scale_rule` `:2150-2177` — without it,
joint_mse-rendered experts re-derived under `static_6` flipped 43% of packed bytes. Residual
gap: joint scale *levels* are not in the lever dict, so a non-default
`PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` must match between cache-build and export.
`_pack_production_cached_2d` `:2180-2279` re-packs only — no re-run of GPTQ/scale-sweep, which
would measure a different artifact.

**Block-output match** (`prismaquant/block_output_match.py`), `PRISMAQUANT_BLOCK_OUTPUT_MATCH`
**default `"1"` = ON** (`:6168-6169`, `:6321`, `:6467`): for NVFP4 dense block Linears
(q/k/v/o, gate/up/down) it defers the pack, greedily refines per-Linear group scales against an
FP16 block-reference forward, then finalises (`:6555+`). Its own ~0.05–0.10 PPL estimate
predates JSO and has never been re-measured on the gold lane (§12).

### 6.2 `config_groups` / `ignore` / packed-MoE emission

`build_quantization_config` `:7589-8005` emits explicit per-name targets grouped by format,
remapped to vLLM-internal names via `profile.to_vllm_internal_name`; the catch-all default
group is the format with the most non-BF16 members (`:7951`). Schemes are hand-authored
constants — `NVFP4_SCHEME` `:7247`, `MXFP8_SCHEME` `:7264`, `MXFP4_SCHEME` `:7282`,
`FP8_SOURCE_SCHEME` `:7305`, `FP8_E4M3_SCHEME` `:7325`.

BF16 plus `bf16_passthrough` plus `extra_ignore` go to `ignore` (`:7614-7636`); BF16 **packed**
experts additionally need a per-layer regex over every (expert, projection) because vLLM
scheme-dispatches on per-expert Linear qnames (`_bf16_packed_expert_ignore_regex` `:7386-7481`,
used `:7632`). Fused siblings present in the serving model but absent from the probe (Gemma4
`k_eq_v` with no `v_proj`) are back-filled into `ignore` from `packed_modules_mapping` / the
structure spec (`:7643-7700`). `compute_extra_ignore` `:8055-8095` must *not* add per-expert
source keys when the packed parent is quantized — that marks the FusedMoE un-quantized, the
NVFP4 scale params never register, and weight-load KeyErrors follow (`:8089-8091`).
`_preflight_quantization_config` `:8008-8025` builds the entire config before any GPU render or
shard write (called `:8477`), so metadata violations fail in seconds rather than hours.

**Packed-MoE 3-D.** vLLM's `get_moe_method` probes three *synthetic* names
`<block>.experts.0.{gate_proj,up_proj,down_proj}`, not the on-disk packed qnames
(`experts.gate_up_proj`). Each packed recipe entry is therefore replaced by **one per-layer
regex pinned to that layer index** (`_constrain_per_expert_projection_regex` `:7350-7383`,
`_pin_regex_to_layer` `:7339-7347`, emission `:7779-7800`), with on-disk leaves translated
through `_vllm_moe_scheme_projection_names` `:4503-4525` so LFM2.5's `w1/w3/w2` are advertised
as `gate_proj/up_proj/down_proj` (`:7980-7994`).

**Namespaces on the codebook lane (`export_nvfp4_cb_streaming`).** A wrapped
source puts three spellings of one module in play: the CHECKPOINT
(`model.language_model.layers.N.*`, the bytes on disk), the LIVE module tree
(`language_model.model.layers.N.*`, what the skeleton and the group planner
see), and the RECIPE (`model.layers.N.*`, what the allocator's `layer_config`
and `_resolve_target` speak). Which one an emission uses is a property of the
*consumer*, not a style choice:

- **CB group targets** keep the CHECKPOINT spelling (`_cb_target_name`
  `:5026`, `_base_name` `:3172`) — the names their tensors were written under.
- **Delegated (stock-CT) and source-passthrough group targets** are emitted in
  the CANONICAL namespace (`_delegated_target_name` `:5034`): the profile's
  vLLM-internal rename with the wrapper canonicalized
  (`language_model.model.` → `model.`, `language_model.<rest>` →
  `model.<rest>`). This mirrors — and is cited from, not asserted about — the
  pinned consumer's `gridbook/config.py::_canonical_prefix` /
  `_candidate_bases`, which try a serving prefix as given *and* in canonical
  form. A canonical target therefore resolves from every namespace vintage;
  a full live-tree target resolves only from its own, which on a wrapped
  Qwen3.5 source left the delegated Linears unquantized until gridbook refused
  them fail-closed at weight load. Scoped to this container:
  `export_native_compressed` keeps live-tree targets, which vanilla
  compressed-tensors matches.
- **Expert-group keys** are normalized to the RECIPE spelling when the planner
  matched a live name (`:2960`), derived from the group's own member tensors;
  recipe- and checkpoint-keyed groups (DSv4-class) are left exactly as
  planned, and a collision on either the normalized key or the
  recipe→checkpoint bridge refuses. Packed expert stacks are then *named* by
  their group's checkpoint prefix, because the packed parent is not a
  checkpoint leaf and gridbook's top-level loader rename
  (`moe_toplevel_loader::_hf_mapper_rename`) reuses the model's own
  `hf_to_vllm_mapper`, i.e. checkpoint prefixes only.
- Consequence for the contract below: on a wrapped source, per-expert (Tier-2)
  config files address groups in the RECIPE namespace. Unchanged for
  DSv4-class sources.

**PROPOSED per-expert CB producer contract (v1; consumer reconciliation
pending).** `export_nvfp4_cb_streaming --per-expert-config <json>` consumes the
Tier-2 flat `qname -> format` allocation and emits one expert sub-stack per
`(layer, family, format)`, with `family ∈ {w13,w2}`. Within a sub-stack expert
ids are ascending. A mixed family's physical prefix is
`<legacy-prefix>.format_group_<lowercase-wire-id>`; a one-group family retains
the legacy prefix, so a single-format layer is byte-identical to an artifact
exported without the flag. `quant_config.json` then carries:

```
per_expert_format_groups = {
  "version": 1,
  "layers": {"<layer>": {
    "w13": [{"format_wire_id": str, "expert_ids": [int, ...], "tensor_prefix": str}],
    "w2":  [{"format_wire_id": str, "expert_ids": [int, ...], "tensor_prefix": str}]
  }}
}
```

The key is omitted for a single-format layer (absence keeps the legacy
uniform contract). Every family must partition the architecture's expert ids
exactly once; `artifact_completeness.py` checks gaps/duplicates, referenced and
unreferenced physical tensors, and the persisted subgroup byte sums. CB
payload accounting sums physical sub-stacks and charges each sub-stack's
codebook sidecar once. MXFP4_SOURCE groups keep the checkpoint's verbatim
per-expert element/scale slices; their entry in this declaration is the sole
routing authority, so the same expert module is deliberately absent from
`source_passthrough.units`. This path is opt-in and **PROPOSED**, not production
eligible until the independently pinned Gridbook consumer reconciles v1 and
passes load/generation plus served speed/quality gates. Implementation:
`export_nvfp4_cb_streaming.py`, `cb_export_config.py`, `footprint.py`, and
`artifact_completeness.py`; CPU contract coverage:
`tests/test_per_expert_cb_export.py`.

**The `quantized_embedding` wire contract (v1, gridbook lane only, 2026-08-14).**
A token embedding is a *lookup*, not a GEMM, so it cannot ride `config_groups`:
vLLM's compressed-tensors embedding path accepts weight-only INT (WNA16) and
**raises** for FP8/NVFP4, meaning a stock `config_groups` entry naming the
embedding does not mis-route the artifact, it refuses to load it. The CB lane
therefore declares it out-of-band:

```
quantized_embedding = {"version": 1, "units": {"<module>": "<wire id>"}}
```

Producer `build_quantized_embedding_declaration` (`cb_export_config.py:450-520`),
attached `:1198-1200`; classification and emission in `export_nvfp4_cb.py`.
Three properties are load-bearing:

- **Weight-only.** Three tensors (`weight_packed`, `weight_scale`,
  `weight_global_scale`) and no `input_global_scale` — a lookup has no input
  activation, the serving method registers no such parameter, and emitting one
  is an unmatched checkpoint key at load.
- **Out of both `config_groups` and `ignore`.** Before this, the exporter copied
  the embedding verbatim *and* appended it to `ignore`, so the 1.70 GiB was
  never actually recoverable.
- **Never `lm_head`.** `ParallelLMHead` subclasses `VocabParallelEmbedding`, so
  a producer naming the head here would take the output projection off the GEMM
  path. Refused at write (`:484-491`), at parse, and at `create_weights`.

Classification is deliberately **two independent conditions** — the name
(`.embed_tokens`) and the shape (rows == declared vocab, and not the head) —
which must agree or the export fails rather than guessing. The declared vocab is
read through a `text_config` walk because multimodal checkpoints keep the LM's
vocab there: Qwen3.8-27B has **no top-level `vocab_size`**, so a naive read would
see zero and silently disable the shape half of the cross-check.

Measured on Qwen3.8-27B (exact full-vocab KL, isolated marginal on a BF16 body;
model total gold KL 0.0338): NVFP4 group-16 costs **0.666 GiB vs BF16's 2.368**
at KL 0.001063 weight-only / **0.000948 under a 4-bit activation floor** — i.e.
it gets *better* in the regime artifact A actually serves. FP8 per-row and INT8
W8A16 are both 1.185 GiB; INT4 W4A16 g128 is 0.611 GiB at 0.001963.

### 6.3 FP8_SOURCE verbatim, MTP, audits

`_build_fp8_source_map` `:5747-5854`: a tensor qualifies when `<base>.weight` has a sibling
`<base>.weight_scale_inv` in the index (the 128×128 block convention of MiniMax-M2 /
DeepSeek-V3 / NVIDIA FP8 releases). Bytes are copied unchanged; only the suffix is renamed
(`weight_scale_inv` ≡ compressed-tensors `weight_scale`). A non-FP8 source returns `{}`, which
makes FP8_SOURCE inert — the allocator's passthrough-integrity filter then drops it everywhere
(`:5766-5769`). Overlay `_fp8_source_config_overlay` `:5885-5929`.

**Passthrough integrity now uses the allocator's own vocabulary** (#29, `b6ec9cb`). The
coercion never passed `source_kind`, so `check_format_applicability` judged **every**
FP8_SOURCE Linear illegal and rewrote it to BF16. That was inert in the bytes — materialization
copies the source fp8 verbatim and the config overlay restores the scheme — but it filled every
DSv4 / Hy3 / MiniMax `runtime_coercions` with demotions that never happened, hiding any real
one, and it forced a passthrough exemption in the group-escalation path. `source_kind` now
comes from `_scan_source_dtype_manifest`, the same recipe-keyed map that gates the allocator's
passthrough candidates, scanned lazily so a BF16-source export does no extra header IO
(`:1530-1545`). Bogus rows went 4/4 → 0 on a synthetic fp8 checkpoint; the exemption is deleted,
so a genuine passthrough mismatch inside a serving unit escalates like any other illegality.
Per-expert 2-D checkpoints whose live Transformers module is packed additionally fold every
leaf's header-derived kind onto the profile-declared packed recipe parent. Qwen3-30B-A3B's
18,432 BF16 expert leaves therefore populate all 96 live w13/w2 source kinds instead of losing
BF16 fallback because only the indexed checkpoint names were present
(`allocator_candidates.py:_per_expert_packed_recipe_name`).

transformers v5 does not instantiate MTP for Qwen3.5/3.6 MoE, so `_materialize_mtp_tensors`
`:8939-9002` rebuilds a standalone MTP module under a parent named `mtp` and materialises it in
memory, keeping checkpoint-convention names. **Since 2026-07-30 (R12) it goes through the
profile**: `profile.build_mtp_module(text_config)` builds it, `profile.read_mtp_source_state_dict()`
pulls the source tensors keyed on `profile.mtp_source_prefix()` (default `"mtp."`), and
`profile.load_mtp_state_dict()` loads them, folding per-expert checkpoint keys into the packed
3D expert Parameters. `build_mtp_module`'s contract is that the returned module's names, once
wrapped in that `mtp` parent, equal the allocator's recipe names (`mtp.fc.*`, `mtp.layers.0.*`) —
which is why the recipe filter here stays `mtp.` regardless of the source prefix.
`validate_mtp_assignment_coverage` `:9195-9222` **hard-fails** when the source has tensors under
`mtp_source_prefix()`, the profile `has_mtp()`, and the recipe has no `mtp.*` entries.

That export-time synthesis is now mirrored by a production-cache producer, not postponed until
packing. `build_production_cache` invokes `fill_profile_mtp_production_cache` after the body
fill in both resident and streaming-assignment modes. A non-BF16 MTP recipe must therefore be
present in the attached `ProductionWeightCache`; the exporter cache-fingerprint gate treats
missing `mtp.*` pairs like any other missing assignment pair and points the operator back to
the same layer config plus the probe `--activation-cache-dir`. Source/module/activation
coverage and `--include-qnames-file` stripe ownership are §5.4. BF16 `MTP_FORMAT` remains the
default and preserves legacy body-only cache behavior when no activation directory is supplied
(`mtp_production_cache.py`, `build_production_cache.py`,
`export_native_compressed.py:_production_cache_expected_keys`).

**DeepSeek-V4 DSpark has two explicit producer modes.** The release-default mode is a
source-format metadata overlay, not a second quantization pass. The released Flash checkpoint
already carries three complete
`mtp.{0,1,2}` stages: 2,304 routed-expert projection bases as packed MXFP4
E2M1 + E8M0 and 25 dense/shared/attention bases as block-FP8 E4M3 + E8M0.
`dspark_source_metadata.py` derives the released layout from model config and
validates all 4,705 MTP tensors from safetensors headers before it emits
anything: exact dtype and shape for all 2,329 weight/scale pairs (including
group-32 / block-128 scale grids), six 2-D BF16 router/confidence/Markov
matrices, fourteen BF16 norms, and twenty-seven F32 sink/router/hyper-connection/
head tensors. Missing or unfamiliar glue is a refusal, as are duplicate or
out-of-range `dspark_target_layer_ids`. The streaming exporter's ordinary copy
loop keeps those bytes unchanged; the overlay removes exactly the quantized
bases from `ignore`, extends the source-layout config groups under their
physical `mtp.*` names, and leaves the six unscaled 2-D Linears in `ignore`.
Norms and F32 parameters are loader glue rather than quantization targets, but
their exact presence, dtype, and shape are still part of the closed layout.

Routing uses vLLM's *construction* namespace, which is intentionally different
from both the physical checkpoint and registered module names. For a body with
`L` decoder layers the declaration names seven units at each of
`model.layers.{L,L+1,L+2}` plus `model.main_proj` (22 total); fused `wq_a/wkv`
and shared `w1/w3` pairs map to their constructed fused modules, and each routed
expert stage maps once to its whole `ffn.experts` unit. Only an artifact carrying
this fully validated declaration gets `config.json:n_mtp_layers = 3`. Existing
artifacts can receive the identical contract with
`python -m prismaquant.dspark_source_metadata ARTIFACT --output-artifact
ARTIFACT-dspark`: it hardlinks the immutable model/container files into a
hidden sibling staging tree, writes only new `config.json` and
`quant_config.json` sidecars, recomputes the self-sized artifact inventory,
validates completeness with no `mtp.` exemption, and publishes the complete
new directory with one `renameat2(RENAME_NOREPLACE)`. A launcher therefore sees
either no output path or both new sidecars, never a half-applied pair; the
source artifact remains unchanged. Provenance schema
`prismaquant.dspark_source_overlay.v1` records `tensor_bytes_rewritten: 0`;
tests additionally pin the model container's SHA-256, inode, size, and mtime.
All non-MTP config groups, ignores, and pre-existing delegated routes must remain
identical, and a route-pending source format still requires the artifact's prior
ship acknowledgement. With `--dspark-cb-sidecar` omitted or explicitly false,
the target artifact retains the source checkpoint's exact physical tensor
inventory, dtype/shape map, and payload bytes and carries no sidecar/render
attestation provenance.

The experimental `--dspark-cb-sidecar` mode is the deliberate exception. Its expanded recipe
contains all 2,328 decoder Linears, but exactly 2,325 are K12 CB. The three
`mtp.{0,1,2}.attn.wo_a` bases remain released block-128 source-FP8 W8A16 because vLLM marks
that projection as a grouped BMM and Gridbook's generic CB Linear implements only ordinary
dense GEMM; flattening `[T,G,K]` would mix groups and produce the wrong output shape.
`mtp.0.main_proj` is the fourth explicit W8A16 source base. The 2,325 logical CB decisions
collapse to exactly 27 physical CB qweights under `mtp.{0,1,2}.*`, declared at Gridbook's
construction names `model.layers.{L,L+1,L+2}.*`. Routed experts are validated as a closed
3-stage × 256-expert × {gate,up,down} set and collapse into six packed expert stacks.

The weight-only safetensors header is therefore exactly 82 tensors: 27 CB qweights, the
unchanged 47 non-Linear glue tensors, and eight W8A16 source planes (`weight` + UE8M0 `scale`
for `main_proj` and all three `wo_a`). The CB parameter count is 19,623,051,264. Relative to
the unsafe all-CB K12 draft, retaining `wo_a` adds 78,256,128 bytes (about 74.63 MiB) before
small JSON/header variance. The input builder refuses anything except the released
three-stage layout and records all four physical→construction source routes. The streaming
exporter refuses a subset other than `mtp.`, any CB `wo_a`, unexpected decoder routes,
incomplete stages, or a non-source member of that four-base set; it records exact physical CB,
construction CB, and source physical→construction sets in
`prismaquant.dspark_cb_sidecar.v1` provenance. When an activation execution contract is
present, the artifact additionally carries a validated one-to-one `dspark_target_bridge`;
weight-only experiments intentionally omit it. All four source bases retain both physical
planes through the raw-copy path; recognizing an MTP `.scale` sibling for source decoding does
not consume or suppress that verbatim scale plane.

The sidecar decoder gate does not trust a config-group label as its own layout proof. Every
CB label must be a canonical producer rung from `cb_layout`; its scheme must exactly match that
family's grid, mode, K, subtable count, superblock, group/vector dimensions, packed type size,
activation width, and production scale-coding record. The group format is then expanded back
to the finalized physical `provenance.tensor_formats` per-Linear assignment. In particular,
each packed routed-expert `gate_up_proj` group must match every expert's separate `gate_proj`
and `up_proj` decision, and `down_proj` must match every expert's separate decision. Thus a
self-consistent label/scheme/qweight/codebook-header rewrite cannot redefine the allocator's
assignment. The `.pqcb` header must contain exactly the canonical subtables and shapes, and
the gate loads each real safetensors tensor and recomputes the producer definition
`sha256(tensor.to(float16).cpu().numpy().tobytes())` against every declared
`provenance.codebook_sha256`; changing payload values without changing dtype or shape refuses.

**A production sidecar must be rendered again from a source-complete recipe.** The earlier
research draft's `render_identity_verified=false` marker cannot be upgraded by editing JSON,
and `--allow-unstamped-research` is never accepted by a ship gate. The input builder requires
the streamed-model identity cache and stamps
`prismaquant.dspark_cb_render_recipe.v1`: compact complete checkpoint identity plus exact
digests of source `config.json`, the validated MTP safetensors header, canonical assignment,
imatrix/column weights, and the pristine incomplete render-identity seed. During the one-pass
export, `CBRenderSourceIdentityCollector` observes the decoded shape and content digest of
every CB source weight exactly once. Dense weights are observed from the decoded host tensor
before H2D, so attestation adds no immediate full-tensor D2H copy; routed-expert members are
likewise observed as each host member is decoded. Any duplicate observation (including a
changed value under the same qname), post-finalize mutation, or incomplete finalize refuses.
Final output publication happens only after that observed set equals the exact assignment
scope; the artifact then records a source-complete `cb_render_identity`,
`render_identity_verified=true`, and the full recipe plus its digest in
`dspark_render_attestation`. `validate_dspark_production_render_attestation` replays the
recipe, assignment and seed digests and requires exact map equality between the finalized
`provenance.tensor_formats` CB qname→format scope and both the render-identity and complete
source-weight scopes. A self-consistent reduced render with recomputed column-weight, source,
seed, recipe, and attestation digests therefore still refuses; W8A16 source/glue routes are
correctly outside that CB map. It also rejects a missing source digest, immutable recipe
stamp, or source-weight closure. Consequently the existing unstamped K12 draft is research
evidence only; a fresh export from the stamped inputs is a release prerequisite.

This producer does not promote the lane. The Gridbook producer record (0.8.5/v3 when this was
written, 0.8.11/v4 since 2026-08-21) remains the
producer and source-overlay handoff evidence; the 0.8.5 record predates the
physical→construction→registered
DSpark bridge. A candidate consumer implements that bridge under
`gridbook.runtime-contract.v4` (`contract_version=4`,
`dspark_construction_physical_bridge=1`, `source_fp8_block128_w8a16=1`). The draft decode
receipt requires both features: the namespace bridge and the delegated grouped-BMM W8A16 route
are each load-bearing. The consumer must be released as Gridbook 0.8.6 and selected through
the separate current-serving pin by its exact release commit and reviewed wheel SHA-256.
Until both identities are resolved, serving fails closed. The separate draft artifact must
then pass load, generation, measured residency/scratch, acceptance-by-position, and paired
throughput evidence. It is served from its own path and never mutates the immutable target
artifact.

The MTP claim has its own paired endpoint/shipcard path rather than borrowing a target-only
smoke result. `serve_fingerprint.py write` takes both `--artifact-dir /model` and
`--draft-artifact-dir /draft`; `pre` and `post` snapshots must bracket one unchanged process
tree/session and bind both exact artifact SHAs. `attestation_phase` is chronology metadata and
is excluded from the stable serve fingerprint; every resident-extension, package,
runtime-capability, artifact, and session field remains bound, and the attester plus compact
receipt replay require exact pre/post fingerprint equality. Between the snapshots, the MTP
arm of `python -m prismaquant.dspark_matched_performance_collector collect-arm` runs eight
fixed prompts for exactly 128 tokens each (`temperature=0`, `ignore_eos=true`) and emits the
acceptance file from that exact measured interval; the older
`validate_dspark_target_draft run-suite` entry point remains diagnostic and is not a release
report producer. The collector reconciles uncontaminated Prometheus before/after counters.
Five-token DSpark drafting must be active, per-position accept counts must be
monotone, and position zero must accept at least 0.60. `attest` additionally requires the
exact 256K/FP8-KV/Marlin/Gridbook launch contract and server-log proof that FULL_DECODE_ONLY
graphs for sizes `[5,6]` were captured without downgrade. The manifests are replayed through
the separate `dspark_serving_profile.v1` receipt and exact Gridbook-0.8.6 runtime evidence,
not the producer environment (0.8.5 at the time of that measurement). The same MTP log must
provide an exact 92-route
`dspark_route_census.v1`: 70 target FP4-v2 routes, six uniform-K12 draft FP4-v2 routes, 16
target source-FP8 inherited routes, and zero fallback/unscoped routes
(`prismaquant/dspark_serving_profile.py:validate_dspark_serve_manifest`,
`validate_route_census`).

An MTP claim also requires a separately served target-only arm and the strict matched release
gate in `prismaquant/dspark_matched_performance.py`. The only release-authoritative report
producer is `prismaquant/dspark_matched_performance_collector.py`; it never starts or patches
vLLM and accepts no operator-authored manifest, response, counter, memory, KV, graph, or route
JSON. From one verified `prismaquant.runtime_source_snapshot.v1`, `declare-policy` first writes
one no-clobber policy declaration shared by both arms. Inside each future serving container,
`start-sampler` must run before vLLM and starts a fresh stdlib-only one-second
MemAvailable/cgroup-OOM observer (so the parent package's ML import footprint is not retained);
after READY, `collect-arm` adopts that exact one-shot process, writes the live pre manifest,
issues one warm and one measured workload, writes the live post manifest, snapshots the real
serve log, and only then stops the sampler. It parses graph, KV, OOM, and (for MTP) route and
speculative-acceptance evidence from those observations before invoking the report validator
and publishing the no-clobber report.

The `python -m` spelling names the module CLI, but a release invocation runs it through
`tools/prismaquant_source_bootstrap.py run-module --source-root <snapshot>` with module
`prismaquant.dspark_matched_performance_collector`. That bootstrap selects the exact read-only
snapshot without executing the package's heavyweight compatibility initialization; the
collector then independently verifies the snapshot's complete tracked-file ledger.

Each source report is self-digesting; `validate_arm_report` also records its actual JSON-file
digest in the compact evidence. Tool schema v2 binds the collector bytes and an exact digest
map of the stdlib observer, collector, report validator, serving profile, pair attester,
source-snapshot/source-bootstrap boundaries, and live fingerprint writer. Full-report
validation recomputes
that map from the attester's local source, while the collector itself first verifies the whole
tracked runtime snapshot closure. The reports additionally bind the exact target artifact,
image, 0.8.6 wheel/runtime/profile, host/GPU, closed process environment, resident extensions,
256K model limit, exactly
1,717,986,918 bytes of FP8 KV, `max_num_seqs=1`, `max_num_batched_tokens=512`, tool-enabled
launch, and one warm plus one measured pass of the same eight prompts × 128 fixed output
tokens. The only permitted launch differences are DSpark K=5 plus their necessary compiled
FULL_DECODE_ONLY graph contracts: `[1]` for no-MTP versus `[5,6]` for MTP. Distinct live
sessions are mandatory; the MTP measured responses and interval must be the exact acceptance
suite above, and both arms reconcile completed/generation/failure/timeout counters so another
request contaminating the interval refuses.

The threshold is predeclared in both reports and is a strict non-regression:
`mtp_warm_output_tokens_per_second / no_mtp_warm_output_tokens_per_second >= 1.0`; the receipt
records both that ratio and the absolute tokens/s delta. Each full report contains a continuous
one-second `/proc/meminfo:MemAvailable` ledger (maximum 2.5-second gap) spanning startup,
READY, warmup, measurement, and post-run. It replays model residency, startup/measured
transients, exact 256K KV capacity, no OOM/watchdog trip, a live post-run server, and the
existing production floors from `scripts/serve_dsv4_cb_validate.sh`: at least 110 GiB before
startup, 8 GiB at READY/post-run, and 4 GiB throughout. Reciprocal cards retain a compact
phase/KV/headroom receipt plus sample-ledger, report-payload, and report-file digests rather
than duplicating thousands of sampler rows (`validate_arm_report`, `validate_matched_result`).
A stack, budget, artifact, report, workload, counter, headroom, or route mismatch, insufficient
256K capacity, or slower MTP is therefore a hard failure.

The same claim digest-binds `dspark_baseline_comparison.v1`, an explicitly reference-only
Entrpi `ds4-on-spark` v0.5 / `ds4` v0.5.0 GB10 record pinned by exact commits and source-file
hashes. Its published DeepSeek-V4-Flash-0731 IQ2_XXS/Q2_K (~81-GiB) observations—about
960 tok/s at 2K, 933 tok/s at 64K, 776.4 tok/s over 517,963 tokens, about 28 tok/s single-chat
decode near 12K, engine-side 45.7 ms/token near 240K, and about 59 aggregate tok/s at
12-way concurrency—are comparison coordinates, never release thresholds. Local rows bind
the exact prompt/tokenization digest, uncached token count, prefix-cache-off/hit-zero state,
context start/end, chunk/batch/concurrency, endpoint counters, wall and internal timings,
loaded bytes/quant profile, and evidence-file digest. Mandatory comparability gaps record the
different engine/container, quantization/loaded bytes, unavailable exact upstream prompts,
and timing basis; the selected 256K profile truthfully marks the 517,963-token local row
unavailable instead of fabricating it
(`prismaquant/dspark_serving_profile.py:validate_baseline_comparison_evidence`). No Entrpi
headline derives a pass.

After all of those checks, `attest` writes one self-digesting paired
receipt into reciprocal optional `mtp.dspark` slots on the target and draft shipcards, binding
roles and peer model SHAs. The slot remains non-blocking when a target-only artifact omits it
or leaves it null. Every present non-null recognized optional claim is automatically replayed
by ordinary verification, while on-disk `provenance.dspark_cb_sidecar` makes `mtp.dspark`
mandatory for the draft even if its mutable card slot is deleted or nulled. Thus normal
publication verifies both reciprocal claims without an additional CLI flag; an operator may
still use `--require-slot mtp.dspark` as an explicit target-only demand. The attester also
requires a clean exact
PrismaQuant commit, so a dirty development tree cannot produce the release claim.

`_bf16_upgrade_audit` `:1965-2087` (emitted `:8622`) classifies each BF16 Linear as
passthrough/immutable, runtime-coerced, or a genuine budget choice — a manifest, not a policy;
serving-unit coercions are reported as such (`serving_group` key), because a whole FusedMoE
shipping unquantized is a different and louder fact than one Linear whose own shape was illegal.
`_coerce_runtime_legal_assignment` `:1452-1756` is the defensive legality re-check for stale or
hand-written recipes; it resolves whole serving-atomic components (§6.4).
`_unify_input_global_scales_across_fused_siblings` `:4624-4685` +
`_compute_nvfp4_joint_global` `:4688-4734` force one global scale per fused group (vLLM warns
and degrades otherwise). `_production_cache_fingerprint` `:1125-1182` /
`_production_cache_expected_keys` `:1088-1122` gate cache↔assignment coverage.

### 6.4 Hard serving invariants

Violating any of these yields a checkpoint that crashes vLLM at load or — worse — loads and
silently corrupts.

| Invariant | Enforced at | Failure mode |
|---|---|---|
| Native compressed-tensors fused siblings (q/k/v, gate/up) share **one** format; Gridbook role composites may use different storage formats | Native-lane DP aggregation over the intersection of member candidates; legality-aware union-find `promote_serving_units` `allocator_solver.py:302-327` + `_choose_group_format` `:192-231`; hard assert `promote_fused` `:362-406`; export re-check `:7896-7944`. DeepSeek-V4 intentionally declares no global dense `fused_groups`: Gridbook decodes each owned role independently to the common FP8 execution type before concatenation, so a native-only constraint must not couple its CB allocation. | On a native merged method, ≥2 schemes can crash at load and quantized + BF16 can silently corrupt (measured 4.3× worse served KL on Qwen3.x DeltaNet `in_proj_ba`, 0.106 vs 0.025). Applying that rule globally to Gridbook instead destroys legal per-role choices. |
| Packed MoE experts uniform per FusedMoE on released consumer contracts (mix across layers, never within) | pre-DP `aggregate_packed_serving_groups` (§4.5) + the same union-find pass via `profile.packed_expert_format_group`; native export raise `:7780-7792`; streaming CB legacy mode collapses uniformly | released Gridbook versions cannot consume a mixed bank; the opt-in PROPOSED per-expert v1 producer record above is the explicit exception under consumer reconciliation, not a relaxation of released serving invariants |
| PROPOSED per-expert sub-stacks partition each w13/w2 family exactly | streaming producer split planner + `artifact_completeness.py`; allocator-side bytes via `footprint.per_expert_format_group_payload_breakdown` | a missing/duplicate expert, undeclared tensor, or subgroup-byte mismatch is refused with layer/family/expert ids before the artifact can be treated as complete |
| A single-method serving unit is never left **mixed** by promotion or by export coercion | promotion picks the cheapest legal-for-all format ≥ max rank and writes **every** member unconditionally (`allocator_solver.py:192-299`); export coercion resolves whole unioned components, raising when a quantized format is legal for all and coercing the *whole* unit to BF16 only when none is (`:1452-1756`). An explicit Gridbook role composite is several role-owned methods under one vLLM module and is not a mixed single-method unit. | previously reachable via the un-aggregated solve path and, silently, via Pareto seed-JSON promotion (which `compute_achieved` never prices); the fused-coherence gate reported it only at the very END of export, and as a wrong-model-profile problem it is not |
| Incomplete fused groups → BF16 + `ignore` | `allocator.py:1482`; ignore back-fill `:7643-7700` | the fused loader expects all siblings; a missing `v_proj` breaks the merged Linear |
| Packed `config_groups` use vLLM **canonical** scheme names | `:7980-7994` | no scheme binds to FusedMoE; `w2_input_global_scale` never registers; `load_weights` KeyError |
| Multi-format menu must not resolve to `DefaultProfile` | `validate_default_profile_format_menu` `allocator.py:961-988`, called `:1550-1554` | silently produces the fused-coherence bug class above |
| A fixed quantized output head is independent, measured, and served by the selected native scheme | `allocator --lm-head-format` refuses tied heads, missing probe/cost rows, and unavailable candidates; `fixed_head.remaining_profile_pins` lifts only head aliases; allocator metadata controls export pinning; production rung `FP8_E4M3` | treating the fixed head as a body-DP item corrupts bpp/Δloss accounting; retaining its profile pin silently ships BF16 bytes; treating it as an embedding routes the wrong operation; a CB head has no `ParallelLMHead` method and fails load |
| Final serving promotion is a no-op | `validate_final_serving_promotion_noop` `allocator.py:1046-1063`, called `:2669` | a late promotion means the DP priced an assignment that is not the one shipped |
| Passthrough integrity (BF16/FP8_SOURCE only if the source already is) | `allocator_candidates.py:24-27`, `:112-120`; export judges it against the *same* `_scan_source_dtype_manifest` vocabulary (§6.3) | synthesising BF16 from a dequantised FP8 source burns 8 bpp for nothing |
| A candidate's exact per-unit payload never exceeds its known source-format payload | scaled owners derive from `SOURCE_PASSTHROUGH_CONTRACTS`, plain owners from their safetensors dtype; exact integer-byte gate `_source_bpp_applicability`; common byte authority in `footprint.py`; complete eliminations persisted under `format_applicability.json:source_bpp_legality` | without the gate, a quality-favoured rung can spend more bytes than the representation it replaces while still appearing in a compression frontier; an unknown source owner could hide the same error behind a guessed bpp |
| Every format in the assignment must have an emit path | `EXPORTABLE_FORMATS` `:7517`, checked `:1548`; the serving profile's `export_lane.codec_formats_from` bounds the allocator's menu by that same constant (`serving_profiles.py:252-330`) | a format with no `config_groups` scheme used to be silently rewritten to BF16 at 16 bpp, blowing the selected byte budget (#27) |
| Registry ↔ served metadata agree on bits/group | **not enforced** — `FormatSpec` (`format_registry.py:44-168`) and the export `*_SCHEME` constants (`:7247-7336`) are independent sources of truth with no reconciling test | a divergence mis-prices bpp or mis-declares the served scheme; §12 D17 |

### 6.5 Post-allocation LDLQ refinement (DSv4 A-FAST re-export, 2026-08-07)

The A-FAST burn's cost table was measured **without** LDLQ (`cbl_semantics.ldlq_in_measurement=false`
on every burn cell) but the per-tensor `cb_serialized_identity` already claimed `ldlq:true`
for the intended export bytes.  The cost and the bytes therefore disagreed, and the
`cb_render_identity` that would have made the mismatch fail-closed was absent from the
research-assembled `cost_merged` path (the Pareto writer then KeyErrored before it could
even stamp one).

The honest fix is **not** to relabel the raw cost as LDLQ and not to weaken any guard.
Instead the allocator's assignment (2.53 bpw, `c525f4025eac7061`, `predicted_dloss 619.71`)
stays on its raw cost, and the exporter applies a **byte-neutral, per-unit gated LDLQ
reassignment** on top of the already-chosen codebooks/scales:

* `nvfp4_cb_formats.ldlq_reassign_cb_fields_gated`
  (`PRISMAQUANT_CB_LDLQ_GATE=holdout`, default-on) keeps the raw indices per Linear
  (2-D) or per expert slice (3-D, mixing only the winning slices) unless LDLQ earns
  a **held-out certificate**: the decision comes from an LDLQ fitted on a
  deterministic, content-keyed half of the calibration rows and scored on the half
  it never saw, requiring strict improvement (ties keep raw). The **shipped**
  assignment remains the all-rows fit, which sees strictly more data than the arm
  that earned the certificate. Tensors with fewer than
  `LDLQ_GATE_MIN_ROWS = 16` rows are *uncertifiable* and keep raw
  (`nvfp4_cb_formats.py:2261`, enforced in `_ldlq_holdout_split` `:2309` and at the
  two decision sites `:2988`/`:3073`). The constant is the code's own evidence floor
  — at least eight fit and eight decision rows after the even split (`:2318-2319`) —
  and it is explicitly **not** a claim that sixteen rows deliver a population-level
  guarantee; the later model-level disjoint-corpus A/B remains the authority on
  whether LDLQ helps at all. This document previously said `= 2`, which was never
  the code.
  `ldlq_reassign_cb_fields` without the gate remains the verbatim assignment for
  cost-measurement parity.
* **Why held-out, not in-sample (2026-08-08).** The previous gate scored on the
  same rows that fitted the Hessian, so it could not fail. Measured across four
  support bands (L17 gate_proj, K12), its error was *anti-correlated* with the true
  benefit — 20× overstatement at 64 activation rows rising to 48.5× at 1–3 rows —
  because fewer rows are easier to fit exactly. Pricing from it would have inverted
  the allocator's ranking, not merely inflated it. Acceptance on held-out rows the
  gate never saw: degeneration **7/96 → 1/96**, and on full-support `down_proj` the
  new gate rejected exactly the one regressing expert. Evidence:
  `dq-runs/dsv4-flash-0731/ldlq-delta/{LDLQ_DIAGNOSIS,GATE_FIX}.md`. Legacy
  behaviour remains reachable as `PRISMAQUANT_CB_LDLQ_GATE=in_sample` for artifact
  reproduction only.
* `gate_info["holdout_ratio_per_expert"]` is the honest per-tensor out-of-sample
  LDLQ/raw output-MSE ratio, emitted as a by-product of the decision the gate must
  make anyway — so LDLQ pricing needs no separate measurement campaign. The ratio
  is constant in `K` (0.4692 / 0.4798 / 0.4770 at K12/K15/K18, certified across
  parity), but varies by support level and projection.
* The gate is byte-neutral by construction (fixed codebook, fixed scales, only the
  `k`-bit indices move), so `cb_tensor_payload_breakdown` and `whole_artifact_budget`
  are unchanged for the post-allocation refinement path.  Allocator optimality for
  that path is claimed only for the raw cost basis it actually optimized; LDLQ-cost
  optimality is not implied and requires the dual-basis reallocation below.
* Truthful provenance is `prismaquant.cb_ldlq_refinement.v1`
  (`cb_ldlq_refinement.py:build_refinement_provenance`): `cost_ldlq=false`,
  `export_ldlq=true`, `gate=holdout_activation_output_mse` (default since 2026-08-08;
  `activation_output_mse` remains accepted for pre-existing artifacts), `byte_neutral=true`, plus the
  creation timestamp.  The derived `layer_config.json` carries it under
  `__prismaquant__.post_allocation_refinement`, and both CB exporters copy it
  into `quant_config.json/provenance.post_allocation_refinement` after
  `validate_refinement_provenance` (invalid provenance aborts, it is never
  silently dropped).  A forged context stamp (claiming the cost was LDLQ) is
  never written.
* The **dual-basis** production recipe (scope `nvfp4`, §6.5.1) keeps the raw NVFP4
  bank as the immutable interpolation basis for FP8_CB, while the allocator-facing
  NVFP4 cost plane, the allocator itself, and the exporter all use the gated LDLQ
  NVFP4 plane.  Per-tensor identities therefore stamp `ldlq:true` for NVFP4_CB
  and `ldlq:false` for FP8_CB, and the global recipe stamps `ldlq_scope:nvfp4`.

`cb_fields_for_context` consults the gate (`_ldlq_gate_enabled`) and the scope
(`_ldlq_for_format`) so every production render — cost or export — shares the
same fixed-codebook LDLQ math under the declared scope, but only the dual-basis
reallocation makes the raw→LDLQ bridge an allocator-plane change rather than a
post-hoc polish.

#### 6.5.1 Dual-basis cost construction (scope `nvfp4`)

The production recipe keeps **three planes** in memory and on disk, never
re-labeling one as the other:

1. **NVFP4_CB raw** — immutable interpolation basis only. The burn's raw bank
   (`cbl_semantics.ldlq_in_measurement=false`) is preserved byte-for-byte for
   FP8_CB interpolation; it is never overwritten and its `cost_merged.pkl` is
   never patched in place.
2. **NVFP4_CB LDLQ** — the measured cost / allocator / export plane.  Each
   NVFP4 entry is re-measured with the fixed-codebook, fixed-scale LDLQ encoder
   (`PRISMAQUANT_CB_LDLQ_SCOPE=nvfp4`, `activation_output_mse` gate) and carries
   its own provenance (`raw_source_digest`, `ldlq_context`, `gate_metric`,
   `measured_vs_interpolated`, `output_metric`).  Direct measurement is preferred
   for the allocator-critical rungs (`K12/K15/K18` plus an independent `K16`
   holdout); if a saving law `saving(K)=mse_ldlq/mse_raw` is used to fill
   `K13/K14/K17`, it must pass a held-out composition gate
   `raw_interpolation × saving_interpolation` vs direct LDLQ at `K16` within the
   stated tolerance, otherwise all seven rungs are direct-measured.  The law is
   fit per-projection at minimum and tested for per-tensor/per-expert residuals.
3. **FP8_CB raw** — raw/interpolated plane.  All FP8_CB costs remain
   `ldlq:false` and are interpolated/projected from the **raw** NVFP4 bank
   (1), even after (2) replaces the NVFP4 cost plane.  Ordering and provenance
   are explicit: FP8 interpolation reads the raw bank, not the LDLQ bank, and
   each FP8 entry records `interpolation_source:raw`.

Gated LDLQ is used identically in cost and export for the NVFP4 plane: if a
unit falls back to raw, its allocator cost is the gated (raw) result and the
exporter makes the same deterministic decision from identical activation
evidence.  Aggregate and per-unit gate decisions are recorded durably
(`ldlq_gate_telemetry.json` plus per-tensor `gate` fields) and the final report
is based on observed counts, not a declared flag.  The raw interpolation plane
is always ungated raw by definition.

Re-allocation from the derived dual-basis table emits a fresh `layer_config`
with freshly computed, exact per-tensor identities under scope `nvfp4`
(`ldlq:true` for NVFP4_CB, `ldlq:false` for FP8_CB, global `ldlq_scope:nvfp4`);
the old identity map is preserved as the raw-cost optimum and a diff (bytes,
`predicted_dloss`, and assignment histogram) is published.  Allocator optimality
is claimed only for the cost plane actually measured — the raw plane for the
old artifact, the dual-basis LDLQ plane for the new one.

#### 6.5.2 The raw (no-LDLQ) render sidecar — one burn, two cost tables

An LDLQ-gated CB cost run already computes the exact no-LDLQ assignment internally, so
it costs nothing to keep it. Since `96bbf09` it does: the fields `cb_fields_for_context`
encodes **before** the gated reassignment ARE the identical-env raw render — same encode
tier, same codebook, same scale sweep and scale coding, same `col_weights` — and that
pre-gate assignment is captured through a caller-supplied `raw_fields_out` mapping
(`nvfp4_cb_footprint.py:1067`, populated `:1124-1129`/`:1145-1153`) and priced alongside
the primary. This is why the sidecar is sound rather than an approximation: it is not a
re-render, it is the render the gate declined to keep.

* **Row fields.** An LDLQ-covered CB row additionally carries
  `weight_mse_raw_render`, `predicted_dloss_raw_render` and — exactly where the primary
  has its per-expert vector — `weight_mse_per_expert_raw_render`
  (`measure_quant_cost.py:141-143`, emitted `:205-224`). The raw `predicted_dloss` runs
  the **same** Fisher math as the primary, including the sampled-expert `E/S` scaling, and
  `_extrapolate_expert_costs` carries the raw scalars so `PRISMAQUANT_EXPERT_COST_SAMPLE`
  groups stay extractable. Raw metrics **reconstruct**, never re-encode, and packed stacks
  are priced expert-slice-by-expert-slice through the holdout gate's chunked helper
  (`reconstruct_packed_cb_expert`) — a second full-stack fp32 residency is 16 GiB on the
  DSv4 fused `gate_up` 256×4096×4096 stack.
* **Output-side metrics are NOT re-measured for the raw arm.** The allocator prices
  `predicted_dloss`/`weight_mse`; a raw `output_mse` would require exactly the full-stack
  forward the sidecar exists to avoid. The extractor therefore stamps
  `output_mse=0.0`/`output_mse_measured=false` on every swapped row rather than inventing
  a number.
* **Provenance.** `prismaquant.cb_ldlq_raw_render_sidecar.v1`
  (`measure_quant_cost.py:139`, stamped into the payload at `:245-249`) states the
  identical-env no-LDLQ derivation.
* **Strict no-op when LDLQ is off.** `raw_fields_out` stays untouched, no sidecar keys are
  emitted, and cost pickles are byte-identical to the pre-`96bbf09` schema
  (`tests/test_cb_ldlq_raw_cost_sidecar.py` asserts the legacy row schema key-for-key).
  Nothing in the gated table's serialized identity moves either: scoring internals and
  additive sidecar fields are not part of the byte contract, and
  `packed_ldlq_artifact_stamp` / `cb_serialization_context_stamp` are untouched.
* **Ladder-rejected slices record no sidecar** — their rows mix interpolated values — and
  the extractor refuses them rather than silently averaging two bases.
* **`tools/extract_raw_cost_table.py`** turns a gated cost pickle into an
  allocator-consumable raw one: it swaps LDLQ-covered CB rows' metrics for the sidecar
  values (`cost_source="ldlq_raw_render_sidecar"`), copies rows LDLQ never touched
  (non-CB rows; the fp8 family under `scope=nvfp4`) verbatim, and re-stamps
  `cb_serialized_payload`/`cb_render_identity` as `ldlq=false, scope=none` — a model-free
  rebuild, because the col-weights and source-weights digests are LDLQ-independent. The
  source stamp is recorded under `derived_from_ldlq_gated_cost` (`:186`), and the result
  must pass `validate_cb_cost_provenance` under the no-LDLQ context before it is written
  (`:208-217`). It fail-closes on error rows, on a missing or partial sidecar, and on an
  already-raw input (`:89`, `:110`).
* **Why it exists.** It makes the LDLQ-contribution A/B — the isolate that says what LDLQ
  is actually worth on the serving metric — reachable from ONE cost run instead of a
  second multi-hour burn.
* **The never-routed hole is closed explicitly, not silently** (`cf0420e`). Declared
  never-routed experts (51 on the DSv4 capture) have no calibration activations by
  construction, so `cb_fields_for_context`'s pre-gate guard refused to render them under an
  LDLQ context and `_emit_weight_only_rows` crashed. The identity-correct row for those
  cells IS the raw render — the export-time holdout gate fail-closes them to raw
  (`raw_uncertifiable_too_few_rows`) — so the weight-only unrouted path passes an explicit
  `ldlq_missing_activation_ok` opt-in (`nvfp4_cb_footprint.py:1068`, honoured `:1115`;
  call-site `measure_quant_cost.py:60`), which returns the pre-gate raw fields and populates
  the sidecar capture with `raw == primary`, keeping the extractor's completeness check
  satisfied. **The default path still raises**, so a broken activation loader can never
  silently produce an all-raw table stamped as LDLQ.

## 7. Validation & ship gates

### 7.1 What runs where

| Stage | Tool | Run by the pipeline? | Verdict? |
|---|---|---|---|
| PrismaSnap BF16 fold fidelity | `tools/measure_vllm_full_kl.py` → `tools/prismasnap.py attest-fold-fidelity` | no — optional pre-pipeline candidate workflow | **blocking for a marked source**: original-BF16 → snapped-BF16 all-position forward KL `≤5e-4`, with BF16/no-spec-decode launch contracts and source, teacher payload, calibration, serve-fingerprint, index, weight-map, and shard-content replay; atomically transitions `MATERIALIZED` to `VERIFIED` |
| PrismaSnap source admission | `python -m prismaquant.prismasnap_contract --model "$MODEL_PATH"`; exporter-local replay/refusal helpers | yes, but only when the source carries the marker | binary: verified content is admitted only to native compressed-tensors; missing marker is a no-op; malformed/unverified marker and every marked GGUF/Gridbook source refuse before expensive work/output |
| Candidate real-KL (selection) | `validate_assignments_kl.py` | yes, only under `SELECTION_MODE=validated-surrogate` (`run-pipeline.sh:1223-1278`) | ranks, does not gate |
| Artifact survey (PPL/MMLU/end-KL) | `validation_harness.py` | no | **no thresholds at all** |
| vLLM load + greedy smoke | `validate_native_export.py` | **echoed only** (`run-pipeline.sh:1704-1705`) | binary |
| DSv4 CB exact eager + CUDA-graph load/generation | `scripts/serve_dsv4_cb_validate.sh {eager,graph}` → `validate_cb_endpoint.py` | no — operator-run, one fresh container per arm | **binary; each arm closes its matching `native_export.*` slot; eager also runs the independently recorded numeric gate before teardown** |
| Strict Qwen3.8 RTX 4090 FP8-CB eager + mandatory full-graph proof | `SERVE_ARM=eager|graph scripts/serve_qwen38_rtx4090_fp8_cb.sh` → `validate_rtx4090_fp8_cb.py` + `rtx4090_graph_contract.py` | no — operator-run on one physical RTX 4090 per fresh arm | **blocking for an on-disk strict artifact: exact RTX 4090/sm89, 32K, seq=1, FP8 KV exactly 4 GiB, immutable released Gridbook v11/device-qualified lane-v2 receipt, deterministic generation, and graph mode 3 + explicit Inductor + `FULL_AND_PIECEWISE` captures `[1,2,4,8,16,32,64]` with `fullgraph=True,dynamic=False` and no fallback. Graph fills `native_export.graph` and `rtx4090.fp8_cb`; eager fills `native_export.eager`. No physical run exists yet.** |
| Numeric ship gate | `validate_quantized_model.py` | never by the build pipeline; the DSv4 CB eager serve driver invokes it against its already-bound live session | yes, exit 0/1; closes `ship_gate` |
| Gold lane | `tools/measure_vllm_full_kl.py`, `tools/measure_vllm_wikitext_ppl.py` | never | manual, authoritative |
| DSv4 CB matched-budget performance | `python -m prismaquant.validate_cb_performance` | no — operator-run after export | **blocking paired prefill/decode/mixed parity against the exact displaced container** |
| DSv4 paired DSpark claim | `python -m prismaquant.dspark_matched_performance_collector {declare-policy,start-sampler,collect-arm}` → `python -m prismaquant.validate_dspark_target_draft attest` | no — lifecycle-owned collection around both exact operator-started serves; the collector does not launch or patch vLLM | **blocking only when `mtp.dspark` is claimed: source-closed/no-clobber arm reports, exact acceptance/routes plus matched no-MTP throughput non-regression, 256K KV, and 110/8/4-GiB headroom; target-only publication remains valid without the optional claim** |
| Ship record | `exported/shipcard.json` — native lane: `export_native_compressed._write_shipcard`; Tessera lane: `python -m prismaquant.lane_shipcard open --lane tessera` → `python -m prismaquant.shipcard_cli verify` | opened by the native and Tessera lanes; **the GGUF arm opens none** | **refuses** until every serve-lane slot is closed |
| **Publication** | `tools/publish_artifact.py` | no — operator-run | **BLOCKING**: refuses to upload (or even print the upload command) unless `shipcard.verify` passes |

Nothing in the ordinary quantization pipeline blocks on an artifact-quality number — and it
should not: `vllm` is not
importable in the build venv, so embedding a serve inside `run-pipeline.sh` would make the
build tool own the serving stack. PrismaSnap's `≤5e-4` check is different in kind: it is a
prerequisite source-integrity/fold-fidelity transition completed before `run-pipeline.sh`, not a
quantized-artifact promotion result. Once its `VERIFIED` record exists, the build pipeline only
replays that record and the current source bytes. The ordinary build/serve boundary remains
physical, so its contract is a **record**, not CI.

**The bar is defined once, per lane** (`prismaquant/lane_specs/*.json` + `lane_spec.py`, re-vet
**R16**). Each lane declares its `{serve command/scripts, endpoint, gate set, KL evaluator}`,
and every gate names the shipcard slot its record closes — so `LaneSpec` is the *runner's*
description and the shipcard is the *refusal*. Two things this made visible rather than
assumed: the CB half was pure wiring (native and CB declare the **same** ship-gate runner on
the **same** endpoint kind, because `validate_quantized_model.py` is endpoint-agnostic), and
GGUF's missing frontier evaluator is a thin adapter over its own harness
(`gguf_kl_evaluator.py`, §9.3). **Gates stay advisory in the pipeline; the blocking point is PUBLICATION** (Robert's ruling on
R16, 2026-07-30). Nothing in `run-pipeline.sh` blocks on a quality number and nothing should —
the build/serve boundary is physical. But an artifact only becomes a claim when it goes public,
so that is where the record is enforced: `tools/publish_artifact.py <artifact_dir> --repo-id
rdtand/<name>` calls `prismaquant.shipcard.verify` as a **library call** (never a subprocess —
the ambient python may not have the package) and **refuses before it uploads anything or even
prints the command it would run**, listing every unfilled or failing slot; a refusal that still
hands over a copy-pasteable command is not a refusal. Model publication accepts only the
canonical in-tree regular, non-symlink `<artifact_dir>/shipcard.json`; an external or aliased
receipt is never publication authority, including under the escape hatch. A parseable file
**A declared gate becomes an enforceable one through four links, and every one of
them is a value.** `shipcard_slot` is the whole mechanism: the build lane opens
the slot, the serve lane fills it, and `publish_artifact` refuses a card whose
slots are not closed. So a gate with no slot appears in `python -m
prismaquant.lane_spec`'s output and nowhere else. That is legitimate for a
diagnostic and illegitimate for a check an artifact's honesty rests on, and
telling the two apart cannot be an omission: `lane_spec.LaneGate.from_dict`
refuses `shipcard_slot: null` unless the gate also declares
`unrecorded_reason`, and refuses a gate that carries both, because a gate that
records is not unrecorded. `LaneSpec.unrecorded_gates` lists what remains, and
the CLI prints it as `UNRECORDED (advisory by declaration)` beside its reason
rather than as `UNFILLED`.

The second link is the card. `shipcard.lane_gate_slots(lane)` reads
`lane_specs/<lane>.json`'s own `gates[]`, so a lane opens a slot by declaring
the gate that closes it and no branch here names a lane;
`shipcard.build_shipcard(..., lane=)` stamps `card["lane"]` and opens those
slots, and `prismaquant.lane_shipcard open --lane <lane> --artifact <dir>` is
the entry point a driver arm calls. It refuses to clobber an existing card
without `--overwrite`, because re-opening one discards every slot the serve
lane has already filled. The third link is the union:
`shipcard.required_slots` extends `REQUIRED_SLOTS` with the lane's slots and
never replaces it, so **a lane can add a requirement and can never subtract
one** — the GGUF lane declares no `native_export.graph` gate and is still
required to close that slot. The vocabulary is derived from every lane spec
(`shipcard.all_slots`), and each derived slot names the verifier
`shipcard.verify` replays for it (`shipcard.LANE_SLOT_VERIFIERS`, dispatched
by registry, #162 closed). The fourth link is `publish_artifact`, above.

One lane holds all four links, and it is Tessera. Native
compressed-tensors holds the first, third and fourth but not the second:
`export_native_compressed._write_shipcard` calls `build_shipcard(out_dir,
build=build)` (`export_native_compressed.py:8487`) with no `lane=`, so a native
card carries no `lane` key and `required_slots` derives nothing from the lane.
That omission is deliberate -- `build_shipcard`'s own docstring says omitting
the argument reproduces the historical card exactly -- and a native artifact is
held to the same bar anyway, but **by coincidence rather than by derivation**:
`lane_gate_slots("compressed-tensors")` returns exactly `REQUIRED_SLOTS`, the
same five slots, so the lane-gated card and the base card are one card on this
lane. Read that as an equality that happens to hold today, not as a link. Add a
sixth gate to `lane_specs/compressed-tensors.json` and the native card will not
open its slot. **The GGUF arm opens no card at all**, so every gate
`lane_specs/gguf.json` declares is still enforced by nothing — `publish_artifact` refuses a GGUF artifact for
*absence* of a card, which an operator dissolves by writing a base card by
hand, and a hand-written base card never carries the lane's own gates. That is
the same defect on the remaining lane, recorded here rather than presented as
closed.

with the wrong schema, missing identity/build fields or required slot keys, or non-object slot
records is structurally malformed and is likewise never force-overrideable; force applies only
to a structurally valid card's unfilled or failed evidence. Allow/ignore
filters are forbidden because excluding a claimed artifact member or the shipcard would make
the uploaded model a different object, and model publication is repository-root only.

The publication authority is a **frozen snapshot**, not a final check followed by a live path.
The preflight shipcard replay is payload-cheap: it rejects failed slots and stat/model-identity
problems before paying for the tree scan, but does not traverse weight payloads. After that
preflight (and after an override stamp, if any), the publisher opens the complete tree
with no-follow descriptors and rejects symlinks and non-regular members. Small files are copied
into the private snapshot as bytes. Large files stay zero-copy: one sequential pass records
their full and per-8-MiB SHA-256 identities while retaining the open inode, the shipcard is
replayed against an internal `/proc/self/fd` view, and upload consumes that held-inode view.
Replacing an original path or swapping it to a symlink therefore cannot retarget the upload;
the post-commit whole-file replay makes any same-inode mutation across the upload window a loud
failure rather than verified success (if the remote commit already exists, the operator is told
to inspect it). This avoids a second 112-GB local copy while keeping one immutable inode as the
authority (`tools/publish_artifact.py:_freeze_artifact`, `_replay_frozen_digests`).

For the strict RTX4090 FP8-CB lane, that same freeze pass additionally parses each root
safetensors header through the shared exact-geometry validator and feeds every raw-tensor digest
from the same bytes used for the whole-container and upload-block hashes. The resulting
process-local receipt is bound to the source root and each shard's device/inode/size/mtime/ctime.
Authoritative frozen replay compares it directly with the frozen closed weight manifest,
canonical tensor payload identity, and exact shard index; it does not reopen mutable source
paths or invoke the independent path-based content scanner. Generic artifacts retain their
historical freeze behavior, and the post-upload held-descriptor replay above is unchanged.

Remote replacement uses the low-level Hub commit API rather than `upload_folder`. The publisher
resolves an explicit destination revision to one full parent commit, enumerates the remote file
set at that immutable commit, pre-uploads the frozen LFS objects with ignore rules disabled,
then submits every local addition plus every stale-file deletion in one `create_commit` with
both `revision` and `parent_commit`. Identical additions are deliberately retained in the
payload so the Hub client's no-op shortcut cannot bypass CAS. A concurrent head change is a
refusal with no automatic retry: the operator must restart the complete local freeze/verification
and remote enumeration. Root `.gitattributes` is the one intentional Hub-managed survivor when
the artifact does not replace it; the committed file set is enumerated again at the returned
commit before success is reported. There is no raw `hf upload` fallback because it cannot
preserve these guarantees; a missing compatible `huggingface_hub` is a refusal, while
`--dry-run` reports only the locally verified frozen manifest. The escape hatch remains
deliberately expensive: `--force-unverified` requires the
operator to **re-type the artifact directory's basename** (interactively, or `--confirm-name`
for scripts) and stamps `forced_unverified: true` plus the overridden problems into the
shipcard, so the artifact itself carries the record that it shipped ungated. Tests:
`tests/test_publish_artifact.py`.

**`uniform_control` (rate-axis artifacts only; PrismaQuant #121).** The
shipcard of a Tessera artifact carries one more required slot: the verdict of
its allocation against the byte-matched uniform control (`shipcard.py`
`_verify_uniform_control_record`, `make_uniform_control_record`,
`record_uniform_control_override`; CLI `fill-control` / `override-control`).
`verify` refuses a card whose allocation LOST to spending the same bytes
uniformly, and refuses the substitutes for a loss -- a control that is not
byte-matched (the match is replayed from integer bit counts, the carried flag
cross-checked), a self-widened tolerance above `MAX_CONTROL_RELATIVE_SLACK`, a
control arm whose `model_sha` equals the candidate's ("compared against
itself is not a control"), a `passed` flag that disagrees with the KL beside
it, or a control that never SERVED. The override hatch re-types the basename
at stamp time and binds to the card (`model_sha`, forgiven ratio) -- not to
the directory, which the publisher's snapshot renames. `publish_artifact`
prints the verdict on the same line as the bpp claim (principle 12: a size
claim travels with its quality caveat). Known limit: `uniform_control_summary`
prints producer-declared fields (`candidate_bpp`, `control_bpp`,
`relative_slack_ppm`) beside the bpp rather than the replayed values; `verify`
still refuses on the replay.

**`route.census` (Tessera-lane cards; PrismaQuant #136).** The shipcard of a
Tessera-lane card carries the lane's required `route.census` slot: the receipt
that the routes the serve actually emitted -- each stamped by the plugin with
the decoder that ran -- are the routes the artifact priced
(`prismaquant/tessera_route_receipt.py`, `make_route_census_record`; CLI
`fill-route-census`; the lane spec's `route.census` gate names this slot).
The requirement travels with the lane declaration (`required_slots` UNIONS
the lane's slots), not with a rate-axis special case. `verify` replays the
priced-vs-served comparison from the carried records
rather than trusting the carried boolean, and refuses a census run on a known
substitute decoder (derived from the pinned contract's `when_unavailable`,
never hardcoded), without a decoder, empty, or disagreeing with the priced
routes in either direction -- plus a hand-set `passed` flag that disagrees
with the replay. "No census was ever compared" reads as `UNFILLED`. Known
limit: coverage strictness is uncalibrated against a real serve (nothing has
been served yet on this side) -- both directions refuse, and relaxing either
needs a measured serve, not an argument.

The strict RTX4090 FP8-CB lane is the explicit exception to that generic
escape hatch. Strictness is re-derived from the canonical policy/profile,
specialized slot topology, and on-disk quantization manifest. Any invalid or
unfilled strict slot, frozen replay problem, or complete-tree size breach
returns before force confirmation or stamping; neither `--force-unverified`
nor `--confirm-name` can publish it (`tools/publish_artifact.py`).

**The ship record (`exported/shipcard.json`).** Native export opens a card
carrying the build-lane
facts it already holds — git commit, `assignment_hash`, `layer_config_sha`, achieved bpp *with
its provenance named* (read from the **recipe's own metadata**, never recomputed under a
different accounting convention: `selected_achieved_bits` when a validated frontier selected the
point, else the allocator's `achieved_bits`, and only then falling back to `pareto.knees.json`.
The knee file is a *separate* artifact describing the **surrogate** frontier, so under
`SELECTION_MODE=validated-surrogate` it does not describe the recipe beside it — reading it
first put a 1.25 bpp false claim on the Qwen3.8-27B arm-B card, 5.9994 for bytes that were the
validated 4.7496.

**That precedence is now backed by a cross-check, because precedence alone cannot catch a
right field naming the wrong point.** `shipcard.recipe_priced_bpp` sums
`cb_serialized_identity.tensor_payload_bytes` and `.params` over the units that declare them,
so numerator and denominator come from the same recipe entries and the result is scope-matched
by construction — no probe, no source manifest, no header or sidecar estimate. It is a **lower
bound**: units carrying no per-unit price (FP8_SOURCE passthrough Linears) leave both sums, and
they can only add bytes. `allocator_achieved_bpp` attaches the comparison as a `cross_check`
block on the card (advisory at export, so a gate bug can never strand a finished export), and
`shipcard.verify` **refuses** on `verdict == "DISAGREE"` — publication is the blocking point,
per R16. Tolerance is `RECIPE_BPP_CROSS_CHECK_TOLERANCE = 0.10`, far wider than label-scope
drift and far narrower than what it exists to catch. A card written before the gate existed
carries no verdict and is left alone. The motivating case is DSv4-Flash
`artifact-aura-cb-112p69`, which published **4.3065** bpp read from a sibling
`pareto.knees.json` for a recipe that prices to **2.7385** — 57% wide, and a false public bpp
silently breaks every matched-bpp comparison built on it. (Post-mortem: that recipe's own
metadata *did* carry the correct 2.7555, byte-identical to the `layer_config_sha` the card
recorded, so the precedence fix above already closes that specific instance; the cross-check is
the independent net that does not depend on getting precedence right.) Tests:
`tests/test_shipcard.py::test_recipe_priced_bpp_*`,
`::test_achieved_bpp_cross_check_*`.

**`read_gb_per_token` — the second rate, stamped wherever `achieved_bpp` is**
(`prismaquant/read_traffic.py`, campaign rule R2, 2026-08-21). bpp measures what an
artifact costs on disk; it does **not** measure what decode costs, because decode
throughput is governed by the bytes streamed per generated token and a sparse MoE reads
only `topk/E` of its expert mass on any one token. Measured on the shipped DSv4-Flash
87 GB artifact: the dense path is **8.3% of the checkpoint but 76.8% of decode read
traffic** (8.0576 GB/token at batch 1); re-measured on Ornith-1.5-35B-A3B's 24.62 GB
export it is **10.4% of the checkpoint and 81.5% of the read** (3.127 GB/token). The
allocator's byte budget is blind to that ~40x divergence in dense-vs-expert marginal
pricing, so it systematically overspends decode bandwidth on the dense path. Per
principle 1 that is a **measurement gap**, and this is the measurement; nothing in
`allocator_solver.py` changed, and pricing the axis inside the DP is a separate decision.

The definition is exact, not an estimate:
`read_bytes_per_token = Σ_tensor stored_bytes(tensor) × read_probability(tensor)`.
`READ_CLASS_TABLE` (`read_traffic.py`) is the single authority for the second factor:
routed-expert stacks get `num_experts_per_tok / n_routed_experts` (exact as an
expectation under the per-layer-uniform expert invariant §6.4 — routing skew changes
*which* experts are read, not how many bytes); allocator-assigned always-active units
(`dense`) and always-active tensors the allocator never decided (`held_fixed` — norms,
biases, routers, a pinned `lm_head`; DSv4's grouped `attn.wo_a`, which the probe prices
since the grouped Fisher accumulator landed and which sits here only while no
assignment covers it) get `1.0`. Both `topk` and `E` are read from the architecture's own config
declarations and **cross-checked against the tensors' measured stack depth**; an MoE
model that declares neither is refused rather than defaulted (principle 2 — the
`moe_imatrix` "assume 8" fallback would mis-price the largest term in the ledger).

Four classes are **excluded but itemized**, never silently dropped: an **untied** input
embedding (one row is gathered per token, not the table), **indexed lookup tables** (below),
the MTP/draft sidecar (read every token under spec-decode and never without it — the honest
default is excluded, and `excluded.mtp_bytes` lets a spec-decode serve add it back exactly),
and anything the model profile's own `checkpoint_to_live_name` declines to map into the live
text graph (vision/audio towers) — read through the shared name-projection
layer (§8.8.1), whose `declared_out_of_graph` outcome is what that rule
branches on. `ModelProfile.embedding_name()` (`model_profiles/base.py`, spec key
`shard_regexes.embedding_name`) is the twin of `lm_head_name()` and exists so "which
tensor is the embedding" is a declaration rather than a substring test.

**Tied embeddings are streamed, and that is decided by observation.** Under
`tie_word_embeddings` the embedding table *is* the output projection and the logits matmul
reads all of it every token, so excluding it would drop one of the largest always-active
tensors in the model (Qwen3-0.6B and LFM2.5 both tie). The invariant
`read_traffic.resolve_embedding_disposition` rests on is that **the logits projection is
streamed exactly once per token**: the embedding is `excluded_embedding` when the
checkpoint carries a separate output-projection tensor and `held_fixed` when it does not.
The config's `tie_word_embeddings` is a cross-check, not the decision — a config declaring
*untied* over a checkpoint with no output projection raises rather than picking a story,
and a config that merely omits the key is answered by the tensors (transformers defaults
that key to `True`, exactly the implicit default principle 2 forbids leaning on). The
resolution is reported per artifact under `embedding` (`streamed_per_token`, `read_class`,
`lm_head_tensor_present`, `config_tie_word_embeddings`, `reason`). It reads both namespaces
because the two declarations live in different ones: `lm_head_name()` is the checkpoint
spelling (DSv4 says `head`), `embedding_name()` the live one.

**An indexed lookup takes three facts, because each weaker rule was falsified on a real
artifact.** DSv4 ships `ffn.gate.tid2eid` — I64 `[129280, 6]`, token id to its six expert
ids — 18.6 MB read one row at a time, not streamed. But integer dtype alone does not mean
lookup: the packed weight payload of **both** quantized lanes is `U8` (81.65 GB of
`cb_qweight` on the DSv4 body, 15.7 GB of NVFP4 `weight_packed` on Ornith), so a dtype-only
rule excludes 94% of that artifact. Adding "leading axis is the vocabulary" is still not
enough — a **quantized `lm_head`** is both, and on the `embed-smoke` CB-head export that
two-fact rule dropped 857,736 B of real logits traffic, an *under*-count. So
`read_traffic.is_indexed_lookup` requires all three: integer dtype **and**
`shape[0] == vocab_size` **and** the module is not declared a quantized weight — neither by
a float scale sidecar (suffixes from `footprint._SIDECAR_SUFFIXES`) nor by the artifact's own
`config_groups[*].targets` (`quantization_targets()`; the CB lane needs this half, since a
CB payload keeps its scales in the codebook sidecar and has no scale tensor in the shard set).
The residual is one-directional by construction: anything the three facts cannot establish
stays at `p=1` and **over**-counts. Every report carries an `indexed_lookups` block naming
the rule, the vocab size, and how many integer bytes were read in full, so the over-count is
visible rather than silent.

CB codebook tables are reported as `resident_bytes`, not stream traffic — and the CB lane
ships them in a globbed sidecar *outside* the shard set, invisible to the safetensors
ledger, so the post-export form reads the artifact's own `quant_config.json`
`codebook_file` declaration, counts it as resident after the shard reconciliation has
passed, and **refuses** when the declared file is absent (687 KB on the shipped DSv4 CB
artifact; a reported `0` there would have been the same silent zero this module refuses
everywhere else).

Stored bytes are **not** a second copy of the byte math. Assigned units are priced by
`footprint.format_tensor_payload_breakdown` / the CB payload breakdown; every other
tensor is priced from the checkpoint's own safetensors spans via the new
`footprint.source_tensor_span_bytes`, which asserts it partitions the checkpoint exactly.
The two halves are then **reconciled against `footprint.assignment_artifact_bytes`
before any probability is applied**, and a one-byte disagreement raises — a ledger that
cannot drift without failing is what makes this reuse rather than duplication (project
memory: a silent zero in a per-tensor score ranks the broken arm first).

Where it is stamped, always beside the bpp: `export_native_compressed._write_shipcard`
and `shipcard.open_cb_export_shipcard` stamp `build.read_gb_per_token` measured from the
shards just written (`read_traffic.read_traffic_claim`), which cannot describe a
different assignment than the one on disk; `validate_assignments_kl` stamps a per-result
`read_gb_per_token` computed from the candidate assignment
(`assignment_read_traffic_claim`). All three are **advisory** in the same sense
`allocator_achieved_bpp`'s cross-check is: a failure reports a named `reason` rather than
stranding a finished export. Every claim carries `scope` (weights only, batch 1, KV cache
and activations excluded — so the figure is a lower bound on real decode traffic), the
four-key `breakdown` (`dense` / `routed` / `held_fixed` / `resident_codebooks`), the
itemized `excluded` bytes, and the `routing` factor with the config keys it came from.
Tests: `tests/test_read_traffic.py` (exact hand-computed ledger on a synthetic 4-expert
model, the `p = topk/E` and `p = 1` property, the classification table, tied/untied
embedding disposition, the three-fact lookup rule with both foils, the CB codebook sidecar,
and every refusal). Measured against real artifacts on both lanes: the shipped DSv4-Flash CB
87.08 GB body reconciles exactly and lands at **8.0576 GB/token**, matching the campaign's
independently measured 8.0576 to seven significant figures, with `excluded_indexed_lookup`
18.6 MB, `resident_codebooks` 687 KB and `excluded_non_text_graph` 0; Ornith-1.5-35B-A3B's
compressed-tensors export reconciles at 24,623,875,824 B for 3.127 GB/token. The
post-export form cannot split `dense` from `held_fixed` — a shipped checkpoint carries no
allocator/floor distinction on disk, so `breakdown.dense` is `0` there by construction and
the split is only meaningful in the pre-export `assignment_read_traffic` form.

Also on the card: exact `artifact_bytes`, format histogram, the render-lever
echo (`_render_lever_provenance()`, shared with the export cache's fingerprint so the two
cannot drift), and the `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` / `PRISMAQUANT_KV_COTANGENT` state
so an allocation that rode an unvalidated Fisher correction is visible on the artifact rather
than only in a probe log (D24) — and since 2026-09-03 `verify` refuses a card carrying
`unvalidated_kv_fisher_correction=true`, and shape-replays the stamped forensic hashes
(`layer_config_sha`, the two assignment digests), the histogram counts, and the
read-traffic value, so a fabricated forensic value no longer verifies (#158).
The `source_model` / `layer_config` path strings stay audit trail by design: they name
build-machine paths no verifier can resolve, and the same strings already travel in
`mixed_native_manifest.json`. Plus five base **empty, required** serve-lane
slots: `native_export.eager`, `native_export.graph`, `ship_gate`, `gold.kl`,
`gold.ppl`. Gridbook CB artifacts open a sixth blocking slot,
`perf.matched_budget_parity`; the generic record importer cannot fill it.
An artifact whose immutable on-disk producer policy is
`qwen38_27b_rtx4090_fp8_cb` additionally opens `rtx4090.fp8_cb`; verification
re-derives that obligation from `quant_config.json`, so deleting or nulling the
mutable slot cannot waive the exact physical-Ada, lane-v2, 32K/4-GiB, and
full-graph receipt (`shipcard.RTX4090_REQUIRED_SLOTS`,
`validate_rtx4090_fp8_cb.verify_rtx4090_shipcard_record`).

The card reserves a fixed 256 KiB (`shipcard.SHIPCARD_RESERVED_BYTES`) and every rewrite pads
with trailing JSON whitespace. That fixed size is load-bearing for CB: `shipcard.json` is
included in `provenance.artifact_inventory` and the exact whole-artifact budget before atomic
publication, yet its verdict slots are intentionally filled later. An oversized record fails
before writing; a normal fill therefore cannot stale `file_bytes`, change
`export_directory_bytes`, or cross the already-enforced budget. Transactional exporters resolve
the displayed `model_dir` through `directory_publication_target`, so it names the final artifact
rather than the private `.tmp-*` staging root.

`python -m prismaquant.shipcard_cli verify <card>` defaults the on-disk identity check to the
card's parent directory (an explicit `--model-dir` remains available) and exits non-zero unless
every slot holds a *passing* record whose `model_sha` matches the artifact. The flag is not
the whole gate: `verify` replays each slot's stamped evidence. `ship_gate` replays the
threshold contract against the fixed catastrophic bounds (a validator that silently lowers
`max_ppl` no longer closes the card), the exact four-check ledger, positive scored-token
evidence with known spec-decode state, and the endpoint binding's presence-and-shape
(#156). Each `native_export.*` slot replays its arm identity, residency
(`enforce_eager`), and generation evidence (`generated_chars > 0` on a pass) against the
slot suffix, so a mislabeled or empty smoke receipt no longer verifies (#157). Every recognized
optional slot whose value is non-null joins that default replay set. Missing or null
`mtp.dspark` remains nonblocking for a target-only artifact, but physical
`provenance.dspark_cb_sidecar` independently makes it required. CB identity adds
canonical `quant_config.json` with only its self-sized inventory excluded, an exporter-time
SHA-256 manifest of every final safetensors container, plus exact `.pqcb` content digests, to
the ordinary config-sha/per-shard-size identity. The production streaming exporter computes
the container digest over the exact header and tensor bytes as it writes them, binds the
in-stream byte count against the published file, and therefore does not make a second
100 GB-class NVMe pass; the resident exporter retains the one-time boundary hash fallback.
The shipcard caches size/mtime/ctime for fast
post-export mutation detection, so routine gates do not reread ~100 GB; a legitimate
cross-filesystem copy must run `shipcard_cli reattest`, which full-hashes the weights against
the immutable manifest before refreshing only that stat cache. CB native records must also
name `validate_cb_endpoint.py` and carry a canonical self-hashed endpoint
contract. Verification replays its exact closed launch options and switches,
artifact-conditional Marlin choice, the complete 31-variable Gridbook-0.8.11
environment snapshot (including affirmative absence), the endpoint preload/cache override,
current Gridbook/vLLM/image/GB10/TP=1 stack, exact imported-package origin,
affirmative absence of a server-side `PYTHONPATH`, complete
artifact plus released three-stage DSpark overlay, resident extensions,
deterministic endpoint smoke, raw serve-manifest digest, and positive
graph-log/capture evidence for the graph arm. Unknown or duplicate launch
arguments fail rather than hiding behind a required-flag subset.
Both `gold.*` records must report `spec_decode_detected: false`. `show` prints the remaining
unfilled slots. Validators fill their own structured slots via `--shipcard`; the generic `fill`
command is restricted to `gold.kl` / `gold.ppl` measurement JSONs. This turns "the numeric ship
gate was never run" (the row above)
from a silent omission into an explicit refusal. `verify` is not yet wired into
`run-pipeline.sh`'s closing echo — that is a follow-up wave.

**`validate_assignments_kl.py`** — the pipeline passes `--kl-scope full_sequence`
(`run-pipeline.sh:269`, `:1208`; option `:832`), `--n-calib-samples 32`, `--calib-seqlen 1024`,
and `--calib-skip-first $NSAMPLES` for held-out disjointness (`:1194-1219`); the CLI's own
defaults (2 × 128, `:767-925`) are not what ships. `_kl_repeat_summary` emits
`kl_mean/kl_std/kl_stderr/kl_ucb`. GPU-only via `gpu_guard.require_cuda_hot_path`.

*Key rename, 2026-07-30 (R28):* the mean is now `kl_mean` — it was `last_token_kl` under **both**
scopes, which had already misled a doc. `last_token_kl` is still emitted as a **deprecated alias
for one cycle**, and `select_validated_frontier._row_metric` resolves either, so pre-rename
result JSONs select identically.

*Per-sequence tail, 2026-07-30 (R9):* both measurement paths — `_measure_inplace_assignment_kl`
and `measure_assignment_kl(..., return_per_sequence=True)` — return `(mean, per_seq, stats)`
instead of discarding the per-sequence values they already accumulate. Each row therefore also
carries `kl_per_sample`, `kl_p95`, `kl_p99`, `kl_max` (**the same key names
`tools/measure_vllm_full_kl.py` emits**, so a selection row and a served row are comparable for
the first time; `kl_tail_domain: "sequence"` records the one honest difference — the sample unit
is a sequence, not a position) and the rung-2 term `nll_mean`/`nll_p99`, from one `gather` +
`logsumexp` over student logits already in hand (`kl_measurement.sequence_token_nll`, chunked so
the fp32 upcast stays bounded; `None` under the last-token scope, which has no next-token
label). **Zero extra forwards.** §4.6 is the consumer.

*Held-out disjointness is mechanized, 2026-07-30 (R14):* cost/probe artifacts stamp the
canonical `perturbed_x_cache.calibration_data_hash` into their output meta
(`incremental_probe` per shard, unioned as `calib_hashes` at merge; `aura_cost.provenance`;
`build_production_cache` onto `cache.metadata`, inherited by `production_render_cost`), and
`validate_assignments_kl` **hard-errors** when its own `calib_repeat_hashes` intersect the
probe's or the cost table's. Pre-R14 artifacts stamp nothing and the check stays inert on them
rather than guessing. Relatedly, `--calib-skip-first` on the wikitext branch was a **silent
no-op** (computed, never applied) and now raises — the mechanism that guarantees the held-out
split cannot quietly do nothing.

**`validation_harness.py`** — `validate_artifact` `:77-153` records `{ppl_wikitext, end_kl,
ppl_mmlu_acc, model_sha, layer_config_sha, eval_split, metric_era}` into `artifact_registry`
(`:18`); defaults 65,536 wikitext tokens, 200 MMLU questions, calib 8 × 512 on split `test`
(`:84-89`). Raises on non-finite metrics (`:156`), otherwise passes everything: measurement and
provenance, not a gate. `metric_era` matters — records lacking `eval_split` were measured on
wikitext **train** and are not face-value comparable (`:147-152`).

**`validate_native_export.py`** — does vLLM accept the checkpoint and emit tokens. Defaults
`--max-new-tokens 16`, `--gpu-memory-utilization 0.55`, `--max-model-len 2048` (`:206-209`);
eager by default, `--no-enforce-eager` `:226` is the graph-mode arm, and **`--both-arms`
`:229` runs both in one invocation** — the run-both-arms rule used to live only in the CLI
help text with nothing in code enforcing the second arm; it is now two named shipcard slots
(`_run_arm` `:112`, `_record_arm` `:174`, `--shipcard` `:234`), and each arm tears its engine
down before the next loads. A failed arm exits 1 instead of raising. Flashinfer pinned from
the profile's `runtime_package("flashinfer")` (`:30-71`); `--speculative-config` exercises MTP
(and marks the record `spec_decode_detected`).

**The profile flashinfer version is a FLOOR, not an exact pin (2026-08-14).**
`maybe_upgrade_flashinfer` compared `installed == version` and pip-installed the profile
version otherwise — so a *newer* flashinfer read as wrong and got **downgraded**. Measured on
the Qwen3.8-27B native-export gate: `gridbook:0.8.6-clean-dde15e0` ships flashinfer **0.6.18**,
`vllm_packed_moe` pins **0.6.8.post1**, and the gate downgraded a container that had just served
the same artifact cleanly, killing engine init with `ImportError: cannot import name
'set_autotune_process_group' from 'flashinfer.autotuner'` (vLLM 0.26 needs the newer API). The
upgrade runs **once per process**, before either arm, so **both arms died** — `flashinfer_autotune`
is on the engine-init path in each. (An earlier draft of this note guessed the graph arm survived
on a lazy path; the completed `--both-arms` log says otherwise, and the guess is corrected here
rather than left standing.) One consequence worth keeping: because the upgrade is process-scoped
and the arms are not, a harness-level environment fault can never present as an artifact-level
arm difference — if the two arms disagree, the artifact is the reason.
The comparison is now `installed >= version` with `.postN`/`.devN` suffixes ignored, preserving
the pin's original intent (images too *old* to dispatch the NVFP4 MoE backend on Blackwell)
while making it structurally unable to break a working container. `--no-flashinfer-upgrade`
`:219` remains the escape hatch. Note the container is `--rm`, so the bad downgrade was discarded
rather than persisted — do not assume that holds for a gate run in a long-lived container.

**Strict RTX 4090 FP8-CB graph/device gate (candidate, unrun).** This is not the
historical DSv4 `FULL_DECODE_ONLY` contract with a different GPU string. The
strict artifact's on-disk policy stamp makes `rtx4090.fp8_cb` mandatory even if
the mutable card omits the slot (`shipcard.RTX4090_REQUIRED_SLOTS`,
`shipcard._is_rtx4090_fp8_cb_artifact`). The operator runs
`scripts/serve_qwen38_rtx4090_fp8_cb.sh` twice, once with `SERVE_ARM=eager` and
once with `SERVE_ARM=graph`, in fresh evidence/cache directories. Both arms
require exactly one `NVIDIA GeForce RTX 4090`, compute capability `[8,9]`, TP=1,
`max_model_len=32768`, scheduler `max_num_seqs=64`,
`max_num_batched_tokens=32768`, FP8 KV,
and `kv_cache_memory_bytes=4294967296`; the artifact's complete recursive
inventory must still be at most 18,000,000,000 bytes
(`validate_rtx4090_fp8_cb.py`). A Spark, an sm89 cross-compile, or a reported
minimum capability cannot satisfy that identity.

The graph arm is fail-closed on both source shape and observed execution. The
in-container fingerprint binds the installed vLLM distribution/RECORD/import
origin and AST-checks exactly one direct `torch.compile` call with literal
`fullgraph=True`, literal `dynamic=False`, and an explicit backend. The launch
then requires compilation mode 3, backend `inductor`,
`FULL_AND_PIECEWISE`, and exact capture sizes `[1,2,4,8,16,32,64]` in a fresh
dedicated compile-cache root. `validate_rtx4090_graph_log` requires positive
Dynamo, compile-range, total-compile, FULL completion, seven-of-seven PIECEWISE,
and final capture markers and refuses every known compile/graph downgrade or
fallback marker. The scheduler ceiling of 64 is required for vLLM to admit the
FULL-decode capture ladder; the deterministic validation workload remains one
live request with `n=1`, and the contract does not claim that 64 concurrent
32K contexts fit in the fixed KV allocation. The graph arm closes both
`native_export.graph` and the
specialized `rtx4090.fp8_cb` receipt; eager closes `native_export.eager`.
Gold KL/PPL, the numeric ship gate, and any applicable performance gate retain
their ordinary independent slots. As of this stamp there is no physical RTX
4090 receipt, so none of these claims is promoted.

The GB10 validation-only producer is a different policy identity, not a weak
mode of this gate. `require_compile_only_gridbook_routes` accepts exact backed,
flag-free compile-only SM89 cells only for that identity; production continues
to call `require_device_qualified_gridbook_routes`. Its launcher first requires
one physical GB10 at compute capability 12.1, then reuses the strict pipeline
and FP8-only finalized census. The output keeps top-level `format: fp8_cb` and
cannot contain NVFP4 in assignments, config groups, or tensor formats, but it
also carries the immutable `UNRELEASABLE_VALIDATION_ONLY` disposition.
`shipcard.verify`, `tools/publish_artifact.py` (before any override/stamp path),
and `validate_rtx4090_fp8_cb.py` each refuse that disposition categorically.
Only `validate_rtx4090_fp8_cb_validation_only.py` may replay its structural
contract, and that command emits no serving or ship evidence.

A completed allocation can bypass the stock retained-format-menu workflow via
`rtx4090_validation_export.py`. Its preflight replays the exact assignment-bound
whole-artifact budget, requires a value-bearing/source-complete
`cb_render_identity`, hashes the supplied column-weight values through the
existing render-identity validator, and consumes the same compile-only v11
policy. It then constructs exactly one `export_nvfp4_cb_streaming` invocation
with lattice/v1/no-activation settings and the validation-only producer policy.
The exporter streams the source and rerenders only selected FP8-CB assignment
entries; it does not rebuild or retain a full candidate menu. Final recursive
inventory and the dedicated structural validator still enforce the artifact's
actual byte count at or below the assignment's ceiling and the absolute decimal
18 GB policy ceiling.

**DSv4 CB two-arm native gate.** `scripts/serve_dsv4_cb_validate.sh` owns the exact one-Spark
load/generation proof for Gridbook CB artifacts. The launcher first requires the artifact's
`shipcard.build.git` to identify one clean full PrismaQuant commit and the bootstrap checkout to
be clean at that commit. It materializes the complete tracked tree into the existing
commit/tree-addressed runtime-source cache, re-executes the launcher from that snapshot, removes
`PYTHONPATH`, and requires safe-path mode plus exact bootstrap import origin. The complete
snapshot closure is re-hashed before Gridbook preparation, before and after host validators,
inside the serving container before evidence capture, and around the terminal deferred shipcard
mutation; `/repo` is that read-only snapshot, never the live checkout. Host and container
bootstrap interpreters additionally require active no-bytecode and disabled-user-site modes, so
validation cannot mutate the cached snapshot with `__pycache__` or import user packages. The
container's stdlib-only fingerprint writer runs through the bootstrap's explicit
`serve-fingerprint` tool allowlist from neutral `/`, with no `PYTHONPATH`, and proves its lazy
`prismaquant.shipcard` import resolves to `/repo` before inspecting or writing evidence. Each arm starts a separate ephemeral container
from image digest `sha256:7bf752…`, requires the tracked released Gridbook (0.8.5 for the
recorded arms, the pinned 0.8.11 for a fresh run) from the tracked immutable
commit through the verified `git+file://<copied-checkout>@<pin.commit>` VCS target (never a
bare-directory install), and requires one `NVIDIA GB10`, TP=1, `--quantization gridbook`, FP8 KV, no speculative
decode, a resident reviewed Gridbook-native CUDA extension, and deterministic non-empty repeated
completions. The eager arm requires `--enforce-eager`. The graph arm instead pins
`FULL_DECODE_ONLY` with capture size 1 and refuses without the server log's positive
`Graph capturing finished …` marker after a compatible generation; merely omitting
`--enforce-eager` is not evidence. Both arms enforce the shared GPU lock, start/READY/watchdog
memory floors of 110/8/4 GiB, server-side process/extension fingerprinting, and a final 8-GiB
check. `validate_cb_endpoint` writes a deferred result first; only after the shell's final
process, watchdog, and memory checks does `commit_deferred_result` mutate the matching fixed-size
shipcard slot. The deferred commit rereads and hashes the serve manifest and graph log, so a
pre-commit file substitution invalidates the record. The launcher refuses an operator-supplied
served name and generates `dsv4-flash-gridbook-<32 lowercase hex>` from a fresh 128-bit nonce.
The endpoint receipt binds the manifest and mounted artifact to the exact process identities,
listener/socket ownership, physical GPU UUID, and serve-session fingerprint. Its `/v1/models`
identity is the stable one-model projection (`id`, `object`, `owned_by`, `root`, and
`max_model_len`); raw response bytes remain digested, but nondeterministic `created` and
`permission` fields are deliberately excluded from that projection. After manifest capture,
the smoke client re-observes `/v1/models` at the same endpoint and requires the same projection
before issuing deterministic completions, so a healthy unrelated listener cannot satisfy the
gate (`validate_cb_endpoint._validate_live_server_session`,
`validate_cb_endpoint.run_endpoint_smoke`,
`serve_fingerprint.models_endpoint_binding_identity`). The endpoint gate itself proves exact
load/capture/generation identity, not quality or speed. After that endpoint proof, the eager
driver runs `validate_quantized_model` against the same still-live nonce-bound process and
explicitly calls `shipcard.verify(required=("ship_gate",))` against the mounted artifact. It
also requires the written record's served-model nonce, base URL, and artifact path to match the
current session, so a warned-away shipcard write or stale passing record fails closed. Only then
may the driver tear down the server and commit the deferred `native_export.eager` result. The two
records remain semantically independent; the graph arm does not rerun the numeric gate, and the
two gold slots remain independent.

**DSv4 CB matched-budget performance gate.**
`prismaquant.validate_cb_performance` consumes a predeclared Cartesian matrix
of paired `gridbook.vllm-bench-serve.v2` reports and closes only
`perf.matched_budget_parity`. Candidate and baseline must use the same host
boot and physical GPU UUID and the exact released Gridbook/vLLM/image/GB10/TP=1
performance stack, closed server environment, normalized launch argv, workload,
and scheduling settings. They intentionally use distinct live server sessions
and artifact identities; one process identity may never be reused for two
artifacts. The matrix covers prefill, decode, and mixed traffic; concurrency
1/2/4/8/shipped-max; chunked prefill off/on; and plain and shipped decode modes.

Every arm of every matrix cell has a digest-bound **pre → report → post**
live attestation. The pre snapshot must be a report attachment, the post snapshot
must not predate the report, and their timestamps must satisfy
`pre.created ≤ report.started < report.finished ≤ post.created`. Apart from the
snapshot timestamp, phase, and resulting snapshot hash, every observed field must
be identical across the bracket. This pins one live serve session, exact process
identities and process-tree environment, listener/socket census and base URL,
mounted artifact identity, normalized argv, resident extensions, Gridbook/vLLM
runtime pins, host boot, and GPU throughout the measurement
(`validate_cb_performance._load_performance_serve_manifests`). Pairing then
requires the candidate and baseline stack fingerprints to match while preserving
their distinct session/artifact bindings.

Each report is unique and inventory-bound. Its concrete execution-assignment
ledger must enumerate exactly the certified DSv4 serving units and reconcile
every unit to the finalized artifact's sanctioned route and backend; CB and
delegated source/native units are distinct routes. For per-expert split stacks,
the execution-assignment ID is the complete consumer route
`<tensor_prefix>/<family>/<format_wire_id>`, not the physical tensor prefix alone.
Source-backed `w13` and `w2` routes may deliberately share that prefix, so including
family and wire id prevents their collision before the uniqueness and route-reconciliation
checks (`validate_cb_performance._derive_expected_execution_assignments`). The report-level
backend is one concrete backend when all assignments agree and `mixed` iff they differ;
its fallback summary is derived the same way, and every unit must attest no
fallback. Runtime routing is therefore replayed from the artifact and concrete
execution ledger rather than trusted from a label; an invented route or silent
fallback cannot pass. Four digest-bound
telemetry ledgers cover routing, occupancy, active experts, and the complete
grouped-MoE operator for both arms, all cells, all 43 layers, and every step; the
validator requires identical step coordinates across ledgers and recomputes
routed-token counts, expert histograms, and occupancy fractions before accepting
them (`validate_cb_performance._validate_telemetry`).

The compact shipcard persists every raw candidate/baseline block pair as
`paired_values`. `shipcard.verify` replays the ratio direction, every paired
ratio, median, conservative p05, per-cell verdict, release minimum, and matrix
digest from those values; derived summaries are not trusted. Conservative
block-level ratios must clear the predeclared phase-specific floor; tolerance is
capped at 5% and a strict release may set it to zero.

The release denominator is the exact container this artifact displaces, as
required by `AGENTS.md`, not a self-asserted synthetic optimum. Its recursive
inventory, current shipcard/endpoint eligibility, source identity, assignment
receipt, whole-artifact budget, and explicit displacement reason are bound in
the manifest and it is re-benchmarked in the same session. This does not make a
global-optimality claim. Separately,
`tools/certify_native_baseline_feasibility.py` reconstructs the complete DSv4
33,325-member/344-serving-unit body plus 22 DSpark construction units and every
legal no-CB option. The exact 112.690 GB proof currently gives a
165,024,004,576-byte lower bound (52,334,004,576 bytes over budget). That
certificate rules out an all-native comparator but never substitutes for the
served displaced-container arm.

### 7.2 `validate_quantized_model.py` — the numeric ship gate

Check order `:12-31`: serve → generation sanity → boundary behavior →
perplexity/NLL → MTP acceptance. Fixed
12-prompt PPL suite `:94-107`, 4-prompt generation suite `:112-118`,
5-prompt boundary suite `:137-144`. Thresholds `:163-181`,
CLI-overridable:

| Constant | Value | Rationale |
|---|---|---|
| `DEFAULT_MAX_PPL` | 25.0 | catastrophic-breakage bound only (BF16 ~3–5, 4-bit ~4–8) |
| `DEFAULT_MAX_P99_NLL` | 6.0 | ~2σ above BF16 mean; implemented as the **worst per-prompt** NLL guard (legacy flag name), true p99 reported separately (`:20-23`, `:65-69`, `:275-278`). Added after a broken 27B passed on the mean while 80% of prompts were broken — a mean cannot see a tail |
| `DEFAULT_MAX_MEAN_NLL` | 3.0 | mean NLL |
| `DEFAULT_MIN_GEN_LEN` | 30 chars | per completion |
| `DEFAULT_MIN_MTP_ACCEPT_P0` | 0.60 | position-0 draft acceptance |
| `DEFAULT_MAX_BOUNDARY_DEFECTS` | 0 | fail-closed historical value, **not a calibrated universal bound**: stock Qwen3-8B produced 10/15 at a 600-token cap. Replacement requires a same-session control-relative policy; #87 remains open |
| `DEFAULT_BOUNDARY_TEMPERATURE` | 1.0 | the unmodified distribution — any temperature > 0 leaves the argmax path greedy-smoke takes; temp 0 is refused, not sampled |
| `DEFAULT_BOUNDARY_MAX_TOKENS` | 64 | fail-closed historical value, **known too short**: healthy DSV4 produced 7/30 cap truncations. It remains only until the paired control derives its finishing cap from the model/context contract |
| `DEFAULT_BOUNDARY_REPS` | 6 | the published battery's own replication count (30 prompts × 6 reps): 5 prompts × 6 reps = 30 sampled generations |

**Sampled boundary behavior (#87).** KL/PPL (distribution distance) and
greedy-smoke (argmax agreement) are structurally blind to boundary-token
distribution defects that only manifest under sampling: three DSV4-Flash
quants within ~3% PPL spanned a 6x behavioral gap (14/180 to 83/180) on the
frozen battery, because greedy takes the argmax path where the boundary token
still wins and KL/PPL average a per-token near-tie at one boundary position
into noise. `check_boundary_behavior` samples the terse boundary-stressing
prompts (ultra-short numeric, terse QA, short recall — the first three
verbatim from the report) at temperature > 0 and scores every generation with
the server-free `score_boundary_text` for the closed defect vocabulary
`zero_tag` / `think_stutter` / `cap_truncation`. The request contract is
`prismaquant.boundary_chat_request/1`: POST `/v1/chat/completions`, one user
`messages` row, thinking enabled in the chat-template kwargs, reasoning
included, and special tokens retained. The response contract is
`prismaquant.boundary_chat_response/1`: raw `message.content` is scored
directly; when vLLM's reasoning parser has consumed the first close token and
split the response into `reasoning` (or legacy `reasoning_content`) plus
`content`, the client reconstructs exactly that one boundary only when both
reasoning-side and answer-side content are non-empty. A later close remains
visible as stutter, while either empty side remains zero-tag/cap-truncation.
Both schema identities and the endpoint are filed in the shipcard and replayed
offline.

The live check refuses an empty prompt/repetition population, non-finite or
nonpositive temperature, nonpositive integer token cap, or negative/noninteger
defect bound before contacting the endpoint. An empty sample cannot certify a
boundary check, regardless of its nominal defect count.

This endpoint fix is necessary and insufficient. Physical evidence invalidated
the current 64-token cap and a universal zero roster: healthy DSV4 still filed
7/30 cap truncations at 64, while stock Qwen3-8B filed 10/15 even at 600. The
pending policy is a same-session BF16 control whose cap grows until the control
reaches its own finishing fixed point, bounded by the declared model context
and an explicit backstop; the quantized arm is then scored control-relative at
that exact cap. The opt-in `boundary_control.py` instrument now specifies and
replays that paired receipt: every prompt/seed outcome, cap-growth step, score
and aggregate is checked, and the candidate names the exact control digest.
The CLI `tools/measure_boundary_control.py` reads the context from the live
`/v1/models` identity and input lengths from chat `/tokenize`; it retains
pre/post process/residency manifests and refuses a changed serve. It can
record raw-request A versus chat-request B at the historical initial cap,
followed by the uncensored control budget. Relative counts are per stratum,
without an invented significance or tolerance threshold. The budget is fit on
this frozen schedule, so the report is a detector and must not select artifact
calibration content. Until a physical paired measurement and a replacement
shipping policy are approved, the old 64/zero values remain fail-closed, #87
remains `needs-decision`, and this instrument cannot promote an artifact.

**Spec-decode refusal.** `_spec_decode_on` scrapes `/metrics` for
`vllm:spec_decode`; if present the perplexity check **refuses a verdict** rather than return
draft-model NLL. MTP artifacts need the two-serve workflow (`:37-54`): serve
without `--speculative-config` for the PPL verdict, re-serve with it for MTP acceptance;
ship-ready requires both. The same refusal now also guards the gold lane (§7.3) — it used to
exist only here.

**The guard fails closed, and the URL it uses is not `--base-url` verbatim (2026-08-14).**
`--base-url` is the **server** root: the module appends its `/v1/*` endpoints itself and reads
`/health` and `/metrics` off the root. The `compressed_tensors` lane spec published the OpenAI
root (`http://127.0.0.1:8000/v1`), so on the Qwen3.8-27B ship gate `wait_for_ready` polled
`/v1/health` — 404 — for 11 minutes and would have burned its whole 900 s timeout without
sending one prompt. Worse, `_spec_decode_on` hit `/v1/metrics` the same way, swallowed the
exception and returned `False`, i.e. **"no spec-decode detected"**: the one guard that stops a
draft-model NLL reaching a model card was issuing a confident all-clear it had no basis for, and
on the standard invocation it could never have fired at all. Now `_server_root` strips a trailing
`/v1` (preserving a `--root-path` prefix), `run_validation` normalizes once at entry so
completions is corrected too, and an unreadable `/metrics` raises `SpecDecodeUndetermined` →
the perplexity check fails with `spec_decode_detected: None`. Regression tests drive the `/v1`
spelling specifically; every pre-existing test passed the bare root, which is exactly why the
gap survived. The lane spec's `ship_gate` runner now spells the server root explicitly, and its
serve command names `gridbook:0.8.6-clean-dde15e0` — `vllm-fresh-b12x:latest` does not exist on
this box (see §"serving image reality") — with `vllm serve` given explicitly, since the image
ENTRYPOINT is `nvidia_entrypoint.sh` and passes argv through. The `gold.ppl` runner pointed at
`tools/measure_wikitext_ppl.py`, which has never existed; the file is
`tools/measure_vllm_wikitext_ppl.py`.

`--shipcard` (`:594`) appends this run's whole verdict block (per-check pass/fail, metrics,
thresholds, `base_url`, served model name, detected spec-decode state) to the `ship_gate` slot;
`--artifact-dir` (`:598`) names the local directory the `model_sha` is computed from, since the
validator drives an HTTP endpoint and cannot otherwise know what the server loaded
(`_fill_shipcard` `:516`, `_resolve_artifact_dir` `:502`). The DSv4 CB eager serve driver passes
the fixed default thresholds explicitly, supplies its generated nonce as `--model-name`, and
replays only `ship_gate` plus the current-session bindings before server teardown. This makes the
numeric run part of the manual eager release operation without merging its evidence semantics
into `native_export.eager`.

Honest limit of the `ship_gate` binding replay (#156): an offline `verify` has no live
session to compare against, so it refuses a record that names no server, no served model,
and no artifact path — but a well-formed binding to the WRONG server still passes here.
The `model_sha` match refuses a receipt measured against a different artifact's bytes;
binding a receipt to the live session it was measured in needs a nonce check before
teardown (the retired CB eager driver did exactly that). That check does not exist on
this lane: recorded as the open half of #156, not as a refusal.

`verify` replays that filing on every lane (#156): the slot closed on a bare `passed` flag
while the producer filed a threshold contract, a check ledger, token evidence and an endpoint
binding nobody read, so a receipt measured against the wrong server, with silently lowered
bounds, or with an empty ledger passed. The replay now demands the validator's tool identity
and a full producer commit, the exact catastrophic bounds against the filed thresholds, every
check present and passed, the perplexity numerics re-cleared against the bounds with a positive
scored-token count and a known non-spec-decode state, the boundary evidence replayed
(positive sampled-generation count, zero defects against the zero bound, sampling
temperature > 0 — a temp-0 "sampled" check is the old argmax-blind gate wearing the new
name), and the endpoint binding present and
shaped (`base_url`, `served_model_name`, `model_sha_source`). The binding half is
presence-and-shape by construction: an offline `verify` has no live session to compare
against, so a well-formed binding to the wrong server still passes here — catching that needs
the nonce-bound live-session check, which was the retired CB eager driver's pattern, not this
gate's.

### 7.3 The gold lane (manual)

**Served-artifact vLLM KL-vs-BF16** — `tools/measure_vllm_full_kl.py` retains the
exact-full-vocabulary path for teachers that fit its ordinary vLLM two-pass
workflow. DSv4Flash must instead use the digest-bound streamed-teacher path; its
release statistic is explicitly **all-position top-8192 support plus one tail
bucket**, not full-vocabulary KL.

**Why 8192 and not the 1024 this lane originally specified.** The support width
is set by measurement, not convention. The contract requires every one of the
4,088 scored positions to carry ≥ 0.90 probability mass inside the teacher's
top-K. On the first execution of this lane — it had never been run, on either
DSv4 artifact — a K=1024 teacher missed that floor on 34 positions (worst
0.6943), so the gate refused every teacher the lane could build. Sweeping K in
one pass over the pinned calibration (`topk` returns sorted, so a single K=16384
reduction yields every smaller K) gives min coverage 0.6943 / 0.7829 / 0.8606 /
0.9231 / 0.9658 at K = 1024 / 2048 / 4096 / 8192 / 16384. K=8192 is the smallest
width clearing the floor. **The 0.90 floor is unchanged** — the support was
widened until it could meet the guarantee, rather than the guarantee lowered to
fit a support that could not. `PROMPT_TOP_K` in `tools/full_kl_teacher_payload.py`
is the single source of that number; `shipcard.py`'s three DSv4 gold literals and
`tests/test_shipcard_gold_replay.py` all track it.

**Gridbook version on this lane.** The gold measurement tools bind the **serving**
pin: both call `load_gridbook_serving_runtime_pin()` and hand the attestation to
`self_manifest`, which compares it against the live runtime. A driver that
installed the 0.8.5/v3 *producer* pin therefore could not complete the KL step —
`run_gold_after_export.sh` does exactly that, which is consistent with that lane
never having produced a gold result. `run_gold_92gb.sh` installs the 0.8.8/v4
serving pin, matching both the tools and the pair that carried the artifact
through the serve gate. Since the 2026-08-21 lockstep the two pins name the same
release, so a producer-pin container no longer fails on the *version* leg; the
serving attestation additionally binds a reviewed wheel digest that the
producer helper's VCS install does not carry, and that leg has not been
re-tested here. Bind the gold tools to the serving pin regardless — it is the
pin they attest.

The exact DSv4 serving image intentionally does not install Hugging Face
`datasets`. Before either GPU measurement,
`tools/prepare_dsv4_wikitext_inputs.py` runs in the CPU preparation environment
with `datasets==4.6.0` and emits one strict-JSON
`prismaquant.dsv4_wikitext_inputs/1` payload. The loader binds the immutable
dataset revision, producer version, train/test fingerprints and complete-corpus
digests, full tokenizer-file identity, total token counts, exact KL windows and
PPL prefix, their value digests, and a whole-payload semantic digest. Both the
streamed teacher and DSv4 PPL command require `--wikitext-inputs`; neither
imports `datasets` in the GPU container. The legacy in-process DSv4 teacher
mode is refused rather than silently recovering the corpus at runtime.

`tools/build_streamed_full_kl_teacher.py` extends the existing
`cost_streaming.build_streamed_causal_lm` layer streamer: BF16 source weights,
one source `LayerCache` slot, one prefetch worker, and zero lookahead. It reduces
logits to FP32 top-8192 log probabilities on GPU before releasing the streamed
model. The closed calibration is WikiText-2 raw **train** revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, verbatim nonempty rows joined by
two newlines, tokenizer special tokens disabled, Python window seed 42,
8 samples × 512 tokens, and every next-token position: 511 per sample, exactly
4,088 positions. Student KL reconstructs the remaining teacher and student mass
as one tail bucket (`measure_vllm_full_kl._position_kl`).

The teacher payload is value-bearing evidence, not a cache hint.
`tools/full_kl_teacher_payload.py` binds the full streamed source identity and
its compact projection, tokenizer-file identity, dataset revision/fingerprint
and corpus digest, window starts and token-id digest, byte descriptors for
`calib_ids`/`topk_ids`/`topk_lps`, semantic payload digest, serialized payload
bytes, and metadata-file digest. Payload and metadata publish atomically; their
digests and semantic identities are the release evidence. Every tensor-payload load uses
`torch.load(..., weights_only=True)` through `safe_load_torch_payload`; a pickle object that
requires arbitrary reduction/code execution is rejected before semantic validation. The
serialized top-K rows are revalidated from their tensor values: ids are unique in-range
`int32`, FP32 log probabilities are finite, non-positive, and nonincreasing, and their summed
probability mass is finite and at most `1 + 1e-6`. Contract
`prismaquant.topk_tail_coverage_policy/1` additionally requires **at least 0.90 mass at every
position** (therefore at most 0.10 declared tail mass), and records recomputed mean/minimum
coverage; a caller-supplied summary is never trusted.

**Coverage is not fidelity, and a second gate says so.** A confidently *wrong*
distribution is still sharply peaked, so the coverage policy above is silent on
whether the teacher's forward pass is correct at all. On 2026-08-16 that
silence shipped a teacher whose own teacher-forced perplexity was 262 — far
worse than the 2.34-bpp student it was grading — and every existing gate
passed. `teacher_forward_fidelity_summary` (`tools/full_kl_teacher_payload.py`,
contract `prismaquant.teacher_forward_fidelity_policy/1`) closes it, and runs
inside `validate_teacher_payload`, so build, `teacher_meta` and
`load_teacher_evidence` all refuse: an unfaithful payload cannot be replayed
into shipcard evidence either. It recovers the teacher's own per-position NLL
from the payload alone — no extra forward — imputing out-of-support targets by
the two exact bounds the row already carries (declared tail mass, and the
smallest probability in a sorted top-K), which is a *lower* bound, so the
imputation can never manufacture a refusal. Positions are partitioned into
octaves of available context (the scale-free partition of a power law, and
fully determined by `SEQLEN`), and every ordered octave pair is compared by a
one-sided Welch t-test on per-position NLL. The enforced property is
**context-monotonicity**: a correct autoregressive model on contiguous natural
text improves with context and never inverts, whatever its absolute quality —
so the gate is scale-free and needs no absolute NLL threshold. "Materially
worse" is measured in the payload's own dispersion; the only convention is the
significance level, fixed at family-wise `1/scored_positions` split Bonferroni
across the comparisons, and the verdict is insensitive to it across ~60 decades.
A secondary absolute ceiling of `ln(vocab_size)` refuses a teacher no more
informative than uniform over its own vocabulary. Both the profile and the
worst comparison print on every build, refusal or not.

Student measurement must load and replay both files, carry the compact
`teacher_evidence` into its result, report exactly 4,088 positions, and require
the teacher source identity to equal the candidate artifact's source identity.
`resolved_commit: null` is an exact legitimate value for the pinned local DSv4
source and must compare equal; it is never a wildcard (`load_teacher_evidence`,
`measure_vllm_full_kl._assert_teacher_matches_candidate_source`, and
`shipcard._verify_dsv4_gridbook_gold_contract`).

**Direct WikiText PPL** — `tools/measure_vllm_wikitext_ppl.py` pins WikiText-2 raw `test` to
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, keeps verbatim nonempty rows joined by
two newlines, disables tokenizer special tokens, tokenizes the complete corpus, then selects
the first 8,192 token ids. Contract `prismaquant.wikitext_ppl_calibration/1` binds dataset
fingerprint and corpus SHA-256, artifact tokenizer-file identity, selected-token canonical-JSON
digest, and the exact 16 non-overlapping 512-token windows (8,176 next-token positions,
`prompt_logprobs=1`, no detokenization). The result carries the canonical contract digest and
`shipcard.verify` replays the revision, split, construction, tokenizer identity, token-prefix
identity, and window geometry instead of trusting only `split/n_tokens/seqlen`. Promotion
authority is §2.4; KL and this PPL are its instruments.

For DSv4 both tools must activate `tools.dsv4_gridbook_contract.exact_llm_contract`
before importing Gridbook/vLLM. Its one-Spark kwargs are closed to
`trust_remote_code=true`, BF16 dtype, TP=1, GPU utilization 0.84,
`max_logprobs=248320`, `quantization=gridbook`, FP8 KV, tokenizer mode
`deepseek_v4`, generation config `vllm`, prefix caching off, max model length
8192, max sequences 1, max batched tokens 512, 1,073,741,824 KV-cache bytes,
seed 0, eager execution, and log stats disabled; speculative decoding is off.
`moe_backend=marlin` is added iff the finalized artifact's live
`source_passthrough` or `per_expert_format_groups` assignment declares
`mxfp4_e2m1_ue8m0_g32`. Menus, provenance strings, and other metadata cannot
select that backend (`prismaquant.gridbook_assignment`). The closed relevant
environment is the complete 31-name Gridbook-0.8.11 snapshot in
`prismaquant.gridbook_environment`, not a two-variable subset. Gold clears the namespace first,
sets its 14 canonical values (including `VLLM_USE_DEEP_GEMM=0` and
`PRISMAQUANT_PRELOAD_FUSED=0`), and carries all 17 required
absences as explicit nulls. In particular retired `PRISMAQUANT_CB_DECODE` is absent, never
inherited and `GRIDBOOK_MXFP8_DENSE` is absent so the direct W8A8 route cannot replace the
source W8A16 method; runtime-pin override variables are removed separately before the first runtime
import. Endpoint and performance evidence use the same map with the one numerical override
`PRISMAQUANT_PRELOAD_FUSED=1` to equalize extension residency (the endpoint additionally binds
its persistent `PRISMAQUANT_CB_EXT_DIR`). The result carries the exact kwargs/environment
receipt and shipcard verification derives the expected Marlin choice again from the on-disk
artifact.

**That contract belongs to the DSv4 lane, not to CB** (corrected 2026-08-15).
`shipcard.verify` gated `_verify_dsv4_gridbook_gold_contract` on
`is_gridbook_cb`, which was true when `80a1c25` introduced it — DSv4 was then
the only CB lane with gold slots — and is not a property of CB. The contract
pins `tokenizer_mode=deepseek_v4` and `max_logprobs=248320`, so the first CB
artifact off that lane (Qwen3.8-27B CB) could not fill `gold.kl`/`gold.ppl` at
any effort: a gate no correct artifact can pass is a measurement gap, not a
missing measurement (§2.1 principle 1), the same class as the
`TOKENIZER_IDENTITY_SHA256` module constant that already refused this lane's
gold-inputs tool. The flag is now the lane itself,
`shipcard._is_dsv4_gridbook_artifact`, read off `config.json` —
already bound into `model_sha` as `config_sha`, so the lane cannot be flipped
without breaking artifact identity — and **fail-closed**: an artifact whose
architecture cannot be read keeps the strict contract, so a DSv4 release can
never shed it by hiding or corrupting its config. Both call sites move
together (`shipcard.verify`, `shipcard_cli._cmd_fill`), because a fill that is
stricter than publication would refuse evidence that ships fine and a looser
one would defer the refusal to the last gate. Everything generic stays on every
lane: finite slot metric, exact serve fingerprint, full producer commit,
position/token counts, and `score_positions=all`.

**The rest of the CB ship-gate stack was DSv4-shaped too** (2026-08-15). Fixing
the gold contract exposed that it was one instance of a pattern, not the
defect. A survey of every CB shipcard on this box found **none with a single
filled slot — DSv4's own included** — so the stack had never been executed
end-to-end, and every DSv4 constant in it had been written as a constant rather
than as a lane's value:

| Pin | Where | Verdict |
|---|---|---|
| `dsv4-flash-gridbook-` served-model prefix | `_SESSION_MODEL_RE` | lane-varying (brand only; the 32-hex nonce is the freshness proof and stays) |
| `--tokenizer-mode deepseek_v4` | `_canonical_launch_contract` | lane-varying — names a tokenizer only DSv4's vendored code registers |
| eugr Spark image digest | `validate_serve_manifest`, both `stack` blocks | lane-varying — which container carries the Gridbook wheel |
| `model_type == "deepseek_v4"` | `validate_cb_artifact` | **removed** — this gate proves the bytes are the shipcard's `nvfp4_cb` export served by the pinned stack; the architecture selects the lane, it does not gate admission |
| `perf.matched_budget_parity` required | `CB_REQUIRED_SLOTS` | lane-varying — its verifier needs five `displaced_container_*` digests, which a net-new size class cannot produce |
| vLLM version/commit, GPU, graph-capture config, the 8192/1-seq/1 GiB-KV/0.90-util gate parameters, the environment profile | throughout | **lane-invariant, unchanged** — the Gridbook image is built *from* the eugr base and reports the identical vLLM version and commit |

The three that vary live in one table, `validate_cb_endpoint.CB_SERVING_LANE_SPECS`,
keyed by `shipcard.cb_serving_lane`. `scripts/serve_dsv4_cb_validate.sh` (name
historical; the launcher is now lane-generic) **reads** that table instead of
restating it, so a serve it can produce is exactly a serve the gate accepts —
restating the constants on both sides is what let them drift apart. The
contract records its lane and publication re-derives the lane from
`config.json` independently: a receipt may declare which contract it was built
against, but not which contract it is judged by, and lane mismatch is refused
in **both** directions. Scoping the *demand* for `perf.matched_budget_parity`
does not create a hole in the *check* — a card that volunteers the slot off-lane
is still fully replayed.

**A third decode topology, and the vacuous pass it exposed** (2026-08-15). The
same survey reached `validate_cb_artifact_decode_contract`, which demanded
"exactly one of `dspark_source_overlay` or `dspark_cb_sidecar`" — two DSpark
topologies a plain Gridbook CB export has neither of. It now dispatches on the
lane: the DSv4 lane requires a declared topology (declaring neither would shed
the bridge contract its stack depends on — unless the omission is itself
declared, see the split release below), declaring *both* is refused everywhere,
and declaring a DSpark topology off the DSv4 lane is refused too. Off-lane
artifacts take `CB_PLAIN_MODE` under
`ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN` — a distinct mode, not a version bump,
because the evidence it can offer is *different*, not newer.

Writing that mode surfaced the real defect. `artifact_completeness` enumerated
candidate units by "`.weight` in an FP8 dtype", which no codebook-coded tensor
is: on the 27B NVFP4-CB export it classified **1 unit out of 818** and returned
`complete: True`. A cover proof over one tensor is a gate no artifact can fail,
which is the same defect as a gate no artifact can pass. The enumerator now also
reads `.cb_qweight` and `.weight_packed` planes, recognizes the artifact's
`quantized_embedding` declaration as a claiming mechanism, and bridges a
**fourth** namespace — a fused checkpoint unit (`…gate_up_proj`) claimed by its
unfused halves, via the profile's `fused_sibling_leaf_mapping`, requiring
*every* member to be claimed since a half-claimed fused group is unservable.
Qwen3.8-27B CB-A goes from 1 classified unit to 498 (497 CB + 1 NVFP4
embedding).

The plain mode's evidence is the cover itself, all of it recomputed from the
artifact's bytes rather than restated from `weight_content_manifest` /
`tensor_formats`: every quantized plane claimed by exactly one mechanism, every
group's targets resolved to tensors that exist, every group's units carrying an
identical plane signature (so a half-exported group refuses), and every
`codebook_ref` present in the `.pqcb` with the F16 payload digest its provenance
claims. `required_runtime_features` is *derived* — a pin missing a feature the
artifact never uses must not refuse it, and one missing a feature it does use
must — so a dense CB artifact requires none and a routed-MoE one requires the
per-role LUT.

**A fourth topology: the body half of a split release** (2026-08-16). At 92 GB
the DSv4-Flash body ships with `mtp.` excluded and the draft as a *second*
artifact, so it declares neither DSpark topology — and the lane guard refused
it. Both old branches were the wrong reading: it is not an in-band overlay and
it is not the sidecar. A DSv4 artifact may now decline both **iff** it records
`mtp.` in `provenance.excluded_namespaces`. The omission is the declaration, so
a body that merely *lost* its overlay (records nothing) still refuses, and
overlay-plus-exclusion is refused as a contradiction — the overlay is
constructed from exactly those source layers. It then takes `CB_PLAIN_MODE`, and
the disk-free receipt replay accepts exactly the one permitted omission (`[]` or
`["mtp."]`, a closed set — not "any exclusion").

Running the plain cover against a real FP8-source artifact for the first time
then broke three assumptions it had never had to hold (all 2026-08-16):

- **A group's target is a tensor.** 22 of 48 groups resolved to zero tensors,
  because 11 learned layers name the *halves* (`…experts.gate_proj`) of a packed
  `gate_up_proj` stack. That is the CB ABI, not a defect: a packed target binds
  exactly one `codebook_ref`, so a per-role learned book must be named per role,
  while a lattice layer's single shared book legally names the packed stack.
  Cover is now computed per **role**, decomposed through the profile's
  `packed_expert_projection_names` — matching gridbook's
  `_resolve_moe_codebook_roles` read *at the pin* rather than widening past it
  (read at 0.8.5, re-read unchanged at the pinned 0.8.11) —
  and a stack whose roles are only partly claimed is refused as unservable, the
  same rule the fused-sibling bridge already applies.
- **`source_passthrough` did not exist to the cover.** The third declared
  claiming mechanism was invisible, so 336 verbatim-FP8 units read as claimed by
  nothing. It is now read exactly as `artifact_completeness` reads it: by exact
  variant, not by ancestry. A unit legitimately carries **both** a passthrough
  declaration and a scheme-less config group — the group states LAYOUT, the
  declaration states ROUTING (`cb_export_config.py`) — and 272 of the 336 carry
  the pair, so refusing it would refuse every FP8-source release.
- **Groups partition units.** `covered == cb_unit_count` cannot hold once two
  per-role groups share one packed stack (398 vs 115). The assembly site, which
  holds both sets, now asserts set equality against the classified CB units
  after subtracting passthrough; the disk-free replay, which does not hold them,
  keeps only the bound it can honestly assert — the cover may not fall *short*.

With those, `artifact-aura-cb-92gb` validates `cb_plain` complete: 115 CB + 336
passthrough over 1736 tensors. `artifact-aura-cb-112p69`'s `evidence_sha256` is
byte-identical across the change, and the ordinary plain fixture's
`cover_sha256`/`evidence_sha256` are pinned in
`tests/test_validate_cb_endpoint_plain.py` from the pre-change code, so a future
cover change cannot silently move every artifact's receipt.

*Closed with them:* the 8 `…ffn.experts.gate_up_proj` stacks previously reported
on `artifact-aura-cb-112p69` as claimed by nothing were the same per-role blind
spot mirrored in `artifact_completeness`, not a `fused_sibling_leaf_mapping`
gap.

**The gate could not run its own fix** (2026-08-16). Both fixes above landed
after the 92 GB body was built, and `scripts/serve_dsv4_cb_validate.sh` refused
unless `REPO_HEAD == ARTIFACT_BUILD_COMMIT`, then materialized and re-exec'd the
snapshot **at that commit**. So the gate would have executed the *pre-fix*
validator against the artifact, and the only remedy it admitted for a wrong
verdict was rebuilding bytes that were never wrong — five to six GPU-hours to
reproduce the same tensors under a newer stamp. That is a defect in the gate,
not a property of the artifact: the binding covered the **judge** as well as the
**runtime**, and `validate_cb_endpoint.py` decodes nothing.

The two roles are now separate identities, and the launcher materializes and
verifies **both**:

| | commit | role |
|---|---|---|
| **runtime** | artifact `build.git.commit` | mounted at `/repo`, sources the Gridbook serving runtime, executes every in-container step |
| **judge** | the live checkout's `HEAD` | runs the launcher and every host-side verdict |

Nothing about the serve is relaxed — the container still runs exactly the stack
the bytes were made for, and its in-container closure verify still expects the
build commit. What changed is that a gate fix now reaches artifacts already on
disk. Two guards keep it from becoming a way to judge with unrelated code. The
judge must be a **descendant** of the build commit (`git merge-base
--is-ancestor`) — forward only, never older than the producer. And
`prismaquant_runtime_snapshot.py judge-divergence` proves every closure path
that differs between the two snapshots is in `JUDGE_ONLY_PATHS`: `docs/`,
`tests/`, and three named host-side modules. A divergence anywhere in the serve
path refuses, because a judge that also moved the serve path may be expecting
something the build-commit runtime does not produce — and *there* re-export is
the honest answer. The claim is re-proved at every `verify_runtime_snapshot`
checkpoint rather than trusted once at bootstrap, and `evidence/judge_split.json`
records both identities with the divergence list, so a receipt can always say
which code rendered its verdict. On the 92 GB body the divergent set is exactly
the eight paths of `1467c90` + `2027c60` + this entry.

Two subtleties the split had to respect. `PQ_RUNTIME_PRISMAQUANT_ROOT` is the
*source-bootstrap* contract — `activate_prismaquant_source` refuses unless it
equals the directory of the tool being run — so on the host it names the judge
root, while the container keeps its own explicit `/repo`. And
`PRISMAQUANT_IDENTITY_GIT_COMMIT` stays the **artifact build commit**: a judge
running newer does not restamp what produced the bytes.

`serve_manifest.json` is excluded from `compute_model_sha` for the same reason
it is written at all: it is the R15 fingerprint of a *serve*, not artifact
content, so hashing it made validating an artifact invalidate its own card
(observed on Qwen3.8-27B CB-A, `677f278a` → `bf6abc17` after the eager smoke).
Its integrity is bound where the claim is made — each record's
`*serve_manifest_sha256`.

**`README.md` is excluded on the same principle, one step later in the
release.** On the Hub a `README.md` in the model directory *is* the model card,
and `publish_artifact.py` takes no `--model-card`, so the card can only be
delivered by writing that file into the artifact — which, while it was hashed,
invalidated the very shipcard that authorizes the upload. It had already
happened: `qwen38-27b-arm-b/exported` carries card `e7ac09f8…` against a disk
`3c4a83a1…` and cannot publish. The deeper failure is ordering, not
bookkeeping: every gate record binds `model_sha`, the gold KL/PPL only exist
*after* the gates run, and writing them into the card broke the records that
produced them — so an artifact could not quote its own measured numbers. Gating
twice is not a way out, because KL is bit-identical only *within* a docker
session (`kl_session_arithmetic_drift`), so a second round would print numbers
its own records disagreed with.

`compute_model_sha(..., legacy_readme_hashed=True)` reproduces the old identity
for cards stamped before this, and both `verify` and `reattest` consult one
`accepted_model_shas()` helper rather than each carrying its own tolerance —
the two drifting apart on which legacy identities they accept is precisely the
bug this shape prevents. Because the excluded set is identical either way, a
directory that had no README when it was stamped hashes the same under both
code paths: Qwen3.8-27B CB-A was stamped `fe89b79a…` before the patch and still
verifies at `fe89b79a…` with its card in place. Tests:
`test_documenting_an_artifact_does_not_invalidate_its_own_card`,
`test_a_card_stamped_while_the_readme_was_hashed_still_verifies`.

**`tools/measure_served_gold.py` — the served lane's gold tool.** Both DSv4
tools own an in-process `LLM`, so every other lane had been measuring its gold
numbers with run-local scripts that emitted a bare metric and none of the
identity a ship record binds. This one talks to an ordinary
`vllm serve` + `/v1/completions` and emits records
`shipcard_cli fill` accepts: `dump` saves one arm's top-K prompt logprobs,
`kl` folds a teacher and a student dump into a `gold.kl` record, `ppl`
teacher-forces a `gold.ppl` record at `prompt_logprobs=0`. The serve
fingerprint is *read* from the in-container `serve_manifest.json`, never
computed client-side (the measuring client cannot see the server's address
space — §7.4), and the tool refuses a manifest that describes a different
served model, a manifest recording a speculative config, a `/metrics`
spec-decode state that is not a clean `false`, two arms dumped at different
top-K, and two arms that scored different text. Producer identity comes from
`serve_fingerprint.gold_producer_identity`, which refuses a dirty tree.
**The one honest caveat, carried on every record rather than in prose:** a
served endpoint returns top-K prompt logprobs, so this is a top-K KL with a
declared tail, not the exact full-vocab KL of §2.4 authority #1. Records carry
`prompt_top_k`, recomputed `topk_coverage_mean`/`topk_coverage_min`, and
`student_tail_model` — the untabulated tail is the max-entropy
`log(residual / (V − K))` clamped at the K-th value, because the K-th logprob
is an upper bound on everything below it and substituting it (what the ad-hoc
script did) can only make the student look closer than it is. `kl_mean` over
all positions is floor-inflated; `kl_confident_mean`, restricted to teacher
top-1 > 0.5, is the number to quote.

Both DSv4 tools own an in-process `LLM`; on current vLLM the measurement process
is the parent and EngineCore is a child. Two guards ride on that:

* **Spec-decode refusal** (`tools/spec_decode_guard.py`). Rung-1 authority had no spec-decode
  guard at all until R13 — the refusal existed only in §7.2. `_load_llm` now inspects the live
  engine's `speculative_config` and raises with the draft-NLL diagnostic (`--allow-spec-decode`
  overrides, and the shipcard then refuses the record). Every result dict carries
  `spec_decode_detected`; `None` means "could not inspect" and is refused too — an unverified
  negative is what the original trap looked like.
* **Parent + EngineCore live attestation** (`tools.serve_fingerprint.self_manifest`).
  DSv4 gold collection fingerprints the measurement parent and its complete
  live descendant process tree, requires an EngineCore/vLLM-engine descendant
  proven by that tree (never a host-global `pgrep`), and unions extension
  residency across their address spaces. Process identities, environment,
  listener census, artifact binding, exact Gridbook/vLLM/image/GPU stack,
  runtime pin, effective kwargs, and the resulting serve-session identity are
  replayed by `shipcard.verify`. Each result dict carries `git_commit`,
  `serve_fingerprint`, and the full `serve_manifest`; missing, unreadable, or
  unrelated engine evidence cannot close a gold slot.
  **The environment half of that attestation requires `SPT_NOENV=1`, which both
  runtime Docker vectors now set** (`prismaquant/gridbook_runtime/gridbook_runtime.sh`,
  `gridbook_serving_runtime.sh`). The census reads `/proc/<pid>/environ`, and
  vLLM's EngineCore renames itself through `setproctitle`
  (`vllm/v1/engine/core.py` → `vllm.utils.system_utils.set_process_title`),
  which on Linux overwrites the contiguous argv+envp block and destroys that
  file while leaving the process's real `os.environ` intact. Measured in the
  pinned serve image across that one call, `/proc/self/environ` went from all
  six probed variables to zero while `os.environ` kept all six. Without the
  flag the census reads a destroyed remnant for the one process that runs the
  CB kernels, `consistent` is `False`, and a correct server is refused —
  structurally, on every lane, at every commit. `SPT_NOENV` is not a
  relaxation and is deliberately excluded from every compared allowlist: values
  are still matched exactly, so a genuinely mismatched EngineCore environment
  still fails (`tests/test_gridbook_runtime_boundary.py`).
* **Clean producer and installed-runtime closure.** Each gold manifest binds a full
  PrismaQuant commit, independently observed `git_dirty=false`, optional tree id, and byte
  descriptors for the exact common/tool source-file closure. It separately attests the
  installed Gridbook distribution: package/version, PEP 610 exact VCS
  requested/resolved commit or independently pinned release-wheel SHA-256,
  `direct_url.json`, `METADATA`, `RECORD`, and every installed Python/CUDA/package-data file
  checked back against its RECORD SHA-256 and size. It also requires the lazy top-level
  package's resolved `__file__`, `__spec__.origin`, and every `__path__` entry to stay within
  the selected distribution's real package root, with the imported version equal to the pin.
  The complete server-process environment projection separately proves that
  `PYTHONPATH` is absent. A same-version CWD/`PYTHONPATH` shadow, dirty producer,
  bare local install, or post-install source mutation cannot close a slot
  (`serve_fingerprint.gold_producer_identity`, `gridbook_distribution_provenance`).
* **Assignment-derived extension closure.** Gold verification parses the finalized
  `quant_config.json` rather than accepting "some Gridbook `.so`". Any CB assignment requires
  `prismaquant_cb_ext`; layout-v2 NVFP4-CB additionally requires
  `prismaquant_cb_v2_ext`; a direct group-32 MXFP8 assignment requires
  `pq_mxfp8_dense_*`; and the block-128 source W8A16 route requires both
  `pq_fp8_source_w8a16_*` and the large-M `pq_cb_bf16_grouped_*` bridge. Only the families
  actually implied by `config_groups`,
  `provenance.tensor_formats`, `source_passthrough`, and per-expert groups are demanded
  (`shipcard._gold_extension_requirements`).

### 7.4 Reproducibility contract

KL is **bit-identical within one docker session** and drifts 4–8× **across** sessions, so
provenance is baked into every KL output JSON: `_git_provenance`
`validate_assignments_kl.py:280`, `_calibration_provenance` `:307` (calib sha256),
`assignment_hash` `:1344`/`:1380`, cache `cache_hit_count` / `rtn_fallback_count` `:371-373`.
An output without these is quarantined, not compared.

**Mechanism of the cross-session drift (2026-07-19).** Loading *any* CUDA extension into the
serving process shifts allocator addresses → activations get different pointer alignments →
alignment-sensitive cuBLAS/CUTLASS heuristic selection elsewhere → ULP-level logit drift. On
the 27B this reads as two bit-reproducible states, conf-KL 0.01134 vs 0.01328 (**±17%**), keyed
purely on whether the gridbook extension `.so` was resident during the dump; ~97% of positions
drift uniformly, so it is global, not path-local.

**Rule:** A/B arms must have identical extension residency and ideally identical
pre-measurement traffic. conf-KL deltas below ~±20% across differing serving stacks are not
evidence either way and should be quoted as a range.

**Mechanized (R15).** The rule is no longer prose an author has to remember.

* **`serve_manifest.json`, written server-side.** Each `scripts/serve_*.sh`, once READY, calls
  `write_serve_manifest` (`scripts/lib/serve_manifest.sh`), which `docker exec`s
  `tools/serve_fingerprint.py write` **inside the container**: launch argv, image tag,
  `vllm`/`torch`/driver versions (`importlib.metadata` + NVML only — the writer never imports
  torch or touches CUDA, so it cannot add a context to a 121 GiB pool), GPU name,
  `enforce_eager`, `--quantization`, `PRISMAQUANT_*` env, and the resident-extension basenames
  read from the **server's** `/proc/<pid>/maps` via `matches_tracked_extension`, which has
   two arms: `SUBSTRING_EXTENSION_PATTERN`
   (`prismaquant|pq_(?:cb|mxfp8|fp8_source)|flashinfer|causal_conv1d|/fla/` — `gridbook`
   was dropped on 2026-09-02 with the lane, and bare `fla` became the package
   directory `/fla/` on 2026-09-03: three free letters match `libflac.so`) for
   libraries this repository recognises by
  name, and `TESSERA_NATIVE_EXTENSIONS` for a runtime that publishes its own loadable
  libraries, matched by the rule that runtime names (§9.4),
  unioned over the API-server *and* EngineCore processes — it is the engine that holds the
  kernels). Client-side is not an option: the measuring client cannot see the server's address
  space — reading a root-owned container process's maps from the host is *denied*, and the
  denial is indistinguishable from "nothing is resident", which is exactly why the ±17% stayed
  invisible. The manifest therefore records `residency_readable` and folds it into the
  fingerprint, so an unverified scan can never match a verified empty one. Never fatal — a
  serve that came up is not torn down over a JSON.
* **Two fingerprints, two identities.** `performance_stack_fingerprint` is SHA-256 over the
  canonical performance projection: image, physical GPU/driver, package and Gridbook
  distribution/import-origin identities, resident extensions/readability, normalized server argv, closed
  process environment, and listener stack. It intentionally excludes the arm artifact and
  live-session identity, so independently served A/B arms can match. `serve_fingerprint`
  remains the full stable artifact/session attestation; independent arms normally have
  different values. Observation-only creation/process fields and `attestation_phase` are
  excluded, so DSpark pre/post observations of one unchanged server must have the exact same
  value, while a changed resident extension, package, runtime capability, artifact, or session
  changes it. In-process gold-lane runs use `self_manifest` over the measurement parent
  and its proven transitive descendants; DSv4 additionally requires an EngineCore/vLLM engine
  in that exact tree and unions residency across all readable address spaces.
* **`tools/kl_ab.py A.json B.json` validates both, then compares only the performance stack.**
  It recomputes each manifest's `performance_stack_fingerprint` and `serve_fingerprint`, checks
  any top-level copies, and refuses stale, missing, or manifest-less current attestations.
  Matching performance stacks permit a delta even though the validated per-run serve
  fingerprints differ. Different performance stacks exit 3 with **no delta quoted** and name
  only the differing performance-projection keys; `--allow-cross-fingerprint` downgrades the
  output to a **range** that prints the ±20% band and says plainly whether the difference
  clears it. Two genuinely legacy bare metric JSONs still compare with a warning; mixing one
  legacy arm with one attested arm refuses.

### 7.5 Validation landmines

| Landmine | Symptom | Handling |
|---|---|---|
| Spec-decode poisons PPL | `/v1/completions` echo+logprobs returns the **draft** model's NLL under `--speculative-config` | detected and refused (§7.2); run PPL on a no-spec serve |
| Gemma / instruct BOS | raw PPL ≈ ln(vocab) garbage when BOS is dropped | use KL-vs-BF16 (`/home/rob/dq-runs/kl_tool.py`); raw PPL cannot separate quantizations of instruct models anyway |
| Activation CPU-residency | tensors from `_LazyActivationCache.get()` are CPU-resident; the matmul silently runs on CPU — no error, no speedup | `.to(device, float32)` explicitly in every batched/sweep path; recurs across export work |
| In-sample "validation" | selection KL measured on text the surrogates saw | `--calib-skip-first $NSAMPLES` (`run-pipeline.sh:1194-1219`); an audit found this had regressed once already |
| Metric-era mixing | old harness records measured on wikitext **train** | check `eval_split`/`metric_era` (`validation_harness.py:147-152`) before comparing |
| Tied embeddings (`tie_word_embeddings`) | the cost stage died on the `lm_head` shard with `NotImplementedError: Cannot copy out of meta tensor` — the checkpoint ships no `lm_head` tensor at all, so the head is a meta alias of `embed_tokens` | `prismaquant/tied_embeddings.py` (landed `d058267`). The head is **materialized** — phase-2's CE backward runs through it, so meta is never acceptable — via transformers' own `get_output_embeddings()`/`get_input_embeddings()`, and **excluded from probe/cost/DP**: a tie means one Parameter, so quantizing the head quantizes the embedding, and probe/cost measure only the head's *output* MSE while the identical perturbation enters every token embedding and thus layer 0 for the whole forward — a cost no surrogate, not even L2 perturbed-X, can observe. There is also nothing to re-encode (no `lm_head.weight` bytes), so `footprint` would either fail to resolve the name or subtract the embedding from the floor while it still ships verbatim. Detection = config declaration AND a source index with no head tensor, never a name guess; a meta head with no declared tie raises immediately. The allocator exclusion (`allocator.py:1010-1043`, called `:1465`) also covers probes built before the fix. It ignores `--allow-pinned lm_head` by design — the tie is a property of the checkpoint, not of the serving profile. Gemma4-31B completed probe → cost → allocate → export for the first time on this fix (**enablement, not a quality claim** — unserved, no KL/PPL) |
| KV-sharing layers (`num_kv_shared_layers > 0`) | phase-3 forwards each layer in isolation and handed the consumer a **detached** K/V, so the storing layer's `k_proj`/`v_proj` Fisher never saw any consumer's contribution — and phase-3 chains each layer's input gradient downward, so the truncation was inherited by every layer *below* the producer too | The KV-cotangent path (`b6ec9cb`): consumers get grad-enabled leaf clones whose `.grad` is the cotangent they contribute, accumulated per storing layer and used to seed that layer's backward alongside its own output cotangent, in one reverse pass (`sensitivity_probe.py:1269-1299`, `:3185-3222`; `incremental_probe.py:1943-2409`). Verified by **exact equivalence** on an fp64 synthetic model — h_trace bit-identical to one end-to-end autograd backward (rel err 0.00e+00) — where the pre-fix protocol under-counts `k_proj` 85.1% and `v_proj` 38.5%. Guard semantics were **inverted, not deleted**: `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` no longer gates KV-sharing models generally; the probe hard-errors only when the path is turned *off* (`PRISMAQUANT_KV_COTANGENT=0`) on a model that needs it, and `PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` still reproduces a pre-fix probe (`incremental_probe.py:1035-1060`). Models without KV sharing are bit-for-bit unaffected either way. **Honest limit:** no real `num_kv_shared_layers > 0` checkpoint has been probed; those percentages are a toy correctness demonstration, not a quality claim |

## 8. Model support: the plugin architecture

Adding an architecture is a registration exercise, not a fork. Three registries hold everything
a model needs; the allocator, solver, caches, exporter and `pipeline.py` contain zero
architecture conditionals. Re-verified 2026-07-30 by AST scan: string literals naming an
architecture that reach **control flow** (a comparison, `startswith`/`endswith`, a dict lookup)
anywhere under `prismaquant/` outside `model_profiles/` and `vendored/` numbered exactly
**three** — the MiniMax hardcodes of §8.5 L4 — and are now **zero**: R27 routed both through
profile accessors (`bypass_hf_fp8_module_rewrite()`,
`packed_expert_module_class_names()`), declared in `specs/minimax_m2.json`.
Gridbook is a separate repository and therefore outside this AST scan. Its supported producer
profiles and serving aliases are imported as data from the installed package's
`runtime_contract.json`; PrismaQuant carries no copy of those tables. An earlier, laxer count
("5 and 2") could not be reproduced and is withdrawn; the
remaining arch-named literals in the core stack are argparse help, log/error text, and
`vendored/`'s registration machinery, which is arch-specific by design — the cosmetic list at
the end of §8.5 is the audited set.

**DIAGRAM-3 — Plugin registries:** the three registries plus the gridbook per-arch loader
chain, what auto-derives from the vLLM class, and the four places production bypasses a
declared extension point.

```mermaid
flowchart TD
  subgraph R1["registry 1 -- model structure"]
    VLLMCLS["vLLM model class<br/>packed_modules_mapping, hf_to_vllm_mapper"]
    DERIVE["auto-derivation -- model_profiles/vllm_registry.py:25-195<br/>fused_sibling_group, fused_sibling_leaf_mapping,<br/>to_vllm_internal_name (prefix mappers only)"]
    SPEC["structure spec JSON<br/>model_profiles/specs/ARCH.json<br/>schema prismaquant.model_structure.v1<br/>match, priority, naming (+ naming_variants), fused_groups, packed_experts,<br/>pinned_names, passthrough_prefixes, default_serving_profile,<br/>supported_lanes / preferred_lane"]
    PROF["ModelProfile subclass -- model_profiles/ARCH.py<br/>only matches() and name are abstract (base.py:103-112)<br/>Python-only: MTP, streaming adapters, forward state"]
    REGY["model_profiles/registry.py _REGISTERED + detection_order()<br/>ordered by ModelProfile.priority (lower first, ties keep list order);<br/>SpecMatchProfile per unclaimed spec; DefaultProfile terminal fallback"]
  end

  VLLMCLS --> DERIVE
  DERIVE -->|"tier 1"| PROF
  SPEC -->|"tier 2"| PROF
  SPEC -->|"match + priority"| REGY
  PROF --> REGY

  CONSUMERS["consumers -- ~30 detect_profile call sites across 22 modules<br/>probe, cost, cache, allocator, exporters, validators"]
  REGY --> CONSUMERS

  subgraph R2["registry 2 -- serving profiles"]
    SPROF["serving_profile_specs/ID.json<br/>research, vllm_packed_moe, gguf, nvfp4_cb<br/>allow/deny formats, shape rules, runtime validators"]
    RESOLVE["resolve_target_profile -- serving_profiles.py:611-633<br/>explicit request wins first (:623-624)"]
  end

  SPEC -->|"default_serving_profile"| RESOLVE
  SPROF --> RESOLVE
  RESOLVE --> ALLOCGATE["allocator candidate legality<br/>allocator_candidates.py + allocator.py:1661"]

  subgraph R3["registry 3 -- pipeline contract"]
    PIPE["pipeline.py -- declarative, not executive<br/>APPROVED_RESOURCE_OWNERS (:19-26), 14 artifacts, 9 stages<br/>validation is tautological in the production path"]
  end
  CONSUMERS --> PIPE

  subgraph GB["external Gridbook repository -- sole serving/runtime owner"]
    GBPLUG["packaged runtime_contract.json<br/>consumer aliases, format ABI, supported producer profiles"]
    GBSCAN["Gridbook-owned architecture loader registry<br/>version-robust, inert for non-CB checkpoints"]
    GBINST["Gridbook-owned top-level expert loader + fill guard<br/>missing coverage fails closed before execution"]
    GBCFG["Gridbook-owned quantization config<br/>native CUDA/CUTLASS-only, fail-closed<br/>CB / ignore / delegated stock CT / embedding / routed experts"]
  end

  GBPLUG --> GBSCAN
  GBSCAN --> GBINST
  GBPLUG --> GBINST
  GBINST --> GBCFG

  L1["LEAK 1 -- run-pipeline.sh:91<br/>TARGET_PROFILE hardcoded to vllm_packed_moe and passed<br/>unconditionally (:471, :1081); spec.default_serving_profile<br/>can never win. hy_v3 declares gguf, laguna declares nvfp4_cb.<br/>MEASURED 2026-07-11: 226 dense FP8 Linears silently -> BF16<br/>on the Hy3 CT export. PRISMAQUANT_TARGET_PROFILE is the audit<br/>escape hatch and run-pipeline.sh does not set it."]
  L2["LEAK 2 -- FIXED 2026-07-30 (R12)<br/>MTP now routed through profile.build_mtp_module /<br/>read_mtp_source_state_dict / load_mtp_state_dict at all three sites.<br/>mtp_module.py deleted; DSv4 takes the hy_v3 passthrough route."]
  L3["LEAK 3 -- FIXED + OWNERSHIP MOVED 2026-08-01<br/>Gridbook alone owns loader wiring and the fill guard; missing coverage<br/>raises before execution. PrismaQuant carries no loader table and CI compares<br/>its eligible producer profiles with the exact pinned consumer contract."]
  L4["LEAK 4 -- FIXED 2026-07-30 (R27)<br/>streaming_model FP8-rewrite bypass -> profile.bypass_hf_fp8_module_rewrite()<br/>(spec staging.bypass_hf_fp8_module_rewrite); incremental_probe expert<br/>container -> profile.packed_expert_module_class_names().<br/>Zero arch literals in core-stack control flow."]

  L1 -.->|"leak"| RESOLVE
  L2 -.->|"leak"| PROF
  L3 -.->|"leak"| GBPLUG
  L4 -.->|"was leak"| CONSUMERS

  classDef leak stroke:#c0392b,stroke-width:2px
  class L1,L3 leak
```

### 8.1 The three registries

| Registry | Where | Holds |
|---|---|---|
| Model structure | `model_profiles/<arch>.py` (`ModelProfile` subclass) + `model_profiles/specs/<name>.json` (`ModelStructureSpec`, schema `prismaquant.model_structure.v1`, `structure.py:20`) | detection (`match`, `priority`), naming across five name spaces, routed-expert packed/unpacked layout and format groups, `concat_merges` (N→1 source concatenations, below), pinned/passthrough names, staging, shard regexes, probe skips, `default_serving_profile`, `supported_lanes`/`preferred_lane` |
| Serving constraints | `serving_profiles.py` + `serving_profile_specs/<id>.json` (schema `prismaquant.serving_profile.v1`) | per-format allow/deny rules with name conditions, shape rules, runtime shape validators, runtime package requirements; `extends` composition (`serving_profiles.py:557-609`) |
| Pipeline contract | `pipeline.py` | almost nothing — `target_profile` as a kwarg (`:644`), run metadata (`:688`), CLI passthrough (`:1115`, `:1151`), one `model.structure_graph` stage spec (`:877-884`). Zero architecture names, which is correct: the contract layer should not know models (§3.6) |

Detection is **priority-ordered, not list-ordered** (R8, 2026-07-30). Subset profiles must
still precede supersets — `Qwen3_5DenseProfile` before `Qwen3_5Profile` — but that used to be
encoded in `_REGISTERED`'s literal order plus comments. Original Qwen3 dense and routed-MoE
now share the contract-aligned `Qwen3Profile`. Priority is a class `int` (**lower is consulted first**, like a sort
rank), declared both on the Python class and in its spec, so the ordering survives the Python
body being deleted. Built-ins take 100–210 in the historical order; `ModelProfile.priority`
defaults to **0**, which is what keeps `register_profile`'s documented insert-at-front override
true for third parties. `detect_profile` keys on `config.json` `model_type` + `architectures`
and dispatches through `_resolve`, which walks `detection_order()`; unmatched models fall to
`DefaultProfile(architectures=archs)`. `tests/test_spec_match_profile.py` asserts that priority
order still reproduces the list literal exactly.

`detection_order()` folds in a second kind of candidate: a **`SpecMatchProfile`**
(`model_profiles/spec_profile.py`) per `specs/<id>.json` whose `id` no registered Python
profile claims, matched by its declarative `match` block. All eleven shipped specs are claimed
by a Python profile, so today the live order contains none — landing the reader changed
detection for exactly zero shipped models, which is the point (see §8.3 Tier A).
`qwen4_exp` and `glm5_next` were registered on 2026-08-26; while their specs sat unclaimed,
`detection_order()` folded in two `SpecMatchProfile`s and
`tests/test_spec_match_profile.py::test_python_profile_wins_over_a_same_named_spec` was red.

**`concat_merges` — N→1 source concatenations.** A checkpoint may store one live parameter as
several separate tensors that the modelling code concatenates on load (transformers'
`Concatenate(dim=…)` conversion entries). `checkpoint_to_live_name` is a 1:1-or-drop contract
and cannot express a merge, so it passes the sources through unchanged and the merge is
declared in the spec — `{"target", "sources", "dim"}`, suffix-matched like `fused_groups`,
source order load-bearing (`structure.py` `ConcatMergeSpec`, accessor
`ModelProfile.concat_merge_groups()`). The loader bridge `layer_streaming._merge_concat_sources`
executes it, wired by `_build_concat_merger` into `_read_layer_to_device` on the streaming
probe/cost path (`streaming_model.py`) and the exporter (`export_native_compressed.py`), so
every path loads the same bytes. It is the sibling of `_pack_per_expert_into_packed`, and
carries no architecture names. The merge is cast-free: mixed source dtypes are refused rather
than promoted, and the assembled shape is checked against the live parameter. This is
deliberately **not** `fused_groups`: a fused group is a claim about what a serving runtime
fuses (principle 14), while a concat merge is a fact about the checkpoint's own layout.
Only `glm5_next` declares one today — its KDA `self_attn.{q,k,v}_conv1d.weight` →
`self_attn.conv1d.weight` (34 layers × 3 BF16 `[8192,1,4]` sources → `[24576,1,4]`); unbridged,
the short convolution would have run on uninitialised weights. Gate:
`tests/test_conv1d_concat_bridge.py`. The exporter carries the **emit-side inverse**
(2026-08-27): the merged live key never existed in the source checkpoint, so the 3e
layer-passthrough skips any concat-merge target and copies the SOURCE tensors verbatim
from the shards under their checkpoint keys — shipping the merged spelling would hand the
serving runtime's loader a key it does not know while dropping the three it expects. Gate:
`tests/test_glm53_stock_campaign.py::test_glm5_next_concat_merge_source_keys_roundtrip`.

**Multimodal-forced export skeleton (2026-08-27).** `materialize_tensors_streaming`
historically hardcoded `stage_text_only` + `AutoModelForCausalLM` — unresolvable for a
family with no `<Arch>ForCausalLM` auto-route. It now mirrors `_build_streaming_context`'s
flip: `profile.requires_multimodal_skeleton()` → `stage_multimodal` +
`_skeleton_config_and_class(multimodal=True)`, with `_build_weight_map(multimodal=…)` to
match, and the visual tower stays on meta (shipped via the source-passthrough merge, never
the body walk). Two companion fixes ride with it: the tensor-sink rename composes
`live_to_recipe_name` before `export_tensor_name` (body-walk keys are LIVE; the composition
is an idempotent no-op for every text-only-skeleton family), and
`fill_packed_expert_cache_entries` probes its compute device from the first **non-meta**
parameter — `next(model.parameters())` returned a meta visual weight on the wrapper and
silently `.to(meta)`'d the activation snapshot (the same fragile probe still exists in
`aura_cost.py:1158`, `kl_measurement.py:1078`, `validate_assignments_kl.py:987/:1579`,
`validation_harness.py:63`; audit before running those stages on a partially-meta skeleton).

**glm5_next streamed-forward wiring (2026-08-26).** The concat bridge was necessary but not
sufficient: `glm5_next` is also the first architecture here whose decoder stack runs on
`hc_mult` **parallel residual streams** (manifold-constrained hyper-connections) and whose
attention schedule declares `deepseek_sparse_attention`. Two existing seams carry it, with no
new contract:

* `Glm5NextProfile.expand_hidden_for_layers` / `collapse_hidden_after_layers` mirror
  `Glm5NextTextModel.forward` — expand `[B,T,H] → [B,T,hc_mult,H]` before the loop, collapse
  with the model's own `hc_head` after it (the DSv4 precedent for the same pair of hooks).
* `layer_streaming._compute_attention_mask` gained a `deepseek_sparse_attention` entry: those
  layers consume a **2D boolean padding mask** `[B,S]` — the DSA indexer applies causality and
  padding itself — and it is never optional, so an all-ones mask is substituted when the
  recurrent helper yields `None`, exactly as upstream does "to guarantee the mask to exist for
  the indexer". `linear_attention` keeps the existing recurrent shim.

Neither is checkable by inspection: a wrong mask is a plausible tensor and a wrong collapse is
a plausible mean. Gate: `tests/test_glm5_next_streamed_forward_parity.py` compares
`StreamedCausalLM` logits against `Glm5NextForConditionalGeneration` on a tiny random config,
on CPU. It runs the comparison at **1024 tokens as well as 12**, because the short cases cannot
see this architecture's failure modes: the KDA scan is *chunked* with recurrent state crossing
chunk boundaries, the short convolution carries a 4-wide window across the same boundaries, and
the DSA indexer only begins sparsifying past `index_topk`. None of the three is exercised at 12
tokens — an injected mask defect that blinds the past beyond position 512 leaves the 12-token
logits **bit-identical** and moves the 1024-token logits by 1.15. The long case is also run with
each layer type in isolation, so a future failure names its suspect (KDA state, conv window, or
DSA long-range masking) rather than just reporting disagreement. Two related facts the streamed
loop depends on and asserts rather than assumes:
`glm5_next` is NoPE (`qk_rope_head_dim = 0`, no standalone rotary, upstream passes
`position_embeddings=None`), and cross-layer top-k sharing is **not** threaded — correct only
while every DSA layer declares `indexer_types == "full"`, which the GLM-5.3-Flash checkpoint
does and the forward-fidelity gate re-checks per checkpoint. transformers 5.16 ships no
`Glm5NextForCausalLM`, so the streamed skeleton is built through the **multimodal** path;
the profile declares this via `requires_multimodal_skeleton()` and
`_build_streaming_context` flips itself, so probe/cost/validation call sites need no
per-family threading (`multimodal=True` explicit still works). The path yields the `model.language_model` base
prefix and root `lm_head` the spec already describes.

`_resolve` also **refuses to hand back a profile whose vendored-modelling override is known
dead** (`_refuse_dead_vendored_override`, added by #19 / `29f3cff`). Its `except Exception:
pass` around `register_vendored_modeling()` is right for keeping *detection* alive, but the old
comment assumed "the eventual model load error" would surface a failure — true only for a
failure that raises. The failure it actually hid is the opposite: `register_qwen3()` returned
cleanly on transformers ≥ 5.13.0 and did nothing, after which the probe ran **upstream** Qwen3
modelling code — on the family behind most shipped artifacts — with no exception anywhere. Root
cause is upstream: `_LazyAutoMapping.register` returns early when the config key's `__module__`
starts with `transformers.`, so no override of a natively-supported `model_type` can land that
way. The fix registers a PrismaQuant-owned subclass of the native config through
`AutoConfig.register` (public API, no internals patched), engages only when the direct route is
verified dead, and verifies every registration by a config-only resolution before setting the
"done" flag. Boundary measured, not assumed: healthy through 5.12.1, broken from 5.13.0.

The `DefaultProfile` fallback is *guarded, not silent*: `allocator.py:1550-1554` calls
`validate_default_profile_format_menu(...)` (`:961-988`), which refuses a multi-format menu
under `DefaultProfile` unless `--allow-default-profile`, on the grounds that fused-sibling
coherence and packed-expert uniformity (§6.4) cannot be enforced without arch knowledge.

### 8.2 Resolution precedence and vLLM auto-derivation

Every `ModelProfile` accessor resolves in one fixed order:

```
vLLM class metadata  →  declarative JSON spec  →  generic hardcoded default
```

Only `matches()` (`base.py:103-107`) and `name` (`:109-112`) are abstract — and `matches()` is now
also spec-expressible, via `SpecMatchProfile` (§8.1). The `match` vocabulary is deliberately
tiny, because all nine in-tree predicates are expressible as `(model_type ∈ set, architecture
glob)` tests:

| key | form | why it exists |
|---|---|---|
| `model_type` | exact strings | the common case |
| `architectures` | `fnmatch` globs (a bare class name is a valid exact glob) | exact Qwen3 dense/MoE entrypoints and Qwen3.5/3.6 family prefixes |
| `architectures_exclude` | globs; any hit **vetoes** the whole match | `qwen3_5_dense`'s `not any("Moe" in arch)` |
| `priority` (top level) | int, lower first | replaces the comment-encoded `_REGISTERED` order |

Unknown keys raise at parse time. That matters: `match` was declared in all nine spec files and
parsed since day one with **no reader**, and `qwen3_5_dense.json` had silently drifted out of
agreement with its Python (it was missing the Moe exclusion) — dead config decays.
`tests/test_spec_match_profile.py` is the standing gate: for every registered profile, on every
representative config in the family, the spec verdict must equal the Python verdict. Only after
that is green for a release does a `matches()` body get deleted, one architecture at a time.

What `base.py` reads off the vLLM class named by `vllm_architecture_class()` (`:114-120`, resolved
lazily at `:76-84`, `None` permitted):

| Derived | vLLM attribute | base.py | Spec fallback |
|---|---|---|---|
| `fused_sibling_group()` | `packed_modules_mapping` | `:89-118` | `spec.fused_groups` `:110-115` |
| `fused_sibling_leaf_mapping()` | `packed_modules_mapping` | `:120-164` | same |
| `to_vllm_internal_name()` | `hf_to_vllm_mapper.orig_to_new_prefix` | `:290-319` | `spec.recipe_to_vllm` rules take **precedence** `:314-318` |

**One profile, one spec, more than one serving class.** A family can ship a multimodal wrapper
and a text-only carve-out that share every structural rule *except* the namespace vLLM builds
them under — Qwen3.5/3.6 dense is the case in tree. `Qwen3_5ForConditionalGeneration` puts the
body at `language_model.model.` and the head at `language_model.lm_head`;
`Qwen3_5ForCausalLM` puts them at `model.` and a bare `lm_head`, because its
`hf_to_vllm_mapper` **strips** `model.language_model.` rather than adding a prefix. A single
`naming` block cannot be right for both, and the wrong one is only observable by serving: the
Qwen3.8-27B CB artifact emitted its one delegated `config_groups` target as
`re:^language_model[.]lm_head$`, matched no module, left the head unquantized and died at load
on an orphaned `lm_head.weight_scale` — one wrong string in 12.98 GB.

So the spec may declare **`naming_variants`** (`structure.py`, `NamingVariant`): a list of
`{when: <SpecMatch>, naming: {…}}` entries whose first `when`-match overrides only the naming
maps it names, the rest inheriting the base block. Selection input is what the checkpoint
declared: `registry._resolve` hands every profile its `model_type`/`architectures` via
`ModelProfile.declare_config`, and `structure_spec()` applies `spec.for_config(...)` **only**
when the spec declares variants — every other spec is returned untouched, and a profile
constructed by hand declares nothing and keeps its historical derivation. Both
`qwen3_5_dense.json` and `qwen3_5.json` now carry variants. A profile that specializes naming
must also specialize `vllm_architecture_class()` (`qwen3_5_dense.py`, `qwen3_5.py`), or the two
sources of the map disagree and a build host with vLLM installed would emit different names
than one without.

The MoE case needs one more distinction: serving namespace and source namespace are independent
facts. An official native-causal Qwen3.5-family checkpoint uses `model.layers.*` in its index
and vLLM tree, while the multimodal wrapper uses `model.language_model.layers.*` on disk and
`language_model.model.layers.*` in vLLM. Internal `stage_text_only` can legitimately declare the
causal serving class while retaining a symlinked wrapper-named source index. Path-based
`detect_profile` therefore records the checkpoint root as private intake evidence;
`Qwen3_5Profile._checkpoint_source_layout` censuses the actual index for source lookup while the
exact declared architecture continues to select the vLLM class. Mixed, absent, or unknown
namespace evidence fails closed rather than inheriting the wrapper default
(`model_profiles/registry.py:detect_profile`; `model_profiles/qwen3_5.py`). This lands namespace
intake for the future MoE PrismaSnap gate; it does not implement that gate or assert the layout
of an unreleased checkpoint.

Offline function-preserving folds also need the stored RMSNorm parameter convention, which is
not inferable safely from values. `ModelProfile.rms_norm_parameter_offset()` therefore defaults
to `None`; a transform must refuse that profile unless it explicitly returns `0.0` for literal
gamma or the architecture's declared offset. Qwen3.5/Qwen3.8 returns `1.0`, matching execution
with effective `gamma = 1 + weight`, so its materialized update is
`weight' = (weight + 1)d - 1` (`model_profiles/base.py`, `qwen3_5.py`;
`prismasnap_checkpoint._discover_dense_layer_graph`). This accessor is source-algebra metadata,
not a quantization or serving default.

The adapter is `model_profiles/vllm_registry.py`: `vllm_class_for_architecture` (`:25-102`)
tries four registry APIs plus internal-table fallbacks and degrades to `None` when vLLM is
absent. It consumes **prefix-substitution mappers only** (`:123-125`) — regex/substring mappers
are skipped, which is why LFM2.5 (`lfm2_moe.py:115-141`), MiniMax (`minimax_m2.py:110-131`) and
HyV3 (`hy_v3.py:75-89`) still hand-override `to_vllm_internal_name`. Spec `regex` rewrite rules
can now express those; `lfm2_moe.json` already does.

Roughly 25 further accessors are pure spec reads (packed-expert names/classes, pinned names,
unpacked expert projection names, per-expert regexes, source/recipe/live name mapping, format groups, passthrough prefixes,
staging, layer prefixes, lm_head, probe skips, export-lane eligibility,
`bypass_hf_fp8_module_rewrite`), `base.py:208-859`. Deliberately Python-only,
because they are forward-pass *behaviour* rather than naming: MTP (`:248-272`),
streaming-probe adapters (`:823-947` — `checkpoint_to_live_name`, `fp8_scale_pairs`,
`head_resident_extra_prefixes`, `init_rotaries`, `expand_hidden_for_layers`,
`extra_layer_kwargs`, …), cross-layer forward state for Gemma4 KV sharing (`:949+`, which the
KV-cotangent path now grafts through — §7.5), `register_vendored_modeling()` (`:974-979`).
`vllm_fused_moe_scheme_projection_names` (`:443-468`) is intentionally hardcoded to vLLM's
canonical names — §6.2.

Routed-expert classification for the AURA hybrid is also a profile boundary, not a shape
heuristic. `routed_experts.py` treats `packed_expert_format_group(qname)` as the membership
answer, validates it against the packed/unpacked/vLLM projection accessors, and maps live,
recipe, and vLLM names before deciding. `deepseek_v4.json` therefore declares live unpacked
`gate_proj` / `up_proj` / `down_proj` explicitly; its vendored probe topology is per-expert
Linears even though Gridbook serves their profile-declared virtual packed parents. Core cost
code contains no DSv4 architecture literal or rank-based expert predicate.

**Two plugin-contract additions landed on this branch.**

`ModelProfile.probe_linear_exclude_extra()` (`base.py:247`, default `""`) makes the probe's
Linear-exclusion regex **profile-owned**. `incremental_probe.resolve_linear_exclude()`
(`:423-437`) ORs the profile's fragment into the router baseline and replaced four literal
regex sites, so hook installs and the shard-reuse meta stamp (`:830`) can no longer disagree —
a mismatch there silently invalidates shard reuse. `DeepseekV4Profile` overrides it
(`deepseek_v4.py:115`) to exclude `self_attn.{compressor,indexer}`. The reason is a contract
fact, not a preference: the faithful vendored forward (`87ca027`) instantiates and loads the
compressor and indexer, so their `nn.Linear` leaves became visible to the probe's enumeration,
but they sit **outside the gridbook D0.1 serving contract's quantizable set** — served
source-format, charged to the immutable floor — and on this FP8-source checkpoint BF16 is
masked model-wide. An inventory row for them therefore carries **zero legal candidates** and
trips the allocator's coverage refusal *after* the cost run has already been paid for. The
override restores the 33,325-selectable-Linear inventory the DSv4 byte accounting assumes
(`deepseek_v4.py:6`, `:127`; commit `d62bace`; `tests/test_probe_linear_exclude.py`).

`ModelProfile.init_rotaries` gained an optional `base_model` kwarg (`base.py:1131`, commit
`9cee20d`) — a profile-plugin **signature** change, so the in-tree overrides moved in step
(`gemma4.py:60`, `deepseek_v4.py:339`). It exists because the DSv4 faithful forward gives every
compressor and indexer its **own** `rotary_emb`, and a meta-built skeleton leaves those nested
`inv_freq` buffers on meta — "Cannot copy out of meta tensor" at the first CSA forward. The
DSV4 override now walks the skeleton from `base_model` and materializes every nested rotary,
not just the model-level one; the caller passes it at `streaming_model.py:217`.

`structure.py`'s `build_model_graph` (five parallel name spaces per tensor) is a declared
contract, not an executor — `base.py:1268-1278`, "intentionally not called from hot paths yet";
production reads the accessors.

### 8.3 Adding a model, end-to-end, as it stands today

**Tier A — pure JSON.** Now *possible*, still never done. The obstacle was `matches()` being
abstract; `SpecMatchProfile` (§8.1–§8.2) removes it, so a spec file with a `match` block, a
`priority`, and declared `fused_groups`/`naming` resolves on its own with no Python. Tier A does
**not** get vLLM tier-1 auto-derivation — that is keyed on a Python
`vllm_architecture_class()` — so a spec-only architecture must declare its fused groups
outright. Every one of the nine shipped specs is still claimed by a Python profile, which is
deliberate: the R8 mitigation lands the reader alongside the Python and deletes `matches()`
bodies one architecture at a time, only after a release of green equivalence.

**Tier B — the realistic minimum (5 items).** (1) `model_profiles/<arch>.py` — subclass with
`matches()`, `name`, `vllm_architecture_class()` (may return `None`); 34–172 LoC in practice.
(2) `model_profiles/specs/<name>.json` — the declarative contract (§8.1). (3)
`registry.py:46-57` — import + one line, **in the right order**. (4) Serving profile — reuse
`vllm_packed_moe`, or add `serving_profile_specs/<id>.json` (`extends` supported). (5)
`TARGET_PROFILE=<id>` on the run invocation — leak L1 means the spec field alone does not take
effect.

**Tier C — commonly also needed.** MTP is profile-owned: implement `has_mtp()`,
`build_mtp_module(text_config)`, `mtp_source_prefix()`, and, only when the checkpoint layout
needs it, the source-reader/loader overrides. Probe, cost, native production cache, and export
all use that one contract; `prismaquant/mtp_module.py` no longer exists (L2). Architectures
whose MTP remains source-passthrough use the hy_v3 route — `has_mtp → False` plus
`passthrough_prefixes` and out-of-band CB encoding scripts. Common remaining work is streaming
overrides (`checkpoint_to_live_name` for flat naming, `init_rotaries` for multi-layer-type
rope, `head_resident_extra_prefixes`), cross-layer forward state, and vendored modeling. Then
run the conformance validator (§8.6), which nothing else does, and the discovery walker (§8.8)
on a meta load — an unclaimed matmul-fed parameter at intake is a `wo_a` waiting to ship.

**Tier D — the gridbook CB lane (§9.2) adds per-arch work.** (6) `default_serving_profile:
"nvfp4_cb"` in the spec **and** `TARGET_PROFILE=nvfp4_cb` (gated `run-pipeline.sh:124-125`),
plus `"nvfp4_cb"` in the spec's `supported_lanes` — the lane declaration is what
`require_lane_supported` (`serving_profiles.py`) checks, and it must be added *with* the loader
wiring of (7), never ahead of it.
(7) **Read the architecture's vLLM `load_weights`** and implement any required
top-level expert-loader hook in the external Gridbook repository. Gridbook owns
that registry and its fail-closed fill guard; its packaged consumer contract is
the only architecture-capability list PrismaQuant checks. Do not copy module
paths here. (8) A CB-quantized MTP/drafter needs corresponding Gridbook loader
coverage and a contract update before the PrismaQuant lane declaration lands.

Serving-side registry keys are owned by the pinned Gridbook consumer contract:
canonical `"gridbook"`, with `"prismaquant"` retained only as a read alias for
artifacts exported before the rename.

### 8.4 Conformance matrix

| Arch | profile | prio | structure spec | `default_serving_profile` | `supported_lanes` (preferred) | gridbook opt-in | MTP |
|---|---|---|---|---|---|---|---|
| qwen3 (dense + routed MoE; smoke: Qwen3-30B-A3B) | `qwen3.py` | 120 | ✅ | `vllm_packed_moe` | **CT, tessera** (CT) | `Qwen3MoeForCausalLM` uses the generic per-layer FusedMoE loader. Its `nvfp4_cb` declaration was removed 2026-09-02 with the lane (§9.2). **`tessera` declared 2026-09-03** and it is the only profile that declares it: Tessera's exporter stacks exactly this profile's two fused groups one-blob-per-vLLM-module, and the 2026-09-02 allocated-vs-uniform serve built Qwen3-0.6B Tessera checkpoints from a PrismaQuant allocation and served them on the plugin. The MoE architecture this profile also claims is refused by the lane preflight, which reads the checkpoint's own expert count against the contract's `structures: ["dense"]` (§9.4) | none |
| qwen3_5 / 3.6 MoE | `qwen3_5.py` | 110 | ✅ | `vllm_packed_moe` | **CT** (CT) | `nvfp4_cb` declaration removed 2026-09-02 with the lane (§9.2) | `build_mtp_module` → `MtpModule` (live; R12) |
| qwen3_5_dense | `qwen3_5_dense.py` | 100 | ✅ | `vllm_packed_moe` | **CT** (CT) | no expert-loader hook; `nvfp4_cb` declaration removed 2026-09-02 with the lane (§9.2) | inherits `Qwen3_5Profile.build_mtp_module` (dead copy removed, R12) |
| gemma4 | `gemma4.py` | 140 | ✅ | `vllm_packed_moe` | CT | ⚠ none | none |
| lfm2_moe (LFM2.5) | `lfm2_moe.py` | 150 | ✅ | `vllm_packed_moe` | CT | ⚠ none | `has_mtp → False` |
| minimax_m2 | `minimax_m2.py` | 160 | ✅ **added R22** — all 8 overrides declared | `vllm_packed_moe` **(added R22)** | CT | ⚠ none | `has_mtp → False` |
| deepseek_v4 | `deepseek_v4.py` | 170 | ✅ | `vllm_packed_moe` **(added R22)** | **CT** (CT) | `nvfp4_cb` declaration removed 2026-09-02 with the lane (§9.2). Until then it was declared by the exact Gridbook producer pin and carried a streaming CB exporter, W8A16 source passthrough, a top-level loader and a routed per-role LUT ABI; all of that is archived at `archive/gridbook_lane_2026-09-02/`, and the DSpark hybrid sidecar it fed is unreachable from here | `has_mtp → False`; three source-quantized DSpark stages are declared by the header-validated physical→construction overlay (§6.3), with no tensor rewrite. The experimental hybrid sidecar emits 27 physical K12 CB targets plus four physical W8A16 bases (`main_proj` and three grouped-BMM `wo_a`) with exact construction declarations. It remains non-shipping until the separately pinned Gridbook 0.8.6/v4 consumer is immutable and the sidecar clears load, memory, acceptance, and paired endpoint/throughput-evidence gates. |
| hy_v3 | `hy_v3.py` | 180 | ✅ | `gguf` (overridden, L1) | CT, **gguf** (gguf) | `nvfp4_cb` declaration removed 2026-09-02 with the lane (§9.2) | `has_mtp → False`; MTP passthrough (its out-of-band CB scripts went with the lane) |
| laguna (poolside S/XS 2.x) | `laguna.py` | 190 | ✅ | spec declares none (2026-09-02) | **CT** (CT) | 2026-09-02: this profile preferred `nvfp4_cb` and pinned it as its `default_serving_profile`; both went with the lane (§9.2), so it now derives its profile like every other. Drafter still separate | `has_mtp → False` |
| qwen4_exp (Qwen3.8-Flash-Next 177B) | `qwen4_exp.py` | 200 | ✅ | ⚠ **spec declares none** | ⚠ **spec declares none** → accessor default CT | ⚠ none | `has_mtp → False`; `mtp_source_prefix "mtp."` + `mtp.` in `passthrough_prefixes` |
| glm5_next (GLM-5.3-Flash 314B) | `glm5_next.py` | 210 | ✅ | `vllm_glm5_next_packed_moe` (extends `vllm_packed_moe`) | CT (CT) | ⚠ none | `has_mtp → False`; body-indexed nextn at layer 45, passthrough (hy_v3 route) |
| default | `default.py` | — (terminal) | n/a by design | — | CT (default) | n/a | none |

`prio` = detection priority, lower first (§8.1); the same number is declared on the Python class
and in the spec, and a test asserts they agree. **CT** = `compressed-tensors`. The lane column is
the *declared* set (R6, spec `supported_lanes`/`preferred_lane`), and required CI compares the
six CB producer profiles with Gridbook's packaged contract; GGUF has one. Over-declaring is the exact
failure the field exists to prevent: an undeclared lane does not fail loudly, it serves
uninitialised expert memory. `require_lane_supported(profile, EXPORT_CONTAINER)`
(`serving_profiles.py`) runs in `run-pipeline.sh` before profile resolution and export.

The two 2026-08-26 additions are **enablement scaffolds, not export-ready lanes**. Neither has
a vLLM class importable from the pinned serving stacks, so both return `vllm_architecture_class
→ None`. `qwen4_exp` still declares `fused_groups` read from HF modelling code (`in_proj_qkvz`,
`in_proj_ba`, the shared-expert `gate_up_proj`), the hy_v3 precedent. `glm5_next` declares
**none**, and has nothing it may honestly declare: its KDA attention exposes separate q/k/v
Linears live, and the only fusion evidence is the checkpoint's own
`quantization_config.modules_to_not_convert` naming `self_attn.qkv_proj` /
`self_attn.fused_qkvbfg_a_proj`, names that appear in no index key — a lead about someone's
serving tree, not a contract (principle 14). The gap is harmless only because every fusable
attention projection is pinned, so no fused group can be split across formats; that pairing is
ratcheted in `tests/test_model_profile_conformance.py`
(`UNATTESTED_FUSED_SOURCE_XFAIL`), which goes red if either half changes.
`qwen4_exp` declares no `default_serving_profile` and
no `supported_lanes` **on purpose** — each is a statement about what a serving runtime does, and
principle 14 refuses an unattested one; they are owed once a class exists to attest them, and
until then `require_lane_supported` refuses the arch rather than guessing. `glm5_next` declares
both, and its large `pinned_names` list is a principle-9 gate input recorded against vLLM PR
#53906 (`ZJY0516/vllm@933876c`), which builds only routed experts, shared experts and dense
MLPs with a `quant_config`; that citation must be re-attested against a pinned release before
export. Its `default_serving_profile` is its own
`serving_profile_specs/vllm_glm5_next_packed_moe.json`, which `extends`
`vllm_packed_moe` and adds one rule denying every quantized format outside
`mlp.{experts,shared_experts}` and the dense `mlp.{gate,up,down}_proj` — the
same PR-#53906 fact as the pins, re-stated where the serving profile can act on
it (defense in depth, principle 9). Its declared lane set is
**`["compressed-tensors"]` only**: the CB lane was withdrawn on 2026-08-29
(`58eb69d`) because the pinned Gridbook 0.8.11 contract's
`producer_profiles.supported_ids` is
`[deepseek_v4, hy_v3, laguna, qwen3, qwen3_5, qwen3_5_dense]` and `glm5_next` is
absent from it in every release through 0.9.0 — the over-declaration this column
exists to prevent, which does not crash but serves uninitialised expert memory.
The exact re-enable condition (advance the pin to a release whose `supported_ids`
contains `glm5_next`, in the same commit that re-adds the lane — never by
widening the test) is recorded in the spec's `_supported_lanes_rationale`
(`specs/glm5_next.json:6`). Nothing shipped is affected: the 2026-08-27
GLM-5.3 campaign served through compressed-tensors.
The one contradiction found in recon is recorded in the spec
(`_verified_source_layout.serving_restriction.CONTRADICTION_to_resolve_before_export`): the PR
comments that MLA projections are BF16, while the checkpoint ships them FP8 with
`.weight_scale_inv`.

Gaps beyond the four leaks. **minimax_m2's missing spec is closed** (R22, 2026-07-30): all
eight overrides (`:69,:86,:91,:101,:104,:110,:133,:137`) are now declared in
`specs/minimax_m2.json` — `fused_groups`, `packed_experts` (+`projection_splits`,
`format_groups`), `moe.per_expert_regex`, and three `naming.recipe_to_vllm` regex rules for the
`block_sparse_moe.experts.N.w{1,2,3}` → `mlp.experts.N.{gate,down,up}_proj` rename. The Python
overrides stay for now; `tests/test_minimax_m2_spec.py` compares a spec-only profile against
the pre-spec Python behaviour accessor by accessor, and that gate must hold for a release
before the Python comes out. It closed a latent bug on the way: without a spec,
`packed_expert_format_group` fell through to the legacy fallback, whose first group is
`(gate_up_proj, down_proj)`, so MiniMax's `down_proj` got a *different* coupling key than its
`gate_proj`/`up_proj` — one expert bank in two format groups, which violates §6.4 and would
have been unservable. **`deepseek_v4.json` now declares `default_serving_profile:
vllm_packed_moe`** (R22) — the conservative, provably-tighter choice for both its native and
Gridbook lanes; `research` carries no format allow-list at all. Its dense
`fused_groups` remain empty deliberately: Gridbook's constructed merged Linear
owns independent role decoders and can consume a different codebook format or
physical activation scalar per role before the common FP8 execution path. A
compressed-tensors-only uniformity rule must therefore live in that lane rather
than globally coupling the producer assignment. The
spec did gain `_verified_source_layout` (`2b5b937`, closing #26): the real
DeepSeek-V4-Flash-Base headers say routed experts are I8 nibble-packed MXFP4 with F8_E8M0
scales while **shared experts are block-FP8 E4M3, not fp4** — settled against the checkpoint
and the authors' `convert.py`, not inferred. **`serving_profile_specs/vllm_qwen3_5_packed_moe.json`
is an empty `extends: [vllm_packed_moe]` alias whose own description says not to use it** — but it
is NOT unreferenced: `.github/scripts/check_installed.py`, `tests/test_allocator_packed_group_units.py`,
and `tests/test_serving_profiles.py` all name it, so deletion means retiring those references and
checking shipped artifact metadata first.
**Mistral-Medium-3.5-128B is in the shipped family table (§1.2) with no profile** — no Mistral
profile class or spec exists (the sole textual mention is a comment at
`model_profiles/default.py:6`), so it ran under
`DefaultProfile`; the `allocator.py:1550-1554` gate would refuse that menu today. Finally, the
never-declared `unpacked_expert_projection_names` (`base.py:651-661`) **is now declared** — by
`specs/minimax_m2.json` (R27), which also required adding it as a real
`ModelStructureSpec` field; `base.py` had been reading it off the spec with `getattr`, so no
declaration could ever have taken effect. Other architectures still ride the `('w1','w2','w3')`
default, which is correct for them.

### 8.5 Known contract leaks

These four are the canonical statement; §12 references them rather than restating them. **All four are now FIXED** (L1/R11, L2/R12, L3/R10, L4/R27, all 2026-07-30) and are kept in the table with their fix, so each leak and its resolution stay in one place.

| # | Leak | Severity |
|---|---|---|
| L1 | **FIXED 2026-07-30 (R11).** Was: `run-pipeline.sh` hardcoded `TARGET_PROFILE:=vllm_packed_moe` and passed it unconditionally, and `resolve_target_profile` gives an explicit request precedence (`serving_profiles.py`), so `spec.default_serving_profile` was **never consulted through the production orchestrator** — `hy_v3.json` (`gguf`) and `laguna.json` (`nvfp4_cb`) silently overridden. **The leak had a measured cost:** because export re-resolved the profile it judges legality under, on 2026-07-11 **226 dense FP8 Linears were silently demoted to BF16** on the Hy3 compressed-tensors export. **Mechanism of the fix:** (i) the shell default is now empty and `--target-profile` is passed to the allocator **only when non-empty**, with a new `--target-profile-default vllm_packed_moe` supplying the fallback for architectures that declare nothing — never `research`, whose menu is unbounded. (ii) The allocator stamps its **resolved** profile into `layer_config.json`'s reserved `__prismaquant__` metadata block (`layer_config.LAYER_CONFIG_META_KEY`, skipped by every assignment parser and by the schema), and `export_native_compressed._allocator_target_profile_for_audit` reads it, with `PRISMAQUANT_TARGET_PROFILE` kept as the operator override for direct exporter invocations. `select_validated_frontier` carries the block forward when it overwrites the layer config, so the validated path keeps it too. Allocator and export can no longer disagree, and the channel travels **with** the artifact. **Non-regression:** re-solving the shipped 27B and 35B from their stored probe/cost artifacts changed **0 of 614** and **0 of 500** assignments vs the same code without the change (the 35B differs from its *shipped* config by 32/500 for an unrelated, pre-existing reason — the Fisher renormalization fix that landed after that artifact shipped). Every in-tree launch script sets `TARGET_PROFILE` explicitly, so all eight are bit-identical. | ~~high~~ FIXED |
| L2 | **FIXED 2026-07-30 (R12).** MTP construction bypassed the profile: `prismaquant/mtp_module.py` was Qwen3.5-specific yet imported **directly** by `incremental_probe.py`, `incremental_measure_quant_cost.py` and `export_native_compressed.py`, gated only on the arch-agnostic `profile.has_mtp()`, so `deepseek_v4` (`has_mtp → True`, `build_mtp_module → None`) would have been handed a Qwen3.5 decoder layer. **Mechanism of the fix:** a fourth accessor `ModelProfile.mtp_source_prefix()` (`base.py:294-311`, spec-expressible as `shard_regexes.mtp_source_prefix`, default `"mtp."`) plus a generic `read_mtp_source_state_dict()` (`:348-365`) and a packed-expert-aware `load_mtp_state_dict()` (`:387-435`, absorbed from the deleted `_load_into_mtp`); `build_mtp_module`'s docstring now states the naming contract (names under an `mtp` parent must equal the recipe names). The Qwen body moved verbatim into `model_profiles/qwen3_5.py:124` (`MtpModule`) and the dead near-copies in `qwen3_5.py` and `qwen3_5_dense.py` were reconciled into it; all three call sites now go through the profile and hard-fail with a named error if `has_mtp()` and `build_mtp_module()` disagree. `prismaquant/mtp_module.py` is **deleted**. DSv4 takes the hy_v3 route (`has_mtp → False` + `"mtp."` in `passthrough_prefixes`) until its nextn block is actually quantized. Gates: `tests/test_mtp_module_arch.py` pins parameter-name-set equality against the pre-move layout for both the dense and MoE profile; `tests/test_model_profile_conformance.py::test_has_mtp_implies_a_buildable_mtp_module` is the standing ratchet. | ~~high~~ FIXED |
| L3 | **FIXED 2026-07-30 (R10), ownership boundary hardened 2026-08-01.** Was: a hand-maintained `try/except ImportError` opt-in chain whose missing-line failure mode was **coherent-looking garbage generation**. Gridbook now owns the module-path registry, fill guard, and tests in its sole canonical repository. PrismaQuant owns no loader or runtime copy; its required CI job installs the exact pinned Gridbook commit, validates PEP 610 provenance, and compares the producer profile set with Gridbook's packaged `runtime_contract.json`. The runtime stamps CB expert parameters unfilled, stamps them after either loader path, and fails closed before execution if any local registered expert remains unfilled. **No env bypass.** | ~~high~~ closed |
| L4 | **FIXED 2026-07-30 (R27).** Both MiniMax hardcodes now go through profile accessors. `streaming_model.py`'s FP8-rewrite bypass was already half config-derived (`quant_method == "fp8"` and `weight_block_size`); the architecture half is a static property, so it became `staging.bypass_hf_fp8_module_rewrite` in the spec behind `profile.bypass_hf_fp8_module_rewrite()` (`base.py`), leaving the per-checkpoint half a config read where it belongs. `incremental_probe.py`'s `type(module).__name__ == "MiniMaxM2Experts"` became `profile.packed_expert_module_class_names()` (`base.py:221-231`) — the accessor that already existed for exactly this lookup — plus the structural shape test; the declared class stays **required**, because the replacement forward implements one specific expert-loop signature and applying it to a lookalike container would silently change a forward pass. `specs/minimax_m2.json` declares both, and `unpacked_expert_projection_names` with them. | closed |

Cosmetic, listed so they are not re-discovered as leaks:
`export_native_compressed.py:94,151-152` imports `Qwen3_5Profile` for `_COMPAT_QWEN_PROFILE`
(verified test-only back-compat); `_fast_kernel_guard.py:86-90`'s Qwen substring list is a
labelled fallback for remote HF IDs with no local `config.json`; `layer_streaming.py:1914-1920`
imports an upstream transformers Gemma3 masking helper under config-driven selection;
`gridbook/config.py:174-194` shared-prefix aliasing is HunYuan-motivated but written
structurally.

### 8.6 The conformance validator

`python -m prismaquant.model_profiles.validate --model <path>` implements 8 conformance checks
(docstring `validate.py:17-53`): profile claim `:136`, vLLM class resolvable `:153`,
fused-sibling self-consistency against vLLM's own sibling lists `:191`, name-remap fixed points
`:219`, MTP module construction `:246`, source-passthrough prefixes matching ≥1 real tensor
`:270`, packed-expert param names `:355`, serving profile exists and its validator callables
import `:462`. Exit 0/1, CI-shaped.

**It now has callers** (2026-07-30). `tests/test_model_profile_conformance.py` runs the
CPU-safe part over every registered profile — checks 1, 6 (against synthetic index fixtures)
and 8, plus four structural invariants (spec presence, fused-sibling source, registry order,
name uniqueness); the vLLM-registry checks 2/3/4 sit behind an `integration` marker (their
answer is vLLM-version-dependent) and the real-checkpoint index checks 6/7 behind `slow`.
Check 5 (MTP) is deliberately absent: `build_mtp_module()` materialises a full decoder layer,
a multi-GB CPU allocation — use the manual CLI for it. Its cheap declarative half IS automated
since 2026-07-30 (R12): `test_has_mtp_implies_a_buildable_mtp_module` fails any profile that
answers `has_mtp()` without a real `build_mtp_module` override or `mtp_source_prefix()`, which
is the L2/D2 defect class. The old no-spec xfail ratchet is empty. The fused-
source check carries one named, passing exception: DeepSeek still returns
`None` from `vllm_architecture_class()`, and its Gridbook lane is role-composite
rather than uniform-format. Direct profile coverage asserts the spec stays
empty so a native-lane assumption cannot silently constrain the Gridbook
allocation.
And there is CI to run it — `.github/workflows/ci.yml` (#18, `1cc7b90`) executes the suite on
every push and PR, on Python 3.12 with CPU torch. Before PrismaQuant is
installed, both jobs use the stdlib-only `tools/resolve_tessera_dev_pin.py` to
derive the exact Tessera checkout from `TESSERA_DEV_PIN_COMMIT`, stop
without publishing a ref if that resolution fails, and install that checkout
before exercising the package. Ordinary checkout access is sufficient once
Tessera is public; while it remains private pending fixes and audit, checkout
failure is expected. §12 D11.

### 8.7 A fourth plug-in point: `FormatCostPlugin` (formats, not models)

The three registries above are the **model-support** contract and stay three.
`format_cost_protocol.FormatCostPlugin` (`:284-300`) is a different axis: it is
what a **format** must implement to be priced, and it exists only on the
Sensitivity Card path (§4.8) — no production stage consumes it today.

The whole protocol is one attribute and one method: a `FormatDescriptor`
(storage bits, group size, passthrough requirement, the explicit
`quantizes_activations` predicate, an optional `speed_index`) and
`weight_error(unit, weight) -> [out, in]` squared error computed **from the
weight alone** under the card's declared render basis. That is the point of the
seam: adding a format is adding a plugin, not re-probing the model and not
re-rendering a menu cache. The consumer already has the weights; what they
cannot compute is the sensitivity, and that is `O(out + in)` in the card.

Two properties keep it honest rather than merely convenient:

- **Reference plugins call the real quantizers.** `format_cost_registry.RegistryFormatPlugin`
  (`:78-160`) invokes `FormatSpec.quantize_dequantize` and
  `activation_quantize_dequantize` directly, so the error a plugin reports is
  produced by the same code that renders (§5.1). A plugin that re-implemented a
  format's rounding would reintroduce precisely the rendering confound the
  one-cache rule exists to prevent (§5.4). Passthrough formats return exactly
  zero error rather than a measured epsilon, so float noise cannot decide a
  passthrough-vs-quantized comparison; and `FormatDescriptor.is_legal_for`
  (`:129-135`) refuses a passthrough on a mismatched source dtype, which is the
  §6.4 passthrough-integrity invariant restated on this path.
- **`act_quant_changes_input` has one definition.** The descriptor carries the
  registry's predicate as explicit data (`descriptor_for:65`,
  `format_registry.py:75-106`) rather than re-deriving it from an `act_bits`
  width; consumers that re-derived it disagreed with the allocator's own gate
  (§4.5, bit-exact re-encode), and
  `tests/test_bit_exact_cost_pricing.py::test_activation_quant_predicate_has_one_definition`
  pins the single definition. This is the
  entire difference between `NVFP4` (W4A4) and `NVFP4A16` — two registry entries
  with identical weight bits — which is what makes them a clean A-side isolation
  menu.

### 8.8 The discovery walker: requirement discovery by traversal

`prismaquant/model_walk.py` (2026-08-21, campaign R5; normative frame
`docs/design/model_coverage_ledgers.md`) discovers every weight-bearing
computation in a model by traversal instead of by the pipeline's own
enumeration. The failure class it kills is structural: when the probe
enumerates the units, then cost, allocation, bpp, and coverage are all computed
over that same enumeration, an omission is invisible by construction. DSv4's
`attn.wo_a` — a parameter consumed by a grouped bmm on a module class the probe
skips, 17.9% of decode read traffic — shipped as passthrough by omission while
every coverage statistic read complete.

The walk has one root pair and one output (`WalkResult`, JSON-serializable):

- **Root A — the module tree.** Every named parameter and buffer becomes a
  `WalkNode` (`remove_duplicate=False`, so tied weights keep all names).
- **Root B — one traced forward.** A `TorchFunctionMode` interceptor
  (`WeightUseInterceptor`) records every matmul-family call (`F.linear`,
  `matmul`, `bmm`, `mm`, `mv`, `addmm`/`addbmm`/`addmv`/`baddbmm`, `einsum`,
  `tensordot`, `@`) with the parameters that feed it, resolved by **storage
  identity** so a view or per-expert slice maps to its parent parameter. The
  trace runs under `FakeTensorMode` on a meta-loaded model — intake costs no
  GPU and no weight I/O — and the interceptor is host-mode agnostic:
  `execution="real"` runs the same capture over a real CPU forward, the
  fallback for forwards fake tensors cannot execute (measured: DSv4's
  `int(position_ids[0, 0])` raises `DataDependentOutputException`; the ratchet
  is `tests/test_model_walk.py::test_dsv4_fake_trace_block_is_still_real`).
  `F.scaled_dot_product_attention` carries no weights and is excluded;
  `F.embedding` produces no edge but its weight still requires a claim.

Every discovered node must be **claimed** by exactly one disposition —
`decide` (the allocator's domain), `pin(reason)` (held at source precision on
purpose), or `exclude(reason)` (outside the artifact's scope). Claims come
from ordered `ClaimRule` lists supplied by the profile:
`ModelProfile.walk_claim_rules()` (`model_profiles/base.py`) derives the base
set from the profile's own declarations — spec
`probe_skip_module_class_names` → pin (the mechanism that held DSv4's
grouped-BMM `attn.wo_a` as a named debt until it could be priced;
2026-08-22 the grouped Fisher accumulator landed, `DeepseekV4GroupedLinear`
moved to the spec's `probe_grouped_module_class_names`, and its weight
became an ordinary `decide` — see §8.9),
`pinned_names` → pin, MTP/visual
prefixes and `nn.Embedding` weights and non-persistent buffers and
non-floating or ≤1-D tensors → exclude with reasons, remaining `nn.Linear`
weights → decide. **A matmul-fed node no rule matches fails the walk** with
the node named and the op cited (einsum equations included). Reasons are
first-class output for the shipcard, not log prose.

Two implementation facts are load-bearing and measured (torch 2.11):
`data_ptr()` is 0 for every meta storage, so identity keys on the
`StorageImpl` address; and under fake mode a view of a non-fake parameter gets
fresh fake storage, so the interceptor propagates identity through a fixed
alias/cast allowlist at `__torch_function__` level (recorded per edge as
`via`). Every tensor whose storage key enters the resolution maps is kept
alive for the trace, so a freed address can never be reused and
misattributed. An unresolved floating multiplicand — a weight the walk cannot
name — is itself a failure, never a guess.

First real yield: walking the actual DSv4 modeling code surfaced **three more
unclaimed matmul-fed families beyond `wo_a`** — the MoE router gates
(`DeepseekV4TopKRouter`/`DeepseekV4HashRouter.weight`), the mHC
hyper-connection mixers (`attn_hc.fn`/`ffn_hc.fn`), and the hyper head
(`hc_head.hc_fn`) — all bare Parameters fed to `F.linear` on classes outside
the probe's enumeration. `DeepseekV4Profile.walk_claim_rules()` now pins each
with its reason (plus the compressor/indexer Linears the serve contract keeps
source-format).

The gate half of the contract is WIRED (2026-08-22, `walker/r5-export-gate`):
`run-pipeline.sh` stage **[3d]** runs `python3 -m prismaquant.model_walk
--model $MODEL_PATH` on a meta load immediately before EVERY export lane and
refuses the run (`exit 2`, `set -e`) when :func:`evaluate_walk_gate` — the
gate over :class:`WalkResult` in the same module, schema
`prismaquant.model_walk_gate.v1` — reports an unclaimed matmul-fed node, an
unresolved floating multiplicand, an unknown walk-failure kind, or a
decided-but-unpriced node. The verdict is STRUCTURED: refusal kinds plus
per-entry `(node, op, equation, module)` lists in
`artifacts/model_walk.json`; prose (`detail`, `refusal_reason`) explains to
humans and nothing branches on it. The per-run override
(`PRISMAQUANT_WALK_GATE_OVERRIDE=<reason>`) excuses trace incompleteness only
— DSv4's DSA scalar aborts the fake trace today — and is stamped into the
report; claim failures have no override. Because identity and dispositions
live on whole logical tensors, Tensor-Parallel degree cannot move the gate's
universe; byte fields are pinned as logical totals behind the explicit
`per_device_bytes(total, tp_degree, policy)` seam, and any future walk-failure
kind (e.g. quantization-group vs shard-boundary misalignment) refuses on
today's gate via the unknown-kind catch-all. Walks are also cacheable now:
`save_walk`/`load_walk` wrap the payload in `{schema,
prismaquant.model_walk.v1, provenance, result}` with fail-closed reload
(foreign schema, execution mismatch, claim-rule digest mismatch) and
trace-time provenance (model+config digest, versions, input contract, applied
rule list). The same commit landed the R5 sweep's claim-table fixes: universal
base rules pin router gates (hy_v3, qwen3_5, laguna, minimax_m2, qwen3-moe;
DSv4 keeps its specific pins) and decide packed expert stacks (seven
`*Experts` families), and gemma4's router — decided by rule 9 while the
probe's own name exclusion made pricing impossible, i.e. wrong polarity
inside the claim table — is pinned like every other router; the gate's
`find_decided_but_unpriced` checker turns that contradiction class into a
refusal so it cannot recur silently.

Two distinct migrations are in play here, and only one has landed. The
first wave is the shared **name-projection layer** (below), and as of
2026-08-29 **all four consumers route their private NAME mappings through
it**: read-traffic (`walker/consumer-readtraffic`), footprint
(`walker/consumer-footprint`), probe (`walker/consumer-probe`) and cost
(`walker/consumer-cost`). No consumer holds its own checkpoint→live→recipe
string surgery any more. One deliberate exception remains and is not a
projection: `production_render_cost.canonical_cost_name`'s umbrella-infix half
stays a **total** normalizer, because render/cost payloads key costed MTP rows
physically and a projection that may decline must never drop them.

The **edge-list migration proper is still open for all four**, and is the
larger of the two: every consumer's INVENTORY remains its own enumeration —
the probe still builds its tracked set from `named_modules()` plus shard
regexes — so the walker's edge list is not yet the single enumeration the
stages derive from. Sharing a name projection is not the same as sharing a
requirement set; the first makes the four agree on what a unit is *called*,
the second would make them agree on which units *exist*.

#### 8.8.1 The shared name-projection layer

`prismaquant/name_projection.py` (2026-08-22, `walker/name-projection`)
lands the ONE projection between the pipeline's parameter-name namespaces
that the R5 consumer analyses each said they were re-deriving ad hoc: a
Linear's live name (`WalkNode.name`), its allocator/probe recipe name,
its source-checkpoint key, its exported-artifact key, and its vLLM
scheme-dispatch qname (`NameProjection`, five `LIVE/RECIPE/CHECKPOINT/
EXPORT/VLLM` constants). The contract, pinned by
`tests/test_name_projection.py`:

- **The profile owns the knowledge; the layer owns the discipline.**
  Every mapping routes through existing `ModelProfile` accessors
  (`checkpoint_to_live_name`, `live_to_recipe_name`,
  `to_vllm_internal_name`, `source_tensor_name`, `export_tensor_name`,
  `fused_sibling_group`, `packed_expert_format_group`) — no second
  mapping lives anywhere, and consumers keep no private
  `_recipe_name`/`_strip_weight` helpers (the leaf rule itself is
  `strip_weight_leaf` here).
- **Fail closed and loud.** An unmappable or ambiguous name raises
  `NameProjectionError` with structured fields — `code`
  (`unmapped_in_universe`, `ambiguous`, `no_universe_supplied`,
  `malformed_profile_result`, `profile_accessor_failed`,
  `unknown_namespace`, `unsupported_pair`, `uncovered_unit`),
  source/target namespace, and what was tried. Unlike
  `decision_units.fused_group_key`, nothing swallows a profile failure
  into an identity fallback. The profile's DECLARED drops (visual/MTP/
  scale keys) are data instead: `project()` returns a `ProjectedName`
  whose `outcome == declared_out_of_graph`, so read-traffic can classify
  `excluded_non_text_graph` by branching on a field.
- **Non-totality is in the type.** Fused siblings and packed experts
  surface as `ServingGroup(kind, key, members)` with packed precedence
  over fused for expert stacks; reverse lookups refuse group KEYS rather
  than returning an arbitrary member; the coverage index mirrors
  footprint's dual-entry manifest convention so one logical tensor
  covered by many split checkpoint spellings answers under either
  naming (`checkpoint_keys_for` / `require_checkpoint_span`, the
  coverage assertion footprint.md asked for).
- **Round-trip property tested where the profile's rules are total**:
  Qwen3 dense (identity), Qwen3.5 multimodal (umbrella-infix rewrite),
  DSv4 flat naming — plus the recorded gap the round trip SURFACED:
  DSv4's spec generates `hc_head.hc_*` source spellings while the real
  checkpoint stores flat `hc_head_*`, so that family's inverse declines;
  the layer reports it, it does not guess.
- **TP seam held**: names are logical, whole-tensor facts; no rank,
  shard index, or degree exists in any signature (a constructor-kwarg
  refusal test pins this). Byte accounting stays in
  `model_walk.per_device_bytes`.

Still open per the design doc: migrating the consumers onto the walker's
EDGE LIST. The first wave of that migration is the shared
**name-projection layer** (below), and as of 2026-08-29 all four
consumers route their NAME derivations through it — cost
(`walker/consumer-cost`), probe (`walker/consumer-probe`), footprint
(`walker/consumer-footprint`) and read-traffic
(`walker/consumer-readtraffic`). Their INVENTORIES are still their own
enumerations — the probe still builds its tracked set from
`named_modules()` plus shard regexes — so the edge-list migration
proper remains open for all four.

### 8.9 Pricing a grouped operand: the `wo_a` accumulator (2026-08-22)

DSv4's `attn.wo_a` — 33,554,432 parameters × 43 layers, 1.443 GB/token,
17.9% of decode read traffic — is consumed by a view + per-group
`torch.bmm` inside `DeepseekV4GroupedLinear` (`y[...,g,r] = Σ_d x[...,g,d]·W[g,r,d]`).
The class sat in the probe's skip list because the dense accumulator cannot
represent that consumption: its chunk_h comes out `[R, D]` against a `[G*R, D]`
plane (the `chunk_h * w.pow(2)` broadcast), and flattening groups into the
token axis destroys the per-channel marginals. The walker then held the weight
as a named pin — correct polarity, real debt.

The debt is closed by ONE grouped mechanism (`prismaquant/sensitivity_probe.py`,
shared by both probe backends):

- **Math.** The elementwise Fisher of a grouped operand is block-diagonal in
  `g`, so the exact per-token-summed empirical Fisher (audit M3's estimator,
  never sum-then-square) reduces over one batched matmul:
  `chunk_h[g,r,d] = Σ_t gy²[t,g,r]·x²[t,g,d]`. Every scalar and marginal —
  `h_trace`, `fisher_row[g*R+r]`, `fisher_col[d]`, `g_sq_sum`, `act_sq_sum`,
  `act_absmax`, `h_trace_per_group_raw[g]` — reduces the SAME fp32
  `chunk_h`, so `sum(fisher_row) == sum(fisher_col) == h_trace_raw` holds by
  construction, which is the card's wiring check.
- **Schema.** The unit stays the whole logical tensor in FLAT-PLANE
  coordinates: `out_features = G*R`, `in_features = D`, `n_params` counts all
  groups, and `num_groups` distinguishes it from a same-shape dense Linear —
  deliberately NOT `num_experts`, which `_shape_from_stats` and
  `_stats_indicates_packed_expert` read as a packed expert stack. TP note:
  identity/dispositions/byte totals are logical; a future rowwise shard cuts
  the plane's row axis and must not straddle a group boundary.
- **Normalization.** Unchanged and global: rows stay raw token sums until
  `finalize_fisher_stats` divides every row by the GLOBAL calibration token
  count. Grouped operands are attention output projections — every group sees
  every token — so the shared denominator is exact, not merely consistent;
  `n_tokens_seen` counts tokens, not token-group pairs.
- **Dispatch.** Explicit declaration only: spec field
  `probe.grouped_module_class_names` (conformance-tested like every other
  spec field). A declared class whose instance lacks `n_groups` fails fast;
  nothing heuristic falls through to the dense accumulator.
- **Cost cells.** They flow from probe keys with no new plumbing; the weight
  render is exact on the flat plane for every menu format because row-wise
  quantization never mixes rows. The joint-output-MSE screen ships honestly
  UNMEASURED (`output_mse_measured=False`) for grouped units in both cost
  paths: its `y = X @ W.T` model would score each group slice against all
  `G*R` outputs, inflating the output term ~G-fold with cross-group error no
  token sees.
- **Serving unchanged.** Candidacy is not assignment: the DSpark sidecar
  contract keeps all three `wo_a` bases on source-FP8 W8A16
  (`dspark_source_metadata.dspark_cb_expected_physical_targets`), CB export
  still refuses grouped-BMM semantics, and Gridbook's pinned contract declares
  no grouped structure lane. "Priced, kept on FP8_SOURCE" is now an honest
  allocator decision where silence used to be.

## 9. Serving lanes

Three artifact containers, one allocator (a fourth, the Gridbook codebook lane,
was retired 2026-09-02 — §9.2). `EXPORT_CONTAINER` picks the lane (§3.3) and
constrains the whole run: the GGUF lane hard-gates `PRODUCTION_CACHE=0` and a matching
`TARGET_PROFILE` (`exit 2`), and must pass the render-faithfulness assertion (§4.7) — its
exporter ships imatrix-weighted bytes, so the cost render must be weighted too. That is no
longer the same as requiring `COST_MODE=local`: the production cache can
render that family weighted (`--col-weights`), so a cached-menu render is admissible on
the lane, opt-in and non-default. Each lane's serve command, endpoint, gate set and KL
evaluator are declared in `prismaquant/lane_specs/*.json` (re-vet **R16**); the gates are
advisory and the shipcard is what refuses — at **publication**, via `tools/publish_artifact.py`
(§7.1).

**DIAGRAM-2 — Serving lanes (re-drawn 2026-09-03):** the three artifact containers the
`EXPORT_CONTAINER` lanes can act on and the runtime each requires (§1.1). The Gridbook
codebook container and its plugin, its RTX 4090 candidate and its sm_120
validation profile were removed with the lane (§9.2) and are gone from the
diagram; the Tessera container (§9.4) is drawn **fail-closed on the release pin**:
since 2026-09-03 `run-pipeline.sh` has a real `EXPORT_CONTAINER=tessera` arm that
plans and encodes through Tessera's own tools under `TESSERA_REPO` — which the lane
declares as `producer_tool`s rather than vendoring — and the pin's PENDING sentinels
are the only refusal left (`tessera_lane_attested` answers False for every rung
until a Tessera release tag exists). No served measurement covers the lane yet, so
its runtime edge stays dashed.

```mermaid
flowchart LR
  subgraph CONT["artifact containers"]
    A1["compressed-tensors<br/>maintained: NVFP4 / FP8_DYNAMIC / BF16<br/>FP8_SOURCE: source-artifact compatibility<br/>export_native_compressed.py"]
    A3["GGUF<br/>Q2_K..Q8_0 + IQ family<br/>export_gguf.py"]
    A4["Tessera wire<br/>quant_method = tessera<br/>planned + encoded via TESSERA_REPO tools<br/>fail-closed: PENDING release pin"]
  end

  subgraph RT["runtimes"]
    R1["vanilla vLLM<br/>no plugin, no forked runtime, no custom kernels<br/>CUTLASS NVFP4 path on Blackwell"]
    R3["llama.cpp"]
    R4["vLLM GGUF path<br/>in-tree up to vLLM 0.19; official vllm-gguf-plugin after"]
    R5["Tessera's own vLLM plugin<br/>tessera.serving, stock image + pin<br/>no served measurement yet"]
  end

  subgraph HW["hardware"]
    H1["NVIDIA GB10 DGX Spark<br/>Blackwell sm_121, 128 GB unified memory<br/>~121 GB usable serving budget"]
    H2["Strix Halo<br/>CANCELED / UNSUPPORTED<br/>prototype removed after hardware access was lost;<br/>no qualified backend"]
  end

  A1 -->|"serving profile vllm_packed_moe"| R1
  A3 -->|"serving profile gguf"| R3
  A3 -->|"serving profile gguf"| R4
  A4 -->|"lane spec tessera, fail-closed on PENDING pin"| R5

  R1 -->|"Spark-proven -- shipped rdtand artifacts"| H1
  R3 -->|"Spark-proven -- 295B-class at 2.8 bpp; the KL harness for this lane"| H1
  R4 -->|"smoke-verified on the 0.19.2 venv only, never KL-measured"| H1
  R5 -.->|"no admittable artifact until a Tessera release tag exists"| H1

  R3 -.->|"no qualified deployment"| H2
  R4 -.->|"no qualified deployment"| H2

  classDef proven stroke:#2d7a2d,stroke-width:2px
  classDef pending stroke:#c07800,stroke-width:2px,stroke-dasharray:4
  classDef unsupported stroke:#c0392b,stroke-width:2px,stroke-dasharray:4
  class H1 proven
  class A4,R5 pending
  class H2 unsupported
```

PrismaQuant paths below are repo-root-relative.

### 9.1 Native compressed-tensors — the default lane

`export_native_compressed.py` writes a stock checkpoint: no forked runtime, no plugin, no custom
kernel — the only lane whose correctness depends on nothing we maintain. All of §6 belongs to
it; §7 owns its gates. Validation runs in-process (`validate_native_export.py:171` constructs
`LLM(...)`), so it needs a venv or container carrying vLLM (§10).

### 9.2 codebook (CB) / gridbook — RETIRED 2026-09-02

Robert, 2026-09-02: *"put Tessera in PrismaQuant and remove Gridbook."* The
fourth container is gone from the producer. `EXPORT_CONTAINER=nvfp4_cb` now
fails with `exit 2` (§3.5), the producer and serving pins, the exporter
(`export_nvfp4_cb*.py`), the lane spec (`lane_specs/nvfp4_cb.json`), the four
CB serving profiles, the CB ship-gate slots and 73 test modules (1,691 node IDs) are archived whole at
`archive/gridbook_lane_2026-09-02/`, together with the lane's 27 documents
(`archive/gridbook_lane_2026-09-02/docs/lanes/nvfp4-cb/`) and every served
measurement it produced. Read that directory's `README.md` before proposing
anything CB-shaped: it records what the lane was, what it was worth, and what
capability went with it.

**The refusal names the wall.** `model_profiles.structure.RETIRED_EXPORT_LANES`
maps a retired lane to the archive directory its code went behind, and
`canonical_export_lane` appends that to `unknown export lane` — so a stale
driver or spec naming `nvfp4_cb` tells an operator where the code is instead of
reading as a typo. Retiring a lane moves one name from `EXPORT_LANES` to that
dict in a single edit, and
`tests/test_profile_export_lanes.py::test_a_lane_outside_the_vocabulary_is_refused`
asserts the property over both rosters rather than re-typing either: no lane is
both live and retired, a generated non-member is refused, and every retired
lane's refusal carries its wall.

Three consequences the removal actually has, stated once so no gate has to
infer them:

- **`FP8_BLOCK_UE8M0_SOURCE` is `ROUTE_STATUS_BLOCKED`.** Those bytes executed
  on the plugin's `Fp8SourceW8A16LinearMethod` and on nothing else, so the
  verdict changed because the runtime left, not because anyone reweighed it.
  Per principle 1 the rung stays **priced** — an allocator that wants it is
  reporting a serving gap — and per principle 9 export fails closed on it.
- **`MXFP4_SOURCE` keeps `ROUTE_STATUS_BACKED`** (stock vLLM Marlin MoE, which
  was never Gridbook's route) but has **no writer and no serving profile**: the
  `nvfp4_cb` container was its only exporter and the `nvfp4_cb` profile its
  only offer. It is an exporter gap, not a route gap, and the two are not
  conflated.
- **`lane_eligibility.py`** (renamed from `gridbook_lane_eligibility.py`) is the
  **generic** closed-world engine and stays: it is what admits Tessera (§9.4).
  Its Gridbook schema string, asset directory and contract index are gone, and
  `load_eligibility_table()` now demands an explicit `contract_path=` — with
  none, every unit resolves UNATTESTED and export fails closed.

The CB **format / cost / render plumbing** (`cb_layout.py`,
`nvfp4_cb_formats.py`, `nvfp4_cb_footprint.py`, `cb_ldlq*.py`, `cb_minchain.py`,
`cb_warm_state.py`, `cb_banked_books.py`, `cb_learned_promotion.py`,
`cb_anchored_cost.py`, `cb_ladder_cross_family.py`, `routed_moe_codebooks.py`,
`mxfp4_widen.py`, `source_class_format_plan.py`, and CB branches inside
`production_weight_cache.py`, `allocator.py`, `format_registry.py`,
`export_native_compressed.py`, `layer_config.py`, `lane_spec.py`,
`serve_constraints.py`, `model_profiles/*`) is **still in the tree** and is
recorded as debt D34 (§12). Removing the lane made those rungs unexportable and
unservable — which is the property that matters — but the code that prices and
renders them was not excised. Do not read its presence as a live lane.


### 9.3 GGUF

A single `.gguf` that llama.cpp serves natively and vLLM through the official
`vllm-gguf-plugin`. No custom kernels anywhere; the only lane reaching 2–3 bpw, where NVFP4 is
the compressed-tensors floor. Menu: k-quants Q2_K–Q6_K/Q8_0 plus the IQ family
(IQ2_XXS…IQ4_NL), all with `gguf-py dequantize(pack(w))` pinned **bit-identical** to the
registry emulation, so measured cost and shipped bytes cannot diverge (`docs/lanes/gguf.md`).
Container correctness is delegated: we requantize llama.cpp's own
`convert_hf_to_gguf --outtype bf16` skeleton and own only tensor bytes.

Three measured facts carry the lane. **imatrix is the dominant lever at ~3 bpw** — 0.6B KLD
2.728 → 0.913 from activation weighting alone, applied in lockstep to the batched cost path and
the exporter under one flag (`PRISMAQUANT_GGUF_IMATRIX`, default on, §3.3).
**GPTQ-into-k-quant** freezes the two-tier scales from the weighted search and re-decides only
`q` under full-Hessian OBS: 0.6B at matched 347 MB, KLD 0.890 / 56.9% top-1 vs llama.cpp's best
stack at 0.913 / 55.6% — the first arm to beat them on their own harness. **The 4B scale check
is honest about the gap**: byte-matched, the fully consistent stack lands at 0.510 vs their
0.461 (+10.6%) = ~+7.7% residual render (Hessian rank — 1024 activation rows is 10.5% rank at
4B) plus ~+2.6% allocation. Deep-bpw surrogate mis-ranking is the known regime failure;
validated-frontier selection is the house answer, and since re-vet **R16** the lane HAS its
evaluator — `gguf_kl_evaluator.measure_assignment_kl` wraps `llama-perplexity
--kl-divergence-base` behind the `validate_assignments_kl` interface and returns
`(mean, per_sequence, stats)` under the gold lane's key names, with `per_sequence` empty and
`kl_tail_domain="aggregate"` because llama.cpp reports token-domain quantiles. The frontier
loop is not wired to it yet, so `SELECTION_MODE=surrogate` is still what this lane runs
(§12 D26).

Shipped: `rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` — 103.686 GB at 2.799 bpp from the
prod `tencent/Hy3` base, measured allocation with IQ rungs displacing Q2_K/Q4_K entirely,
single Spark, vLLM smoke only (no quality claims). IQ vs k-quant at matched bytes: decode 17.8
vs 18.7 tok/s (−5%), TEB 87 vs 86 (churn at one plateau), and **prefill 42 tok/s is the whole
IQ tax** — k-quants have CUDA MMQ, IQ falls to MMVQ/Triton. That number is what the CB lane
exists to remove (§9.2). Open work: MoE expert stacking in cost/export;
**running** the ship gate on this lane — no analog is needed, `validate_quantized_model` is
endpoint-agnostic and llama-server speaks OpenAI, which is what `lane_specs/gguf.json` declares
(the pipeline smoke still proves load+generate only); embedding/head format as a measured
decision rather than operator policy. Strix Halo enters this lane first, serving-only
(re-vet R7): `docs/lanes/gguf.md` carries the dated tracking table.

### 9.4 tessera — the Tessera wire on Tessera's own vLLM plugin

**Declared, driveable, and fail-closed on one thing.** This lane existed as a
*bar* before it existed as a shipping path: `prismaquant/lane_specs/tessera.json`
states the serve, the endpoint, the gate set, the KL evaluator and the executed
activation contracts. Since 2026-09-03 `run-pipeline.sh` has a real
`EXPORT_CONTAINER=tessera` arm to go with it, and the refusal an operator now
meets is the release pin rather than `unknown export lane` from a vocabulary
check three layers up.

**The arm calls out; it does not vendor in.** `export_native_compressed.py` still
has no Tessera codec and is not getting one: the layer_config → plan translation
(`experiments/plan_from_layer_config.py`) and the encode
(`experiments/export_tessera_serving.py`) both live in the Tessera repository and
are NAMED by the arm under `TESSERA_REPO`, the same boundary the lane spec already
uses for the serve script and the route census. A second copy of either here would
be a second place a wire recipe can drift, which is the failure principle 14
exists to prevent. `TESSERA_PLAN_COVER` (`as-allocated` by default) decides whether
a partial allocation is planned as-is with every other body Linear spelled BF16, or
broadcast by role and stamped as the extrapolation it is; silence must never become
a 4-bit rung.

**Four preflight gates, each fail-closed on its own** (`tessera_export_lane.py`,
run before any GPU work): the checkpoint's structure must be one the packaged
contract declares (`require_declared_structure`, read from the artifact's own
`config.json`, because the `qwen3` profile claims both the dense and the MoE
architecture and only one is declared); the lane spec's
`served_activation_quantization.executes` must EQUAL what the contract's
`formats[]` rows imply (`require_executes_derived_from_contract`), principle 14
in the field that asserts what the runtime executes; every tool the lane
declares it shells out to must exist under the env var that declaration names
(`require_producer_tools`); and the pinned Tessera serving runtime must be an
exact reviewed release (`require_release_pin`). The last refuses every run
today and is the only thing that does.

The producer-tool gate reads `lane_specs/tessera.json`'s `producer_tools`. It
was a hardcoded `for` loop over two paths in `run-pipeline.sh`, which named two
files for one lane, was invisible to a reader of the lane spec, and would have
needed a fourth loop for a fourth lane. `TESSERA_REPO` is `export`ed by the
driver for the same reason: the gate resolves each tool through the env var its
own declaration names, so the value has to reach a child process
(`run-pipeline.sh:143-149`). `preflight()` returns `producer_tools`,
`unsupported_producer_tools`, `shipcard_slots` and `unrecorded_gates`, and the
CLI echoes the last three, so the debt is printed where it is being incurred.
Beside these the arm also checks, in the same up-front block, that
`TESSERA_SERVE_MODE` and `TESSERA_PLAN_COVER` each carry one of their two legal
values — not left to the translator's own `argparse` `choices`, which does not
run until stage 4, because the point of this block is to refuse before GPU
hours rather than after them.

**The runtime is Tessera's.** Package `tessera.serving` in the Tessera
repository: a `vllm.general_plugins` entry point
`tessera = "tessera.serving:register"` registering `quant_method = "tessera"`,
the routes and executed decoders published by its packaged contract. The
reviewed development contract includes NVFP4 W4A4 at
`e2m1_group16_ue4m3_static`, FP8 W8A8 at `fp8_per_token_dynamic` and BF16 at
`bf16_unquantized`; each route's cells name its residency and launches.
**There is no enable flag**: the checkpoint's
`quantization_config.quant_method` selects the plugin, and the single operator
knob is `TESSERA_SERVE_MODE=resident|streamed`, declared rather than defaulted
because it changes the artifact's footprint and is folded into vLLM's
compile-cache key. The serve installs the plugin into the *stock* vLLM image
(`pip install --no-deps --no-build-isolation -e <tessera>`); no core patch, no
forked runtime. The reference serve script
(`/home/rob/tessera/experiments/tessera_plugin_served.sh`) and the route census
(`/home/rob/tessera/tools/tessera_route_census.py`) live in that repository and
are NAMED by the lane spec, never vendored.

**The boundary is two objects, both machine-readable.** The pin,
`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`
(`prismaquant.tessera_serving_runtime_pin.v1`), read by
`tessera_serving_runtime_pin.py`; and the contract the plugin packages,
`tessera/serving/runtime_contract.json` (`tessera.runtime-contract.v1`, lane
table `tessera.lane-eligibility.v4`), read through `importlib.resources`.
PrismaQuant never vendors or imports the serving half. Unlike the Gridbook
serving pin this one binds no wheel digest: Tessera publishes no wheel and is
installed from a source checkout, so a digest would be a claim about an
artifact that does not exist.

**Every cell is plugin-gated, in a field a gate reads.** Each cell publishes
`requires_plugin: "tessera"` — stock vLLM has no reader for these bytes, so the
route is not merely flag-gated, and an export gate has to be able to refuse an
artifact whose serve command would not install the runtime. The shared parser
carries it as a required v4 cell key (optional only in legacy v3) and
aggregates it as `requires_plugins`
through `RegimeRoute` / `UnitRoute` / `EligibilityTable.provenance()`;
`tessera_lane_attested` RAISES on a cell that claims a native route without it,
because a contract defect must be loud rather than silently admitted.

**Scope, from the table and not from prose.** `structures: ["dense"]` and no
`routed_moe` cell — no served measurement covers routed experts, and absence is
the honest state under a closed-world v4 table, not a refusal. The published
`tensor_parallel` units declare `max_world_size: 1`: a Tessera unit is one blob
per vLLM module against a shared rate schedule, so a sharded form needs
per-rank wires rather than a byte range. `expert_parallel.units` is empty.
Both residency modes are receipted and both must be exercised.

**Admission is False until a release tag exists.** The pin's `commit` and
`version` are the sentinels `PENDING_TESSERA_RELEASE_COMMIT` /
`PENDING_TESSERA_RELEASE_VERSION` with `version_is_release: false`;
`require_exact_tessera_runtime_release` refuses them, and `tessera_lane_attested`
ANDs that refusal in (§5.7), as does the container arm's preflight. Cutting the
tag is Rob's decision and resolves the
pin file *and* the reader's two release constants in one reviewed commit —
neither half admits anything alone.

**The lane's gates are recorded on a card the publisher refuses on.** Gates are
ADVISORY to the build run on this lane like every other; the shipcard is what
refuses, at publication (§7.1). Until 2026-09-03 that sentence was not true
here, twice over. `route.census` — the only place this lane compares the route
it PRICED against the route it SERVED, principle 12's second leg — carried
`shipcard_slot: null`, so it was recorded nowhere and nothing could refuse on
it. And the arm called Tessera's exporter, which has no concept of a
PrismaQuant shipcard, then `exit 0`ed about 130 lines above the driver's
shipcard block, so no card existed to carry any of the six gates. Both are
closed: `route.census` names its slot, the derived vocabulary carries it
(`shipcard.lane_scoped_slots`, #162 closed),
and the arm runs `python -m prismaquant.lane_shipcard open --lane tessera
--artifact <exported>` before exiting (`run-pipeline.sh:2452-2458`), so an
un-run gate is an unfilled slot on a real card rather than a sentence in a JSON
file. A Tessera card owes **seven** slots, and they arrive from two
derivations that know nothing about each other and compose by union. The lane's
six declared gates close six slots (`lane_gate_slots("tessera")` —
`native_export.eager`, `native_export.graph`, `route.census`, `ship_gate`,
`gold.kl`, `gold.ppl`), five of which are the base `REQUIRED_SLOTS` the lane
happens to re-declare and one, `route.census`, that only this lane opens. The
seventh is `uniform_control`, which `required_slots` adds because the artifact
has a rate axis, not because any lane asked for it (#121, §7.1).
`open_lane_shipcard` stamps `export_container` into the card's build block so
that second obligation rests on the card as well as on the checkpoint's
`config.json`.

**What the card does not do is run anything.** Each of the six gates needs a
fresh vLLM container with the pinned plugin editable-installed, and both
residency modes, because the two decode the same bytes by different paths; the
build lane must not spawn those inside a pipeline run. Building that runner is
R16's open half and stays with RobTand/prismaquant#119. `verify` replays
`route.census` through its registered verifier (`_verify_route_census_record`):
the record must carry the priced-vs-served `route_census` block and its
verdict must agree with it, so the slot refuses a *wrong* census as well as
*silence* (#162).

**The lane declares its own architecture roster and its own build-tool
dependencies.** `wired_architectures` (`["qwen3"]`) is the set of model-profile
names permitted to export through the lane; it is REQUIRED and required to be
non-empty, and `["*"]` (`LaneSpec.ANY_ARCHITECTURE`) is how the default lane
says "every registered profile" without keeping a second copy of
`model_profiles/registry.py`. The roster lived in
`tests/test_profile_export_lanes.py` as two module-level sets named after two
specific lanes, which a third non-default lane would have escaped entirely
because the test asserted two lanes by name. It is now a property quantified
over `EXPORT_LANES`: a profile declares a lane exactly when that lane declares
the profile. `producer_tools` declares each external tool the arm shells out
to, with `repo_env`, `path`, `stability` ∈ {`supported`,
`unsupported_experiments`} and — mandatory for the second value —
`tracking_issue`. Both Tessera tools are `unsupported_experiments` today: they
live under that repository's `experiments/` with no stability promise, so a
tidy-up there breaks a shipping lane here, and that is stated on the lane's own
declaration rather than only in an issue tracker. Promotion rides the
release-tag commit (RobTand/tessera#17, tracked as
RobTand/prismaquant#119).

**The pin also carries the serve fingerprint's half of §7.4, and it is DERIVED.**
Tessera contract v7 publishes `native_extensions` — what the plugin *loads*, beside
the routes it *executes* — as three values a consumer can act on: the
`module_name_prefix` its JIT load path itself passes to `cpp_extension.load`
(`tessera_nvfp4_`), the `filename_glob` that produces (`tessera_nvfp4_*.so`; there
is no exact basename, the module name carries a build-identity hash), and `match`,
the name of the **rule** a gate applies (`basename_fnmatch` — fnmatch the glob
against the basename of a mapped `.so`). The chain is **contract → pin →
fingerprint, with a refusal at each link**:

* `tessera_runtime_contract` parses the table and **refuses a contract that does
  not publish it** — "does not say what it loads" is not "loads nothing" — refuses
  an empty table, and refuses a `match` rule it does not implement rather than
  approximating the predicate.
* `require_pin_native_extensions_match_contract` refuses a pin whose
  `serving_native_extensions` is not the pinned contract's table, in both
  directions, with a field-level diff. **Scope, stated:** `load_tessera_contract`
  calls it, so an allocation under the **dev pin** refuses on drift, and the test
  suite runs it against the installed contract on every run — but the *release*
  admission path (`tessera_lane_attested` → `_pinned_serving_table` →
  `lane_eligibility`) does not read this module at all, so on the day a tag is
  cut the pin↔contract link is test-enforced, not runtime-enforced. Wiring it
  into that path is a separate change. The table is part of
  `contract_answer`, so a library rename re-stales the reviewed answer instead of
  widening silently (`source`/`loaded_by` stay identity and travel into
  provenance).
* `tools/serve_fingerprint.py` is stdlib-only and runs inside the serving container
  from a bootstrapped snapshot with no installed package, so it can read neither
  the contract nor the pin's reader module — but the pin is JSON, so the tool reads
  the transported `tessera_serving_runtime_pin.json` beside itself (a member of its
  gold-producer source closure, hence digest-covered) and refuses a missing or
  malformed one instead of falling back to a constant;
  `tests/test_tessera_serve_fingerprint.py` refuses any disagreement with the pin,
  **and refuses a predicate that is not the rule the contract names**. The refusal
  is therefore in the container, not only in the test suite on the tree the
  snapshot came from: a snapshot of the tool from an older commit, beside a newer
  Tessera whose extension was renamed, refuses rather than fingerprinting "no lane
  extension resident".

Two holes this closed. Until 2026-09-03 no Tessera name was matched at all, so a
serve running Tessera's own native span-2 decode fingerprinted identically to a
stock serve — the one lane whose entire point is a custom decoder was the one lane
§7.4's identical-residency rule could not see. Then it was matched by
`re.escape("tessera_nvfp4")` anywhere in the mapped path, from a hand-written pin
that was already wrong by one character (`tessera_nvfp4` where the load path's
constant is `tessera_nvfp4_`) — a predicate that is not the runtime's, and that
answers yes for `/root/.cache/torch_extensions/tessera_nvfp4_9f2c/unrelated.so`.

**Open, and named:** the contract's `when_unavailable` block says a *resident*-mode
serve whose `.so` cannot build keeps serving on a named substitute decoder
(`torch_materialize_stock`) while a *streamed*-mode serve refuses. The fingerprint
observes `/proc/maps`, not configuration, so those two serves already hash
differently — but the manifest does not record *which* pinned library was expected
and what runs instead, so a differing fingerprint reads as "±20% alignment drift,
not evidence" when the honest reading is "this arm measured a substituted decoder
and says nothing about the lane" (#142, #143).
The residency-mode half of those items is closed since 2026-09-03 (#138): the
environment allowlist carries the pin's `serving_residency_env` name, so two serves
that both map the `.so` in different residency modes no longer hash identically —
only the substitution half above stays open.

## 10. Hardware & environment

The primary build hardware is now two NVIDIA GB10 / DGX Sparks, **Sparky** and **Sparklina**,
each Blackwell sm_121 with **128 GB unified memory** shared physically between CPU and GPU
(~121 GB usable). Sparky has the historically documented 1.8 TB NVMe; campaign manifests bind
each host's own absolute work root rather than assuming identical free space. Two consequences
that catch every newcomer: "move it to CPU to spare the GPU" is a **no-op** for memory pressure,
and a production run owns each assigned box — concurrent serves, heavy agents, or downloads
starve the launch-bound loop. Every production hot path must be GPU-bound;
`prismaquant/gpu_guard.py:7` (`require_cuda_hot_path`) refuses to run otherwise (though seven
stages never call it — §12 D9).

The current PrismaSnap candidate uses both Sparks for source preparation. The two
hosts are peers only for explicitly manifest-partitioned planning/materialization: neither may
infer work from the other's progress. Both plans must bind the same portable source, probe,
canonical search, producer source closure, and attested container-rootfs identity; the
coordinator then verifies exact layer/shard covers and all transferred content. The final
low-space merge runs where both part directories reside on one filesystem and requires
hardlinks. Those orchestration and transfer stages are offline lifecycle work; after the
verified BF16 checkpoint is committed, the ordinary production pipeline retains its existing
GPU-resident cache/prefetch contracts (§3.0; `prismaquant.cluster_campaign`,
`prismasnap_checkpoint.merge_checkpoint_parts`).

The strict Qwen3.8 FP8-CB campaign adds a second target, not a second qualified
machine: one physical 24 GiB RTX 4090/sm89. GB10 can exercise architecture-independent code and
learned-table determinism, and a no-device toolchain can cross-compile explicit sm89 SASS, but
neither emulates Ada execution, its 24 GiB memory boundary, vLLM graph capture, or served
performance. Those claims remain pending until the two-arm physical gate in §7 runs on the exact
GPU identity.

**OOM discipline.** The pool has no evictable slack, so an allocation that would merely swap on
a discrete-GPU box kills the machine instead. Rules, all learned from kills: serve at util
**0.90 or below** for spec-decode + compiled configs (0.94/0.95 died under long-prefill
activation spikes with a drafter resident, `prod_hy3_results.md`); arm the slack gate and
watchdog (§9.2); never bench a new kernel while a serve holds the pool. An idle serve is not
safe — one killed the box ~1.75 h after going quiet.

| environment | use | note |
|---|---|---|
| `/home/rob/dq-runs/venvs/prismaquant-cu130` | build / probe / cost / export / PPL | torch 2.11+cu130; `PYTHONPATH=.` for tests; the host `.venv` has no torch |
| `/home/rob/dq-runs/venvs/prismaquant-hy3` | Hy3 (`hy_v3`) chain | transformers 5.13; the cu130 venv lacks `hy_v3` |
| `/home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521` | vLLM 0.19.2 in-tree GGUF | the working local GGUF-serving venv |
| `vllm-node:latest` | all four CB serve scripts; the Hy3 GGUF stack | native HYV3; the only serving image the current scripts reference |
| `~/.cache/prismaquant-cb-ext` (or `PRISMAQUANT_CB_EXT_DIR`) | Gridbook JIT build cache | never `/tmp` (external Gridbook `gridbook/cuda_ext.py`) |

`transformers` pins are model-specific and have cost hours: MiniMax requires 4.57.5,
Qwen3.5/3.6 need ≥5.5 (4.57.5 raises `KeyError` on the model type). Older launchers and
`CLAUDE.md` name images (`vllm-fresh-b12x`, `vllm-node-tf5-cu132-lfm`) that are **not present on
the box today** — treat those references as historical.

**Disk.** Keep ≥10% of the 1.8 TB free (224 GB at time of writing). A 27B production cache is
~90 GB and a multi-arm matrix is bounded by peak, not final state: `df -h /home/rob` before
launching, build → measure → delete before the next arm. **Never write to `/tmp`** — an OOM
cleared it in 2026-04 and took the MiniMax artifacts with it. Set `TMPDIR` explicitly for any
tool reaching for `mkdtemp()`.

**Strix Halo / ROCm — CANCELED 2026-07-31; no supported backend.** Access to the only gfx1151
machine was lost before build ABI, dispatch, fallback, graph, wheel-install, vLLM, or served
quality gates could be completed. The prototype sources and dispatch hook were deleted from the
canonical Gridbook tree; PrismaQuant contains no copy. ROCm is therefore unsupported and must
fail through the ordinary absence of a qualified backend. Reintroduction requires new hardware,
hard architecture attestation, installed-wheel tests, and the full served promotion ladder.

The remainder of this subsection is a **frozen historical measurement record**, not an active
implementation description, support claim, or build plan. Paths named below belonged to the
now-deleted prototype.

**Historical Strix snapshot.** Robert funded kernel
authoring on 2026-07-30 ("build fp8 vllm kernels that target strix halo and support codebook
formats"), which **supersedes re-vet R7's serving-only framing** (R7 accepted GGUF-first with
"no ROCm build stack in scope" and left option B — a HIP port of the CB kernels — unfunded; the
outcome note in `docs/audits/architecture_re-vet_2026-07-30.md` §R7 records the change). A box
arrived the same day: AMD Ryzen AI MAX+ 395 / Radeon 8060S, **gfx1151**, RDNA 3.5, Fedora 44,
ROCm 7.1.1, torch 2.9.1 with HIP, **58 GB** unified memory (not 128 — size test shapes
accordingly), 919 GB free.

The deleted prototype's `csrc_hip/` held the result: a wave32 decode GEMV (fp8-CB **and**
NVFP4_CB two-tier v2), a bf16 **WMMA** decode-in-prologue prefill GEMM, the transient expander,
and the fused fp8 activation QDQ, sharing one format header with the CUDA lane. It **compiles,
runs, and passes parity at a 1-bf16-ULP gate against an fp64 torch reference** across the odd
rungs, every register-tile M boundary, ragged edge tiles and the fused multi-role codebook case.

**Benchmark discipline on this box is itself a finding:** gfx1151 idles at ~1.2 GHz and needs
~45 s of sustained load to reach ~2.6 GHz, so a conventional 5-iteration warmup measures the
idle clock. Every number taken that way was discarded and re-taken behind a `sustain_clock()`
that prints the achieved clock. The related correction matters more: there is **no large
bandwidth headroom to reclaim** — stock bf16 GEMV already measures 201–233 GB/s against a
~210 GB/s copy ceiling. **The decode GEMV's justification is the index stream, not out-coding
hipBLAS**: a CB rung reads `k/8` bits per weight (4.5 bpw at K36) where bf16 reads 16, and the
right question is how much of that byte advantage survives the decode. Measured against a
perfectly bandwidth-bound bf16 GEMV of the same logical matrix, at N=K=4096: **K36 = 1.55×,
K40 = 1.40×, then a cliff to a flat 0.80–0.89× from K42 through K48**. Flat-while-bytes-rise
means the top rungs are **decode-bound, not bandwidth-bound**, and the cliff coincides exactly
with the codebook LUT ceasing to be LDS-resident. Actionable consequence: **on Strix prefer the
K≤40 rungs** — which is also where a 58 GB box wants to be. The WMMA prefill GEMM reaches
**11.9 TFLOP/s** (K44, M=128), ~22% of a spec-derived peak.

**The load-bearing measured fact: RDNA 3.5 has no fp8 matrix instruction.** A device-pass
`__has_builtin` probe (`csrc_hip/wmma_probe.hip`) shows bf16/f16/**iu8** WMMA present and every
fp8 variant absent — those are gfx12/RDNA4. So fp8-CB here is a **storage** format decoded to
bf16 fragments; since e4m3's 3 mantissa bits fit bf16's 7 the decode is lossless and the
arithmetic is *better* than an fp8-MMA path, not worse. The same probe established the wave32
fragment layout on device (`D[2*i + lane/16][lane%16]`), which is what lets the GEMM decode B
straight into registers with no LDS tile, and measured that the WMMA unit is **not** exactly
rounded (~0.5 f32 ULP), which is why every parity gate here is relative rather than bit-exact.

**The codebook grid is a free choice on this platform, and the kernels are built for that.**
Since an FP8_CB codeword must be materialised as bf16 for WMMA regardless, a bf16-grid codebook
is same-bytes, same-speed and a strict grid superset of the e4m3 one — so the likely artifact
design is one index stream plus a per-grid codebook (~0.02% of artifact bytes), Blackwell
reading e4m3 and Strix bf16. The HIP kernels are therefore **dtype-agnostic at materialisation**:
the LDS LUT is always filled as bf16 and any e4m3→bf16 conversion happens **once at fill time,
never per gather** (the ALU term R6 removed on Blackwell stays removed), pinned by a test that
asserts a bf16 sidecar is **bit-identical** to the e4m3 one. The cost is LDS footprint —
a materialised LUT is 2 B/element, so K48 and NVFP4_CB K24 need the full 64 KiB. Public NVFP4 K25
exceeds that old whole-codebook LDS envelope and therefore requires an attested
global/partitioned lookup route; the registry expansion does not claim the historical LDS kernel
covers it. Research-only K26..K32 (K32 is 1 MiB across its two subtables) are unsupported public
formats. K24 and below still *fit* here because A and B are
register-resident and the LUT is the only LDS consumer; the top historical rungs use the
global-gather arm because it is measurably faster, not because the LUT overflows.
`iu8`/`iu4` WMMA are present, and measured on this box at **1.56× bf16 LDS-fed but only 1.06×
register-resident** (iu8) and 2.86× (iu4) — iu8's gain is halved LDS traffic, not faster math,
which buys little in a GEMM whose operands are already in registers. Both need integer
activations and an accuracy gate is running; **neither is started**.

**Not yet true, and the docs must keep saying so:** no vLLM-ROCm serve has been attempted, so
there is **no serving-metric claim** — no KL, no PPL, no tok/s under a real engine; the
`linear_hip.py` dispatch is authored but never exercised inside a live serve; MoE/grouped-expert
HIP kernels do not exist, so an MoE artifact falls back to Triton on ROCm; and the signed S-rung
path is compiled but untested. Quantization stays on the Spark either way — R7's load-bearing
half is unchanged, since probe/cost/render/export are CUDA and need zero ROCm work. GGUF remains
the zero-code serving lane (§9.3). Full status, the LDS budget table per rung, the measured
LDS-vs-global LUT policy and its shape-dependence, seven Fedora/ROCm bring-up landmines and the
deferred-work list was kept in the deleted prototype's `csrc_hip/README.md`; the dated audit is
the surviving record.

## 11. History — what was tried and rejected

Two conventions. (a) Every rejection gets a **dated wall**: `archive/<name>_YYYY-MM-DD/` with a
top-level `README.md` banner stating the kill order and the lesson. (b) Four of those walls are
**load-bearing for the orchestrator** — `run-pipeline.sh` fail-fast messages name them by path,
so `archive/` cannot be moved or renamed without editing the `exit 2` gates of §3.5. Doc-only
walls live under `docs/archive/`; code walls under repo-root `archive/`.

| Method | Why it lost (the lesson) | Wall / gate |
|---|---|---|
| grouped-KL cost surrogate | "−3.52% PPL" was a local screen; lost the vLLM A/B. Promote on the serving metric. | `archive/grouped_kl_2026-05-28/` · gate §3.5 |
| Fisher-weighted GPTQ / Fisher output-MSE allocator | Killed by order; no demonstrated utility on a production model. | `archive/fisher_2026-05-15/` · gate §3.5 |
| Hadamard-DuQuant (HDQ) | Fold-only preconditioner, no served win. | `archive/hdq_2026-05-14/` · gate §3.5 |
| Multi-shot recalibration | Double-negative: ΔKL=0 at production calib, −153% on a small calib. | `archive/multi_shot_2026-05-19/` · gate §3.5 |
| CLADO full IQP solver | O(N²) per-pair measurement; the O(N) cascade matched it to 1–2%. Framing kept (`decision_units.py`), solver dropped. | `archive/cross_layer_2026-05-09/` · docs `docs/archive/block_clado/` |
| Sparse pairwise QUBO / SMRF | 8-of-~500-Linear coverage is homeopathic; too local to fix global non-additivity. | same wall |
| Top-K Hessian covering | Blind to the propagation graph; misses small-eigenvalue Linears with long downstream paths. | same wall |
| L3-polish-of-many DP | Per-Linear L3 costs measured under L2 context do not sum when many units flip at once. | `archive/polish_2026-05-15/` |
| Top-down / ceiling-start polish | Spends its budget on cheap ~12-bit flips, never reaches the knee bpp range. | same wall |
| Coordinate-descent polish (as a shipped stage) | Overfits at n=8 (train→val sign flip); provable only under its own polish-time evaluator. | same wall |
| HALO / Hadamard-Fisher rotations | Worked once on Qwen3.5 dense, never on a production model; cut in the 2026-05-15 consolidation. ParoQuant (`2511.10645`) is the tracked replacement. | `archive/halo_2026-05-15/` |
| ReSpinQuant / layer-wise rotations | Needs a residual-transition adapter (a custom kernel) at serve time — forbidden in the vanilla-vLLM container. | `archive/respinquant_2026-05-13/` |
| Fold-scale / OrthoG, DuQuant++ fold | Preconditioner family, no served win at matched bpp. | `archive/foldscale_orthog_2026-05-13/`, `archive/duquant_dqpp_2026-05-13/` |
| PrismaClip / PrismaFisherClip | Subsumed by JSO's per-block scale grid — clipping is another way of asking what the right scale is. | `archive/prismaclip_2026-05-14/` |
| `scale_sweep` as a default lever | +77.5% KL on 4B: re-picks block scales *after* GPTQ, mis-calibrating its error compensation. Still reachable via `--enable scale_sweep` for ablations. | no wall (menu-only) |
| SAO (column permutation) | Failed on its own objective; redundant with GPTQ's full-Hessian propagation. | `archive/sao_2026-05-15/` |
| REAP / expert pruning | Cost model under-counts token redistribution and misrouting. Hit size via format/factorization, not pruning. | `archive/reap_2026-05-15/` |
| Entmoot expert-merge | Never wired into the runtime. | `archive/entmoot_2026-05-03/` |
| Analytical / closed-form GPTQ damp | +100–161% KL vs the discrete sweep; the fit's 2.4× per-Linear error compounds. Then the sweep itself fell (below). | `docs/design/unified_render_theory.md` |
| GPTQ damp sweep (as default) | Its evaluator was in-sample; held-out basins invert 31/31; served A/B null per role. Fixed damp 1.0 (§5.3). | flag-only, `PRISMAQUANT_GPTQ_DAMP_SWEEP=1` reproduces |
| Surrogate-only knee | On 27B the surrogate knee picks 5.857/0.056, validated picks 5.31/0.015. Outside the additive trust region, bpp order ≠ KL order. | superseded by `SELECTION_MODE=validated-surrogate` |
| Kneedle as the ship rule | Axis-dependent and LOO-unstable (fp32 4B: elbow at 5.00 in 454/1000 bootstraps). Byte budget + saturation B* replaced it; `allocator.py:1247-1252` says so in the CLI itself. | demoted, not removed |
| Lagrangian λ-bisection (as selector) | The discrete frontier has non-convex pockets no λ selects. Kept as a candidate *generator*. | demoted |
| The three-level cost cascade (L1→L2→L3) | **Retired from the spine 2026-07-30 (R4).** L2 beat additive L1 by −1.5% while AURA beat L1 by −38.5%; pairwise residuals are +5–12% diffuse with 3/1180 pairs significant and the apparent non-additivity is a bf16 artifact; L3-polish-of-many does not compose. One faithful cost + real-KL selection replaced all three (§2.2, §4.4). | `archive/l3_propagated_2026-07-30/` |
| `COST_MODE=production-render-staged` | Rendered NVFP4 first, promoted only the top-30% error tail — so every Linear outside the tail carried an `unavailable` cost the DP could not consider. On 27B its last-token-KL screen improved (0.0232 vs 0.0280) while **direct WikiText PPL regressed 10.83 vs 8.33**; the result doc says "Do not ship". The canonical screen-vs-gold inversion. | `archive/production_render_staged_2026-07-30/` |
| `MSE_PROMOTION` post-frontier rewrite | Re-ranked the *already measured-KL-selected* point by local `output_mse_per_bit`. On 35B it beat the strategic baseline but lost to both the shipped 4.75 and the 5.16 kneedle. A post-allocator rewrite cannot beat a better cost inside the DP (AURA). No shipped run carries `layer_config_before_mse_promotion.json`. | `archive/mse_promotion_2026-07-30/` |
| `PRODUCTION_CACHE_UNION` smart-union cache | Saved ~40% of the frontier render by offering an FP8 rung only above a percentile of the NVFP4 `output_mse` surrogate — a render-budget heuristic deciding the allocator's candidate set (principle 1). Never used by a shipped artifact. This archived selector is unrelated to §5.4's live exact set union, which never chooses or prunes candidates. | `archive/union_cache_2026-07-30/` |
| Block-output match (quality lever #12) | **Unreachable, not unmeasured.** The production-cache pack `continue`s first, so on the shipping recipe no dense NVFP4 Linear ever reached the branch (0 hits in two real export logs). Had it run it would have re-derived NVFP4 scales outside `_export_match_render_scale_rule` and discarded the render's `joint_mse` scales (the −6.6% M19 defect). Its `{0.95,1.0,1.05}` gain search is subsumed by JSO; its "~0.05–0.10 PPL" was a pre-JSO expectation. | `archive/block_output_match_2026-07-30/` |
| Orphan modules and tools (4 + 6) | Zero references tree-wide; three belong to threads with recorded verdicts (damp-sweep OFF-final, xlayer null, export-config collapse subsumed). `_fast_kernel_guard` is the exception — an orphan that is a **missing caller**, booked as debt in §12 rather than declared dead. | `archive/orphans_2026-07-30/` |
| MXFP8 in the default menu | E8M0 pow2 scale wastes ~√2 of a binade (+13.8% output MSE over 410 Gemma Linears); exact-scale FP8 Pareto-dominates. Registry entry retained. | de-menued (§5.1) |
| MXFP4-grid codebooks | Shares NVFP4's element grid exactly and differs only in the scale plane — but E8M0's pow2-only scale costs **~25× at the 4-bit grid what it costs at 8-bit**, and the 8-bit figure (+13.8%) is what de-menued MXFP8. The cross-platform premise is also false as stated: gfx1151 has no fp4 matrix path at all, so an MXFP4 grid buys nothing on the only non-Blackwell box we own. Revisit only with hardware that runs MXFP4 natively (RDNA4/CDNA4/Intel). | `docs/design/mxfp4_cb_feasibility.md` |
| MXFP6-grid codebooks | **In a codebook the grid is not a storage dial.** The stored stream is the k-bit index stream, so an MXFP6-grid rung stores byte-for-byte what FP8-CB stores at the same k; both MXFP6 grids are *exact subsets* of e4m3 (63/63 values round-trip, verified), so the codebook is an FP8-CB codebook handicapped to a subset, decoding to the same e4m3 tile and the same GEMM. Strict dominance, not a tradeoff — no measurement warranted. Would change only on a part with a genuine 6-bit matrix rate (CDNA4/MI355X runs MXFP6 at 2× fp8). | `docs/design/mxfp6_cb_feasibility.md` |
| CB persistent-N dense prefill; decode contract v2; w2 `rowpack`; chunked expand/GEMM overlap | Parity-green, 0.74–5.7× slower. Quarantined behind flags, kept as measured negatives. | `docs/lanes/nvfp4-cb/STANDARDS.md` |
| CB `l2_pipeline` MoE prefill | Wedged live serving three times; DIAGNOSTIC-ONLY, excluded from `auto` in external Gridbook (`gridbook/moe.py`; evidence commit `afc64ec`). | same |

Derivations and the additivity/cancellation analysis behind the CLADO/QUBO rejections belong to
`paper/main.tex` §`sec:additivity`; the retired PrismaSCOUT paper (cascade spine, monotone
polish, full rejected-methods catalog) is at
`paper/archive/prismascout_paper_2026-06-05.tex`. Dated measurement records are under
`docs/results/`; superseded narrative docs under `docs/archive/`.

## 12. Known gaps and debt register

Honest register, code-cited, as of 2026-08-03 (`release/prismaquant-0.8.0`, implementation
baseline commit `7183d21`; external Gridbook pin
`9011a19228ddb96b8a49e11a20ac75c99c83998e`, v0.8.0). The DSv4 study's working tree carried a
proposed **D29** ("the native-only Gridbook candidate is measured but not yet an attested
runtime"). It is deliberately **not** ported: that candidate was subsequently released and
the former 0.8.4 consumer was independently attested; re-adding the row would assert a
stale pin (`59cebf9f…`, v0.4.1) that no longer exists in this tree. The study's measurement
half survives as the dated §9.2 record.
Severity is operational risk, not effort. Plugin-contract leaks are stated in §8.5 and only
referenced here. Entries closed on 2026-07-30 are kept, marked, for one cycle so a reader
returning with a stale copy sees the resolution rather than silence.

**State after all four re-vet waves.** Closed today: D2, D3, D4, D5, D6, D7, D8, D9, D10, D12,
D13, **D14**, **D15**, D16, D19, D20, D21, D27 — and D1 is implemented but shipped default-off,
so its *default* remains a decision for Robert rather than debt.
Still open, unchanged by the waves: **D11** (no profile-validator preflight for the actual
`MODEL_PATH`), **D17** (registry vs export-scheme metadata unreconciled), **D18** (two dead
PrismaQuant flag tokens not yet deleted), **D23** (no
accounting-era stamp), **D24** (KV-cotangent path never run on a real KV-sharing checkpoint),
**D25** (Gemma4 tied-embeddings result is enablement, not quality), **D28** (fast-kernel guard
has no caller) — and **D26**, whose measurement half closed (a GGUF KL evaluator exists) while
its plumbing half did not (the frontier loop is not wired to it; `PACKED_ROLE_SPLIT` is still
unplumbed).

| # | Item | Evidence | Sev | Suggested action |
|---|---|---|---|---|
| D1 | **FIXED 2026-07-30 (R9).** Tail-veto was unimplemented since 2026-06-05 — and it had stalled on an assumed cost (a second eval pass) that does not exist. **Mechanism:** every KL site already accumulated per-sequence values and discarded them at the return; both selection paths now return `(mean, per_seq, stats)`, so `kl_p95/kl_p99/kl_max` and the rung-2 `nll_mean/nll_p99` (one `gather` + `logsumexp` on logits already in hand) cost **zero extra forwards**. `_frontier_from_rows` gained a second admission condition — `row[tail] <= incumbent[tail] * (1 + tail_eta)` — behind `--tail-veto {none,kl_p99,kl_max,nll_p99}` / `--tail-eta`, with vetoed rows retained under `vetoed_rows` + `veto_reason` so a refusal is visible. **DEFAULT-ON since 2026-07-30** with `kl_max` (the worst sequence) as the contract statistic — Robert's ruling; `--tail-veto none` still reproduces the pre-R9 envelope byte-for-byte (pinned by a frontier-identity regression test). The slack is **derived, not chosen**: `--tail-eta auto` (default) is the incumbent's between-repeat relative stderr of the tail statistic, degrading to a strict 0 **with a printed warning** on a single repeat. A pre-R9 validation JSON (no tail column on any row) makes the veto go inert with a warning rather than empty the frontier. §4.6, §7.1. | `select_validated_frontier.py` `_frontier_from_rows`, `measured_rows`, `tail_eta_auto`, `tail_veto_inert_reason`, `TAIL_VETO_COLUMNS`/`TAIL_REPEAT_COLUMNS`; `validate_assignments_kl._kl_repeat_summary` (per-repeat tails); `kl_measurement.sequence_token_nll` / `summarize_per_sequence_kl`; `tests/test_select_validated_frontier.py`, `tests/test_kl_per_sequence_tail.py` | — | CLOSED — default-on, `kl_max`, repeat-derived eta (ruled 2026-07-30). |
| D2 | **FIXED 2026-07-30 (R12).** MTP construction bypassed the profile — §8.5 L2. All three import sites now call `profile.build_mtp_module()` / `read_mtp_source_state_dict()` / `load_mtp_state_dict()`, keyed on the new `mtp_source_prefix()` accessor; `prismaquant/mtp_module.py` is deleted and DSv4 declares `has_mtp → False` + `"mtp."` passthrough. | §8.5 L2 | ~~HIGH~~ CLOSED | — |
| D3 | **CLOSED 2026-08-01 (R10 ownership follow-through)** — was: Gridbook per-arch CB expert opt-in as a hand-maintained list inside PrismaQuant, with a missing line failing silently as coherent garbage generation. Gridbook now solely owns its loader registry and unbypassable fill guard; PrismaQuant carries no runtime copy and required CI checks its one exact pin, PEP 610 provenance, producer-profile set, and emitted artifacts against Gridbook's packaged contract. | §8.5 L3; `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; `tests/test_gridbook_runtime_contract.py` | ~~HIGH~~ closed | — |
| ~~D4~~ | **CLOSED 2026-07-30 (re-vet R11).** `TARGET_PROFILE` has no shell default; `--target-profile` reaches the allocator only when requested, with `--target-profile-default vllm_packed_moe` as the fallback; the allocator stamps its resolved profile into `layer_config.json`'s reserved `__prismaquant__` block and the exporter reads it (env override kept). Non-regression 0/614 and 0/500 on the shipped 27B/35B. See §8.5 L1. | §8.5 L1 | ~~HIGH~~ | closed |
| D5 | **RESOLVED 2026-07-30.** `PRISMAQUANT_GPTQ_DAMP_SWEEP` had two readers with opposite defaults — `"0"` in the exporter, `"1"` in a forked lever-defaulting copy inside the KL sensitivity probe (stale from `9c91d62`, missed by the sweep-OFF policy in `f2363e2`), so any A/B touching both compared different renders. `_normalized_production_cache_levers` now delegates to `production_weight_cache._resolve_production_render_levers` — one contract, and the probe's stamped provenance can no longer disagree with the render that produced it. | `archive/l3_propagated_2026-07-30/prismaquant/kl_sensitivity_probe.py:272-285`; `tests/test_production_weight_cache.py` | — | Done, and fully closed later the same day: R4 walled the probe itself, so the forked reader no longer exists in the live tree. The follow-up landed too — the delegation contract is pinned by `tests/test_production_weight_cache.py`, which carries it as a local shim and keeps every assertion (sweep OFF by default; sweep-off renders must record their fixed damp). |
| ~~D6~~ | **CLOSED 2026-07-30 (re-vet R5).** Closed by *mechanism*, not enumeration: `pipeline.STAGE_SETTINGS_KEYS` declares each artifact's key set, `run-pipeline.sh` supplies values once, and the guard now covers every skip-if-exists artifact (16 call sites / 15 artifacts). `cost.pkl` additionally carries a `provenance["cost_mode"]` stamp so a mode change cannot silently reuse the other estimator's table (R2 precondition (i)). See §3.4. | §3.4; `pipeline.py` `STAGE_SETTINGS_KEYS` | ~~HIGH~~ | closed |
| D7 | **RESOLVED 2026-07-30 — and the original diagnosis was wrong.** The register previously read "`pyproject.toml` on `main` is `0.1.0` while PyPI serves `0.4.1` from a tag that is not an ancestor of `main`", implying the release had been cut off-trunk. It had not: `origin/main` *was* the release source all along (`v0.2.0` `4745887` → `v0.2.1` → `v0.3.x` → `v0.4.1` `d058267`, each an ancestor of `origin/main`), and the **local** `main` ref was simply 54 commits behind. Merging `origin/main` into this branch (`8f14400`) brings the whole release stack: `pyproject.toml:7` is `0.4.1`, `requires-python = ">=3.11"` (`:14`), plus the tag-driven PyPI pipeline, packaging gates and `docs/RELEASING.md`. `git merge-base --is-ancestor v0.4.1 HEAD` → true. Lesson: verify a divergence claim against the **remote** ref before filing it as debt. | `pyproject.toml:6-14`; `.github/workflows/release.yml`; `git merge-base --is-ancestor v0.4.1 HEAD` | — | Done. Follow-up: fast-forward the local `main` ref so the next reader's `git log main` is not 54 commits stale. |
| ~~D8~~ | **CLOSED 2026-07-30 (re-vet R24).** `_production_cache_prefetch_assignment` gained a `require` mode mirroring `production_weight_cache.prefetch_assignment(require=…)`, exposed as `--production-cache-prefetch {require,warn}`; `run-pipeline.sh` passes `require` on the native lane (matching `VALIDATED_SOURCE_PREFETCH=require`), and the CB/GGUF lanes read no production cache at all. A total miss is now a named failure instead of a silent NVMe-bound export. | `export_native_compressed._production_cache_prefetch_assignment` | ~~MED~~ | closed |
| ~~D9~~ | **CLOSED 2026-07-30 (re-vet R24).** The guard is at `main()` entry (not import time) in all seven — `incremental_probe`, `incremental_measure_quant_cost`, `aura_cost`, `production_render_cost`, `export_nvfp4_cb[_streaming]`, `export_gguf`, `select_validated_frontier` — verified against every CPU-only test import first, and a parametrized test pins all twelve callers so a refactor cannot drop one. | `gpu_guard.py` | ~~MED~~ | closed |
| ~~D10~~ | **CLOSED 2026-07-30 (re-vet R5).** `pipeline.py` now has one real job — settings-hash authority (§3.4) — and the bookkeeping is honest: the two owner names that existed nowhere in the tree are deleted, `streaming_model_weights` names `layer_streaming.LayerCache`, and a test asserts every approved owner has a class behind it. `QuantWeightCache` went to the archive wall with L3, so it is no longer an unmodelled holder. The *spec* half stays explicitly descriptive (§3.6); modelling the ten executed-but-unmodelled stages was refused as fiction-surface. | §3.6; `pipeline.py` | ~~MED~~ | closed |
| D11 | **MOSTLY FIXED 2026-07-30.** `model_profiles/validate.py`'s 8 conformance checks had zero callers and there were no workflow files in the tree. Both halves closed: `.github/workflows/ci.yml` (#18, `1cc7b90`) runs the suite on every push and PR (Python 3.12, CPU torch), and `tests/test_model_profile_conformance.py` drives the CPU-safe checks (1, 6, 8 + four structural invariants) over every registered profile, with 2/3/4 behind `integration` and 6/7 behind `slow`, and known gaps encoded as ratchets rather than bare xfails. **Residual (2026-07-30, R12): the check-5 half is now covered** — `test_has_mtp_implies_a_buildable_mtp_module` asserts `build_mtp_module` is a real override (and `mtp_source_prefix()` non-empty) whenever `has_mtp()`, which is the declarative part of the check that would catch L2/D2; check 5 proper still materialises a decoder layer and stays out of CI. Remaining: nothing invokes the validator as a `run-pipeline.sh` preflight for the actual `MODEL_PATH`. | `.github/workflows/ci.yml`; `tests/test_model_profile_conformance.py:9-31,223-249` | LOW (was MED) | Add a preflight invocation for `MODEL_PATH`. |
| ~~D12~~ | **CLOSED 2026-07-30 (re-vet R1).** `TARGET_DISK_GB` is plumbed through `run-pipeline.sh`: it overrides `TARGET_BITS`, narrows the Pareto sweep to the byte-feasible bracket, flips `SELECTION_MODE` to `validated-surrogate` and the frontier pick to `budget` = min measured KL among the rows that fit. Kneedle stays the default without a card and stays a diagnostic. See §4.6. | §4.6; `select_validated_frontier --mode budget` | ~~MED~~ | closed |
| D13 | **FIXED 2026-07-30 (R22 + R27).** The two hardcoded MiniMax arch tests now route through `profile.bypass_hf_fp8_module_rewrite()` and `profile.packed_expert_module_class_names()`; `specs/minimax_m2.json` exists and declares all eight of that profile's overrides; `deepseek_v4.json` declares `default_serving_profile: vllm_packed_moe`. Core-stack arch literals in control flow: **0**. Residual (not debt, sequencing): the MiniMax Python overrides stay until the equivalence gate `tests/test_minimax_m2_spec.py` has held for a release. | §8.4, §8.5 L4 | closed | — |
| ~~D14~~ | **CLOSED 2026-08-01.** Runtime documentation now lives with the sole canonical Gridbook package. PrismaQuant documents only its producer/export contract and points to the pinned package's machine-readable runtime contract; the former in-tree README was deleted with the vendored runtime. | external Gridbook `README.md`; `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` | ~~MED~~ | closed |
| ~~D15~~ | **CLOSED 2026-07-30 (wave 4, R28/R3).** The approved option was taken — **flip the defaults to the shipped values**, not defend the conservative one: `CB_SCALE_CODING=two_tier` (layout-v2 shipped in the Hy3 295B and Laguna-S-2.1 artifacts, and `STANDARDS.md` calls it the production fp4 scale coding with v1 legacy read-compat only — the old "serve gates pending, do NOT ship" comment predated its own ship; the knob is inert on fp8-CB-only menus, which is why two 27B/35B drivers set `v1` without contradicting it) and `CB_EXPERT_EMPIRICAL=0` (every shipped MoE CB driver sets it). Both are pinned by `tests/test_architecture_doc.py::test_cb_defaults_match_the_shipped_drivers`, which asserts the shell default against the drivers themselves so the two cannot drift apart again. No shipped run changes: every driver sets both explicitly. | `run-pipeline.sh`; `tests/test_architecture_doc.py` | ~~MED~~ | closed |
| D16 | **RESOLVED 2026-07-30 (R25) — as *unreachable*, not unmeasured, and the A/B was never needed.** The register asked for a gold-lane A/B on a 27B-class artifact; reading the emit-loop dispatch order answered it for free: the production-cache pack fires first and `continue`s, so with `PRODUCTION_CACHE=1` (the shipping default) **no dense NVFP4 Linear ever reached the branch** — confirmed by `grep -c "block-output-match"` → 0 on two real production export logs. Two further findings made keeping it indefensible: had it run, `_finalize_compute_only` would have re-derived per-group scales **outside** `_export_match_render_scale_rule`, discarding the render's `joint_mse` scales (the −6.6% KL defect M19 fixed everywhere else); and its `{0.95, 1.0, 1.05}` per-tensor gain re-search is subsumed by JSO, wrapped in `except Exception → WARN` so failure was invisible. Walled with `_finalize_compute_only` and the three export branches; `main()` now hard-`SystemExit`s if the flag is set truthy (§3.5). **Lesson: before funding a measurement, check the code under test executes on the recipe you ship.** | `archive/block_output_match_2026-07-30/README.md`; `export_native_compressed.py::_refuse_archived_block_output_match` | ~~MED~~ closed | — |
| D17 | **Registry and export metadata are unreconciled sources of truth** for bits/group per format — `FormatSpec` vs the `*_SCHEME` constants, with no test comparing them (§6.4, last row). | `format_registry.py:44-168`; `export_native_compressed.py:7247-7336` | MED | Add a parametrized test asserting scheme ↔ spec agreement per production format. |
| D18 | **PARTIALLY FIXED 2026-08-01.** The Gridbook-documentation half is closed: Gridbook owns and publishes its runtime flags, and PrismaQuant no longer mirrors that external catalog. The only remaining debt is producer-local cleanup: the dead `PRISMAQUANT_L2_CUDA_GRAPHS` and `PRISMAQUANT_DO_NO_HARM_MIN_GAIN` tokens remain in the historical/dead section even though the former's sole code occurrence is a comment at `perturbed_x_cache.py:1225` and the latter has no code occurrence. The live producer analogue remains `PRISMAQUANT_RENDER_GATE_MIN_GAIN`. | `docs/design/runtime_flags.md`; external pinned Gridbook runtime contract | LOW | Delete the two dead PrismaQuant entries once no reader is chasing them. |
| D19 | **FIXED 2026-07-30.** The count was low: **14** launchers under `examples/launchers/`, not 8, invoke `python -m prismaquant.<module>` for a module that no longer exists (`iterate_block_clado`, `measure_block_clado`, `block_clado`, `validate_block_clado`, `measure_output_fisher`, `dense_cone`, `polish_from_assignment`, `coord_descent_polish`, `measure_adjoint_l3`, `adjoint_l3_frontier`). Walled at `archive/launchers_2026-07-30/` with a banner README enumerating each file and its dead invocation, per the dated-wall convention of §11. | `archive/launchers_2026-07-30/README.md`; `examples/launchers/README.md` | — | Done. |
| D20 | **RESOLVED 2026-07-30.** Two archive walls had no banner README (`archive/prismaclip_2026-05-14/`, `archive/reap_2026-05-15/`) — the latter walls off live-adjacent code (`expert_prune.py`, `allocator_prune.py`, `observers/`, 5 tests) and encodes a policy the code still enforces. Two more walls violated the dated-directory convention. Banners written; `archive/entmoot/` → `archive/entmoot_2026-05-03/` (date from `193f313`) and `archive/minimax_m2p7/` → `archive/minimax_m2p7_2026-04-24/` (date from its own banner). Neither renamed wall is cited by a `run-pipeline.sh` `exit 2` message. | `ls archive/*/README.md` | — | Done. Follow-up: a test asserting every `archive/*/` carries a `README.md`. |
| D21 | **RESOLVED 2026-07-30 (R28).** Three ids appeared across the docs for one Hy3 artifact; the premise "at most one is live" was wrong — they are *renames*, not rivals, so the older ids **307-redirect** rather than 404. Canonical id (verified against the Hub 2026-07-30): **`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`** — the one to cite in all new material. The two `prod_hy3_results.md` citations are the dated ship ledger and were **annotated in place** ("now redirects to …"), not rewritten: a ledger records what was posted on the day. §9.2's unresolved paragraph now carries the resolution. | `docs/lanes/nvfp4-cb/prod_hy3_results.md:248-251,313-320`; §9.2 | ~~LOW~~ closed | Done. Follow-up: `scratch/gridbook-launch-post.md:24,179` still carries a third variant (`…-prismaquant-codebook-2.9bit-vllm`) — `scratch/` is out of the doc contract's scope, so it is left as-is; do not cite from it. |
| D23 | **bpp labels are not comparable across accounting eras.** The public "5.31" artifact's body bpp is ~4.76 under current accounting (§1.2); nothing in the tree records which era an artifact's label came from. | §1.2 | LOW | Stamp an accounting-era field into exported artifact metadata. |

New with the 2026-07-30 merge:

| # | Item | Evidence | Sev | Suggested action |
|---|---|---|---|---|
| D24 | **The KV-cotangent path has never touched a real KV-sharing checkpoint.** Its correctness is established by exact fp64 equivalence on a synthetic model (rel err 0.00e+00 vs one end-to-end autograd backward; the pre-fix protocol under-counts `k_proj` 85.1% / `v_proj` 38.5%) — a demonstration, not a measurement. No `num_kv_shared_layers > 0` model has been probed, so the magnitude of the correction on a shipping architecture is unknown, and the guard it replaced (`PRISMAQUANT_ALLOW_KV_SHARED_FISHER`) was the only thing previously stopping such a probe. | §7.5; `tests/test_kv_cotangent_path.py`; commit `b6ec9cb` | MED | Probe one real KV-sharing checkpoint (Gemma4-class) with the path on and off, and record the h_trace delta before any allocation claim rides on it. |
| D25 | **Gemma4-31B tied-embeddings result is enablement, not quality.** The first end-to-end probe → cost → allocate → export on a tied model (244 NVFP4 / 119 FP8 / 27 BF16 at achieved 6.000 bpp, 27.18 GB, `tie_word_embeddings` preserved and no duplicated `lm_head` bytes) ran at **2 samples × seqlen 512** to reach failures fast. The artifact has not been served and no KL/PPL exists for it. Nothing in §1.2 should cite it. | §7.5; commit `d058267` | MED | Re-run at production calibration and take it through the §7 gates before the family table gains a row. |
| D26 | **MEASUREMENT HALF CLOSED 2026-07-30 (wave 4, R16); the plumbing half is open.** The lane now has a KL evaluator: `prismaquant/gguf_kl_evaluator.py:measure_assignment_kl` wraps `llama-perplexity --kl-divergence-base` behind the `validate_assignments_kl` interface and returns `(mean, per_sequence, stats)` under the gold lane's key names — with the honest caveat that `per_sequence` is empty and `kl_tail_domain="aggregate"` (llama.cpp reports token-domain quantiles). Parsing is pinned against canned output in both shipped spellings; the live path is integration and unrun. **Still open:** `run-pipeline.sh`'s frontier loop is not wired to it (GGUF selection is still `surrogate`), and there is still no `PACKED_ROLE_SPLIT` plumbing, so every use of the split is a manual `allocator.py` invocation. | `prismaquant/gguf_kl_evaluator.py`; `prismaquant/lane_specs/gguf.json`; `grep -c PACKED_ROLE_SPLIT prismaquant/run-pipeline.sh` → 0 | LOW | Wire the frontier loop to the adapter, and plumb `PACKED_ROLE_SPLIT`. |
| D27 | **CLOSED 2026-08-01; import resolution hardened 2026-08-12; immutable-wheel parity added 2026-08-13.** The version skew was not benign enough to preserve: the vendored package, mirror, and sync test were deleted. PrismaQuant consumes one full-commit pin, verifies package version plus PEP 610 exact-VCS identity or an independently pinned release-wheel digest, launches from a neutral directory in Python safe-path mode, rejects an import outside the selected distribution root (including CWD/`PYTHONPATH` shadows), and fingerprints that import origin plus the complete RECORD-bound source closure for every pinned serve. | `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; `prismaquant/gridbook_runtime/gridbook_runtime.sh`; `tools/serve_fingerprint.py`; `tests/test_serve_fingerprint_descendants.py` | ~~LOW~~ closed | — |
| D28 | **Serve-time fast-kernel enforcement has no caller.** `require_fast_kernels(model)` — which reads the model profile's kernel requirements and hard-fails at startup when a required fast kernel (`causal-conv1d`, `flash-linear-attention`, …) is not importable — lost its only caller when `polish_from_assignment` was archived on **2026-05-15**, and was itself walled 2026-07-30 (R19) as an orphan. It is the only mechanized piece of **core principle 9's** "routed to a *performant* kernel (not a slow fallback)" gate, so that gate is **manual today**: nothing in the build or serve path refuses a checkpoint whose arch would silently fall back to the slow PyTorch implementation. The mechanism is written and tested — only the call site is missing. | `archive/orphans_2026-07-30/prismaquant/_fast_kernel_guard.py` + `tests/test_fast_kernel_guard.py`; sole historical caller `archive/polish_2026-05-15/prismaquant/polish_from_assignment.py:202` | LOW | Move the guard back and call it from `validate_native_export` / the serve launcher, keyed on the resolved profile — or, if serve-time enforcement belongs to the lane scripts, say so in §7 and delete the row. |
| D29 | **The FP8-CB row scale is not bit-reproducible across CPU architectures.** It is the scalar argmin of a scale sweep whose objective reduces over every column of the row, and that reduction reorders differently on x86 than on aarch64: on the fixed `test_cbl_scope_identity` fixture the packed index bytes -- the payload that actually ships -- are **identical** on both, while the single float32 scale differs in the low bits. Found 2026-08-11 when a byte-identity test recorded on the aarch64 build box failed on x86 CI. Consequence for the provenance gate (§5): artifact byte-reproducibility is a **within-platform** guarantee, not a cross-platform one; a rebuild on a different architecture may differ in scale bytes without differing in indices. Artifacts are built on the Spark, so nothing shipped is affected. The test now pins the packed plane by exact digest everywhere and the scale by value within float32's own worst-case reordering bound (n·2^-23), keeping the exact digest assertion on the recording platform. | `tests/test_cbl_scope_identity.py::test_unset_scopes_pin_76666bd_stamp_and_rendered_bytes`; `nvfp4_cb_formats._sweep_encode_moment` | LOW | Decide whether cross-architecture byte reproducibility is a goal at all. If it is, the sweep objective needs a fixed reduction order; if it is not (the likely answer -- artifacts are Spark-built), say so in §5 so a future reader does not read a cross-platform promise into the provenance gate. |
| D30 | **The Sensitivity Card's non-scalar tiers are screening surrogates, and its probe wiring has two soft spots** (added 2026-08-14, §4.8). Four honest gaps, none of them closed: (1) **No served A/B.** The `MARGINAL` tier and AQUA-AURA have never been measured on exact full-vocab vLLM KL-vs-BF16 or direct WikiText PPL. `SCALAR` is a byte-identical reproduction of today's model and carries no such debt; the other two must not be cited as results (§2.5). (2) **The rank-1 reconstruction's error is unquantified on real layers.** `H = Σ_t outer(g_t², x_t²)` is exactly rank-1 only when one token dominates; `outer(row, col)/h_trace_raw` is provably exact in that case (`rtol=1e-10`) and an approximation of unknown magnitude everywhere else. Nothing has compared it against a materialized `H` on a real Linear. (3) **The marginal identity is exact only at the two streaming sites.** `sum(fisher_row) == sum(fisher_col) == h_trace_raw` holds by construction where `h_trace_raw` is literally `chunk_h.sum()` in fp32 (`incremental_probe.py:2520`, `:2751`). On the **resident** path `h_trace_raw` comes from the bf16 outer-product-norm identity `(gy2_sq.sum(1) · x2_sq.sum(1)).sum()` (`:1667-1668`) while the marginals reduce the fp32 `chunk_h`, so the two agree mathematically but not bitwise; `SensitivityUnit.validate`'s `rtol=1e-3` is what absorbs that, and nothing measures the actual spread. (4) **One accumulation site is dead on the shipping path and therefore untested.** The batched MoE block-flush hook (`:2276-2362`) fires only for blocks whose immediate children are per-expert containers exposing the profile's projection names as `nn.Linear` — the *unpacked*-expert layout. The shipping recipe's MoE models do not take it, and `tests/test_probe_marginals.py` covers the helpers and the two streaming sites but not that branch, so its marginal emission has never executed. A transposed axis or a wrong merge rule there would surface first on a new unpacked-expert architecture, which is exactly the class of silent-garbage failure §8.5 L3 is about. | §4.8; `prismaquant/sensitivity_card.py`, `format_cost_protocol.py`, `sensitivity_card_allocate.py`; `incremental_probe.py:97-199,1667-1672,2276-2360,2501-2520,2735-2751`; `tests/test_sensitivity_card.py`, `tests/test_probe_marginals.py`; `docs/design/sensitivity_card_contract.md` §8 | MED | (1)-(2) run the rank-agreement check against measured `output_mse` on Qwen3-0.6B and an allocation-churn check against a shipped `cost.pkl` before any tier but `SCALAR` is proposed for a default; (3) record the resident-vs-streaming identity spread on one real probe, or tighten the resident path to reduce `chunk_h` for both; (4) cover the MoE block flush with a synthetic unpacked-expert fixture, or state that the branch is retired. |
| D31 | **Shipcard replay binds recorded evidence to the serving pin at HEAD** (added 2026-08-18). Every gate slot records the runtime that actually gated it (serve-manifest `gridbook_distribution`, endpoint-contract stack), but the replay compares those records against `load_gridbook_serving_runtime_pin()` at HEAD — so the 0.8.9 pin bump made the already-published DSv4 flagship unpublishable for a docs-only README update: six slot refusals, all "is not the tracked pin", on evidence that exactly matches the pin that was tracked when it was measured. Worked around honestly for the 0.8.9 card update by running the publisher from a worktree at `0266662` (the pre-bump commit; publisher and verifier code there are byte-identical to HEAD — the bump commit `6a883bc` touched pin data and docs only — so this verifies the card against the pin that gated it, with zero tool divergence). Recurs on every serving-pin bump for every historical artifact. | `prismaquant/shipcard.py:1225,2374,2521`; `tools/publish_artifact.py` dry-run refusal 2026-08-18 | MED | Decision for Robert: accept a declarative superseded-pins record in `gridbook_serving_runtime_pin.py` (version/commit/wheel of prior released pins; replay accepts recorded == current OR recorded ∈ superseded, and the verdict names which) — keeps fail-closed against unreviewed runtimes without rotting history — or rule that docs updates to historical artifacts always re-run the publisher at the artifact's pin era. |
| D32 | **The Fisher probe is not bit-reproducible, and nothing in the tree said so** (added 2026-08-20). Two runs of `incremental_probe` with byte-identical calibration, the same commit and the same `--layers-per-shard` differ on **379/402 units**, median `|Δh_trace|/h_trace` **2.5e-4** (max 1.1e-2); `n_tokens_seen` and the per-expert Fisher *support* are bit-identical on every unit, so the forward and the routing are exactly deterministic and only the backward moves. Mechanism: 30 of Ornith-1.5-35B-A3B's 40 layers are Gated DeltaNet, whose `fla` Triton kernels reduce over chunks in a non-deterministic order. **Why it is debt rather than a bug:** the jitter is unbiased (signed mean +6.5e-5 against its own sd 5.7e-4) and three orders below the 23% cost CV that §9 records as producing 3% assignment churn and 0σ served — but a probe-side change gated on bit-identity refuses for reasons that have nothing to do with the change, and `--layers-per-shard auto` (sized from free RAM at launch) adds a second, *avoidable* source on top. **Consequence for provenance:** probe-derived artifacts (`cost_baseline.pkl`, `cost_aura.pkl`, `cost.pkl`, the sensitivity card) must be rebuilt together from one probe run rather than half-reused, or `cost.pkl`'s stamped provenance names a probe that produced only some of its numbers. | `incremental_probe.py`; `sensitivity_probe.py` `_accumulate_packed_per_token_fisher`; measured Ornith-1.5-35B-A3B 2026-08-20 | LOW | Gate probe changes on what is invariant (`n_tokens_seen`, per-expert support, an unbiased signed mean within a *measured* floor), never on bit-identity; pin `--layers-per-shard` for any A/B. |
| D33 | **OPEN 2026-09-02, narrowed twice.** Tessera is priced and rendered by name (§5.7), has a *declared* lane (§9.4, `lane_specs/tessera.json`), a real serving runtime of its own (`tessera.serving`, `quant_method: "tessera"`), and since **2026-09-03** a real `EXPORT_CONTAINER=tessera` arm in `run-pipeline.sh` that plans and encodes through Tessera's OWN tools under `TESSERA_REPO` (`plan_from_layer_config.py`, `export_tessera_serving.py`). **The "no exporter codec" half is re-scoped, not closed**: this repository still writes no Tessera bytes and deliberately never will — a wire recipe with two homes is how the two halves of one format drift apart — so the debt is now *the boundary*, not *the absence*: the two Tessera scripts the arm names live in `experiments/`, which their own README calls drivers rather than a supported interface. Producer eligibility is still False, and the reason has narrowed to one thing: **no Tessera release tag exists**. The packaged contract publishes both families with `device_qualified` native cells; the PENDING pin is what withholds them, and it is now also what the driver's preflight refuses on. No ship gate has been run on the lane, and no runner exists to run one: the lane's six declared gates are now RECORDED — `route.census` names a shipcard slot and the arm opens a lane-gated card (§7.1, §9.4, 2026-09-03) — so an un-run gate is an unfilled slot the publisher refuses on rather than a sentence nothing reads, but nothing spawns the container that would fill it. The `experiments/` boundary is likewise declared rather than fixed: both tools carry `stability: "unsupported_experiments"` and a `tracking_issue` on `lane_specs/tessera.json`, which makes the debt visible to a reader and a gate without promoting anything. Gridbook's Tessera lane (contract v14) is withdrawn and was never released. | `tessera_render.py` (`tessera_lane_attested`), `tessera_export_lane.py` (the arm's four gates), `run-pipeline.sh` (`EXPORT_CONTAINER=tessera`), `tessera_runtime/tessera_serving_runtime_pin.json` (PENDING sentinels), `tessera_serving_runtime_pin.py` | Med | Rob cuts a Tessera release tag, then one reviewed commit resolves the pin file and the reader's two release constants together — after which the arm can build. Independently, and both still open under RobTand/prismaquant#119: promote the two named Tessera scripts from `experiments/` to a supported entry point so the boundary is an interface rather than a path — after which their `stability` becomes `supported` and the `tracking_issue` goes — and build the lane runner that executes a lane's declared gates in a fresh plugin container and fills the slots the card already opens. |
| D34 | **The Gridbook lane is retired but its format/cost/render plumbing is not** (added 2026-09-02). The lane, its pins, exporter, serving profiles, ship-gate slots, 73 test modules (1,691 node IDs) and 27 documents were archived at `archive/gridbook_lane_2026-09-02/` and `EXPORT_CONTAINER=nvfp4_cb` now `exit 2`s (§3.5, §9.2) — so no CB rung can be exported or served, which is the property principle 9 cares about. What remains is the machinery that *prices and renders* those rungs: `cb_layout.py`, `nvfp4_cb_formats.py`, `nvfp4_cb_footprint.py`, `cb_ldlq*.py`, `cb_minchain.py`, `cb_warm_state.py`, `cb_banked_books.py`, `cb_learned_promotion.py`, `cb_anchored_cost.py`, `cb_ladder_cross_family.py`, `routed_moe_codebooks.py`, `mxfp4_widen.py`, `source_class_format_plan.py`, plus CB branches inside `production_weight_cache.py`, `allocator.py`, `format_registry.py`, `export_native_compressed.py`, `layer_config.py`, `lane_spec.py`, `serve_constraints.py` and `model_profiles/*`, and roughly 60 tests that exercise them. **Why it was left:** the excision is several hundred diffuse edits concentrated in exactly the files the continuous-menu branch is rewriting, and merging that against a live branch is more dangerous than the debt. **The risk it carries:** a `FORMATS` menu can still name a `*_CB_*` rung, the DP can still price it, and the only thing that stops it is the exporter and the `production-render-score` pairing guard — a *refusal*, not an *absence*. Four consequences are recorded separately because they are capability losses, not debt. (i) `FP8_BLOCK_UE8M0_SOURCE` is now `ROUTE_STATUS_BLOCKED` — its only route was the plugin. (ii) `MXFP4_SOURCE` keeps a backed stock-Marlin route but has no writer and no serving profile, and `MXFP8_UE8M0_G32` is the same shape — never a compressed-tensors scheme, written only by the CB *streaming* exporter, which is archived. Both keep a live `FormatSpec` and a working render; neither has a writer. (iii) **The `serving_lanes` block of a serving-profile spec now has zero live declarations.** `serving_profile_specs/nvfp4_cb.json` was the only spec that ever declared one (verified against `d263f54`), so the per-lane structured `route_status` / `activation_contract` / `fused_mid_m` table that principle 9 reads is a parser with nothing left to parse; the native lane's route status has always come from the source-passthrough contracts instead. The parser and its `route_status_source` machinery are kept because that is the shape the Tessera lane must declare in. (iv) **The sample-parallel incremental probe is unavailable**: its `prepare-run-contract` minter and its per-worker source-census revalidation were both built on `prismaquant/rtx4090_artifact_census.py`, the strict-Ada FP8-CB campaign's closed Qwen3.8-27B layout. `incremental_probe.py --global-calibration-tensor` now refuses up front rather than admitting a pre-retirement contract with one leg of its identity replay missing (`docs/design/sample_parallel_probe.md` carries the banner). Reviving it means giving the census a lane-independent source of truth. Two production observations were surfaced by the removal, deferred at the time, and **both fixed 2026-09-03** (RobTand/tessera#20): `check_serving_shape` failed **open** on an unknown profile id — it caught `FileNotFoundError` and resolved silently to `research`, which permits every shape, while `serving_lane_route`/`serving_lane_catalog`/`check_serving_format` all fail **closed**. It now returns the same `profile_mismatch` refusal `check_serving_format` does; `profile_id=None` still resolves to `research`, which is the declared default and loads, so no legal call changed. And `activation_pricing_branches["unrecorded"]` is re-homed as its own profile-independent test in `tests/test_serving_lane_metadata.py` rather than left riding a deleted CB test. A fifth item is dead-but-kept rather than lost: `shipcard.py`'s `safetensors_content_receipt` trio has no live caller since the strict-RTX4090 publication gate retired, and is kept so receipts already on disk stay readable. `ROLE_COMPOSITE_FUSED_SOURCE_EXEMPT` still exempts `DeepseekV4Profile` from declaring a fused-sibling source, but the lane that justified the exemption is gone; discharging it is a producer-behaviour decision, not a removal. | `archive/gridbook_lane_2026-09-02/README.md`; `docs/measurements/gridbook-lane-retired-2026-09-02.md`; §9.2 | MED | Excise the CB plumbing after the continuous-menu branch merges, in one commit whose diff is deletions plus the tests that go with them; or, if a codebook rung is wanted again for the Tessera lane, port the parts worth keeping deliberately rather than inheriting them. |

**Open items carried from session handovers.** Of the 41 items the handover census could not
map to a verified closure, the prior FP4-CB fast-expander/Triton item is now closed by the
exact formerly pinned Gridbook 0.8.4 runtime: FP4-v2 prepares its native expander at model load, decode
uses native CUDA GEMV, M>8 uses native BF16 expansion plus Gridbook's owned CUTLASS grouped
bridge, and a missing operation fails closed. The remaining re-verified items are folded in
above: tail-veto (D1), `TARGET_DISK_GB` (D12), the DSv4 CB lane (D3), and the shipped
Mistral-Medium-3.5-128B artifact with no profile or spec (§8.4). Two are standing
research questions rather than debt: deriving the GPTQ damp constant from the weights, and the
XLAYER Q4 LFM2.5 routing-channel measurement. The remaining ~34 — mostly PrismaSCOUT-era items
that died with their subsystem — are enumerated with verdicts in
`scratch/doc-consolidation-2026-07-30/census_handovers.md` §POSSIBLY-STILL-OPEN.
