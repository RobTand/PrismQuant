> **ARCHIVED 2026-09-02.** This document belongs to the Gridbook codebook
> serving lane, retired that day on Robert's decision (*"put Tessera in
> PrismaQuant and remove Gridbook"*). It is kept because it records what was
> measured and asked for, not because any of it is actionable. See
> `archive/gridbook_lane_2026-09-02/README.md`.

# Lane eligibility in Gridbook's packaged runtime contract

**Status: SUPERSEDED as a specification (2026-08-30), kept as the record of
the ask.** Gridbook has now published a `lane_eligibility` table — commit
`30287aa`, runtime contract v12, schema `gridbook.lane-eligibility.v3` — and it
is **not** the shape proposed below. Read this file for *why* each field was
demanded; read the published contract for what the table actually is, and
`prismaquant/gridbook_lane_eligibility.py` for what PrismaQuant parses.

What the published v3 does differently, and where this document is now wrong:

- Cells are **platform-scoped** (`{schema, platforms, regimes, structures,
  cells}`), not a flat regime/lane map. A rule that names no platform matches
  nothing; there is no match-any.
- Rungs live in two disjoint vocabularies — `rungs` (CB codebook K) and
  `rungs_q256` (trellis body bits per 256 weights) — and the `kind`
  discriminator (`cb_product` / `tcq_trellis`) sits on the `formats[]` rows,
  not on the cells.
- Cells carry `qualification` (`compile_only` / `device_qualified`) and an
  `activation_contract` string, neither of which this proposal anticipated.
- The cell status set is closed at `backed | backed_with_serve_flag |
  fallback`. **There is no `unbacked` cell**: Gridbook declines to publish a
  negative claim, so *absence is the only negative signal*, and a rate or rung
  the table does not list must resolve `unattested`. That inverts this
  document's assumption that the table would enumerate refusals.

Still true, and still the reason the lane reads `unattested` today: PrismaQuant
pins Gridbook 0.8.11, whose contract is v4 and carries no table at all. Moving
the pins to a build that has one is release-gated (`30287aa` is untagged); see
ARCHITECTURE.md §9.2.1 for the exact prerequisites.

**Audience:** whoever implements the Gridbook side. This document specifies the
exact table to package and why each field is required.

**Owner:** PrismaQuant, campaign rule R3
(`/home/rob/dq-runs/dsv4-flash-0731/REBURN_CAMPAIGN_PLAN.md`).

## Why this is needed

PrismaQuant's principle 14 says a claim about another runtime is derived from a
machine-readable table that the runtime publishes, or refused. Principle 9 makes
serving-route status a gate input at export: a selected unit with no backed
route fails the export closed.

Neither rule can operate on lane routes today, because the packaged contract
publishes no lane-eligibility table. The measured consequence:

- The shipped DSv4 87 GB artifact carries 11 routed FP8-CB layers whose
  `gate_proj` and `up_proj` bind distinct learned codebooks. Gridbook's
  persistent-B prefill lane refuses per-role split books, so above the token
  threshold those layers take the announced expand-and-bridge route. The
  exporter priced and shipped them with no gate consuming that fact, and a user
  found it at serve time.
- PrismaQuant cannot fix this on its own side without transcribing Gridbook's
  kernel predicates into local constants, which principle 14 treats as an
  assertion and therefore refuses.

## What Gridbook 0.8.11 publishes today

`gridbook/runtime_contract.json`, schema `gridbook.runtime-contract.v4`, carries
exactly these top-level fields (measured on 0.8.10; the 0.8.11 file is
byte-identical, sha256 `0e2c32f3…`):

| Field | Contents |
|---|---|
| `schema`, `contract_version` | Contract identity. |
| `abi_features` | Three integer feature flags. |
| `quant_method` | Canonical and accepted `quant_method` strings. |
| `packing` | Vector dim, superblock size, index bytes per k, bit order. |
| `layout` | Layout versions, scale coding, per-grid scale plane bytes. |
| `formats` | Per family: `grid`, `mode`, `n_sub`, `rungs`, layout versions. |
| `producer_profiles` | Supported profile ids and loader modules. |

PrismaQuant already derives real facts from `formats`: a unit's payload family,
sub-table split, and rung legality come from that table rather than from a local
constant (`prismaquant/gridbook_lane_eligibility.py`, `load_published_formats`).

None of these fields determines which serving lane a unit's bytes ride. The
persistent-B role-split refusal, the fused mid-M `k % 4` law, the token-count
regime thresholds, and the operator serve flags all live in Gridbook source that
the contract does not summarize.

Do not read `abi_features.routed_moe_per_role_codebook_lut = 1` as a
persistent-B eligibility claim. It attests fused-mainloop LUT support, which is a
different question from whether the persistent-B lane accepts a role-split stack.

## What to add

Add a `lane_eligibility` object to `runtime_contract.json`.

`gridbook/runtime_contract.py::validate_runtime_contract` enforces an exact
top-level key set, so the new key is currently rejected rather than merely
missing. Adding it requires widening that set, which is a contract change: bump
`contract_version` to 5 and `schema` to `gridbook.runtime-contract.v5`.
PrismaQuant's pin records the schema string, so the bump is what tells the
producer that the table arrived.

### Schema

```json
"lane_eligibility": {
  "schema": "gridbook.lane-eligibility.v1",
  "regimes": ["decode", "batch"],
  "lanes": [
    {
      "id": "cb_moe_persistent_b",
      "regime": "batch",
      "structure": "routed_moe",
      "route_status": "backed",
      "requires_serve_flags": [],
      "predicates": [
        {"fact": "payload_family", "op": "in", "value": ["FP8_CB_K"]},
        {"fact": "role_split", "op": "equals", "value": false},
        {"fact": "k", "op": "multiple_of", "value": 4}
      ],
      "detail": "decode-in-mainloop prefill; one canonical book per stack"
    }
  ]
}
```

### Field reference

`regimes` lists the dispatch bands a unit passes through. Gridbook selects by
token count, not by phase, so name the bands after the predicate rather than
after prefill and decode. Two bands cover the current runtime: routed
`num_tokens <= 16` and dense `M <= 8` on one side, everything above on the other.

Each entry in `lanes` describes one route:

`id`
: Stable identifier for the kernel lane. PrismaQuant records it per unit, so a
  fallback is attributable to a named route rather than to a category.

`regime`
: Which band this rule applies in. Must appear in `regimes`.

`structure`
: Either `dense` or `routed_moe`. The two take different dispatch paths, so a
  rule that does not say which one it covers cannot be evaluated.

`route_status`
: One of `backed`, `backed_with_serve_flag`, `fallback`, or `unbacked`.
  Use `fallback` for a route that serves by an announced non-native path, such
  as expand-and-bridge. PrismaQuant records `fallback` per unit and refuses only
  when no band backs the unit at all, which is what keeps the measured DSv4
  state a recorded fact rather than a refusal.

`requires_serve_flags`
: The environment variables or command-line flags an operator must set for this
  route to fire, written as `NAME=value`. Required and non-empty when
  `route_status` is `backed_with_serve_flag`, and empty otherwise. A flag-gated
  route with an unnamed flag is unreachable, so PrismaQuant refuses that
  contract.

`predicates`
: The load-time conditions, all of which must hold. Each is
  `{"fact": ..., "op": ..., "value": ...}`. The facts are the closed set below;
  the operators are `equals`, `in`, `multiple_of`, `at_least`, and `at_most`.
  An unrecognized fact is a malformed contract, not a rule that is skipped: a
  predicate that no-ops would make a narrower rule read as unconditional.

`detail`
: Prose for a human reader. No gate reads it.

### The closed fact set

These are the structural facts a producer can know about the bytes it is about
to write, so they are the only facts a predicate may name.

| Fact | Meaning |
|---|---|
| `payload_family` | Family prefix, matching a `formats[].family` value. |
| `k` | Rung. `null` when the release does not instantiate the rung. |
| `n_sub` | Sub-table split, matching `formats[].n_sub`. |
| `role_split` | Whether an expert stack binds more than one codebook across its projections. |
| `in_features` | Reduction-dimension size. |
| `out_features` | Output-dimension size. |

`role_split` is the fact the DSv4 defect turned on, and the one no producer-side
structure carried. Adding it is the point of this proposal. The producer resolves
it only at export, after the per-`(qname, format)` codebook cells settle, which
is why the PrismaQuant gate runs at export rather than at allocation.

If a lane's real predicate needs a fact outside this set, propose the fact rather
than approximating it. A predicate PrismaQuant cannot evaluate must fail closed,
so an approximation silently narrows the backed set.

### Rules to publish first

The DSv4 case needs four rules, and they are the minimum useful table:

1. Routed CB GEMV decode, `regime: decode`, `structure: routed_moe`, backed,
   with the family and shape predicates the load gate applies.
2. Persistent-B prefill, `regime: batch`, `structure: routed_moe`, backed,
   predicated on `role_split == false`.
3. Expand-and-bridge prefill, `regime: batch`, `structure: routed_moe`,
   `fallback`, covering the stacks rule 2 rejects.
4. Dense mid-M, `regime: batch`, `structure: dense`, with
   `backed_with_serve_flag` and the flag named where the route is opt-in.

Rules are evaluated in order and the strongest match wins, so an explicit
`fallback` rule that covers what a `backed` rule rejects is how an announced
fallback stays distinguishable from a dead route. A band with no matching rule
resolves to `unbacked`, which is the fail-closed direction.

## What PrismaQuant does with it

The consumption side is implemented and needs no further change when the table
arrives:

- `prismaquant/gridbook_lane_eligibility.py` reads the table from the
  materialized copy of the packaged contract in
  `prismaquant/gridbook_runtime/`, keyed to the serving pin.
- `prismaquant/cb_route_status_gate.py` resolves every selected unit at export
  and refuses, records, or reports unattested.
- `prismaquant/serving_profile_specs/nvfp4_cb.json` declares which structural
  classes each lane consults. It declares the key; it never declares a verdict.

One constraint on whoever authors the table: a rule that carries no
`payload_family` predicate claims its `route_status` for **every** format that
reaches its `structure`. Per-unit resolution at export applies the rule's own
predicates, so an unpredicated rule is a deliberate whole-structure claim. The
serving-profile spec resolves each lane by structure alone and reports
`unit_dependent` for any rule that carries predicates, which means an
unpredicated rule is also the only kind that can announce a lane-level verdict.
Predicate on `payload_family` whenever the claim is narrower than the structure.

To flip PrismaQuant from unattested to attested, advance the serving pin, add the
release's packaged contract to `prismaquant/gridbook_runtime/`, and set the index
entry's `lane_eligibility` field to `present`. The pinned-compatibility CI job
asserts that the materialized copy equals the installed wheel's file, so the two
cannot drift.

## Related

- `prismaquant/CLAUDE.md`, principles 9, 12, and 14.
- `AGENTS.md:38`, which sanctions the pin and the contract as the only crossing
  of the repository boundary.
- `tests/test_cb_route_status_gate.py`, whose synthetic table is a fixture
  modelling the measured 0.8.10 behaviour and is not a shipped attestation.
