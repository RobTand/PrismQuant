# Tessera serving-runtime pin

`tessera_serving_runtime_pin.json` is PrismaQuant's whole producer-side
boundary to Tessera's vLLM serving plugin, read by
`prismaquant/tessera_serving_runtime_pin.py`. PrismaQuant never vendors or
imports the serving half of that runtime; compatibility crosses the repository
boundary through this pin and through the contract Tessera packages
(`tessera/serving/runtime_contract.json`, read via `importlib.resources`).

**This pin is PENDING on purpose, and admission is fail-closed because of it.**
There is no Tessera release tag — cutting one is Rob's decision — so `commit`
and `version` are the conspicuous sentinels
`PENDING_TESSERA_RELEASE_COMMIT` / `PENDING_TESSERA_RELEASE_VERSION`,
`version_is_release` is `false`, and
`require_exact_tessera_runtime_release()` refuses them. That refusal is what
makes `tessera_render.tessera_lane_attested` answer False today: not an edit,
not a module constant, and not the absence of a route table — Tessera's
packaged contract publishes `device_qualified` cells for both families, and the
pin is what withholds them.

**Cutting the release is one reviewed commit** that resolves, together: the
three fields in this file, and the constants
`TESSERA_SERVING_RUNTIME_RELEASE_VERSION` /
`TESSERA_SERVING_RUNTIME_RELEASE_COMMIT` in
`prismaquant/tessera_serving_runtime_pin.py`. The reader requires the file to
equal the constants, so neither half admits anything alone.

**`serving_native_extensions` names what the plugin LOADS, not what it
executes — and it is DERIVED, not asserted.** Tessera's serving plugin
JIT-builds a CUDA decoder and loads it as `tessera_nvfp4_<identity>.so`
(`tessera/serving/ext.py`), and §7.4's reproducibility contract keys KL
comparability on whether a lane's `.so` was resident in the serving process:
two KLs are comparable only across serves whose native-extension residency
matches. Since Tessera contract v7 the runtime publishes that itself, in
`native_extensions`, as four values a consumer can act on — the
`module_name_prefix` the JIT load path itself passes to `cpp_extension.load`,
the `filename_glob` that produces (there is no exact basename: the module name
carries a build-identity hash), `match`, the name of the RULE a gate
applies (`basename_fnmatch`), and `when_unavailable`, what a serve does when
the library is absent (per residency mode: the substitute decoder it keeps
running on, or that there is no serve at all).  The fourth is what lets a
manifest say "the Tessera decoder was expected and is missing" instead of
recording the same absent-basename list as a stack with no Tessera in it
(PrismaQuant #142).

The chain is **contract → pin → fingerprint, with a refusal at each link**:

* `tessera_runtime_contract.require_pin_native_extensions_match_contract`
  refuses a pin whose rows are not the pinned contract's table, in both
  directions — a library the contract publishes and the pin omits makes the
  fingerprint go quietly short, and a library the pin invents is a claim about
  a runtime that does not load it. `contract_answer` carries the table, so a
  Tessera commit that renames the library re-stales the dev pin with a
  field-level diff instead of widening silently.
* `tools/serve_fingerprint.py` is stdlib-only by construction — it runs
  *inside* the serving container from a bootstrapped snapshot of five tool
  files and no package data — so it cannot read either file at runtime and
  carries the same rows as a constant;
  `tests/test_tessera_serve_fingerprint.py` refuses any disagreement, and also
  refuses a tool whose PREDICATE stops being the rule the contract names.
* the member is required, so the JSON and `tessera_serving_runtime_pin.py`'s
  member set move in one commit, the same rule the release constants carry.

History worth keeping: until 2026-09-03 this field was a hand-written
`serving_extension_basenames` with nothing here able to refuse it on drift
(principle 14 read backwards), and it was already wrong by one character —
`"tessera_nvfp4"` where the load path's constant is `"tessera_nvfp4_"`. The
fingerprint also matched it with a bare substring search over the whole mapped
path, which answers yes for
`/root/.cache/torch_extensions/tessera_nvfp4_9f2c/unrelated.so` and is not the
runtime's predicate. RobTand/tessera#28 published the table; PrismaQuant #133
consumed it.

**The repository is local-only today.** `repository` is the reviewed identity
the release will be cut from; the tree lives at `/home/rob/tessera` and has not
been pushed. The field is not a reachability claim, and nothing in this
repository fetches it.

**No wheel digest.** Gridbook's serving pin binds an exact reviewed wheel
SHA-256 because Gridbook is installed into a serving container from a published
archive. Tessera's plugin is installed from a source checkout
(`pip install --no-deps --no-build-isolation -e <tessera>`) and publishes no
wheel; asserting a digest for an archive that does not exist would be exactly
the hand-asserted claim principle 14 refuses. When Tessera publishes wheels, a
`wheel_sha256` member is added here and to the reader in one reviewed commit.

---

## The flip, once the tag is cut

`RobTand/tessera#17` is the blocker this pin names, and it is the only one:
Tessera's packaged contract already publishes `device_qualified` native cells
for every family, the `tessera` package is importable here, and the
`serving_native_extensions` table below already transcribes both rows the
runtime publishes (`tessera_nvfp4_`, `tessera_window_gemv`) — verified against
contract v16 / `tessera.lane-eligibility.v5` on 2026-09-04. What is missing is
the tag.

**Five values, two files, one commit.** `version` is not a choice: it is
`versions.tessera` in the contract the plugin packages, which reads `0.1.0`.
`commit` is what the tag points at.

```bash
TAG=v0.1.0
SHA=$(git -C /home/rob/tessera rev-list -n 1 "$TAG")
VER=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["versions"]["tessera"])' \
        /home/rob/tessera/src/tessera/serving/runtime_contract.json)

# 1/2 -- the JSON pin
python3 - "$SHA" "$VER" <<'PY'
import json, pathlib, sys
sha, ver = sys.argv[1], sys.argv[2]
p = pathlib.Path("prismaquant/tessera_runtime/tessera_serving_runtime_pin.json")
t = p.read_text(encoding="utf-8")
t = t.replace('"commit": "PENDING_TESSERA_RELEASE_COMMIT"', f'"commit": "{sha}"')
t = t.replace('"version": "PENDING_TESSERA_RELEASE_VERSION"', f'"version": "{ver}"')
t = t.replace('"version_is_release": false', '"version_is_release": true')
p.write_text(t, encoding="utf-8")
PY

# 2/2 -- the two reader constants, which the reader requires to EQUAL the pin
python3 - "$SHA" "$VER" <<'PY'
import pathlib, sys
sha, ver = sys.argv[1], sys.argv[2]
p = pathlib.Path("prismaquant/tessera_serving_runtime_pin.py")
t = p.read_text(encoding="utf-8")
old = ("TESSERA_SERVING_RUNTIME_RELEASE_VERSION = TESSERA_SERVING_RUNTIME_VERSION_PENDING\n"
       "TESSERA_SERVING_RUNTIME_RELEASE_COMMIT = TESSERA_SERVING_RUNTIME_COMMIT_PENDING")
assert old in t, "the constants already moved; review by hand"
p.write_text(t.replace(old, f'TESSERA_SERVING_RUNTIME_RELEASE_VERSION = "{ver}"\n'
                            f'TESSERA_SERVING_RUNTIME_RELEASE_COMMIT = "{sha}"'), encoding="utf-8")
PY
```

**Verify:**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. TRITON_CACHE_DIR=/home/rob/tmp/triton-cache \
  python -m pytest -q -p no:cacheprovider tests/test_tessera_release_pin_flip.py
```

`tests/test_tessera_release_pin_flip.py` is the merge guard: three of its four
tests FAIL while the sentinels stand, so a branch that prepares the flip cannot
be merged before the tag exists, and the same file is the flip's verification.
Its fourth test — the pin-vs-contract extension transcription — passes today
and is independent of the tag.

**Four existing tests assert the PENDING state and invert in the same commit.**
They are not collateral damage; their content *is* "no release tag exists", so
the reviewed release commit rewrites them:

| Test | What it asserts today | What it becomes |
|---|---|---|
| `test_tessera_lane_admission.py::test_the_tracked_pin_is_pending_and_the_release_gate_refuses_it` | tracked pin is both sentinels, `version_is_release is False`, `_release_pin_satisfied() is False` | tracked pin is the reviewed release, `_release_pin_satisfied() is True`; keep the "a sentinel pin is refused" half on a fixture |
| `test_tessera_lane_admission.py::test_a_pending_pin_cannot_be_marked_released` | flipping only `version_is_release` on the tracked payload raises "cannot be marked" | run the same structural rule on a synthetic PENDING payload, not on the tracked one |
| `test_tessera_export_lane.py::test_the_preflight_refuses_on_the_pending_pin_and_says_so` | `require_release_pin()` raises, message names `RobTand/tessera#17` | the preflight passes; keep the refusal text under the released-pin fixture's inverse |
| `test_tessera_formats.py::test_no_tessera_rung_is_producer_eligible_on_the_pinned_release` | no Tessera rung is producer-eligible, *because* of the pin | rungs the contract's cells cover become eligible; the "because of the pin" half moves to a sentinel fixture |

**Independent of the tag, and blocking today:** `TESSERA_DEV_PIN_ANSWER` in
`prismaquant/tessera_runtime_contract.py` was reviewed at Tessera contract v14
(commit `1221d2a`) and the installed contract is v16 — lane schema v4 → v5, a
per-cell `attested_on` block, and two NEW `routed_moe` E4M3 cells. Ten tests
fail on `main` for that reason alone. Re-reviewing that answer means deciding
whether PrismaQuant admits routed-MoE Tessera cells; it is a promotion
decision, not a pin flip, and it does not gate the tag.
