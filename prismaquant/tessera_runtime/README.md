# Tessera serving-runtime pin

`tessera_serving_runtime_pin.json` is PrismaQuant's whole producer-side
boundary to Tessera's vLLM serving plugin, read by
`prismaquant/tessera_serving_runtime_pin.py`. PrismaQuant never vendors or
imports the serving half of that runtime; compatibility crosses the repository
boundary through this pin and through the contract Tessera packages
(`tessera/serving/runtime_contract.json`, read via `importlib.resources`).

**This pin is an exact commit plus the contract's digest — not a release
tag.** Until 2026-09-04 `commit` and `version` were the sentinels
`PENDING_TESSERA_RELEASE_COMMIT` / `PENDING_TESSERA_RELEASE_VERSION` and
admission was fail-closed because no Tessera release tag existed. Rob retired
that: *"can we just pin prismaquant to latest version of tessera? then we won't
have to keep cutting releases."* "Latest" is read here as **an exact commit
plus the packaged contract's raw SHA-256**, never as a floating ref — `main`,
"the installed source tree" and "whatever imports" are precisely the failure
principle 14 exists to prevent.

Of the two, the **digest is the enforced half**. PrismaQuant cannot verify a
sibling checkout's git history from inside its own process, but it can hash the
contract bytes it is about to read, and those bytes are the only thing about
the runtime a gate here consumes. So `require_pinned_tessera_runtime()` refuses
whenever the installed `tessera/serving/runtime_contract.json` does not hash to
`contract_sha256` — which keeps the stray-checkout property the PENDING
sentinels used to provide: a Tessera source tree on `PYTHONPATH` that is not
the pinned one is refused exactly as before. The `commit` is recorded identity:
it says which reviewed tree those bytes came from, and `git` settles any
question the digest raises. The two are bound at review time by one pair of
commands, so they cannot become two independent assertions about one runtime:

```bash
SHA=$(git -C /home/rob/tessera rev-parse HEAD)
git -C /home/rob/tessera show "$SHA:src/tessera/serving/runtime_contract.json" | sha256sum
```

The pin below was first bound that way on 2026-09-04, when master was
`5acc2a6f` (contract v17), and re-bound on 2026-09-05 to Tessera master at
`b8b1cb38` — the merge of Tessera #313 (contract v21, lane schema v8). The
release checkout Rob named, `e78959ed`, carried contract v20; #313 is the one
published move past it (the routed-MoE cells' smoke, re-measured and now
`recorded`; PrismaQuant #198 option C). Re-check what is recorded against the
COMMIT, never against a later `HEAD`:

```bash
git -C /home/rob/tessera show b8b1cb38d581721d084860db1ae4a25b6afae50f:src/tessera/serving/runtime_contract.json | sha256sum
```

At the re-pin, master (`8446227d`, nine merges / 44 commits ahead of the pin,
through Tessera #315 and on) still packaged byte-identical contract bytes
(`sha256sum` of the file at that commit gives the digest above), so the pin
admits master too. A Tessera commit re-stales this pin only when it changes what the runtime
publishes. No tag names `b8b1cb38`, so `version_is_release` stays `false`.

**`version_is_release` is advisory.** Still required, still parsed, still
recorded, and still unable to be `true` over a PENDING commit — so it keeps
saying something true for an actual release. It gates nothing: a gate that
demanded it would re-impose the tag Rob just removed, and the immutability it
stood in for is now carried directly by the digest.

**Moving the pin is one reviewed commit** that resolves, together: `commit`,
`version` and `contract_sha256` in this file, and
`TESSERA_SERVING_RUNTIME_PINNED_VERSION` /
`TESSERA_SERVING_RUNTIME_PINNED_COMMIT` /
`TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256` in
`prismaquant/tessera_serving_runtime_pin.py`. The reader requires the file to
equal the constants, so neither half admits anything alone.

**A consequence, by design.** A developer checkout of Tessera that has moved
past the pin makes PrismaQuant's Tessera tests red: the installed contract is
not the pinned contract, and fail-closed is the whole point. The fix is
environmental — install Tessera at the pinned commit — never a check that reads
whatever is installed.

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

**`repository` is the reviewed identity of the runtime**, not a reachability
claim: nothing in this repository fetches it, and the pin is satisfied by the
contract bytes an installed Tessera packages, never by an origin.

**No wheel digest.** Gridbook's serving pin binds an exact reviewed wheel
SHA-256 because Gridbook is installed into a serving container from a published
archive. Tessera's plugin is installed from a source checkout
(`pip install --no-deps --no-build-isolation -e <tessera>`) and publishes no
wheel; asserting a digest for an archive that does not exist would be exactly
the hand-asserted claim principle 14 refuses. What it DOES bind, since
2026-09-04, is `contract_sha256` — the digest of the one Tessera artifact that
both exists and is read by a gate on this side. When Tessera publishes wheels, a
`wheel_sha256` member joins it here and in the reader in one reviewed commit.

---

## Moving the pin

Verified against the installed Tessera on 2026-09-05:

```
commit           b8b1cb38d581721d084860db1ae4a25b6afae50f
contract_sha256  34f1da7977f1aa155cd2ff18b584e5c35a6089b648bb4a51d449fc25082a2c3e
versions.tessera 0.1.0
contract_version 21
lane schema      tessera.lane-eligibility.v8
```

Five values, two files, one commit. Resolve the new commit, digest and version
first — from one `git` object, so they name the same tree:

```bash
TS=/home/rob/tessera
SHA=$(git -C "$TS" rev-parse HEAD)
BLOB="$SHA:src/tessera/serving/runtime_contract.json"
DIGEST=$(git -C "$TS" show "$BLOB" | sha256sum | cut -d' ' -f1)
VER=$(git -C "$TS" show "$BLOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['versions']['tessera'])")
echo "$SHA $DIGEST $VER"
```

Then edit both halves in one commit — the JSON:

```bash
python3 - "$SHA" "$DIGEST" "$VER" <<'EDIT_JSON'
import json, sys, pathlib
sha, digest, ver = sys.argv[1:4]
p = pathlib.Path("prismaquant/tessera_runtime/tessera_serving_runtime_pin.json")
d = json.loads(p.read_text())
d["commit"], d["contract_sha256"], d["version"] = sha, digest, ver
p.write_text(json.dumps(d, indent=2) + "\n")
EDIT_JSON
```

...and the three reader constants, which the reader requires to EQUAL the pin:

```bash
python3 - "$SHA" "$DIGEST" "$VER" <<'EDIT_CONSTS'
import re, sys, pathlib
sha, digest, ver = sys.argv[1:4]
p = pathlib.Path("prismaquant/tessera_serving_runtime_pin.py")
s = p.read_text()
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_COMMIT = \(\n    "[^"]*"\n\)',
           'TESSERA_SERVING_RUNTIME_PINNED_COMMIT = (\n    "%s"\n)' % sha, s, count=1)
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 = \(\n    "[^"]*"\n\)',
           'TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 = (\n    "%s"\n)' % digest, s, count=1)
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_VERSION = "[^"]*"',
           'TESSERA_SERVING_RUNTIME_PINNED_VERSION = "%s"' % ver, s, count=1)
p.write_text(s)
EDIT_CONSTS
```

Verify — ONE shell segment, and the empty `CUDA_VISIBLE_DEVICES` prefix is
required by this repository's pool hook:

```bash
cd /home/rob/prismaquant && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  TRITON_CACHE_DIR=/home/rob/tmp/triton-cache nice -n 10 \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_tessera_serving_pin.py tests/test_tessera_lane_v6.py \
  tests/test_tessera_lane_admission.py tests/test_tessera_export_lane.py
```

**A moved contract is a re-review, not a bump.** The development pin
(`TESSERA_DEV_PIN_COMMIT` / `TESSERA_DEV_PIN_CONTRACT_SHA256` /
`TESSERA_DEV_PIN_ANSWER` in `prismaquant/tessera_runtime_contract.py`) names
the SAME Tessera object, and `tests/test_tessera_serving_pin.py` refuses a
drift between the two. So moving the serving pin means regenerating the
dev-pin answer in the same commit, and the git diff of that literal IS the
review: it shows every value an admission gate decides on, including each
cell's `evidence` block. That is the mechanism which keeps promoting a
`routed_moe` cell a human decision under principle 9, rather than a
consequence of Tessera merging a PR.

**Tests whose content is "the pin refuses everything" are spent.** The
2026-09-04 commit deleted `tests/test_tessera_release_pin_flip.py` (its whole
subject was a tag that no longer gates anything) and inverted the four
PENDING-asserting tests in `test_tessera_lane_admission.py`,
`test_tessera_export_lane.py`, `test_tessera_formats.py` and
`test_tessera_contract_v4.py`. What replaced them asserts the property the tag
stood in for: an exact commit, the installed contract's digest, and a refusal
for any other Tessera.
