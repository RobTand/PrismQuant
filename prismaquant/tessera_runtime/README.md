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
executes -- and it is DERIVED.** Tessera's serving plugin JIT-builds a CUDA
decoder and loads it as `tessera_nvfp4_<build identity>.so`
(`tessera/serving/ext.py`), and §7.4's reproducibility contract keys KL
comparability on whether a lane's `.so` was resident in the serving process. The
module name always carries a source/toolchain/arch hash, so **no exact basename
exists to publish**: the runtime publishes a prefix, a glob, and the name of the
rule that turns the glob into a decision.

Until 2026-09-03 this file hand-wrote `serving_extension_basenames` and
`tools/serve_fingerprint.py` hand-copied it -- a claim about Tessera's runtime,
maintained here, that nothing on this side could refuse on drift. Tessera
contract v7 (RobTand/tessera#28) publishes `native_extensions`, so the member is
now produced by `tessera_runtime_contract.native_extension_pin_payload` from
that table and refused against it by `require_serving_pin_matches_contract`,
which `load_tessera_contract` calls on every read. The three fields are also in
`contract_answer`, so a Tessera rename refuses the dev pin with a field-level
diff rather than silently turning a Tessera serve back into "no lane extension
resident". Changing the member set moved the pin schema to `.v2`: a pin does not
move by halves.

The last link is still a constant. `tools/serve_fingerprint.py` is stdlib-only
by construction, is invoked by path inside the serving container, and its
`SERVER_ENV_ALLOWLIST` requires PYTHONPATH to be absent, so it cannot import
this package; the gold lane's bound source closure is five `.py` files and no
package data, so this JSON is not proven to travel with it either.
`tests/test_tessera_serve_fingerprint.py` therefore refuses the tool's
*behaviour* against the runtime's table, and binding the pin into that closure
so the tool reads it in-container is issue #137.

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
