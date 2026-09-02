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
