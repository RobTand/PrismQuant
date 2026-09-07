# Returned PrismaBuild fixes: PR 308, PR 310, PR 311, PR 312

Review scope: complete diffs from common base `dc083644646144e995863a64d9eb31a995dd6269` to PR 308 head `ba476cc4a2498fc12624d2802cdc0f5d10439f53` and PR 311 head `3b62b3690a7384ba925bbc0e75ab72e7d3a1a9b5`, their associated tests, and the terminal/CAS evidence listed in `delegated-validation.json`. This was a read-only review; no tests were repeated because the inspected evidence addressed the original findings and raised no new concrete concern.

## PR 308 — Preserve existing Netdata jobs during metrics installation

**READY. No blocker found against issue 304.**

The installer now parses the existing collector YAML before service changes, requires `jobs` to be a sequence, preserves unrelated jobs, treats an existing PrismaBuild job idempotently, refuses malformed/ambiguous YAML and symlink destinations, writes a unique recovery copy while preserving mode/ownership, and propagates restart/inactive-service failures. PyYAML is declared for development tests and its installer-time absence has an actionable failure. Tests cover the real installer through temporary paths and stubbed privileged commands, including duplicate keys/jobs, aliases/merges/tags, ambiguous numeric forms, backup behavior, idempotence, missing dependency, symlinks, and service failures.

Validation action `4166faea755cb8fdd927e77f521c062930a6f68e97cf23cdad2309ed92a1542d` finished `executed` on `dl380g10` with return code 0. Its immutable payload `1785b7ebc39db771efd5790dd6c7e619bfcf5d0a3d89409638367d72710497e0` contains `58 passed, 4 subtests passed in 2.76s`. The receipt's embedded canonical SHA-256 is `a0451208dbd13287cebc7bcb7909063f015d710b6f0cb5453fd249537e2f09c8`; the payload bytes hash to the CAS filename. `git diff --check` is clean.

Evidence limit: the suite uses a temporary filesystem and stubbed privileged commands. Live Netdata collector adoption and endpoint scrapeability remain deployment verification, as the PR already states; they do not block the source fix.

## PR 311 — Make mount-latency Netdata entry point executable and emit protocol

**READY. No blocker found against issue 306.**

The file mode is executable, and a positional Netdata update interval now selects external-plugin protocol mode unless one-shot or JSON output is explicitly requested. Protocol-mode records default to `/var/lib/netdata/prismabuild`; explicit record and worker-lock paths remain supported. Documentation correctly identifies Netdata command-options configuration and the access arrangement required when the Netdata UID cannot traverse the worker's private directory. Subprocess tests exercise direct executable invocation, the installed Python symlink argument shape, CHART-before-sample protocol output, one-shot behavior, the record default, and explicit overrides.

Validation action `e50db8c7a2275fc2241cbaebf04cc503aa7ebaa629c61e7a27679f28dbbbf343` finished `executed` on `dl380g10` with return code 0. Its immutable payload `9428f9373b73b9410911cf11ef4d5868b91f1b82f7d991c07fa8163be3f5b7fb` contains `31 passed, 9 warnings in 8.17s`. The receipt's embedded canonical SHA-256 is `bb89fcfe09c8182047101f3c5ec7f080cb86af48755f12217eedc7e210b47cd3`; the payload bytes hash to the CAS filename. The warnings are the existing Python 3.12 `os.fork()` warning from a multithreaded pytest-xdist process. `git diff --check` is clean.

Evidence limit: this validates the executable/protocol contract with local temporary probe and record paths. Publication, installed-plugin adoption, and access to the production worker lock directory remain operational verification, as the PR already states; they do not block the source fix.

## PR 310 — Retain contradictory-holder claims

**READY at c881eb4e80a61a0048359e25162ae36b8a6fb1d0. No blocker against PB301.** The typed AmbiguousClaimHolder exception is distinct from absent evidence; the reaper handles it before cleanup, preserves the exact claim and every reservation, reports hosts/key, and continues other claims. Withdrawal propagates its existing contract-error refusal before recording a fresh cancellation. Tests cover both empty contradictory-holder directories and actual CPU tokens, preservation, independent recovery, and resumption after removing contradictory evidence. The conservative holder predicate is preserved rather than redefined.

Independently checked terminal statuses/rc, immutable attempt stdout/stderr hashes, canonical CAS receipt hashes, and payload hashes/byte sizes for red 359263322668 (4 failed), focused a22d88c7a2d4 (23 passed), integration 6f64c4567974 (175 passed), and final focused 8a85c9b44e44 (24 passed), with no skips in green runs. Sealed Git bundles were imported into an isolated evidence repository: the integration pool.py is byte-identical to final PR production code; the later 24-test bundle also matches both changed test files. The contradiction remains a fabricated recovery-invariant state, not a production incident claim.

Evidence: /home/rob/tmp/codex-pb-maintenance-evidence/301-review-independent.json.

## PR 312 — Cover no-scope cleanup failures

**READY at 77b560f896bcfa86c5655b0bd531acf39bb830a4. No blocker against PB302.** The marker stat and ownership/hostname checks now lie inside the cleanup guard; unexpected Docker and scope-creation errors return incomplete cleanup while BaseException subclasses still propagate. The tests drive the real finish and reaper paths separately, asserting original outcome persistence, held tokens, pending-retry accounting, and no READY/DONE publication on uncertain cleanup.

Actual red action c76d164f7cc5e06d8cbe38d47dbed38f7661dec46f516bd66f42b59d6109f466 on sparky: 6 failed, 1 passed. Control9a632f54d30974420cee1fbbe08a590d8e44a46663a9349a9b7033729642ec1c: 2874 passed, 3 skipped, 4 subtests. Branch1ec88e3f262e93c8642265054ea17524da79cc618a60f58db2f80fbd38ae7d90: 2881 passed, 3 skipped, 4 subtests. Green terminal rc0 and actual immutable log hashes, canonical receipt hashes, and payload hashes/bytes checked. Imported sealed bundle confirms branch pool.py and added test file match final77b560f; control and red pool.py matchdc08364.

Branch receipt SHA: c95b337f70244429786949608cc52818342b7365992d298d9be5cdab1db8a491. Payload: /mnt/shared/prismabuild-fleet/cas/blobs/92/920ca96cf256bccbdc96653d3d3918fbae52c22d15abc827b0d4393ef873ddb0. Evidence: /home/rob/tmp/codex-pb-maintenance-evidence/302-review-independent.json.

## Final source attribution and disposition

Additional sealed-bundle comparison for PR308 and311 confirms both modified production scripts, test files, and executable modes match their reviewed final heads. See /home/rob/tmp/codex-pb-maintenance-evidence/304-306-review-source-identity.json. All four are ready for root's final merge review; no merge was performed. No repeat tests were run because existing correctly attributed results answer the findings. Dirty main was preserved. Runtime publication and installed collector/plugin adoption remain to verify operationally.
