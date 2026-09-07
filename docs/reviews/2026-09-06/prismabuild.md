# PrismaBuild review 2026-09-06 (America/New_York)

Reviewed range: `d39cb006410907a6d036299084c573ab4d0d732e..645f440c742d6011caf22b3b64236e07f4f6687e`. Cutoff: 2026-09-07 02:50Z. Main checkout was dirty and untouched; review snapshot `/home/rob/tmp/pb-review-20260906`. 53 landed changes fall within the supplied window; earlier PR metadata in the inventory is context only.

## Confirmed core findings

- **P1 PB301** https://github.com/RobTand/prismabuild/issues/301 — helper returns None on contradictory holder ledgers, but reaper falls back to its own ledger and requeues anyway. Existing contradiction test covers only helper. Reproduced on fabricated two-holder state; not observed in fleet.
- **P1 PB302** https://github.com/RobTand/prismabuild/issues/302 — no-scope Docker marker stat sits outside the new broad cleanup handler. Injected ESTALE escapes instead of retaining a pending result. No new RDMA failure inferred.

All findings are off PB234, explicitly assigned in issue prose to claude-triage (there is no claude-triage GitHub label). No production change, commit, or merge.

- **P1 PB303** https://github.com/RobTand/prismabuild/issues/303 — incomplete cgroup fallback samples enter recent-core rate counters before completeness gating, yielding false zero with measured-job coverage. Source independently read; PB calculation receipt and payload verified.
- **P2 PB304** https://github.com/RobTand/prismabuild/issues/304 — installer appends a second YAML jobs mapping to an existing Netdata collector config. Deterministic source path; no live config changed.
- **P2 PB306** https://github.com/RobTand/prismabuild/issues/306 — documented direct Netdata plugin symlink targets a non-executable 0444 published file and would emit human output without --netdata even if executable. Pure CLI protocol run through PB confirms mode mismatch.

## Core validation

New targeted regressions only: action `9b95878c9c7d3a9ef751b1ff73f9158b619a7bd1f81946f5df38dac143821416`, sparky, CPU-only 1 CPU / 2 GiB with native threads 1, portable placement. Terminal and immutable attempt reviewed; rc=1 and exactly 2 expected regression failures in 0.95s. No successful CAS receipt from failed command. `pb-review-regression-result.json` is sanitized log evidence, and the two tests are retained in review snapshot for triage.

Existing committed suite.json: all four real terminal records read, executed rc0 on dl380g10. 2747 passed, 3 skipped, 2 xfailed. All four CAS payload hashes match their receipts. These requests seal dirty snapshots parented at `768563a` and executed using runtime `aba92a3`; they cannot qualify later changes or final tip. See `pb-existing-suite-receipts.json`. Later PR #291/#296/#299 state 2822/2867/2871 passes, but their bodies do not provide action keys. Those are author-reported results, not independently resolved receipts.

## PB234 boundary

The diff does not complete PB234. PR252 preserves budget for no-lease/no-attempt claims, but existing stale leases still charge attempts, default max_attempts remains one, and execution wall-time timeout remains enforced. PR245 adds progress observations, not an autonomous progress-based recovery state machine. Late claimant launch after reaping is explicitly acknowledged in PR252; no claim-fencing repair is included in this range. The user’s latest issue comment permits submitter redispatch and asks primarily for knowing dead versus live, so do not treat the original issue title as the latest authority.

PB266 already carries updated RDMA probe evidence and recommends waiting for recurrence before restructuring shared bookkeeping. That probe does not establish current state-recovery-fault latency. PB217 and PB298 are existing tracked work; PB300 memory clamp race appeared during review and was not duplicated.

## Coverage ledger

| PR | Change | Review surface | Result |
|---|---|---|---|
| #198 | Include required agent policies in sealed runtime generations | fleet tools review | reviewed; see fleet tools report |
| #199 | Automate recurring PrismaBuild issue triage and repair | fleet tools review | reviewed; see fleet tools report |
| #200 | Find Rob’s installed GitHub CLI in the maintenance service | fleet tools review | reviewed; see fleet tools report |
| #203 | Require issue-linked pull requests for main | fleet tools review | reviewed; see fleet tools report |
| #204 | Dispatch full Tessera exports as verified layer campaigns | client/dispatch review | reviewed; no confirmed defect, see client report |
| #207 | Give the supervisor a boot-time owner, not a five-minute poll | fleet tools review | reviewed; see fleet tools report |
| #211 | Read a vanished worker offer as a box that left, not as a failure | core pool/admission | reviewed, no additional defect found |
| #210 | Accept a sidebar-linked issue, and pin the check's action properly | fleet tools review | reviewed; see fleet tools report |
| #218 | Write the fleet measurement discipline into the agent execution policy | fleet tools review | reviewed; see fleet tools report |
| #220 | Record the vLLM serving exemption, and that its load is still external | fleet tools review | reviewed; see fleet tools report |
| #228 | Widen the vLLM exemption to everything vLLM | fleet tools review | reviewed; see fleet tools report |
| #221 | Refuse the instance, not the mention, in a script's arguments | fleet tools review | reviewed; see fleet tools report |
| #226 | Sweep once per heartbeat per box, not once per poll per loop (#205) | core pool/admission | reviewed, no additional defect found |
| #235 | Reading the process table is not running what it finds | fleet tools review | reviewed; see fleet tools report |
| #229 | Ask a shard whether pytest ran, and size the worker to the box it was given | client/dispatch review | reviewed; no confirmed defect, see client report |
| #238 | Import the published tools when a command runs, not when the module loads (#237) | client/dispatch review | reviewed; no confirmed defect, see client report |
| #240 | Read a here-document body as shell only when a shell reads it | fleet tools review | reviewed; see fleet tools report |
| #242 | Green means the shard ran, and a cache hit is a result (#241) | client/dispatch review | reviewed; no confirmed defect, see client report |
| #243 | A rename follows the box, and the offer follows the rename (#236) | fleet tools review | reviewed; see fleet tools report |
| #216 | Let the ready sweeps outlive an entry, and say why the reaper may not | core pool/admission | reviewed, no additional defect found |
| #245 | Report whether a claim is moving, not only whether it reported | client/dispatch review | reviewed; no confirmed defect, see client report |
| #248 | Run the queue exporter, and let Netdata keep what it says | fleet tools review | reviewed; see fleet tools report |
| #251 | Stop throwing away the freshest sample because another box stamped it | core pool/admission | reviewed, no additional defect found |
| #253 | Say where Netdata actually files these, before someone hunts for them | fleet tools review | reviewed; see fleet tools report |
| #249 | Ask the finish guard again on the read the reaper acts from | core pool/admission | reviewed, no additional defect found |
| #250 | Report a lost claim against the box that held it | client/dispatch review | reviewed; no confirmed defect, see client report |
| #260 | Accept pytest-subtests parts in pbtest's terminal-summary grammar | client/dispatch review | reviewed; no confirmed defect, see client report |
| #252 | Release a claim that never started instead of charging it an attempt | core pool/admission | reviewed, PB234 boundary noted |
| #246 | Size worker loops on claims held, not on a backlog nobody can attribute (#231) | fleet tools review | reviewed; see fleet tools report |
| #255 | Measure the shared filesystem the fleet runs on (#230) | fleet tools review | reviewed; see fleet tools report |
| #257 | Read an attached -m value as the runner it names | fleet tools review | reviewed; see fleet tools report |
| #258 | Leave a here-document inside a substitution to the pass that owns it | fleet tools review | reviewed; see fleet tools report |
| #259 | Give a dead worker name a supported way out of workers/ | fleet tools review | reviewed; see fleet tools report |
| #268 | Pad the lock key, or read healthy on the only box that stalled (#264) | core pool/admission | reviewed, no additional defect found |
| #267 | Refuse box admission instead of waiting on a peer's shared-mount latency (#233) | core pool/admission | reviewed, no additional defect found |
| #270 | Bound the admission tests so a blocking regression fails instead of hanging | core pool/admission | reviewed, no additional defect found |
| #276 | Name the admission holder by device and inode, not by inode alone | core pool/admission | reviewed, no additional defect found |
| #279 | Correct two docs that still say admission queues behind a slow mount | core pool/admission | reviewed, no additional defect found |
| #273 | Release a lost claim's tokens on the box that committed them | core pool/admission | reviewed, no additional defect found |
| #274 | Name the box the work was on in pbstatus and pbwait | client/dispatch review | reviewed; no confirmed defect, see client report |
| #283 | Name the holder by inode, and let the device only break a tie (#282) | core pool/admission | reviewed, no additional defect found |
| #277 | Scan the body a -c switch is about to run | fleet tools review | reviewed; see fleet tools report |
| #280 | Read the release count back out where someone is looking | client/dispatch review | reviewed; no confirmed defect, see client report |
| #284 | Keep the suite out of the box's own admission directory (#265) | core pool/admission | reviewed, no additional defect found |
| #285 | Repair an admission-directory mode this uid owns (#281) | core pool/admission | reviewed, no additional defect found |
| #287 | Count a box's worker loops on the offer it writes (#254) | fleet tools review | reviewed; see fleet tools report |
| #289 | Stop a failed learning hook from costing an action its lease (#286) | core pool/admission | reviewed, no additional defect found |
| #290 | Resolve the holder in withdraw the way the reaper does (#271) | core pool/admission | reviewed, no additional defect found |
| #291 | Name the winner of a claim race from the ledger, not the marker (#272) | core pool/admission | reviewed; PB301 follow-up |
| #294 | Refuse work the chosen box could not open the runtime for (#292) | client/dispatch review | reviewed; no confirmed defect, see client report |
| #295 | Say what deadline governs, at submit and in the receipt (#293) | client/dispatch review | reviewed; no confirmed defect, see client report |
| #296 | Fail the token-return gate closed, and let the retry loop raise its hand (#288) | core pool/admission | reviewed; PB302 follow-up |
| #299 | Retire a withdrawal decision that no longer names the live run (#297) | core pool/admission | reviewed, no additional defect found |

## Additional independently checked evidence

`pb-subreview-receipts.json` records terminal rc0, receipt identity, and matching payload hashes for PB303 calculation (`e1016cc9abed`), PB306 CLI protocol (`a77982a5f573`), and dismissed source-mount-overlap hypothesis (`03caa2bf02ae`), all on sparky. No speed or new-topology NFS claim is drawn. Docker refuses an exact duplicate mount point (negative action `1a4bf756b0c1`); with nested source RO and ancestor output RW, the real uid1000 invocation retained source read-only protection. Directory/symlink population in the adapter was inspected but not elevated without a demonstrated current producer/consumer path to unverified served bytes.

## Surface reports and disposition

- `prismabuild-client-dispatch-review.md`: client/Tessera source coverage, PR evidence limits, dismissed hypotheses.
- `prismabuild-fleet-tools-review.md`: fleet tooling coverage and PB303/PB304/PB306 evidence.
- The main checkout remained dirty with its original owner’s edits; no production code was changed. Detached review snapshots and the two failing tests are retained as bounded, attributable evidence for triage. No merges.
