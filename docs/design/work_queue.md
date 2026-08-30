# The shared work queue (`pqwork`)

**Status: LIVE since 2026-08-29.** Workers running on sparky and sparklina.
Supersedes the SLURM+Dagster stack in `prismabuild.md`; that document's
problem statement still stands, its chosen stack does not.

## What it is

A directory of JSON files on `/mnt/shared` and a worker loop per box.
Agents *enqueue* computational work; each box *pulls* the items it is
eligible for. Nothing dispatches.

```
/mnt/shared/pq-queue/
  bin/pqwork.py       the deployed tool (installed from tools/pqwork.py)
  ready/<id>.json     queued
  claimed/<id>.json   running, with <id>.lease alongside
  done/<id>.json      complete
  failed/<id>.json    complete and wrong
  logs/<id>.log       stdout+stderr, appended across attempts
  reaped/<id>.lease   archived orphan lease evidence
  reserved/<host>.gpu legacy single-host external GPU hold
  reserved/<host>.mem legacy single-host external memory hold
  reserved/reservations.v2.json atomic multi-host reservation sets
  workers/<host>.json live PID, heartbeat, and loaded source SHA-256
```

## Why it exists

Every job used to be dispatched by hand from an agent session: ssh, then
`systemd-run`, then watch. That makes the agent the scheduler, and on
2026-08-29 it failed in all three of the available ways in one evening —
sparklina idle at 4.5 W for ~20 minutes with work waiting, wsl-gpu at load
0.00 while the board said "queued", and three jobs dying with nothing
rescheduled when a session hit its limit at 20:40 ET.

Inverting the direction fixes all three at once. An agent that goes away
stops new work from being *created*; it can no longer stop queued work from
*running*.

## Using it

```bash
Q=/mnt/shared/pq-queue/bin/pqwork.py

# Enqueue. --receipt is both the completion test and the skip-if-done gate.
python3 $Q enqueue --id glm-probe-4b \
  --desc "4B probe, unified sweep" \
  --cmd  "bash run-probe.sh --dataset wikitext" \
  --cwd  /home/rob/dq-runs/glm53 \
  --gpu --host sparky \
  --receipt /home/rob/dq-runs/glm53/artifacts/probe.pkl \
  --priority 80

# Sequence work by receipt, not by a scheduler DAG.
python3 $Q enqueue --id glm-cost-4b --cmd "..." \
  --after /home/rob/dq-runs/glm53/artifacts/probe.pkl \
  --receipt /home/rob/dq-runs/glm53/artifacts/cost.pkl

python3 $Q status          # the board
python3 $Q requeue <id>    # failed -> ready
python3 $Q cancel <id>     # ready/claimed -> failed
```

## The seven properties that make it safe

1. **Claiming is `rename()`.** Atomic on NFSv4; the loser gets `ENOENT` and
   moves on. The final capacity recheck, rename, and lease write run under
   the same short NFS namespace guard as external reservation acquisition,
   so a queue claim and a conflicting reservation cannot both publish.
   `flock` over NFS is version-dependent and was not used.
   Verified across boxes: 6 items, 8 racing workers on two machines, 6
   executions, 0 double-claims.
2. **A claim is a lease.** The holder rewrites its timestamp every 30 s; any
   worker may requeue a claim stale for 300 s. This is the recovery path for
   a box that dies mid-job — without it the queue develops permanent holes,
   which is the original bug one level down. The same runner moves an orphan
   `claimed/*.lease` into `reaped/` as soon as its job is terminal, or after
   the heartbeat TTL when no item record remains; stale sidecars therefore
   need no manual cleanup.
3. **Completion is the receipt, never the exit code or unit state.** A
   command that exits 0 without producing its declared receipt lands in
   `failed/`. This encodes a real incident: a collected systemd unit reported
   success for work that had not happened.

   **And the receipt must be produced by the work, not by the wrapper.**
   Writing `<command> && touch <receipt>` recreates the exact bug the receipt
   exists to prevent: it attests that the command exited 0, which is what the
   exit code already said. First tried on 2026-08-29 with queued Codex jobs;
   Codex's sandbox failed, Codex reported the blockage as its *answer* and
   exited 0, `touch` fired, and two items sat in `done/` having done nothing.
   The fix is a receipt only the work itself can write -- the task's own
   output file, or a value the task must compute -- and a gate that checks
   it (`...; test -s <receipt>`), never `&&`.
4. **Receipt-gating makes requeue safe.** An item whose receipt already
   exists skips without executing, so a stale-lease requeue cannot
   double-run finished work, and a whole campaign can be re-enqueued after a
   partial run without redoing its finished arms.
5. **Out-of-queue work declares itself — memory as well as GPU slots.**
   Legacy single-host work uses `reserved/<host>.gpu` and
   `reserved/<host>.mem`. A gang launch uses one entry in
   `reserved/reservations.v2.json`; one same-directory rename publishes or
   releases every host and resource together. Acquisition and the worker's
   final admission share the NFS-safe admission guard, re-read claims and
   reservations inside it, and return temporary-failure status 75 without
   publishing anything when capacity was lost. This is what rules out both
   partial box-A holds and the worker race between an earlier capacity read
   and its later claim.

   A gated actor owns **nothing while it waits**. It calls `reserve-set
   acquire` only after the gate passes; if it loses admission, it returns to
   the gate. Releases and verification require the 0600 owner capability
   created during acquisition, so another actor cannot release the set by
   guessing its public ID. A malformed ledger fails closed. Timestamp-based
   stealing of the admission guard is intentionally forbidden because it is
   not fencing; an old paused holder could otherwise resume and overwrite a
   successor.

   The first successful claim for a host also records its GPU and memory
   capacities in the ledger. Releases retain that registry, and later actors
   must supply the same values; an actor cannot manufacture headroom by
   inflating its own capacity argument.

   These declarations hold resources for jobs the queue cannot see. The
   memory half exists because
   of a failure: on 2026-08-30 a hand-launched job took sparklina from 68 GB
   to 4 GB in about 100 seconds, filled 16 GB of swap, and left the box
   unreachable over SSH for five minutes. The governor could not evict it --
   it was not a queue item -- and no admission policy over MemAvailable alone
   could have prevented it, because MemAvailable does not say how much more a
   neighbour intends to take. Only a declaration can.

   Two things that failure also settled. **Swap makes it worse, not better:**
   MemAvailable sat pinned at 4.0 GB for four minutes and the OOM killer
   never fired, because swap absorbed the pressure, so the box never
   self-healed. Without swap the kernel kills one process in seconds and
   stays reachable. And **aggressive admission is only safe when the governor
   can evict everything on the box**; where a non-evictable neighbour exists,
   admission has to be conservative.

   The alternative — inferring "the box is busy" from telemetry — is
   unavailable here: on GB10 `gpu_utilization` reads 96% for a stalled kernel
   and a saturated one alike, so such a check would be a guess wearing a
   measurement's clothes.

6. **A declared memory budget is enforced, not trusted.** An item with
   `--mem-gb N` runs inside a transient systemd unit with `MemoryMax=NG` and
   `MemorySwapMax=0`, so a job that outgrows its own declaration is killed by
   its own cgroup. This is the half the governor cannot do. A governor picks
   a victim by policy and can only reach jobs it started; a cgroup limit
   reaches the one process that is actually over budget, and reaches it
   whoever started it. It is also the shape the PrismaBuild spec asked for —
   allocation-time enforcement rather than a reactive monitor — because a
   userspace memory monitor on unified memory is the recorded Ray failure,
   where the monitor killed healthy ranks.

   Measured on sparky: a 1 GB cap against a runaway allocator gives
   `memory.events oom-kill` and exit 9 with nothing else on the box
   disturbed; an under-budget job completes normally through the same path,
   keeping its `cwd`, its environment and its log. Two consequences worth
   knowing:

   * **A budget-exceeded job fails once and is not retried.** The same
     command under the same budget dies the same way, so the outcome names
     the fix (`re-enqueue with a larger --mem-gb`) instead of spending two
     more attempts reproducing it.
   * **Undeclared jobs stay on the plain child path**, uncapped and
     governor-evictable. The cap is what declaring buys; an undeclared
     appetite can only be managed reactively, which is the weaker tool and
     should read as the weaker option.

   The scope form (`systemd-run --user --scope`) would have been more
   convenient — the job stays a direct child, so stdout inherits and
   `killpg` still evicts. It was tried first and hangs under the cap rather
   than killing, twice. The service form enforces, so eviction of a capped
   job goes through `systemctl --user stop` instead; that path is tested to
   terminate in well under a second and leave no orphaned children.

7. **Eviction measures the trend, attributes the event, and ages work into
   protection.** The box-wide trigger is the median of every consecutive
   `MemAvailable` slope in the 30-second window, with at least three valid
   intervals required. A single endpoint step therefore cannot trigger an
   eviction. The median is deliberate: one interior outlier creates one steep
   fall and one steep recovery, while an endpoint outlier creates one bad
   interval; neither can dominate a sustained majority trend. A genuine fast
   fall across the window still projects through the reserve and fires. lina
   had swap free during the incident, so its configured 60-second horizon was
   the branch in force; the earlier 120-second reconstruction was wrong.

   Victim selection is still lowest-priority-first, but newest-first is gone.
   Within a priority it chooses the least cumulative **measured sunk runtime**,
   including runtime discarded by prior evictions. Restarting an item no
   longer makes it newest and therefore the next victim of the policy that just
   restarted it.

   At each box sample the worker also records every running job's own footprint
   from its cgroup `memory.current` (or the direct process tree's RSS when the
   job is uncapped). The record distinguishes `evicted_for_own_memory_growth`,
   `evicted_as_collateral`, and `evicted_without_attribution`; exact box and
   victim slopes, footprint endpoints, source, sample count, and window travel
   with the item. Missing measurement is never converted into blame.

   `evictions` is now explicitly a **policy-aging count**, not an impossibility
   budget. Every eviction earns age and a fixed 60-second cooldown. After two
   losses, the next attempt is non-evictable and holds admission for its
   declared footprint, falling back to its observed peak. If neither exists,
   the queue drains current work and gives it an exclusive attempt. New
   arrivals cannot consume the reservation while it waits, including during
   the scheduler-imposed cooldown. Receipt dependencies remain genuine gates
   and do not reserve a box before they are satisfied. Thus a fitting item
   loses at most two attempts before the queue guarantees one; it never ages
   into a permanent bench. The old exponential cooldown and
   `evicted_Nx_does_not_fit` terminal outcome no longer exist.

   Producer-pinned `--no-evict` work follows the irreversible side of that
   admission contract from its first attempt. Once its placement, receipt and
   time gates pass, it holds *admission order* while its complete declared (or
   previously observed) footprint does not fit. It remains in `ready/`, owns
   no claim and publishes no external memory/GPU reservation while the box
   drains. With no footprint it waits for an exclusive attempt. This is
   deliberately different from ordinary optimistic backfill: the governor
   cannot walk a non-evictable overcommit back, so negative committed headroom
   and non-evictability are an invalid combination. Pinning admission order is
   also necessary because a refused non-evictable item earns no eviction age;
   allowing an unbounded stream of backfill would otherwise starve it forever.

   The only memory-fit terminal is
   `measured_footprint_exceeds_box_capacity`, reached when a declared or
   observed footprint is larger than `MemTotal - mem_reserve_gb`. Its record
   carries the source footprint, total memory, reserve, capacity, and timestamp.
   Ordinary command/receipt failures retain their separate attempt contract.

   Optimistic admission is unchanged for new **evictable** work: the queue
   still starts it whenever headroom is positive and measures what happens.
   The 2026-08-30
   `pq-currency-4b-v2` log is the incident behind this split: its application
   samples stayed near 81 GB with a 24 GB reserve, so it demonstrably fit, but
   those sparse four-minute samples cannot reconstruct the governor's unlogged
   two-second series. They establish that the repeated fit verdict was false;
   they do not establish which individual endpoint sample caused each trigger.

## Adding a box that is not a Spark

Eligibility is opt-in by declaration: `is_runnable` filters on the item's
`hosts`, `requires` and `needs_gpu`. An item that declares none of them is
eligible **everywhere** -- correct for two interchangeable GB10s, and wrong
the moment a third machine differs, because a job written for a Spark's venv
and repo checkout would be claimed by an unlike box and fail there, looking
like the job's fault rather than the placement's.

So an unlike box runs with `--only-declared`, which claims only items that
name it by `--host` or carry a tag it declares. Adding the box then cannot
break work enqueued before it existed. Give it tags that describe what it
actually is (`--tag x86 --tag cpu`), not what you hope to run on it, and let
work opt in with `--require`.

## What it deliberately does not do

* **It does not migrate work whose inputs are box-local.** `/home/rob` is
  local per box, not shared. An item is portable only if its inputs live on
  `/mnt/shared`; otherwise it pins itself with `--host`. **Receipts belong on
  `/mnt/shared`** for the same reason: a `--after` dependency on a
  box-local path can never be observed by the other box, so the dependent
  item would wait forever. The utilization win
  here is backfill-on-idle per box, and saying otherwise would overstate it.
* **It does not generate work.** Agents remain the producers. Rob asked for
  agents off the critical path of work *distribution*, and that is the scope.
* **It does not schedule cleverly.** One GPU item per box by default, because
  unified memory means two large jobs contend for one physical pool. With two
  GB10s there is no scheduling problem worth solving.

## Codex jobs

Codex runs through the queue like anything else, but two things bite:

* **Use the default sandbox, not `--sandbox workspace-write`.** The latter
  needs a user namespace, and this fleet runs with
  `kernel.apparmor_restrict_unprivileged_userns=1`, so bwrap fails its own
  loopback setup before any command runs. The failure is box-wide, not
  specific to the worker unit -- an interactive shell hits it too. The
  default sandbox executes commands and writes files fine.
* **Have Codex write the receipt itself** as its final instructed action --
  the PR number it opened, say -- and gate with `; test -s <receipt>`. See
  property 3 above for why `&& touch` does not work.

`gh` and `codex` live in `~/.local/bin` on sparklina, which a non-login shell
does not have on `PATH`; jobs there pass it explicitly with `--env PATH=...`.

## Operating it

The worker is a systemd **user** unit (`prismaquant/systemd/pqwork.service`),
enabled with linger on, so it survives logout and reboot.

```bash
systemctl --user status pqwork          # is the puller alive
journalctl --user -u pqwork -f          # what it is pulling
python3 $Q reserve --slots 1 --why "…"  # hold the GPU for a hand-launched job
python3 $Q reserve --release            # give it back

# A two-box actor calls this only after its start gate passes. Status 75
# means it lost admission and must return to the gate with no hold retained.
python3 $Q reserve-set acquire --id campaign-run-id \
  --owner-file /home/rob/campaign-state/reservation-owner.json \
  --host-resource sparky:6:115:6:121.6 \
  --host-resource gx10-6b77:6:115:6:121.6 \
  --require-no-claimed --why "two-box endpoint"
python3 $Q reserve-set release --id campaign-run-id \
  --owner-file /home/rob/campaign-state/reservation-owner.json
```

### Deploying a worker change

Merging the repo file does not change either live worker. From the merged
checkout, one command propagates the exact source through both deployment
layers and restarts the services:

```bash
python3 tools/deploy_pqwork.py
```

It atomically installs `tools/pqwork.py` to
`/mnt/shared/pq-queue/bin/pqwork.py`, copies that shared artifact atomically to
`/home/rob/.local/bin/pqwork.py` on sparky and sparklina, verifies one SHA-256
across all three deployed copies, and restarts both `pqwork` user services.

**This command is safe only between claims, never mid-claim.** A runner may
reread the script at a claim boundary; swapping it while a claim is active can
split one execution across two contracts. The deployer creates the shared
`.deploying` admission hold first and refuses to change any file while
`claimed/*.json` is non-empty. Workers carrying this version admit no new
claims while the hold exists. For the first rollout from an older worker that
does not yet understand the hold, stop both old workers after the board shows
no claims, recheck that no claim raced the stops, and deploy with
`--no-restart`. Create the admission hold while both remain stopped, start
both new workers, and verify each `workers/<host>.json` reports the active
service PID and deployed source SHA before removing the hold. The refusal
check remains a second guard, but cannot close a race in code not deployed
yet. A refusal before any replacement removes its hold. Once any deployed byte
has changed, a failure intentionally retains the hold so a split-version fleet
cannot claim new work. Clear that marker only after confirming no deployment
is active, all copies and loaded worker heartbeats match, and no item is
claimed. Rerun only after the board is idle again.

Restarting the unit kills any job it is running. That is intentional: the
lease then goes stale, the reaper requeues, and receipt-gating makes the
re-run a no-op for whatever already finished. One recovery path, not two.

To add a box: mount `/mnt/shared`, copy the unit, `systemctl --user enable
--now pqwork`. Tag it (`--tag rocm`) and items can require the tag. That is
the seam a real scheduler would slot into if the fleet ever justifies one.

## Migrating hand-launched campaigns

The default migration is to enqueue the synchronous campaign payload with
`pqwork enqueue --gpu`. Do not reserve the host first: the worker owns GPU
admission for a queued item, keeps its lease alive, retries it, and closes it
only when the work-produced receipt is non-empty.

For a concrete example, `tools/run_glm53_stock_harvest.sh` defines its remote
paths and source identity at lines 55-91, then generates a synchronous payload,
checks it, and hand-launches it as a transient user unit at lines 197-286. The
payload's final command writes
`work/artifacts/cost_stock_anchored.pkl`; that is a real campaign output, not a
wrapper-created marker. Once the preparation block has produced
`stock_harvest_payload.sh`, replace that script's `systemd-run` block with the
queue submission below. The command, working directory, environment, host-local
paths, and receipt are taken from
[`tools/run_glm53_stock_harvest.sh:55-91`](../../tools/run_glm53_stock_harvest.sh#L55-L91)
and
[`tools/run_glm53_stock_harvest.sh:197-286`](../../tools/run_glm53_stock_harvest.sh#L197-L286).

```bash
Q=/mnt/shared/pq-queue/bin/pqwork.py
COMMIT=$(git rev-parse HEAD)

python3 "$Q" enqueue \
  --id glm53-stock-harvest \
  --desc "GLM-5.3 stock-anchor harvest and campaign pricing" \
  --cmd "bash /home/rob/dq-runs/glm53-flash/stock_harvest_payload.sh" \
  --cwd /home/rob/prismaquant-glm53-gate \
  --gpu --host gx10-6b77 \
  --env PYTHONPATH=/home/rob/prismaquant-glm53-gate \
  --env LD_LIBRARY_PATH=/home/rob/dq-runs/glm53-flash/lib \
  --env TMPDIR=/home/rob/dq-runs/glm53-flash/tmp \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env CACHE_HEADROOM_GB=90 \
  --env PREFETCH_WORKERS=1 \
  --env PRISMAQUANT_IDENTITY_GIT_COMMIT="$COMMIT" \
  --receipt /home/rob/dq-runs/glm53-flash/work/artifacts/cost_stock_anchored.pkl
```

If a campaign truly cannot run through the queue and must use
`pqwork reserve`, releasing the reservation is part of the campaign's own
lifecycle. Install the `EXIT` trap immediately after a successful reservation,
preserve the campaign's exit status, and do not `exec` the campaign out of the
shell that owns the trap:

```bash
set -euo pipefail
Q=/mnt/shared/pq-queue/bin/pqwork.py
HOST=sparky

python3 "$Q" reserve --host "$HOST" --slots 1 \
  --why "hand-launched <campaign-id>"
release_gpu_reservation() {
  rc=$?
  trap - EXIT
  python3 "$Q" reserve --host "$HOST" --release || true
  exit "$rc"
}
trap release_gpu_reservation EXIT

bash /absolute/path/to/campaign.sh
test -s /absolute/path/to/work-produced-receipt
```

An `EXIT` trap cannot survive `SIGKILL` or a host loss, which is another reason
the queue is the recommended launcher. When retrofitting a campaign that is
already running, use a separate user unit that waits on the task-owned receipt
file and then releases the reservation. Never wait or clean up with
`pgrep -cf PATTERN` or `pkill -f PATTERN`: both can match the waiter's own
command line. Do not use `systemd-run --collect` as evidence; a collected unit's
reported state is not a completion receipt.
