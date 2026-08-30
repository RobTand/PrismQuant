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
  reserved/<host>.gpu GPU slots held by work outside the queue
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

## The five properties that make it safe

1. **Claiming is `rename()`.** Atomic on NFSv4; the loser gets `ENOENT` and
   moves on. `flock` over NFS is version-dependent and was not used.
   Verified across boxes: 6 items, 8 racing workers on two machines, 6
   executions, 0 double-claims.
2. **A claim is a lease.** The holder rewrites its timestamp every 30 s; any
   worker may requeue a claim stale for 300 s. This is the recovery path for
   a box that dies mid-job — without it the queue develops permanent holes,
   which is the original bug one level down.
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
   `reserved/<host>.gpu` holds slots and `reserved/<host>.mem` holds
   gigabytes, for jobs the queue cannot see. The memory half exists because
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
```

Restarting the unit kills any job it is running. That is intentional: the
lease then goes stale, the reaper requeues, and receipt-gating makes the
re-run a no-op for whatever already finished. One recovery path, not two.

To add a box: mount `/mnt/shared`, copy the unit, `systemctl --user enable
--now pqwork`. Tag it (`--tag rocm`) and items can require the tag. That is
the seam a real scheduler would slot into if the fleet ever justifies one.
