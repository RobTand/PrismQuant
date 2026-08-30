# Installing pqteld

Step 1 of the observability plan: the recorder and `pqtel health`. Nothing
here needs sudo on sparky. Two of the lina steps do.

## Status

| Box | Recorder | Verified by |
|---|---|---|
| sparky | installed, enabled, running | `systemctl --user is-active pqteld`, CSV rows at 2 Hz, three SIGKILL trials |
| lina | **not installed** | this session has no shell on lina — see below |

lina is reachable from sparky **only** over its Netdata HTTP port
(`http://192.168.1.110:19999`). There is no SSH key here and the hostname
`lina` does not resolve, so nothing in this repo has run on lina. Do not read
any claim about lina's local state as verified.

## sparky (done)

```
mkdir -p ~/.config/systemd/user
cp /home/rob/prismaquant/pqtel/units/pqteld.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pqteld
```

`Linger=yes` already holds for `rob` on sparky, so the user unit starts at
boot without a login.

## lina (owed — run these on lina)

The repo is not on lina, so copy the package across first, from sparky, with
whatever credential you use interactively. Address lina by IP, not by name: `lina` does not resolve from sparky, so any
`ssh lina` / `rsync ... lina:` line fails on paste. Your interactive key may
work where this session's did not (`ssh 192.168.1.110` returned Permission
denied here).

```
ssh rob@192.168.1.110 'mkdir -p /home/rob/pqtel/csv /home/rob/.config/systemd/user /home/rob/prismaquant /home/rob/.local/bin'
rsync -a /home/rob/prismaquant/pqtel/ rob@192.168.1.110:/home/rob/prismaquant/pqtel/
rsync -a /home/rob/.local/bin/pqtel   rob@192.168.1.110:/home/rob/.local/bin/pqtel
rsync -a /home/rob/prismaquant/pqtel/units/pqteld.service \
         rob@192.168.1.110:/home/rob/.config/systemd/user/pqteld.service
```

Then on lina:

```
systemctl --user daemon-reload
systemctl --user enable --now pqteld
systemctl --user is-active pqteld
pqtel health
```

**Needs sudo on lina.** `Linger` reads `no` there, so a `--user` unit cannot
start at boot and the recorder would be gone after the next reboot — which is
the scenario it exists for. `enable-linger` is polkit-gated, so run it with
sudo even for your own user:

```
sudo loginctl enable-linger rob
loginctl show-user rob | grep Linger      # expect Linger=yes
```

Until linger is enabled, a login-scoped fallback keeps the recorder up for the
current session only:

```
systemd-run --user --unit=pqteld \
  --setenv=PYTHONPATH=/home/rob/prismaquant \
  /usr/bin/python3 -m pqtel.recorder --csv-dir /home/rob/pqtel/csv
```

## Also owed, both boxes (sudo)

Netdata's own unit is unhardened against the event it collects:
`Restart=on-failure` with `OOMScoreAdjust=0`. A drop-in fixes it. Unlike a
`--user` unit, a system unit *may* set a negative `OOMScoreAdjust`, which
needs `CAP_SYS_RESOURCE`:

```
sudo mkdir -p /etc/systemd/system/netdata.service.d
printf '[Service]\nRestart=always\nOOMScoreAdjust=-1000\n' \
  | sudo tee /etc/systemd/system/netdata.service.d/hardening.conf
sudo systemctl daemon-reload && sudo systemctl restart netdata
systemctl show netdata.service -p Restart -p OOMScoreAdjust
```

## Not owed any more

The observability plan's section 5 lists two sudo items that are **already
applied on sparky**, measured 2026-08-28:

- `kernel.perf_event_paranoid` reads `2`, not the `4` the plan records. CPU
  sampling and `--cpuctxsw` work unprivileged today.
- `RmProfilingAdminOnly` reads `0`, not `1`. GPU performance counters,
  `nsys --gpu-metrics-devices` and `ncu` are open; no reboot is owed.

Both are **UNVERIFIED on lina**. `pqtel health` re-reads them live on
whichever box it runs on, so run it on lina rather than trusting this note.

## Process-name safety

`/mnt/shared/mem-guard-glm53.sh` runs

```
pkill -9 -f "ray::RayWorkerP"; pkill -9 -f "vllm serve"; pkill -9 -f "sglang.launch_server"
```

`pkill -f` matches the whole command line, so the check is against the unit's
full `ExecStart`, not just the executable name:

```
/usr/bin/python3 -m pqtel.recorder --csv-dir /home/rob/pqtel/csv
```

No substring match on any of the three patterns. Verified 2026-08-28. Re-check
this if either the unit's `ExecStart` or the guard's patterns change.
