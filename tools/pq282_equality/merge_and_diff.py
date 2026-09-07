import subprocess, sys, pathlib
V = "/mnt/shared/tessera-measurements/pq282-2026-09-06/v7"
py = sys.executable
rc = subprocess.call([py, "tools/dispatch_tessera_campaign.py",
                      "merge", "--workspace", f"{V}/fanout",
                      "--out", f"{V}/merged/cost.pkl"])
print(f"[driver] merge rc={rc}", flush=True)
if rc:
    raise SystemExit(rc)
rc = subprocess.call([py, "tools/pq282_equality/equality_diff.py",
                      f"{V}/monolith/cost.pkl", f"{V}/merged/cost.pkl",
                      f"{V}/monolith/cost.anchors.json",
                      f"{V}/merged/cost.anchors.json"])
print(f"[driver] equality_diff rc={rc}", flush=True)
raise SystemExit(rc)
