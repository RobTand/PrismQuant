"""pqtel — crash-proof box telemetry for the DGX Spark boxes (sparky, lina).

Stdlib only, under /usr/bin/python3, no virtualenv. This package is a top-level
sibling of `prismaquant/` and imports nothing from it: `import prismaquant`
costs 4.67 s against 0.02 s for a stdlib-only start, which disqualifies it for
anything a hook, a daemon, or a 2 Hz sampler calls. The dependency runs the
other way -- PrismaQuant hot paths may `import pqtel.*` under try/except.

Step 1 of the observability plan: the recorder (`pqtel.recorder`) and the
health verb (`pqtel.health`). Capture, nsys, MCP, window and incident are
deliberately not here.
"""

__version__ = "0.1.0"

# Where data lives. Never /tmp (an OOM cleared it and wiped the MiniMax
# artifacts), never the /mnt/shared NFS mount.
DATA_DIR = "/home/rob/pqtel"
CSV_DIR = DATA_DIR + "/csv"
