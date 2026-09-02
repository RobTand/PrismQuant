#!/usr/bin/env python3
"""Post-install gate: exercise the wheel from site-packages, not the checkout.

Run this from a directory that is NOT the repo root, so `import prismaquant`
cannot resolve to the source tree. It asserts the install is genuinely
non-editable and then does the one thing an import smoke test does not: reads
the runtime JSON specs back out of site-packages, which is where a
package-data regression actually shows up.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import prismaquant

where = os.path.dirname(prismaquant.__file__)
print(f"prismaquant resolved from: {where}")
if "site-packages" not in where and "dist-packages" not in where:
    sys.exit("prismaquant did not resolve from site-packages — this check must "
             "run outside the repo root, or the wheel is not installed")

# Serving-constraint specs: read from the installed package's JSON.
from prismaquant import serving_profiles as sp  # noqa: E402

for name in ("vllm_packed_moe", "vllm_qwen3_5_packed_moe", "gguf", "research"):
    profile = sp.load_serving_profile(name)
    if profile is None:
        sys.exit(f"serving profile {name!r} did not load from the install")
    print(f"  serving profile OK: {name}")

# Model-structure specs: the spec directory must exist inside the package and
# carry every arch the source tree ships.
spec_dir = os.path.join(where, "model_profiles", "specs")
specs = sorted(f for f in os.listdir(spec_dir) if f.endswith(".json"))
if not specs:
    sys.exit(f"no model-structure specs under {spec_dir} — package-data broke")
print(f"  model-structure specs OK: {len(specs)} ({', '.join(specs)})")

# Lane specs (re-vet R16): the per-lane ship-gate declaration is package data
# too — an install that cannot resolve it silently loses the only place the
# bar is defined.
from prismaquant.lane_spec import all_lane_specs  # noqa: E402

lanes = all_lane_specs()
if len(lanes) < 3:
    sys.exit(f"lane specs missing from the install (found {len(lanes)}) — "
             "package-data broke")
print(f"  lane specs OK: {len(lanes)} "
      f"({', '.join(spec.id for spec in lanes)})")

# run-pipeline.sh is the orchestrator, shipped as package data.
pipeline = os.path.join(where, "run-pipeline.sh")
if not os.path.isfile(pipeline):
    sys.exit(f"run-pipeline.sh missing from the install ({pipeline})")
print("  run-pipeline.sh OK")

# Canonical IQ and CB tables must ship. Without IQ tables the first real GGUF
# encode raises FileNotFoundError; without CB lattices the package silently
# regenerates them, which is expensive and can be nondeterministic on CUDA.
from prismaquant.gguf_iq_formats import _tables as iq_tables  # noqa: E402
from prismaquant.nvfp4_cb_formats import _lattice_file  # noqa: E402

iq = iq_tables("cpu")
required_iq = {
    "grid_iq2_xxs", "grid_iq2_xs", "grid_iq2_s",
    "grid_iq3_xxs", "grid_iq3_s", "ksigns", "kvalues_iq4nl",
}
if set(iq) != required_iq:
    sys.exit(f"installed IQ tables differ from canonical keys: {sorted(iq)}")
print(f"  IQ tables OK: {len(iq)}")

lattices = _lattice_file()
if len(lattices) != 20:
    sys.exit(f"installed CB lattice table count is {len(lattices)}, expected 20")
print(f"  CB lattices OK: {len(lattices)}")

# The Gridbook codebook lane's packaged runtime helper and immutable pin were
# checked here until 2026-09-02, when that lane was retired
# (archive/gridbook_lane_2026-09-02/). Nothing replaces the check: the Tessera
# lane's pin is not packaged data, it is read from the installed `tessera`
# distribution's own runtime_contract.json.

# The pipeline's final refusal display is packaged rather than pointing at the
# repository-only tools/ tree.
shipcard = subprocess.run(
    [sys.executable, "-m", "prismaquant.shipcard_cli", "--help"],
    capture_output=True, text=True,
)
if shipcard.returncode != 0:
    sys.exit(f"installed shipcard CLI failed:\n{shipcard.stderr}")
print("  shipcard CLI OK")

# The CLI entry point every downstream user touches first.
r = subprocess.run([sys.executable, "-m", "prismaquant.allocator", "--help"],
                   capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(f"`python -m prismaquant.allocator --help` failed:\n{r.stderr}")
print("  allocator CLI OK")

print("installed wheel verified")
