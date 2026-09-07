"""Recheck frozen campaign artifacts with the repaired merger, without encoding.

The recorded monolith remains unchanged. A separate comparison copy applies
only the current canonical refusal ordering, after proving multiset equality
with the historical merged result. The ordinary equality tool compares all
remaining price, population, capture and journal identity fields.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dispatch_tessera_campaign import cmd_merge
from prismaquant.tessera_campaign import canonical_refusals

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--recorded", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
paths = {"monolith": args.recorded / "monolith/cost.pkl",
         "historical_merge": args.recorded / "merged/cost.pkl"}
raw = {name: path.read_bytes() for name, path in paths.items()}
payloads = {name: pickle.loads(data) for name, data in raw.items()}
counts = {name: Counter(json.dumps(item, sort_keys=True) for item in payload["non_interpolable"])
          for name, payload in payloads.items()}
if counts["monolith"] != counts["historical_merge"]:
    raise RuntimeError("historical refusal records differ in content or multiplicity")
print("Historical refusals: identical content and multiplicity; order alone differed", flush=True)
merged = args.out / "merged/cost.pkl"
cmd_merge(argparse.Namespace(workspace=args.recorded / "fanout", out=merged))
normalized = args.out / "monolith-canonical-refusals.pkl"
monolith = payloads["monolith"]
monolith["non_interpolable"] = canonical_refusals(monolith["non_interpolable"])
normalized.write_bytes(pickle.dumps(monolith, protocol=pickle.HIGHEST_PROTOCOL))
command = [sys.executable, str(Path(__file__).with_name("equality_diff.py")),
           str(normalized), str(merged),
           str(args.recorded / "monolith/cost.anchors.json"),
           str(merged.with_suffix(".anchors.json"))]
result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout, flush=True)
(args.out / "comparison.log").write_text(result.stdout)
(args.out / "recheck.json").write_text(json.dumps({
    "inputs": {name: {"path": str(paths[name]), "sha256": hashlib.sha256(data).hexdigest()}
               for name, data in raw.items()},
    "normalization": "canonical_refusals only, on a separate monolith comparison copy",
    "historical_refusals_multiset_equal": True,
    "no_new_encodes": True,
    "comparison_returncode": result.returncode,
    "comparison_command": command,
}, indent=2) + "\n")
raise SystemExit(result.returncode)
