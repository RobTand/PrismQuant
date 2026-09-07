"""Compare a merged fan-out cost table with a whole-scope one, field by field.

Equality is defined modulo what a wall clock and a path write: encode seconds,
wall seconds, rounds run, the early-stop flag, the cache and wire directories,
and the fan-out's own provenance. Everything a price is made of must match.
"""
import json
import pickle
import sys

WALL_CLOCK = {"encode_seconds", "wall_seconds", "rounds_run", "stopped_early",
              "cache_dir", "wire_dir", "capture_path", "path",
              "campaign_fanout", "seed_checkpoint", "unit_selection",
              "campaign_scope", "capture_sha256", "calibration_identity",
              "hessian", "activation_static_scales", "surfaces", "population",
              # Compared on its own terms below: a seeded side carries the
              # evidence a monolith of the same menu never saw, so equality
              # here is a claim about the PRICES, not about the evidence.
              "unservable"}


def strip(value, drop):
    if isinstance(value, dict):
        return {k: strip(v, drop) for k, v in sorted(value.items()) if k not in drop}
    if isinstance(value, (list, tuple)):
        return [strip(v, drop) for v in value]
    return value


def report(name, left, right):
    same = left == right
    print(f"{'OK  ' if same else 'DIFF'}  {name}")
    if not same:
        if isinstance(left, dict) and isinstance(right, dict):
            only_l = sorted(set(left) - set(right))
            only_r = sorted(set(right) - set(left))
            if only_l:
                print(f"        only in A: {only_l[:6]}")
            if only_r:
                print(f"        only in B: {only_r[:6]}")
            for key in sorted(set(left) & set(right)):
                if left[key] != right[key]:
                    print(f"        {key}:\n          A={json.dumps(left[key], default=str)[:400]}"
                          f"\n          B={json.dumps(right[key], default=str)[:400]}")
        else:
            print(f"        A={json.dumps(left, default=str)[:400]}")
            print(f"        B={json.dumps(right, default=str)[:400]}")
    return same


sys.path.insert(0, "/home/rob/tessera-runs/pq282")
from prismaquant.tessera_expert_projection import POPULATION_KEY

a = pickle.load(open(sys.argv[1], "rb"))
b = pickle.load(open(sys.argv[2], "rb"))
print(f"A (whole scope) {sys.argv[1]}")
print(f"B (merged fan-out) {sys.argv[2]}\n")

ok = True
ok &= report("schema/currency", (a["schema"], a["currency"]), (b["schema"], b["currency"]))
ok &= report("units priced", sorted(a["costs"]), sorted(b["costs"]))
ok &= report("formats", a["formats"], b["formats"])
ok &= report("menu_sizes", a["menu_sizes"], b["menu_sizes"])
ok &= report("anchor_counts", a["anchor_counts"], b["anchor_counts"])
ok &= report("leave_one_anchor_out", a["leave_one_anchor_out"], b["leave_one_anchor_out"])
ok &= report("non_interpolable", a["non_interpolable"], b["non_interpolable"])

# Priced rows, modulo the encode clock. The Hessian identity is compared
# whole EXCEPT capture_sha256, which the merge recomputes over the union.
drop_row = {"encode_seconds"}
rows_a = {q: {f: strip(r, drop_row) for f, r in rows.items()} for q, rows in a["costs"].items()}
rows_b = {q: {f: strip(r, drop_row) for f, r in rows.items()} for q, rows in b["costs"].items()}
for q in sorted(set(rows_a) | set(rows_b)):
    ra = {f: {k: v for k, v in r.items() if k != "hessian_identity"}
          for f, r in rows_a.get(q, {}).items()}
    rb = {f: {k: v for k, v in r.items() if k != "hessian_identity"}
          for f, r in rows_b.get(q, {}).items()}
    ok &= report(f"rows[{q}]", ra, rb)
    ha = {f: {k: v for k, v in r["hessian_identity"].items() if k != "capture_sha256"}
          for f, r in rows_a.get(q, {}).items() if "hessian_identity" in r}
    hb = {f: {k: v for k, v in r["hessian_identity"].items() if k != "capture_sha256"}
          for f, r in rows_b.get(q, {}).items() if "hessian_identity" in r}
    ok &= report(f"hessian_identity[{q}] (minus capture digest)", ha, hb)

pa, pb = a["provenance"], b["provenance"]
ok &= report("provenance (minus wall clock and fan-out)",
             strip(pa, WALL_CLOCK), strip(pb, WALL_CLOCK))
ok &= report("static input scales",
             pa["activation_static_scales"]["units"],
             pb["activation_static_scales"]["units"])
ok &= report("hessian identity block (minus capture digest and paths)",
             strip(pa["hessian"], {"capture_path", "capture_sha256"}),
             strip(pb["hessian"], {"capture_path", "capture_sha256"}))
ok &= report("surfaces (minus encode seconds)",
             strip(pa["surfaces"], {"encode_seconds"}),
             strip(pb["surfaces"], {"encode_seconds"}))
ok &= report("population coverage", pa[POPULATION_KEY], pb[POPULATION_KEY])

# The two claims that make this an equality rather than a resemblance.
#
# The capture digest is computed over the Hessian union and every unit's H
# bytes.  Equal digests say the merged capture IS the whole-scope object the
# export leg would have read, not a re-stamp that merely looks like one.
ok &= report("hessian capture_sha256",
             pa["hessian"]["capture_sha256"], pb["hessian"]["capture_sha256"])

# And the journal identity: both sides compute it from the same tree over the
# same units and the same calibration, with the fan-out's own flags popped.
# Equal identities say a later whole-scope run can RESUME the merged journal,
# which is what makes it a cost.anchors.json and not a bundle of shards.
if len(sys.argv) > 4:
    ja = json.loads(open(sys.argv[3]).read())
    jb = json.loads(open(sys.argv[4]).read())
    ok &= report("journal identity_sha256",
                 ja["identity_sha256"], jb["identity_sha256"])
    ok &= report("journal identity", ja["identity"], jb["identity"])
    ok &= report("journal units", sorted(u["qname"] for u in ja["units"]),
                 sorted(u["qname"] for u in jb["units"]))

# Evidence is reported, not equated: what must hold is that nothing carried as
# unservable was also priced -- the two sides of the menu line, on both arms.
for label, payload in (("A", a), ("B", b)):
    ev = payload["provenance"].get("unservable", {})
    priced = {(q, f) for q, rows in payload["costs"].items() for f in rows}
    overlap = sorted((q, f) for q, rungs in ev.items() for f in rungs
                     if (q, f) in priced)
    print(f"{'OK  ' if not overlap else 'DIFF'}  unservable[{label}] "
          f"{sum(len(v) for v in ev.values())} row(s), priced overlap {overlap}")
    ok &= not overlap

print("\nEQUAL" if ok else "\nNOT EQUAL")
sys.exit(0 if ok else 1)
