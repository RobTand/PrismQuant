"""Empirical: regret(e) from perturbing one layer's interpolated prices.

Builds reduced allocation problem from layers with full truth (L14, L21, L0,
plus L1/L2 shards if available). For each scaled budget (whole-model 88/92/97/102.8GB
mapped to per-expert avg bpp), runs REAL solver (prismaquant.allocator_solver)
on true menu vs perturbed menu (one layer's interpolated rows shifted by e uniformly).

Structure of error: law residuals are CORRELATED within a layer x projection
(audit measures layer-level bias); per-expert idiosyncratic is backstop-handled.
Bar governs correlated component -> model as uniform relative shift of one layer's
interpolated rows (not per-expert independent noise).

Evaluates realized quality regret vs true-menu allocation (realized = evaluate
chosen rungs against TRUE errors). Produces regret(e) curve with n, both signs,
per budget.

CPU only; never touches gpu.lock; writes incrementally to host mount
if run >15min (content-keyed per (budget, victim_layer, e) file).
Reuse existing cache via content hashing: checks output dir for existing result.

Run:
  PYTHONPATH=/w python3 interp-diagnosis/harness/bound_empirical.py
  pytest interp-diagnosis/harness/bound_empirical.py -v
"""
from __future__ import annotations
import pickle, glob, pathlib, json, sys, os, hashlib, time, resource
import numpy as np

sys.path.insert(0, "/w")

RUN = pathlib.Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq")
OUTDIR = pathlib.Path("/w/interp-diagnosis/harness/output_regret")
OUTDIR2 = pathlib.Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/interp-diagnosis/harness_output")
# ensure dirs
for d in [OUTDIR, OUTDIR2]:
    d.mkdir(parents=True, exist_ok=True)

from prismaquant.cb_layout import type_size
from prismaquant.allocator_solver import Candidate, solve_allocation
from prismaquant import format_registry as fr

SB_PER_PROJ = 4194304 // 256
def bytes_per_proj(k:int)->int:
    return type_size(int(k),"fp8","v1") * SB_PER_PROJ
def bpp(k:int)->float:
    return 8*bytes_per_proj(k)/4194304

# Budget mapping: per-expert average bpp covering priced domain
# We define 4 scaled budgets that span the regime: low (tight), mid-low, mid-high, high
# Chosen to correspond to avg K ≈ 29,31,33,36 which maps to whole-model 88,92,97,102.8GB under sensible fixed overhead
BUDGETS = {
    "88GB_tight": 3.625,   # K29
    "92GB_mid_low": 3.875, # K31
    "97GB_mid": 4.125,     # K33 (knee)
    "102.8GB_high": 4.50,  # K36
}
# Also linear in bytes per 3-proj expert total (for solver we use total bytes)
def budget_bytes_per_expert(bpp_val:float)->int:
    # invert: bytes = bpp * params/8; params per expert total = 3*4194304=12582912
    return int(bpp_val * 12582912 / 8)

E_LIST = [0.02,0.04,0.06,0.08,0.12,0.20]
SIGNS = [+1,-1]

# Loaders (same as analytic)
def load_L14(proj):
    d=pickle.load(open(RUN/"pilot2/shards/layer_014.pkl","rb"))
    cells=d["projections"][proj]["cells"]
    return {int(k): np.array(cells[k]["selected_weight_mse"],dtype=float) for k in cells if 28<=int(k)<=38}
def load_L21(proj):
    d=pickle.load(open(RUN/"PILOT_FULL_MEASUREMENTS.pkl","rb"))
    return {int(k): np.array(d["measurements"][proj][k]["weight_mse_per_expert"],dtype=float) for k in d["measurements"][proj] if 28<=int(k)<=38}
def load_full_layer(layer:int, proj:str):
    out={}
    for p in glob.glob(str(RUN/f"burn-shards/layer_{layer:03d}_{proj}_v2s-full-layer_K*.pkl")):
        pay=pickle.load(open(p,"rb"))
        k=int(pay["cell"]["rung"])
        if 28<=int(k)<=38:
            out[k]=np.array(pay["cell"]["free_weight_mse"],dtype=float)
    return out

def discover_layers():
    layers={}
    # L14, L21 are pilot probes (pure-incumbent vs FusedMoE)
    for proj in ["gate_proj","up_proj","down_proj"]:
        layers[("L14",proj)] = load_L14(proj)
        layers[("L21",proj)] = load_L21(proj)
    # shards L0..5
    for L in [0,1,2,3,4,5]:
        for proj in ["gate_proj","up_proj","down_proj"]:
            per=load_full_layer(L, proj)
            if len(per)>=8:  # require near-full ladder
                layers[(f"L{L}",proj)] = per
    return layers

def build_packed_stats(layers):
    """Aggregate each (layer,proj) into packed unit: total dloss per K and memory bytes per K.
    Returns dict unit_name -> { "n_params":..., "per_K": {K: total_dloss} }
    """
    stats={}
    for (lname,proj), per in layers.items():
        unit=f"{lname}.{proj}"
        # n_params per unit = 4194304*256*3? No per proj unit is 256 experts *4194304 params
        n_params = 4194304*256
        # but we treat packed group as one decision; total bytes = bytes_per_proj*256
        entry={"n_params": n_params, "per_K": {}}
        for K, arr in per.items():
            total = float(np.sum(arr))
            entry["per_K"][K]=total
        stats[unit]=entry
    return stats

def build_candidates_from_stats(stats, target_bpp=None):
    """Build candidates dict: unit -> list[Candidate] for K28..38 plus maybe baseline?
    For allocation we need baseline = cheapest format (K28). We restrict menu to 28..38.
    """
    candidates={}
    # format specs: we need to define FormatSpec for each K. For solver we only need Candidate memory_bytes and predicted_dloss
    # We'll create synthetic format names FP8_CB_Kxx
    for unit, entry in stats.items():
        n_params = entry["n_params"]
        cands=[]
        for K in range(28,39):
            if K not in entry["per_K"]:
                continue
            fmt=f"FP8_CB_K{K}"
            bpc = bytes_per_proj(K)*256  # total bytes for packed unit (256 experts)
            bits_per_param = 8*bpc/n_params  # = bpp(K)
            dloss = entry["per_K"][K]
            cands.append(Candidate(fmt=fmt, bits_per_param=bits_per_param, memory_bytes=bpc, predicted_dloss=dloss))
        # sort by bits
        cands.sort(key=lambda c: c.bits_per_param)
        candidates[unit]=cands
    return candidates

def _stats_dict_for_solver(stats):
    """Need stats dict mapping unit -> {n_params} for solver."""
    return {unit: {"n_params": entry["n_params"]} for unit, entry in stats.items()}

def solve_for_target(candidates, stats_for_solver, target_bpp):
    """Run solver at target_bpp, return assignment dict unit->fmt and achieved bpp."""
    res=solve_allocation(stats_for_solver, candidates, target_bpp, bit_precision=0.001)
    if res is None:
        return None, None
    assign, _ = res
    # compute achieved
    total_params=sum(stats_for_solver[u]["n_params"] for u in assign)
    total_bytes=sum(next(c.memory_bytes for c in candidates[u] if c.fmt==assign[u]) for u in assign)
    achieved = 8*total_bytes/total_params if total_params>0 else 0
    return assign, achieved

def evaluate_true_dloss(assign, stats):
    """Sum true dloss for assignment using stats per_K."""
    total=0.0
    for unit, fmt in assign.items():
        K=int(fmt.split("_K")[1])
        total+= stats[unit]["per_K"][K]
    return total

def perturbed_candidates(stats, victim_units, e, sign):
    """Create perturbed candidates where victim_units' dloss multiplied by (1+ sign*e) for interpolated rows.
    For this study, interpolated rows are those that would be law-predicted: we model as ALL K in 28..38 shifted uniformly,
    since law's residual is correlated shift across interpolated rungs (the audit measures layer-level bias).
    To be precise we shift all candidate dlosses for victim by factor, but keep true evaluation at unperturbed.
    """
    factor = 1.0 + sign*e
    cand={}
    for unit, entry in stats.items():
        n_params = entry["n_params"]
        cands=[]
        for K in range(28,39):
            if K not in entry["per_K"]:
                continue
            fmt=f"FP8_CB_K{K}"
            bpc = bytes_per_proj(K)*256
            bits_per_param = 8*bpc/n_params
            true_dloss = entry["per_K"][K]
            dloss = true_dloss * factor if unit in victim_units else true_dloss
            cands.append(Candidate(fmt=fmt, bits_per_param=bits_per_param, memory_bytes=bpc, predicted_dloss=dloss))
        cands.sort(key=lambda c: c.bits_per_param)
        cand[unit]=cands
    return cand

def content_key(budget_name, victim, e, sign):
    s=f"{budget_name}|{victim}|{e:.3f}|{sign}"
    return hashlib.sha1(s.encode()).hexdigest()[:12]

def run_one_budget(budget_name, target_bpp, stats, candidates_true, stats_solver):
    # solve true
    assign_true, achieved_true = solve_for_target(candidates_true, stats_solver, target_bpp)
    if assign_true is None:
        print(f"WARN true solve infeasible at {budget_name} {target_bpp}")
        return []
    true_dloss = evaluate_true_dloss(assign_true, stats)
    results=[]
    for victim in sorted(stats.keys()):
        # victim is one layer.proj (e.g., L0.gate_proj) -> correlated shift applies to that whole unit
        # But task says "one layer's interpolated prices" — we interpret as one layer x projection (since law is per-projection)
        # This maximizes sensitivity (small victim fraction). Also test whole layer (all 3 projs) as variant? Stick to per-proj.
        for e in E_LIST:
            for sign in SIGNS:
                ck = content_key(budget_name, victim, e, sign)
                out_path = OUTDIR/f"{ck}.json"
                out_path2 = OUTDIR2/f"{ck}.json"
                # resume check
                cached=None
                for p in [out_path, out_path2]:
                    if p.exists():
                        try:
                            cached=json.load(open(p))
                            break
                        except: pass
                if cached is not None:
                    results.append(cached)
                    continue
                # perturb only victim
                cand_pert = perturbed_candidates(stats, {victim}, e, sign)
                assign_pert, achieved_pert = solve_for_target(cand_pert, stats_solver, target_bpp)
                if assign_pert is None:
                    # infeasible due to perturbation? treat as same as true but mark
                    regret=np.nan
                    realized_dloss=np.nan
                    flip=False
                else:
                    realized_dloss = evaluate_true_dloss(assign_pert, stats)
                    regret = max(0.0, realized_dloss - true_dloss)
                    flip = assign_pert != assign_true  # at least one diff ; more precisely victim diff?
                    # check if victim choice flipped
                    victim_flip = assign_pert.get(victim)!=assign_true.get(victim)
                rec={
                    "budget": budget_name,
                    "target_bpp": target_bpp,
                    "victim": victim,
                    "e": e,
                    "sign": sign,
                    "true_dloss": true_dloss,
                    "realized_dloss": float(realized_dloss) if not np.isnan(realized_dloss) else None,
                    "regret": float(regret) if not np.isnan(regret) else None,
                    "regret_rel": float(regret/true_dloss) if not np.isnan(regret) and true_dloss>0 else None,
                    "flip": bool(flip),
                    "victim_flip": bool(victim_flip) if 'victim_flip' in locals() else None,
                    "assign_true": assign_true,
                    "assign_pert": assign_pert,
                    "achieved_true": achieved_true,
                    "achieved_pert": achieved_pert,
                    "held": "packed"
                }
                # persist incrementally
                for p in [out_path, out_path2]:
                    tmp=p.with_suffix(".tmp")
                    with open(tmp,"w") as f:
                        json.dump(rec,f,indent=2)
                    tmp.rename(p)
                results.append(rec)
                # RSS guard
                try:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on linux
                    # rss in KB, convert to GB
                    rss_gb = rss/1024/1024
                    if rss_gb > 60:
                        print(f"RSS guard: {rss_gb:.1f}GB >60 abort")
                        break
                except: pass
    return results

def run_all():
    print("Discovering layers...")
    layers=discover_layers()
    print(f"Found {len(layers)} layer.proj units: {sorted(layers.keys())}")
    stats=build_packed_stats(layers)
    candidates_true=build_candidates_from_stats(stats)
    stats_solver=_stats_dict_for_solver(stats)
    print(f"Stats built: {list(stats.keys())[:5]}")
    # quick sanity: check true solve at each budget
    for bname, tbpp in BUDGETS.items():
        ass,_=solve_for_target(candidates_true, stats_solver, tbpp)
        print(f"Budget {bname} tbpp {tbpp} -> {'feasible' if ass else 'infeasible'} {ass}")

    all_results=[]
    for bname, tbpp in BUDGETS.items():
        print(f"\n=== Budget {bname} target_bpp {tbpp:.3f} ===")
        res=run_one_budget(bname, tbpp, stats, candidates_true, stats_solver)
        all_results.extend(res)
        # summarize regret stats
        import collections
        for e in E_LIST:
            vals=[r for r in res if r["e"]==e]
            if not vals: continue
            regs=[r["regret_rel"] for r in vals if r["regret_rel"] is not None]
            flips=np.mean([r["victim_flip"] for r in vals])
            p50=np.median(regs) if regs else 0
            p95=np.percentile(regs,95) if regs else 0
            mx=np.max(regs) if regs else 0
            mean=np.mean(regs) if regs else 0
            n=len(vals)
            print(f" e={e:4.0%}  n={n:3d}  flip_rate {flips:4.0%}  regret_rel median {p50:.3%} p95 {p95:.3%} max {mx:.3%} mean {mean:.3%}")
            # also count non-zero regrets
            nz=np.mean(np.array(regs)>1e-9) if regs else 0
            print(f"          non-zero regret {nz:4.0%}")

    # write summary table
    summary_path = pathlib.Path("/w/interp-diagnosis/harness/bound_empirical_summary.json")
    summary_path2 = OUTDIR2/"bound_empirical_summary.json"
    summary={"budgets": BUDGETS, "e_list": E_LIST, "results": all_results}
    for p in [summary_path, summary_path2, OUTDIR/"bound_empirical_summary.json"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p,"w") as f:
            json.dump(summary,f,indent=2)
    print(f"Wrote summary to {summary_path}")

    # also produce regret curve table for AUDIT_BOUND
    print("\n=== Regret(e) aggregated curve ===")
    for bname in BUDGETS:
        print(f"-- {bname} --")
        for e in E_LIST:
            for sign in SIGNS:
                vals=[r for r in all_results if r["budget"]==bname and r["e"]==e and r["sign"]==sign]
                regs=[r["regret_rel"] for r in vals if r["regret_rel"] is not None]
                if not regs: continue
                print(f"  e={'+' if sign>0 else '-'}{e:4.0%}  median {np.median(regs):.3%} p95 {np.percentile(regs,95):.3%} max {np.max(regs):.3%} flip {np.mean([r['victim_flip'] for r in vals]):.0%}")

    return all_results

if __name__=="__main__":
    run_all()

# pytest helpers
def test_solver_feasible():
    layers=discover_layers()
    assert len(layers)>=6, f"need at least 6 units found {len(layers)}"
    stats=build_packed_stats(layers)
    cand=build_candidates_from_stats(stats)
    solver_stats=_stats_dict_for_solver(stats)
    for tbpp in BUDGETS.values():
        a,_=solve_for_target(cand, solver_stats, tbpp)
        assert a is not None, f"infeasible at {tbpp}"

def test_regret_nonnegative():
    layers=discover_layers()
    stats=build_packed_stats(layers)
    cand=build_candidates_from_stats(stats)
    solver_stats=_stats_dict_for_solver(stats)
    # pick one budget
    tbpp=list(BUDGETS.values())[2]
    a_true,_=solve_for_target(cand, solver_stats, tbpp)
    true_dl=evaluate_true_dloss(a_true, stats)
    for e in [0.08,0.20]:
        cand_p=perturbed_candidates(stats, {sorted(stats.keys())[0]}, e, 1)
        a_p,_=solve_for_target(cand_p, solver_stats, tbpp)
        if a_p is not None:
            realised=evaluate_true_dloss(a_p, stats)
            assert realised >= true_dl -1e-9
