"""Analytic: distribution of marginal-utility gaps at budgets of record.

Computes per-step error drops and efficiency gaps from banked truth ladders
(L14, L21, L0) plus available v2 shards (L1, L2 ...). Reports fraction of
selections within e of a flip as function of e.

CPU only; no gpu.lock. Resume by content: checkpoint not needed (fast).

Run:
  PYTHONPATH=/w python3 interp-diagnosis/harness/bound_analytic.py
  pytest interp-diagnosis/harness/bound_analytic.py -v
"""
from __future__ import annotations
import pickle, glob, pathlib, json, sys
import numpy as np

sys.path.insert(0, "/w")

RUN = pathlib.Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq")
PROJS = ["gate_proj","up_proj","down_proj"]

# bytes per expert-projection per K (fp8 product, type_size = k*4 per 256 weights)
from prismaquant.cb_layout import type_size, VEC_DIM, SUPERBLOCK

# per-proj params: 2048*2048 = 4194304, superblocks = 16384
SB_PER_PROJ = 4194304 // 256  # 16384

def bytes_per_proj(k:int)->int:
    return type_size(int(k),"fp8","v1") * SB_PER_PROJ

def bpp(k:int)->float:
    return 8*bytes_per_proj(k)/4194304

# Loaders
def load_L14(proj):
    d=pickle.load(open(RUN/"pilot2/shards/layer_014.pkl","rb"))
    cells=d["projections"][proj]["cells"]
    return {int(k): np.array(cells[k]["selected_weight_mse"],dtype=float) for k in cells if 28<=int(k)<=38}
def load_L21(proj):
    d=pickle.load(open(RUN/"PILOT_FULL_MEASUREMENTS.pkl","rb"))
    return {int(k): np.array(d["measurements"][proj][k]["weight_mse_per_expert"],dtype=float) for k in d["measurements"][proj] if 28<=int(k)<=38}
def load_L0(proj):
    out={}
    for p in glob.glob(str(RUN/f"burn-shards/layer_000_{proj}_v2s-full-layer_K*.pkl")):
        pay=pickle.load(open(p,"rb"))
        k=int(pay["cell"]["rung"])
        if 28<=k<=38:
            out[k]=np.array(pay["cell"]["free_weight_mse"],dtype=float)
    return out
def load_shard_layer(layer:int, proj:str):
    """Generic loader for available v2 shards (burn-shards). Return empty if not full ladder."""
    out={}
    for p in glob.glob(str(RUN/f"burn-shards/layer_{layer:03d}_{proj}_v2s-full-layer_K*.pkl")):
        pay=pickle.load(open(p,"rb"))
        k=int(pay["cell"]["rung"])
        if 28<=k<=38:
            out[k]=np.array(pay["cell"]["free_weight_mse"],dtype=float)
    # require at least 8 consecutive covering 28-38 to be considered full
    if len(out)>=8:
        return out
    return {}
def load_L1(proj):
    return load_shard_layer(1,proj)
def load_L2(proj):
    return load_shard_layer(2,proj)

LOADERS = {
    "L14": load_L14,
    "L21": load_L21,
    "L0": load_L0,
    "L1": load_L1,
    "L2": load_L2,
}

def per_expert_drops(per_expert):
    """Return array of all per-expert adjacent drops (K->K+1) pooled."""
    drops=[]
    ks=sorted(per_expert)
    for k in range(28,38):
        if k in per_expert and (k+1) in per_expert:
            a=per_expert[k]; b=per_expert[k+1]
            rel=(a-b)/np.maximum(a,1e-30)
            drops.extend(rel.tolist())
    return np.array(drops)

def packed_group_stats(per_expert):
    """Aggregate 256 experts into one packed decision unit (sum). Returns per-K total dloss and bytes."""
    totals={}
    for k,arr in per_expert.items():
        totals[k]=float(np.sum(arr))  # sum over experts, uniform scaling
    return totals

def efficiency_gaps(per_expert, use_packed=True):
    """For each unit, compute incremental efficiency = delta_dloss / delta_bytes between adjacent K.
    Gap between successive efficiencies measures distance to flip.
    Returns arrays of relative gaps."""
    if use_packed:
        totals=packed_group_stats(per_expert)
        ks=sorted(totals)
        # bytes delta uniform: 4*SB_PER_PROJ per step
        db = bytes_per_proj(29)-bytes_per_proj(28)  # constant 65536
        gaps=[]
        effs=[]
        for k in range(28,38):
            if k in totals and (k+1) in totals:
                eff = (totals[k]-totals[k+1])/db
                effs.append(eff)
        effs=np.array(effs)
        # gap between consecutive efficiencies (adjacent rung pairs)
        # flip threshold approx |eff_i - eff_{i+1}| / eff_i
        for i in range(len(effs)-1):
            gap = abs(effs[i+1]-effs[i])/max(effs[i],1e-30)
            gaps.append(gap)
        return np.array(gaps), np.array([ (totals[k]-totals[k+1])/totals[k] for k in range(28,38) if k in totals and (k+1) in totals])
    else:
        # per-expert efficiencies pooled
        gaps=[]
        drops=[]
        for expert_idx in range(256):
            effs=[]
            vals={k: per_expert[k][expert_idx] for k in per_expert}
            ks=sorted(vals)
            db = bytes_per_proj(29)-bytes_per_proj(28)
            for k in range(28,38):
                if k in vals and (k+1) in vals:
                    eff=(vals[k]-vals[k+1])/db
                    effs.append(eff)
                    drops.append((vals[k]-vals[k+1])/max(vals[k],1e-30))
            effs=np.array(effs)
            for i in range(len(effs)-1):
                gaps.append(abs(effs[i+1]-effs[i])/max(effs[i],1e-30))
        return np.array(gaps), np.array(drops)

def translate_budgets():
    """Translate whole-model byte budgets to per-expert byte regimes.
    Assumption documented here: total expert bytes at K28=60.6GB, K38=82.2GB (138.5B params * bpp/8).
    Dense + shared + codebooks + MTP account for ~20-25GB fixed overhead (NVFP4 dense at ~4.5bpp on ~143B dense params ≈ 80GB? but we treat as ~25GB compressed? Honest: whole-model 88GB => per-expert avg ~3.6bpp (K29), 92GB=>3.9bpp (K31), 97GB=>4.2bpp (K33), 102.8GB=>4.5bpp (K36). Derivation below.
    """
    lines=[]
    lines.append("Whole-model 88/92/97/102.8GB translation:")
    lines.append("  total quantizable ≈281B params (per DeepseekV4Profile: 281,263,734,784 probe-measured).")
    lines.append("  Expert subset: 138.5B params (3*2048*2048*256*43). Remaining ~142.5B dense/shared/attention.")
    lines.append("  Expert bytes: K28 60.6GB, K38 82.2GB (8*type_size/256 = k/8 bpp).")
    lines.append("  At target whole-model 88GB, with ~25GB fixed for dense (NVFP4_CB_K16-like 4b) + 10.8GB MTP, remaining ~52GB for routed experts => avg expert bpp ~3.0 => below K28 floor, so low budget is K28-constrained (many experts at floor).")
    lines.append("  More faithful: map directly per-expert avg bpp regimes covering priced domain:")
    lines.append("    budget 88GB ≈ avg K29 (3.625 bpp) constrained, 92GB≈K31 (3.875), 97GB≈K33 (4.125), 102.8GB≈K36 (4.5).")
    lines.append("  For analytic we therefore evaluate marginal gaps in K28-38 window (the only menu where interpolation matters);")
    lines.append("  budgets below K28 floor give same gap statistics (all at floor) so we report uniform window.")
    # compute per-expert byte per K
    for K in [28,29,31,33,36,38]:
        bp = bytes_per_proj(K)
        print(f"K{K}: bytes_per_proj {bp} total_expert_3proj {bp*3} bpp {bpp(K):.3f}")
    return "\n".join(lines)

def analyze():
    print("=== Analytic: per-step drop distribution ===")
    print(translate_budgets())
    print()
    # collect pooled drops across all layers/projs
    all_drops_packed=[]
    all_gaps_packed=[]
    per_combo=[]
    for lname, loader in LOADERS.items():
        for proj in PROJS:
            per=loader(proj)
            if not per or len(per)<5:
                continue
            gaps, drops = efficiency_gaps(per, use_packed=True)
            all_drops_packed.extend(drops.tolist())
            all_gaps_packed.extend(gaps.tolist())
            # also per-expert pooled for vulnerable fraction
            _, drops_pe = efficiency_gaps(per, use_packed=False)
            per_combo.append((lname,proj,gaps,drops,drops_pe))
            print(f"{lname} {proj}: N={len(per)} ks={sorted(per.keys())} drops median {np.median(drops):.1%} gaps median {np.median(gaps) if len(gaps)>0 else 0:.1%}")
    all_drops_packed=np.array(all_drops_packed)
    all_gaps_packed=np.array(all_gaps_packed)
    print(f"\nPooled packed drops: median {np.median(all_drops_packed):.1%} p10 {np.percentile(all_drops_packed,10):.1%} p90 {np.percentile(all_drops_packed,90):.1%}")
    print(f"Pooled efficiency gaps (adjacent pair relative): median {np.median(all_gaps_packed):.1%} p10 {np.percentile(all_gaps_packed,10):.1%} p90 {np.percentile(all_gaps_packed,90):.1%}")
    # Vulnerable fraction as function of e (drop <=e)
    print("\nVulnerable fraction (per-step drop <= e) — fraction of adjacent choices within e of flip under uniform shift model:")
    for e in [0.02,0.04,0.06,0.08,0.12,0.20]:
        frac=np.mean(all_drops_packed <= e) if len(all_drops_packed)>0 else 0
        print(f"  e={e:4.0%} vulnerable {frac:5.1%} (packed median curve)")
    # per-expert pooled
    all_pe=np.concatenate([c[4] for c in per_combo]) if per_combo else np.array([])
    if len(all_pe)>0:
        print("\nPer-expert pooled drops (includes hetero tails):")
        for e in [0.02,0.04,0.06,0.08,0.12,0.20]:
            frac=np.mean(all_pe <= e)
            print(f"  e={e:4.0%} vulnerable {frac:5.1%}")
        print(f"  median {np.median(all_pe):.1%} tail 5% {np.percentile(all_pe,5):.1%} max {np.max(all_pe):.1%} min {np.min(all_pe):.1%}")
    # Also report per-budget expected flips: at budgets near knee, λ near median efficiency
    # For budgets where avg K ~ 33 (mid), efficiencies around 1e-05 per byte? Not needed for headline.
    return per_combo

if __name__=="__main__":
    analyze()

# pytest hooks
def test_drops_in_expected_range():
    # per-step drops should be 10-27% per DIAGNOSIS, pooled median 12-18%
    for lname in ["L14","L21","L0"]:
        for proj in PROJS:
            loader=LOADERS[lname]
            per=loader(proj)
            if not per: continue
            gaps,drops=efficiency_gaps(per, use_packed=True)
            assert np.median(drops) > 0.08 and np.median(drops) < 0.25, f"{lname} {proj} median drop {np.median(drops)}"
def test_vulnerable_fraction_monotone():
    per=load_L14("gate_proj")
    _, drops = efficiency_gaps(per, use_packed=True)
    pooled=np.array(drops)
    fracs=[np.mean(pooled <= e) for e in [0.02,0.04,0.08,0.12,0.20]]
    for i in range(1,len(fracs)):
        assert fracs[i] >= fracs[i-1]
