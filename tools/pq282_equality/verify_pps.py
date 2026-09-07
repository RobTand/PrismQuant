"""Check the sampler's arithmetic without importing the tessera runtime."""
import ast, pathlib, types, collections, math, statistics, pickle, re

src = pathlib.Path("/home/rob/tessera-runs/pq282/prismaquant/tessera_campaign.py").read_text()
tree = ast.parse(src)
want = {"_keystream", "_permute", "draw_stack_sample", "audit_subsample"}
mod = ast.Module(body=[n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name in want],
                 type_ignores=[])
ns = {"Mapping": dict, "Sequence": list}
exec(compile(mod, "<sampler>", "exec"), ns)
draw = ns["draw_stack_sample"]; audit = ns["audit_subsample"]

# Real h_trace from campaign-01, layer 18 w1.
o = pickle.loads(pathlib.Path("/mnt/shared/tessera-measurements/pq275-2026-09-06/campaign-01/probe_expanded.pkl").read_bytes())
h = {n: v["h_trace"] for n, v in o["stats"].items() if re.search(r"experts\.\d+\.w1$", n)}
print(f"frame E={len(h)}  CV(h)={statistics.pstdev(list(h.values()))/statistics.mean(list(h.values())):.3f}")

for n_s in (4, 8, 16):
    d = draw(h, n_s, seed=0, stack="s:L18|w1")
    tot = sum(d["inclusion_probability"].values())
    print(f"n_s={n_s:2d} drawn={len(d['units']):2d} certainty={len(d['certainty']):2d} "
          f"random={d['random_draws']:2d} sum(pi)={tot:.12f} "
          f"max(pi)={max(d['inclusion_probability'].values()):.4f} "
          f"audit={audit(d['units'], rate=10, seed=0, stack='s:L18|w1')}")
    assert abs(tot - n_s) < 1e-9 and len(d["units"]) == n_s

# Closed form pi_i = min(1, c*h_i) with sum = n.
d = draw(h, 8, seed=0, stack="s:L18|w1")
lo, hi = 0.0, 1e9
for _ in range(200):
    c = (lo + hi) / 2
    if sum(min(1.0, c * v) for v in h.values()) < 8: lo = c
    else: hi = c
err = max(abs(d["inclusion_probability"][k] - min(1.0, c * h[k])) for k in h)
print(f"closed-form pi agreement: max abs err {err:.3e}")
assert err < 1e-6

# Empirical first-order inclusion frequency == pi, over seeds.
trials = 4000
count = collections.Counter()
for seed in range(trials):
    for name in draw(h, 8, seed=seed, stack="s:L18|w1")["units"]:
        count[name] += 1
pi = draw(h, 8, seed=0, stack="s:L18|w1")["inclusion_probability"]
worst = max((abs(count[k]/trials - pi[k]), k) for k in h)
print(f"empirical inclusion vs pi over {trials} seeds: worst |dev| {worst[0]:.4f} at {worst[1].split('.')[-2:]}")
assert worst[0] < 0.035, worst

# HT unbiasedness on a synthetic mse, averaged over seeds.
import random
rng = random.Random(1234)
mse = {k: math.exp(rng.gauss(0, 0.45)) for k in h}
T = sum(h[k] * mse[k] for k in h)
ests = []
for seed in range(trials):
    d = draw(h, 8, seed=seed, stack="s:L18|w1")
    p = d["inclusion_probability"]
    ests.append(sum(h[k] * mse[k] / p[k] for k in d["units"]))
mean = statistics.mean(ests)
rse = statistics.pstdev(ests) / T
print(f"HT: bias {100*(mean-T)/T:+.3f}%  RSE {100*rse:.2f}%  (n_s=8, E={len(h)}, CV(mse)=0.45)")
assert abs(mean - T) / T < 0.01

# Refusals.
for args, msg in (((h, 1), "one randomly drawn"),):
    try:
        draw(*args, seed=0, stack="s:x")
        print("NO REFUSAL for", args[1])
    except RuntimeError as e:
        print("refused n_s=1:", str(e)[:80])
print("ALL CHECKS PASS")
