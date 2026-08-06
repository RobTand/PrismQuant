"""Within-basis PCHIP validation on the 'failing' layer 0.

The archived full-layer cells hold INCUMBENT measurements at all 18
non-adopted rungs. Fit PCHIP through 5 incumbent anchors mirroring the v1
anchor spacing, predict every held-out incumbent rung, and score with the
production audit stat (|pred-truth|/truth, median/p95 over 256 experts).
If these layers' curves are smooth within one basis, medians land under the
5% bar everywhere -- proving the 34% was the encoder-basis gap alone.
"""
import pickle
import statistics

ROOT = "/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq"
N = 256
INCUMBENT_RUNGS = [r for r in range(28, 49) if r not in (28, 33, 38)]
FIT_ANCHORS = (29, 34, 39, 44, 47)      # mirrors v1 five-anchor spacing
HOLDOUT = [r for r in INCUMBENT_RUNGS
           if r not in FIT_ANCHORS and min(FIT_ANCHORS) <= r <= max(FIT_ANCHORS)]


def pchip_slopes(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def pchip_eval(x, y, xq):
    m = pchip_slopes(x, y)
    for i in range(len(x) - 1):
        if x[i] <= xq <= x[i + 1]:
            h = x[i + 1] - x[i]
            t = (xq - x[i]) / h
            return ((2 * t**3 - 3 * t**2 + 1) * y[i]
                    + (t**3 - 2 * t**2 + t) * h * m[i]
                    + (-2 * t**3 + 3 * t**2) * y[i + 1]
                    + (t**3 - t**2) * h * m[i + 1])
    raise ValueError(xq)


def cell(proj, rung):
    p = f"{ROOT}/burn-shards/layer_000_{proj}_v2s-full-layer_K{rung}.pkl"
    return pickle.load(open(p, "rb"))["cell"]


print(f"{'proj':>10} {'rung':>4} {'median':>8} {'p95':>8}  bar: med<=5% p95<=15%")
worst_med, worst_p95, fails = 0.0, 0.0, 0
for proj in ("gate_proj", "up_proj", "down_proj"):
    free = {r: [float(v) for v in cell(proj, r)["free_weight_mse"]]
            for r in INCUMBENT_RUNGS}
    for rq in HOLDOUT:
        rels = []
        for e in range(N):
            pred = pchip_eval(list(FIT_ANCHORS),
                              [free[r][e] for r in FIT_ANCHORS], rq)
            truth = free[rq][e]
            rels.append(abs(pred - truth) / max(abs(truth), 1e-30))
        med = statistics.median(rels)
        p95 = sorted(rels)[int(0.95 * N)]
        worst_med = max(worst_med, med)
        worst_p95 = max(worst_p95, p95)
        bad = med > 0.05 or p95 > 0.15
        fails += int(bad)
        print(f"{proj:>10} K{rq:>3} {med:8.4f} {p95:8.4f}"
              f"{'  <-- FAIL' if bad else ''}")
print(f"\nheld-out cells failing the production bar: {fails}/39")
print(f"worst median {worst_med:.4f}   worst p95 {worst_p95:.4f}")
