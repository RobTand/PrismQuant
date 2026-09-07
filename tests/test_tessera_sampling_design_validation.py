"""Cross-check the production randomized systematic draw on measured anchors."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

from prismaquant.tessera_campaign import draw_stack_sample, _horvitz_thompson_stack


def test_real_stack_fixed_size_pps_point_estimate_and_paired_rungs():
    data = json.loads((Path(__file__).parent / 'fixtures/tessera_stack_lfm_layer18.json').read_text())
    stem = 'model.layers.18.feed_forward.experts'
    h = data['packed_probe_rows'][stem + '.down_proj']['h_trace_per_expert']
    weights = {str(e): x for e, x in enumerate(h)}
    formats = ('TESSERA_BF16_K1_R256', 'TESSERA_BF16_K1_R1142')
    y = [{e: h[e] * next(a['dloss'] for a in data['member_anchors'][f'{stem}.{e}.w2']
                        if a['format_name'] == fmt) for e in range(len(h))} for fmt in formats]
    y.append({e: y[0][e] - y[1][e] for e in range(len(h))})
    trials = 4000
    for n in (8, 16):
        estimates = [[], [], []]
        variances = [[], [], []]
        counts = [0] * len(h)
        for seed in range(trials):
            draw = draw_stack_sample(weights, n, seed=seed, stack=stem + '.down_proj')
            pi = {int(e): p for e, p in draw['inclusion_probability'].items()}
            ids = tuple(int(e) for e in draw['units'])
            assert len(ids) == len(set(ids)) == n
            sample = SimpleNamespace(sampled_experts=ids, inclusion_prob=pi,
                                     packed_qname=stem + '.down_proj')
            for e in ids:
                counts[e] += 1
            for k, contributions in enumerate(y):
                estimate, se, _ = _horvitz_thompson_stack(sample, contributions)
                estimates[k].append(estimate)
                variances[k].append(se * se)
            assert math.isclose(estimates[2][-1], estimates[0][-1] - estimates[1][-1], rel_tol=1e-12)
        for e, p in pi.items():
            assert abs(counts[e] / trials - p) <= 6 * math.sqrt(p * (1-p) / trials) + 1/trials
        metrics = []
        for vals, v, population in zip(estimates, variances, y):
            mean = math.fsum(vals) / trials
            true_total = math.fsum(population.values())
            empirical = math.fsum((value - mean)**2 for value in vals) / (trials - 1)
            assert abs(mean - true_total) < 6 * math.sqrt(empirical / trials)
            metrics.append({'relative_bias': mean / true_total - 1,
                            'empirical_variance': empirical,
                            'mean_reported_variance': math.fsum(v)/trials,
                            'variance_ratio': math.fsum(v)/trials/empirical})
        print(json.dumps({'sample_size': n, 'trials': trials, 'rungs_and_difference': metrics}))
