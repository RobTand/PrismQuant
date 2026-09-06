"""Portable preparation guards; actual producer/capture are separate PB actions."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments import lfm_mixed_preparation as prep


def fixture():
    dense = {}
    for layer in range(24):
        prefix = f'model.layers.{layer}'
        if layer in [2, 6, 10, 14, 18, 21]:
            for leaf, rows in [('q_proj', 2048), ('k_proj', 512), ('v_proj', 512), ('out_proj', 2048)]:
                dense[f'{prefix}.self_attn.{leaf}.weight'] = (rows, 2048)
        else:
            dense[f'{prefix}.conv.in_proj.weight'] = (6144, 2048)
            dense[f'{prefix}.conv.out_proj.weight'] = (2048, 2048)
        if layer < 2:
            for role in (1, 2, 3):
                dense[f'{prefix}.feed_forward.w{role}.weight'] = (2048, 7168) if role == 2 else (7168, 2048)
        else:
            dense[f'{prefix}.feed_forward.gate.weight'] = (32, 2048)
    stacks = {f'model.layers.{i}.feed_forward.experts': {} for i in range(2, 24)}
    routed = {f'{s}.{expert}.w{role}.weight': (1792, 2048) for s in stacks
              for expert in range(32) for role in (1, 2, 3)}
    def fused(n):
        if re.search(r'\.self_attn\.(q_proj|k_proj|v_proj)\.weight$', n):
            root = n.rsplit('.', 2)[0]
            return root + '.qkv_proj', [root + '.' + x + '.weight' for x in ('q_proj', 'k_proj', 'v_proj')]
        if re.search(r'\.feed_forward\.(w1|w3)\.weight$', n):
            root = n.rsplit('.', 2)[0]
            return root + '.w13', [root + '.' + x + '.weight' for x in ('w1', 'w3')]
    def project(shapes, config, request):
        return {'schema': 'tessera.expert_projection.v1', 'stacks': {
            s: {**request[s], 'units': [{'tensor': n} for n in routed if n.startswith(s + '.')]}
            for s in request}}
    api = types.SimpleNamespace(MOE_ROUTER=re.compile(r'.*\.feed_forward\.gate\.weight'),
        MOE_SOURCE_UNPACKED='producer_owned_layout', expert_stacks=lambda r: stacks,
        packed_expert_stacks=lambda p: {}, project_expert_plan=project,
        fused_module=fused, module_of=lambda n: n.removesuffix('.weight'))
    config = {'architectures': ['Lfm2MoeForCausalLM'], 'num_hidden_layers': 24, 'num_experts': 32}
    return api, dense, {}, routed, config


class PlanTests(unittest.TestCase):
    def test_full_body_uses_real_trellis16_and_producer_layout(self):
        plan, projection, population = prep.mixed_plan(*fixture())
        self.assertEqual(sum(s['grid'] == 'E2M1x2' for s in plan.values()), 6)
        self.assertEqual(sum(s['grid'] == 'BF16' for s in plan.values()), 60)
        self.assertEqual(sum(s['grid'] == 'E4M3' for s in plan.values()), 22)
        self.assertEqual(population['served_owners'], 74)
        self.assertEqual(len(population['retained_router_tensors']), 22)
        self.assertTrue(all(s['source_layout'] == 'producer_owned_layout' for s in projection['stacks'].values()))
        self.assertFalse(set(plan) & set(population['retained_router_tensors']))

    def test_missing_source_expert_cannot_look_complete(self):
        args = list(fixture())
        args[3] = dict(args[3]); args[3].pop(next(iter(args[3])))
        with self.assertRaisesRegex(ValueError, 'lost or duplicated'):
            prep.mixed_plan(*args)

    def test_duplicate_projected_expert_refuses(self):
        args = list(fixture()); old = args[0].project_expert_plan
        def duplicate(*a):
            obj = old(*a); stack = next(iter(obj['stacks'].values()))
            stack['units'][1] = stack['units'][0]
            return obj
        args[0].project_expert_plan = duplicate
        with self.assertRaisesRegex(ValueError, 'lost or duplicated'):
            prep.mixed_plan(*args)

    def test_incomplete_fused_group_refuses(self):
        args = list(fixture()); old = args[0].fused_module
        args[0].fused_module = lambda n: ('bad', [n, 'absent.weight']) if n in prep.DENSE4 else old(n)
        with self.assertRaisesRegex(ValueError, 'incomplete fused'):
            prep.mixed_plan(*args)

    def test_cross_family_fusion_refuses(self):
        args = list(fixture()); old = args[0].fused_module
        args[0].fused_module = lambda n: ('bad', [n, 'model.layers.0.conv.in_proj.weight']) if n in prep.DENSE4 else old(n)
        with self.assertRaisesRegex(ValueError, 'mixed-family'):
            prep.mixed_plan(*args)

    def test_dense_geometry_refuses_without_passthrough(self):
        args = list(fixture()); args[1][next(iter(prep.DENSE4))] = (63, 2048)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            prep.mixed_plan(*args)


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.pop = prep.mixed_plan(*fixture())[2]
        self.names = {n.removesuffix('.weight') for n in prep.DENSE4}
        self.maximum = {n: 12.0 for n in self.names}
        self.counts = {n: 16384 for n in self.names}
        self.scales = {n: 224.0 for n in self.names}
        self.calibration = {'nsamples': 32, 'seqlen': 512, 'seed': 0,
                            'token_sha256': 'a' * 64, 'corpus_sha256': 'b' * 64}

    def record(self):
        return prep.scale_record(self.maximum, self.counts, self.scales,
                                 'fixture-policy', self.pop, calibration=self.calibration)

    def test_scale_policy_receives_producer_fusion_for_lfm_names(self):
        profile = prep.dense_scale_profile(self.pop)
        self.assertEqual(profile.fused_sibling_group('model.layers.0.feed_forward.w1'),
                         profile.fused_sibling_group('model.layers.0.feed_forward.w3'))
        self.assertNotEqual(profile.fused_sibling_group('model.layers.0.feed_forward.w1'),
                            profile.fused_sibling_group('model.layers.0.feed_forward.w2'))
        self.assertIsNone(profile.fused_sibling_group('not.a.planned.unit'))

    def test_scale_receipt_preserves_all_rows_and_actual_maximum(self):
        r = self.record()
        self.assertEqual(r['max_abs'], self.maximum)
        self.assertEqual(r['rows'], self.counts)
        self.assertFalse(r['hessian_captured'])
        self.assertFalse(r['expert_rows_claimed'])
        self.assertEqual(r['calibration'], self.calibration)

    def test_wrong_draw_refuses(self):
        self.calibration['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'calibration identity'):
            self.record()

    def test_prefix_only_scale_observation_refuses(self):
        self.counts[next(iter(self.names))] = 512
        with self.assertRaisesRegex(ValueError, 'incomplete dense'):
            self.record()

    def test_missing_scale_refuses(self):
        self.scales.pop(next(iter(self.names)))
        with self.assertRaisesRegex(ValueError, 'roster'):
            self.record()

    def test_nonfinite_maximum_refuses(self):
        self.maximum[next(iter(self.names))] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid calibrated'):
            self.record()

    def test_fused_scale_drift_refuses(self):
        self.scales['model.layers.0.feed_forward.w1'] = 1.0
        with self.assertRaisesRegex(ValueError, 'fused siblings'):
            self.record()


class SourceSealTests(unittest.TestCase):
    def test_complete_source_seal_refuses_changed_and_extra_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / 'source'; root.mkdir(); (root / 'code.py').write_text('sealed\n')
            manifest = Path(t) / 'manifest.json'
            manifest.write_text(json.dumps({'schema': 'prismaquant.lfm-mixed-tessera-source.v1',
                'commit': prep.TS_COMMIT, 'files': {'code.py': prep.sha(root / 'code.py')}}))
            digest = prep.sha(manifest)
            self.assertEqual(prep.verify_source(root, manifest, digest)['files'], 1)
            (root / 'extra.py').write_text('unsealed')
            with self.assertRaisesRegex(ValueError, 'full source roster'):
                prep.verify_source(root, manifest, digest)
            (root / 'extra.py').unlink(); (root / 'code.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'full source roster'):
                prep.verify_source(root, manifest, digest)


if __name__ == '__main__':
    unittest.main()
