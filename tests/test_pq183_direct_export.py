"""Portable direct-finalization fixtures; actual producer gates run in PB GPU continuation."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class DirectExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'out'
        self.exported = self.out / 'exported'
        self.exported.mkdir(parents=True)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.ts = self.root / 'ts'
        self.commit = 'a' * 40
        self.producer = 'sha256:' + 'b' * 64
        self.target = 'example/runtime@sha256:' + 'c' * 64
        self.args = types.SimpleNamespace(out=self.out, model=self.model, tessera_repo=self.ts,
                                         campaign_input_manifest_sha256='d' * 64)
        self.source = {'config_sha256': 'e' * 64, 'files': {'model.safetensors': 'f' * 64},
                       'auxiliary_sha256': {}, 'tensors': {'body.weight': 'model.safetensors'}}
        self.plan = {'body.weight': 'BF16', 'model.layers.13.feed_forward.experts':
                     {'grid': 'E4M3', 'q256': 1024, 'source_layout': 'unpacked'}}
        self.dump('plan.json', self.plan)
        self.dump('exported/config.json', {'quantization_config': {'config_groups': {'group': {}}}})
        self.manifest = {'source': str(self.model), 'git': self.commit, 'plan': self.plan,
                         'modules': {'model.layers.13.feed_forward.experts': {'fixture': True}}}
        self.dump('exported/tessera_serving_manifest.json', self.manifest)
        self.dump('cached.json', {'fixture': 'validated transport'})
        self.dump('build.json', {'cached_expert_units': str(self.out / 'cached.json')})
        (self.out / 'hessian.pt').write_bytes(b'owned fixture H')
        argv = ['python3', str(self.ts / 'experiments/export_tessera_serving.py'),
                str(self.model), str(self.exported), '--plan-json', str(self.out / 'plan.json'),
                '--priced-inputs', str(self.out / 'build.json'), '--priced-inputs-sha256',
                self.sha('build.json'), '--hessian', str(self.out / 'hessian.pt'),
                '--cached-expert-units', str(self.out / 'cached.json'), '--device', 'cuda']
        self.dump('export.command.json', {'argv': argv, 'exit_code': 0})
        self.dump('host-status.json', {'schema': 'prismaquant.pq183-host-observation.v1',
                  'source_snapshot': '1' * 40, 'producer_image_id': self.producer,
                  'serving_image': self.target, 'tessera_source':
                  {'commit': self.commit, 'manifest_sha256': '2' * 64, 'files': 1017}})
        self.parts = types.ModuleType('tessera.serving_parts')
        self.parts.source_identity = lambda path: (
            {'manifest_sha256': self.sha('exported/tessera_serving_manifest.json')}
            if Path(path) == self.exported else copy.deepcopy(self.source))
        self.gate_calls = []
        self.parts.validate_explicit_plan = lambda *a, **kw: self.gate_calls.append((a, kw))
        self.modules = patch.dict('sys.modules', {'tessera.serving_parts': self.parts})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.ns = runpy.run_path(str(ROOT / 'experiments/pq183_direct_export.py'))

    def dump(self, path, obj):
        (self.out / path).write_text(json.dumps(obj))

    def sha(self, path):
        return hashlib.sha256((self.out / path).read_bytes()).hexdigest()

    def supplement(self, **changes):
        options = dict(expected_source=self.source, producer_image_id=self.producer,
                       serving_target_image=self.target, tessera_commit=self.commit)
        options.update(changes)
        return self.ns['supplement_direct_export'](self.args, **options)

    def refuses_without_mutation(self, message, **changes):
        before = self.sha('exported/tessera_serving_manifest.json')
        with self.assertRaisesRegex(ValueError, message):
            self.supplement(**changes)
        self.assertEqual(before, self.sha('exported/tessera_serving_manifest.json'))
        self.assertFalse((self.out / 'direct-export-manifest.original.json').exists())

    def test_direct_manifest_keeps_original_and_distinct_images(self):
        before = (self.exported / 'tessera_serving_manifest.json').read_bytes()
        result = self.supplement()
        identity = result['export_identity']
        self.assertEqual((self.out / 'direct-export-manifest.original.json').read_bytes(), before)
        self.assertEqual(identity['producer_image_id'], self.producer)
        self.assertEqual(identity['serving_target_image'], self.target)
        self.assertNotIn('runtime_image', identity)
        self.assertNotIn('export_partition', result)
        self.assertEqual(identity['options']['plan'], self.plan)
        self.assertEqual(identity['source'], self.source)
        self.assertEqual(self.gate_calls[0][1]['source_tensors'], self.source['tensors'])
        self.assertEqual(json.loads((self.exported / 'tessera_serving_manifest.json').read_text()), result)

    def test_plan_drift_refuses_before_manifest_mutation(self):
        bad = copy.deepcopy(self.manifest)
        bad['plan'] = {}
        self.dump('exported/tessera_serving_manifest.json', bad)
        before = self.sha('exported/tessera_serving_manifest.json')
        with self.assertRaisesRegex(ValueError, 'actual plan'):
            self.supplement()
        self.assertEqual(before, self.sha('exported/tessera_serving_manifest.json'))
        self.assertFalse((self.out / 'direct-export-manifest.original.json').exists())

    def test_source_drift_refuses(self):
        self.refuses_without_mutation('priced producer source', expected_source={})

    def test_producer_plan_gate_failure_is_preserved(self):
        def refuse(*args, **kwargs):
            raise ValueError('actual producer explicit-plan obligation refused')
        self.parts.validate_explicit_plan = refuse
        with self.assertRaisesRegex(ValueError, 'explicit-plan obligation'):
            self.supplement()
        self.assertNotIn('export_identity', json.loads((self.exported / 'tessera_serving_manifest.json').read_text()))

    def test_failed_or_changed_export_command_refuses(self):
        p = self.out / 'export.command.json'
        original = json.loads(p.read_text())
        for case in ('exit', 'input'):
            record = copy.deepcopy(original)
            if case == 'exit':
                record['exit_code'] = 1
            else:
                record['argv'][9] = '0' * 64
            self.dump('export.command.json', record)
            self.refuses_without_mutation('command')

    def test_misbound_host_image_refuses(self):
        self.refuses_without_mutation('host/source/image', producer_image_id=self.target)

    def test_partition_and_double_finalization_refuse(self):
        self.dump('exported/tessera_serving_manifest.json', {**self.manifest, 'export_partition': {}})
        with self.assertRaisesRegex(ValueError, 'non-partition'):
            self.supplement()
        self.dump('exported/tessera_serving_manifest.json', self.manifest)
        self.supplement()
        with self.assertRaisesRegex(ValueError, 'unsupplemented'):
            self.supplement()

    def test_existing_seal_accepts_direct_manifest_after_finalization(self):
        ns = runpy.run_path(str(ROOT / 'experiments/pq183_lfm_bound.py'))
        self.dump('layer_config.json', {'__prismaquant__': {
            'tessera_expert_projection': {'producer': {'source': self.source}}}})
        with patch.dict(ns['seal'].__globals__, wire_audit=lambda _: {'passed': True},
                        PRODUCER_IMAGE=self.producer, IMAGE=self.target, TESSERA_COMMIT=self.commit):
            ns['seal'](self.args)
        seal = json.loads((self.out / 'artifact-seal.json').read_text())
        self.assertEqual(seal['export_identity']['source'], self.source)
        self.assertEqual(seal['checkpoint_identity']['manifest_sha256'],
                         self.sha('exported/tessera_serving_manifest.json'))
        self.assertTrue((self.out / 'direct-export-manifest.original.json').is_file())


if __name__ == '__main__':
    unittest.main()
