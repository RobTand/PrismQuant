"""PB-only checks of retained #323 producer artifacts; no timing/admission claim.

This explicitly selected experiment is outside ordinary test discovery. Shared
artifacts are hash-bound by the retained relation. The current pair must refuse:
it differs in physical GPU and one generated production library's bytes.
"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from prismaquant.measured_runtime_prices import load_runtime_context, RuntimePriceError
from prismaquant.runtime_provenance import (
    ArtifactReader, _observe_run, _package_source, load_runtime_relation,
)

ROOT = Path('/mnt/shared/tessera-native376-resource/runtime-provenance-323')
RELATION = ROOT / 'relation-prepare05-engine-r4.json'


def test_retained_source_archive_and_each_individual_runtime():
    relation = json.loads(RELATION.read_text())
    reader = ArtifactReader(ROOT)
    source = _package_source(relation['package_source'], reader)
    assert source['source_tree_sha256'] == '57809bff862b880dc397e6d271a80c04d6c87d1af3bba12c076648fc5443355c'
    assert source['installed_source_sha256'] == '8239d56568b0b2298a05c61b7a2dc0c85b7fc76a25c4bb6dae0fe2d7ecdf428b'
    _, configuration = reader.json(relation['configuration'], 'original selected configuration')
    _, manifest = reader.json(relation['image_manifest'], 'original image manifest')
    image = {'manifest_digest': 'sha256:' + relation['image_manifest']['sha256'],
             'config_digest': manifest['config']['digest']}
    context = load_runtime_context(ROOT / 'context-panel03.json')
    for name, run in relation['runs'].items():
        _, envelope = reader.json(run['runtime'], 'original runtime')
        raw = envelope if run['runtime_field'] is None else envelope['runtime']
        base = raw.get('base', raw)
        # Independently check each original device; this is NOT a relation.
        observed = _observe_run(run, reader=reader, configuration=configuration,
            configuration_sha256=relation['configuration']['sha256'], image_manifest=image,
            package_source=source, context=replace(context, gpu_identity=base['gpu']['uuid']))
        assert observed['raw'] == raw
        assert observed['common']['package_sha256'] == source['installed_source_sha256']


def test_retained_cross_device_pair_refuses():
    raw = RELATION.read_bytes()
    reference = {'path': str(RELATION), 'sha256': hashlib.sha256(raw).hexdigest()}
    with pytest.raises(RuntimePriceError, match='actual GPU UUID'):
        load_runtime_relation(reference, context=load_runtime_context(ROOT / 'context-panel03.json'), root=ROOT)
