"""Transparent finalization of the bounded direct export, before its first seal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def supplement_direct_export(args, *, expected_source, producer_image_id,
                             serving_target_image, tessera_commit):
    """Bind the actual direct output without claiming a partition or merged run.

    The original producer manifest is preserved outside the checkpoint. Census
    subsequently hashes and validates the supplemented manifest itself.
    """
    from tessera.serving_parts import source_identity, validate_explicit_plan

    out = args.out
    path = out / 'exported/tessera_serving_manifest.json'
    original = path.read_bytes()
    manifest = json.loads(original)
    _require('export_identity' not in manifest and 'export_partition' not in manifest,
             'direct finalization requires an unsupplemented, non-partition export')
    original_path = out / 'direct-export-manifest.original.json'
    _require(not original_path.exists(), 'direct manifest preservation path already exists')
    plan = _read(out / 'plan.json')
    _require(isinstance(plan, dict) and manifest.get('plan') == plan,
             'direct producer manifest does not bind the actual plan contents')
    _require(manifest.get('source') == str(args.model) and manifest.get('git') == tessera_commit,
             'direct producer source path or frozen Tessera commit differs')
    build_path = out / 'build.json'
    build = _read(build_path)
    command_path = out / 'export.command.json'
    command = _read(command_path)
    argv = command.get('argv')
    _require(command.get('exit_code') == 0 and isinstance(argv, list) and len(argv) == 16,
             'direct export needs its successful exact command receipt')
    # This is the actual fixed export invocation, not reconstructed default
    # argparse values or a claim that the serving image ran the producer.
    hessian = Path(argv[11])
    expected = [str(args.tessera_repo / 'experiments/export_tessera_serving.py'),
                str(args.model), str(out / 'exported'), '--plan-json', str(out / 'plan.json'),
                '--priced-inputs', str(build_path), '--priced-inputs-sha256', _sha(build_path),
                '--hessian', str(hessian), '--cached-expert-units', build['cached_expert_units'],
                '--device', 'cuda']
    _require(argv[1:] == expected, 'direct export command differs from its declared inputs')
    _require(hessian.is_file(), 'direct export Hessian is absent')
    host = _read(out / 'host-status.json')
    snapshot = host.get('source_snapshot')
    ts_source = host.get('tessera_source', {})
    _require(host.get('schema') == 'prismaquant.pq183-host-observation.v1'
             and isinstance(snapshot, str) and re.fullmatch(r'[0-9a-f]{40}', snapshot)
             and host.get('producer_image_id') == producer_image_id
             and host.get('serving_image') == serving_target_image
             and ts_source.get('commit') == tessera_commit
             and re.fullmatch(r'[0-9a-f]{64}', str(ts_source.get('manifest_sha256', ''))),
             'direct finalization host/source/image binding is absent or differs')
    source = source_identity(args.model)
    _require(source == expected_source, 'direct source differs from the priced producer source')
    config = _read(out / 'exported/config.json')
    validate_explicit_plan(plan, manifest['modules'],
                           config['quantization_config']['config_groups'],
                           source_tensors=source['tensors'])
    # The producer's atomic safetensors writer creates mode 0600 files. The
    # next admitted host and serving readers must read these task-owned bytes.
    # Change only regular checkpoint shards, after all provenance gates pass.
    shards = sorted((out / 'exported').glob('*.safetensors'))
    _require(bool(shards), 'direct export has no checkpoint shards')
    modes = {}
    for shard in shards:
        info = shard.lstat()
        _require(stat.S_ISREG(info.st_mode), 'checkpoint shard is not a regular file')
        before = stat.S_IMODE(info.st_mode)
        modes[shard.name] = {'before': oct(before), 'after': '0o644',
                             'sha256_before': _sha(shard)}
    identity = {
        'schema': 'prismaquant.pq183-direct-export-identity.v1',
        'source': source,
        'options': {'plan': plan, 'command_argv': argv,
                    'priced_inputs_sha256': _sha(build_path),
                    'hessian_sha256': _sha(hessian),
                    'cached_expert_units_sha256': _sha(build['cached_expert_units'])},
        'producer_image_id': producer_image_id,
        'serving_target_image': serving_target_image,
        'tessera_source': ts_source,
        'prismaquant_source_snapshot': snapshot,
        'origin': {
            'schema': 'prismaquant.pq183-direct-manifest-supplement.v1',
            'kind': 'direct_export_finalization',
            'original_manifest_sha256': hashlib.sha256(original).hexdigest(),
            'original_manifest_archive': str(original_path),
            'export_command_sha256': _sha(command_path),
            'host_receipt_sha256_at_finalization': _sha(out / 'host-status.json'),
            'campaign_input_manifest_sha256': getattr(args, 'campaign_input_manifest_sha256', None),
            'partitioned': False,
            'checkpoint_shard_readability': modes,
        },
    }
    # Every fallible provenance/obligation check precedes modification. These
    # records are retained even if a subsequent wire audit or serve refuses.
    with original_path.open('xb') as stream:
        stream.write(original)
    for shard in shards:
        shard.chmod(int(modes[shard.name]['after'], 8))
        modes[shard.name]['sha256_after'] = _sha(shard)
        _require(modes[shard.name]['sha256_before'] == modes[shard.name]['sha256_after'],
                 'checkpoint bytes changed during readability finalization')
    augmented = {**manifest, 'export_identity': identity}
    path.write_text(json.dumps(augmented, indent=2, sort_keys=True) + '\n')
    return augmented
