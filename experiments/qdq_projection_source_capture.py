"""Record actual B1 routed projection inputs/cotangents from original rows.

The source model, streaming path, complete sequences, probe seeds and global
normalization are those of the stopped full joint run. This records evidence;
it neither prices candidates nor adopts the old prepared cache for a new run.
"""
import argparse
import gc
import json
from pathlib import Path
import socket
import time

import torch

from experiments.qdq_constant_residency import NAME, ROOT, sha, tensor_sha
from prismaquant.calibration_data import load_calibration_input
from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
from prismaquant.joint_aura import SignedJointProjectionLease, source_execution_identity
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.model_profiles import detect_profile
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from prismaquant.sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled
from prismaquant.tessera_joint_aura import _source_prefetch
from prismaquant import format_registry as fr


class CaptureLease(SignedJointProjectionLease):
    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if not self.active:
            raise RuntimeError('source capture outside active probe')
        value = x.detach().clone()
        probe, row = self.capture_key

        def record(gradient):
            selected = gradient if row_slice is None else gradient[row_slice]
            selected = selected if output_slice is None else selected[..., output_slice]
            assert value.shape[:-1] == selected.shape[:-1]
            self.records.append({'probe': probe, 'row': row, 'x': value.cpu(),
                                 'g': selected.detach().contiguous().cpu()})
            return gradient

        output.register_hook(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    plan_path = ROOT / 'full-model-joint-aura/plan-02.json'
    plan = json.loads(plan_path.read_text())
    prepared_path = ROOT / 'full-model-joint-aura/run-02/prepare/prepared.json'
    assert sha(prepared_path) == '69a56bff2d29beaf38759b309dee1ac94ca9e091e50f846590d8b51cb0ba5908'
    prepared = json.loads(prepared_path.read_text())
    assert sha(plan_path) == prepared['plan_sha256']
    assert plan['execution']['probe_microbatch'] == 1
    ids, calibration = load_calibration_input(plan['calibration_input']['path'],
        expected_sha256=plan['calibration_input']['sha256'], n_samples=512, seqlen=512)
    assert calibration == prepared['calibration_input']
    receipt = {'schema': 'pq.actual_routed_projection_capture.v1', 'passed': False,
               'env': {'started_epoch': time.time(), 'host': socket.gethostname(),
                       'torch': str(torch.__version__), 'cuda': torch.version.cuda},
               'original_plan_sha256': sha(plan_path), 'calibration': calibration,
               'rows': [0, 1, 2, 3], 'input_ids_sha256': tensor_sha(ids[:4]),
               'global_token_count': ids.numel(), 'seed_base': 7000, 'n_probes': 4,
               'unit': NAME, 'phases': []}

    def save():
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))

    save()
    profile = detect_profile(plan['model'])
    runner = build_streamed_causal_lm(plan['model'], device=torch.device('cuda'), dtype=torch.bfloat16,
        offload_folder=str(args.output / 'offload'), profile=profile, attn_implementation='eager',
        **_source_prefetch(plan))
    try:
        source = build_streamed_model_identity(runner, plan['model'], identity_cache_path=args.output / 'source-identity.json')
        assert source == prepared['source_model_identity']
        execution = source_execution_identity(runner.model)
        assert execution == prepared['source_execution']
        receipt.update(source_model_identity=source, source_execution=execution,
                       source_prefetch=_source_prefetch(plan))
        batches = []
        for row in receipt['rows']:
            batches.append(runner.capture_boundaries(ids[row:row + 1]))
            print(json.dumps({'captured_boundary_row': row}), flush=True)
        layer = 23
        runner.context.install(layer, require_prefetched=runner.require_prefetched_residency)
        members = profile_declared_packed_expert_projections(runner.model, profile=profile)
        target = next(member for member in members if member.qname == NAME)
        receipt['source_weight'] = _cb_cache_tensor_identity(target.weight)
        source_weight = target.weight.detach().cpu().clone()
        target.parameter.requires_grad_(True)

        def clear(parameter):
            parameter.grad = None

        handle = target.parameter.register_post_accumulate_grad_hook(clear)
        lease = CaptureLease({NAME: target}, {NAME: {'BF16': fr.get_format('BF16')}},
                             {(NAME, 'BF16'): torch.zeros_like(target.weight, dtype=torch.float32)})
        lease.records = []
        with lease:
            for probe in range(4):
                lease.begin_probe()
                for row, batch in zip(receipt['rows'], batches):
                    tail = batch.activations_cpu[-1].to(device=runner.device, dtype=runner.dtype).detach().requires_grad_(True)
                    logits = runner.tail_logits(batch, tail)
                    scalar = fisher_probe_scalar(logits, seed=7000 + probe, token_scope='all',
                        temperature=1.0, distribution='rademacher', token_count_override=ids.numel(),
                        global_row_offset=row)
                    scalar.backward()
                    incoming = tail.grad.detach().clone()
                    del tail, scalar, logits
                    x_in = batch.activations_cpu[layer].to(device=runner.device, dtype=runner.dtype).detach().requires_grad_(True)
                    cotangents = SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                    isolated = profile.isolated_layer_pass_state(batch.shared_pass_state, runner.layers[layer])
                    isolated = cotangents.graft(isolated)
                    lease.capture_key = probe, row
                    out = runner.isolated_layer(batch, layer, x_in, pass_state=isolated)
                    roots, root_grads = cotangents.produced_roots()
                    torch.autograd.backward((out, *roots), (incoming, *root_grads))
                    del out, x_in, incoming, isolated, cotangents, roots, root_grads
                lease.finish_probe()
                print(json.dumps({'captured_probe': probe, 'calls': len(lease.records)}), flush=True)
        handle.remove()
        assert len(lease.records) == 16
        assert {(v['probe'], v['row']) for v in lease.records} == {(p, r) for p in range(4) for r in range(4)}
        for record in lease.records:
            assert record['x'].dtype == record['g'].dtype == torch.bfloat16
            assert record['x'].ndim == 2 and record['x'].shape[1] == 2048 and record['g'].shape[1] == 1792
        payload = args.output / 'projection-inputs.pt'
        torch.save({'records': lease.records, 'source_weight': source_weight}, payload)
        receipt['records'] = [{k: v for k, v in record.items() if k not in ('x', 'g')} |
                              {'rows': len(record['x']), 'x_sha256': tensor_sha(record['x']), 'g_sha256': tensor_sha(record['g'])}
                              for record in lease.records]
        receipt['payload'] = {'path': str(payload), 'sha256': sha(payload)}
        receipt.update(passed=True, cuda_max_allocated=torch.cuda.max_memory_allocated(),
                       cuda_max_reserved=torch.cuda.max_memory_reserved())
        receipt['env']['finished_epoch'] = time.time()
        save()
        print(json.dumps({'passed': True, 'records': receipt['records']}), flush=True)
    finally:
        runner.shutdown()
        gc.collect()


if __name__ == '__main__':
    main()
