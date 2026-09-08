"""Tiny original-layout GLM source gate; execute in a PB-admitted GPU image."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['PRISMAQUANT_TMPDIR'] = str(args.out/'staging')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from test_glm5_next_streamed_forward_parity import _tiny_config, _build_model
    from test_glm_campaign_streaming import write_original_layout_checkpoint
    from transformers import Glm5NextForConditionalGeneration
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.tessera_campaign import _collect_activations
    assert torch.cuda.is_available(), 'GPU source gate requires CUDA'
    config = _tiny_config()
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.moe_intermediate_size = 128
    config.text_config.linear_attn_config['head_dim'] = 64
    config.text_config.qk_nope_head_dim = 64
    config.text_config.v_head_dim = 64
    config.vision_config.out_hidden_size = 256
    config = type(config).from_dict(config.to_dict())
    torch.manual_seed(20260826)
    source = args.out/'source'
    model = _build_model(config).to(torch.bfloat16)
    # The real source stores recurrence/router controls in FP32. Nonzero
    # values make accidental downcasts observable instead of testing zeros.
    with torch.no_grad():
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith(('dt_bias', 'A_log', 'e_score_correction_bias')):
                tensor.data = tensor.data.float()
                tensor.copy_(torch.randn_like(tensor)*0.05)
    write_original_layout_checkpoint(model, source)
    del model
    model = Glm5NextForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation='eager').eval().cuda()
    profile = Glm5NextProfile()
    dense = [name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and '.layers.' in name
             and not profile.is_pinned_name(name)]
    targets = [*dense, *(m.qname for m in profile_declared_packed_expert_projections(model, profile))]
    torch.manual_seed(441)
    tokens = [torch.randint(2, 128, (1, 257)), torch.randint(2, 128, (1, 257))]
    stopped = threading.Event()
    errors = []
    def monitor():
        with (args.out/'netdata.jsonl').open('w') as out:
            while not stopped.is_set():
                for host in ('sparky', 'sparklina'):
                    try:
                        with urllib.request.urlopen(f'http://{host}:19999/api/v1/allmetrics?format=json', timeout=5) as response:
                            metrics = json.load(response)
                        out.write(json.dumps(dict(host=host, time=time.time(), metrics=metrics))+'\n')
                        out.flush()
                    except Exception as exc:
                        errors.append(dict(host=host, error=repr(exc)))
                stopped.wait(2)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    results = {'torch':torch.__version__, 'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(), 'dtype':'bfloat16', 'samples':2, 'seqlen':257,
        'prefix_rows':7, 'units':len(targets), 'telemetry_errors':errors, 'arms':{}}
    def run_profile(name, call):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            value = call()
        torch.cuda.synchronize()
        results['arms'][name] = dict(start=start, end=time.time(),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        prof.export_chrome_trace(str(args.out/(name+'.trace.json')))
        (args.out/(name+'.profile.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        return value
    runner = None
    try:
        with torch.inference_mode():
            expected_logits = [model(input_ids=ids.cuda(), use_cache=False).logits.cpu() for ids in tokens]
            baseline = run_profile('source', lambda: _collect_activations(model, targets, tokens, 7, 'cuda', want_hessian=True, profile=profile))
            reference_state = {name: tensor.detach().cpu() for name, tensor in [*model.named_parameters(), *model.named_buffers()]}
            del model
            torch.cuda.empty_cache()
            runner = build_streamed_causal_lm(str(source), device=torch.device('cuda'), dtype=torch.bfloat16,
                offload_folder=str(args.out/'offload'), profile=profile, max_cache_slots=2,
                prefetch_workers=1, prefetch_min_available_gb=0, cache_headroom_gb=0,
                prefetch_lookahead=1, require_prefetched_residency=True, attn_implementation='eager')
            runner.context.begin_source_initialization_audit()
            captured = [{}, {}, {}, {}]
            actual_logits = []
            results['state_mismatches'] = {}
            def visit(layer, forward_batch):
                for name, tensor in [*runner.model.named_parameters(), *runner.model.named_buffers()]:
                    if not name.startswith(f'{runner.context.layers_prefix}{layer}.'):
                        continue
                    assert not tensor.is_meta, name
                    if name not in reference_state:
                        continue
                    expected = reference_state[name]
                    if tensor.dtype != expected.dtype or not torch.equal(tensor.cpu(), expected):
                        results['state_mismatches'][name] = dict(actual_dtype=str(tensor.dtype), expected_dtype=str(expected.dtype), max_abs=float((tensor.cpu().float()-expected.float()).abs().max()))
                names = [n for n in targets if runner.layer_index_for_qname(n) == layer]
                local = _collect_activations(runner.model, names, tokens, 7, 'cuda',
                    want_hessian=True, profile=profile, forward_batch=forward_batch)
                for full, part in zip(captured, local):
                    full.update(part)
            run_profile('streamed', lambda: runner.visit_layer_batches(tokens, visit,
                output_consumer=lambda _, logits: actual_logits.append(logits.cpu())))
            results['count_mismatches'] = {n:[baseline[2][n],captured[2][n]] for n in baseline[2] if baseline[2][n] != captured[2][n]}
            results['amax_mismatches'] = {n:[baseline[3][n],captured[3][n]] for n in baseline[3] if baseline[3][n] != captured[3][n]}
            results['tensor_mismatches'] = {str(kind):{n:float((captured[kind][n]-baseline[kind][n]).abs().max()) if captured[kind][n].shape==baseline[kind][n].shape else 'shape mismatch' for n in baseline[kind] if not torch.equal(captured[kind][n],baseline[kind][n])} for kind in (0,1)}
            results['logit_max_abs'] = [float((a-b).abs().max()) for a,b in zip(actual_logits,expected_logits)]
            assert not results['state_mismatches'], results['state_mismatches']
            assert captured[2:] == list(baseline[2:]), 'counts or amax differ'
            for actual, expected in zip(captured[:2], baseline[:2]):
                assert actual.keys() == expected.keys()
                for name in actual:
                    assert torch.equal(actual[name], expected[name]), name
            for actual, expected in zip(actual_logits, expected_logits):
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            results['initialization'] = runner.context.source_initialization_contract()
            results['exact_x_h_counts_amax_logits'] = True
    finally:
        if runner is not None:
            runner.shutdown()
        stopped.set()
        monitor_thread.join(timeout=12)
        (args.out/'result.json').write_text(json.dumps(results, indent=2)+'\n')
    assert not errors, errors
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
