"""Packed probe -> persisted draw -> packed campaign prices -> scoped allocator."""
import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from test_tessera_stack_sample_cost import _profile, _packed_probe_row, _anchor, _campaign, _sample, _payload, STEM


def _plan(tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: _profile())
    data = json.loads((Path(__file__).parent / 'fixtures/tessera_stack_lfm_layer18.json').read_text())
    stats = data['packed_probe_rows']
    probe = tmp_path / 'probe.pkl'
    probe.write_bytes(pickle.dumps({'stats': stats, 'meta': {'model': 'lfm'}}))
    groups = {'s:' + STEM: sorted(data['member_anchors'])}
    (tmp_path / 'census.json').write_text(json.dumps({
        'model': 'lfm', 'layer_stride': 1, 'anchor_groups': groups}))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'model': 'lfm', 'cwd': str(tmp_path),
        'python': 'python', 'env': {}, 'campaign_argv': []}))
    args = SimpleNamespace(spec=spec, workspace=tmp_path, stack_sample=8,
        probe=probe, stack_sample_seed=5, audit_rate=10, groups_per_row=1,
        seed_checkpoint=None, seed_wire_dir=None, rows_per_box=1, timeout_s=300)
    assert dispatch.cmd_plan(args) == 0
    return json.loads((tmp_path / 'units/row-0000.json').read_text()), stats


def test_real_packed_probe_plans_and_rehydrates_shared_projection_draw(tmp_path, monkeypatch):
    from prismaquant import tessera_campaign as campaign
    selection, stats = _plan(tmp_path, monkeypatch)
    samples = campaign.selection_stack_samples(selection, _profile())
    assert set(samples) == set(stats)
    for name, sample in samples.items():
        assert sample.stack_h_trace == stats[name]['h_trace']
        assert sample.h_trace_per_expert == tuple(stats[name]['h_trace_per_expert'])
        assert set(sample.inclusion_prob) == set(range(32))
        assert len(sample.sampled_experts) == 8
        expected = {m for names in sample.members.values() for m in names}
        assert expected <= set(selection['groups'][0]['sampled'])
    # Full-frame metadata survives JSON, including experts outside the draw.
    assert len(selection['groups'][0]['inclusion_probability']) == 96
    anchors = {}
    for sample in samples.values():
        for members in sample.members.values():
            for member in members:
                anchors[member] = {'TESSERA_E2M1_K2': [_anchor(campaign, member,
                    'TESSERA_E2M1_K2', 'TESSERA_E2M1_K2_R896', 896, 0.02)]}
    payload = campaign.campaign_cost_payload(anchors, {}, loo={}, provenance={}, stack_samples=samples)
    assert set(payload['costs']) == set(stats)


def test_selection_rejects_independent_gate_and_up_draws(tmp_path, monkeypatch):
    from prismaquant import tessera_campaign as campaign
    selection, _ = _plan(tmp_path, monkeypatch)
    selection['groups'][0]['sampled'].pop()
    with pytest.raises(campaign.StackSampleError, match='sampled members'):
        campaign.selection_stack_samples(selection, _profile())


def test_a_stack_payload_allocates_through_the_real_allocator(tmp_path, monkeypatch):
    """Criterion 4 end to end: allocator.main() places the STACK, scoped.

    The probe is the packed probe -- not the per-expert expansion #290
    retired -- so the packed qname is a row the candidate builder can see, the
    explicit Tessera scope resolves it as ``routed_moe`` with no fallback to
    the serving profile, and the layer config names the stack, never a member.
    """
    import pickle
    import sys

    from test_tessera_scope_endpoints import _cli_scope, _v5_contract

    from prismaquant import allocator

    campaign = _campaign()
    _v5_contract(monkeypatch)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}))

    fmt, family, q256 = "TESSERA_E2M1_K2_R896", "TESSERA_E2M1_K2", 896
    gate_up = f"{STEM}.gate_up_proj"
    stats = {
        gate_up: _packed_probe_row(4, [1.0, 2.0, 3.0, 4.0],
                                   out_features=512, in_features=256),
        "model.layers.18.self_attn.out_proj": {
            "h_trace": 1.0, "n_params": 256 * 256, "in_features": 256,
            "out_features": 256, "router_path": None, "expert_id": None},
    }
    probe = tmp_path / "probe.pkl"
    probe.write_bytes(pickle.dumps(
        {"stats": stats, "meta": {"model": str(model_dir)}}))

    anchors = {
        f"{STEM}.{e}.{role}": {family: [
            _anchor(campaign, f"{STEM}.{e}.{role}", family, fmt, q256, 0.02)]}
        for e in range(4) for role in ("w1", "w3")}
    sample = _sample(campaign, stats[gate_up])
    payload = _payload(campaign, anchors, {sample.packed_qname: sample})
    # The dense unit is priced by the same table so the run has one currency.
    payload["costs"]["model.layers.18.self_attn.out_proj"] = {
        fmt: {"output_mse": 4e-4, "output_mse_measured": True,
              "currency": campaign.CURRENCY}}
    payload["formats"] = sorted(set(payload["formats"]) | {fmt})
    costs = tmp_path / "cost.pkl"
    costs.write_bytes(pickle.dumps(payload))

    layer_config = tmp_path / "layer.json"
    from prismaquant import allocator_candidates
    seen = {}
    original = allocator_candidates.selection_serving_lane_provenance

    def provenance(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(allocator, "selection_serving_lane_provenance", provenance)
    monkeypatch.setattr(sys, "argv", [
        "allocator", "--probe", str(probe), "--costs", str(costs),
        "--formats", fmt, "--allow-legacy-fisher-norm",
        "--target-profile", "tessera_research_sm121", "--target-bits", "16",
        "--pareto-targets", "16", "--bit-precision", "0.1",
        "--layer-config", str(layer_config),
        "--pareto-csv", str(tmp_path / "pareto.csv"), *_cli_scope()])
    allocator.main()

    assert seen["context_by_unit"][gate_up].structure == "routed_moe"
    written = json.loads(layer_config.read_text())
    assert gate_up in written
    assert f"{STEM}.0.w1" not in written
    route = written["__prismaquant__"]["serving_lane_provenance"]["by_unit"][gate_up]
    assert route["format"] == fmt
    assert route["serving_context"]["structure"] == "routed_moe"


def test_draw_receipt_must_replay_from_the_carried_probe(tmp_path, monkeypatch):
    from prismaquant import tessera_campaign as campaign
    selection, _ = _plan(tmp_path, monkeypatch)
    record = next(iter(selection['groups'][0]['stack_samples'].values()))
    record['seed'] += 1
    with pytest.raises(campaign.StackSampleError, match='does not replay'):
        campaign.selection_stack_samples(selection, _profile())


def test_checkpoint_binds_fisher_and_probability_values_with_same_measured_units(monkeypatch):
    from prismaquant import tessera_campaign as campaign
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    monkeypatch.setattr(campaign, '_checkpoint_identity_api', lambda: SimpleNamespace(
        encoder_source_sha256=lambda: 'encoder', tensor_identity=lambda t: t))
    monkeypatch.setattr(campaign.th, 'encoder_recipe', lambda: {})
    monkeypatch.setattr('prismaquant.production_weight_cache._production_cache_source_sha256', lambda: 'pq')
    common = dict(weights={'a': 'tensor'}, acts={}, hessians={}, menus={'a': []},
        args=SimpleNamespace(units='/a.json'), calibration_identity={}, serving_scope=None,
        static_scales={}, static_scale_policy='policy')
    record = {'stack': {'probe_row': {'h_trace': 4.0}, 'inclusion_prob': {'0': 0.5}}}
    first = campaign._campaign_checkpoint_identity(**common, stack_sampling_identity=record)
    changed = copy.deepcopy(record)
    changed['stack']['inclusion_prob']['0'] = 0.75
    second = campaign._campaign_checkpoint_identity(**common, stack_sampling_identity=changed)
    assert first['units'] == second['units']
    assert canonical_json_sha256(first, where="first") != canonical_json_sha256(second, where="second")


def test_campaign_main_consumes_persisted_draw_and_emits_packed_population(tmp_path, monkeypatch):
    """Real producer projection/capture/receipts, with the established CPU fixture."""
    import dispatch_tessera_campaign as dispatch
    from test_tessera_campaign_packed import _bridge_main_fixture, STACK, RUNG
    campaign, argv, model, encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    profile = _profile()
    population = campaign._require_campaign_population(model, profile, 1)
    dense = 'model.layers.0.attention'
    groups = campaign.resolve_anchor_groups([dense, *population.qnames], profile=profile,
        expert_members={member.qname: member for member in population.members})
    probe_rows = {}
    for name, shape in population.packed_in_scope.items():
        probe_rows[name] = _packed_probe_row(shape[0], [1.0] * shape[0],
            packed_param=name.rsplit('.', 1)[-1], out_features=shape[1], in_features=shape[2])
        probe_rows[name]['_packed_experts_module'] = STACK
    sampled = dispatch.sample_stack_groups(groups, probe_rows, profile=profile,
        stack_sample=2, seed=5, audit_rate=10)
    selection = {'schema': campaign.UNITS_SCHEMA_V2, 'model': argv[1], 'layer_stride': 1,
        'groups': [{'key': key, 'members': members, **sampled.get(key, {})}
                   for key, members in sorted(groups.items())]}
    path = tmp_path / 'units.json'
    path.write_text(json.dumps(selection))
    assert campaign.main([*argv, '--units', str(path)]) == 0
    payload = pickle.loads((tmp_path / 'cost.pkl').read_bytes())
    assert set(payload['costs']) == {dense, *probe_rows}
    assert payload['provenance']['population']['priced']['routed_experts'] == sorted(probe_rows)
    assert len(payload['tessera_expert_wires']) == 6
    assert payload['provenance']['unit_selection']['groups'] == selection['groups']
    checkpoint = json.loads((tmp_path / 'cost.anchors.json').read_text())
    assert set(checkpoint['identity']['stack_sampling_identity']) == set(probe_rows)
    for packed in probe_rows:
        assert set(payload['costs'][packed]) == {RUNG}
