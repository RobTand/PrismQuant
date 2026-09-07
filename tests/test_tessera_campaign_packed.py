"""A campaign must account for the live packed expert population."""
from types import ModuleType, SimpleNamespace
import json
import pickle
import sys

import pytest

torch = pytest.importorskip("torch")


class Lfm2MoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.arange(64.).reshape(2, 8, 4) / 64)
        self.down_proj = torch.nn.Parameter(torch.arange(32.).reshape(2, 4, 4) / 32)
        self.num_experts = 2
        self.act_fn = torch.nn.functional.silu
        self.observed = {0: {"gate_up": [], "down": []},
                         1: {"gate_up": [], "down": []}}

    def forward(self, hidden, indices, weights):
        output = torch.zeros_like(hidden)
        for expert in range(self.num_experts):
            tokens, positions = torch.where(indices == expert)
            x = hidden[tokens]
            gate, up = torch.nn.functional.linear(
                x, self.gate_up_proj[expert]).chunk(2, dim=-1)
            down = self.act_fn(gate) * up
            self.observed[expert]["gate_up"].append(x.detach().clone())
            self.observed[expert]["down"].append(down.detach().clone())
            output.index_add_(0, tokens, torch.nn.functional.linear(
                down, self.down_proj[expert]) * weights[tokens, positions, None])
        return output


class _Router(torch.nn.Module):
    def forward(self, hidden):
        return torch.stack((hidden[:, 0], -hidden[:, 0]), dim=-1)


class _RoutedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Router()
        self.experts = Lfm2MoeExperts()

    def route_tokens_to_experts(self, logits):
        values, indices = torch.softmax(logits, dim=-1).topk(1, dim=-1)
        return indices, values / values.sum(dim=-1, keepdim=True)

    def forward(self, hidden):
        indices, weights = self.route_tokens_to_experts(self.gate(hidden))
        return self.experts(hidden, indices, weights)


class _RoutedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="lfm2_moe", architectures=[])
        self.model = _mixed_model().model
        self.model.layers[2].feed_forward = _RoutedBlock()

    def forward(self, hidden):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        self.model.layers[0].attention(hidden)
        return self.model.layers[2].feed_forward(hidden)


EXPERT_PREFIX = "model.layers.2.feed_forward.experts"


def _routed_tokens():
    return [torch.tensor([[[1., 2., 0., 1.], [-1., 3., 1., 0.],
                           [2., 1., 1., 0.], [-2., 1., 0., 2.]]]),
            torch.tensor([[[3., 2., 0., 1.], [-3., 3., 1., 0.],
                           [4., 1., 1., 0.], [-4., 1., 0., 2.]]])]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capture_packed_projections_uses_every_actual_routed_row(dtype):
    from prismaquant.tessera_campaign import _collect_activations

    model = _RoutedModel().to(dtype=dtype)
    targets = [f"{EXPERT_PREFIX}.{expert}.{projection}"
               for expert in range(2) for projection in ("w1", "w3", "w2")]
    dense = "model.layers.0.attention"
    tokens = [batch.to(dtype=dtype) for batch in _routed_tokens()]
    rows, hessians, counts, max_abs = _collect_activations(
        model, [dense, *targets], tokens, 1, "cpu", want_hessian=True)
    from prismaquant.routed_experts import profile_declared_packed_expert_projections

    observed = model.model.layers[2].feed_forward.experts.observed
    units = {unit.qname: unit for unit in profile_declared_packed_expert_projections(model)}
    assert set(units) == set(targets)
    for expert in range(2):
        for projection, input_kind in (("w1", "gate_up"),
                                       ("w3", "gate_up"), ("w2", "down")):
            name = f"{EXPERT_PREFIX}.{expert}.{projection}"
            actual = torch.cat(observed[expert][input_kind]).float()
            assert actual.shape[0] == 4  # The score cap must be binding.
            assert counts[name] == actual.shape[0]
            torch.testing.assert_close(rows[name], actual[:1])
            torch.testing.assert_close(hessians[name], actual.T @ actual)
            # The static-scale calibration input is the SAME accumulator a
            # dense Linear feeds: max|x| over every routed row, never over
            # the capped scoring prefix.
            assert max_abs[name] == pytest.approx(float(actual.abs().amax()))
            unit = units[name]
            packed = getattr(unit.module, unit.param_name)
            expected = (packed[expert] if projection == "w2" else
                        packed[expert, :4] if projection == "w1" else packed[expert, 4:])
            torch.testing.assert_close(unit.weight, expected)
            assert unit.weight.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    flat = torch.cat(tokens).reshape(-1, 4).float()
    assert counts[dense] == 8
    torch.testing.assert_close(hessians[dense], flat.T @ flat)
    assert max_abs[dense] == pytest.approx(float(flat.abs().amax()))
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks
    assert not model.model.layers[0].attention._forward_pre_hooks


def test_packed_capture_subset_keeps_the_same_routed_rows_and_hessian():
    from prismaquant.tessera_campaign import _collect_activations

    all_targets = [f"{EXPERT_PREFIX}.{expert}.{projection}"
                   for expert in range(2) for projection in ("w1", "w3", "w2")]
    selected = f"{EXPERT_PREFIX}.1.w2"
    whole = _collect_activations(
        _RoutedModel(), all_targets, _routed_tokens(), 2, "cpu", want_hessian=True)
    subset = _collect_activations(
        _RoutedModel(), [selected], _routed_tokens(), 2, "cpu", want_hessian=True)
    for index in (0, 1):
        torch.testing.assert_close(subset[index][selected], whole[index][selected])
    assert subset[2][selected] == whole[2][selected] == 4
    assert subset[3][selected] == whole[3][selected] > 0.0


@pytest.mark.parametrize("want_hessian", [False, True])
def test_packed_capture_refuses_an_unobserved_expert(want_hessian):
    from prismaquant.tessera_campaign import _collect_activations

    model = _RoutedModel()
    missing = f"{EXPERT_PREFIX}.1.w1"
    with pytest.raises(RuntimeError, match="no routed calibration rows") as error:
        _collect_activations(model, [missing], [torch.ones(1, 3, 4)], 1,
                             "cpu", want_hessian=want_hessian)
    assert missing in str(error.value)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks


def _mixed_model():
    net = torch.nn.Module()
    net.model = torch.nn.Module()
    net.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    net.model.layers[0].attention = torch.nn.Linear(4, 4, bias=False)
    net.model.layers[2].feed_forward = torch.nn.Module()
    net.model.layers[2].feed_forward.experts = Lfm2MoeExperts()
    return net


def _refusing_main_fixture(monkeypatch, model):
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: Lfm2MoeProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status",
                        lambda: {"accepted": True})

    def calibration_would_hide_the_omission(*_args, **_kwargs):
        raise AssertionError("started calibration with a population the bridge cannot carry")

    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        calibration_would_hide_the_omission)
    return tessera_campaign


def _main_argv(tmp_path, *, stride=1):
    return ["--model", "synthetic-lfm", "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--hessian", "off",
            "--menu-mode", "research", "--layer-stride", str(stride)]


@pytest.mark.parametrize("stride", [1, 2])
def test_main_refuses_a_packed_population_without_the_producer_tool(monkeypatch, tmp_path, stride):
    """The bridge shells out to the producer; no producer, no packed price.

    The refusal names the packed parameters it cannot price and arrives
    before calibration, so a campaign started without ``TESSERA_REPO`` does
    not spend an hour on a dense-only table (PrismaQuant #183).
    """
    monkeypatch.delenv("TESSERA_REPO", raising=False)
    campaign = _refusing_main_fixture(monkeypatch, _mixed_model())
    with pytest.raises(RuntimeError, match="cannot ask the producer for its expert projection") as error:
        campaign.main(_main_argv(tmp_path, stride=stride))
    for parameter in ("gate_up_proj", "down_proj"):
        assert f"model.layers.2.feed_forward.experts.{parameter}" in str(error.value)
    assert "TESSERA_REPO" in str(error.value)
    assert not (tmp_path / "cost.pkl").exists()


def test_main_refuses_a_packed_parameter_the_profile_does_not_split(monkeypatch, tmp_path):
    """A packed parameter without a declared per-expert split is refused by name.

    The gate is the SAME derivation the capture uses: if the profile's split
    does not cover a routed 3-D parameter, nothing downstream could have priced
    it, and a dense-only payload is the failure this refusal exists to stop.
    """
    from prismaquant import routed_experts

    monkeypatch.setenv("TESSERA_REPO", "/nonexistent/tessera")
    campaign = _refusing_main_fixture(monkeypatch, _mixed_model())
    declared = routed_experts.profile_declared_packed_expert_projections

    def without_down(model, profile=None):
        return [member for member in declared(model, profile)
                if member.param_name != "down_proj"]

    monkeypatch.setattr(routed_experts, "profile_declared_packed_expert_projections", without_down)
    with pytest.raises(RuntimeError, match="cannot price the packed expert population") as error:
        campaign.main(_main_argv(tmp_path))
    assert "model.layers.2.feed_forward.experts.down_proj (2, 4, 4)" in str(error.value)
    assert "gate_up_proj" not in str(error.value)
    assert not (tmp_path / "cost.pkl").exists()


# ---------------------------------------------------------------------------
# The bridge: the producer's projection priced through main()
# ---------------------------------------------------------------------------
HIDDEN, INTER, EXPERTS = 64, 64, 2
STACK = "model.layers.2.feed_forward.experts"
RUNG = "TESSERA_E4M3_K1_R1024"


class _WideExperts(torch.nn.Module):
    """A packed LFM2-shaped expert stack the producer's tile rules accept.

    ``Lfm2MoeExperts`` above is 8x4 per expert -- too small for the producer's
    ``rows % (arity*32) == 0 and cols % 16 == 0`` cut -- so the bridge tests
    use hidden 64 / intermediate 64.
    """

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(183)
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=generator))
        self.down_proj = torch.nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, INTER, generator=generator))
        self.num_experts = EXPERTS
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden, indices, weights):
        output = torch.zeros_like(hidden)
        for expert in range(self.num_experts):
            tokens, positions = torch.where(indices == expert)
            x = hidden[tokens]
            gate, up = torch.nn.functional.linear(x, self.gate_up_proj[expert]).chunk(2, dim=-1)
            down = self.act_fn(gate) * up
            output.index_add_(0, tokens, torch.nn.functional.linear(
                down, self.down_proj[expert]) * weights[tokens, positions, None])
        return output


class _WideRoutedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="lfm2_moe", architectures=[])
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
        self.model.layers[0].attention = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.model.layers[2].feed_forward = _RoutedBlock()
        self.model.layers[2].feed_forward.experts = _WideExperts()

    def forward(self, hidden):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        self.model.layers[0].attention(hidden)
        return self.model.layers[2].feed_forward(hidden)


def _wide_tokens():
    generator = torch.Generator().manual_seed(1830)
    batches = []
    for _ in range(2):
        batch = torch.randn(1, 8, HIDDEN, generator=generator)
        batch[0, ::2, 0] = 2.0   # routes half the rows to expert 0 ...
        batch[0, 1::2, 0] = -2.0  # ... and half to expert 1
        batches.append(batch)
    return batches


def _expert_unit_names():
    return [f"{STACK}.{expert}.{projection}"
            for expert in range(EXPERTS) for projection in ("w1", "w3", "w2")]


def _write_source_checkpoint(model, root, *, perturb=None):
    """The unpacked per-expert LFM2.5 source the live packed views came from."""
    from safetensors.torch import save_file

    experts = model.model.layers[2].feed_forward.experts
    tensors = {}
    for expert in range(EXPERTS):
        gate_up = experts.gate_up_proj[expert].detach()
        tensors[f"{STACK}.{expert}.w1.weight"] = gate_up[:INTER].contiguous()
        tensors[f"{STACK}.{expert}.w3.weight"] = gate_up[INTER:].contiguous()
        tensors[f"{STACK}.{expert}.w2.weight"] = experts.down_proj[expert].detach().contiguous()
    tensors["model.layers.0.attention.weight"] = model.model.layers[0].attention.weight.detach().contiguous()
    if perturb is not None:
        tensors[perturb] = tensors[perturb].clone()
        tensors[perturb][0, 0] = tensors[perturb][0, 0] + 1.0
    root.mkdir(parents=True)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": HIDDEN, "intermediate_size": HIDDEN,
        "moe_intermediate_size": INTER, "num_experts": EXPERTS,
        "num_hidden_layers": 3,
    }, indent=1))


def _bridge_main_fixture(monkeypatch, tmp_path, *, perturb=None):
    """main() with a real capture, projection, encode and receipt; no route scoring.

    ``_measure_anchor`` is replaced by the producer's real encode without its
    served-route admission: this CPU fixture exercises the bridge, while
    runtime-scoped GPU pricing requires its own measurement. Everything the
    bridge adds -- the population
    gate, the producer request, the binding, the source-byte check, the real
    Tessera bytes under ``unit_input_identity`` receipts and the payload's
    population/projection/wire blocks -- runs for real.
    """
    pytest.importorskip("tessera.cached_unit")
    pytest.importorskip("safetensors")
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
    from prismaquant.tessera_expert_projection import ExpertProjectionError, producer_plan_tool

    try:
        producer_plan_tool()
    except ExpertProjectionError as exc:
        pytest.skip(f"producer projection tool unavailable: {exc}")

    model = _WideRoutedModel().to(dtype=torch.bfloat16)
    source = tmp_path / "source"
    _write_source_checkpoint(model, source, perturb=perturb)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: Lfm2MoeProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {
        "accepted": True, "reason": "CPU test fixture", "kwargs": [], "recipe": {},
    })
    tokens = [batch.to(dtype=torch.bfloat16) for batch in _wide_tokens()]
    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        lambda *_args: (tokens, "synthetic routed draw"))
    menu = [SimpleNamespace(format_name=RUNG, family="TESSERA_E4M3_K1",
                            body_rate_q256=1024, bpp=4.0)]
    monkeypatch.setattr(tessera_campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: {name: list(menu) for name in targets})
    encoded = []

    def measure_without_route_admission(*, qname, weight, format_name, wire_dir, **_kwargs):
        # The producer's real encode and real bytes (so the cached-unit receipt
        # is a receipt for a Tessera artifact); only the served-route scoring
        # of ``_measure_anchor`` is left out, for the reason in the docstring.
        render, blob = tessera_campaign._encode_and_render(
            weight, format_name, hessian_required=False)
        path = tessera_campaign._wire_path(wire_dir, qname, format_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        encoded.append((qname, format_name, tuple(weight.shape)))
        return tessera_campaign.CampaignAnchor(
            qname=qname, family="TESSERA_E4M3_K1", format_name=format_name,
            body_rate_q256=1024,
            dloss=float(((render.float() - weight.float()) ** 2).mean()), dloss_stderr=0.0,
            memory_bytes=len(blob), bits_per_param=8 * len(blob) / weight.numel(),
            activation_contract="w8a8-dynamic-e4m3-channel", activation_quantized=True,
            wire_bytes=len(blob), seconds=0.01, hessian_applied=False)

    monkeypatch.setattr(tessera_campaign, "_measure_anchor", measure_without_route_admission)
    argv = ["--model", str(source), "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--hessian", "off",
            "--menu-mode", "research", "--max-rounds", "1"]
    return tessera_campaign, argv, model, encoded


def test_main_prices_the_producer_projected_expert_population(monkeypatch, tmp_path):
    from prismaquant.tessera_expert_projection import (
        CARRIED_PROJECTION_SCHEMA, EXPERT_WIRES_KEY, POPULATION_KEY, POPULATION_SCHEMA,
        PROJECTION_KEY, carried_units,
    )

    campaign, argv, model, encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    assert campaign.main(argv) == 0
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    units = _expert_unit_names()
    dense = "model.layers.0.attention"

    # Every projected unit was priced as the 2-D view the profile declares.
    assert sorted(name for name, _fmt, _shape in encoded) == sorted([dense, *units])
    assert {shape for name, _fmt, shape in encoded if name in units} == {(INTER, HIDDEN), (HIDDEN, INTER)}
    assert set(payload["costs"]) == {dense, *units}

    # The payload says which population it priced and which it omitted.
    population = payload["provenance"][POPULATION_KEY]
    assert population["schema"] == POPULATION_SCHEMA
    assert population["priced"]["routed_experts"] == sorted(units)
    assert population["priced"]["dense"] == [dense]
    assert population["priced"]["stacks"] == [STACK]
    assert population["priced"]["packed_parameters"] == {
        f"{STACK}.gate_up_proj": [EXPERTS, 2 * INTER, HIDDEN],
        f"{STACK}.down_proj": [EXPERTS, HIDDEN, INTER]}
    assert population["omitted"] == {"dense_outside_layer_stride": [],
                                     "packed_outside_layer_stride": {}, "pinned": []}
    assert population["counts"]["routed_experts_priced"] == len(units)

    # The producer's projection rides in the payload verbatim, bound to the
    # profile-declared names; PrismaQuant chose no slice of its own.
    carried = payload["provenance"][PROJECTION_KEY]
    assert carried["schema"] == CARRIED_PROJECTION_SCHEMA
    assert carried["request"] == {STACK: {"grid": "E4M3", "q256": 1024,
                                          "source_layout": "unpacked_per_expert"}}
    source, flat, stack_of = carried_units(carried)
    assert sorted(flat) == sorted(units)
    assert set(stack_of.values()) == {STACK}
    for name, unit in flat.items():
        assert unit["source_tensor"] == f"{name}.weight"
        assert unit["source_slice"] == {"expert": int(name.split(".")[-2]),
                                        "selector": "whole", "transpose": False}
    assert set(source["tensors"]) >= {f"{name}.weight" for name in units}
    request_path = tmp_path / "cache" / "expert_projection.json.request.json"
    assert json.loads(request_path.read_text()) == carried["request"]

    # Every priced expert wire carries the producer's cached-unit receipt,
    # sealed with the projection record (unit_input_identity), so the exporter
    # can verify the bytes without a second encode.  Dense units keep their
    # encoding_input_identity receipts in the checkpoint, not here.
    wires = payload[EXPERT_WIRES_KEY]
    assert sorted(wires) == sorted(units)
    for name in units:
        record = wires[name][RUNG]
        assert set(record) == {"file", "blob_sha256", "blob_bytes", "identity"}
        identity = record["identity"]
        assert identity["unit"] == name
        assert identity["projection"] == flat[name]
        assert identity["recipe"]["grid"] == "E4M3"
        assert identity["recipe"]["q256"] == 1024
        wire = tmp_path / "cache" / "wire" / record["file"]
        assert wire.is_file() and wire.stat().st_size == record["blob_bytes"]
    assert dense not in wires
    # The checkpoint identity binds the projection the rows were priced under.
    assert campaign._campaign_checkpoint_identity.__doc__  # (bound in the journal; see resume tests)


def test_main_refuses_a_live_view_that_disagrees_with_the_producer_source(monkeypatch, tmp_path):
    """The exporter re-reads the source tensor; the campaign must have priced those bytes."""
    campaign, argv, _model, encoded = _bridge_main_fixture(
        monkeypatch, tmp_path, perturb=f"{STACK}.1.w2.weight")
    with pytest.raises(RuntimeError, match="disagrees byte-for-byte") as error:
        campaign.main(argv)
    assert f"{STACK}.1.w2" in str(error.value)
    assert f"{STACK}.0.w1" not in str(error.value)
    assert not encoded, "priced a rung on bytes the exporter would not read"
    assert not (tmp_path / "cost.pkl").exists()


@pytest.mark.parametrize("failure", ["empty_menu", "failed_anchor", "failed_expert"])
def test_main_reports_unpriced_targets_without_claiming_coverage(monkeypatch, tmp_path, failure):
    campaign, argv, _model, _encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    dense = "model.layers.0.attention"
    target = _expert_unit_names()[0] if failure == "failed_expert" else dense
    if failure == "empty_menu":
        expand = campaign.expand_menus_for_targets

        def menus(*args, **kwargs):
            result = expand(*args, **kwargs)
            result[dense] = []
            return result

        monkeypatch.setattr(campaign, "expand_menus_for_targets", menus)
    else:
        measure = campaign._measure_anchor

        def failing_measure(**kwargs):
            if kwargs["qname"] == target:
                raise RuntimeError("synthetic encode failure")
            return measure(**kwargs)

        monkeypatch.setattr(campaign, "_measure_anchor", failing_measure)
    assert campaign.main(argv) == 0
    payload = pickle.loads((tmp_path / "cost.pkl").read_bytes())
    population = payload["provenance"]["population"]
    assert target not in payload["costs"]
    if failure == "failed_expert":
        assert target not in population["priced"]["routed_experts"]
        assert population["enumerated"]["routed_experts"] == sorted(_expert_unit_names())
        assert population["unpriced"]["routed_experts"] == {target: "no_successful_anchor"}
        assert population["priced"]["stacks"] == []
        assert population["priced"]["packed_parameters"] == {}
        assert population["counts"]["routed_experts_priced"] == len(_expert_unit_names()) - 1
        return
    assert population["priced"]["dense"] == []
    assert population["enumerated"]["dense"] == [dense]
    assert population["unpriced"]["dense"] == {
        dense: "no_admitted_menu" if failure == "empty_menu" else "no_successful_anchor"}
    assert population["counts"]["dense_priced"] == 0
    assert population["counts"]["dense_unpriced"] == 1
    assert population["priced"]["routed_experts"] == sorted(_expert_unit_names())


def test_wire_backed_units_keep_only_measured_rows():
    """A priced expert wire IS the exported wire, so no rung without bytes is offered.

    A dense unit's interpolated rows are re-encoded at export; a projected
    expert unit's row is exported from its priced blob (``--cached-expert-units``),
    so an interpolated rung would be a price with no bytes behind it.
    """
    from prismaquant.tessera_campaign import CampaignAnchor, campaign_cost_payload

    fam = "TESSERA_E2M1_K2"

    def anchor(qname, rung, dloss):
        return CampaignAnchor(
            qname=qname, family=fam, format_name=f"{fam}_R{rung}", body_rate_q256=rung,
            dloss=dloss, dloss_stderr=0.0, memory_bytes=1024, bits_per_param=rung / 256.0,
            activation_contract="w4a4-nvfp4-e2m1-group16-ue4m3", activation_quantized=True,
            wire_bytes=1024, seconds=1.0)

    # Menu rows as the payload reads them (the real research menu needs the
    # producer's lane table, which is #192's re-pin, not this test's subject).
    rows = [SimpleNamespace(
        format_name=f"{fam}_R{rung}", family=fam, body_rate_q256=rung, bpp=rung / 256.0,
        admission=SimpleNamespace(activation_contract="w4a4-nvfp4-e2m1-group16-ue4m3"))
        for rung in (128, 384, 512, 896)]
    dense, expert = "m.0.q_proj", f"{STACK}.0.w1"
    anchors = {name: {fam: [anchor(name, r, d) for r, d in ((128, 1e-2), (512, 1e-3), (896, 1e-4))]}
               for name in (dense, expert)}
    payload = campaign_cost_payload(anchors, {dense: rows, expert: rows}, loo={}, provenance={},
                                    wire_backed={expert})
    assert set(payload["costs"][dense]) == {f"{fam}_R{r}" for r in (128, 384, 512, 896)}
    assert set(payload["costs"][expert]) == {f"{fam}_R{r}" for r in (128, 512, 896)}
    assert all(row["output_mse_measured"] for row in payload["costs"][expert].values())


def test_projection_walks_the_menu_to_a_family_with_an_expert_route(monkeypatch, tmp_path):
    """The cheapest rung's family need not have an expert route (#280).

    The menu is ordered by rate, so its first rung is NVFP4 here.  The pinned
    producer refuses an NVFP4 expert stack -- ``scheme.MOE_BUILDERS`` names
    only ``TESSERA_FP8`` on this build -- and the campaign must ask the next
    family rather than refuse the whole population.  The refusal is the
    producer's real one: nothing about the route is mocked.
    """
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    campaign, _argv, model, _encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    population = campaign._require_campaign_population(model, Lfm2MoeProfile(), 1)
    assert population.declared, "fixture must declare a packed expert stack"
    weights = {member.qname: member.weight.detach() for member in population.members}
    ladder = [SimpleNamespace(format_name="TESSERA_E2M1_K2_R128", family="TESSERA_E2M1_K2",
                              body_rate_q256=128, bpp=0.5),
              SimpleNamespace(format_name=RUNG, family="TESSERA_E4M3_K1",
                              body_rate_q256=1024, bpp=4.0)]
    menus = {name: list(ladder) for name in weights}

    carried, projected = campaign._project_expert_population(
        population, weights=weights, menus=menus,
        model_path=str(tmp_path / "source"), cache_dir=tmp_path / "cache")

    assert projected, "the routable family must yield the projected units"
    attempts = carried["plan_attempts"]
    assert len(attempts) == 2, attempts
    first = attempts[0]
    assert first["refused"], "the producer must have refused the NVFP4 stack"
    assert "no expert route" in first["refused"]
    assert {entry["grid"] for entry in first["request"].values()} == {"E2M1x2"}
    assert attempts[-1]["refused"] is None
    grids = {entry["grid"] for entry in carried["request"].values()}
    assert grids == {entry["grid"] for entry in attempts[-1]["request"].values()}
    assert "E2M1" not in "".join(grids)


def test_projection_matches_family_names_across_different_stack_menus(monkeypatch, tmp_path):
    """Different menu positions must not hide a common routable family."""
    from safetensors.torch import load_file, save_file
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    campaign, _argv, model, _encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    second_stack = STACK.replace("layers.2", "layers.3")
    layer = torch.nn.Module()
    layer.feed_forward = _RoutedBlock()
    layer.feed_forward.experts = _WideExperts().to(dtype=torch.bfloat16)
    model.model.layers.append(layer)
    source = tmp_path / "source"
    tensors = load_file(str(source / "model.safetensors"))
    tensors.update({name.replace(STACK, second_stack): value.clone()
                    for name, value in list(tensors.items()) if name.startswith(STACK)})
    save_file(tensors, str(source / "model.safetensors"))
    config = json.loads((source / "config.json").read_text())
    config["num_hidden_layers"] = 4
    (source / "config.json").write_text(json.dumps(config))

    population = campaign._require_campaign_population(model, Lfm2MoeProfile(), 1)
    assert set(population.declared) == {STACK, second_stack}
    weights = {member.qname: member.weight.detach() for member in population.members}
    first = ["TESSERA_E2M1_K2_R128", RUNG]
    second = [RUNG, "TESSERA_BF16_K1_R1792"]
    menus = {name: [SimpleNamespace(format_name=fmt) for fmt in
                   (first if name.startswith(STACK + ".") else second)]
             for name in weights}

    carried, projected = campaign._project_expert_population(
        population, weights=weights, menus=menus,
        model_path=str(source), cache_dir=tmp_path / "cache")

    assert set(projected) == set(weights)
    assert {row["grid"] for row in carried["request"].values()} == {"E4M3"}
    assert carried["plan_attempts"][-1]["refused"] is None
