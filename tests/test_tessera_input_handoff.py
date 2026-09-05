"""Campaign publication -> real preflight -> producer intake, CPU only."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import save_file
from prismaquant import tessera_export_lane as lane
from prismaquant.tessera_campaign import write_export_inputs

UNIT = "model.layers.0.self_attn.o_proj"
TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64, "fit_tokens": 4096}


@pytest.fixture(scope="module", autouse=True)
def _one_native_thread():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


@pytest.fixture
def producer():
    import tessera

    path = Path(tessera.__file__).resolve().parents[2] / "experiments/export_tessera_serving.py"
    if not path.is_file():
        pytest.skip("requires the Tessera repository producer script")
    spec = importlib.util.spec_from_file_location("handoff_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, h=1.0, scale=1.0):
    return write_export_inputs(
        root, hessians={UNIT: torch.eye(32) * h} if h is not None else None,
        hessian_rows={UNIT: 4}, hessian_identity=TRIPLE,
        static_scales={UNIT: scale} if scale is not None else {},
        static_scale_policy="legacy_6_over_calibration_amax.v1")


def _case(tmp_path, monkeypatch, capsys, *, h=1.0, scale=1.0, bf16=False):
    capture, scales, digest = _write(tmp_path, h, scale)
    assignment = tmp_path / "allocation.json"
    assignment.write_text(json.dumps({
        UNIT: "BF16" if bf16 else {"data_type": "tessera", "bits": 4,
            "tessera_format": "TESSERA_E2M1_K2_R896" if scale is not None else "TESSERA_E4M3_K1_R1024"},
        "__prismaquant__": {
            "tessera_hessian": {"supplied": h is not None, **TRIPLE, "capture_sha256": digest},
            "tessera_activation_static_scales": {"schema": lane.PRICED_STATIC_SCALES_SCHEMA,
                                                "units": {UNIT: scale} if scale is not None else {}}}}))
    # Scope/ship gates have their own tests; keep the real input validation,
    # preflight build assembly, CLI publication and producer intake here.
    monkeypatch.setattr(lane, "require_declared_structure", lambda _: "dense")
    monkeypatch.setattr(lane, "require_serving_target", lambda _: None)
    monkeypatch.setattr(lane, "require_executes_derived_from_contract", lambda: ())
    monkeypatch.setattr(lane, "require_producer_tools", lambda: ())
    monkeypatch.setattr(lane, "require_producer_repo_is_pinned", lambda: ())
    monkeypatch.setattr(lane, "require_release_pin", lambda: None)
    monkeypatch.setattr(lane, "require_assignment_scope", lambda *a, **k: None)
    from prismaquant import tessera_serving_runtime_pin as pin
    monkeypatch.setattr(pin, "load_tessera_serving_runtime_pin",
                        lambda: SimpleNamespace(version="fixture", commit="f" * 40))
    paths = []
    if capture:
        paths += ["--hessian", str(capture)]
    if scales:
        paths += ["--input-scales", str(scales)]
    build = tmp_path / "build.json"
    capsys.readouterr()
    assert lane.main(["--model", str(tmp_path), "--assignment", str(assignment),
                      "--write-build-json", str(build), "--print-build-sha256", *paths]) == 0
    build_sha = capsys.readouterr().out.strip()
    assert build_sha == hashlib.sha256(build.read_bytes()).hexdigest()
    assert json.loads(build.read_bytes())["priced_inputs"] == {
        "schema": "tessera.priced_export_inputs.v1", "hessian_capture_sha256": digest,
        "input_global_scales": {UNIT + ".input_global_scale": scale} if scale is not None else {}}
    src = tmp_path / "src"
    src.mkdir()
    weight = torch.randn(64, 32, generator=torch.Generator().manual_seed(0)).bfloat16()
    save_file({UNIT + ".weight": weight}, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 32, "intermediate_size": 32}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({UNIT + ".weight": "PASSTHROUGH" if bf16 else {
        "grid": "E2M1x2" if scale is not None else "E4M3", "q256": 896 if scale is not None else 1024}}))
    out = tmp_path / "out"
    argv = ["export_tessera_serving.py", str(src), str(out), "--plan-json", str(plan),
            "--device", "cpu", "--no-verify", "--priced-inputs", str(build),
            "--priced-inputs-sha256", build_sha, *paths]
    monkeypatch.setattr("sys.argv", argv)
    return out, build, digest


@pytest.mark.parametrize("replacement", ["hessian", "scale", "both", "build"])
def test_campaign_republication_after_preflight_refuses_before_output(
        tmp_path, monkeypatch, capsys, producer, replacement):
    out, build, _ = _case(tmp_path, monkeypatch, capsys)
    _write(tmp_path, h=2.0 if replacement in {"hessian", "both"} else 1.0,
           scale=4.0 if replacement in {"scale", "both"} else 1.0)
    if replacement == "build":
        payload = json.loads(build.read_bytes())
        payload["priced_inputs"]["input_global_scales"][UNIT + ".input_global_scale"] = 4.0
        build.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="(--priced-inputs.*allocation|build SHA-256)"):
        producer.main()
    assert not out.exists()


@pytest.mark.parametrize("mode", ["priced", "no_h", "no_scale", "bf16"])
def test_unchanged_campaign_and_input_free_controls_export(
        tmp_path, monkeypatch, capsys, producer, mode):
    out, _, _ = _case(tmp_path, monkeypatch, capsys,
                     h=None if mode in {"no_h", "bf16"} else 1.0,
                     scale=None if mode in {"no_scale", "bf16"} else 1.0,
                     bf16=mode == "bf16")
    if mode == "bf16":
        # Preflight accepts this input-free allocation. The standalone
        # producer still refuses an empty Tessera config unless a deliberate
        # passthrough copy was requested, as it did before the handoff gate.
        with pytest.raises(SystemExit, match="nothing was planned"):
            producer.main()
        assert not out.exists()
        return
    producer.main()
    assert (out / "config.json").is_file()
    if mode in {"priced", "no_h"}:
        with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
            assert handle.get_tensor(UNIT + ".trellis_input_global_scale").item() == 1.0


def test_republication_after_intake_keeps_the_consumed_snapshot(
        tmp_path, monkeypatch, capsys, producer):
    out, build, digest = _case(tmp_path, monkeypatch, capsys)
    require = producer.PricedInputsSnapshot.require

    def republish_after_intake(self, activation, scales):
        require(self, activation, scales)
        _write(tmp_path, h=2.0, scale=4.0)
        build.write_text("{}")
        assert activation.capture_sha256() == digest
        assert scales[UNIT + ".input_global_scale"] == 1.0

    monkeypatch.setattr(producer.PricedInputsSnapshot, "require", republish_after_intake)
    producer.main()
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["activation_aware"]["hessian"]["capture_sha256"] == digest
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        assert handle.get_tensor(UNIT + ".trellis_input_global_scale").item() == 1.0


def test_preflight_returns_the_digest_of_its_owned_build_after_path_replacement(
        tmp_path, monkeypatch, capsys, producer):
    build = tmp_path / "build.json"
    report = {"structure": "dense", "executes": [], "pinned_version": "fixture",
              "pinned_commit": "f" * 40, "shipcard_slots": [],
              "unrecorded_gates": [], "unsupported_producer_tools": [],
              "build": {"priced_inputs": {"schema": "tessera.priced_export_inputs.v1",
                        "hessian_capture_sha256": None, "input_global_scales": {}}}}
    monkeypatch.setattr(lane, "preflight", lambda *args, **kwargs: report)
    replace = Path.replace
    written = []

    def republish(self, target):
        written.append(self.read_bytes())
        result = replace(self, target)
        Path(target).write_text("{}")
        return result

    monkeypatch.setattr(Path, "replace", republish)
    assert lane.main(["--model", str(tmp_path), "--assignment", "unused.json",
                      "--write-build-json", str(build), "--print-build-sha256"]) == 0
    digest = capsys.readouterr().out.strip()
    assert digest == hashlib.sha256(written[0]).hexdigest()
    with pytest.raises(SystemExit, match="build SHA-256"):
        producer.PricedInputsSnapshot(build, digest)
