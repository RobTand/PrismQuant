from __future__ import annotations

import hashlib
import json
import math
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import prismaquant.trellis_bf16_corpus as corpus


_SMALL_SHAPES = {
    ("dense", "gate_proj"): (3, 2),
    ("dense", "up_proj"): (3, 2),
    ("dense", "down_proj"): (2, 3),
    ("routed", "gate_proj"): (2, 2),
    ("routed", "up_proj"): (2, 2),
    ("routed", "down_proj"): (2, 2),
}


def _small_shape(population: str, projection: str) -> tuple[int, int]:
    return _SMALL_SHAPES[(population, projection)]


def _tensor_bytes(value: torch.Tensor, dtype: str) -> bytes:
    if dtype == "F32":
        return value.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    assert dtype == "BF16"
    return value.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def _write_safetensors(path: Path, rows: dict[str, tuple[str, torch.Tensor]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, value) in rows.items():
        raw = _tensor_bytes(value, dtype)
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _calibration() -> dict[str, object]:
    payload: dict[str, object] = {
        "probe_calib_hash": "a" * 32,
        "dataset": "test",
        "nsamples": 2,
        "seqlen": 8,
        "seed": 7,
        "tokens": 16,
    }
    payload["identity_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    return payload


def _glm_names() -> list[str]:
    names: list[str] = []
    for layer in corpus.GLM_DENSE_LAYERS:
        for projection in corpus.GLM_PROJECTIONS:
            names.append(
                f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            )
    for layer in corpus.GLM_ROUTED_LAYERS:
        for projection in corpus.GLM_PROJECTIONS:
            names.append(
                f"model.language_model.layers.{layer}.mlp.experts.0."
                f"{projection}.weight"
            )
    return names


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "glm_inputs.safetensors"
    manifest = tmp_path / "glm_inputs_manifest.json"
    rows: dict[str, tuple[str, torch.Tensor]] = {}
    expected_raw: dict[str, bytes] = {}
    tensors: dict[str, object] = {}
    for index, name in enumerate(_glm_names(), 1):
        population, _layer, projection, _expert = corpus._classify_glm_name(name)
        shape = _small_shape(population, projection)
        value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        value = (value + index).to(torch.bfloat16)
        rows[name] = ("BF16", value)
        expected_raw[name] = _tensor_bytes(value, "BF16")
        tensors[name] = {
            "shape": list(shape),
            "role": population,
            "shard": "source",
            "distinct_source_values": int(value.unique().numel()),
        }
    _write_safetensors(source, rows)
    manifest.write_text(json.dumps({
        "schema": corpus.INCOMPLETE_GLM_SCHEMA,
        "model": "/model/glm",
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tensors": tensors,
    }))
    return manifest, source, expected_raw


def _probe_fixture(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, object]] = {}
    expected: dict[str, torch.Tensor] = {}
    for name in _glm_names():
        population, _layer, projection, expert = corpus._classify_glm_name(name)
        width = _small_shape(population, projection)[1]
        if population == "dense":
            qname = name.removesuffix(".weight")
            value = torch.arange(1, width + 1, dtype=torch.float32)
            stats[qname] = {
                "act_sq_sum": value * 4,
                "n_tokens_seen": 4,
                "in_features": width,
            }
            expected[name] = value
        else:
            prefix = name.split(".experts.0.", 1)[0]
            packed = "down_proj" if projection == "down_proj" else "gate_up_proj"
            qname = f"{prefix}.experts.{packed}"
            if qname not in stats:
                values = torch.stack((
                    torch.arange(1, width + 1, dtype=torch.float32) * 3,
                    torch.arange(1, width + 1, dtype=torch.float32) * 10,
                ))
                stats[qname] = {
                    "expert_act_sq_sum": values,
                    "expert_tokens": torch.tensor([3, 5]),
                }
            expected[name] = torch.arange(1, width + 1, dtype=torch.float32)
            assert expert == 0
    probe = tmp_path / "probe.pkl"
    with probe.open("wb") as handle:
        pickle.dump({
            "stats": stats,
            "meta": {"calibration_hash": "a" * 32},
        }, handle)
    return probe, expected


def _finalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, source, expected_raw = _source_fixture(tmp_path)
    probe, expected_importance = _probe_fixture(tmp_path)
    importance, identity = corpus.adapt_glm_importance_from_probe(incomplete, probe)
    output = tmp_path / "glm_inputs_final.safetensors"
    manifest = tmp_path / "glm_inputs_final_manifest.json"
    source_before = source.read_bytes()
    corpus.finalize_glm_bf16_corpus(
        incomplete_manifest_path=incomplete,
        source_artifact_path=source,
        importance=importance,
        importance_identity=identity,
        output_artifact_path=output,
        output_manifest_path=manifest,
        calibration=_calibration(),
        model_config_sha256="b" * 64,
        prismaquant_commit="c" * 40,
        generated="2026-08-30T00:00:00+00:00",
        host="test-host",
    )
    assert source.read_bytes() == source_before
    return (
        manifest, output, expected_raw, expected_importance,
        incomplete, source, importance, identity,
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def _rewrite_artifact_header(path: Path, mutate) -> None:
    path.chmod(0o644)
    raw = path.read_bytes()
    old_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + old_len])
    mutate(header)
    new_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    new_header += b" " * ((-len(new_header)) % 8)
    path.write_bytes(len(new_header).to_bytes(8, "little") + new_header + raw[8 + old_len:])


def test_adapter_maps_packed_gate_up_and_down_without_population_pooling(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, _source, _ = _source_fixture(tmp_path)
    probe, expected = _probe_fixture(tmp_path)
    values, identity = corpus.adapt_glm_importance_from_probe(incomplete, probe)

    assert len(values) == 33
    assert identity["schema"] == corpus.IMPORTANCE_SCHEMA
    for name, item in values.items():
        population, _layer, projection, expert = corpus._classify_glm_name(name)
        assert torch.equal(item.value, expected[name])
        if population == "routed":
            assert item.source_expert == expert == 0
            assert item.denominator_name == "expert_tokens"
            assert item.denominator == 3
            if projection in ("gate_proj", "up_proj"):
                assert item.source_qname.endswith("experts.gate_up_proj")
            else:
                assert item.source_qname.endswith("experts.down_proj")
        else:
            assert item.source_expert is None
            assert item.denominator_name == "n_tokens_seen"
            assert item.denominator == 4


def test_adapter_ignores_unrouted_nonselected_experts_but_requires_expert_zero(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, _source, _ = _source_fixture(tmp_path)
    probe, expected = _probe_fixture(tmp_path)
    with probe.open("rb") as handle:
        payload = pickle.load(handle)
    for raw in payload["stats"].values():
        if "expert_tokens" in raw:
            raw["expert_tokens"][1] = 0
            raw["expert_act_sq_sum"][1].zero_()
    with probe.open("wb") as handle:
        pickle.dump(payload, handle)
    values, _identity = corpus.adapt_glm_importance_from_probe(incomplete, probe)
    for name, item in values.items():
        assert torch.equal(item.value, expected[name])

    with probe.open("rb") as handle:
        payload = pickle.load(handle)
    first = next(raw for raw in payload["stats"].values()
                 if "expert_tokens" in raw)
    first["expert_tokens"][0] = 0
    first["expert_act_sq_sum"][0].zero_()
    with probe.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(corpus.CorpusContractError, match="positive integer"):
        corpus.adapt_glm_importance_from_probe(incomplete, probe)


def test_finalizer_preserves_weights_and_strict_loader_separates_populations(
    tmp_path, monkeypatch,
):
    manifest, _output, expected_raw, expected_importance, *_ = _finalized(
        tmp_path, monkeypatch
    )
    loaded = corpus.load_finalized_bf16_corpus(manifest)

    assert len(loaded.populations["dense"]) == 9
    assert len(loaded.populations["routed"]) == 24
    for entry in loaded.entries:
        weight, importance = loaded.load_tensor(entry)
        assert _tensor_bytes(weight, "BF16") == expected_raw[entry.name]
        assert torch.equal(importance, expected_importance[entry.name])


def test_finalizer_is_no_clobber_and_never_mutates_incomplete_artifact(
    tmp_path, monkeypatch,
):
    (manifest, output, _raw, _expected, incomplete, source,
     importance, identity) = _finalized(tmp_path, monkeypatch)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(corpus.CorpusContractError, match="must not exist"):
        corpus.finalize_glm_bf16_corpus(
            incomplete_manifest_path=incomplete,
            source_artifact_path=source,
            importance=importance,
            importance_identity=identity,
            output_artifact_path=output,
            output_manifest_path=manifest,
            calibration=_calibration(),
            model_config_sha256="b" * 64,
            prismaquant_commit="c" * 40,
            generated="2026-08-30T00:00:01+00:00",
            host="host",
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_concurrent_finalizers_have_exactly_one_live_owner(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, source, _ = _source_fixture(tmp_path)
    probe, _ = _probe_fixture(tmp_path)
    importance, identity = corpus.adapt_glm_importance_from_probe(incomplete, probe)
    output = tmp_path / "final.safetensors"
    manifest = tmp_path / "manifest.json"
    kwargs = dict(
        incomplete_manifest_path=incomplete,
        source_artifact_path=source,
        importance=importance,
        importance_identity=identity,
        output_artifact_path=output,
        output_manifest_path=manifest,
        calibration=_calibration(),
        model_config_sha256="b" * 64,
        prismaquant_commit="c" * 40,
        generated="2026-08-30T00:00:01+00:00",
        host="host-a",
    )
    entered_copy = threading.Event()
    release_copy = threading.Event()
    original_copy = corpus._copy_payload_pread

    def blocking_copy(*args, **inner_kwargs):
        entered_copy.set()
        assert release_copy.wait(timeout=10)
        return original_copy(*args, **inner_kwargs)

    monkeypatch.setattr(corpus, "_copy_payload_pread", blocking_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(corpus.finalize_glm_bf16_corpus, **kwargs)
        assert entered_copy.wait(timeout=10)
        loser = pool.submit(corpus.finalize_glm_bf16_corpus, **kwargs)
        with pytest.raises(corpus.CorpusContractError, match="competing producer"):
            loser.result(timeout=10)
        release_copy.set()
        assert winner.result(timeout=10) == manifest.resolve()

    loaded = corpus.load_finalized_bf16_corpus(manifest)
    assert loaded.artifact_path == output.resolve()


def test_artifact_only_crash_is_adopted_by_same_identity_retry(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, source, _ = _source_fixture(tmp_path)
    probe, _ = _probe_fixture(tmp_path)
    importance, identity = corpus.adapt_glm_importance_from_probe(incomplete, probe)
    output = tmp_path / "final.safetensors"
    manifest = tmp_path / "manifest.json"
    kwargs = dict(
        incomplete_manifest_path=incomplete,
        source_artifact_path=source,
        importance=importance,
        importance_identity=identity,
        output_artifact_path=output,
        output_manifest_path=manifest,
        calibration=_calibration(),
        model_config_sha256="b" * 64,
        prismaquant_commit="c" * 40,
        generated="2026-08-30T00:00:01+00:00",
        host="host-a",
    )
    original_publish = corpus._publish_staged_file_no_replace
    injected = False

    def crash_before_manifest(staged, destination):
        nonlocal injected
        if destination == manifest.resolve() and not injected:
            injected = True
            raise corpus.CorpusContractError("injected crash before commit marker")
        return original_publish(staged, destination)

    monkeypatch.setattr(
        corpus, "_publish_staged_file_no_replace", crash_before_manifest
    )
    with pytest.raises(corpus.CorpusContractError, match="injected crash"):
        corpus.finalize_glm_bf16_corpus(**kwargs)
    assert output.is_file()
    assert not manifest.exists()
    orphan_bytes = output.read_bytes()
    orphan_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    monkeypatch.setattr(
        corpus, "_publish_staged_file_no_replace", original_publish
    )
    output.chmod(0o644)
    output.write_bytes(orphan_bytes + b"corrupt")
    with pytest.raises(corpus.CorpusContractError, match="recovery bytes differ"):
        corpus.finalize_glm_bf16_corpus(**kwargs)
    output.write_bytes(orphan_bytes)
    output.chmod(0o444)
    kwargs.update(generated="2026-08-30T00:00:02+00:00", host="host-b")
    assert corpus.finalize_glm_bf16_corpus(**kwargs) == manifest.resolve()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == orphan_hash
    assert corpus.load_finalized_bf16_corpus(manifest).manifest["generated"] == "2026-08-30T00:00:02+00:00"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("corpus_label", "some other GLM corpus", "corpus_label"),
        ("generated", "", "nonempty string"),
        ("generated", "sometime", "ISO-8601"),
        ("generated", "2026-08-30T00:00:00", "timezone"),
        ("host", "", "nonempty string"),
    ],
)
def test_loader_refuses_semantically_unbound_manifest_identity_fields(
    tmp_path, monkeypatch, field, value, match,
):
    manifest, *_ = _finalized(tmp_path, monkeypatch)
    _rewrite_manifest(manifest, lambda payload: payload.__setitem__(field, value))
    with pytest.raises(corpus.CorpusContractError, match=match):
        corpus.load_finalized_bf16_corpus(manifest)


def test_loader_refuses_incomplete_and_duplicate_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, _source, _ = _source_fixture(tmp_path)
    with pytest.raises(corpus.CorpusContractError, match="incomplete"):
        corpus.load_finalized_bf16_corpus(incomplete)

    manifest, *_ = _finalized(tmp_path / "final", monkeypatch)
    def duplicate(payload):
        payload["entries"][-1] = dict(payload["entries"][0])
    _rewrite_manifest(manifest, duplicate)
    with pytest.raises(corpus.CorpusContractError, match="duplicate corpus tensor"):
        corpus.load_finalized_bf16_corpus(manifest)


@pytest.mark.parametrize(
    ("where", "unknown"),
    [
        ("top", "production_eligible"),
        ("entry", "unrecognized_claim"),
        ("importance_source", "source_confidence"),
        ("importance_identity", "normalization_claim"),
        ("source_artifact", "copy_verified"),
        ("census", "estimated_values"),
    ],
)
def test_loader_refuses_unknown_claim_bearing_members(
    tmp_path, monkeypatch, where, unknown,
):
    manifest, *_ = _finalized(tmp_path, monkeypatch)

    def mutate(payload):
        target = {
            "top": payload,
            "entry": payload["entries"][0],
            "importance_source": payload["entries"][0]["importance_source"],
            "importance_identity": payload["importance_identity"],
            "source_artifact": payload["source_artifact"],
            "census": payload["entries"][0]["census"],
        }[where]
        target[unknown] = True

    _rewrite_manifest(manifest, mutate)
    with pytest.raises(corpus.CorpusContractError, match=f"unknown=.*{unknown}"):
        corpus.load_finalized_bf16_corpus(manifest)


def test_loader_refuses_declared_hash_drift(tmp_path, monkeypatch):
    manifest, *_ = _finalized(tmp_path, monkeypatch)
    _rewrite_manifest(
        manifest,
        lambda payload: payload["entries"][0].__setitem__("importance_sha256", "d" * 64),
    )
    with pytest.raises(corpus.CorpusContractError, match="importance hash differs"):
        corpus.load_finalized_bf16_corpus(manifest)


def test_loader_refuses_probe_calibration_identity_drift(tmp_path, monkeypatch):
    manifest, *_ = _finalized(tmp_path, monkeypatch)
    _rewrite_manifest(
        manifest,
        lambda payload: payload["calibration"].__setitem__(
            "probe_calib_hash", "b" * 32
        ),
    )
    with pytest.raises(
        corpus.CorpusContractError,
        match="identity_sha256 does not bind",
    ):
        corpus.load_finalized_bf16_corpus(manifest)


def test_adapter_refuses_sha256_mislabelled_as_probe_calibration_hash(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(corpus, "_expected_shape", _small_shape)
    incomplete, _source, _ = _source_fixture(tmp_path)
    probe, _expected = _probe_fixture(tmp_path)
    with probe.open("rb") as handle:
        payload = pickle.load(handle)
    payload["meta"]["calibration_hash"] = "a" * 64
    with probe.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(corpus.CorpusContractError, match="BLAKE2b-128"):
        corpus.adapt_glm_importance_from_probe(incomplete, probe)


@pytest.mark.parametrize("kind", ["dtype", "shape"])
def test_loader_refuses_importance_header_dtype_or_shape(
    tmp_path, monkeypatch, kind,
):
    manifest, output, *_ = _finalized(tmp_path, monkeypatch)
    manifest_payload = json.loads(manifest.read_text())
    key = manifest_payload["entries"][0]["importance_key"]

    def mutate(header):
        if kind == "dtype":
            # Keep the byte span legal under BF16 by doubling the logical count.
            header[key]["dtype"] = "BF16"
            header[key]["shape"][0] *= 2
        else:
            header[key]["shape"] = [1, header[key]["shape"][0]]
    _rewrite_artifact_header(output, mutate)
    _rewrite_manifest(manifest, lambda payload: payload.update({
        "file_size_bytes": output.stat().st_size,
        "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }))
    with pytest.raises(corpus.CorpusContractError, match="importance artifact header differs"):
        corpus.load_finalized_bf16_corpus(manifest)


def test_loader_refuses_nonfinite_importance_even_when_raw_hash_is_redeclared(
    tmp_path, monkeypatch,
):
    manifest, output, *_ = _finalized(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    entry = payload["entries"][0]
    layout = corpus._read_safetensors_layout(output)
    start, _end = layout.tensors[entry["importance_key"]]["data_offsets"]
    output.chmod(0o644)
    with output.open("r+b") as handle:
        handle.seek(layout.data_start + start)
        handle.write(torch.tensor([float("nan")], dtype=torch.float32).numpy().tobytes())
    new_layout = corpus._read_safetensors_layout(output)
    bad_raw = corpus._tensor_bytes(new_layout, entry["importance_key"])
    entry["importance_sha256"] = hashlib.sha256(bad_raw).hexdigest()
    payload["file_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(corpus.CorpusContractError, match="importance is non-finite"):
        corpus.load_finalized_bf16_corpus(manifest)


@pytest.mark.parametrize(
    ("kind", "message"),
    [("weight", "source weight changed"), ("importance", "importance changed")],
)
def test_load_tensor_rehashes_bound_bytes_after_initial_validation(
    tmp_path, monkeypatch, kind, message,
):
    manifest, output, *_ = _finalized(tmp_path, monkeypatch)
    loaded = corpus.load_finalized_bf16_corpus(manifest)
    entry = loaded.entries[0]
    original = output.read_bytes()
    tensor_name = entry.name if kind == "weight" else entry.importance_key
    start, _end = loaded._layout.tensors[tensor_name]["data_offsets"]
    output.chmod(0o644)
    try:
        with output.open("r+b") as handle:
            position = loaded._layout.data_start + start
            handle.seek(position)
            byte = handle.read(1)
            handle.seek(position)
            handle.write(bytes([byte[0] ^ 1]))
        with pytest.raises(corpus.CorpusContractError, match=message):
            loaded.load_tensor(entry)
    finally:
        output.write_bytes(original)
        output.chmod(0o444)


def test_load_tensor_pins_one_fd_across_atomic_path_swap(
    tmp_path, monkeypatch,
):
    manifest, output, expected_raw, expected_importance, *_ = _finalized(
        tmp_path, monkeypatch
    )
    loaded = corpus.load_finalized_bf16_corpus(manifest)
    entry = loaded.entries[0]
    replacement = output.with_name("replacement.safetensors")
    replacement.write_bytes(output.read_bytes())
    replacement.chmod(0o644)
    replacement_bytes = bytearray(replacement.read_bytes())
    replacement_bytes[-1] ^= 1
    replacement.write_bytes(replacement_bytes)
    backup = output.with_name("original-pinned.safetensors")
    original_reader = corpus._tensor_bytes_from_fd
    swapped = False
    restored = False
    displaced = output.with_name("displaced-replacement.safetensors")

    def swap_after_first_read(layout, name, descriptor):
        nonlocal swapped, restored
        raw = original_reader(layout, name, descriptor)
        if not swapped:
            swapped = True
            output.replace(backup)
            replacement.replace(output)
        elif not restored:
            output.replace(displaced)
            backup.replace(output)
            restored = True
        return raw

    monkeypatch.setattr(corpus, "_tensor_bytes_from_fd", swap_after_first_read)
    try:
        weight, importance = loaded.load_tensor(entry)
        assert _tensor_bytes(weight, "BF16") == expected_raw[entry.name]
        assert torch.equal(importance, expected_importance[entry.name])
        assert swapped
        assert restored
    finally:
        if backup.exists():
            output.replace(displaced)
            backup.replace(output)
