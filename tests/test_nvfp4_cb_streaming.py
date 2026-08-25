"""Streaming NVFP4-CB exporter — CPU-only, tiny synthetic, compile-off.

Pins the streaming exporter (prismaquant.export_nvfp4_cb_streaming) against the
in-memory export_nvfp4_cb: byte-identical packed output, bounded peak
residency, per-expert->stacked bridging, fp8-source dequant-on-read, and the
stock-CT scope gate. No GPU, no torch.compile (PRISMAQUANT_CB_ENCODE_COMPILE=0).
"""
from __future__ import annotations

import hashlib
import json
import importlib
import os
import shutil
import struct
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant import nvfp4_cb_formats as cb  # noqa: E402
from prismaquant.export_nvfp4_cb import (  # noqa: E402
    export_nvfp4_cb as _export_nvfp4_cb,
)
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _LazySkeleton,
    _StreamWriter,
    export_nvfp4_cb_streaming as _export_nvfp4_cb_streaming,
    main as _cb_stream_main,
)
from prismaquant.export_native_compressed import (  # noqa: E402
    _quantize_2d,
    build_quantization_config,
    compute_nvfp4_global_real,
)
from prismaquant.cb_export_config import (  # noqa: E402
    parse_quantized_embedding_declaration,
)
from prismaquant.model_profiles import detect_profile  # noqa: E402
from prismaquant.gridbook_validation_only_policy import (  # noqa: E402
    SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
    SM120_VALIDATION_POLICY_ID,
    VALIDATION_ONLY_DISPOSITION,
    validate_sm120_validation_only_quant_config,
)
from prismaquant.shipcard import (  # noqa: E402
    CB_REQUIRED_SLOTS,
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
    verify,
)


def export_nvfp4_cb(*args, **kwargs):
    """This module's synthetic direct calls are explicit research renders."""
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb(*args, **kwargs)


def export_nvfp4_cb_streaming(*args, **kwargs):
    """This module's synthetic direct calls are explicit research renders."""
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb_streaming(*args, **kwargs)


def _st_header(path: Path) -> tuple[dict, int]:
    """Parse a safetensors file's header dict and data-start offset."""
    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + hlen]), 8 + hlen


def _assert_offsets_consistent(path: Path) -> dict:
    """The streaming header must lay tensors out gap-free, in order, with
    data_offsets matching dtype x shape (requirement a)."""
    header, _ = _st_header(path)
    _bytes = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E8M0": 1,
              "F16": 2, "BF16": 2, "I32": 4, "F32": 4, "I64": 8}
    off = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        n = 1
        for d in meta["shape"]:
            n *= int(d)
        assert lo == off, f"{name}: gap/overlap at {lo} != {off}"
        assert hi - lo == n * _bytes[meta["dtype"]], f"{name}: nbytes mismatch"
        off = hi
    return header


def _stock_by_scheme(quant_config: dict) -> dict:
    """Config groups WITHOUT a 'scheme' key (stock CT / FP8_SOURCE), normalized
    by target-set so group-key ordering doesn't matter."""
    return {tuple(sorted(g["targets"])):
            {k: v for k, v in g.items() if k != "targets"}
            for g in quant_config["config_groups"].values() if "scheme" not in g}

@pytest.fixture
def workdir(tmp_path: Path):
    """Keep synthetic exports isolated and portable across CI runners."""
    return tmp_path


def _write_model(mdl: Path, tensors: dict, hid: int = 256):
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _write_sharded_model(mdl: Path, tensors: dict, hid: int = 256):
    mdl.mkdir(parents=True, exist_ok=True)
    items = list(tensors.items())
    midpoint = max(1, len(items) // 2)
    shards = [items[:midpoint], items[midpoint:]]
    shards = [shard for shard in shards if shard]
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(dict(shard), str(mdl / name))
        weight_map.update({tensor_name: name for tensor_name, _ in shard})
    (mdl / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": weight_map,
    }))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _assign(path: Path, mapping: dict):
    path.write_text(json.dumps(mapping))


def _tensors_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_in_place_single_source_before_mutation(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        exporter(mdl, assignment, mdl, {}, device="cpu")

    assert _tree_bytes(mdl) == before


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_output_nested_under_source(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    output = mdl / "exported"
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="ancestor/descendant"):
        exporter(mdl, assignment, output, {}, device="cpu")

    assert not output.exists()
    assert _tree_bytes(mdl) == before


def test_streaming_rejects_in_place_sharded_source_before_mutation(workdir):
    mdl = workdir / "sharded-model"
    _write_sharded_model(
        mdl,
        {
            "model.embed_tokens.weight": torch.ones(2, 2),
            "model.norm.weight": torch.ones(2),
        },
        hid=2,
    )
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        export_nvfp4_cb_streaming(
            mdl,
            assignment,
            mdl,
            {},
            device="cpu",
        )

    assert _tree_bytes(mdl) == before


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_symlink_alias_and_stale_aux(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    alias = workdir / "source-alias"
    alias.symlink_to(mdl, target_is_directory=True)
    with pytest.raises(RuntimeError, match="resolve to the same path"):
        exporter(mdl, assignment, alias, {}, device="cpu")
    assert alias.is_symlink()
    assert _tree_bytes(mdl) == before

    output = workdir / "stale-output"
    output.mkdir()
    stale = output / "tokenizer_config.json"
    stale.write_text('{"old": true}')
    with pytest.raises(RuntimeError, match="is not empty"):
        exporter(mdl, assignment, output, {}, device="cpu")
    assert stale.read_text() == '{"old": true}'
    assert set(output.iterdir()) == {stale}


@pytest.mark.parametrize(
    ("module_name", "exporter"),
    [
        ("prismaquant.export_nvfp4_cb", export_nvfp4_cb),
        (
            "prismaquant.export_nvfp4_cb_streaming",
            export_nvfp4_cb_streaming,
        ),
    ],
    ids=["batch", "streaming"],
)
def test_cb_export_transaction_preserves_post_model_budget_failure(
    workdir,
    monkeypatch,
    module_name,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    output = workdir / "artifact"
    module = importlib.import_module(module_name)

    def fail_final_inventory(out_dir, *_args, **_kwargs):
        staged = Path(out_dir)
        assert staged != output
        assert (staged / "model.safetensors").is_file()
        raise RuntimeError("hard whole-artifact budget exceeded")

    monkeypatch.setattr(
        module,
        "finalize_cb_export_artifact_inventory",
        fail_final_inventory,
    )
    with pytest.raises(RuntimeError, match="hard whole-artifact budget"):
        exporter(mdl, assignment, output, {}, device="cpu")

    assert not output.exists()
    # A late inventory/budget gate runs after the model payload has been
    # written.  Preserve that transaction root so a subsequent export can use
    # it as a verified --reuse-prior instead of discarding hours of work.  The
    # final destination must remain unpublished.
    preserved = list(workdir.glob(f".{output.name}.tmp-*"))
    assert len(preserved) == 1, preserved
    assert (preserved[0] / "model.safetensors").is_file()


# --- byte-identity: dense CB + BF16 + stacked-3D experts + fp8_cb -----------

def test_streaming_byte_identical_dense_and_stacked(workdir):
    torch.manual_seed(0)
    mdl = workdir / "model"
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(3, 64, 256) * 0.3).to(torch.bfloat16),   # stacked
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                            "cb_k": 16},
        "model.layers.0.mlp.experts.gate_up_proj": {"data_type": "fp8_cb",
                                                    "cb_k": 40}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
          "model.layers.0.mlp.experts.gate_up_proj": torch.rand(3, 1, 256)
          + 0.05}
    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(cm) == dict(cs)
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    assert qm["ignore"] == qs["ignore"]
    assert qs["provenance"]["streaming"] is True
    for root, config in ((workdir / "m", qm), (workdir / "s", qs)):
        card = load_shipcard(root / "shipcard.json")
        assert Path(card["model_dir"]) == root.resolve()
        assert card["model_sha"] == compute_model_sha(root)
        assert verify(card, model_dir=root) == [
            f"{slot}: UNFILLED"
            for slot in REQUIRED_SLOTS + CB_REQUIRED_SLOTS
        ]
        assert card["build"]["quant_method"] == "gridbook"
        assert card["artifact_bytes"] == sum(
            path.stat().st_size
            for pattern in ("*.safetensors", "*.pqcb")
            for path in root.glob(pattern)
        )
        inventory = config["provenance"]["artifact_inventory"]
        files = {
            path.relative_to(root).as_posix(): path.stat().st_size
            for path in root.rglob("*") if path.is_file()
        }
        assert inventory["file_bytes"] == files
        assert inventory["file_bytes"]["shipcard.json"] == (
            root / "shipcard.json"
        ).stat().st_size
        assert inventory["file_bytes"]["shipcard.json"] == card[
            "reserved_file_bytes"
        ]
        assert inventory["export_directory_bytes"] == sum(files.values())
        assert inventory["cb_serialized_payload_bytes"] == (
            config["provenance"]["serialized_payload"]["total_bytes"]
        )
        # Every serve/gold verdict mutates the refusal receipt, but its fixed
        # reservation keeps the export-time recursive inventory and exact hard
        # budget valid through the fully closed card.
        pin = json.loads((
            Path(__file__).resolve().parents[1]
            / "prismaquant/gridbook_runtime/gridbook_runtime_pin.json"
        ).read_text())
        for slot in REQUIRED_SLOTS:
            arm = slot.rsplit(".", 1)[-1] if slot.startswith(
                "native_export."
            ) else None
            metrics = {"detail": "served"}
            tool = "inventory-stability-test"
            fingerprint = None
            if arm is not None:
                tool = "validate_cb_endpoint.py"
                fingerprint = "f" * 64
                metrics.update({
                    "arm": arm,
                    "enforce_eager": arm == "eager",
                    "quantization": "gridbook",
                    "kv_cache_dtype": "fp8",
                    "tensor_parallel_size": 1,
                    "gridbook_runtime_commit": pin["commit"],
                    "gridbook_runtime_version": pin["version"],
                })
                if arm == "graph":
                    metrics["cuda_graph"] = {
                        "capture_marker": (
                            "Graph capturing finished in 1 secs, took 1.00 GiB"
                        ),
                        "serve_log_sha256": "a" * 64,
                    }
            fill_slot(root / "shipcard.json", slot, make_record(
                slot=slot,
                tool=tool,
                passed=True,
                model_sha=card["model_sha"],
                metrics=metrics,
                spec_decode_detected=(
                    False if slot in GOLD_SLOTS or arm is not None else None
                ),
                serve_fingerprint=fingerprint,
            ))
        final_files = {
            path.relative_to(root).as_posix(): path.stat().st_size
            for path in root.rglob("*") if path.is_file()
        }
        assert final_files == inventory["file_bytes"]
        assert sum(final_files.values()) == inventory["export_directory_bytes"]
        assert compute_model_sha(root) == card["model_sha"]
        # This test owns byte identity and the fixed-size inventory
        # reservation.  Canonical Gridbook release-record semantics are
        # exercised by test_shipcard.py and the slot-specific validators; do
        # not duplicate that evolving policy in a streaming exporter fixture.
    # codebook sidecars identical
    cbm = load_file(str(workdir / "m" / "cm.pqcb")) if (
        workdir / "m" / "cm.pqcb").exists() else None
    if qs.get("codebook_file"):
        cb_s = load_file(str(workdir / "s" / qs["codebook_file"]))
        cb_m = load_file(str(workdir / "m" / qm["codebook_file"]))
        assert _tensors_equal(cb_m, cb_s)


@pytest.mark.parametrize(
    ("format_name", "expected_type_size", "expected_book_shapes"),
    [
        ("NVFP4_CB_K1", 13, [(1, 4), (2, 4)]),
        ("NVFP4_CB_K25", 109, [(4096, 4), (8192, 4)]),
    ],
    ids=["k1", "k25"],
)
def test_public_endpoint_resident_and_streaming_exports_are_byte_identical(
    workdir,
    format_name,
    expected_type_size,
    expected_book_shapes,
):
    torch.manual_seed(101)
    qname = "model.layers.0.self_attn.q_proj"
    suffix = format_name.lower()
    mdl = workdir / f"model-{suffix}"
    _write_model(mdl, {
        f"{qname}.weight": (torch.randn(8, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / f"{suffix}.json"
    _assign(assignment, {qname: format_name})
    col_weights = {qname: torch.rand(256) + 0.05}
    resident_dir = workdir / f"{suffix}-resident"
    streaming_dir = workdir / f"{suffix}-streaming"

    export_nvfp4_cb(
        mdl, assignment, resident_dir, col_weights, device="cpu"
    )
    export_nvfp4_cb_streaming(
        mdl, assignment, streaming_dir, col_weights, device="cpu"
    )
    resident = load_file(str(resident_dir / "model.safetensors"))
    streaming = load_file(str(streaming_dir / "model.safetensors"))
    assert _tensors_equal(resident, streaming)
    resident_config = json.loads(
        (resident_dir / "quant_config.json").read_text()
    )
    streaming_config = json.loads(
        (streaming_dir / "quant_config.json").read_text()
    )
    assert resident_config["config_groups"] == streaming_config["config_groups"]
    assert next(iter(resident_config["config_groups"].values()))["scheme"][
        "type_size"
    ] == expected_type_size
    resident_books = load_file(str(
        resident_dir / resident_config["codebook_file"]
    ))
    streaming_books = load_file(str(
        streaming_dir / streaming_config["codebook_file"]
    ))
    assert _tensors_equal(resident_books, streaming_books)
    assert sorted(tuple(tensor.shape) for tensor in resident_books.values()) == (
        expected_book_shapes
    )


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
@pytest.mark.parametrize(
    "format_name",
    ["NVFP4_CB_K26", "NVFP4_CB_K32"],
    ids=["k26", "k32"],
)
def test_unsupported_nvfp4_rungs_are_refused_before_output_transaction(
    workdir, exporter, format_name,
):
    """Research codec rungs have no producer transaction or filesystem trace."""
    qname = "model.layers.0.self_attn.q_proj"
    suffix = format_name.lower()
    mdl = workdir / f"model-{suffix}"
    _write_model(mdl, {
        f"{qname}.weight": torch.zeros(8, 256, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / f"{suffix}.json"
    _assign(assignment, {qname: format_name})
    col_weights = {qname: torch.ones(256)}
    out = workdir / f"{suffix}-{exporter.__name__}"
    with pytest.raises(ValueError, match="unsupported format string"):
        exporter(mdl, assignment, out, col_weights, device="cpu")
    assert not out.exists()
    assert list(workdir.glob(f".{out.name}.tmp-*")) == []


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
@pytest.mark.parametrize(
    "format_name",
    ["FP8_SOURCE", "FP8_BLOCK_UE8M0_SOURCE"],
)
def test_sm120_w8a16_is_refused_before_output_transaction(
    workdir, exporter, format_name,
):
    """Compatibility readers do not make W8A16 materializable for RTX50."""
    qname = "model.layers.0.self_attn.q_proj"
    mdl = workdir / f"model-{format_name.lower()}"
    _write_model(mdl, {
        f"{qname}.weight": torch.zeros(8, 256, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / f"{format_name.lower()}.json"
    _assign(assignment, {
        qname: format_name,
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": "qwen38_sm120_cb_validation_only",
        },
    })
    out = workdir / f"{format_name.lower()}-{exporter.__name__}"

    with pytest.raises(ValueError, match="target profile.*refuses assignment"):
        exporter(
            mdl,
            assignment,
            out,
            {qname: torch.ones(256)},
            device="cpu",
        )
    assert not out.exists()
    assert list(workdir.glob(f".{out.name}.tmp-*")) == []


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
@pytest.mark.parametrize(
    ("producer_policy", "contract_mode", "message"),
    (
        (None, "none", "requires producer_policy"),
        ("some_other_policy", "exact", "requires producer_policy"),
        (
            SM120_VALIDATION_POLICY_ID,
            "none",
            "requires an explicit exact",
        ),
        (
            SM120_VALIDATION_POLICY_ID,
            "tampered",
            "differs from exact candidate",
        ),
    ),
)
def test_both_cb_exporters_fail_closed_on_sm120_policy_contract_preflight(
    workdir,
    exporter,
    producer_policy,
    contract_mode,
    message,
):
    qname = "model.layers.0.self_attn.q_proj"
    mdl = workdir / f"model-{exporter.__name__}-{contract_mode}"
    _write_model(mdl, {
        f"{qname}.weight": torch.zeros(8, 256, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / (
        f"assignment-{exporter.__name__}-{contract_mode}.json"
    )
    _assign(assignment, {
        qname: "NVFP4_CB_K16",
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": "qwen38_sm120_cb_validation_only",
        },
    })
    if contract_mode == "exact":
        runtime_contract = SM120_VALIDATION_CANDIDATE_CONTRACT_PATH
    elif contract_mode == "tampered":
        runtime_contract = json.loads(
            SM120_VALIDATION_CANDIDATE_CONTRACT_PATH.read_text()
        )
        runtime_contract["contract_version"] = 12
    else:
        runtime_contract = None
    out = workdir / f"out-{exporter.__name__}-{contract_mode}"

    with pytest.raises(ValueError, match=message):
        exporter(
            mdl,
            assignment,
            out,
            {qname: torch.ones(256)},
            device="cpu",
            producer_policy=producer_policy,
            producer_runtime_contract=runtime_contract,
        )
    assert not out.exists()
    assert list(workdir.glob(f".{out.name}.tmp-*")) == []


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_both_cb_exporters_stamp_and_finalize_exact_sm120_policy(
    workdir,
    exporter,
):
    qname = "model.layers.0.self_attn.q_proj"
    mdl = workdir / f"model-{exporter.__name__}"
    _write_model(mdl, {
        f"{qname}.weight": torch.zeros(8, 256, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / f"assignment-{exporter.__name__}.json"
    _assign(assignment, {
        qname: "NVFP4_CB_K16",
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": "qwen38_sm120_cb_validation_only",
        },
    })
    out = workdir / f"out-{exporter.__name__}"

    assert exporter(
        mdl,
        assignment,
        out,
        {qname: torch.ones(256)},
        device="cpu",
        producer_policy=SM120_VALIDATION_POLICY_ID,
        producer_runtime_contract=SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
    )["NVFP4_CB_K16"] == 1

    quant = json.loads((out / "quant_config.json").read_text())
    validate_sm120_validation_only_quant_config(quant)
    assert quant["provenance"]["artifact_inventory"]["schema"] == (
        "prismaquant.cb_export_artifact_inventory.v1"
    )
    card = load_shipcard(out / "shipcard.json")
    assert card["build"]["producer_policy"] == SM120_VALIDATION_POLICY_ID
    assert card["build"]["artifact_disposition"] == (
        VALIDATION_ONLY_DISPOSITION
    )
    assert any(
        VALIDATION_ONLY_DISPOSITION in problem
        for problem in verify(card, model_dir=out, required=())
    )


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_emit_gated_ldlq_bytes(
    workdir, exporter, monkeypatch,
):
    """Both final exporters must ship the gate's arm, not bare LDLQ."""
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1")
    monkeypatch.delenv("PRISMAQUANT_CB_LDLQ_SCOPE", raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "holdout")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")

    generator = torch.Generator().manual_seed(0)
    qname = "model.layers.0.self_attn.q_proj"
    weight = (torch.randn(8, 256, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    col_weights = torch.rand(256, generator=generator) + 0.05
    activation_rows = torch.randn(1, 256, generator=generator)
    model = workdir / "model"
    _write_model(model, {f"{qname}.weight": weight})
    assignment = workdir / "assignment.json"
    _assign(assignment, {qname: {"data_type": "nvfp4_cb", "cb_k": 12}})
    activation_cache = workdir / "activation-cache"
    activation_cache.mkdir()
    torch.save(
        {"inputs": activation_rows, "name": qname},
        activation_cache / "model__layers__0__self_attn__q_proj.pt",
    )

    # One activation row is uncertifiable under the holdout gate, hence the
    # expected artifact bytes are the ordinary same-format assignment.
    expected, _ = cb.nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="balanced",
    )
    out = workdir / exporter.__name__
    exporter(
        model,
        assignment,
        out,
        {qname: col_weights},
        device="cpu",
        activation_cache_dir=activation_cache,
    )
    shipped = load_file(str(out / "model.safetensors"))[
        f"{qname}.cb_qweight"
    ]
    assert torch.equal(shipped, expected)


def test_streaming_warm_fallback_counts_are_artifact_provenance(workdir):
    qname = "model.layers.0.self_attn.q_proj"
    mdl = workdir / "warm-provenance-model"
    _write_model(
        mdl,
        {f"{qname}.weight": torch.randn(2, 256).to(torch.bfloat16)},
    )
    assignment = workdir / "warm-provenance-assignment.json"
    _assign(assignment, {qname: "FP8_CB_K28"})

    counts = export_nvfp4_cb_streaming(
        mdl,
        assignment,
        workdir / "warm-provenance-export",
        {qname: torch.ones(256)},
        device="cpu",
        warm_state_dir=workdir / "empty-warm-state",
        warm_verify_sample=32,
    )

    config = json.loads(
        (workdir / "warm-provenance-export" / "quant_config.json").read_text()
    )
    expected = {"warm_used": 0, "cold_fallback": 1, "verified_n": 0}
    assert config["provenance"]["encoder_warm_start"] == expected
    assert {key: counts[key] for key in expected} == expected


def test_streaming_global_cb_stamp_does_not_bypass_missing_render_identity(
    workdir,
):
    from prismaquant.nvfp4_cb_footprint import (
        CBSerializationContext,
        cb_serialization_context_stamp,
    )

    qname = "model.layers.0.self_attn.o_proj"
    mdl = workdir / "model"
    _write_model(
        mdl,
        {f"{qname}.weight": torch.randn(2, 256).to(torch.bfloat16)},
    )
    ap = workdir / "a.json"
    _assign(ap, {
        qname: {"data_type": "nvfp4_cb", "cb_k": 16},
        "__prismaquant__": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.production(),
                formats=["NVFP4_CB_K16"],
            ),
        },
    })
    with pytest.raises(ValueError, match="missing its value-bearing render identity"):
        export_nvfp4_cb_streaming(
            mdl,
            ap,
            workdir / "s",
            {qname: torch.ones(256)},
            device="cpu",
        )


def test_streaming_rejects_unstamped_cb_recipe_by_default(workdir):
    qname = "model.layers.0.self_attn.o_proj"
    mdl = workdir / "model"
    _write_model(
        mdl,
        {f"{qname}.weight": torch.randn(2, 256).to(torch.bfloat16)},
    )
    ap = workdir / "a.json"
    _assign(ap, {qname: "NVFP4_CB_K16"})
    with pytest.raises(ValueError, match="value-bearing render identity"):
        _export_nvfp4_cb_streaming(
            mdl,
            ap,
            workdir / "s",
            {qname: torch.ones(256)},
            device="cpu",
        )


@pytest.mark.parametrize("failure_stage", ["producer", "before_publish"])
def test_stream_writer_removes_owned_temp_before_publish(workdir, failure_stage):
    output = workdir / "model.safetensors"
    writer = _StreamWriter()

    def producer():
        if failure_stage == "producer":
            raise RuntimeError("producer failed")
        return torch.ones(4, dtype=torch.float32)

    def before_publish():
        if failure_stage == "before_publish":
            raise RuntimeError("coverage failed")

    writer.add("value", torch.float32, (4,), producer)
    with pytest.raises(RuntimeError, match="failed"):
        writer.write(output, before_publish=before_publish)
    assert not output.exists()
    assert not (workdir / ".model.safetensors.tmp").exists()
    assert writer.last_content_sha256 is None
    assert writer.last_content_bytes is None


def test_stream_writer_attests_exact_published_bytes_without_a_reread(workdir):
    output = workdir / "model.safetensors"
    writer = _StreamWriter()
    writer.add(
        "value",
        torch.float32,
        (4,),
        lambda: torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    writer.write(output)

    assert writer.last_content_bytes == output.stat().st_size
    assert writer.last_content_sha256 == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


# --- per-expert -> stacked bridging (Hy3 layout) ---------------------------

def _per_expert_model(mdl: Path, E=3, inter=256, hid=256, seed=1):
    torch.manual_seed(seed)
    pe = {}
    for e in range(E):
        pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = (
            torch.randn(hid, inter) * 0.3).to(torch.bfloat16)
    pe["model.norm.weight"] = torch.ones(hid, dtype=torch.bfloat16)
    _write_model(mdl, pe, hid)
    # equivalent pre-stacked model for the in-memory reference
    gu = torch.stack([
        torch.cat([pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"],
                   pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"]], 0)
        for e in range(E)])
    dn = torch.stack([pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"]
                      for e in range(E)])
    st = {"model.layers.1.mlp.experts.gate_up_proj.weight": gu,
          "model.layers.1.mlp.experts.down_proj.weight": dn,
          "model.norm.weight": pe["model.norm.weight"]}
    return st


def test_streaming_per_expert_bridging(workdir):
    E, inter, hid = 3, 256, 256
    st = _per_expert_model(workdir / "pe", E, inter, hid)
    _write_model(workdir / "st", st, hid)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.1.mlp.experts.gate_up_proj": {"data_type": "nvfp4_cb",
                                                    "cb_k": 16},
        "model.layers.1.mlp.experts.down_proj": {"data_type": "nvfp4_cb",
                                                 "cb_k": 16}})
    cw = {"model.layers.1.mlp.experts.gate_up_proj": torch.rand(E, 1, hid)
          + 0.05,
          "model.layers.1.mlp.experts.down_proj": torch.rand(E, 1, inter)
          + 0.05}
    export_nvfp4_cb(workdir / "st", ap, workdir / "m", cw, device="cpu")
    export_nvfp4_cb_streaming(workdir / "pe", ap, workdir / "s", cw,
                              device="cpu")
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    for key in ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
                "model.layers.1.mlp.experts.down_proj.cb_qweight"):
        assert torch.equal(tm[key], ts[key]), key


def _lfm_per_expert_model(mdl: Path, layers=(2,), E=2, inter=256, hid=256,
                          seed=17):
    """Tiny LFM checkpoint layout: per-expert w1/w3 fuse to gate_up and w2
    maps to down. Dimensions stay superblock-legal for real CB packing."""
    torch.manual_seed(seed)
    tensors = {"model.embedding_norm.weight":
               torch.ones(hid, dtype=torch.bfloat16)}
    for layer in layers:
        prefix = f"model.layers.{layer}.feed_forward.experts"
        for e in range(E):
            tensors[f"{prefix}.{e}.w1.weight"] = (
                torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
            tensors[f"{prefix}.{e}.w3.weight"] = (
                torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
            tensors[f"{prefix}.{e}.w2.weight"] = (
                torch.randn(hid, inter) * 0.3).to(torch.bfloat16)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": inter,
        "num_local_experts": E,
    }))
    return tensors


def _lfm_cb_assignment(layer: int) -> dict:
    prefix = f"model.layers.{layer}.feed_forward.experts"
    return {
        f"{prefix}.gate_up_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        f"{prefix}.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    }


def _lfm_col_weights(layer: int, E=2, inter=256, hid=256) -> dict:
    prefix = f"model.layers.{layer}.feed_forward.experts"
    return {
        f"{prefix}.gate_up_proj": torch.rand(E, 1, hid) + 0.05,
        f"{prefix}.down_proj": torch.rand(E, 1, inter) + 0.05,
    }


def test_streaming_lfm_profile_declared_expert_projections(workdir):
    """Regression: streaming must discover LFM's w1/w2/w3 source tensors via
    the profile rather than a gate_proj/up_proj/down_proj-only regex."""
    mdl = workdir / "lfm"
    _lfm_per_expert_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _lfm_cb_assignment(2))
    cw = _lfm_col_weights(2)

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert cm == cs
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    prefix = "model.layers.2.feed_forward.experts"
    assert f"{prefix}.gate_up_proj.cb_qweight" in ts
    assert f"{prefix}.down_proj.cb_qweight" in ts
    assert not any(k.startswith(prefix + ".0.") for k in ts)


def _lfm_direct_packed_model(mdl: Path, layers=(2,), E=2, inter=256,
                             hid=256, seed=19):
    """LFM's live ``nn.Parameter`` save layout has no ``.weight`` suffix."""
    torch.manual_seed(seed)
    tensors = {"model.embedding_norm.weight":
               torch.ones(hid, dtype=torch.bfloat16)}
    for layer in layers:
        prefix = f"model.layers.{layer}.feed_forward.experts"
        tensors[f"{prefix}.gate_up_proj"] = (
            torch.randn(E, 2 * inter, hid) * 0.3).to(torch.bfloat16)
        tensors[f"{prefix}.down_proj"] = (
            torch.randn(E, hid, inter) * 0.3).to(torch.bfloat16)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": inter,
        "num_local_experts": E,
    }))
    return tensors


def test_streaming_lfm_direct_packed_source_is_validated_and_exported(workdir):
    """Regression for packed LFM saves whose expert parameter keys are the
    direct qnames. Both exporters must quantize selected rank-3 tensors while
    preserving direct BF16 banks and marking them ignored."""
    mdl = workdir / "lfm-packed"
    original = _lfm_direct_packed_model(mdl, layers=(2, 3))
    ap = workdir / "a.json"
    assignment = _lfm_cb_assignment(2)
    bf16_prefix = "model.layers.3.feed_forward.experts"
    assignment.update({
        f"{bf16_prefix}.gate_up_proj": "BF16",
        f"{bf16_prefix}.down_proj": "BF16",
    })
    _assign(ap, assignment)
    cw = _lfm_col_weights(2)

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert cm == cs
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    cb_prefix = "model.layers.2.feed_forward.experts"
    assert f"{cb_prefix}.gate_up_proj.cb_qweight" in ts
    assert f"{cb_prefix}.down_proj.cb_qweight" in ts
    for projection in ("gate_up_proj", "down_proj"):
        key = f"{bf16_prefix}.{projection}"
        assert torch.equal(ts[key], original[key])
    quant_config = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert {f"{bf16_prefix}.gate_up_proj", f"{bf16_prefix}.down_proj"} \
        <= set(quant_config["ignore"])


def test_streaming_lfm_direct_packed_source_rejects_non_3d(workdir):
    mdl = workdir / "lfm-packed-bad"
    tensors = _lfm_direct_packed_model(mdl)
    key = "model.layers.2.feed_forward.experts.gate_up_proj"
    tensors[key] = torch.randn(512, 256).to(torch.bfloat16)
    save_file(tensors, str(mdl / "model.safetensors"))
    ap = workdir / "a.json"
    _assign(ap, _lfm_cb_assignment(2))
    with pytest.raises(ValueError, match="direct packed expert source.*rank-3"):
        export_nvfp4_cb_streaming(
            mdl, ap, workdir / "s", _lfm_col_weights(2), device="cpu")


@pytest.mark.parametrize("streaming", [False, True])
def test_partial_lfm_mixed_export_preserves_bf16_per_expert(
        workdir, streaming):
    """A CB layer may coexist with an explicitly BF16 LFM layer. Only the CB
    parents are stacked; BF16 stays as the original per-expert checkpoint
    tensors expected by LFM's vLLM loader."""
    mdl = workdir / "lfm"
    original = _lfm_per_expert_model(mdl, layers=(2, 3))
    ap = workdir / "a.json"
    assignment = _lfm_cb_assignment(2)
    bf16_prefix = "model.layers.3.feed_forward.experts"
    assignment.update({
        f"{bf16_prefix}.gate_up_proj": {
            "data_type": "bfloat16", "bits": 16},
        f"{bf16_prefix}.down_proj": {
            "data_type": "bfloat16", "bits": 16},
    })
    _assign(ap, assignment)
    out = workdir / ("streaming" if streaming else "memory")
    exporter = export_nvfp4_cb_streaming if streaming else export_nvfp4_cb
    exporter(mdl, ap, out, _lfm_col_weights(2), device="cpu")

    exported = load_file(str(out / "model.safetensors"))
    cb_prefix = "model.layers.2.feed_forward.experts"
    assert f"{cb_prefix}.gate_up_proj.cb_qweight" in exported
    assert f"{cb_prefix}.down_proj.cb_qweight" in exported
    assert not any(k.startswith(cb_prefix + ".0.") for k in exported)
    assert f"{bf16_prefix}.gate_up_proj.weight" not in exported
    assert f"{bf16_prefix}.down_proj.weight" not in exported
    expected_ignore = set()
    for e in range(2):
        for proj in ("w1", "w2", "w3"):
            key = f"{bf16_prefix}.{e}.{proj}.weight"
            assert torch.equal(exported[key], original[key])
            expected_ignore.add(key[:-len(".weight")])
    quant_config = json.loads((out / "quant_config.json").read_text())
    assert expected_ignore <= set(quant_config["ignore"])


# --- bounded peak residency ------------------------------------------------

def test_streaming_peak_residency(workdir, monkeypatch):
    torch.manual_seed(2)
    tens = {"model.norm.weight": torch.ones(256, dtype=torch.bfloat16)}
    # many passthrough tensors so full materialization would be obvious
    for i in range(20):
        tens[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(
            256, dtype=torch.bfloat16)
    tens["model.layers.0.self_attn.q_proj.weight"] = (
        torch.randn(128, 256) * 0.3).to(torch.bfloat16)
    mdl = workdir / "model"
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                                     "cb_k": 16}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}

    live = {"n": 0, "peak": 0}
    orig = _LazySkeleton.load

    def counting(self, name):
        t = orig(self, name)
        live["n"] += 1
        live["peak"] = max(live["peak"], live["n"])
        weakref.finalize(t, lambda: live.__setitem__("n", live["n"] - 1))
        return t
    monkeypatch.setattr(_LazySkeleton, "load", counting)
    export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    # 22 source tensors total; peak resident must be a tiny constant, not ~22.
    assert live["peak"] <= 4, f"peak residency {live['peak']} too high"


# --- fp8-source dequant-on-read (DSv4 ingestion) ---------------------------

def test_streaming_fp8_source_dequant_on_read(workdir):
    from prismaquant.layer_streaming import _dequant_fp8_block_weight
    torch.manual_seed(3)
    out_f, in_f = 256, 256
    w_fp8 = (torch.randn(out_f, in_f) * 0.3).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(out_f // 128, in_f // 128) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.down_proj.weight": w_fp8,
        "model.layers.0.mlp.down_proj.weight_scale_inv": scale_inv,
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)},
        hid=256)
    (mdl / "config.json").write_text(json.dumps({
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]}}))
    sk = _LazySkeleton(mdl)
    got = sk.dequant_weight("model.layers.0.mlp.down_proj.weight")
    ref = _dequant_fp8_block_weight(w_fp8, scale_inv, block=(128, 128)).float()
    assert torch.equal(got, ref)


def test_fp8_cb_resident_and_streaming_exports_are_byte_identical(workdir):
    torch.manual_seed(31)
    qname = "model.layers.0.mlp.down_proj"
    w_fp8 = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        f"{qname}.weight": w_fp8,
        f"{qname}.weight_scale_inv": scale_inv,
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    (mdl / "config.json").write_text(json.dumps({
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    ap = workdir / "a.json"
    _assign(ap, {qname: "FP8_CB_K36"})
    cw = {qname: torch.rand(256) + 0.05}

    export_nvfp4_cb(mdl, ap, workdir / "batch", cw, device="cpu")
    export_nvfp4_cb_streaming(
        mdl, ap, workdir / "stream", cw, device="cpu"
    )
    batch = load_file(str(workdir / "batch" / "model.safetensors"))
    stream = load_file(str(workdir / "stream" / "model.safetensors"))
    assert _tensors_equal(batch, stream)


def test_lazy_skeleton_rejects_unscaled_raw_fp8_codes(workdir, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_ALLOW_UNSCALED_FP8", raising=False)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.down_proj.weight":
            torch.ones(128, 256).to(torch.float8_e4m3fn),
    })
    skeleton = _LazySkeleton(mdl)
    with pytest.raises(RuntimeError, match="has no entry"):
        skeleton.dequant_weight("model.layers.0.mlp.down_proj.weight")


def test_lazy_skeleton_uses_profile_defined_scale_pair(workdir):
    from prismaquant.layer_streaming import _dequant_fp8_block_weight

    torch.manual_seed(32)
    weight_key = "layers.0.attn.q_proj.weight"
    w_fp8 = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {weight_key: w_fp8, "layers.0.attn.q_proj.scale": scale})
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    skeleton = _LazySkeleton(mdl)
    got = skeleton.dequant_weight(weight_key)
    ref = _dequant_fp8_block_weight(
        w_fp8, scale, block=(128, 128), name=weight_key
    ).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(got, ref)


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_fp8_source_uses_profile_defined_scale_pair(workdir, exporter):
    torch.manual_seed(33)
    checkpoint_base = "layers.0.attn.q_proj"
    live_base = "model.layers.0.self_attn.q_proj"
    weight = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        checkpoint_base + ".weight": weight,
        checkpoint_base + ".scale": scale,
    })
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    ap = workdir / "a.json"
    _assign(ap, {live_base: {
        "data_type": "fp8_e4m3", "bits": 8, "group_size": 128,
    }})
    out = workdir / "out"
    exporter(mdl, ap, out, {}, device="cpu")
    tensors = load_file(str(out / "model.safetensors"))
    assert torch.equal(
        tensors[checkpoint_base + ".weight"].view(torch.uint8),
        weight.view(torch.uint8),
    )
    assert torch.equal(tensors[checkpoint_base + ".weight_scale"], scale)
    assert checkpoint_base + ".scale" not in tensors


def test_resident_export_rejects_profile_scaled_per_expert_source(workdir):
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    mdl = workdir / "model"
    _write_model(mdl, {
        weight_key: torch.ones(128, 256).to(torch.float8_e4m3fn),
        "layers.0.ffn.experts.0.w1.scale": torch.ones(1, 2),
    })
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    qname = "model.layers.0.mlp.experts.gate_up_proj"
    ap = workdir / "a.json"
    _assign(ap, {qname: "FP8_CB_K36"})
    with pytest.raises(ValueError, match="profile-scaled FP8/MXFP4"):
        export_nvfp4_cb(
            mdl,
            ap,
            workdir / "batch",
            {qname: torch.ones(256)},
            device="cpu",
        )


# --- mixed-menu: CB + stock NVFP4 + stock FP8_DYNAMIC + BF16 ----------------

def _mixed_menu_model(mdl: Path):
    """q/k on stock NVFP4 (fused siblings — shared global), gate on stock
    FP8_DYNAMIC, down on CB (nvfp4_cb), up on BF16 passthrough."""
    torch.manual_seed(7)
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight":
            (torch.randn(64, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight":
            (torch.randn(64, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    return tens


_MIXED_ASSIGN = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nv_fp", "bits": 4},
    "model.layers.0.self_attn.k_proj": {"data_type": "nv_fp", "bits": 4},
    "model.layers.0.mlp.gate_proj": {"data_type": "fp8_e4m3", "bits": 8,
                                     "group_size": 0},               # FP8_DYNAMIC
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.up_proj": {"data_type": "bfloat16", "bits": 16},
}


def test_streaming_mixed_menu_byte_identical(workdir):
    mdl = workdir / "model"
    tens = _mixed_menu_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _MIXED_ASSIGN)
    cw = {"model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(cm) == dict(cs)
    assert cs["NVFP4"] == 2 and cs["FP8_E4M3"] == 1 and cs["NVFP4_CB_K16"] == 1

    # (a) header/offsets consistency of the streamed file.
    _assert_offsets_consistent(workdir / "s" / "model.safetensors")

    # (b) every tensor byte-identical to the in-memory exporter (which itself
    #     calls the export_native_compressed packers).
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)

    # (b) stock tensor BYTES identical to the packers called directly. q/k are
    #     fused NVFP4 siblings -> they share the max global_real.
    gq = compute_nvfp4_global_real(
        tens["model.layers.0.self_attn.q_proj.weight"].float(), 16).reshape(())
    gk = compute_nvfp4_global_real(
        tens["model.layers.0.self_attn.k_proj.weight"].float(), 16).reshape(())
    shared = torch.stack([gq, gk]).max()
    for leaf in ("q_proj", "k_proj"):
        direct = _quantize_2d(
            tens[f"model.layers.0.self_attn.{leaf}.weight"].float(), "NVFP4",
            nvfp4_global_real_override=shared)
        for suffix, t in direct.items():
            assert torch.equal(
                ts[f"model.layers.0.self_attn.{leaf}.{suffix}"], t), \
                f"{leaf}.{suffix}"
    fp8 = _quantize_2d(
        tens["model.layers.0.mlp.gate_proj.weight"].float(), "FP8_E4M3")
    for suffix, t in fp8.items():
        assert torch.equal(ts[f"model.layers.0.mlp.gate_proj.{suffix}"], t), \
            suffix

    # stock groups have NO "scheme" key; CB groups DO (the dispatch marker).
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    stock = _stock_by_scheme(qs)
    assert len(stock) == 2                       # one NVFP4 group, one FP8 group
    assert any(g["format"] == "nvfp4-pack-quantized" for g in stock.values())
    assert any(g["format"] == "float-quantized" for g in stock.values())
    assert "model.layers.0.mlp.up_proj" in qs["ignore"]      # BF16 passthrough

    # (c) stock config vocabulary equals build_quantization_config's (flat model
    #     -> DefaultProfile -> no greedy per-expert catch-all regex, so exact).
    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    prof = detect_profile(str(mdl))
    bqc = build_quantization_config(
        {"model.layers.0.self_attn.q_proj": "NVFP4",
         "model.layers.0.self_attn.k_proj": "NVFP4",
         "model.layers.0.mlp.gate_proj": "FP8_E4M3"}, set(), profile=prof)
    assert _stock_by_scheme(qs) == _stock_by_scheme(
        {"config_groups": bqc["config_groups"]})


def test_streaming_resume_fails_closed_without_producer_identity(workdir):
    # A matching tensor header cannot prove that source/imatrix/codebook bytes
    # or exporter code are unchanged, even at a valid group boundary.
    mdl = workdir / "model"
    _mixed_menu_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _MIXED_ASSIGN)
    cw = {"model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}
    out = workdir / "s"
    export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu")
    ref = (out / "model.safetensors").read_bytes()

    header, data0 = _st_header(out / "model.safetensors")
    # cut mid-weight_scale of the stock NVFP4 q_proj group (after weight_packed,
    # before the group ends) so RESUME must re-enter the stock group.
    wp = header["model.layers.0.self_attn.q_proj.weight_packed"]["data_offsets"]
    ws = header["model.layers.0.self_attn.q_proj.weight_scale"]["data_offsets"]
    cut = data0 + (ws[0] + ws[1]) // 2
    assert wp[1] <= ws[0] <= (cut - data0) < ws[1]
    (out / "model.safetensors").write_bytes(ref[:cut])   # truncate mid-group

    with pytest.raises(RuntimeError, match="output directory .* is not empty"):
        export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu")


# --- stock rungs on MoE expert stacks are gated off ------------------------

def test_streaming_rejects_stock_expert_stack(workdir):
    # Per-expert on-disk MoE, packed parent assigned a stock format -> gated.
    E, inter, hid = 3, 256, 256
    _per_expert_model(workdir / "pe", E, inter, hid)
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.1.mlp.experts.gate_up_proj":
                 {"data_type": "nv_fp", "bits": 4}})     # NVFP4 on an expert stack
    cw = {"model.layers.1.mlp.experts.gate_up_proj":
          torch.rand(E, 1, hid) + 0.05}
    with pytest.raises(ValueError, match="expert-stack"):
        export_nvfp4_cb_streaming(workdir / "pe", ap, workdir / "s", cw,
                                  device="cpu")


def test_streaming_rejects_stock_stacked_3d_tensor(workdir):
    # Already-stacked 3-D expert tensor assigned a stock format -> gated too.
    torch.manual_seed(8)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(3, 64, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)})
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.mlp.experts.gate_up_proj":
                 {"data_type": "fp8_e4m3", "bits": 8, "group_size": 0}})
    with pytest.raises(ValueError, match="expert-stack"):
        export_nvfp4_cb_streaming(mdl, ap, workdir / "s", {}, device="cpu")


# --- hy_v3 shared_mlp: stock config target collapses via to_vllm_internal_name

def test_streaming_stock_shared_mlp_vllm_target(workdir):
    torch.manual_seed(9)
    mdl = workdir / "hy"
    mdl.mkdir(parents=True, exist_ok=True)
    save_file({
        "model.layers.5.mlp.shared_mlp.gate_up_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.5.mlp.shared_mlp.down_proj.weight":
            (torch.randn(256, 64) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"model_type": "hy_v3", "hidden_size": 256}))
    ap = workdir / "a.json"
    # recipe (live) names use `shared_experts`; checkpoint uses `shared_mlp`.
    _assign(ap, {
        "model.layers.5.mlp.shared_experts.gate_up_proj":
            {"data_type": "fp8_e4m3", "bits": 8, "group_size": 0},
        "model.layers.5.mlp.shared_experts.down_proj":
            {"data_type": "nv_fp", "bits": 4}})
    export_nvfp4_cb_streaming(mdl, ap, workdir / "s", {}, device="cpu")

    ts = load_file(str(workdir / "s" / "model.safetensors"))
    # tensors keep the CHECKPOINT name (params live under .shared_mlp.*).
    assert "model.layers.5.mlp.shared_mlp.gate_up_proj.weight" in ts
    assert "model.layers.5.mlp.shared_mlp.down_proj.weight_packed" in ts

    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    stock_targets = {t for g in qs["config_groups"].values()
                     if "scheme" not in g for t in g["targets"]}
    # config targets COLLAPSE .shared_mlp. -> .mlp. (to_vllm_internal_name),
    # matching vLLM's dispatch prefix and build_quantization_config (28b6862).
    assert "re:^model[.]layers[.]5[.]mlp[.]gate_up_proj$" in stock_targets
    assert "re:^model[.]layers[.]5[.]mlp[.]down_proj$" in stock_targets
    assert not any(".shared_mlp." in t or ".shared_experts." in t
                   for t in stock_targets)
    prof = detect_profile(str(mdl))
    bqc = build_quantization_config(
        {"model.layers.5.mlp.shared_experts.gate_up_proj": "FP8_E4M3",
         "model.layers.5.mlp.shared_experts.down_proj": "NVFP4"},
        set(), profile=prof)
    bqc_targets = {t for g in bqc["config_groups"].values()
                   for t in g["targets"]}
    # the collapsed explicit targets we emit are exactly the ones vLLM's own
    # config builder emits (modulo its greedy per-expert catch-all regex).
    assert stock_targets <= bqc_targets


# --- lazy skeleton: single-file + sharded, metadata without data load ------

def test_lazy_skeleton_metadata(workdir):
    torch.manual_seed(5)
    mdl = workdir / "model"
    _write_model(mdl, {
        "a.weight": torch.randn(16, 32),
        "b.weight": (torch.randn(8, 8)).to(torch.bfloat16)})
    sk = _LazySkeleton(mdl)
    assert "a.weight" in sk and "b.weight" in sk
    assert sk.get_shape("a.weight") == (16, 32)
    assert sk.get_dtype("b.weight") == torch.bfloat16
    assert torch.equal(sk.load("a.weight"), load_file(
        str(mdl / "model.safetensors"))["a.weight"])


# --- DELTA-EXPORT reuse (PRISMAQUANT_EXPORT_REUSE_PRIOR) --------------------
#
# Two dense CB Linears + a BF16 passthrough norm. On a re-allocation most CB
# targets keep their (format, scheme, codebook), so a re-encode reproduces the
# exact bytes — the reuse path byte-copies them from a prior artifact instead.

_REUSE_ASSIGN_A = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
}
# down_proj alone moves K16 -> K20 (a different CB format string) in B.
_REUSE_ASSIGN_B = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 20},
}


def _reuse_model(mdl: Path, seed: int = 11) -> dict:
    torch.manual_seed(seed)
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    return tens


def _reuse_cw() -> dict:
    torch.manual_seed(99)
    return {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
            "model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}


def _reshard(src: Path, dst: Path):
    """Re-serialize a single-file artifact as a 2-shard artifact + index."""
    dst.mkdir(parents=True, exist_ok=True)
    tens = load_file(str(src / "model.safetensors"))
    keys = list(tens)
    half = max(1, len(keys) // 2)
    g1 = {k: tens[k] for k in keys[:half]}
    g2 = {k: tens[k] for k in keys[half:]}
    save_file(g1, str(dst / "model-00001-of-00002.safetensors"),
              metadata={"format": "pt"})
    save_file(g2, str(dst / "model-00002-of-00002.safetensors"),
              metadata={"format": "pt"})
    wm = {**{k: "model-00001-of-00002.safetensors" for k in g1},
          **{k: "model-00002-of-00002.safetensors" for k in g2}}
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
    shutil.copy(src / "quant_config.json", dst / "quant_config.json")
    qp = json.loads((src / "quant_config.json").read_text())
    if qp.get("codebook_file"):
        shutil.copy(src / qp["codebook_file"], dst / qp["codebook_file"])
    if (src / "config.json").exists():
        shutil.copy(src / "config.json", dst / "config.json")


# (1) reuse disabled == today: byte-identical + no reuse_* keys leak.
def test_reuse_disabled_is_noop(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    cw = _reuse_cw()
    c0 = export_nvfp4_cb_streaming(mdl, ap, workdir / "s0", cw, device="cpu")
    c1 = export_nvfp4_cb_streaming(mdl, ap, workdir / "s1", cw, device="cpu",
                                   reuse_prior=None)
    assert (workdir / "s0" / "model.safetensors").read_bytes() == \
        (workdir / "s1" / "model.safetensors").read_bytes()
    assert dict(c0) == dict(c1)
    assert not any(str(k).startswith("reuse_") for k in c0)
    assert not any(str(k).startswith("reuse_") for k in c1)


# (2) reuse is fail-closed until every copied tensor has an immutable producer
# identity (source bytes + imatrix + codebook + scheme + exporter ABI).
def test_reuse_prior_is_blocked_before_any_output(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    cw = _reuse_cw()
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")   # fresh
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior, reuse_verify=2
        )
    assert not (out / "model.safetensors").exists()


# (3) changed-format target re-encodes; unchanged one still copies.
def test_reuse_changed_format_is_still_blocked_globally(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    apA = workdir / "a.json"
    _assign(apA, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, apA, prior, cw, device="cpu")
    apB = workdir / "b.json"
    _assign(apB, _REUSE_ASSIGN_B)
    fresh_b = workdir / "freshB"
    export_nvfp4_cb_streaming(mdl, apB, fresh_b, cw, device="cpu")   # reference
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, apB, out, cw, device="cpu", reuse_prior=prior, reuse_verify=5
        )


# (4) codebook byte-mismatch makes every CB target on that group ineligible.
def test_reuse_codebook_mismatch_does_not_reenable_copy(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    qp = json.loads((prior / "quant_config.json").read_text())
    cbf = prior / qp["codebook_file"]
    cbt = load_file(str(cbf))
    cbt = {k: (v + 1.0).to(v.dtype).contiguous() for k, v in cbt.items()}
    save_file(cbt, str(cbf), metadata={"format": "pt"})   # perturb codebook
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior
        )


# (5) even a positive sample count cannot turn an unbound prior into proof.
def test_reuse_sampling_cannot_override_missing_identity(workdir):
    torch.manual_seed(13)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj":
                 {"data_type": "nvfp4_cb", "cb_k": 16}})
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    header, data0 = _st_header(prior / "model.safetensors")
    off = header["model.layers.0.self_attn.q_proj.cb_qweight"]["data_offsets"]
    raw = bytearray((prior / "model.safetensors").read_bytes())
    raw[data0 + off[0]] ^= 0xFF                   # flip one packed-code byte
    (prior / "model.safetensors").write_bytes(bytes(raw))
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu",
                                  reuse_prior=prior, reuse_verify=1)
    assert not (out / "model.safetensors").exists()   # nothing shipped


# (6) sharded prior artifact (index.json + model-XXXXX-of-XXXXX) read path.
def test_reuse_sharded_prior_is_blocked_before_index_read(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    single = workdir / "prior_single"
    export_nvfp4_cb_streaming(mdl, ap, single, cw, device="cpu")
    sharded = workdir / "prior_sharded"
    _reshard(single, sharded)
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=sharded, reuse_verify=2
        )


# (7) main()/CLI env fallback (PRISMAQUANT_EXPORT_REUSE_PRIOR) — the exact path
# run-pipeline.sh drives tonight.
def test_reuse_main_env_fallback(workdir, monkeypatch):
    import pickle
    import prismaquant.gpu_guard as gpu_guard

    # The production CLI is intentionally GPU-only. This test exercises only
    # argument/environment plumbing on tiny CPU tensors, so isolate that policy
    # guard instead of weakening it in production code.
    monkeypatch.setattr(
        gpu_guard,
        "require_cuda_hot_path",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    mdl = workdir / "model"
    _reuse_model(mdl)
    cwp = workdir / "cw.pkl"
    with open(cwp, "wb") as f:
        pickle.dump(_reuse_cw(), f)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    base = ["--model-dir", str(mdl), "--layer-config", str(ap),
            "--col-weights", str(cwp), "--device", "cpu",
            "--allow-unstamped-research"]
    prior = workdir / "prior"
    monkeypatch.delenv("PRISMAQUANT_EXPORT_REUSE_PRIOR", raising=False)
    _cb_stream_main(base + ["--out", str(prior)])                  # fresh
    out = workdir / "delta"
    monkeypatch.setenv("PRISMAQUANT_EXPORT_REUSE_PRIOR", str(prior))
    monkeypatch.setenv("PRISMAQUANT_EXPORT_REUSE_VERIFY", "2")
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        _cb_stream_main(base + ["--out", str(out)])


# (8) reuse is rejected before an existing output can be mistaken for a bound
# resume journal.
def test_reuse_does_not_bypass_resume_gate(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    out = workdir / "delta"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"unbound-partial")
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior, reuse_verify=1
        )


# --- source-passthrough family: byte-verbatim native lanes ------------------
#
# Two formats, one wire contract: the exporter copies the checkpoint's own
# element plane and its own scale plane, under the checkpoint's own names,
# without ever building a tensor. What these tests actually pin is that the
# artifact's bytes ARE the source's bytes — a passthrough that re-serializes
# "equivalently" is not a passthrough, and the failure is invisible to any
# assertion phrased in terms of decoded values.

_SOURCE_HID = 256          # CB needs logical in_features % 256 == 0
_SOURCE_EXPERTS = 2
_MXFP4_RECIPE = {"data_type": "fp4_e2m1", "bits": 4, "group_size": 32}
_UE8M0_RECIPE = {"data_type": "fp8_e4m3", "bits": 8, "group_size": 128,
                 "scale_fmt": "ue8m0"}
_CB_RECIPE = {"data_type": "nvfp4_cb", "cb_k": 16}


def _e8m0_plane(shape, generator) -> torch.Tensor:
    """A real F8_E8M0 exponent plane (the dtype DSv4 ships), not a uint8 stand-in.

    Built through a uint8 view because the exponent codes are what the format
    is; going via a float cast would round-trip through values E8M0 cannot
    represent and quietly change the bytes under test.
    """
    codes = torch.randint(110, 140, tuple(shape), generator=generator)
    return codes.to(torch.uint8).view(torch.float8_e8m0fnu)


def _dsv4_source_model(mdl: Path, *, mxfp4_layers=(1,), cb_layers=(0,),
                       dense_ue8m0=True, floor_fp8=False, seed=5) -> dict:
    """A DSv4-Flash-shaped checkpoint at 1/1000 scale, in its own namespace.

    Routed experts are nibble-packed MXFP4 (`.weight` I8 + `.scale` F8_E8M0)
    on EVERY layer — which is the real checkpoint's state — so the CB layer
    and the passthrough layer differ only in what the recipe asks for, not in
    what is on disk. The dense body Linear is the UE8M0 block-FP8 spelling
    (F8_E4M3 + F8_E8M0), the second member of the same passthrough family.
    """
    mdl.mkdir(parents=True, exist_ok=True)
    hid = _SOURCE_HID
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for layer in sorted({*mxfp4_layers, *cb_layers}):
        for expert in range(_SOURCE_EXPERTS):
            for leaf in ("w1", "w3", "w2"):
                base = f"layers.{layer}.ffn.experts.{expert}.{leaf}"
                tensors[base + ".weight"] = torch.randint(
                    -128, 128, (hid, hid // 2), dtype=torch.int8,
                    generator=generator)
                tensors[base + ".scale"] = _e8m0_plane(
                    (hid, hid // 32), generator)
    if dense_ue8m0:
        tensors["layers.0.attn.wq_a.weight"] = (
            torch.randn(hid, hid, generator=generator) * 0.3
        ).to(torch.float8_e4m3fn)
        tensors["layers.0.attn.wq_a.scale"] = _e8m0_plane((2, 2), generator)
    if floor_fp8:
        # THE wo_a SHAPE: a block-FP8 body Linear the recipe never mentions.
        # Identical on disk to the `dense_ue8m0` unit above — element plane
        # plus UE8M0 scale plane — and different only in that no allocation
        # target claims it, which is precisely the distinction that used to
        # cost it its scale plane and earn it an `ignore` entry.
        tensors["layers.0.attn.wo_a.weight"] = (
            torch.randn(hid, hid, generator=generator) * 0.3
        ).to(torch.float8_e4m3fn)
        tensors["layers.0.attn.wo_a.scale"] = _e8m0_plane((2, 2), generator)
    tensors["norm.weight"] = torch.ones(hid, dtype=torch.bfloat16)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": hid,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8", "fmt": "e4m3",
            "weight_block_size": [128, 128], "scale_fmt": "ue8m0",
        },
    }))
    return tensors


def _dsv4_recipe(*, mxfp4_layers=(1,), cb_layers=(0,), dense_ue8m0=True,
                 floor_fp8=False):
    """Expanded per-tensor assignment + col_weights, as the allocator writes it."""
    assignment: dict[str, object] = {}
    col_weights: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(11)
    for layers, recipe in ((cb_layers, _CB_RECIPE),
                           (mxfp4_layers, _MXFP4_RECIPE)):
        for layer in layers:
            for expert in range(_SOURCE_EXPERTS):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    qname = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                    assignment[qname] = recipe
                    if recipe is _CB_RECIPE:
                        col_weights[qname] = torch.rand(
                            _SOURCE_HID, generator=generator) + 0.05
    if dense_ue8m0:
        assignment["model.layers.0.self_attn.wq_a"] = _UE8M0_RECIPE
    # `floor_fp8` is deliberately NOT given an assignment entry: the whole
    # point of that fixture arm is a unit the recipe never mentions.
    return assignment, col_weights


def _source_bytes(mdl: Path, name: str) -> bytes:
    header, data0 = _st_header(mdl / "model.safetensors")
    lo, hi = header[name]["data_offsets"]
    raw = (mdl / "model.safetensors").read_bytes()
    return raw[data0 + lo:data0 + hi]


def _emitted_bytes(out: Path, name: str) -> bytes:
    return _source_bytes(out, name)


def _export_dsv4(
    workdir,
    out_name="out",
    *,
    target_profile: str | None = None,
    **model_kwargs,
):
    mdl = workdir / "src"
    tensors = _dsv4_source_model(mdl, **model_kwargs)
    assignment, col_weights = _dsv4_recipe(**model_kwargs)
    if target_profile is not None:
        assignment["__prismaquant__"] = {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": target_profile,
        }
    path = workdir / f"{out_name}.json"
    _assign(path, assignment)
    out = workdir / out_name
    counts = export_nvfp4_cb_streaming(
        mdl, path, out, col_weights, device="cpu",
        allow_route_pending_passthrough=True)
    # Every DSv4 artifact these tests produce goes through the completeness
    # gate, so a future change that drops a scale plane or mislabels a unit
    # fails in whichever test produced it rather than only in the one test
    # written to look for that.
    from prismaquant.artifact_completeness import assert_artifact_complete

    assert_artifact_complete(out)
    return mdl, out, tensors, dict(counts)


def test_source_passthrough_streams_checkpoint_bytes_verbatim(workdir):
    """The artifact's bytes ARE the checkpoint's bytes, per tensor, by digest."""
    mdl, out, tensors, counts = _export_dsv4(workdir)
    assert counts["MXFP4_SOURCE"] == _SOURCE_EXPERTS * 3
    assert counts["FP8_BLOCK_UE8M0_SOURCE"] == 1

    header = _assert_offsets_consistent(out / "model.safetensors")
    source_header, _ = _st_header(mdl / "model.safetensors")
    passthrough = [
        f"layers.1.ffn.experts.{expert}.{leaf}.{plane}"
        for expert in range(_SOURCE_EXPERTS)
        for leaf in ("w1", "w3", "w2")
        for plane in ("weight", "scale")
    ] + ["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"]
    for name in passthrough:
        # (a) the CHECKPOINT spelling, which is what the model's own loader
        #     resolves — not the live `model.layers.N.mlp.experts.*` name.
        assert name in header, name
        # (b) dtype and shape untouched: no widening, no repacking.
        assert header[name]["dtype"] == source_header[name]["dtype"], name
        assert header[name]["shape"] == source_header[name]["shape"], name
        # (c) the bytes themselves.
        assert hashlib.sha256(_emitted_bytes(out, name)).hexdigest() == \
            hashlib.sha256(_source_bytes(mdl, name)).hexdigest(), name

    # The dtypes are asserted positively too, so a checkpoint that happened to
    # ship F32 scales could not satisfy (b) vacuously.
    assert header["layers.1.ffn.experts.0.w1.weight"]["dtype"] == "I8"
    assert header["layers.1.ffn.experts.0.w1.scale"]["dtype"] == "F8_E8M0"
    assert header["layers.0.attn.wq_a.weight"]["dtype"] == "F8_E4M3"
    assert header["layers.0.attn.wq_a.scale"]["dtype"] == "F8_E8M0"


def test_generic_profile_keeps_published_source_materialization_compatible(
    workdir,
):
    """The new SM120 deny must not globally disable DSv4 source artifacts."""
    _mdl, _out, _tensors, counts = _export_dsv4(
        workdir,
        target_profile="nvfp4_cb",
    )
    assert counts["MXFP4_SOURCE"] == _SOURCE_EXPERTS * 3
    assert counts["FP8_BLOCK_UE8M0_SOURCE"] == 1


def test_ue8m0_block_scale_is_not_widened_like_fp8_source(workdir):
    """Regression guard on reusing the FP8_SOURCE branch for a UE8M0 source.

    FP8_SOURCE renames `.weight_scale_inv` -> `.weight_scale` and casts it to
    F32 because vLLM's stock block-FP8 path reads an fp32 plane. Doing that to
    a one-byte UE8M0 plane would quadruple its size and emit a tensor the
    checkpoint's own loader does not expect — and every decoded-value assertion
    in the suite would still pass.
    """
    _mdl, out, _tensors, _counts = _export_dsv4(workdir)
    header, _ = _st_header(out / "model.safetensors")
    assert "layers.0.attn.wq_a.weight_scale" not in header
    assert header["layers.0.attn.wq_a.scale"]["dtype"] == "F8_E8M0"
    lo, hi = header["layers.0.attn.wq_a.scale"]["data_offsets"]
    assert hi - lo == 2 * 2          # one byte per 128x128 block, not four


def test_source_passthrough_expert_group_is_not_collapsed_or_ignored(workdir):
    """A delegated group keeps its per-expert tensors and stays out of `ignore`.

    The packed parent gridbook's CB loader anchors on does not exist on disk
    for this route, so naming one would promise a stack that is never written;
    and `ignore` means "unquantized, load as-is", which these 4.25 bpw tensors
    are not.
    """
    _mdl, out, _tensors, _counts = _export_dsv4(workdir)
    emitted = set(load_file(str(out / "model.safetensors")))
    config = json.loads((out / "quant_config.json").read_text())

    assert "layers.1.ffn.experts.gate_up_proj.cb_qweight" not in emitted
    assert "layers.1.ffn.experts.down_proj.cb_qweight" not in emitted
    assert "layers.1.ffn.experts.0.w1.weight" in emitted
    # The CB layer is the control: there the collapse DID happen.
    assert "layers.0.ffn.experts.gate_up_proj.cb_qweight" in emitted
    assert not any(name.startswith("layers.0.ffn.experts.0.")
                   for name in emitted)

    ignored = set(config["ignore"])
    assert not any(name.startswith("layers.1.ffn.experts") for name in ignored)
    assert "layers.0.attn.wq_a" not in ignored

    native = [group for group in config["config_groups"].values()
              if group.get("source_format") == "MXFP4_SOURCE"]
    assert len(native) == 1
    assert native[0]["source_passthrough_id"] == "mxfp4_e2m1_ue8m0_g32"
    assert native[0]["weights"]["scale_dtype"] == "uint8_e8m0"
    # Routing is stated once, in the source_passthrough declaration.
    assert "route" not in native[0] and "route_backed" not in native[0]
    assert native[0]["input_activations"] is None
    assert len(native[0]["targets"]) == _SOURCE_EXPERTS * 3


def test_cb_expert_layer_beside_a_passthrough_layer_keeps_its_gates(workdir):
    """The CB half of a mixed artifact is unaffected by the passthrough half."""
    mdl, out, _tensors, counts = _export_dsv4(workdir)
    assert counts["NVFP4_CB_K16"] == 2          # gate_up + down, collapsed

    reference = workdir / "cb_only"
    assignment, col_weights = _dsv4_recipe(
        mxfp4_layers=(), cb_layers=(0,), dense_ue8m0=False)
    path = workdir / "cb_only.json"
    _assign(path, assignment)
    # The recipe drops `wq_a`, but the SOURCE still has it — so in this export
    # it is a floor block-FP8 unit and needs the same acknowledgement any
    # route-pending passthrough does. Before floor units were declared, this
    # call silently produced a reference artifact with `wq_a`'s scale plane
    # dropped and the unit listed in `ignore`; the comparison below still
    # passed because it only reads the CB tensors.
    export_nvfp4_cb_streaming(mdl, path, reference, col_weights, device="cpu",
                              allow_route_pending_passthrough=True)

    mixed_tensors = load_file(str(out / "model.safetensors"))
    cb_only_tensors = load_file(str(reference / "model.safetensors"))
    for leaf in ("gate_up_proj", "down_proj"):
        name = f"layers.0.ffn.experts.{leaf}.cb_qweight"
        assert torch.equal(mixed_tensors[name], cb_only_tensors[name]), name

    config = json.loads((out / "quant_config.json").read_text())
    cb_groups = [group for group in config["config_groups"].values()
                 if "scheme" in group]
    assert len(cb_groups) == 1
    assert cb_groups[0]["targets"] == [
        "layers.0.ffn.experts.down_proj", "layers.0.ffn.experts.gate_up_proj"]


def _pending_formats_in(counts) -> set[str]:
    """Route-pending formats this export actually carries, per the contract table.

    Read at call time, never pinned: the table is measurement-backed and its
    verdicts move (2026-08-03 inverted both of them). A test that hardcodes
    which format is pending stops testing the gate and starts testing the
    measurement.
    """
    from prismaquant.allocator_candidates import (
        ROUTE_PENDING_PASSTHROUGH_FORMATS,
    )

    return {fmt for fmt in counts if fmt in ROUTE_PENDING_PASSTHROUGH_FORMATS}


def test_route_pending_ship_gate_follows_the_contract_table(workdir):
    """Whatever the table currently marks pending is what must be refused.

    Both branches assert something real, so this stays honest whichever way the
    measured verdicts land.
    """
    mdl = workdir / "src"
    _dsv4_source_model(mdl)
    assignment, col_weights = _dsv4_recipe()
    path = workdir / "a.json"
    _assign(path, assignment)

    shipped = export_nvfp4_cb_streaming(
        mdl, path, workdir / "shipped", col_weights, device="cpu",
        allow_route_pending_passthrough=True)
    pending = _pending_formats_in(shipped)
    config = json.loads(
        (workdir / "shipped" / "quant_config.json").read_text())

    out = workdir / "default"
    if not pending:
        # Nothing pending: the override must not be needed at all.
        export_nvfp4_cb_streaming(mdl, path, out, col_weights, device="cpu")
        assert "route_pending_passthrough_acknowledged" not in \
            config["provenance"]
        return
    with pytest.raises(ValueError) as error:
        export_nvfp4_cb_streaming(mdl, path, out, col_weights, device="cpu")
    message = str(error.value)
    for fmt in pending:
        assert fmt in message
    assert "--allow-route-pending-passthrough" in message
    assert not out.exists() or not list(out.iterdir())
    # The override was one flag on one machine; the artifact carries the fact.
    assert config["provenance"][
        "route_pending_passthrough_acknowledged"] == sorted(pending)


def test_route_pending_ship_gate_refuses_a_format_the_table_marks_pending(
    workdir, monkeypatch,
):
    """The gate itself, pinned independently of today's measured verdicts."""
    from prismaquant import export_nvfp4_cb_streaming as streaming_module
    from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_CONTRACTS

    monkeypatch.setattr(
        streaming_module, "ROUTE_PENDING_PASSTHROUGH_FORMATS",
        frozenset({"MXFP4_SOURCE"}))
    mdl = workdir / "src"
    _dsv4_source_model(mdl, dense_ue8m0=False)
    assignment, col_weights = _dsv4_recipe(dense_ue8m0=False)
    path = workdir / "a.json"
    _assign(path, assignment)
    with pytest.raises(ValueError) as error:
        export_nvfp4_cb_streaming(
            mdl, path, workdir / "refused", col_weights, device="cpu")
    message = str(error.value)
    assert "MXFP4_SOURCE" in message
    assert SOURCE_PASSTHROUGH_CONTRACTS[
        "MXFP4_SOURCE"].serving_route in message
    assert f"{_SOURCE_EXPERTS * 3} unit(s)" in message
    export_nvfp4_cb_streaming(
        mdl, path, workdir / "shipped", col_weights, device="cpu",
        allow_route_pending_passthrough=True)


# --- source_passthrough: the cross-repo declaration -------------------------

def test_source_passthrough_declaration_names_every_delegated_unit(workdir):
    from prismaquant.cb_export_config import (
        SOURCE_PASSTHROUGH_DECLARATION_KEY,
        SOURCE_PASSTHROUGH_DECLARATION_VERSION,
        parse_source_passthrough_declaration,
    )

    _mdl, out, _tensors, _counts = _export_dsv4(workdir)
    config = json.loads((out / "quant_config.json").read_text())
    record = config[SOURCE_PASSTHROUGH_DECLARATION_KEY]
    assert record == {
        "version": SOURCE_PASSTHROUGH_DECLARATION_VERSION,
        "units": {
            "model.layers.0.self_attn.wq_a": "fp8_e4m3_ue8m0_block128",
            "model.layers.1.mlp.experts": "mxfp4_e2m1_ue8m0_g32",
        },
    }
    # The declaration is TOP-LEVEL, not an execution contract.
    assert SOURCE_PASSTHROUGH_DECLARATION_KEY not in config.get(
        "execution_contracts", {})
    assert parse_source_passthrough_declaration(config) == record["units"]

    # A routed-expert group is ONE unit even though it emits 12 tensors under
    # per-expert names, and a dense body Linear is a legal unit too.
    assert "model.layers.1.mlp.experts.0.gate_proj" not in record["units"]
    # The CB layer is not in the declaration at all.
    assert "model.layers.0.mlp.experts" not in record["units"]


def test_source_passthrough_key_is_absent_from_an_all_cb_artifact(workdir):
    """Absence of the key IS the legacy/all-CB signal — never an empty record."""
    from prismaquant.cb_export_config import (
        SOURCE_PASSTHROUGH_DECLARATION_KEY,
        parse_source_passthrough_declaration,
    )

    mdl = workdir / "src"
    _dsv4_source_model(mdl, mxfp4_layers=(), cb_layers=(0,),
                       dense_ue8m0=False)
    assignment, col_weights = _dsv4_recipe(
        mxfp4_layers=(), cb_layers=(0,), dense_ue8m0=False)
    path = workdir / "a.json"
    _assign(path, assignment)
    out = workdir / "cb_only"
    export_nvfp4_cb_streaming(mdl, path, out, col_weights, device="cpu")
    config = json.loads((out / "quant_config.json").read_text())
    assert SOURCE_PASSTHROUGH_DECLARATION_KEY not in config
    assert parse_source_passthrough_declaration(config) is None


def test_source_passthrough_wire_ids_are_a_closed_enum_with_no_silent_gaps():
    """A passthrough format with no wire id must stop the export.

    A unit silently dropped from ``units`` reads to the consumer as "this is
    CB" — the one wrong answer that loads.
    """
    from prismaquant.cb_export_config import (
        DELEGATED_NATIVE_PASSTHROUGH_FORMATS,
        SOURCE_PASSTHROUGH_WIRE_FORMATS,
        SOURCE_PASSTHROUGH_WIRE_IDS,
        source_passthrough_wire_id,
    )

    assert SOURCE_PASSTHROUGH_WIRE_IDS["MXFP4_SOURCE"] == \
        "mxfp4_e2m1_ue8m0_g32"
    assert SOURCE_PASSTHROUGH_WIRE_IDS["FP8_BLOCK_UE8M0_SOURCE"] == \
        "fp8_e4m3_ue8m0_block128"
    assert len(SOURCE_PASSTHROUGH_WIRE_FORMATS) == len(
        SOURCE_PASSTHROUGH_WIRE_IDS)
    # Every format that can reach the byte-verbatim emitter has an id.
    missing = sorted(DELEGATED_NATIVE_PASSTHROUGH_FORMATS
                     - set(SOURCE_PASSTHROUGH_WIRE_IDS))
    assert missing == [], missing
    # FP8_SOURCE is CT-normalized: it never enters this record, and asking for
    # its wire id is a bug rather than a lookup miss to paper over.
    with pytest.raises(ValueError, match="no wire id"):
        source_passthrough_wire_id("FP8_SOURCE")


def test_parse_source_passthrough_declaration_refuses_the_load_failure_cases():
    from prismaquant.cb_export_config import (
        SOURCE_PASSTHROUGH_DECLARATION_KEY as KEY,
        parse_source_passthrough_declaration as parse,
    )

    def config(record):
        return {"config_groups": {}, KEY: record}

    good = {"version": 1,
            "units": {"model.layers.1.mlp.experts": "mxfp4_e2m1_ue8m0_g32"}}
    assert parse(config(good)) == good["units"]

    with pytest.raises(ValueError, match="unsupported .* version"):
        parse(config({**good, "version": 2}))
    with pytest.raises(ValueError, match="must be an object"):
        parse(config(["units"]))
    with pytest.raises(ValueError, match="carries no units"):
        parse(config({"version": 1, "units": {}}))
    with pytest.raises(ValueError, match="string unit ids"):
        parse(config({"version": 1, "units": {"a": 4}}))
    with pytest.raises(ValueError, match="unknown format id"):
        parse(config({"version": 1, "units": {"a": "mxfp4_source"}}))
    # A unit claimed by BOTH gridbook's codec and the passthrough declaration.
    contested = {
        "config_groups": {
            "group_0": {
                "format": "NVFP4_CB_K16",
                "scheme": {"grid": "fp4"},
                "targets": ["model.layers.1.mlp.experts"],
            },
        },
        KEY: good,
    }
    with pytest.raises(ValueError, match="claimed by BOTH"):
        parse(contested)


_RECONCILED = {
    "cb_units": {"model.layers.0.mlp.experts"},
    "passthrough_units": {"model.layers.1.mlp.experts"},
    "cb_tensors": {"layers.0.ffn.experts.gate_up_proj.cb_qweight"},
    "passthrough_tensors": {"layers.1.ffn.experts.0.w1.weight"},
    "cb_modules": {"layers.0.ffn.experts"},
    "passthrough_modules": {"layers.1.ffn.experts"},
    "attested": {"layers.0.ffn.experts"},
}


def test_route_reconciliation_refuses_every_way_the_two_scopes_can_overlap():
    """The producer-side invariant, as a predicate over the four set pairs.

    Each case below is a state the artifact must never reach, and each would be
    invisible in the emitted JSON — which is the point: the wire record no
    longer carries ``cb_activation_contract``, so this is where the claim that
    a delegated group's K0.2 absence is DELIBERATE actually gets proved.
    """
    from prismaquant.export_nvfp4_cb_streaming import assert_routes_reconcile

    assert_routes_reconcile(**_RECONCILED)          # the reconciled state

    for field, mutation, message in (
        # One unit handed to two loaders.
        ("passthrough_units", {"model.layers.0.mlp.experts"},
         "claimed by BOTH"),
        # Unit ids disjoint but a tensor emitted twice (a namespace bug).
        ("passthrough_tensors", {"layers.0.ffn.experts.gate_up_proj.cb_qweight"},
         "emitted by both"),
        # A delegated group that somehow reached the K0.2 attestation.
        ("attested", {"layers.1.ffn.experts"}, "AND declared"),
        # An attested module no CB unit claims — a dropped/renamed group.
        ("attested", {"layers.9.ffn.experts"}, "which no CB expert unit claims"),
    ):
        state = dict(_RECONCILED)
        state[field] = mutation
        with pytest.raises(AssertionError, match=message):
            assert_routes_reconcile(**state)

    # A routed-expert group left entirely on BF16 is on neither route and is
    # correctly absent from both module sets.
    assert_routes_reconcile(**{**_RECONCILED, "attested": set()})


def test_producer_refuses_a_k02_attestation_of_a_delegated_group(
    workdir, monkeypatch,
):
    """A delegated group must never appear in the K0.2 attestation.

    This is what ``cb_activation_contract`` bought on the wire; the field is
    gone but the invariant it protected is not. A delegated group's ABSENCE
    from the K0.2 record is a DECLARATION, not a dropped attestation — which
    only holds while the two scopes are provably disjoint.
    """
    from prismaquant import export_nvfp4_cb_streaming as streaming_module

    mdl = workdir / "src"
    _dsv4_source_model(mdl, dense_ue8m0=False)
    assignment, col_weights = _dsv4_recipe(dense_ue8m0=False)
    path = workdir / "a.json"
    _assign(path, assignment)
    # The delegated group's SERIALIZED module prefix — what K0.2 would name it.
    monkeypatch.setattr(
        streaming_module, "routed_moe_attested_module_names",
        lambda record: ("layers.1.ffn.experts",))
    with pytest.raises(AssertionError, match="AND declared"):
        export_nvfp4_cb_streaming(
            mdl, path, workdir / "conflict", col_weights, device="cpu",
            allow_route_pending_passthrough=True)

    # ... and an attested module that no CB unit claims is equally refused.
    monkeypatch.setattr(
        streaming_module, "routed_moe_attested_module_names",
        lambda record: ("layers.9.ffn.experts",))
    with pytest.raises(AssertionError, match="which no CB expert unit claims"):
        export_nvfp4_cb_streaming(
            mdl, path, workdir / "orphan", col_weights, device="cpu",
            allow_route_pending_passthrough=True)


def test_declared_units_and_cb_targets_are_disjoint_in_the_artifact(workdir):
    """The shipped artifact never lets one unit be read by two loaders."""
    _mdl, out, _tensors, _counts = _export_dsv4(workdir)
    config = json.loads((out / "quant_config.json").read_text())
    declared = set(config["source_passthrough"]["units"])
    cb_targets = {
        target
        for group in config["config_groups"].values() if "scheme" in group
        for target in group["targets"]
    }
    assert declared and cb_targets
    assert not any(
        target == unit or target.startswith(unit + ".")
        for unit in declared for target in cb_targets)


def test_k02_scope_reader_reads_the_attested_modules():
    """The producer-side scope reader, on a real routed-MoE record."""
    from prismaquant.nvfp4_activation_contract import (
        CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
        CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        NVFP4_ROUTED_MOE_STAGE_KEY,
        build_execution_contract,
        routed_moe_attested_module_names,
    )

    module = "model.layers.0.mlp.experts"
    record, _scales = build_execution_contract(
        {f"{module}.gate_up_proj": 4.0, f"{module}.down_proj": 8.0},
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources={
            f"{module}.gate_up_proj":
                CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
            f"{module}.down_proj":
                CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        },
    )
    assert routed_moe_attested_module_names(record) == (module,)
    assert len(routed_moe_attested_module_names(record)) == record[
        NVFP4_ROUTED_MOE_STAGE_KEY]["module_count"]
    # Absent / dense-only records attest nothing, and say so without raising.
    assert routed_moe_attested_module_names(None) == ()
    assert routed_moe_attested_module_names({"schema": "x"}) == ()


def test_stream_writer_refuses_a_tensor_planned_by_two_emit_paths():
    """A duplicate header name keeps one span while both blobs are written."""
    writer = _StreamWriter()
    writer.add("a.weight", torch.uint8, (4,), lambda: torch.zeros(4).to(
        torch.uint8))
    writer.add("a.weight", torch.uint8, (8,), lambda: torch.zeros(8).to(
        torch.uint8))
    with pytest.raises(AssertionError, match="planned twice"):
        writer.write(Path("unused-never-created.safetensors"))


def test_source_passthrough_recipes_round_trip_through_canonicalize_format():
    """The recipe spelling the registry emits is the one the exporter reads.

    Both new carriers sit one field away from a format that already existed —
    MXFP4_SOURCE beside MXFP4, FP8_BLOCK_UE8M0_SOURCE beside FP8_SOURCE — and
    collapsing either pair would silently ship the wrong on-disk contract.
    """
    from prismaquant import format_registry as fr
    from prismaquant.layer_config import canonicalize_format

    for name in ("MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE", "FP8_SOURCE",
                 "MXFP4", "MXFP8_E4M3", "FP8_E4M3"):
        assert canonicalize_format(
            fr.get_format(name).autoround_config()) == name, name
    # The neighbours stay distinct on the ONE field that separates them.
    assert canonicalize_format(
        {"data_type": "fp8_e4m3", "bits": 8, "group_size": 128}) == "FP8_SOURCE"
    assert canonicalize_format({
        "data_type": "fp8_e4m3", "bits": 8, "group_size": 128,
        "scale_fmt": "ue8m0"}) == "FP8_BLOCK_UE8M0_SOURCE"
    assert canonicalize_format({"data_type": "mx_fp", "bits": 4}) == "MXFP4"
    # MXFP4_SOURCE is the OCP-MX group-of-32 claim; another group is another
    # contract, not a variant.
    with pytest.raises(ValueError, match="group-of-32"):
        canonicalize_format(
            {"data_type": "fp4_e2m1", "bits": 4, "group_size": 16})


# ---------------------------------------------------------------------------
# MXFP8_UE8M0_G32 — re-quantized native emission (the third streaming lane)
# ---------------------------------------------------------------------------

_MXFP8_RECIPE = {"data_type": "fp8_e4m3", "bits": 8, "group_size": 32,
                 "scale_fmt": "ue8m0"}


def _export_dsv4_with_mxfp8_body(workdir, out_name="mxfp8"):
    """DSv4 source whose dense body Linear is RE-ENCODED to MXFP8, not copied.

    Same checkpoint as the passthrough tests — the body Linear really is
    block-FP8 on disk — so this exercises the interesting case: a re-quant
    rung reading a block-scaled FP8 source.
    """
    mdl = workdir / "src_mxfp8"
    tensors = _dsv4_source_model(mdl)
    assignment, col_weights = _dsv4_recipe()
    assignment["model.layers.0.self_attn.wq_a"] = _MXFP8_RECIPE
    path = workdir / f"{out_name}.json"
    _assign(path, assignment)
    out = workdir / out_name
    counts = export_nvfp4_cb_streaming(
        mdl, path, out, col_weights, device="cpu",
        allow_route_pending_passthrough=True)
    return mdl, out, tensors, dict(counts)


def test_mxfp8_requant_emits_the_declared_weight_and_scale_pair(workdir):
    """The planned header is the emitted header: names, dtypes, byte counts."""
    _mdl, out, _tensors, counts = _export_dsv4_with_mxfp8_body(workdir)
    assert counts["MXFP8_UE8M0_G32"] == 1

    header, _ = _st_header(out / "model.safetensors")
    base = "layers.0.attn.wq_a"
    assert header[base + ".weight"]["dtype"] == "F8_E4M3"
    assert header[base + ".weight"]["shape"] == [_SOURCE_HID, _SOURCE_HID]
    assert header[base + ".weight_scale"]["dtype"] == "F8_E8M0"
    assert header[base + ".weight_scale"]["shape"] == [
        _SOURCE_HID, _SOURCE_HID // 32]
    # The source's own one-byte block-scale plane is GONE: this unit was
    # re-encoded, so shipping the checkpoint's scale beside it would describe
    # a tensor the new elements are not scaled by.
    assert base + ".scale" not in header
    lo, hi = header[base + ".weight_scale"]["data_offsets"]
    assert hi - lo == _SOURCE_HID * (_SOURCE_HID // 32)   # one byte per group

    # 8 + 8/32 = 8.25 bpw, on the bytes actually written.
    wlo, whi = header[base + ".weight"]["data_offsets"]
    n_params = _SOURCE_HID * _SOURCE_HID
    assert 8.0 * ((whi - wlo) + (hi - lo)) / n_params == 8.25


def test_mxfp8_requant_of_a_block_fp8_body_is_bit_exact_end_to_end(workdir):
    """The exactness property, all the way through the shipped artifact.

    The source Linear is E4M3 codes times a per-128x128 power of two. Decoding
    the emitted MXFP8 planes must return those values EXACTLY — not close.
    """
    mdl, out, tensors, _counts = _export_dsv4_with_mxfp8_body(workdir)
    base = "layers.0.attn.wq_a"

    source = (
        tensors[base + ".weight"].float()
        * tensors[base + ".scale"].float()
        .repeat_interleave(128, 0).repeat_interleave(128, 1)
    )

    emitted = load_file(str(out / "model.safetensors"))
    weight = emitted[base + ".weight"].float()
    scale = emitted[base + ".weight_scale"].float()
    decoded = (
        weight.reshape(_SOURCE_HID, _SOURCE_HID // 32, 32)
        * scale.unsqueeze(-1)
    ).reshape(_SOURCE_HID, _SOURCE_HID)

    assert torch.equal(decoded, source)
    assert float(((decoded - source) ** 2).mean()) == 0.0


def test_mxfp8_requant_declares_a_native_group_and_routes_natively(workdir):
    """It carries its own wire id and joins the delegated-native routing map.

    The consumer's dispatcher reads ONE map (`source_passthrough.units`) to
    decide "native route or CB decoder" and refuses an id it does not know, so
    a unit omitted there would be read as CB — the one wrong answer that
    loads. The byte-verbatim-vs-re-encoded distinction lives per unit in the
    config group's `weights.source_passthrough` flag instead.
    """
    _mdl, out, _tensors, _counts = _export_dsv4_with_mxfp8_body(workdir)
    config = json.loads((out / "quant_config.json").read_text())

    groups = [g for g in config["config_groups"].values()
              if g.get("source_format") == "MXFP8_UE8M0_G32"]
    assert len(groups) == 1
    group = groups[0]
    assert group["format"] == "gridbook-native"
    assert group["wire_format_id"] == "mxfp8_e4m3_e8m0_g32"
    assert group["weights"]["group_size"] == 32
    assert group["weights"]["scale_dtype"] == "uint8_e8m0"
    # The producer WROTE these bytes -- the one claim it must not borrow from
    # its byte-verbatim neighbours.
    assert group["weights"]["source_passthrough"] is False
    # W8A8: the lane quantizes activations to the same per-32 grid.
    acts = group["input_activations"]
    assert acts["num_bits"] == 8 and acts["group_size"] == 32
    assert acts["dynamic"] is True
    from prismaquant.export_native_compressed import _explicit_regex
    assert group["targets"] == [_explicit_regex("layers.0.attn.wq_a")]

    declared = config["source_passthrough"]["units"]
    assert config["source_passthrough"]["version"] == 1
    # Unit ids stay in the RECIPE namespace (as the byte-verbatim lane's do);
    # only the emitted TENSORS take the checkpoint spelling.
    assert declared["model.layers.0.self_attn.wq_a"] == "mxfp8_e4m3_e8m0_g32"
    # The routed-expert byte-verbatim half of the same artifact is unaffected.
    assert any(fmt == "mxfp4_e2m1_ue8m0_g32" for fmt in declared.values())
    assert config["provenance"]["requant_native_targets"] == {
        "MXFP8_UE8M0_G32": 1}
    assert "layers.0.attn.wq_a" not in set(config["ignore"])


def test_mxfp8_requant_declaration_parses_on_the_producer_side(workdir):
    """The producer self-parses what it writes, as it does for passthroughs."""
    from prismaquant.cb_export_config import (
        parse_source_passthrough_declaration,
    )

    _mdl, out, _tensors, _counts = _export_dsv4_with_mxfp8_body(workdir)
    config = json.loads((out / "quant_config.json").read_text())
    parsed = parse_source_passthrough_declaration(config)
    assert parsed["model.layers.0.self_attn.wq_a"] == "mxfp8_e4m3_e8m0_g32"


# --- floor block-FP8: units the recipe never allocated ----------------------
#
# The bug these pin, stated once: a block-FP8 weight that no allocation target
# claims used to reach the verbatim copy loop, lose its `.scale` sibling to a
# skip meant for scales the CB/source lanes had already consumed, and be
# declared `ignore` — i.e. "plain unquantized floats". A consumer honouring
# that casts fp8 to bf16 with no scale, passes every size check, and serves
# weights each off by their own power of two. On DSv4-Flash: 43 `attn.wo_a` +
# 21 `attn.indexer.wq_b` units, 1.44 GB, no error raised anywhere.


def test_floor_block_fp8_ships_its_scale_and_is_declared_not_ignored(workdir):
    """Weight AND scale present, declared passthrough, absent from `ignore`."""
    mdl, out, tensors, counts = _export_dsv4(workdir, floor_fp8=True)

    header = _assert_offsets_consistent(out / "model.safetensors")
    quant_config = json.loads((out / "quant_config.json").read_text())

    unit = "layers.0.attn.wo_a"
    # (a) BOTH planes ship, byte-for-byte. A weight without its scale is not a
    #     smaller artifact, it is an unreadable one.
    for plane in ("weight", "scale"):
        name = f"{unit}.{plane}"
        assert name in header, f"{name} missing from the artifact"
        assert header[name]["dtype"] == \
            _st_header(mdl / "model.safetensors")[0][name]["dtype"], name
        assert hashlib.sha256(_emitted_bytes(out, name)).hexdigest() == \
            hashlib.sha256(_source_bytes(mdl, name)).hexdigest(), name

    # (b) declared under the wire id the consumer routes on.
    declared = (quant_config.get("source_passthrough") or {}).get("units") or {}
    assert declared.get(unit) == "fp8_e4m3_ue8m0_block128", declared

    # (c) and NOT claimed to be unquantized. This is the assertion that would
    #     have caught the original bug on its own.
    assert unit not in (quant_config.get("ignore") or []), \
        "a declared block-FP8 unit must not also be listed in `ignore`"

    assert counts["floor_fp8_declared"] == 1


def test_floor_block_fp8_declaration_survives_the_completeness_gate(workdir):
    from prismaquant.artifact_completeness import assert_artifact_complete

    _, out, _, _ = _export_dsv4(workdir, floor_fp8=True)
    report = assert_artifact_complete(out)
    assert "layers.0.attn.wo_a" in report.passthrough_units
    # The exact Gridbook W8A16 route is backed, so no override record belongs
    # in a newly exported artifact.
    assert report.route_pending_acknowledged == []


def test_completeness_gate_catches_the_original_silent_corruption(workdir):
    """The regression pin: reproduce the OLD artifact shape, demand a failure.

    Built by editing a healthy artifact back into the broken state (drop the
    scale plane, move the unit from the declaration into `ignore`) rather than
    by reverting the exporter, so the gate is tested against the artifact the
    bug actually produced.
    """
    from prismaquant.artifact_completeness import (
        ArtifactIncomplete,
        check_artifact_completeness,
    )

    _, out, _, _ = _export_dsv4(workdir, floor_fp8=True)
    unit = "layers.0.attn.wo_a"

    quant_config = json.loads((out / "quant_config.json").read_text())
    quant_config["source_passthrough"]["units"].pop(unit)
    quant_config.setdefault("ignore", []).append(unit)
    (out / "quant_config.json").write_text(json.dumps(quant_config))

    # Drop the scale plane exactly as the old skip did.
    tensors = load_file(str(out / "model.safetensors"))
    tensors.pop(f"{unit}.scale")
    save_file(tensors, str(out / "model.safetensors"))

    report = check_artifact_completeness(out)
    assert not report.ok
    assert unit in report.fp8_in_ignore
    with pytest.raises(ArtifactIncomplete, match="ignore"):
        from prismaquant.artifact_completeness import assert_artifact_complete
        assert_artifact_complete(out)


def test_completeness_gate_catches_a_declared_unit_with_no_scale(workdir):
    """Declaring a decode the artifact cannot perform is its own failure."""
    from prismaquant.artifact_completeness import check_artifact_completeness

    _, out, _, _ = _export_dsv4(workdir, floor_fp8=True)
    tensors = load_file(str(out / "model.safetensors"))
    tensors.pop("layers.0.attn.wo_a.scale")
    save_file(tensors, str(out / "model.safetensors"))

    report = check_artifact_completeness(out)
    assert not report.ok
    assert "layers.0.attn.wo_a" in report.missing_scale


def test_completeness_gate_passes_a_healthy_artifact_without_floor_units(
        workdir):
    """No false positives on the shape the exporter already produced."""
    from prismaquant.artifact_completeness import assert_artifact_complete

    _, out, _, _ = _export_dsv4(workdir)
    assert_artifact_complete(out)


def test_floor_block_fp8_scale_bytes_are_a_negligible_budget_delta(workdir):
    """Item 4: declaring the floor costs the scale planes and nothing else.

    Block-128 means one scale byte per 16,384 weight elements, so the fix
    cannot plausibly move a byte budget — but "cannot plausibly" is exactly
    the kind of claim that should be measured against the artifact rather
    than argued, since it is what stands between the fix and the 1.06 MB of
    headroom tonight's 92 GB selection actually has.
    """
    _, out_declared, _, _ = _export_dsv4(
        workdir, out_name="declared", floor_fp8=True)
    header = _assert_offsets_consistent(out_declared / "model.safetensors")

    weight = header["layers.0.attn.wo_a.weight"]
    scale = header["layers.0.attn.wo_a.scale"]

    def _nbytes(meta):
        lo, hi = meta["data_offsets"]
        return hi - lo

    weight_bytes, scale_bytes = _nbytes(weight), _nbytes(scale)
    elements = 1
    for dim in weight["shape"]:
        elements *= dim
    # One E8M0 byte per 128x128 block.
    assert scale_bytes * (128 * 128) == elements
    assert scale_bytes / weight_bytes < 1e-3, (
        f"scale plane is {scale_bytes}B against a {weight_bytes}B weight; "
        f"block-128 should make it ~1/16384 of the element plane")


def test_route_pending_ack_env_defaults_off_and_needs_exactly_one(monkeypatch):
    """The acknowledgement must never be ambient."""
    from prismaquant.export_nvfp4_cb_streaming import (
        _ROUTE_PENDING_ACK_ENV,
        _route_pending_ack_from_env,
    )

    monkeypatch.delenv(_ROUTE_PENDING_ACK_ENV, raising=False)
    assert _route_pending_ack_from_env() is False
    for value in ("0", "", "true", "yes", "TRUE", "2"):
        monkeypatch.setenv(_ROUTE_PENDING_ACK_ENV, value)
        assert _route_pending_ack_from_env() is False, value
    monkeypatch.setenv(_ROUTE_PENDING_ACK_ENV, "1")
    assert _route_pending_ack_from_env() is True


def test_backed_floor_block_fp8_ships_without_route_override(workdir):
    """The released W8A16 route needs no stale route-pending acknowledgement."""
    mdl = workdir / "src"
    _dsv4_source_model(mdl, floor_fp8=True)
    assignment, col_weights = _dsv4_recipe(floor_fp8=True)
    path = workdir / "recipe.json"
    _assign(path, assignment)

    out = workdir / "backed"
    export_nvfp4_cb_streaming(
        mdl, path, out, col_weights, device="cpu",
        allow_route_pending_passthrough=False)
    quant = json.loads((out / "quant_config.json").read_text())
    assert "route_pending_passthrough_acknowledged" not in quant["provenance"]


# --- namespace exclusion: omitting a floor namespace entirely ---------------


def _dsv4_source_with_mtp(mdl: Path, **kwargs) -> dict:
    """The DSv4 fixture plus an `mtp.*` block, so exclusion has a target."""
    tensors = _dsv4_source_model(mdl, **kwargs)
    generator = torch.Generator().manual_seed(77)
    hid = _SOURCE_HID
    extra = {
        "mtp.0.attn.wq_a.weight": (
            torch.randn(hid, hid, generator=generator) * 0.3
        ).to(torch.float8_e4m3fn),
        "mtp.0.attn.wq_a.scale": _e8m0_plane((2, 2), generator),
        "mtp.0.attn_norm.weight": torch.ones(hid, dtype=torch.bfloat16),
    }
    for expert in range(_SOURCE_EXPERTS):
        for leaf in ("w1", "w3", "w2"):
            base = f"mtp.0.ffn.experts.{expert}.{leaf}"
            extra[base + ".weight"] = torch.randint(
                -128, 128, (hid, hid // 2), dtype=torch.int8,
                generator=generator)
            extra[base + ".scale"] = _e8m0_plane((hid, hid // 32), generator)
    tensors.update(extra)
    save_file(tensors, str(mdl / "model.safetensors"))
    return tensors


def _export_with_exclusions(workdir, out_name, exclusions, **model_kwargs):
    mdl = workdir / f"src_{out_name}"
    _dsv4_source_with_mtp(mdl, **model_kwargs)
    assignment, col_weights = _dsv4_recipe(**model_kwargs)
    path = workdir / f"{out_name}.json"
    _assign(path, assignment)
    out = workdir / out_name
    counts = export_nvfp4_cb_streaming(
        mdl, path, out, col_weights, device="cpu",
        allow_route_pending_passthrough=True,
        exclude_namespaces=exclusions)
    return mdl, out, dict(counts)


def test_namespace_exclusion_removes_it_everywhere_and_changes_nothing_else(
        workdir):
    """`mtp.*` gone from tensors, index and config; the rest byte-identical."""
    _mdl, kept, _ = _export_with_exclusions(workdir, "kept", None)
    mdl, dropped, counts = _export_with_exclusions(
        workdir, "dropped", ["mtp."])

    kept_header, _ = _st_header(kept / "model.safetensors")
    dropped_header, _ = _st_header(dropped / "model.safetensors")

    # (a) gone from the tensor set / index entirely.
    assert any(n.startswith("mtp.") for n in kept_header)
    assert not any(n.startswith("mtp.") for n in dropped_header)

    # (b) gone from every declaration surface too, not just the payload.
    dropped_config = json.loads((dropped / "quant_config.json").read_text())
    serialized = json.dumps(dropped_config)
    assert "mtp." not in serialized.replace('"excluded_namespaces": ["mtp."]',
                                            ""), \
        "an excluded namespace must not survive in ignore/targets/units"
    assert dropped_config["provenance"]["excluded_namespaces"] == ["mtp."]

    # (c) the byte total drops by exactly the excluded namespace's size.
    def _payload(header):
        return sum(meta["data_offsets"][1] - meta["data_offsets"][0]
                   for name, meta in header.items() if name != "__metadata__")

    excluded_bytes = sum(
        meta["data_offsets"][1] - meta["data_offsets"][0]
        for name, meta in kept_header.items()
        if name.startswith("mtp."))
    assert excluded_bytes > 0
    assert _payload(kept_header) - _payload(dropped_header) == excluded_bytes

    # (d) EVERYTHING else is byte-identical — exclusion removes, never rewrites.
    shared = {n for n in kept_header
              if not n.startswith("mtp.") and n != "__metadata__"}
    assert shared == {n for n in dropped_header if n != "__metadata__"}
    for name in sorted(shared):
        assert hashlib.sha256(_emitted_bytes(kept, name)).hexdigest() == \
            hashlib.sha256(_emitted_bytes(dropped, name)).hexdigest(), name


def test_namespace_exclusion_defaults_to_nothing(workdir, monkeypatch):
    """Unset env and unset argument both mean 'exclude nothing'."""
    from prismaquant.export_nvfp4_cb_streaming import (
        _EXCLUDE_NAMESPACES_ENV,
        _exclude_namespaces_from_env,
    )

    monkeypatch.delenv(_EXCLUDE_NAMESPACES_ENV, raising=False)
    assert _exclude_namespaces_from_env() == ()
    for blank in ("", "   ", ",", " , "):
        monkeypatch.setenv(_EXCLUDE_NAMESPACES_ENV, blank)
        assert _exclude_namespaces_from_env() == ()
    monkeypatch.setenv(_EXCLUDE_NAMESPACES_ENV, " mtp. , visual. ")
    assert _exclude_namespaces_from_env() == ("mtp.", "visual.")

    # Back to unset before exporting: with no argument AND no env, the export
    # must keep everything. (Leaving the variable set above would exclude
    # `mtp.` through the env path — which is the knob working, but not what
    # this test is about.)
    monkeypatch.delenv(_EXCLUDE_NAMESPACES_ENV, raising=False)
    _mdl, out, _ = _export_with_exclusions(workdir, "default", None)
    header, _ = _st_header(out / "model.safetensors")
    assert any(name.startswith("mtp.") for name in header)
    config = json.loads((out / "quant_config.json").read_text())
    assert "excluded_namespaces" not in config["provenance"]


def test_namespace_exclusion_refuses_an_allocated_unit(workdir):
    """Excluding something the recipe priced must hard-fail, not silently drop.

    The allocated unit's bytes are already counted in the selection's achieved
    bits and predicted loss, so omitting it would make the artifact contradict
    the recipe that justifies it.
    """
    with pytest.raises(ValueError, match="allocates"):
        _export_with_exclusions(workdir, "illegal", ["layers.0."])


def test_namespace_exclusion_refuses_via_the_recipe_spelling_too(workdir):
    """A prefix written in the RECIPE vintage must collide just the same."""
    with pytest.raises(ValueError, match="allocates"):
        _export_with_exclusions(workdir, "illegal2", ["model.layers.0."])


def test_excluded_namespace_absence_is_valid_to_the_completeness_gate(workdir):
    """Absent-and-excluded passes; the same absence unrecorded still fails."""
    from prismaquant.artifact_completeness import (
        assert_artifact_complete,
        check_artifact_completeness,
    )

    _mdl, dropped, _ = _export_with_exclusions(workdir, "gate", ["mtp."])
    report = assert_artifact_complete(dropped)
    assert report.excluded_namespaces == ["mtp."]

    # Strip the record: the artifact is byte-identical, but nothing now says
    # the absence was intended. A weight whose scale is missing must fail
    # again, which is what proves the exemption is the RECORD and not the
    # prefix.
    _mdl2, kept, _ = _export_with_exclusions(workdir, "gate2", None)
    tensors = load_file(str(kept / "model.safetensors"))
    tensors.pop("mtp.0.attn.wq_a.scale")
    save_file(tensors, str(kept / "model.safetensors"))
    report = check_artifact_completeness(kept)
    assert not report.ok
    assert "mtp.0.attn.wq_a" in report.missing_scale


def _embedding_model(mdl: Path, vocab: int = 64, hid: int = 256) -> None:
    """A model whose config declares vocab_size, so the embedding cross-check
    has both halves to agree on."""
    torch.manual_seed(3)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file({
        "model.embed_tokens.weight":
            (torch.randn(vocab, hid) * 0.3).to(torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, hid) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(hid, dtype=torch.bfloat16),
    }, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"hidden_size": hid, "vocab_size": vocab})
    )


def test_streaming_ships_the_quantized_embedding_like_the_in_memory_exporter(
    workdir,
):
    """The 13.0 GB card lane needs a quantized embedding AND streaming.

    The embedding declaration lived only in the in-memory exporter, which
    materialises the whole model twice over and OOMs on a 27B. Porting it is
    only safe if the streamed artifact is the in-memory one byte-for-byte, so
    that is what this asserts -- alongside the two properties the consumer
    actually depends on: the unit is claimed by the `quantized_embedding`
    declaration and NOT by a config group (vLLM's compressed-tensors embedding
    path raises for NVFP4), and it carries no `input_global_scale` (a lookup
    has no input activation, so the serving method registers no such
    parameter and an emitted one is an unmatched checkpoint key at load).
    """
    mdl = workdir / "model"
    _embedding_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.embed_tokens": "NVFP4",
        "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                            "cb_k": 16},
    })
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}

    counts_mem = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    counts_str = export_nvfp4_cb_streaming(
        mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(counts_mem) == dict(counts_str)

    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)

    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    assert qm["ignore"] == qs["ignore"]

    # The wire spelling is the consumer's lowercase one, not the recipe's.
    declaration = parse_quantized_embedding_declaration(qs)
    assert declaration == {"model.embed_tokens": "nvfp4"}
    assert declaration == parse_quantized_embedding_declaration(qm)

    claimed = {
        target
        for group in qs["config_groups"].values()
        for target in group.get("targets", ())
    }
    assert not any("embed_tokens" in target for target in claimed)

    embedding_keys = {k for k in ts if k.startswith("model.embed_tokens.")}
    assert embedding_keys == {
        "model.embed_tokens.weight_packed",
        "model.embed_tokens.weight_scale",
        "model.embed_tokens.weight_global_scale",
    }


def test_streaming_refuses_an_embedding_whose_name_and_shape_disagree(workdir):
    """Name and checkpoint shape are INDEPENDENT conditions and must agree.

    Getting this wrong ships an artifact whose embedding is dispatched as a
    Linear, or a Linear dispatched as a lookup -- neither fails at export, both
    fail at serve. A vocab-shaped tensor under another name must refuse.
    """
    mdl = workdir / "model"
    torch.manual_seed(4)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file({
        # Vocab-shaped, but not named like an embedding.
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(64, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"hidden_size": 256, "vocab_size": 64})
    )
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj": "NVFP4"})

    with pytest.raises(ValueError, match="cannot classify as a token embedding"):
        export_nvfp4_cb_streaming(mdl, ap, workdir / "s", {}, device="cpu")
