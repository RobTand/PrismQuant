from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import cb_learned_bundle as bundle
from prismaquant import gridbook_runtime_pin as runtime_pin
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import parse_format_name, subtable_bit_widths


def _inputs():
    base = torch.linspace(-2.0, 2.0, 2 * 256, dtype=torch.float32).reshape(2, 256)
    return {
        "model.layers.1.self_attn.q_proj": base,
        "model.layers.1.self_attn.k_proj": base.flip(0).contiguous(),
    }, {
        "model.layers.1.self_attn.q_proj": torch.linspace(
            0.25, 1.25, 256, dtype=torch.float32
        ),
        "model.layers.1.self_attn.k_proj": torch.linspace(
            1.25, 0.25, 256, dtype=torch.float32
        ),
    }


def _fast_learn_pool(weight, col_weights, rung):
    del weight, col_weights
    parsed = parse_format_name(f"FP8_CBL_K{int(rung)}")
    assert parsed is not None
    family, k = parsed
    widths = subtable_bit_widths(k, family.mode, family.n_sub)
    # A row permutation keeps every value on the exact FP8 grid while ensuring
    # the learned payload is not byte-equal to the canonical lattice payload.
    return tuple(
        cb.fixed_lattice(bits, "fp8", 2).roll(index + 1, dims=0)
        for index, bits in enumerate(widths)
    )


@pytest.fixture
def mixed_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "learn_pool", _fast_learn_pool)
    weights, col_weights = _inputs()
    path = tmp_path / "learned-menu.pqcb"
    calls = []

    def provider(qname):
        calls.append(qname)
        return weights[qname]

    loaded = bundle.train_and_save_bundle_streaming(
        path,
        qnames=weights,
        weight_provider=provider,
        col_weights=col_weights,
        formats=("NVFP4_CB_K12", "FP8_CBL_K28"),
    )
    assert calls == sorted(weights)
    return loaded, weights, col_weights


def test_rung_policy_is_measurement_gated_not_the_2048_rule():
    assert bundle.require_cbl_rung_enabled(28) == 28
    assert bundle.require_cbl_rung_enabled(43) == 43
    assert bundle.CBL_RUNG_POLICY[43]["enabled"] is True
    expected_new_measurements = {
        44: (0.6057, "cbl_k43_k47.log:31"),
        45: (0.6929, "cbl_k43_k47.log:40"),
        46: (0.8312, "cbl_k43_k47.log:51"),
    }
    for rung, (ratio, provenance) in expected_new_measurements.items():
        policy = bundle.CBL_RUNG_POLICY[rung]
        assert bundle.require_cbl_rung_enabled(rung) == rung
        assert policy["enabled"] is True
        assert policy["status"] == "measured_go_sweep_matched"
        assert policy["cbl_over_lattice_base_ratio"] == pytest.approx(ratio)
        assert str(policy["provenance"]).endswith(provenance)
    assert bundle.CBL_RUNG_POLICY[47]["status"] == (
        "measured_no_go_sweep_matched"
    )
    assert bundle.CBL_RUNG_POLICY[47]["cbl_over_lattice_base_ratio"] == (
        pytest.approx(1.0689)
    )
    assert bundle.CBL_RUNG_POLICY[48]["status"] == "measured_no_go"
    with pytest.raises(ValueError, match=r"K47.*measured_no_go_sweep_matched"):
        bundle.require_cbl_rung_enabled(47)
    with pytest.raises(ValueError, match=r"K48.*measured_no_go"):
        bundle.require_cbl_rung_enabled(48)
    with pytest.raises(ValueError, match="explicit learned format name"):
        bundle.require_cbl_rung_enabled("NVFP4_CB_K16")


def test_bundle_has_distinct_learned_cells_and_complete_mixed_sidecar(mixed_bundle):
    loaded, weights, _col_weights = mixed_bundle
    refs = loaded.codebook_refs_by_cell
    qname, kname = sorted(weights)

    q_refs = refs[qname]["FP8_CBL_K28"]
    k_refs = refs[kname]["FP8_CBL_K28"]
    assert q_refs != k_refs
    assert all(qname in ref for ref in q_refs)
    assert all(kname in ref for ref in k_refs)

    q_lattice = refs[qname]["NVFP4_CB_K12"]
    k_lattice = refs[kname]["NVFP4_CB_K12"]
    assert q_lattice == k_lattice
    assert all("cb_codebook.lattice." in ref for ref in q_lattice)

    all_refs = {
        ref
        for formats in refs.values()
        for cell_refs in formats.values()
        for ref in cell_refs
    }
    assert set(loaded.sidecar_tensors) == all_refs
    assert set(loaded.codebook_content_digests) == all_refs
    assert len(all_refs) == 10  # 2 shared NVFP4 + 2 * 4 learned FP8 subtables.
    assert loaded.bundle_content_sha256 == loaded.manifest["bundle_content_sha256"]

    # Bundle construction and footprint accounting must identify the same
    # canonical FP16 lattice bytes; Gridbook verifies the complete name set.
    from prismaquant.nvfp4_cb_footprint import lattice_codebook_content_sha256

    assert tuple(
        loaded.codebook_content_digests[ref] for ref in q_lattice
    ) == lattice_codebook_content_sha256("NVFP4_CB_K12")


def test_learned_sidecar_roundtrip_is_bit_exact_vs_emulation(mixed_bundle):
    loaded, weights, col_weights = mixed_bundle
    qname = "model.layers.1.self_attn.q_proj"
    weight = weights[qname]
    cw = col_weights[qname]
    learned = loaded.codebook_for(
        qname,
        "FP8_CBL_K28",
        weight=weight,
        col_weights=cw,
    )

    packed, emulated_fields = cb.nvfp4_cb_pack(
        weight,
        28,
        grid="fp8",
        mode="product",
        col_weights=cw,
        codebook=learned,
        scale_sweep=False,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    reloaded = bundle.load_bundle(loaded.path)
    disk_book = reloaded.codebook_for(qname, "FP8_CBL_K28")
    unpacked = cb.nvfp4_cb_unpack(
        packed,
        28,
        "fp8",
        "product",
        tuple(weight.shape),
        codebook=disk_book,
        scales=emulated_fields["scales"],
    )
    from_disk = cb.nvfp4_cb_reconstruct(
        unpacked, 28, grid="fp8", mode="product"
    )
    emulated = cb.nvfp4_cb_reconstruct(
        emulated_fields, 28, grid="fp8", mode="product"
    )
    assert torch.equal(from_disk, emulated)

    repacked, _ = cb.nvfp4_cb_pack(
        weight,
        28,
        grid="fp8",
        mode="product",
        col_weights=cw,
        codebook=disk_book,
        scale_sweep=False,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    assert torch.equal(repacked, packed)


def test_bundle_refuses_missing_or_mismatched_values(mixed_bundle, tmp_path):
    loaded, weights, col_weights = mixed_bundle
    qname = "model.layers.1.self_attn.q_proj"
    with pytest.raises(ValueError, match="no FP8_CBL_K32 cell.*lattice fallback"):
        loaded.codebook_for(qname, "FP8_CBL_K32")
    with pytest.raises(ValueError, match="source weight does not match"):
        loaded.codebook_for(
            qname,
            "FP8_CBL_K28",
            weight=weights[qname] + 1,
            col_weights=col_weights[qname],
        )
    with pytest.raises(ValueError, match="col_weights do not match"):
        loaded.codebook_for(
            qname,
            "FP8_CBL_K28",
            weight=weights[qname],
            col_weights=col_weights[qname] + 1,
        )

    with safe_open(str(loaded.path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    first = sorted(tensors)[0]
    tensors[first] = tensors[first].clone()
    tensors[first].view(-1)[0] += torch.tensor(1, dtype=torch.float16)
    corrupt = tmp_path / "digest-mismatch.pqcb"
    save_file(tensors, str(corrupt), metadata=metadata)
    with pytest.raises(ValueError, match="codebook digest mismatch"):
        bundle.load_bundle(corrupt)

    tensors.pop(first)
    missing = tmp_path / "missing-table.pqcb"
    save_file(tensors, str(missing), metadata=metadata)
    with pytest.raises(ValueError, match="digest map does not cover sidecar exactly"):
        bundle.load_bundle(missing)

    # Even a self-consistent digest rewrite cannot relabel arbitrary grid
    # values as the canonical lattice.
    with safe_open(str(loaded.path), framework="pt", device="cpu") as handle:
        lattice_metadata = dict(handle.metadata() or {})
        lattice_tensors = {
            name: handle.get_tensor(name) for name in handle.keys()
        }
    lattice_ref = loaded.codebook_refs_by_cell[qname]["NVFP4_CB_K12"][0]
    lattice_tensors[lattice_ref] = lattice_tensors[lattice_ref].roll(1, dims=0)
    manifest = json.loads(
        lattice_metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY]
    )
    replacement_digest = bundle.codebook_table_sha256(
        lattice_tensors[lattice_ref]
    )
    manifest["codebook_content_sha256"][lattice_ref] = replacement_digest
    for formats in manifest["cells"].values():
        for cell in formats.values():
            if lattice_ref in cell["codebook_ref"]:
                index = cell["codebook_ref"].index(lattice_ref)
                cell["content_sha256"][index] = replacement_digest
    manifest["bundle_content_sha256"] = bundle._bundle_content_sha256(
        manifest["codebook_content_sha256"]
    )
    lattice_metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY] = (
        bundle._canonical_json(manifest)
    )
    relabeled = tmp_path / "relabeled-lattice.pqcb"
    save_file(lattice_tensors, str(relabeled), metadata=lattice_metadata)
    with pytest.raises(ValueError, match="differs from the canonical"):
        bundle.load_bundle(relabeled)


def test_cached_loader_invalidates_on_same_path_file_identity_change(mixed_bundle):
    loaded, _weights, _col_weights = mixed_bundle
    first = bundle.load_bundle_cached(loaded.path)
    second = bundle.load_bundle_cached(loaded.path)
    assert first is second
    stat = loaded.path.stat()
    os.utime(
        loaded.path,
        ns=(int(stat.st_atime_ns), int(stat.st_mtime_ns) + 1_000_000),
    )
    third = bundle.load_bundle_cached(loaded.path)
    assert third is not first


def _gridbook_pin(version, *, commit="a" * 40, version_is_release=False):
    return runtime_pin.parse_gridbook_runtime_pin({
        "schema": runtime_pin.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": commit,
        "version": version,
        "version_is_release": version_is_release,
    })


_ROUTED_DETECTORS = (
    ("model.layers.2.mlp.experts.gate_up_proj", False, (2, 256)),
    ("model.layers.2.mlp.gate_up_proj", True, (2, 256)),
    ("model.layers.2.mlp.gate_up_proj", False, (2, 2, 256)),
)


@pytest.mark.parametrize("qname,routed_flag,shape", _ROUTED_DETECTORS)
def test_routed_moe_learned_ref_fails_with_old_gridbook_pin(
    monkeypatch, qname, routed_flag, shape
):
    old = _gridbook_pin(
        "0.8.2",
        commit="9f915dd868eab2e13ab7847a67c594e2c5c8955c",
        version_is_release=True,
    )
    monkeypatch.setattr(bundle, "load_gridbook_runtime_pin", lambda: old)
    with pytest.raises(
        ValueError,
        match=(
            r"Gridbook 0\.8\.2.*9f915dd868eab2e13ab7847a67c594e2c5c8955c"
            r".*per-row/per-role LUT offset ABI.*Gridbook >=0\.8\.3"
        ),
    ):
        bundle.refuse_routed_moe_learned(
            qname,
            routed_moe=routed_flag,
            weight=torch.zeros(shape),
        )


@pytest.mark.parametrize("qname,routed_flag,shape", _ROUTED_DETECTORS)
def test_routed_moe_learned_ref_lifts_at_gridbook_0_8_3_even_before_tag(
    monkeypatch, qname, routed_flag, shape
):
    supported = _gridbook_pin("0.8.3", version_is_release=False)
    monkeypatch.setattr(
        bundle, "load_gridbook_runtime_pin", lambda: supported
    )
    bundle.refuse_routed_moe_learned(
        qname,
        routed_moe=routed_flag,
        weight=torch.zeros(shape),
    )


def test_routed_moe_learned_ref_fails_closed_on_invalid_pin(monkeypatch):
    def invalid_pin():
        raise runtime_pin.GridbookRuntimePinError("malformed pin fixture")

    monkeypatch.setattr(bundle, "load_gridbook_runtime_pin", invalid_pin)
    with pytest.raises(
        ValueError,
        match=r"runtime pin is invalid.*malformed pin fixture.*Gridbook >=0\.8\.3",
    ):
        bundle.refuse_routed_moe_learned(
            "model.layers.2.mlp.experts.gate_up_proj",
            weight=torch.zeros(2, 256),
        )


def test_shared_expert_dense_name_is_not_misclassified_as_routed():
    bundle.refuse_routed_moe_learned(
        "model.layers.2.mlp.shared_experts.gate_proj",
        weight=torch.zeros(2, 256),
    )


def test_builder_refuses_relabeling_lattice_name_before_training(tmp_path, monkeypatch):
    called = False

    def should_not_train(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("trainer should not run")

    monkeypatch.setattr(bundle, "learn_pool", should_not_train)
    qname = "model.layers.0.self_attn.q_proj"
    with pytest.raises(ValueError, match="contradict source-bearing format"):
        bundle.train_and_save_bundle(
            tmp_path / "ambiguous-old-name.pqcb",
            weights={qname: torch.zeros(2, 256)},
            col_weights={qname: torch.ones(256)},
            formats=("FP8_CB_K28",),
            learned_formats=("FP8_CB_K28",),
        )
    assert called is False


def test_certified_kernel_is_cpu_deterministic_and_tool_delegates(monkeypatch):
    weight = torch.linspace(-1, 1, 2 * 256).reshape(1, 2, 256)
    col_weights = torch.ones(1, 256)
    first = bundle.learn_pool(weight, col_weights, 28)
    second = bundle.learn_pool(weight, col_weights, 28)
    assert len(first) == 4
    assert [tuple(table.shape) for table in first] == [(128, 2)] * 4
    assert all(torch.equal(left, right) for left, right in zip(first, second))

    from tools import dsv4_cbl_kernels as study

    assert all(study.cbl_eligible(rung) for rung in (44, 45, 46))
    assert not study.cbl_eligible(47)
    assert not study.cbl_eligible(48)
    assert study.SEMANTICS_STAMP["cbl_dispatch"] == (
        "per_rung_measurement_policy"
    )
    assert "eligible_max_rung" not in study.SEMANTICS_STAMP
    assert study.SEMANTICS_STAMP["cbl_rung_policy"]["46"] == (
        bundle.CBL_RUNG_POLICY[46]
    )

    marker = object()
    monkeypatch.setattr(study, "_certified_learn_pool", lambda *_args: marker)
    assert study.learn_pool(weight, col_weights, 44) is marker
