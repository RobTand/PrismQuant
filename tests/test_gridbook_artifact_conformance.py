"""Tiny producer artifacts decoded through Gridbook's independent CPU ABI.

This is a cross-repository compatibility test, not a production dependency:
PrismaQuant never imports Gridbook while probing, allocating, or exporting.
The dedicated CI job installs the exact external Gridbook pin, exports tiny
real containers, then checks their config, sidecar, index stream, and scale
planes with Gridbook's packaged contract and torch-only codec helpers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


REQUIRE_CONTRACT = os.environ.get(
    "PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT") == "1"

pytestmark = pytest.mark.skipif(
    not REQUIRE_CONTRACT,
    reason="run by the pinned Gridbook compatibility CI job",
)

_PINNED_ASSIGNMENT = {
    "model.layers.0.mlp.gate_proj": {
        "data_type": "nvfp4_cb", "cb_k": 12,
    },
    "model.layers.1.mlp.gate_proj": {
        "data_type": "nvfp4_cb", "cb_k": 13,
    },
    # A third fp4 rung at a different k. This leaf carried the SIGNED family
    # until the producer deleted it (2026-08-17: n_sub=1 can never satisfy
    # gridbook's native-FP4 predicate and no allocation ever chose it); the
    # runtime contract still lists signed as decodable, but this test exports
    # through the producer, so it can only exercise what the producer emits.
    "model.layers.2.mlp.up_proj": {
        "data_type": "nvfp4_cb", "cb_k": 16,
    },
    "model.layers.3.mlp.down_proj": {
        "data_type": "nvfp4_cb", "cb_k": 24,
    },
    "model.layers.4.mlp.down_proj": {
        "data_type": "fp8_cb", "cb_k": 28,
    },
}

_CANDIDATE_ENDPOINTS = {
    "model.layers.0.mlp.gate_proj": {
        "data_type": "nvfp4_cb", "cb_k": 1,
    },
    "model.layers.3.mlp.down_proj": {
        "data_type": "nvfp4_cb", "cb_k": 25,
    },
}


def _assignment_for_contract(contract: dict) -> dict[str, dict]:
    """Exercise new endpoints only after the installed reader declares them.

    The immutable release-pin CI must validate bytes the pin actually accepts;
    it cannot assert candidate K1/K25 support against Gridbook 0.8.11/v4.  The
    same fixture automatically promotes to the expanded endpoint seam when a
    candidate checkout or future released pin publishes both rungs.
    """

    assignment = {
        qname: dict(entry) for qname, entry in _PINNED_ASSIGNMENT.items()
    }
    nvfp4 = next(
        row for row in contract["formats"] if row["family"] == "NVFP4_CB_K"
    )
    if {1, 25} <= set(nvfp4["rungs"]):
        assignment.update({
            qname: dict(entry)
            for qname, entry in _CANDIDATE_ENDPOINTS.items()
        })
    return assignment


def _write_tiny_model(
    path: Path,
    assignment: dict[str, dict],
) -> dict[str, torch.Tensor]:
    path.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260801)
    tensors = {
        qname + ".weight": (
            torch.randn(16, 256, generator=generator) * 0.25
        ).to(torch.bfloat16)
        for qname in assignment
    }
    tensors["model.norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({
        "architectures": ["GridbookContractTiny"],
        "hidden_size": 256,
    }))
    return tensors


@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory):
    from gridbook.runtime_contract import load_runtime_contract
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    root = tmp_path_factory.mktemp("gridbook-artifact-contract")
    source = root / "source"
    selected_assignment = _assignment_for_contract(load_runtime_contract())
    weights = _write_tiny_model(source, selected_assignment)
    assignment = root / "assignment.json"
    assignment.write_text(json.dumps(selected_assignment))
    col_weights = {
        qname: torch.linspace(0.5, 1.5, 256) for qname in selected_assignment
    }

    outputs = {}
    for coding in ("v1", "two_tier"):
        out = root / coding
        export_nvfp4_cb(
            source,
            assignment,
            out,
            col_weights,
            shared_codebook_spec={"source": "lattice"},
            device="cpu",
            scale_coding=coding,
            allow_unstamped_research=True,
        )
        outputs[coding] = out
    return weights, outputs, selected_assignment


def _family_entry(contract: dict, format_name: str) -> dict:
    hits = [
        entry for entry in contract["formats"]
        if format_name.startswith(entry["family"])
    ]
    assert len(hits) == 1, (format_name, hits)
    return hits[0]


def _ceil_first_widths(k: int, n_sub: int) -> tuple[int, ...]:
    base, extra = divmod(int(k), int(n_sub))
    return tuple(base + (index < extra) for index in range(n_sub))


def _extract_codewords(
    packed: torch.Tensor,
    *,
    k: int,
    type_size: int,
) -> torch.Tensor:
    """CPU mirror of Gridbook's little-endian 8-byte kernel window."""

    rows, row_bytes = packed.shape
    assert row_bytes % type_size == 0
    n_superblocks = row_bytes // type_size
    blocks = packed.reshape(rows, n_superblocks, type_size).to(torch.int64)
    padded = torch.nn.functional.pad(blocks, (0, 8))
    result = torch.empty(rows, n_superblocks, 32, dtype=torch.int64)
    mask = (1 << k) - 1
    for vector in range(32):
        bit_position = vector * k
        byte_position, shift = divmod(bit_position, 8)
        window = torch.zeros(rows, n_superblocks, dtype=torch.int64)
        for offset in range(8):
            window |= padded[..., byte_position + offset] << (8 * offset)
        result[..., vector] = (window >> shift) & mask
    return result


def _gridbook_cpu_decode(
    codec,
    scheme: dict,
    packed: torch.Tensor,
    weight_scale: torch.Tensor | None,
    sidecar: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Decode using Gridbook-owned codebook/scale helpers plus kernel ABI."""

    grid = scheme["grid"]
    mode = scheme["mode"]
    k = int(scheme["k"])
    n_sub = int(scheme["n_sub"])
    type_size = int(scheme["type_size"])
    rows, row_bytes = packed.shape
    n_superblocks = row_bytes // type_size

    refs = scheme["codebook_ref"]
    names = list(refs) if isinstance(refs, list) else [refs]
    subtables = [sidecar[name].to(torch.float32) for name in names]
    # This is the consumer's real sidecar/grid validation and flattening path.
    codec.build_flat_codebook(subtables, "contract fixture", grid)
    assert len(subtables) == n_sub

    codewords = _extract_codewords(packed, k=k, type_size=type_size)
    values = torch.empty(rows, n_superblocks, 32, 8, dtype=torch.float32)
    assert mode == "product"
    widths = _ceil_first_widths(k, n_sub)
    sub_dim = 8 // n_sub
    bit_offset = 0
    for index, width in enumerate(widths):
        sub_index = (codewords >> bit_offset) & ((1 << width) - 1)
        values[..., index * sub_dim:(index + 1) * sub_dim] = (
            subtables[index][sub_index]
        )
        bit_offset += width

    if grid == "fp8":
        assert weight_scale is not None
        scales = weight_scale.to(torch.float32).reshape(rows, 1, 1, 1)
    else:
        assert weight_scale is None
        scale_coding = scheme.get("scale_coding")
        if scale_coding is None:
            scale16 = codec.decode_fp4_scale_plane(packed, k).reshape(
                rows, n_superblocks, 16
            )
        else:
            assert scale_coding["kind"] == codec.SCALE_CODING_TWO_TIER
            blocks = packed.reshape(rows, n_superblocks, type_size)
            super_exponent = blocks[..., 4 * k].to(torch.int64)
            nibble_bytes = blocks[..., 4 * k + 1:4 * k + 9].to(torch.int64)
            scale_codes = torch.empty(
                rows, n_superblocks, 16, dtype=torch.int64
            )
            scale_codes[..., 0::2] = nibble_bytes & 0xF
            scale_codes[..., 1::2] = nibble_bytes >> 4
            compose = codec.build_compose_table(
                scale_coding["table"]
            ).reshape(256, 16)
            scale16 = compose[
                super_exponent.unsqueeze(-1), scale_codes
            ]
        scales = scale16.repeat_interleave(16, dim=-1).reshape(
            rows, n_superblocks, 32, 8
        )
    return (values * scales).reshape(rows, n_superblocks * 256)


@pytest.mark.parametrize(
    "coding, expected_layout", [("v1", 1), ("two_tier", 2)]
)
def test_tiny_export_is_gridbook_config_sidecar_and_decode_compatible(
    tiny_artifacts,
    coding,
    expected_layout,
):
    from gridbook import codec
    from gridbook.runtime_contract import load_runtime_contract
    from prismaquant import nvfp4_cb_formats as producer

    weights, outputs, selected_assignment = tiny_artifacts
    out = outputs[coding]
    contract = load_runtime_contract()
    nvfp4 = _family_entry(contract, "NVFP4_CB_K1")
    selected_nvfp4_rungs = {
        int(entry["cb_k"])
        for entry in selected_assignment.values()
        if entry["data_type"] == "nvfp4_cb"
    }
    if {1, 25} <= set(nvfp4["rungs"]):
        assert {1, 25} <= selected_nvfp4_rungs
    else:
        assert not ({1, 25} & selected_nvfp4_rungs)
    pointer = json.loads((out / "config.json").read_text())["quantization_config"]
    assert pointer["quant_method"] == contract["quant_method"]["canonical"]
    quant_config = json.loads((out / pointer["config_file"]).read_text())
    layout = quant_config.get(
        contract["layout"]["field"],
        contract["layout"]["default_when_absent"],
    )
    assert layout == expected_layout
    assert pointer["codebook_file"] == quant_config["codebook_file"]

    tensors = load_file(str(out / "model.safetensors"))
    sidecar = load_file(str(out / quant_config["codebook_file"]))
    seen_modes = set()
    for group in quant_config["config_groups"].values():
        scheme = group["scheme"]
        format_name = group["format"]
        family = _family_entry(contract, format_name)
        k = int(scheme["k"])
        assert k in family["rungs"]
        assert (
            scheme["grid"], scheme["mode"], int(scheme["n_sub"])
        ) == (family["grid"], family["mode"], family["n_sub"])
        assert scheme["vec_dim"] == contract["packing"]["vector_dim"]
        assert scheme["superblock"] == contract["packing"]["superblock_weights"]

        scale_kind = "v1"
        scheme_coding = scheme.get("scale_coding")
        if scheme_coding is not None:
            scale_kind = scheme_coding["kind"]
        rule = next(
            item for item in contract["layout"]["type_size_rules"]
            if item["grid"] == scheme["grid"]
            and item["scale_coding"] == scale_kind
        )
        assert scheme["type_size"] == (
            contract["packing"]["index_bytes_per_k"] * k
            + rule["scale_plane_bytes"]
        )

        refs = scheme["codebook_ref"]
        names = list(refs) if isinstance(refs, list) else [refs]
        assert len(names) == family["n_sub"]
        assert all(sidecar[name].dtype == torch.float16 for name in names)
        seen_modes.add((scheme["grid"], scheme["mode"]))

        for qname in group["targets"]:
            packed = tensors[qname + ".cb_qweight"]
            weight_scale = tensors.get(qname + ".weight_scale")
            decoded = _gridbook_cpu_decode(
                codec, scheme, packed, weight_scale, sidecar
            )
            codebook = (
                tuple(sidecar[name].float() for name in names)
                if len(names) > 1 else sidecar[names[0]].float()
            )
            expected_fields = producer.nvfp4_cb_unpack(
                packed,
                k,
                scheme["grid"],
                scheme["mode"],
                tuple(weights[qname + ".weight"].shape),
                codebook=codebook,
                scales=(weight_scale.reshape(-1, 1)
                        if weight_scale is not None else None),
                scale_coding=coding if scheme["grid"] == "fp4" else "v1",
            )
            expected = producer.nvfp4_cb_reconstruct(
                expected_fields,
                k,
                grid=scheme["grid"],
                mode=scheme["mode"],
            ).float()
            assert torch.equal(decoded, expected), (coding, format_name, qname)

    assert seen_modes == {("fp4", "product"), ("fp8", "product")}
