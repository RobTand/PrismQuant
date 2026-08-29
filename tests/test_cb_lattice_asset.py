from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from scripts import gen_nvfp4_cb_lattices as generator


_KEY = re.compile(r"(fp4|fp8)(pos)?_d([0-9]+)_k([0-9]+)")
_HISTORICAL_TABLE_SHA256 = {
    "fp4_d4_k10": "fdc497367d503968ba5dfe35c0b2595811d6c227d77c9b161236c3f76c5ae0c5",
    "fp4_d4_k11": "859f0826887ef2d41f1af7ef3a19366ce8af6e45bfd48c86fdd7d94ae58f4aa6",
    "fp4_d4_k12": "ef826908980f1b3e5f6e25f1da364b690f80ea73f781b7135dfd89584eeed0f8",
    "fp4_d4_k6": "e938ed8040103f1016aa2f1763dc00f7c9e60223f98bdeafb544b0f54daf3cf5",
    "fp4_d4_k7": "d07c934dd4323ac8594a2f1198f01577775ae9fa8905c17ad7bbeaf988c8ee80",
    "fp4_d4_k8": "502b439c8c7598cce4b0e1dcd623c2290582d0fe04a6a6c4e1a40502a5bd6ee5",
    "fp4_d4_k9": "ad68d8abfe7347f193ef27fc3c11ca1f0a54fbf924c3434c244cdbe6725644a8",
    "fp4_d8_k12": "3aebf2e59a3dce23f8495bbdd76e365f14508efa4cc1bc49394baa74a1dda85f",
    "fp4_d8_k13": "41737da0ec1c6d8c9943b90d016d1cead8fd503b06554d5a9e7c208a8c492e5b",
    "fp4_d8_k14": "654fef71fb7c3f5d2e8de248936699ac99dc3a2c2e419ab344d3faa93e56ceca",
    "fp4pos_d8_k5": "5c5f6ccd1bcc596279734fbfeebdec2d44076021424c89b93ecdd0f2aa1ee534",
    "fp4pos_d8_k6": "c7b4f040b3e223ad3072776d9f8769122e58e6efbe4408c39fca858a217bcd3b",
    "fp4pos_d8_k7": "52ee75178b9d58013f30f63d9a497fb44712437fe02f1a76fa2bb5c034bf6645",
    "fp4pos_d8_k8": "b0e5f603589ca272c9fd341d3d65d5f2b2b7f0f3ea94e7f4814f39ddd1386cd5",
    "fp8_d2_k1": "41efa9cd1aabd456d99384850dca0daf44ed6f3d3a6f99fb53b8ef36d9a2670c",
    "fp8_d2_k10": "18de5a3b656b3e68a9d25cff57dc4befdaf50df6c2a001b57cbacd6494e75643",
    "fp8_d2_k11": "57f68a24c0d4fe6277628859381fedca9ceab9ecf40123ed45fddec75c6c17fb",
    "fp8_d2_k12": "0f94c3a330910d2216cd9e91d81cc3945e3105978e048381befb3ea64e012acf",
    "fp8_d2_k2": "10a95cff3f64fccb74bc0d0ab8ebf5011422e03fae106be280f56cc8046075ce",
    "fp8_d2_k3": "3b6740caea089cf410390937c91c3a89d34766881ac9178cf44af46b2937a0c7",
    "fp8_d2_k4": "d68a3cb4dba3b7ef0eef65798b1ba18484afeaf2b2ea5258c42e88e71e7ae336",
    "fp8_d2_k5": "8beb2ba1d52068c9417f854792b21f2a8911a2001cc9b033a87a4f2a86ac5d73",
    "fp8_d2_k6": "10e56fdca8ec3146108a9e3e30d2c2248568a847e3a0a92b1971dc91f3f33d1e",
    "fp8_d2_k7": "945c62f0c3b71d5809687c7f1a9cd12132efd492d91ed9353847a574bb6640a6",
    "fp8_d2_k8": "7bf9daee793b0564ad1916c88ca0bfa45f16fbeae8513602b6f5d0ca893ba482",
    "fp8_d2_k9": "75fc6e9de4079a7351f7caa982c1b62f47911152dcc74eb679f1dc3c7fa23770",
}


def _table_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def test_digest_pinned_asset_survives_cache_clear_without_synthesis(monkeypatch):
    assert hashlib.sha256(cb._DATA.read_bytes()).hexdigest() == (
        cb._LATTICE_ASSET_SHA256
    )
    cb._lattice_file.cache_clear()
    cb._fixed_lattice_cpu.cache_clear()
    asset = cb._lattice_file()
    required = cb._production_lattice_keys()
    assert required <= set(asset)

    def no_research_fallback(*_args, **_kwargs):
        raise AssertionError("canonical producer lookup must not synthesize")

    monkeypatch.setattr(cb, "_build_lattice", no_research_fallback)
    for key in sorted(required):
        match = _KEY.fullmatch(key)
        assert match is not None
        grid, positive, raw_d, raw_k = match.groups()
        table = cb.fixed_lattice(
            int(raw_k),
            grid,
            int(raw_d),
            positive=bool(positive),
        )
        assert table.device.type == "cpu"
        assert _table_sha256(table) == _table_sha256(asset[key])


def test_digest_pinned_asset_is_identical_in_fresh_processes():
    source = r'''
import hashlib
import json
import re
from prismaquant import nvfp4_cb_formats as cb

cb._lattice_file.cache_clear()
cb._fixed_lattice_cpu.cache_clear()
cb._build_lattice = lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("canonical producer lookup synthesized")
)
asset = cb._lattice_file()
rows = []
pattern = re.compile(r"(fp4|fp8)(pos)?_d([0-9]+)_k([0-9]+)")
for key in sorted(cb._production_lattice_keys()):
    grid, positive, raw_d, raw_k = pattern.fullmatch(key).groups()
    table = cb.fixed_lattice(
        int(raw_k), grid, int(raw_d), positive=bool(positive)
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(str(table.dtype).encode("ascii"))
    digest.update(json.dumps(list(table.shape)).encode("ascii"))
    digest.update(table.numpy().tobytes(order="C"))
    rows.append([key, digest.hexdigest(), table.device.type])
print(json.dumps({
    "asset": hashlib.sha256(cb._DATA.read_bytes()).hexdigest(),
    "rows": rows,
}, sort_keys=True, separators=(",", ":")))
'''
    outputs = [
        subprocess.run(
            [sys.executable, "-c", source],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    payload = json.loads(outputs[0])
    assert payload["asset"] == cb._LATTICE_ASSET_SHA256
    assert {row[0] for row in payload["rows"]} == (
        cb._production_lattice_keys()
    )
    assert {row[2] for row in payload["rows"]} == {"cpu"}


def test_structured_fp4_d4_low_widths_are_nested_subsets_of_width6():
    cb._structured_fp4_d4_low_master.cache_clear()
    cb._structured_fp4_d4_lattice.cache_clear()
    base = cb._lattice_file()["fp4_d4_k6"].to(torch.float32).contiguous()
    previous = None
    for k in range(0, cb.STRUCTURED_FP4_D4_LOW_MAX_K + 1):
        table = cb._structured_fp4_d4_lattice(k)
        assert tuple(table.shape) == (1 << k, 4)
        assert int(table.unique(dim=0).shape[0]) == 1 << k
        if previous is not None:
            assert torch.equal(table[:previous.shape[0]], previous)
        previous = table
    assert previous is not None
    base_rows = {tuple(row) for row in base.tolist()}
    assert {tuple(row) for row in previous.tolist()} <= base_rows
    assert torch.equal(
        previous[0], base[(base * base).sum(dim=1).argmin()]
    )


def test_structured_fp4_d4_high_widths_are_nested_and_deterministic():
    cb._structured_fp4_d4_high_master.cache_clear()
    cb._structured_fp4_d4_lattice.cache_clear()
    expected_unique = {
        13: 1 << 13,
        14: 1 << 14,
        15: 1 << 15,
        16: 15 ** 4,
    }
    base = cb._lattice_file()["fp4_d4_k12"].to(torch.float32).contiguous()
    previous = base
    for k, unique_rows in expected_unique.items():
        first = cb._structured_fp4_d4_lattice(k)
        cb._structured_fp4_d4_high_master.cache_clear()
        cb._structured_fp4_d4_lattice.cache_clear()
        second = cb._structured_fp4_d4_lattice(k)
        assert tuple(first.shape) == (1 << k, 4)
        assert torch.equal(first, second)
        assert torch.equal(first[:previous.shape[0]], previous)
        assert int(first.unique(dim=0).shape[0]) == unique_rows
        assert _table_sha256(first) == _table_sha256(second)
        previous = first


def test_missing_production_table_refuses_runtime_synthesis(monkeypatch):
    asset = dict(cb._lattice_file())
    asset.pop("fp4_d4_k12")
    monkeypatch.setattr(cb, "_lattice_file", lambda: asset)
    cb._fixed_lattice_cpu.cache_clear()
    with pytest.raises(RuntimeError, match="canonical producer lattice.*missing"):
        cb.fixed_lattice(12, "fp4", 4)


def test_every_historical_asset_table_still_wins_byte_for_byte():
    asset = cb._lattice_file()
    observed = {
        key: _table_sha256(asset[key])
        for key in _HISTORICAL_TABLE_SHA256
    }
    assert observed == _HISTORICAL_TABLE_SHA256


def test_canonical_generator_requests_cpu_for_missing_tables(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "lattices.pt"
    observed_devices: list[str | torch.device | None] = []

    def fake_build(k, grid, d, positive=False, *, device=None):
        observed_devices.append(device)
        return torch.zeros(1 << int(k), int(d), dtype=torch.float32)

    monkeypatch.setattr(cb, "_DATA", output)
    monkeypatch.setattr(cb, "_build_lattice", fake_build)
    monkeypatch.setattr(
        generator,
        "required_lattice_specs",
        lambda: ((2, "fp8", 2, False),),
    )
    generator.main()

    assert observed_devices == ["cpu"]
    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert set(saved) == {"fp8_d2_k2"}
    assert saved["fp8_d2_k2"].device.type == "cpu"
