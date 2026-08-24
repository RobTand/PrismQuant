from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys

import torch

from prismaquant import nvfp4_cb_formats as cb
from scripts import gen_nvfp4_cb_lattices as generator


_KEY = re.compile(r"(fp4|fp8)(pos)?_d([0-9]+)_k([0-9]+)")


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
