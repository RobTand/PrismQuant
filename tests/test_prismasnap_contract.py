"""Failing-branch coverage for the PrismaSnap lane-admission contract.

`prismasnap_contract.main` is the entry `run-pipeline.sh` invokes before any
expensive stage, and `refuse_prismasnap_lane_before_output` decorates four
programmatic exporters.  Both had production call sites and no test, so the
gates that fail closed on a snapped source were themselves unexercised.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import prismaquant.prismasnap_contract as contract


def _provenance(model_dir: Path, payload: object = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    marker = model_dir / contract.PRISMASNAP_PROVENANCE_JSON
    marker.write_text(
        json.dumps(payload if payload is not None else {"schema": "bogus"}),
        encoding="utf-8",
    )
    return marker


def test_main_refuses_an_unverified_or_unsafe_snap_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Positive control: no marker at all means the source is not Snap-prepared
    # and `main` admits it, so the refusals below are the gate and not an
    # unconditional failure.
    plain = tmp_path / "plain-source"
    plain.mkdir()
    monkeypatch.setattr(sys, "argv", ["prismasnap_contract", "--model", str(plain)])
    assert contract.main() == 0

    # A marker that is present but not a regular file is ambiguous state, and
    # ambiguous state fails closed rather than being ignored.
    symlinked = tmp_path / "symlinked-source"
    symlinked.mkdir()
    target = _provenance(tmp_path / "elsewhere")
    (symlinked / contract.PRISMASNAP_PROVENANCE_JSON).symlink_to(target)
    monkeypatch.setattr(sys, "argv", ["prismasnap_contract", "--model", str(symlinked)])
    with pytest.raises(RuntimeError, match="not a regular file"):
        contract.main()

    # A real marker means the full checkpoint validation runs; a marker that
    # does not replay must refuse before the pipeline spends anything.
    snapped = tmp_path / "snapped-source"
    _provenance(snapped)
    monkeypatch.setattr(sys, "argv", ["prismasnap_contract", "--model", str(snapped)])
    with pytest.raises(RuntimeError):
        contract.main()


def test_main_requires_the_model_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prismasnap_contract"])
    with pytest.raises(SystemExit) as excinfo:
        contract.main()
    assert excinfo.value.code == 2


def test_lane_decorator_refuses_a_snapped_source_before_the_exporter_runs(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    @contract.refuse_prismasnap_lane_before_output(lane="Gridbook/codebook")
    def fake_export(model_dir, output_dir, *, shards: int = 1) -> str:
        calls.append(Path(model_dir))
        (Path(output_dir)).mkdir(parents=True, exist_ok=True)
        return "exported"

    plain = tmp_path / "plain-source"
    plain.mkdir()
    assert fake_export(plain, tmp_path / "plain-out") == "exported"
    assert calls == [plain]

    snapped = tmp_path / "snapped-source"
    _provenance(snapped)
    destination = tmp_path / "snapped-out"
    with pytest.raises(RuntimeError, match="not admitted to the Gridbook/codebook lane"):
        fake_export(snapped, destination)
    # "before their transaction" is the contract: the wrapped exporter never
    # ran and no output tree was created.
    assert calls == [plain]
    assert not destination.exists()

    # Positional binding must refuse identically to keyword binding.
    with pytest.raises(RuntimeError, match="not admitted to the"):
        fake_export(model_dir=snapped, output_dir=destination)
    assert not destination.exists()


def test_lane_decorator_refuses_a_function_it_cannot_bind_model_dir_on() -> None:
    @contract.refuse_prismasnap_lane_before_output(lane="GGUF")
    def wrong_signature(source_dir, output_dir) -> str:  # no `model_dir`
        return "exported"

    with pytest.raises(RuntimeError, match="could not bind model_dir"):
        wrong_signature("a", "b")


def test_unvalidated_lane_refusal_names_the_lane_and_the_marker(
    tmp_path: Path,
) -> None:
    snapped = tmp_path / "snapped-source"
    marker = _provenance(snapped)
    with pytest.raises(RuntimeError) as excinfo:
        contract.refuse_prismasnap_for_unvalidated_lane(snapped, lane="GGUF")
    message = str(excinfo.value)
    assert "GGUF" in message and str(marker) in message
