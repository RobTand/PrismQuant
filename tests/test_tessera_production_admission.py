"""Production pin admission preserves the conditions its cells attest."""
import json

import pytest

from prismaquant import tessera_menu as menu, tessera_render as render
from prismaquant import tessera_runtime_contract as contract
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_serving_runtime_pin import require_pinned_tessera_runtime


@pytest.mark.parametrize("residency", ["resident", "streamed"])
def test_production_pin_preserves_attested_conditions(monkeypatch, residency):
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()  # Exercise the real production pin, no substitute.
    table, formats = render._pinned_serving_table()
    name = "TESSERA_E4M3_K1_R1024"
    reference = next(cell for cell in table.cells
                     if cell.family == "TESSERA_E4M3_K1"
                     and cell.structure == "dense"
                     and f"TESSERA_SERVE_MODE={residency}" in cell.requires_serve_flags)
    context = ServingContext(
        platform=reference.platform, structure="dense", residency=residency,
        runtime_image=reference.runtime_image, execution_mode=reference.execution_modes[0])
    cells = render.tessera_attesting_cells(name, table=table, formats=formats,
                                          serving_context=context)
    assert cells
    expected_flags = tuple(dict.fromkeys(flag for cell in cells
                                        for flag in cell.requires_serve_flags))
    assert expected_flags == (f"TESSERA_SERVE_MODE={residency}",)
    assert {cell.route_status for cell in cells} == {"backed_with_serve_flag"}
    admission = menu.route_admission(name, serving_context=context)
    resolved = menu.tessera_resolved_serving_lane(name, serving_context=context).as_dict()
    assert admission.attested
    assert admission.route_status == reference.route_status
    assert admission.requires_serve_flags == expected_flags
    assert resolved["route_status"] == reference.route_status
    assert resolved["requires_serve_flags"] == list(expected_flags)
    payload = json.loads(contract.contract_path().read_text())
    ceiling = next(row["max_world_size"] for row in payload["tensor_parallel"]["units"]
                   if row["unit"] == reference.family)
    assert admission.max_world_size == ceiling == 1
    monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "1")
    dev = menu.route_admission(name, serving_context=context)
    assert (admission.route_status, admission.requires_serve_flags, admission.max_world_size) == (
        dev.route_status, dev.requires_serve_flags, dev.max_world_size)


@pytest.mark.parametrize("dev_pin", [False, True])
def test_conflicting_cell_statuses_refuse_with_named_cells(monkeypatch, dev_pin):
    from dataclasses import replace

    monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "1")
    parsed = contract.load_tessera_contract()
    table, formats = render._pinned_serving_table()
    reference = next(cell for cell in table.cells if cell.family == "TESSERA_E4M3_K1"
                     and cell.structure == "dense" and cell.regime == "decode"
                     and "TESSERA_SERVE_MODE=resident" in cell.requires_serve_flags)
    context = ServingContext(platform=reference.platform, structure="dense", residency="resident",
                             runtime_image=reference.runtime_image,
                             execution_mode=reference.execution_modes[0])
    if dev_pin:
        altered = replace(parsed, cells=tuple(
            replace(cell, route_status="backed") if cell.cell_id == reference.id else cell
            for cell in parsed.cells))
        monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: altered)
    else:
        monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV)
        altered = replace(table, cells=tuple(
            replace(cell, route_status="backed") if cell.id == reference.id else cell
            for cell in table.cells))
        monkeypatch.setattr(render, "_pinned_serving_table", lambda: (altered, formats))
    with pytest.raises(menu.TesseraMenuError, match="native cells disagree about route status") as err:
        menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert reference.id in str(err.value)
    assert "backed_with_serve_flag" in str(err.value)


def test_production_conditions_do_not_override_pin_refusal(monkeypatch):
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: False)
    table, _ = render._pinned_serving_table()
    reference = next(cell for cell in table.cells if cell.family == "TESSERA_E4M3_K1"
                     and cell.structure == "dense" and cell.regime == "decode"
                     and "TESSERA_SERVE_MODE=streamed" in cell.requires_serve_flags)
    context = ServingContext(platform=reference.platform, structure="dense", residency="streamed",
                             runtime_image=reference.runtime_image,
                             execution_mode=reference.execution_modes[0])
    admission = menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert not admission.attested
    assert admission.route_status == "unattested"
    assert admission.requires_serve_flags == ()
    assert admission.max_world_size is None
