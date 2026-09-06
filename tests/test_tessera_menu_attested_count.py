"""An explicitly named Tessera rung is not "attested" until the predicate says so (#278)."""
from prismaquant import format_registry as fr
from prismaquant import tessera_menu as menu

ATTESTED = "TESSERA_E4M3_K1_R1024"
UNATTESTED = "TESSERA_E4M3_K1_R896"


def _admits_only_r1024(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible",
                        lambda name, **_kw: str(name).endswith("R1024"))


def test_explicit_unattested_rung_is_partitioned_out(monkeypatch):
    _admits_only_r1024(monkeypatch)
    admitted, refused = menu.partition_attested([ATTESTED, UNATTESTED, "NVFP4"])
    assert admitted == [ATTESTED]
    assert refused == [UNATTESTED]


def test_explicit_menu_report_counts_the_predicate_not_the_caller(monkeypatch):
    _admits_only_r1024(monkeypatch)
    kept = [ATTESTED, UNATTESTED]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED)
    assert widths["attested_rungs"] == 1
    assert widths["explicit_unattested_rungs"] == 1
    assert widths["menu_mode"] == menu.MENU_ATTESTED
    assert line.startswith("[alloc] Tessera menu: 1 of 2 priced rungs are attested by the pinned runtime")
    assert "1 explicitly named rungs are unattested" in line


def test_all_explicit_rungs_unattested_reads_zero_not_all(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_kw: False)
    kept = [ATTESTED, UNATTESTED, "TESSERA_BF16_K1_R896"]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED)
    assert widths["attested_rungs"] == 0 and refused == kept
    assert line.startswith("[alloc] Tessera menu: 0 of 3 priced rungs are attested")
    assert "sample" not in line


def test_research_mode_is_labelled_admitted_not_attested(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_kw: True)
    kept = [ATTESTED, UNATTESTED]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_RESEARCH)
    assert widths["attested_rungs"] == 0
    assert widths["research_admitted_rungs"] == 2
    assert "PRISMAQUANT_TESSERA_MENU=research" in line
    assert "attested by the pinned runtime" not in line.split("(not attested")[0]
