from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/"
    "glm_e2m1_analysis_receipt.py"
)
_SPEC = importlib.util.spec_from_file_location("glm_e2m1_analysis_receipt", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
M = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_PATH.parent))
sys.modules[_SPEC.name] = M
try:
    _SPEC.loader.exec_module(M)
finally:
    sys.path.pop(0)


def _arm(*, trellis_db: float, nvfp4_wsse: float, bpw: float) -> dict:
    return {
        "weighted_snr_db": trellis_db,
        "footprint": {"exact_bpw": bpw},
        "subset_split": {
            "1": {
                "columns": 8,
                "trellis_db": trellis_db,
                "nvfp4_db": 10.0,
                "nvfp4_wsse": nvfp4_wsse,
                "scalar_subgrid_oracle": {
                    "db": trellis_db - 1.0,
                    "coding_gain_db": 1.0,
                },
                "scalar_subgrid_shared": {
                    "db": trellis_db - 0.5,
                    "coding_gain_db": 0.5,
                },
            }
        },
    }


def _source(tmp_path: Path, *, plan: str) -> Path:
    per_tensor = {
        "dense-a": {
            "population": "dense",
            "weighted_energy": 1.0,
            "arms": {"tcq_v1@1.0": _arm(
                trellis_db=11.0, nvfp4_wsse=0.1, bpw=1.5
            )},
        },
        "routed-a": {
            "population": "routed",
            "weighted_energy": 2.0,
            "arms": {"tcq_v1@1.0": _arm(
                trellis_db=12.0, nvfp4_wsse=0.2, bpw=1.6
            )},
        },
    }
    receipt = {
        "schema": "trellis.e2m1_highrate.v3",
        "status": "ok",
        "partial": False,
        "corpus": "glm",
        "glm_rate_plan": plan,
        "tensors_done": 2,
        "population_counts": {"dense": 1, "routed": 1},
        "publication_identity_sha256": "a" * 64,
        "active_source_identity": {"repo_git_commit": "b" * 40},
        "environment": {"repo_git_commit": "b" * 40},
    }
    body = {"receipt": receipt, "per_tensor": per_tensor}
    document = {
        **body,
        "checkpoint_sha256": hashlib.sha256(
            M.canonical_json_bytes(body)
        ).hexdigest(),
    }
    path = tmp_path / f"{plan}.json"
    path.write_text(json.dumps(document))
    return path


def _analysis(kind: str, source: Path) -> dict:
    document, bound = M._bound_json_object(source)
    if kind == "coding_gain":
        return M._coding_gain_summary(bound.path, document)
    return M._near_four_summary(
        bound.path, document, source_sha256=bound.sha256
    )


def _v1_receipt(v2: dict) -> dict:
    source = copy.deepcopy(v2["source"])
    source.pop("glm_rate_plan")
    body = {
        "schema": "trellis.glm_e2m1_analysis_receipt.v1",
        "status": v2["status"],
        "kind": v2["kind"],
        "source": source,
        "analysis": copy.deepcopy(v2["analysis"]),
        "population_counts": copy.deepcopy(v2["population_counts"]),
        "aggregation_contract": v2["aggregation_contract"],
        "verifier": copy.deepcopy(v2["verifier"]),
    }
    return {**body, "receipt_sha256": M._identity_sha256(body)}


@pytest.mark.parametrize(
    ("kind", "plan", "builder"),
    [
        ("coding_gain", "scaffold", M._coding_gain_summary),
        ("near_four", "high", M._near_four_summary),
    ],
)
def test_exact_analysis_recomputation_is_self_sealed(
    tmp_path: Path, kind: str, plan: str, builder
):
    source = _source(tmp_path, plan=plan).resolve()
    analysis = tmp_path / f"{kind}.json"
    analysis.write_text(json.dumps(_analysis(kind, source)))

    receipt = M.build_receipt(
        kind=kind, source_path=source, analysis_path=analysis
    )
    M.validate_receipt_self_digest(receipt)
    assert receipt["source"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert receipt["analysis"]["sha256"] == hashlib.sha256(
        analysis.read_bytes()
    ).hexdigest()
    assert receipt["source"]["glm_rate_plan"] == plan
    assert receipt["population_counts"] == {"dense": 1, "routed": 1}
    assert set(receipt["population_counts"]) == {"dense", "routed"}
    assert receipt["aggregation_contract"].endswith("no pooled median")

    tampered = copy.deepcopy(receipt)
    tampered["analysis"]["sha256"] = "f" * 64
    with pytest.raises(M.AnalysisReceiptError, match="self-digest differs"):
        M.validate_receipt_self_digest(tampered)


def test_analysis_or_source_mutation_refuses(tmp_path: Path):
    source = _source(tmp_path, plan="high").resolve()
    analysis = tmp_path / "near-four.json"
    expected = _analysis("near_four", source)
    analysis.write_text(json.dumps(expected))

    changed = copy.deepcopy(expected)
    changed["populations"]["dense"]["rows"][0][
        "paired_trellis_minus_scalar_db_median"
    ] += 0.01
    analysis.write_text(json.dumps(changed))
    with pytest.raises(M.AnalysisReceiptError, match="exact source recomputation"):
        M.build_receipt(
            kind="near_four", source_path=source, analysis_path=analysis
        )

    analysis.write_text(json.dumps(expected))
    mutated_source = json.loads(source.read_text())
    mutated_source["per_tensor"]["dense-a"]["weighted_energy"] = 3.0
    source.write_text(json.dumps(mutated_source))
    with pytest.raises(M.AnalysisReceiptError, match="self-digest differs"):
        M.build_receipt(
            kind="near_four", source_path=source, analysis_path=analysis
        )


def test_no_replace_receipt_publication(tmp_path: Path):
    source = _source(tmp_path, plan="high").resolve()
    analysis = tmp_path / "near-four.json"
    analysis.write_text(json.dumps(_analysis("near_four", source)))
    receipt = M.build_receipt(
        kind="near_four", source_path=source, analysis_path=analysis
    )
    output = tmp_path / "receipt.json"
    M.publish_receipt(output, receipt)
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(M.AnalysisReceiptError, match="already exists"):
        M.publish_receipt(output, receipt)


def test_v2_receipt_explicitly_supersedes_matching_v1(tmp_path: Path):
    source = _source(tmp_path, plan="high").resolve()
    analysis = tmp_path / "near-four.json"
    analysis.write_text(json.dumps(_analysis("near_four", source)))
    initial = M.build_receipt(
        kind="near_four", source_path=source, analysis_path=analysis
    )
    old_path = tmp_path / "receipt-v1.json"
    old_path.write_text(json.dumps(_v1_receipt(initial)))

    receipt = M.build_receipt(
        kind="near_four",
        source_path=source,
        analysis_path=analysis,
        supersedes_path=old_path,
    )
    assert receipt["supersedes"]["path"] == str(old_path)
    assert receipt["supersedes"]["schema"].endswith(".v1")
    assert receipt["supersedes"]["reason"] == M._SUPERSESSION_REASON
    M.validate_receipt_self_digest(receipt)

    old = json.loads(old_path.read_text())
    old["analysis"]["sha256"] = "e" * 64
    body = {key: value for key, value in old.items()
            if key != "receipt_sha256"}
    old["receipt_sha256"] = M._identity_sha256(body)
    old_path.write_text(json.dumps(old))
    with pytest.raises(M.AnalysisReceiptError, match="different inputs"):
        M.build_receipt(
            kind="near_four",
            source_path=source,
            analysis_path=analysis,
            supersedes_path=old_path,
        )


@pytest.mark.parametrize(
    ("kind", "wrong_plan"),
    [("coding_gain", "high"), ("near_four", "scaffold")],
)
def test_analysis_kind_requires_exact_glm_plan(
    tmp_path: Path, kind: str, wrong_plan: str
):
    source = _source(tmp_path, plan=wrong_plan).resolve()
    analysis = tmp_path / f"{kind}.json"
    analysis.write_text(json.dumps(_analysis(kind, source)))
    expected = "scaffold" if kind == "coding_gain" else "high"
    with pytest.raises(
        M.AnalysisReceiptError,
        match=rf"final GLM E2M1 v3 {expected} result",
    ):
        M.build_receipt(
            kind=kind, source_path=source, analysis_path=analysis
        )


def test_resigned_wrong_plan_receipt_still_refuses(tmp_path: Path):
    source = _source(tmp_path, plan="high").resolve()
    analysis = tmp_path / "near-four.json"
    analysis.write_text(json.dumps(_analysis("near_four", source)))
    receipt = M.build_receipt(
        kind="near_four", source_path=source, analysis_path=analysis
    )
    receipt["source"]["glm_rate_plan"] = "scaffold"
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    receipt["receipt_sha256"] = M._identity_sha256(body)
    with pytest.raises(M.AnalysisReceiptError, match="semantic identity differs"):
        M.validate_receipt_self_digest(receipt)


def test_bound_json_refuses_symlink(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text('{"value":1}\n')
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(M.AnalysisReceiptError, match="cannot open bound file"):
        M._bound_json_object(link)


def test_bound_json_refuses_path_swap_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "bound.json"
    replacement = tmp_path / "replacement.json"
    path.write_text('{"value":"original"}\n')
    replacement.write_text('{"value":"replacement"}\n')
    real_read = M.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            replacement.replace(path)
        return real_read(descriptor, size)

    monkeypatch.setattr(M.os, "read", swapping_read)
    with pytest.raises(M.AnalysisReceiptError, match="changed during read"):
        M._bound_json_object(path)
