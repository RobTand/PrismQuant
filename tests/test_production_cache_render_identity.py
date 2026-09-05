"""Render-identity guard for the production cache directory (#146).

Resume admits a unit as done by file presence alone, so a ``--cache-dir``
resumed under a different ``--render-scope``, a different
``--include-qnames-file``, a different lever string or a different
calibration would silently mix units rendered under different conditions.
``fill_production_weight_cache`` now writes its render identity into
``render_identity.json`` on first use and refuses on mismatch. A missing
sidecar means a pre-guard directory: warn, do not refuse.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from prismaquant.production_weight_cache import fill_production_weight_cache


class _TinyTwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 32)
        self.l1 = nn.Linear(32, 32, bias=False)
        self.l2 = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[0, 0] = 1.0
            self.embed.weight[1, 1] = 2.0
            self.l1.weight.copy_(torch.eye(32))
            self.l2.weight.fill_(1.0)

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        x = self.l1(x)
        return self.l2(x)


def _fake_render_env(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    monkeypatch.setattr(
        enc,
        "_compute_nvfp4_joint_global",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, fmt, **_kwargs: weight.detach().to(torch.float32),
    )


def _fill(model, calib_ids, cache_dir, **kwargs):
    params = {
        "qnames": ["l1", "l2"],
        "formats": ["NVFP4"],
        "levers": {"gptq": False, "scale_sweep": False},
        "max_act_rows": 8,
        "cache_dir": cache_dir,
        "progress": False,
    }
    params.update(kwargs)
    return fill_production_weight_cache(model, calib_ids, **params)


def _sidecar(cache_dir):
    import prismaquant.production_weight_cache as pwc

    return cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME


def test_resume_refuses_different_render_scope(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="render_scope") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            render_assignment={"l1": "NVFP4"},
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_render_subset(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="rendered_pairs") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            render_qnames=["l1"],
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_hooked_enumeration(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="hooked_qnames") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            qnames=["l1", "l2", "lm_head"],
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_levers(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="levers") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            levers={"gptq": True, "scale_sweep": False},
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_calibration(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    _fill(_TinyTwoLinear(), torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    with pytest.raises(ValueError, match="calib_hash") as exc_info:
        _fill(
            _TinyTwoLinear(),
            torch.tensor([[2, 3]], dtype=torch.long),
            tmp_path,
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_with_identical_identity_reuses_shards(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    first = _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    second = _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    assert set(second.weights) == set(first.weights)
    assert not second.failed


def test_fresh_directory_writes_validated_sidecar(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    sidecar = _sidecar(tmp_path)
    assert sidecar.is_file()
    identity = pwc.validate_production_cache_render_identity(
        json.loads(sidecar.read_text())
    )
    assert identity["schema"] == pwc.RENDER_IDENTITY_SCHEMA
    assert identity["render_scope"] == "format-menu"
    assert identity["requested_formats"] == ["NVFP4"]
    assert identity["hooked_qnames_sha256"] == pwc._qname_set_sha256(
        ["l1", "l2"]
    )
    assert identity["rendered_pairs"] == ["l1|NVFP4", "l2|NVFP4"]


def test_preguard_directory_warns_and_adopts_identity(
    tmp_path, monkeypatch, capsys
):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    assert not _sidecar(tmp_path).exists()

    cache = _fill(model, torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    out = capsys.readouterr().out
    assert "no render_identity.json" in out
    assert "pre-guard directory" in out
    assert _sidecar(tmp_path).is_file()
    assert ("l1", "NVFP4") in cache.weights


def test_corrupt_sidecar_refuses_fail_closed(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    _sidecar(tmp_path).write_text('{"schema": "bogus"')

    with pytest.raises(ValueError, match="rebuild this directory"):
        _fill(_TinyTwoLinear(), calib_ids, tmp_path)


_V1_SIDECAR_KEYS = {
    "schema",
    "render_scope",
    "requested_formats",
    "levers",
    "render_mechanism_order",
    "calib_hash",
    "hooked_qnames_sha256",
    "hooked_qnames",
    "rendered_pairs",
    "max_act_rows",
}


def test_legacy_v1_sidecar_without_admission_field_still_resumes(
    tmp_path, monkeypatch
):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    raw = json.loads(_sidecar(tmp_path).read_text())
    # Hand-built v1: exactly the ten pre-admission fields, spelled out as a
    # literal, with none of the new keys. The hook digest is recomputed
    # through the hash helper rather than copied, so the dict is not a
    # round-trip of the writer's output.
    legacy = {
        "schema": pwc.RENDER_IDENTITY_SCHEMA,
        "render_scope": raw["render_scope"],
        "requested_formats": raw["requested_formats"],
        "levers": raw["levers"],
        "render_mechanism_order": raw["render_mechanism_order"],
        "calib_hash": raw["calib_hash"],
        "hooked_qnames_sha256": pwc._qname_set_sha256(["l1", "l2"]),
        "hooked_qnames": raw["hooked_qnames"],
        "rendered_pairs": raw["rendered_pairs"],
        "max_act_rows": raw["max_act_rows"],
    }
    assert set(legacy) == _V1_SIDECAR_KEYS
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY not in legacy
    pwc.validate_production_cache_render_identity(legacy)
    _sidecar(tmp_path).write_text(json.dumps(legacy, indent=2, sort_keys=True))

    second = _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    assert ("l1", "NVFP4") in second.weights
    assert ("l2", "NVFP4") in second.weights


def test_preguard_shards_record_trust_admission(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    shard_name = pwc._cache_weight_filename("l1", "NVFP4")
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / shard_name,
    )
    assert not _sidecar(tmp_path).exists()

    _fill(model, torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    raw = json.loads(_sidecar(tmp_path).read_text())
    assert raw[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] == {
        "admitted_on_trust": True,
        "trusted_shard_count": 1,
        "trusted_shards": [shard_name],
    }


def test_fresh_directory_carries_no_trust_admission(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    raw = json.loads(_sidecar(tmp_path).read_text())
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY not in raw
    assert set(raw) == _V1_SIDECAR_KEYS


def _mtp_section():
    import prismaquant.production_weight_cache as pwc

    return pwc.build_mtp_append_identity(
        max_act_rows=2,
        activation_source_hash="ab" * 16,
        activation_rows={"mtp.l1": 2},
        source_prefix="mtp",
        source_tensor_count=1,
        profile_name="test",
    )


def test_append_merge_preserves_trust_admission(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    _fill(model, torch.tensor([[0, 1]], dtype=torch.long), tmp_path)
    before = json.loads(_sidecar(tmp_path).read_text())
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY in before

    pwc._check_and_record_append_identity(
        tmp_path,
        pwc.MTP_APPEND_SIDECAR_KEY,
        _mtp_section(),
        progress=False,
    )

    after = json.loads(_sidecar(tmp_path).read_text())
    assert after[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] == before[
        pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY
    ]
    assert after[pwc.MTP_APPEND_SIDECAR_KEY]["profile_name"] == "test"


def test_base_adoption_preserves_trust_admission(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    # An append running first in a pre-guard directory records the admission
    # alongside its section; the base fill then adopts the append-only file.
    pwc._check_and_record_append_identity(
        tmp_path,
        pwc.MTP_APPEND_SIDECAR_KEY,
        _mtp_section(),
        progress=False,
    )
    before = json.loads(_sidecar(tmp_path).read_text())
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY in before
    assert set(before) - {
        pwc.MTP_APPEND_SIDECAR_KEY,
        pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY,
    } == set()

    _fill(model, torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    after = json.loads(_sidecar(tmp_path).read_text())
    assert after[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] == before[
        pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY
    ]
    assert after[pwc.MTP_APPEND_SIDECAR_KEY]["profile_name"] == "test"
    assert after["render_scope"] == "format-menu"


def test_trust_admission_causes_no_false_mismatch(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(model, calib_ids, tmp_path)
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY in json.loads(
        _sidecar(tmp_path).read_text()
    )
    guarded_bytes = _sidecar(tmp_path).read_bytes()

    # The caller computes its identity without the admission field; the
    # guarded directory must still resume cleanly, and the match path must
    # not rewrite (and launder) the sidecar.
    second = _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    assert ("l1", "NVFP4") in second.weights
    assert ("l2", "NVFP4") in second.weights
    assert _sidecar(tmp_path).read_bytes() == guarded_bytes


def test_malformed_trust_admission_refuses_fail_closed(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    raw = json.loads(_sidecar(tmp_path).read_text())
    raw[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] = {"admitted_on_trust": True}
    _sidecar(tmp_path).write_text(json.dumps(raw, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="rebuild this directory"):
        _fill(_TinyTwoLinear(), calib_ids, tmp_path)


# ------------------- the resolved activation-scale policy (#227)
#
# A directory's shards are the same bytes under either input-global-scale
# policy -- the policy only sets the STATIC activation scale G -- but the
# activation-aware render SCORES beside them are priced at that G, and the
# served oracle at G=6/amax and at G=448*6/amax are different quantizers.  So
# the policy is a render input like `nvfp4_scale_rule`, it is resolved once
# into the levers, and the identity carries it.

W4A4 = "TESSERA_E2M1_K2_R896"
POLICY_ENV = "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE"


def _scores(cache):
    return dict(cache.metadata["render_scores"]["records"])


def test_resume_refuses_a_changed_activation_scale_policy(
        tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    monkeypatch.setenv(POLICY_ENV, "0")
    cache = _fill(_TinyTwoLinear(), calib_ids, tmp_path, formats=[W4A4])
    priced = _scores(cache)
    assert priced, "the fill must record W4A4 render scores"
    assert all(
        record["input_global_scale"] == 6.0 / record["activation_max_abs"]
        for record in priced.values()
    )

    # Control: the unchanged policy resumes and keeps the same costs.
    resumed = _fill(_TinyTwoLinear(), calib_ids, tmp_path, formats=[W4A4])
    assert _scores(resumed) == priced

    # The pre-fix behaviour, stated first: this fill used to resume the
    # directory silently, keeping every cost priced at G=6/amax while
    # quantizing activations at G=448*6/amax.  It now refuses, naming the
    # field of the render identity that moved.
    monkeypatch.setenv(POLICY_ENV, "1")
    with pytest.raises(
            ValueError, match="nvfp4_input_global_scale_policy") as exc_info:
        _fill(_TinyTwoLinear(), calib_ids, tmp_path, formats=[W4A4])
    assert "rebuild this directory" in str(exc_info.value)

    from prismaquant.nvfp4_activation_contract import (
        ActivationScalePolicyMismatchError,
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )

    # A pre-guard directory (no sidecar) is admitted on TRUST by that gate, so
    # the retained scores are the line that has to hold -- and they do, by
    # name, comparing the G each cost was priced at against the G this policy
    # derives from the same recorded maximum.
    _sidecar(tmp_path).unlink()
    with pytest.raises(ActivationScalePolicyMismatchError) as exc_info:
        _fill(_TinyTwoLinear(), calib_ids, tmp_path, formats=[W4A4])
    message = str(exc_info.value)
    assert "l1" in message
    assert LEGACY_INPUT_GLOBAL_SCALE_POLICY in message
    assert FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY in message

    # And the provenance the two refusals read: the policy on the identity
    # sidecar and on each cost the fill priced.
    assert {
        record["input_global_scale_policy"] for record in priced.values()
    } == {LEGACY_INPUT_GLOBAL_SCALE_POLICY}
    stored = json.loads(_sidecar(tmp_path).read_text())
    assert stored["levers"]["nvfp4_input_global_scale_policy"] == (
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    )


def test_a_fresh_directory_under_the_full_range_policy_prices_at_that_policy(
        tmp_path, monkeypatch):
    """The other policy is not broken, it is bound: a fresh cache-dir under it
    prices every W4A4 row at 448*6/amax and says so (#205's positive control
    kept honest by #227's provenance)."""
    from prismaquant.nvfp4_activation_contract import (
        FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    monkeypatch.setenv(POLICY_ENV, "1")
    cache = _fill(_TinyTwoLinear(), calib_ids, tmp_path, formats=[W4A4])
    records = _scores(cache)
    assert records
    for record in records.values():
        assert record["input_global_scale_policy"] == (
            FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
        )
        assert record["input_global_scale"] == pytest.approx(
            448.0 * 6.0 / record["activation_max_abs"], rel=1e-6)
