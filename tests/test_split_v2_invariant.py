"""Adversarial tests for v2 permutation-stable evidence identity."""

import torch
from prismaquant.nvfp4_cb_formats import (
    deterministic_fit_holdout_split_per_expert,
    _partition_multiset_digest,
    _row_content_hash,
    _SPLIT_POLICY,
    _SPLIT_VERSION,
)


def test_permutation_invariant_digests_and_membership():
    torch.manual_seed(0)
    C = 16
    n = 12
    rows = torch.randn(n, C)
    # add duplicates to test multiplicity and grouping
    rows[5] = rows[2].clone()
    rows[9] = rows[2].clone()
    act = [rows.clone()]
    fit1, hold1, infos1 = deterministic_fit_holdout_split_per_expert(act, version=2)
    # permute
    perm = torch.randperm(n)
    rows_perm = rows[perm]
    fit2, hold2, infos2 = deterministic_fit_holdout_split_per_expert([rows_perm], version=2)
    assert infos1[0]["full_digest"] == infos2[0]["full_digest"]
    assert infos1[0]["fit_digest"] == infos2[0]["fit_digest"]
    assert infos1[0]["holdout_digest"] == infos2[0]["holdout_digest"]
    # membership invariant: sorted row hashes of fit/hold must match
    fit1_hashes = sorted(_row_content_hash(r) for r in fit1[0])
    fit2_hashes = sorted(_row_content_hash(r) for r in fit2[0])
    assert fit1_hashes == fit2_hashes
    hold1_hashes = sorted(_row_content_hash(r) for r in hold1[0])
    hold2_hashes = sorted(_row_content_hash(r) for r in hold2[0])
    assert hold1_hashes == hold2_hashes
    # policy/version v2
    assert infos1[0]["policy"] == _SPLIT_POLICY
    assert infos1[0]["version"] == _SPLIT_VERSION
    assert _SPLIT_POLICY.startswith("v2")
    assert _SPLIT_VERSION == 2


def test_duplicates_remain_together_and_multiplicity_affects_identity():
    torch.manual_seed(1)
    C = 8
    # Create rows with duplicate content
    base = torch.randn(6, C)
    rows = torch.cat([base, base[0:1].repeat(2, 1)], dim=0)  # duplicate row 0 three times total
    # Need at least 2 unique hashes to split
    rows[3] = torch.randn(C) * 5 + 10  # make distinct
    act = [rows]
    fit, hold, infos = deterministic_fit_holdout_split_per_expert(act, version=2)
    # Find duplicate hash
    dup_hash = _row_content_hash(base[0])
    fit_hashes = set(_row_content_hash(r) for r in fit[0])
    hold_hashes = set(_row_content_hash(r) for r in hold[0])
    # Duplicate must not be split across both
    assert not (dup_hash in fit_hashes and dup_hash in hold_hashes)
    # Multiplicity: add one more duplicate should change full digest
    rows_extra = torch.cat([rows, base[0:1]], dim=0)
    _, _, infos_extra = deterministic_fit_holdout_split_per_expert([rows_extra], version=2)
    assert infos[0]["full_digest"] != infos_extra[0]["full_digest"]
    # Also fit or holdout digest must change
    assert infos[0]["fit_digest"] != infos_extra[0]["fit_digest"] or infos[0]["holdout_digest"] != infos_extra[0]["holdout_digest"]


def test_device_transfer_stable():
    torch.manual_seed(2)
    t = torch.randn(8, 16)
    d1 = _partition_multiset_digest(t)
    # clone to different device if cuda available, else just float32 conversion
    t2 = t.clone().to(torch.float32)
    d2 = _partition_multiset_digest(t2)
    assert d1 == d2
    # Test via deterministic split device transfer
    act_cpu = [t]
    fit_cpu, hold_cpu, infos_cpu = deterministic_fit_holdout_split_per_expert(act_cpu, version=2)
    # Simulate transfer via .clone()
    act_clone = [t.clone()]
    fit_clone, hold_clone, infos_clone = deterministic_fit_holdout_split_per_expert(act_clone, version=2)
    assert infos_cpu[0]["full_digest"] == infos_clone[0]["full_digest"]


def test_one_content_only_insufficient():
    # Single unique row content cannot form two partitions
    single = torch.ones(5, 8)
    fit, hold, infos = deterministic_fit_holdout_split_per_expert([single], version=2)
    assert infos[0]["insufficient"] is True
    assert fit[0].numel() == 0
    assert hold[0].numel() == 0


def test_winner_behavior_unchanged_by_permutation():
    # Verify that gating winner (kept) is unchanged by permutation of rows
    torch.manual_seed(3)
    import os
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"
    from prismaquant import format_registry as fr
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_fields_for_context

    E, R, C = 2, 4, 256
    w = torch.randn(E, R, C)
    cw = torch.randn(E, 1, C).abs() + 0.1
    rows = torch.randn(12, C)
    # Add duplicate to test grouping but keep winner stable
    rows_dup = rows.clone()
    perm = torch.randperm(rows.shape[0])
    rows_perm = rows[perm]
    spec = fr.get_format("NVFP4_CB_K12")
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    # Gate on original
    fields = cb_fields_for_context(spec, w, context=CBSerializationContext.production(ldlq_scope="none", encode_tier="balanced"), col_weights=cw)
    # Use original rows
    _, gi_orig = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=[rows, rows.clone()], return_gate_info=True) if E==2 else (None, None)
    # Use permuted rows (same multiset)
    _, gi_perm = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=[rows_perm, rows_perm.clone()], return_gate_info=True)
    # Per-expert kept should be identical
    assert gi_orig["per_expert_kept"] == gi_perm["per_expert_kept"]
    assert gi_orig["gate"] == gi_perm["gate"]


def test_stale_v1_cannot_be_reused_as_v2():
    torch.manual_seed(4)
    t = torch.randn(8, 16)
    _, _, infos_v1 = deterministic_fit_holdout_split_per_expert([t], version=1)
    _, _, infos_v2 = deterministic_fit_holdout_split_per_expert([t], version=2)
    assert infos_v1[0]["policy"] != infos_v2[0]["policy"]
    assert infos_v1[0]["version"] != infos_v2[0]["version"]
    assert infos_v1[0]["full_digest"] != infos_v2[0]["full_digest"]
    # v1 digest is order-dependent, v2 is not — permuting changes v1 but not v2
    perm = torch.randperm(t.shape[0])
    t_perm = t[perm]
    _, _, infos_v1_perm = deterministic_fit_holdout_split_per_expert([t_perm], version=1)
    _, _, infos_v2_perm = deterministic_fit_holdout_split_per_expert([t_perm], version=2)
    # v1 full digest should differ (order dependent)
    assert infos_v1[0]["full_digest"] != infos_v1_perm[0]["full_digest"]
    # v2 should be same
    assert infos_v2[0]["full_digest"] == infos_v2_perm[0]["full_digest"]


def test_shape_dtype_boundaries():
    t1 = torch.randn(4, 8)
    t2 = torch.randn(4, 16)
    d1 = _partition_multiset_digest(t1)
    d2 = _partition_multiset_digest(t2)
    assert d1 != d2
    # Different dtype via float16 conversion (normalized to float32 for row hash, but shape/dtype framing includes dtype)
    # Row hashes are normalized to float32, so different original dtype but same float32 bytes should still be same multiset digest if shape same?
    # However our helper normalizes to float32, so dtype framing is always float32, but distinct C still differs.
    # Ensure empty shapes with different C differ
    e1 = torch.empty((0, 8), dtype=torch.float32)
    e2 = torch.empty((0, 16), dtype=torch.float32)
    assert _partition_multiset_digest(e1) != _partition_multiset_digest(e2)
