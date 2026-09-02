from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from tools.full_kl_teacher_payload import (
    N_SAMPLES,
    PROMPT_TOP_K,
    SEQLEN,
    TEACHER_PAYLOAD_SCHEMA,
    TOPK_MINIMUM_COVERAGE,
    TeacherPayloadError,
    _octave_position_blocks,
    atomic_json_write,
    atomic_torch_save,
    build_calibration_contract,
    canonical_sha256,
    compact_source_model_identity,
    format_forward_fidelity_profile,
    load_teacher_evidence,
    payload_semantic_sha256,
    safe_load_torch_payload,
    student_t_upper_tail,
    teacher_forward_fidelity_policy,
    teacher_forward_fidelity_summary,
    teacher_forward_nll_per_position,
    teacher_meta,
    tensor_descriptor,
    topk_coverage_policy,
    topk_coverage_summary,
    tokenizer_identity,
    validate_teacher_payload,
)
from tools.build_streamed_full_kl_teacher import (
    _build_payload,
    _tokenizer_vocab_size,
    main as build_main,
)
from tools.measure_vllm_full_kl import _student


def _execute_marker(path: str) -> dict:
    Path(path).write_text("unsafe pickle executed")
    return {}


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _execute_marker, (str(self.marker),)


def _source_identity(tmp_path: Path) -> dict:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"source weights")
    config = {"model_type": "deepseek_v4"}
    weight_map = {"model.layers.0.weight": "layers.0.weight"}
    checkpoint_map = {"layers.0.weight": shard.name}
    shards = [{
        "path": str(shard.resolve()),
        "size": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }]
    value_bearing = {
        "config": config,
        "weight_map": weight_map,
        "shards": shards,
        "checkpoint_weight_map": checkpoint_map,
    }
    return {
        "schema": "prismaquant.streamed_model.identity.v1",
        "source": str(tmp_path),
        # A local source has no Hugging Face resolved commit.  This is the
        # actual DSv4 campaign shape and must remain legal.
        "resolved_commit": None,
        "content_sha256": canonical_json_sha256(
            value_bearing, where="test identity"
        ),
        **value_bearing,
    }


def _payload(tmp_path: Path) -> dict:
    torch.manual_seed(5)
    calib = torch.randint(0, 3000, (N_SAMPLES, SEQLEN), dtype=torch.long)
    starts = list(range(10, 10 + N_SAMPLES))
    tokenizer = {
        "schema": "prismaquant.tokenizer_identity/1",
        "content_sha256": "a" * 64,
        "files": {"tokenizer.json": {"bytes": 1, "sha256": "b" * 64}},
    }
    contract = build_calibration_contract(
        dataset_fingerprint="dataset-fingerprint",
        corpus_sha256="c" * 64,
        tokenizer=tokenizer,
        starts=starts,
        total_tokens=100_000,
        calib_ids=calib,
    )
    identity = _source_identity(tmp_path)
    ids = torch.arange(PROMPT_TOP_K, dtype=torch.int32).reshape(1, 1, -1)
    ids = ids.expand(N_SAMPLES, SEQLEN - 1, -1).contiguous()
    # Strictly decreasing top-K support carrying 99% of the probability mass.
    logits = torch.linspace(4.0, -4.0, PROMPT_TOP_K, dtype=torch.float64)
    lps = (torch.log_softmax(logits, dim=0) + math.log(0.99)).to(torch.float32)
    lps = lps.reshape(1, 1, -1).expand_as(ids).contiguous()
    payload = {
        "schema": TEACHER_PAYLOAD_SCHEMA,
        "score_positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "topk_ids": ids,
        "topk_lps": lps,
        "calib_ids": calib,
        "starts": starts,
        "model": str(tmp_path),
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        # Must exceed the top-K support, so derive it rather than hardcoding a
        # number that silently becomes invalid when PROMPT_TOP_K moves.
        "vocab_size": PROMPT_TOP_K * 4,
        "source_model_identity": identity,
        "source_model": compact_source_model_identity(identity),
        "source_model_identity_sha256": canonical_sha256(identity),
        "calibration_contract": contract,
        "calibration_contract_sha256": canonical_sha256(contract),
    }
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    return payload


def test_teacher_payload_roundtrip_binds_null_commit_and_tensor_bytes(tmp_path):
    payload = _payload(tmp_path)
    assert validate_teacher_payload(payload)["source_model"][
        "resolved_commit"
    ] is None
    descriptor = tensor_descriptor(payload["topk_lps"])
    assert descriptor["dtype"] == "float32"
    assert descriptor["shape"] == [N_SAMPLES, SEQLEN - 1, PROMPT_TOP_K]
    assert len(descriptor["sha256"]) == 64
    coverage = topk_coverage_summary(
        payload["topk_ids"],
        payload["topk_lps"],
        vocab_size=payload["vocab_size"],
    )
    assert coverage["topk_coverage_mean"] == pytest.approx(0.99, abs=1e-7)
    assert coverage["topk_coverage_min"] == pytest.approx(0.99, abs=1e-7)
    assert coverage["topk_coverage_policy"] == topk_coverage_policy()
    assert (
        coverage["topk_coverage_policy"][
            "minimum_probability_mass_per_position"
        ]
        == TOPK_MINIMUM_COVERAGE
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value["topk_lps"].__setitem__((0, 0, -1), -99.0),
         "semantic payload digest"),
        (lambda value: value["calib_ids"].__setitem__((0, 0), 7),
         "calib_ids bytes"),
        (lambda value: value["source_model"].__setitem__(
            "checkpoint_shards", 2), "compact source identity"),
        (lambda value: value.__setitem__("prompt_top_k", 999),
         "scoring dimensions"),
        (lambda value: value.__setitem__("extra", True), "fields are not closed"),
    ],
)
def test_teacher_payload_refuses_semantic_forgery(tmp_path, mutation, error):
    payload = copy.deepcopy(_payload(tmp_path))
    mutation(payload)
    with pytest.raises(TeacherPayloadError, match=error):
        validate_teacher_payload(payload)


def _duplicate_topk_id(payload):
    payload["topk_ids"][0, 0, 1] = payload["topk_ids"][0, 0, 0]


def _unsort_topk_logprobs(payload):
    payload["topk_lps"][0, 0, 1] = payload["topk_lps"][0, 0, 0] + 0.01


def _make_topk_logprob_nonfinite(payload):
    payload["topk_lps"][0, 0, 0] = float("nan")


def _make_topk_id_out_of_range(payload):
    payload["topk_ids"][0, 0, 0] = payload["vocab_size"]


def _over_normalize_topk_mass(payload):
    payload["topk_lps"].fill_(math.log(1.001 / PROMPT_TOP_K))


def _drop_topk_coverage(payload):
    payload["topk_lps"].fill_(math.log(0.50 / PROMPT_TOP_K))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (_duplicate_topk_id, "duplicate token ids"),
        (_unsort_topk_logprobs, "not nonincreasing"),
        (_make_topk_logprob_nonfinite, "non-finite"),
        (_make_topk_id_out_of_range, "out of range"),
        (_over_normalize_topk_mass, "probability mass exceeds one"),
        (_drop_topk_coverage, "coverage falls below"),
    ],
)
def test_teacher_payload_refuses_resigned_topk_semantic_tamper(
    tmp_path, mutation, error
):
    payload = _payload(tmp_path)
    mutation(payload)
    # Model an attacker who can recompute the unkeyed semantic digest.  The
    # row-level probability contract must still reject the tensor values.
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    with pytest.raises(TeacherPayloadError, match=error):
        validate_teacher_payload(payload)


def test_serialized_payload_and_meta_are_replayed_into_compact_evidence(tmp_path):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(
        payload_path=payload_path,
        elapsed_s=12.5,
    )
    atomic_json_write(meta, meta_path)

    loaded, evidence = load_teacher_evidence(payload_path, meta_path)
    assert loaded["payload_semantic_sha256"] == payload["payload_semantic_sha256"]
    assert evidence["schema"] == "prismaquant.full_kl_teacher_evidence/1"
    assert evidence["meta_sha256"] == hashlib.sha256(meta_path.read_bytes()).hexdigest()
    assert evidence["source_model"] == payload["source_model"]
    assert evidence["topk_coverage_mean"] == meta["topk_coverage_mean"]
    assert evidence["topk_coverage_min"] == meta["topk_coverage_min"]
    assert evidence["topk_coverage_policy"] == topk_coverage_policy()

    with payload_path.open("ab") as handle:
        handle.write(b"forgery")
    with pytest.raises(TeacherPayloadError, match="serialized payload bytes"):
        load_teacher_evidence(payload_path, meta_path)


def test_meta_refuses_tensor_descriptor_forgery(tmp_path):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(
        payload_path=payload_path,
        elapsed_s=1.0,
    )
    meta["tensor_descriptors"]["topk_ids"]["sha256"] = "f" * 64
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


@pytest.mark.parametrize("field", ["topk_coverage_mean", "topk_coverage_min"])
def test_meta_refuses_forged_coverage_recomputed_from_tensor_bytes(
    tmp_path, field
):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(payload_path=payload_path, elapsed_s=1.0)
    meta[field] = float(meta[field]) - 0.01
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


def test_meta_refuses_forged_coverage_policy(tmp_path):
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(_payload(tmp_path), payload_path)
    meta = teacher_meta(payload_path=payload_path, elapsed_s=1.0)
    meta["topk_coverage_policy"][
        "minimum_probability_mass_per_position"
    ] = 0.0
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


def test_weights_only_teacher_load_never_executes_reduce(tmp_path):
    marker = tmp_path / "pickle-executed"
    payload_path = tmp_path / "malicious.pt"
    meta_path = tmp_path / "malicious.json"
    torch.save(_MaliciousPayload(marker), payload_path)
    meta_path.write_text("{}")

    with pytest.raises(TeacherPayloadError, match="safely load"):
        safe_load_torch_payload(payload_path)
    assert not marker.exists()

    with pytest.raises(TeacherPayloadError, match="safely load"):
        load_teacher_evidence(payload_path, meta_path)
    assert not marker.exists()

    # The legacy/no-sidecar student entry point uses the same restricted load.
    args = SimpleNamespace(
        teacher_meta=None,
        teacher_payload=str(payload_path),
    )
    with pytest.raises(TeacherPayloadError, match="safely load"):
        _student(args)
    assert not marker.exists()


def test_tokenizer_identity_hashes_only_present_contract_files(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text('{"legacy":true}')
    (tmp_path / "unrelated.bin").write_bytes(b"not tokenizer state")
    identity = tokenizer_identity(tmp_path)
    assert set(identity["files"]) == {"tokenizer.json", "tokenizer_config.json"}
    assert identity["content_sha256"] == canonical_sha256({
        "files": identity["files"]
    })


def test_atomic_publish_refuses_preexisting_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "teacher.json"
    temporary = tmp_path / f".{target.name}.tmp.123"
    temporary.write_text("do not overwrite")
    monkeypatch.setattr("tools.full_kl_teacher_payload.os.getpid", lambda: 123)
    with pytest.raises(TeacherPayloadError, match="temporary metadata"):
        atomic_json_write({"ok": True}, target)
    assert temporary.read_text() == "do not overwrite"
    assert not target.exists()


def test_teacher_uses_added_token_vocabulary_cardinality():
    class Tokenizer:
        vocab_size = 128_000

        def __len__(self):
            return 129_280

    assert _tokenizer_vocab_size(Tokenizer()) == 129_280


# ---------------------------------------------------------------------------
# Forward-fidelity (context-monotonicity) gate
# ---------------------------------------------------------------------------


def _row_probabilities(
    target_probability: torch.Tensor,
    *,
    target_index: int,
    support: int,
    total_mass: float,
) -> torch.Tensor:
    """Build sorted-descending rows whose target entry carries a given mass.

    ``target_index`` is either the head (0) or the last supported rank, which
    is enough to place a target anywhere between "most likely token" and
    "barely inside the support" without inventing a general sorter.
    """
    q = target_probability.to(dtype=torch.float64).unsqueeze(-1)
    remainder = float(total_mass) - q
    if target_index == 0:
        weights = torch.linspace(
            1.0, 1.0 / (support - 1), support - 1, dtype=torch.float64
        )
        rest = weights / weights.sum() * remainder
        if float(rest[..., 0].max()) > float(q.min()):
            raise AssertionError("head target is not the largest probability")
        return torch.cat([q, rest], dim=-1)
    if target_index != support - 1:
        raise AssertionError("target_index must be the head or the last rank")
    weights = torch.linspace(1.0, 0.5, support - 1, dtype=torch.float64)
    rest = weights / weights.sum() * remainder
    if float(rest[..., -1].min()) < float(q.max()):
        raise AssertionError("tail target is not the smallest probability")
    return torch.cat([rest, q], dim=-1)


def _nll_shape(profile: list[tuple[int, float]], *, jitter: float) -> torch.Tensor:
    """Expand ``(context_upper_bound, nll)`` steps over the scored grid."""
    generator = torch.Generator().manual_seed(11)
    nll = torch.empty((N_SAMPLES, SEQLEN - 1), dtype=torch.float64)
    for index in range(SEQLEN - 1):
        context = index + 1
        value = next(
            level for bound, level in profile if context <= bound
        )
        nll[:, index] = value
    if jitter:
        nll += (
            torch.rand(nll.shape, generator=generator, dtype=torch.float64) - 0.5
        ) * (2.0 * jitter)
    return nll


def _shaped_payload(
    tmp_path: Path,
    nll: torch.Tensor,
    *,
    target_index: int = 0,
    total_mass: float = 0.99,
) -> dict:
    """A structurally valid payload whose per-position NLL is prescribed."""
    payload = _payload(tmp_path)
    support = PROMPT_TOP_K
    vocab_size = int(payload["vocab_size"])
    target_token = 0 if target_index == 0 else support - 1

    calib = payload["calib_ids"].clone()
    calib[:, 1:] = target_token
    ids = torch.arange(support, dtype=torch.int32).reshape(1, 1, -1)
    ids = ids.expand(N_SAMPLES, SEQLEN - 1, -1).contiguous()
    lps = torch.empty((N_SAMPLES, SEQLEN - 1, support), dtype=torch.float32)
    for sample in range(N_SAMPLES):
        probabilities = _row_probabilities(
            torch.exp(-nll[sample]),
            target_index=target_index,
            support=support,
            total_mass=total_mass,
        )
        lps[sample] = probabilities.log().to(torch.float32)

    payload["calib_ids"] = calib
    payload["topk_ids"] = ids
    payload["topk_lps"] = lps
    contract = build_calibration_contract(
        dataset_fingerprint="dataset-fingerprint",
        corpus_sha256="c" * 64,
        tokenizer={
            "schema": "prismaquant.tokenizer_identity/1",
            "content_sha256": "a" * 64,
            "files": {"tokenizer.json": {"bytes": 1, "sha256": "b" * 64}},
        },
        starts=payload["starts"],
        total_tokens=100_000,
        calib_ids=calib,
    )
    payload["calibration_contract"] = contract
    payload["calibration_contract_sha256"] = canonical_sha256(contract)
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    assert vocab_size > support
    return payload


# The 2026-08-16 DSv4 streamed teacher, whose own perplexity was 262 while the
# 2.34-bpp student it graded shipped at 9.05. Derived from the measured
# profile, expressed as (context upper bound, NLL) steps.
_DEGRADING_PROFILE = [
    (32, math.log(24.5)),
    (64, math.log(7.6)),
    (128, math.log(34.8)),
    (256, math.log(475.7)),
    (384, math.log(492.4)),
    (SEQLEN, math.log(1013.2)),
]
_IMPROVING_PROFILE = [
    (32, math.log(24.5)),
    (64, math.log(12.0)),
    (128, math.log(9.0)),
    (256, math.log(7.5)),
    (384, math.log(7.0)),
    (SEQLEN, math.log(6.8)),
]
_FLAT_PROFILE = [(SEQLEN, math.log(7.6))]


def test_forward_fidelity_refuses_a_context_degrading_teacher(tmp_path):
    payload = _shaped_payload(
        tmp_path, _nll_shape(_DEGRADING_PROFILE, jitter=0.2)
    )
    with pytest.raises(TeacherPayloadError) as excinfo:
        validate_teacher_payload(payload)
    message = str(excinfo.value)
    assert "not context-monotone" in message
    # The refusal must carry the profile that made this diagnosable, not just
    # the verdict: every context octave, with its NLL and perplexity.
    assert "per-position teacher-forced NLL by context octave" in message
    for octave in ("1-1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127",
                   "128-255", "256-511"):
        first, last = octave.split("-")
        assert f"{int(first):>5d}-{int(last):<5d}" in message
    assert "significant regressions:" in message
    assert "welch_t=" in message and "df=" in message and "p=" in message
    # The healthy 32-64 octave against the blown-out tail must be among them.
    assert "32-63 NLL" in message


def test_forward_fidelity_refuses_zero_variance_degradation(tmp_path):
    # Constant NLL inside each octave leaves the Welch denominator at zero for
    # blocks that fall entirely within one step; exact separation must refuse
    # rather than divide by zero.
    payload = _shaped_payload(
        tmp_path, _nll_shape(_DEGRADING_PROFILE, jitter=0.0)
    )
    with pytest.raises(TeacherPayloadError, match="not context-monotone"):
        validate_teacher_payload(payload)


@pytest.mark.parametrize(
    ("name", "profile"),
    [("flat", _FLAT_PROFILE), ("improving", _IMPROVING_PROFILE)],
)
def test_forward_fidelity_accepts_flat_and_improving_teachers(
    tmp_path, name, profile
):
    payload = _shaped_payload(tmp_path, _nll_shape(profile, jitter=0.2))
    assert validate_teacher_payload(payload)
    summary = teacher_forward_fidelity_summary(
        payload["topk_ids"],
        payload["topk_lps"],
        payload["calib_ids"],
        vocab_size=int(payload["vocab_size"]),
    )
    assert summary["scored_positions"] == N_SAMPLES * (SEQLEN - 1)
    assert summary["out_of_support_targets"] == 0
    assert summary["perplexity"] == pytest.approx(
        math.exp(summary["nll_mean"])
    )
    if name == "improving":
        assert summary["blocks"][-1]["nll_mean"] < summary["blocks"][0]["nll_mean"]


def test_forward_fidelity_reports_the_profile_on_success(tmp_path, capsys):
    payload = _shaped_payload(tmp_path, _nll_shape(_FLAT_PROFILE, jitter=0.1))
    summary = teacher_forward_fidelity_summary(
        payload["topk_ids"],
        payload["topk_lps"],
        payload["calib_ids"],
        vocab_size=int(payload["vocab_size"]),
    )
    print(format_forward_fidelity_profile(summary))
    text = capsys.readouterr().out
    assert "per-position teacher-forced NLL by context octave" in text
    assert "uniform-vocabulary ceiling" in text
    assert "worst context-monotonicity comparison" in text
    for block in summary["blocks"]:
        assert block["perplexity"] == pytest.approx(7.6, rel=0.05)


def test_forward_fidelity_refuses_a_teacher_no_better_than_uniform(tmp_path):
    vocabulary_nll = math.log(PROMPT_TOP_K * 4)
    nll = torch.full(
        (N_SAMPLES, SEQLEN - 1), vocabulary_nll + 1.0, dtype=torch.float64
    )
    payload = _shaped_payload(tmp_path, nll, target_index=PROMPT_TOP_K - 1)
    with pytest.raises(TeacherPayloadError) as excinfo:
        validate_teacher_payload(payload)
    message = str(excinfo.value)
    assert "no more informative than the uniform distribution" in message
    assert f"ln(V)={vocabulary_nll:.4f}" in message
    assert "per-position teacher-forced NLL by context octave" in message


def test_per_position_nll_bounds_out_of_support_targets():
    # Two scored positions over a four-token support: the first target is in
    # support, the second is not.
    support, vocab_size = 4, 64
    probabilities = torch.tensor(
        [[0.50, 0.30, 0.10, 0.05], [0.40, 0.30, 0.10, 0.05]],
        dtype=torch.float64,
    )
    topk_lps = probabilities.log().to(torch.float32).reshape(1, 2, support)
    topk_ids = torch.arange(support, dtype=torch.int32).reshape(1, 1, -1)
    topk_ids = topk_ids.expand(1, 2, support).contiguous()
    calib_ids = torch.tensor([[9, 1, 63]], dtype=torch.long)

    nll, missing = teacher_forward_nll_per_position(
        topk_ids, topk_lps, calib_ids, vocab_size=vocab_size
    )
    assert missing.tolist() == [[False, True]]
    assert float(nll[0, 0]) == pytest.approx(-math.log(0.30), abs=1e-6)
    # Out of support: bounded by min(tail mass, smallest in-support mass),
    # here min(1 - 0.85, 0.05) = 0.05, floored by K * eps(float32).
    floor = support * float(torch.finfo(torch.float32).eps)
    assert float(nll[0, 1]) == pytest.approx(
        -math.log(max(0.05, floor)), abs=1e-6
    )
    # The imputation is a lower bound on the true NLL, never an inflation.
    assert float(nll[0, 1]) <= -math.log(floor) + 1e-9


def test_octave_partition_is_derived_from_the_scored_grid():
    blocks = _octave_position_blocks(SEQLEN - 1)
    assert len(blocks) == math.floor(math.log2(SEQLEN - 1)) + 1
    assert blocks[0] == (0, 1)
    assert blocks[-1] == (255, SEQLEN - 1)
    # Contiguous, non-overlapping, exhaustive.
    assert blocks[0][0] == 0
    assert all(
        later[0] == earlier[1] for earlier, later in zip(blocks, blocks[1:])
    )
    assert sum(last - first for first, last in blocks) == SEQLEN - 1


def test_forward_fidelity_alpha_is_derived_from_the_payload_shape():
    policy = teacher_forward_fidelity_policy(
        scored_positions=N_SAMPLES * (SEQLEN - 1), comparisons=36
    )
    assert policy["family_wise_alpha"] == 1.0 / (N_SAMPLES * (SEQLEN - 1))
    assert policy["per_comparison_alpha"] == pytest.approx(
        policy["family_wise_alpha"] / 36
    )
    assert policy["absolute_ceiling"] == "ln(vocab_size)"


def test_forward_fidelity_verdict_is_insensitive_to_the_alpha_convention(
    tmp_path,
):
    payload = _shaped_payload(
        tmp_path, _nll_shape(_DEGRADING_PROFILE, jitter=0.2)
    )
    summary_error = None
    try:
        teacher_forward_fidelity_summary(
            payload["topk_ids"],
            payload["topk_lps"],
            payload["calib_ids"],
            vocab_size=int(payload["vocab_size"]),
        )
    except TeacherPayloadError as exc:
        summary_error = exc
    assert summary_error is not None
    # Re-derive the worst comparison independently: the verdict must survive an
    # alpha many decades tighter than the one the policy derives.
    nll, _ = teacher_forward_nll_per_position(
        payload["topk_ids"],
        payload["topk_lps"],
        payload["calib_ids"],
        vocab_size=int(payload["vocab_size"]),
    )
    healthy = nll[:, 31:63].reshape(-1)
    degraded = nll[:, 383:].reshape(-1)
    spread = (
        float(healthy.var(unbiased=True)) / healthy.numel()
        + float(degraded.var(unbiased=True)) / degraded.numel()
    )
    statistic = (float(degraded.mean()) - float(healthy.mean())) / math.sqrt(
        spread
    )
    assert statistic > 20.0
    assert student_t_upper_tail(statistic, 1000.0) < 1e-12


@pytest.mark.parametrize(
    ("statistic", "degrees_of_freedom"),
    [
        (0.0, 7.0), (1.0, 7.0), (3.9, 7.0), (7.88, 7.0), (-2.5, 7.0),
        (0.5, 1.0), (2.0, 2.5), (4.3, 35.0), (12.0, 1023.0), (24.0, 4000.0),
    ],
)
def test_student_t_upper_tail_matches_scipy(statistic, degrees_of_freedom):
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = float(scipy_stats.t.sf(statistic, degrees_of_freedom))
    observed = student_t_upper_tail(statistic, degrees_of_freedom)
    assert observed == pytest.approx(expected, rel=1e-9, abs=1e-300)


def test_forward_fidelity_gate_runs_in_the_build_path_before_the_write():
    """The acceptance criterion: the gate is on the real code path."""
    build_source = inspect.getsource(_build_payload)
    assert "validate_teacher_payload(payload)" in build_source
    assert "teacher_forward_fidelity_summary(" in build_source
    assert "format_forward_fidelity_profile(" in build_source
    # validate_teacher_payload is what refuses; it must precede the return, and
    # the caller writes only after _build_payload returns.
    assert build_source.index("validate_teacher_payload(payload)") < (
        build_source.rindex("return payload")
    )
    main_source = inspect.getsource(build_main)
    assert main_source.index("_build_payload(args)") < main_source.index(
        "atomic_torch_save("
    )


def test_meta_and_evidence_replay_refuse_an_unfaithful_written_payload(
    tmp_path, monkeypatch
):
    """A payload that slipped past the gate cannot be replayed into evidence."""
    payload = _shaped_payload(
        tmp_path, _nll_shape(_DEGRADING_PROFILE, jitter=0.2)
    )
    payload_path = tmp_path / "teacher.pt"
    atomic_torch_save(payload, payload_path)
    with pytest.raises(TeacherPayloadError, match="not context-monotone"):
        teacher_meta(payload_path=payload_path, elapsed_s=1.0)


def test_a_refused_teacher_is_never_written_to_disk(tmp_path, monkeypatch):
    """main() writes only what _build_payload returned; a refusal returns nothing."""
    import tools.build_streamed_full_kl_teacher as builder

    payload = _shaped_payload(
        tmp_path, _nll_shape(_DEGRADING_PROFILE, jitter=0.2)
    )
    output = tmp_path / "teacher.pt"
    meta_output = tmp_path / "teacher.json"
    written: list[Path] = []

    def _gated_build(_args):
        # The GPU prefix of the real _build_payload cannot run here; what is
        # under test is that its validation tail refuses and propagates.
        return validate_teacher_payload(payload)

    monkeypatch.setattr(builder, "_build_payload", _gated_build)
    monkeypatch.setattr(
        builder,
        "atomic_torch_save",
        lambda *a, **k: written.append(Path(a[1])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_streamed_full_kl_teacher",
            "--model", str(tmp_path),
            "--identity-cache", str(tmp_path / "identity.json"),
            "--output", str(output),
            "--meta-output", str(meta_output),
            "--offload-folder", str(tmp_path / "offload"),
            "--wikitext-inputs", str(tmp_path / "inputs.json"),
        ],
    )
    with pytest.raises(TeacherPayloadError, match="not context-monotone"):
        build_main()
    assert written == []
    assert not output.exists()
    assert not meta_output.exists()
