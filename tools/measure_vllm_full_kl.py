#!/usr/bin/env python3
"""Measure full-vocab next-token KL between two vLLM-loadable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

_TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_ROOT))
from prismaquant_source_bootstrap import activate_prismaquant_source

activate_prismaquant_source()

try:  # package mode (`python -m tools.measure_vllm_full_kl`)
    from .dsv4_gridbook_contract import exact_llm_contract
    from .full_kl_teacher_payload import (
        EXPECTED_POSITIONS,
        load_teacher_evidence,
        safe_load_torch_payload,
    )
    from .serve_fingerprint import gold_producer_identity, self_manifest
    from .spec_decode_guard import refuse_if_spec_decode
except ImportError:  # script mode (`python /repo/tools/measure_vllm_full_kl.py`)
    from dsv4_gridbook_contract import exact_llm_contract  # type: ignore
    from full_kl_teacher_payload import (  # type: ignore
        EXPECTED_POSITIONS,
        load_teacher_evidence,
        safe_load_torch_payload,
    )
    from serve_fingerprint import (  # type: ignore
        gold_producer_identity,
        self_manifest,
    )
    from spec_decode_guard import refuse_if_spec_decode  # type: ignore

# vLLM / datasets / transformers are imported lazily inside the functions
# that need them so the position-KL math stays unit-testable in environments
# without a serving stack.

#: Set by `_load_llm` once the engine exists; stamped on every result dict.
#: `None` means "could not inspect", which the shipcard refuses — an unverified
#: negative is exactly what the draft-logprobs trap looked like.
_SPEC_DECODE_DETECTED: bool | None = None
_DSV4_GRIDBOOK_CONTRACT: dict | None = None
_DSV4_GRIDBOOK_KWARGS: dict | None = None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_text(value: object) -> str:
    """Serialize release evidence without JavaScript NaN/Infinity tokens."""
    try:
        return json.dumps(value, indent=2, allow_nan=False)
    except ValueError as exc:
        raise RuntimeError(
            "refusing to serialize non-finite full-KL evidence"
        ) from exc


def _activate_dsv4_gridbook_contract(args) -> dict:
    """Apply the closed environment/kwargs before importing the runtime."""
    global _DSV4_GRIDBOOK_CONTRACT, _DSV4_GRIDBOOK_KWARGS
    if not args.dsv4_gridbook_contract:
        return {}
    if _DSV4_GRIDBOOK_KWARGS is None:
        exact_kwargs, receipt = exact_llm_contract(args.model)
        _DSV4_GRIDBOOK_KWARGS = dict(exact_kwargs)
        _DSV4_GRIDBOOK_CONTRACT = dict(receipt)
    args.quantization = "gridbook"
    args.dtype = "bfloat16"
    args.gpu_memory_utilization = 0.84
    args.enforce_eager = True
    args.max_num_batched_tokens = 512
    args.max_logprobs = 248_320
    return dict(_DSV4_GRIDBOOK_KWARGS)


def _provenance(args) -> dict:
    """Serving-stack + code provenance for a result dict (R15).

    This tool builds its own in-process `LLM`, so the measuring process *is*
    the server and `/proc/self/maps` is the authoritative residency read. Two
    result JSONs whose `serve_fingerprint` differs are not comparable as a
    delta (`tools/kl_ab.py` refuses them).
    """
    contract_sha = (
        hashlib.sha256(json.dumps(
            _DSV4_GRIDBOOK_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if _DSV4_GRIDBOOK_CONTRACT is not None else None
    )
    from prismaquant.gridbook_serving_runtime_pin import (
        load_gridbook_serving_runtime_pin,
        require_exact_gridbook_serving_runtime_release,
    )
    from prismaquant.validate_cb_endpoint import DSV4_SPARK_VLLM_IMAGE

    pin = load_gridbook_serving_runtime_pin()
    if args.dsv4_gridbook_contract:
        require_exact_gridbook_serving_runtime_release(pin)
    producer = gold_producer_identity("measure_vllm_full_kl")
    manifest = self_manifest(
        extra={
            "measurement_tool": "measure_vllm_full_kl",
            "producer_identity": producer,
            "effective_llm_kwargs": _DSV4_GRIDBOOK_KWARGS,
            "dsv4_gridbook_contract_sha256": contract_sha,
        },
        image=DSV4_SPARK_VLLM_IMAGE if args.dsv4_gridbook_contract else None,
        artifact_dir=(args.model if args.dsv4_gridbook_contract else None),
        require_engine_descendant=bool(args.dsv4_gridbook_contract),
        gridbook_pin_attestation={
            "repository": pin.repository,
            "commit": pin.commit,
            "version": pin.version,
            "wheel_sha256": pin.wheel_sha256,
        } if args.dsv4_gridbook_contract else None,
    )
    return {
        "git_commit": producer["git_commit"],
        "serve_fingerprint": manifest["serve_fingerprint"],
        "serve_manifest": manifest,
        "spec_decode_detected": _SPEC_DECODE_DETECTED,
        "dsv4_gridbook_contract": _DSV4_GRIDBOOK_CONTRACT,
        "dsv4_gridbook_contract_sha256": contract_sha,
    }


def _load_wikitext_calibration(
    tokenizer,
    *,
    cache_dir: str,
    n_samples: int,
    seqlen: int,
    window_seed: int = 42,
    text_file: str | None = None,
) -> tuple[list[list[int]], list[int], int, dict]:
    """Build the calibration windows, and attest which bytes they came from.

    `text_file` is the materialized corpus (tools/materialize_wikitext_corpus.py)
    and is the preferred source: the measurement containers carry no `datasets`
    / pyarrow / pandas, and installing them at measurement time would mutate the
    very serving stack the number is fingerprinted against. Reading bytes also
    lets teacher and student prove they scored the SAME corpus by sha256, rather
    than each calling load_dataset and trusting two resolutions to agree.

    The `datasets` path is kept for hosts that have it, and the join rule is
    identical in both branches -- the materializer copies it verbatim.
    """
    if text_file:
        raw = Path(text_file).read_bytes()
        text = raw.decode("utf-8")
        corpus = {
            "source": "text_file",
            "path": str(text_file),
            "bytes": len(raw),
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        }
    else:
        from datasets import load_dataset

        ds = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=cache_dir,
        )
        text = "\n\n".join(
            row["text"] for row in ds if row.get("text", "").strip()
        )
        corpus = {
            "source": "datasets",
            "cache_dir": str(cache_dir),
            "bytes": len(text.encode("utf-8")),
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "datasets_fingerprint": getattr(ds, "_fingerprint", None),
        }
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if int(ids.numel()) < seqlen + 1:
        raise RuntimeError(f"not enough calibration tokens: {int(ids.numel())}")
    max_start = int(ids.numel()) - int(seqlen)
    rng = random.Random(window_seed)
    if max_start >= n_samples:
        starts = rng.sample(range(max_start), n_samples)
    else:
        starts = [
            min(max_start, int(i * max_start / max(n_samples, 1)))
            for i in range(n_samples)
        ]
    calib = [ids[s : s + seqlen].tolist() for s in starts]
    corpus["total_tokens"] = int(ids.numel())
    return calib, starts, int(ids.numel()), corpus


def _load_llm(args, *, max_model_len: int) -> "LLM":
    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_logprobs": args.max_logprobs,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.max_num_batched_tokens is not None:
        # Mamba/DeltaNet hybrids (e.g. Qwen3.6-35B-A3B) require
        # max_num_batched_tokens >= their chunk-alignment floor (~2096);
        # the seqlen+16 max_model_len alone can drive it below that.
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.dsv4_gridbook_contract:
        kwargs.update(_activate_dsv4_gridbook_contract(args))
    # Environment/bootstrap above must precede the first Gridbook/vLLM import.
    from vllm import LLM

    llm = LLM(**kwargs)
    global _SPEC_DECODE_DETECTED
    _SPEC_DECODE_DETECTED = refuse_if_spec_decode(
        llm=llm,
        allow=getattr(args, "allow_spec_decode", False),
        context="kl",
    )
    return llm


def _resolve_vocab_size(llm: LLM, tokenizer) -> int:
    hf_config = llm.llm_engine.model_config.hf_config
    vocab_size = int(getattr(hf_config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = int(len(tokenizer))
    if vocab_size <= 0:
        raise RuntimeError("could not resolve model vocabulary size")
    return vocab_size


def _logprob_vector(logprobs, *, vocab_size: int) -> torch.Tensor:
    vec = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    seen_token_ids: set[int] = set()
    for key, value in logprobs.items():
        token_id = int(key)
        if token_id < 0:
            raise RuntimeError(
                f"vLLM returned negative token id {token_id} in full-vocab logprobs"
            )
        if token_id >= vocab_size:
            continue
        if token_id in seen_token_ids:
            raise RuntimeError(
                f"vLLM returned duplicate token id {token_id} in full-vocab logprobs"
            )
        seen_token_ids.add(token_id)
        logprob = getattr(value, "logprob", None)
        if logprob is None and isinstance(value, dict):
            logprob = value.get("logprob")
        if logprob is None and isinstance(value, (tuple, list)):
            logprob = value[0]
        if logprob is None:
            logprob = value
        logprob = float(logprob)
        if not math.isfinite(logprob):
            raise RuntimeError(
                f"vLLM returned non-finite logprob for token id {token_id}"
            )
        if logprob > 0.0:
            raise RuntimeError(
                f"vLLM returned positive logprob for token id {token_id}: {logprob}"
            )
        vec[token_id] = logprob
    missing = int(torch.isneginf(vec).sum().item())
    if missing:
        raise RuntimeError(
            f"vLLM returned {vocab_size - missing}/{vocab_size} logprobs; "
            "full-vocab KL requires the engine to return every token logprob"
        )
    return vec


def _measure_logprobs(
    llm: "LLM",
    prompts: list[list[int]],
    *,
    vocab_size: int,
) -> torch.Tensor:
    from vllm import SamplingParams

    # logprobs=-1 (full vocab) is the primary request, but some
    # model/vLLM combinations (observed: Qwen3-4B on vllm 0.21.1rc1)
    # silently return an EMPTY logprobs list for -1 — for those, retry
    # with an explicit vocab-size count. The reverse also exists: with
    # an explicit count some engines OMIT -inf (padding) tokens
    # (observed: Qwen3.6-35B-A3B, 243 entries short), so -1 must stay
    # the first choice. _logprob_vector's completeness check guards
    # whichever path produced the row.
    def _params(logprob_arg: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=logprob_arg,
            detokenize=False,
        )

    rows = []
    for index, prompt_ids in enumerate(prompts, 1):
        start = time.monotonic()
        logprobs = None
        for logprob_arg in (-1, int(vocab_size)):
            output = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                _params(logprob_arg),
                use_tqdm=False,
            )[0]
            got = output.outputs[0].logprobs
            if got and len(got) and len(got[0]):
                logprobs = got[0]
                break
        if logprobs is None:
            raise RuntimeError(
                "vLLM returned no logprobs under either logprobs=-1 or "
                f"logprobs={vocab_size}")
        rows.append(_logprob_vector(logprobs, vocab_size=vocab_size))
        print(
            f"[kl] sample {index}/{len(prompts)} "
            f"logprobs={len(logprobs)} wall={time.monotonic() - start:.2f}s",
            flush=True,
        )
    return torch.stack(rows, dim=0).contiguous()


def _measure_prompt_topk(
    llm: "LLM",
    prompts: list[list[int]],
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-position scoring: per prompt position, the top-K token ids and
    logprobs of the model's next-token distribution (vLLM prompt_logprobs).

    Returns (ids, lps) shaped [n_prompts, P-1, K] (position 0 has no
    prediction). Full-vocab dicts at every position are infeasible (~620M
    Python objects per pass), so K bounds the support; the tail is handled
    as a single bucket by the caller. K is set by the teacher's measured
    coverage, not by convention -- see PROMPT_TOP_K in full_kl_teacher_payload
    for the sweep that fixes it -- and the truncation floor is shared across
    arms.
    Positions with fewer than K entries are padded with ``(-1, -inf)``;
    ``_position_kl`` masks the pads back out.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=int(top_k),
        detokenize=False,
    )
    all_ids, all_lps = [], []
    for index, prompt_ids in enumerate(prompts, 1):
        start = time.monotonic()
        output = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            params,
            use_tqdm=False,
        )[0]
        plps = output.prompt_logprobs
        if plps is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        ids_rows, lps_rows = [], []
        for pos in range(1, len(prompt_ids)):
            d = plps[pos]
            items = [(int(k), float(getattr(v, "logprob", v)))
                     for k, v in d.items()]
            items.sort(key=lambda kv: kv[1], reverse=True)
            items = items[: int(top_k)]
            if len(items) < int(top_k):
                pad = int(top_k) - len(items)
                items = items + [(-1, float("-inf"))] * pad
            ids_rows.append([kv[0] for kv in items])
            lps_rows.append([kv[1] for kv in items])
        all_ids.append(torch.tensor(ids_rows, dtype=torch.int32))
        all_lps.append(torch.tensor(lps_rows, dtype=torch.float32))
        print(
            f"[kl] sample {index}/{len(prompts)} positions={len(ids_rows)} "
            f"top_k={top_k} wall={time.monotonic() - start:.2f}s",
            flush=True,
        )
    return torch.stack(all_ids), torch.stack(all_lps)


def _teacher(args) -> int:
    from transformers import AutoTokenizer

    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, starts, total_tokens, corpus = _load_wikitext_calibration(
        tokenizer,
        cache_dir=args.dataset_cache_dir,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        window_seed=args.window_seed,
        text_file=args.corpus_text_file,
    )
    print(
        f"[kl] teacher model={args.model} n={args.n_samples} "
        f"seqlen={args.seqlen} total_tokens={total_tokens}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=args.seqlen + 16)
    vocab_size = _resolve_vocab_size(llm, tokenizer)
    if int(args.max_logprobs) < vocab_size:
        raise RuntimeError(
            f"--max-logprobs={args.max_logprobs} is smaller than "
            f"model vocab_size={vocab_size}; full-vocab KL requires "
            "requesting at least the full vocabulary"
        )
    if args.score_positions == "all":
        topk_ids, topk_lps = _measure_prompt_topk(
            llm, prompts, top_k=args.prompt_top_k)
        payload = {
            "score_positions": "all",
            "prompt_top_k": int(args.prompt_top_k),
            "topk_ids": topk_ids,
            # fp32, matching the student side: fp16 teacher logprobs against
            # fp32 student logprobs is an asymmetric rounding that biases the
            # absolute confident-KL (mostly cancels in paired A/Bs, but the
            # published absolute numbers come through here too).
            "topk_lps": topk_lps.to(torch.float32),
            "calib_ids": torch.tensor(prompts, dtype=torch.long),
            "starts": starts,
            "model": args.model,
            "n_samples": int(args.n_samples),
            "seqlen": int(args.seqlen),
            "vocab_size": int(vocab_size),
        }
        torch.save(payload, output)
        cov = topk_lps.double().exp().sum(dim=-1)
        meta = {
            "mode": "teacher",
            "score_positions": "all",
            "prompt_top_k": int(args.prompt_top_k),
            "model": args.model,
            "output": str(output),
            "n_samples": int(args.n_samples),
            "seqlen": int(args.seqlen),
            "starts": starts,
            "total_tokens": total_tokens,
        "corpus": corpus,
            "vocab_size": int(vocab_size),
            "teacher_shape": list(topk_lps.shape),
            "topk_coverage_mean": float(cov.mean()),
            "topk_coverage_min": float(cov.min()),
            "elapsed_s": time.monotonic() - started,
            **_provenance(args),
        }
        meta_text = _strict_json_text(meta)
        Path(args.meta_output).write_text(meta_text)
        print(meta_text, flush=True)
        return 0
    logprobs = _measure_logprobs(llm, prompts, vocab_size=vocab_size)
    payload = {
        "teacher_logprobs": logprobs,
        "calib_ids": torch.tensor(prompts, dtype=torch.long),
        "starts": starts,
        "corpus": corpus,
        "model": args.model,
        "n_samples": int(args.n_samples),
        "seqlen": int(args.seqlen),
        "vocab_size": int(vocab_size),
    }
    torch.save(payload, output)
    meta = {
        "mode": "teacher",
        "model": args.model,
        "output": str(output),
        "n_samples": int(args.n_samples),
        "seqlen": int(args.seqlen),
        "starts": starts,
        "total_tokens": total_tokens,
        "corpus": corpus,
        "vocab_size": int(vocab_size),
        "teacher_shape": list(logprobs.shape),
        "elapsed_s": time.monotonic() - started,
        **_provenance(args),
    }
    meta_text = _strict_json_text(meta)
    Path(args.meta_output).write_text(meta_text)
    print(meta_text, flush=True)
    return 0


def _validated_topk_entries(ids_row, lps_row, *, role: str) -> list[tuple[int, float]]:
    """Return one strict top-K row, excluding only canonical pad entries."""
    token_ids = ids_row.tolist()
    logprobs = lps_row.tolist()
    if len(token_ids) != len(logprobs):
        raise RuntimeError(f"{role} top-K token/logprob lengths differ")

    entries: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_token_id, raw_logprob in zip(token_ids, logprobs):
        token_id = int(raw_token_id)
        logprob = float(raw_logprob)
        if token_id == -1:
            if not math.isinf(logprob) or logprob > 0.0:
                raise RuntimeError(
                    f"{role} top-K padding must be the canonical (-1, -inf) pair"
                )
            continue
        if token_id < 0:
            raise RuntimeError(f"{role} top-K token id {token_id} is invalid")
        if token_id in seen:
            raise RuntimeError(
                f"{role} top-K row contains duplicate token id {token_id}"
            )
        seen.add(token_id)
        if not math.isfinite(logprob):
            raise RuntimeError(
                f"{role} top-K row contains a non-finite logprob for token id "
                f"{token_id}"
            )
        if logprob > 0.0:
            raise RuntimeError(
                f"{role} top-K row contains a positive logprob for token id "
                f"{token_id}: {logprob}"
            )
        entries.append((token_id, logprob))
    if not entries:
        raise RuntimeError(f"{role} position carries no finite top-K entries")
    return entries


def _position_kl(t_ids_row, t_lps_row, s_ids_row, s_lps_row) -> tuple[float, float]:
    """KL(teacher || student) over one position's top-K support + tail bucket.

    Returns ``(kl, teacher_top1_prob)``. Pad entries (id ``-1`` / ``-inf``
    logprob, emitted when a position carries fewer than K entries) are masked
    out of both the KL sum and the accounted probability mass: an unmasked pad
    produces ``0 * (-inf) = NaN`` and poisons the whole run's mean, and a pad
    mapped to the student floor would wrongly consume student mass from the
    tail bucket. Entries with exactly zero teacher probability contribute
    zero KL and zero mass, so masking them is exact, not an approximation.

    The student-floor substitution and the 1e-12 tail clamps are the
    documented relative-compare-only convention (shared across arms); they
    are deliberately left as-is.
    """
    student = _validated_topk_entries(s_ids_row, s_lps_row, role="student")
    smap = dict(student)
    floor = min(smap.values())                        # kl_ab.py convention
    valid = _validated_topk_entries(t_ids_row, t_lps_row, role="teacher")
    tlp = torch.tensor([lp for _t, lp in valid], dtype=torch.float64)
    q = torch.tensor([smap.get(t, floor) for t, _lp in valid],
                     dtype=torch.float64)
    p = tlp.exp()
    kl = float((p * (tlp - q)).sum())
    # tail bucket: remaining teacher mass vs remaining student mass
    pt = max(1.0 - float(p.sum()), 1e-12)
    qt = max(1.0 - float(q.exp().sum()), 1e-12)
    kl += pt * (math.log(pt) - math.log(qt))
    top1 = float(p.max())
    if not math.isfinite(kl) or not math.isfinite(top1):
        raise RuntimeError(
            f"non-finite KL output: kl={kl!r}, teacher_top1_prob={top1!r}"
        )
    return kl, top1


def _assert_teacher_matches_candidate_source(args, teacher_evidence) -> None:
    """Bind streamed BF16 teacher identity to the candidate's exact source."""
    if not args.dsv4_gridbook_contract:
        return
    if not isinstance(teacher_evidence, dict):
        raise RuntimeError(
            "DSv4 Gridbook KL requires digest-bound streamed teacher evidence"
        )
    quant_path = Path(args.model).resolve(strict=True) / "quant_config.json"
    try:
        quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"could not read candidate source identity from {quant_path}"
        ) from exc
    provenance = quant_config.get("provenance") if isinstance(
        quant_config, dict
    ) else None
    candidate_source = provenance.get("source_model_identity") if isinstance(
        provenance, dict
    ) else None
    if candidate_source != teacher_evidence.get("source_model"):
        raise RuntimeError(
            "streamed teacher source identity differs from candidate provenance"
        )


def _student_all_positions(args, payload, teacher_evidence=None) -> int:
    started = time.monotonic()
    prompts = payload["calib_ids"].tolist()
    vocab_size = int(payload["vocab_size"])
    top_k = int(payload["prompt_top_k"])
    t_ids = payload["topk_ids"]                       # [n, P-1, K]
    t_lps = payload["topk_lps"].float()
    print(
        f"[kl] student(all-pos) model={args.model} n={len(prompts)} "
        f"seqlen={int(payload['seqlen'])} top_k={top_k}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=int(payload["seqlen"]) + 16)
    s_ids, s_lps = _measure_prompt_topk(llm, prompts, top_k=top_k)

    n, pm1, k = t_ids.shape
    if args.dsv4_gridbook_contract and n * pm1 != EXPECTED_POSITIONS:
        raise RuntimeError(
            f"DSv4 gold workload requires {EXPECTED_POSITIONS} positions, "
            f"got {n * pm1}"
        )
    kl_pos = torch.zeros((n, pm1), dtype=torch.float64)
    t_top1 = torch.zeros((n, pm1), dtype=torch.float64)
    for i in range(n):
        for j in range(pm1):
            kl, top1 = _position_kl(
                t_ids[i, j], t_lps[i, j], s_ids[i, j], s_lps[i, j],
            )
            kl_pos[i, j] = kl
            t_top1[i, j] = top1
    if not torch.isfinite(kl_pos).all() or not torch.isfinite(t_top1).all():
        raise RuntimeError("non-finite all-position KL output")
    confident = t_top1 > 0.5
    flat = kl_pos.flatten()
    result = {
        "mode": "student",
        "score_positions": "all",
        "prompt_top_k": top_k,
        "model": args.model,
        "teacher_model": payload.get("model"),
        "teacher_payload": str(args.teacher_payload),
        "teacher_payload_sha256": args.teacher_payload_sha256,
        "quantization": args.quantization,
        "n_samples": len(prompts),
        "seqlen": int(payload["seqlen"]),
        "vocab_size": vocab_size,
        "n_positions": int(flat.numel()),
        "kl_mean": float(flat.mean()),
        "kl_p99": float(flat.quantile(0.99)),
        "kl_max": float(flat.max()),
        "kl_confident_mean": float(kl_pos[confident].mean())
        if bool(confident.any()) else None,
        "n_confident": int(confident.sum()),
        "kl_per_sample": [float(x) for x in kl_pos.mean(dim=1).tolist()],
        "elapsed_s": time.monotonic() - started,
        "teacher_evidence": teacher_evidence,
        **_provenance(args),
    }
    result_text = _strict_json_text(result)
    Path(args.output).write_text(result_text)
    print(result_text, flush=True)
    return 0


def _student(args) -> int:
    started = time.monotonic()
    teacher_payload_sha256 = _file_sha256(args.teacher_payload)
    teacher_evidence = None
    if args.teacher_meta:
        payload, teacher_evidence = load_teacher_evidence(
            args.teacher_payload, args.teacher_meta
        )
    else:
        payload = safe_load_torch_payload(args.teacher_payload)
    if _file_sha256(args.teacher_payload) != teacher_payload_sha256:
        raise RuntimeError("teacher payload changed while the student loaded it")
    args.teacher_payload_sha256 = teacher_payload_sha256
    _assert_teacher_matches_candidate_source(args, teacher_evidence)
    if payload.get("score_positions") == "all":
        return _student_all_positions(args, payload, teacher_evidence)
    teacher = payload["teacher_logprobs"].float()
    prompts = payload["calib_ids"].tolist()
    vocab_size = int(payload["vocab_size"])
    print(
        f"[kl] student model={args.model} n={len(prompts)} "
        f"seqlen={int(payload['seqlen'])} vocab={vocab_size}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=int(payload["seqlen"]) + 16)
    student = _measure_logprobs(llm, prompts, vocab_size=vocab_size)
    teacher_probs = teacher.exp()
    per_sample = (teacher_probs * (teacher - student)).sum(dim=-1)
    if not torch.isfinite(per_sample).all():
        raise RuntimeError(f"non-finite KL values: {per_sample.tolist()}")
    result = {
        "mode": "student",
        "model": args.model,
        "teacher_model": payload.get("model"),
        "teacher_payload": str(args.teacher_payload),
        "teacher_payload_sha256": teacher_payload_sha256,
        "quantization": args.quantization,
        "n_samples": len(prompts),
        "seqlen": int(payload["seqlen"]),
        "vocab_size": vocab_size,
        "kl_mean": float(per_sample.mean().item()),
        "kl_min": float(per_sample.min().item()),
        "kl_max": float(per_sample.max().item()),
        "kl_per_sample": [float(x) for x in per_sample.tolist()],
        "elapsed_s": time.monotonic() - started,
        **_provenance(args),
    }
    result_text = _strict_json_text(result)
    Path(args.output).write_text(result_text)
    print(result_text, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teacher", "student"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default="teacher_meta.json")
    parser.add_argument("--teacher-payload")
    parser.add_argument(
        "--teacher-meta",
        help="metadata JSON emitted beside the streamed teacher payload; "
        "required by the DSv4 Gridbook release contract",
    )
    parser.add_argument("--dataset-cache-dir", default="/hfcache/datasets")
    parser.add_argument("--corpus-text-file", default=None,
                        help="Materialized corpus text (see "
                             "tools/materialize_wikitext_corpus.py). Preferred "
                             "over --dataset-cache-dir: the measurement "
                             "containers carry no `datasets`/pyarrow/pandas, "
                             "and installing them at measurement time would "
                             "mutate the serving stack the number is "
                             "fingerprinted against.")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--max-logprobs", type=int, default=248320)
    parser.add_argument(
        "--score-positions", choices=["final", "all"], default="final",
        help="final: full-vocab KL at the window-final context only "
        "(legacy; n_positions = n_samples). all: top-K KL at every prompt "
        "position (n_positions = n_samples*(seqlen-1)).")
    parser.add_argument("--prompt-top-k", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-spec-decode", action="store_true",
        help="proceed even when the engine has a speculative config. The "
        "numbers are then the DRAFT model's, not the artifact's; the shipcard "
        "refuses such a record (see tools/spec_decode_guard.py).")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--dsv4-gridbook-contract",
        action="store_true",
        help="force the exact one-Spark DSv4 Gridbook gold-measurement "
        "runtime contract (FP8 KV, conditional Marlin MoE, closed env, "
        "eager/no-spec)",
    )
    parser.add_argument(
        "--window-seed", type=int, default=42,
        help="RNG seed for the WikiText window draw (teacher mode only; "
        "students replay the windows stored in the teacher payload)")
    args = parser.parse_args()
    if args.mode == "student" and not args.teacher_payload:
        parser.error("--teacher-payload is required in student mode")
    if args.mode == "teacher" and args.dsv4_gridbook_contract:
        parser.error(
            "DSv4 Gridbook teacher mode is streamed-only; use "
            "tools/build_streamed_full_kl_teacher.py with "
            "--wikitext-inputs"
        )
    if (
        args.mode == "student"
        and args.dsv4_gridbook_contract
        and not args.teacher_meta
    ):
        parser.error(
            "--teacher-meta is required for DSv4 Gridbook student evidence"
        )
    if args.dsv4_gridbook_contract:
        _activate_dsv4_gridbook_contract(args)
    if args.mode == "teacher":
        return _teacher(args)
    return _student(args)


if __name__ == "__main__":
    raise SystemExit(main())
