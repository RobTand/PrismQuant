#!/usr/bin/env python3
"""Measure WikiText perplexity for a vLLM-loadable model/artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

_TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_ROOT))
from prismaquant_source_bootstrap import activate_prismaquant_source

activate_prismaquant_source()

try:  # package mode (`python -m tools.measure_vllm_wikitext_ppl`)
    from .dsv4_wikitext_inputs import load_dsv4_wikitext_inputs
    from .serve_fingerprint import gold_producer_identity, self_manifest
    from .spec_decode_guard import refuse_if_spec_decode
except ImportError:  # script mode (`python /repo/tools/measure_vllm_wikitext_ppl.py`)
    from dsv4_wikitext_inputs import load_dsv4_wikitext_inputs  # type: ignore
    from serve_fingerprint import (  # type: ignore
        gold_producer_identity,
        self_manifest,
    )
    from spec_decode_guard import refuse_if_spec_decode  # type: ignore

#: Set by `_load_llm`; `None` (could not inspect) is refused by the shipcard.
_SPEC_DECODE_DETECTED: bool | None = None

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_PPL_CALIBRATION_SCHEMA = (
    "prismaquant.wikitext_ppl_calibration/1"
)
# The DSv4 gold corpus/tokenizer value identities lived here and were asserted
# only under --dsv4-gridbook-contract, retired 2026-09-02 with its lane; see
# archive/gridbook_lane_2026-09-02/. The digests themselves are not lost --
# prismaquant/shipcard.py keeps the same five DSV4_WIKITEXT_*/DSV4_TOKENIZER_*
# receipts, which is where a shipped record is checked against them.
_CORPUS_CONSTRUCTION = {
    "row_filter": "include iff bool(text.strip()); preserve text verbatim",
    "join_separator": "\n\n",
    "normalization": "none",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(value: object) -> str:
    """Pretty-print a result without JavaScript's non-standard NaN values."""
    return json.dumps(value, indent=2, allow_nan=False)


def _validate_workload_args(args) -> None:
    """Refuse impossible or non-gold workloads before runtime construction."""
    for name in ("n_tokens", "seqlen"):
        value = getattr(args, name, None)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 2
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be an integer >= 2")


def _resolve_serve_image(args) -> str:
    """The container image this measurement runs in -- never guessed.

    The image used to come from the Gridbook lane's serving pin, retired
    2026-09-02 (archive/gridbook_lane_2026-09-02/), so a caller could get one
    without naming it. A wrong image tag is a silent reproducibility confound
    (R15), so this fails closed: name it with --serve-image or PQ_SERVE_IMAGE.
    """
    image = (args.serve_image or os.environ.get("PQ_SERVE_IMAGE") or "").strip()
    if not image:
        raise RuntimeError(
            "the serving container image is unknown: pass --serve-image or "
            "set PQ_SERVE_IMAGE. The serve fingerprint that makes two PPL "
            "numbers comparable is not evidence without it."
        )
    return image


def _provenance(args) -> dict:
    """Serving-stack + code provenance for the result dict (R15).

    The tool builds its own in-process `LLM`, so `/proc/self/maps` is the
    authoritative extension-residency read for the numbers below; a PPL from a
    different `serve_fingerprint` is not a comparable delta (tools/kl_ab.py).
    """
    producer = gold_producer_identity("measure_vllm_wikitext_ppl")
    manifest = self_manifest(
        extra={
            "measurement_tool": "measure_vllm_wikitext_ppl",
            "producer_identity": producer,
        },
        image=_resolve_serve_image(args),
    )
    return {
        "git_commit": producer["git_commit"],
        "serve_fingerprint": manifest["serve_fingerprint"],
        "serve_manifest": manifest,
        "spec_decode_detected": _SPEC_DECODE_DETECTED,
    }


def _load_ids(
    tokenizer,
    *,
    cache_dir: str,
    split: str,
    n_tokens: int,
    text_file: str | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Tokenize the corpus prefix to score, and attest which bytes it was.

    `text_file` is the materialized corpus (tools/materialize_wikitext_corpus.py).
    The measurement containers carry no `datasets`/pyarrow/pandas, and installing
    them at measurement time would mutate the serving stack this number is
    fingerprinted against, so bytes-on-disk is the preferred source. The
    `datasets` branch is unchanged for hosts that have it; the join rule is
    identical in both, and `corpus_sha256` is what actually proves two runs
    scored the same text.
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 2:
        raise ValueError("n_tokens must be an integer >= 2")
    if text_file:
        raw = Path(text_file).read_bytes()
        text = raw.decode("utf-8")
        # A materialized corpus has no `datasets` fingerprint to quote, so the
        # content hash IS the fingerprint -- spelled so it can never be mistaken
        # for one.
        fingerprint = "file:sha256:" + hashlib.sha256(raw).hexdigest()
    else:
        from datasets import load_dataset

        ds = load_dataset(
            WIKITEXT_DATASET,
            WIKITEXT_CONFIG,
            split=split,
            cache_dir=cache_dir,
            revision=WIKITEXT_REVISION,
        )
        rows = [
            row["text"]
            for row in ds
            if isinstance(row.get("text"), str) and row["text"].strip()
        ]
        text = "\n\n".join(rows)
        fingerprint = getattr(ds, "_fingerprint", None)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total_tokens = int(ids.numel())
    if total_tokens < n_tokens:
        raise RuntimeError(
            "WikiText tokenization cannot satisfy the requested exact prefix: "
            f"requested={n_tokens}, available={total_tokens}"
        )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("WikiText corpus exposes no immutable fingerprint")
    selected = ids[:n_tokens].to(dtype=torch.long, device="cpu").tolist()
    if len(selected) != n_tokens:
        raise RuntimeError("WikiText selected token prefix has the wrong length")
    return selected, {
        "fingerprint": fingerprint,
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "total_tokens": total_tokens,
    }


def _load_measurement_ids(
    args,
    *,
    tokenizer_attestation: Mapping[str, Any],
    tokenizer=None,
) -> tuple[list[int], dict[str, object]]:
    """Load generic corpus tokens, or a pre-tokenized offline input payload.

    The offline payload used to be selected by the Gridbook lane's contract
    flag (retired 2026-09-02, archive/gridbook_lane_2026-09-02/). It is a
    lane-independent capability -- it keeps `datasets`/pyarrow out of the
    serving image -- so it is now selected by naming --wikitext-inputs.
    """
    if not getattr(args, "wikitext_inputs", None):
        if tokenizer is None:
            raise ValueError("generic WikiText PPL requires a tokenizer")
        return _load_ids(
            tokenizer,
            cache_dir=args.dataset_cache_dir,
            split=args.split,
            n_tokens=args.n_tokens,
            text_file=args.corpus_text_file,
        )
    payload = load_dsv4_wikitext_inputs(
        args.wikitext_inputs,
        expected_tokenizer_identity=tokenizer_attestation,
    )
    ppl_input = payload["ppl"]
    dataset = ppl_input["dataset"]
    return list(ppl_input["token_ids"]), {
        "fingerprint": dataset["fingerprint"],
        "corpus_sha256": dataset["corpus_sha256"],
        "total_tokens": dataset["total_tokens"],
    }


def _build_chunks(ids: list[int], *, seqlen: int) -> list[list[int]]:
    """Build the only permitted non-overlapping, lossless chunk partition."""
    chunks = [
        ids[start : start + seqlen]
        for start in range(0, len(ids), seqlen)
    ]
    if not chunks or any(len(chunk) < 2 for chunk in chunks):
        raise RuntimeError(
            "WikiText PPL chunk law requires at least two tokens per chunk"
        )
    if [token for chunk in chunks for token in chunk] != ids:
        raise RuntimeError("WikiText PPL chunks do not flatten to the token prefix")
    return chunks


def _ppl_calibration_contract(
    *,
    args,
    ids: list[int],
    dataset_evidence: Mapping[str, Any],
    tokenizer_identity_sha256: str,
    chunks: list[list[int]],
    n_tokens_scored: int,
) -> dict[str, object]:
    """Bind the exact corpus, tokenizer, token prefix, and scoring windows."""
    _validate_workload_args(args)
    if set(dataset_evidence) != {
        "fingerprint", "corpus_sha256", "total_tokens",
    }:
        raise RuntimeError("WikiText dataset evidence is not closed and exact")
    fingerprint = dataset_evidence.get("fingerprint")
    corpus_sha256 = dataset_evidence.get("corpus_sha256")
    total_tokens = dataset_evidence.get("total_tokens")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("WikiText dataset fingerprint is missing")
    if (
        not isinstance(corpus_sha256, str)
        or len(corpus_sha256) != 64
        or any(char not in "0123456789abcdef" for char in corpus_sha256)
    ):
        raise RuntimeError("WikiText corpus digest is not a lowercase SHA256")
    if (
        not isinstance(total_tokens, int)
        or isinstance(total_tokens, bool)
        or total_tokens < int(args.n_tokens)
    ):
        raise RuntimeError("WikiText total-token evidence is invalid")
    if not isinstance(tokenizer_identity_sha256, str) or (
        len(tokenizer_identity_sha256) != 64
    ) or any(
        char not in "0123456789abcdef"
        for char in tokenizer_identity_sha256
    ):
        raise RuntimeError("tokenizer identity is not a lowercase SHA256")
    if (
        len(ids) != int(args.n_tokens)
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in ids
        )
    ):
        raise RuntimeError("WikiText selected token prefix is not exact")
    expected_chunks = [
        ids[start : start + int(args.seqlen)]
        for start in range(0, len(ids), int(args.seqlen))
    ]
    if (
        not chunks
        or chunks != expected_chunks
        or any(len(chunk) < 2 for chunk in chunks)
        or [token for chunk in chunks for token in chunk] != ids
    ):
        raise RuntimeError("WikiText PPL chunks violate the exact chunk law")
    expected_scored = sum(len(chunk) - 1 for chunk in chunks)
    if (
        not isinstance(n_tokens_scored, int)
        or isinstance(n_tokens_scored, bool)
        or n_tokens_scored != expected_scored
    ):
        raise RuntimeError("WikiText PPL scored-token count is inconsistent")
    selected_sha256 = _canonical_sha256(ids)
    chunk_starts = list(range(0, len(ids), int(args.seqlen)))
    # The corpus/token/tokenizer value-identity assertions that ran here were
    # gated on --dsv4-gridbook-contract, retired 2026-09-02 with its lane (see
    # archive/gridbook_lane_2026-09-02/). The contract below still records
    # every one of those identities, so a consumer can still check them.
    contract = {
        "schema": WIKITEXT_PPL_CALIBRATION_SCHEMA,
        "dataset": {
            "name": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "split": str(args.split),
            "revision": WIKITEXT_REVISION,
            "fingerprint": fingerprint,
            "corpus_sha256": corpus_sha256,
        },
        "corpus_construction": dict(_CORPUS_CONSTRUCTION),
        "tokenizer": {
            "identity_sha256": tokenizer_identity_sha256,
            "trust_remote_code": True,
            "add_special_tokens": False,
        },
        "token_selection": {
            "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
            "n_tokens_requested": int(args.n_tokens),
            "n_tokens_available": total_tokens,
            "selected_token_count": len(ids),
            "token_ids_sha256": selected_sha256,
            "digest_encoding": "canonical_json_integer_array/v1",
        },
        "scoring": {
            "chunking": "nonoverlapping_contiguous/v1",
            "seqlen": int(args.seqlen),
            "chunk_starts": chunk_starts,
            "chunk_token_counts": [len(chunk) for chunk in chunks],
            "positions": "within_each_chunk_positions_1_through_N_minus_1",
            "n_tokens_scored": int(n_tokens_scored),
            "prompt_logprobs": 1,
            "temperature": 0.0,
            "max_tokens": 1,
            "detokenize": False,
        },
    }
    return contract


def _load_llm(args) -> "LLM":
    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": int(args.seqlen) + 1,
        "max_num_seqs": 1,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.max_num_batched_tokens is not None:
        # Mamba/DeltaNet hybrids need max_num_batched_tokens >= their
        # chunk-alignment floor (~2096); seqlen+1 alone can undershoot it.
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    # Environment/bootstrap above must precede the first vLLM import.
    from vllm import LLM

    llm = LLM(**kwargs)
    # Spec-decode routes /v1/completions echo+logprobs (and prompt_logprobs)
    # through the DRAFT model: the NLL below would be the 1-layer MTP head's.
    # `validate_quantized_model` has refused this since the draft-logprobs
    # postmortem; the gold lane had no such guard until R13.
    global _SPEC_DECODE_DETECTED
    _SPEC_DECODE_DETECTED = refuse_if_spec_decode(
        llm=llm,
        allow=getattr(args, "allow_spec_decode", False),
        context="ppl",
    )
    return llm


def _logprob_value(entry, token_id: int) -> float:
    if entry is None:
        raise KeyError(token_id)
    value = None
    if isinstance(entry, dict):
        value = entry.get(token_id)
        if value is None:
            value = entry.get(str(token_id))
    if value is None:
        raise KeyError(token_id)
    logprob = getattr(value, "logprob", None)
    if logprob is None and isinstance(value, dict):
        logprob = value.get("logprob")
    if logprob is None and isinstance(value, (tuple, list)):
        logprob = value[0]
    if logprob is None:
        logprob = value
    result = float(logprob)
    if not math.isfinite(result):
        raise ValueError(f"token {token_id} has a non-finite logprob")
    if result > 0.0:
        raise ValueError(f"token {token_id} has a positive logprob")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-cache-dir", default="/hfcache/datasets")
    parser.add_argument("--corpus-text-file", default=None,
                        help="Materialized corpus text (see "
                             "tools/materialize_wikitext_corpus.py). Preferred "
                             "over --dataset-cache-dir: the measurement "
                             "containers carry no `datasets`/pyarrow/pandas.")
    parser.add_argument(
        "--wikitext-inputs",
        help="pre-tokenized offline payload from "
        "tools/prepare_dsv4_wikitext_inputs.py; when given, the tool never "
        "imports `datasets` or a tokenizer",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-spec-decode", action="store_true",
        help="proceed even when the engine has a speculative config. The NLL "
        "is then the DRAFT model's, not the artifact's; the shipcard refuses "
        "such a record (see tools/spec_decode_guard.py).")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    # --dsv4-gridbook-contract forced the one-Spark DSv4 gold-measurement
    # runtime contract; retired 2026-09-02 with its lane, see
    # archive/gridbook_lane_2026-09-02/.
    parser.add_argument(
        "--serve-image",
        default=None,
        help="container image tag this measurement runs in. Required (or "
        "PQ_SERVE_IMAGE): the serve fingerprint is not evidence without it.",
    )
    args = parser.parse_args()

    # Resolve the image before the tokenizer or the model loads: an unnamed
    # image is a reproducibility hole (R15), and discovering it only when the
    # provenance block is written would waste the whole measurement.
    try:
        _resolve_serve_image(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    _validate_workload_args(args)
    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .full_kl_teacher_payload import tokenizer_identity
    except ImportError:
        from full_kl_teacher_payload import tokenizer_identity  # type: ignore
    tokenizer_attestation = tokenizer_identity(args.model)
    tokenizer = None
    if not args.wikitext_inputs:
        # Generic workloads still tokenize arbitrary corpus revisions. An
        # offline --wikitext-inputs payload instead verifies pre-tokenized
        # input below and never needs datasets in the serving image.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )
    ids, dataset_evidence = _load_measurement_ids(
        args,
        tokenizer_attestation=tokenizer_attestation,
        tokenizer=tokenizer,
    )
    chunks = _build_chunks(ids, seqlen=int(args.seqlen))
    expected_tokens_scored = sum(len(chunk) - 1 for chunk in chunks)
    calibration_contract = _ppl_calibration_contract(
        args=args,
        ids=ids,
        dataset_evidence=dataset_evidence,
        tokenizer_identity_sha256=str(
            tokenizer_attestation["content_sha256"]
        ),
        chunks=chunks,
        n_tokens_scored=expected_tokens_scored,
    )

    # Constructing SamplingParams imports vLLM, so even that follows the full
    # corpus/token/window preflight.  No GPU memory is committed before this.
    from vllm import SamplingParams

    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
        detokenize=False,
    )
    llm = _load_llm(args)

    nll = 0.0
    count = 0
    chunk_nlls: list[float] = []
    for index, chunk in enumerate(chunks, 1):
        t0 = time.monotonic()
        result = llm.generate(
            [{"prompt_token_ids": chunk}],
            sampling,
            use_tqdm=False,
        )[0]
        prompt_logprobs = result.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        chunk_nll = 0.0
        for pos in range(1, len(chunk)):
            chunk_nll -= _logprob_value(prompt_logprobs[pos], int(chunk[pos]))
            count += 1
        nll += chunk_nll
        chunk_mean_nll = chunk_nll / (len(chunk) - 1)
        if not math.isfinite(chunk_mean_nll) or chunk_mean_nll < 0.0:
            raise RuntimeError("WikiText chunk produced an invalid mean NLL")
        chunk_nlls.append(chunk_mean_nll)
        print(
            f"[ppl] chunk {index}/{len(chunks)} tokens={len(chunk)} "
            f"wall={time.monotonic() - t0:.2f}s",
            flush=True,
        )

    if count != expected_tokens_scored:
        raise RuntimeError("vLLM result count differs from the preflight contract")
    mean_nll = nll / count
    if not math.isfinite(mean_nll) or mean_nll < 0.0:
        raise RuntimeError("WikiText PPL produced an invalid mean NLL")
    ppl = math.exp(mean_nll)
    if not math.isfinite(ppl) or ppl < 1.0:
        raise RuntimeError("WikiText PPL produced an invalid perplexity")
    result = {
        "model": args.model,
        "quantization": args.quantization,
        "split": args.split,
        "n_tokens_requested": int(args.n_tokens),
        "n_tokens_scored": int(count),
        "seqlen": int(args.seqlen),
        "mean_nll": float(mean_nll),
        "ppl": float(ppl),
        "per_chunk_mean_nll": [float(v) for v in chunk_nlls],
        "max_chunk_mean_nll": float(max(chunk_nlls)) if chunk_nlls else None,
        "calibration_contract": calibration_contract,
        "calibration_contract_sha256": _canonical_sha256(
            calibration_contract
        ),
        "elapsed_s": float(time.monotonic() - started),
        **_provenance(args),
    }
    rendered = _strict_json(result)
    output.write_text(rendered + "\n")
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
