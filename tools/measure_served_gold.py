#!/usr/bin/env python3
"""Gold-lane KL and direct PPL against a **served** artifact, over HTTP.

`tools/measure_vllm_full_kl.py` and `tools/measure_vllm_wikitext_ppl.py` are
the DSv4-Flash lane's gold tools: they construct an in-process `LLM`.  They
were written under a closed per-lane runtime contract (the Gridbook lane's,
retired 2026-09-02 -- archive/gridbook_lane_2026-09-02/), which no other model
could satisfy, so every other lane measured its gold numbers with ad-hoc
run-local scripts that emitted a bare metric and none of the identity a ship
record binds -- no serve fingerprint, no producer commit, no spec-decode
state, no workload contract.  That is the gap this fills: the same discipline
for artifacts served the ordinary way, through
`vllm serve` + `/v1/completions`.

Three subcommands, because KL needs two serves and PPL needs one:

    dump  --base-url ... --artifact-dir ... --out arm.json
    kl    --teacher bf16.json --student quant.json --out gold_kl.json
    ppl   --base-url ... --artifact-dir ... --out gold_ppl.json

`kl` and `ppl` write records `python -m prismaquant.shipcard_cli fill` accepts
directly.

**Why the numbers are trustworthy, and where they are not.**

*Serve identity.*  The 64-hex `serve_fingerprint` is not computed here: it is
read from the `serve_manifest.json` that `tools/serve_fingerprint.py write`
produced from INSIDE the serving container (`scripts/lib/serve_manifest.sh`).
The measuring client cannot see the server's address space, which is exactly
why loading a CUDA extension could move the same artifact's conf-KL by +-17%
invisibly (ARCHITECTURE.md 7.4).  This tool checks the manifest describes the
server it is about to talk to -- same served model name, same artifact, no
speculative config -- and refuses otherwise.

*Producer identity.*  `serve_fingerprint.gold_producer_identity` binds the
number to a clean PrismaQuant tree and hashes this file's own bytes.  A gold
measurement from a dirty checkout is refused, not warned about.

*Spec decode.*  Refused from `/metrics` before any request, because with
`--speculative-config` serving, `/v1/completions` logprobs are the DRAFT
model's NLL and the number would silently describe a different network.

*Truncation.*  A served endpoint returns top-K prompt logprobs, never the full
vocabulary, so the KL here is a top-K KL with a declared tail -- NOT the exact
full-vocab KL of metric authority #1.  It is reported honestly as such: every
record carries `prompt_top_k`, the recomputed teacher coverage
(`topk_coverage_mean` / `topk_coverage_min`), and the tail model used for
student probabilities the endpoint did not tabulate.  Two consequences worth
stating plainly:

  * `kl_mean` over ALL positions is floor-inflated wherever the teacher's mass
    is spread past K.  `kl_confident_mean`, restricted to positions where the
    teacher's top-1 probability exceeds 0.5, is the clean number -- there the
    top-K genuinely covers the distribution.  Both ship; the confident one is
    the one to quote.
  * Serving with a larger `--max-logprobs` narrows the gap.  K is recorded so
    two numbers taken at different K are never silently compared.

PPL has no such caveat: `prompt_logprobs=0` returns the logprob of the ACTUAL
token at each position, which is exactly the teacher-forced NLL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import requests

_TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_ROOT))
from prismaquant_source_bootstrap import activate_prismaquant_source

activate_prismaquant_source()

try:  # package mode (`python -m tools.measure_served_gold`)
    from .serve_fingerprint import gold_producer_identity
    from .spec_decode_guard import spec_decode_from_metrics
except ImportError:  # script mode
    from serve_fingerprint import gold_producer_identity  # type: ignore
    from spec_decode_guard import spec_decode_from_metrics  # type: ignore

MEASUREMENT_TOOL = "measure_served_gold"
SERVED_KL_SCHEMA = "prismaquant.served_topk_kl/1"
SERVED_PPL_SCHEMA = "prismaquant.served_wikitext_ppl/1"
SERVED_DUMP_SCHEMA = "prismaquant.served_prompt_logprob_dump/1"
TEACHER_ARM_SCHEMA = "prismaquant.served_kl_teacher_arm/1"
CALIBRATION_SCHEMA = "prismaquant.served_gold_calibration/1"
SERVE_MANIFEST_SCHEMA = "prismaquant.serve_manifest/1"

#: Teacher top-1 probability above which a position's top-K covers the
#: distribution well enough that the untabulated tail cannot matter.
CONFIDENT_TOP1_PROBABILITY = 0.5
DEFAULT_SEQLEN = 512
DEFAULT_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load_serve_manifest(artifact_dir: Path) -> dict[str, Any]:
    """The in-container fingerprint of the serve we are about to measure."""
    path = artifact_dir / "serve_manifest.json"
    if not path.is_file():
        raise SystemExit(
            f"REFUSING: no {path}. A gold number with no serve fingerprint "
            "cannot be compared to any other number (ARCHITECTURE.md 7.4); "
            "run the serve through scripts/lib/serve_manifest.sh."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise SystemExit(f"REFUSING: {path} is not a JSON object")
    if manifest.get("schema") != SERVE_MANIFEST_SCHEMA:
        raise SystemExit(f"REFUSING: {path} has schema {manifest.get('schema')!r}")
    fingerprint = manifest.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise SystemExit(f"REFUSING: {path} carries no serve fingerprint")
    return dict(manifest)


def _check_manifest_describes_this_serve(
    manifest: Mapping[str, Any],
    *,
    base_url: str,
    served_model_name: str,
) -> None:
    """A stale manifest is worse than none: it attests the wrong stack."""
    if manifest.get("served_model_name") != served_model_name:
        raise SystemExit(
            f"REFUSING: serve_manifest.json describes served model "
            f"{manifest.get('served_model_name')!r}, measuring "
            f"{served_model_name!r} -- the manifest is stale"
        )
    if manifest.get("speculative_config") is not None:
        raise SystemExit(
            "REFUSING: serve_manifest.json records a speculative config; "
            "/v1/completions logprobs would be the DRAFT model's"
        )
    try:
        served = requests.get(f"{base_url}/v1/models", timeout=30).json()
    except Exception as exc:                    # noqa: BLE001 - reported as-is
        raise SystemExit(f"REFUSING: cannot reach {base_url}/v1/models: {exc}")
    names = {row.get("id") for row in (served.get("data") or [])}
    if served_model_name not in names:
        raise SystemExit(
            f"REFUSING: {base_url} serves {sorted(n for n in names if n)}, "
            f"not {served_model_name!r}"
        )


def _refuse_spec_decode(base_url: str) -> bool:
    """Return the observed state; refuse anything that is not a clean False."""
    detected = spec_decode_from_metrics(base_url)
    if detected is not False:
        raise SystemExit(
            f"REFUSING: {base_url} reports spec_decode_detected={detected!r}. "
            "None means 'could not inspect', which is what the original trap "
            "looked like; True means the logprobs are the draft model's."
        )
    return False


def _producer_identity() -> dict[str, Any]:
    return gold_producer_identity(MEASUREMENT_TOOL)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def _calibration_contract(
    corpus: Path,
    tokenizer_dir: Path,
    token_ids: list[int],
    *,
    seqlen: int,
    n_tokens_requested: int,
) -> dict[str, Any]:
    """Bind the exact text, tokenizer and window geometry that produced this."""
    tokenizer_files = {
        name: _sha256_file(tokenizer_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
        if (tokenizer_dir / name).is_file()
    }
    return {
        "schema": CALIBRATION_SCHEMA,
        "corpus_path": str(corpus.resolve()),
        "corpus_sha256": _sha256_file(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "tokenizer_files_sha256": tokenizer_files,
        "add_special_tokens": False,
        "n_tokens_requested": n_tokens_requested,
        "n_tokens_selected": len(token_ids),
        "seqlen": seqlen,
        "n_windows": math.ceil(len(token_ids) / seqlen) if token_ids else 0,
        "window_overlap": 0,
        "calib_ids_sha256": hashlib.sha256(
            ",".join(str(i) for i in token_ids).encode("utf-8")
        ).hexdigest(),
    }


def _tokenize(corpus: Path, tokenizer_dir: Path, max_tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=True
    )
    ids = tokenizer.encode(
        corpus.read_text(encoding="utf-8"), add_special_tokens=False
    )
    return list(ids[:max_tokens])


def _windows(token_ids: list[int], seqlen: int) -> list[list[int]]:
    return [
        token_ids[i:i + seqlen]
        for i in range(0, len(token_ids), seqlen)
        if len(token_ids[i:i + seqlen]) >= 2
    ]


def _vocab_size(artifact_dir: Path) -> int:
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    for key in ("vocab_size",):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        value = text_config.get("vocab_size")
        if isinstance(value, int) and value > 0:
            return value
    raise SystemExit(f"REFUSING: no vocab_size in {artifact_dir / 'config.json'}")


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------
def _post_completion(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/v1/completions", json=payload, timeout=1800
    ).json()
    if "choices" not in response:
        raise SystemExit(f"server error: {json.dumps(response)[:400]}")
    return response


def _cmd_dump(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    manifest = _load_serve_manifest(artifact_dir)
    _check_manifest_describes_this_serve(
        manifest, base_url=args.base_url, served_model_name=args.model
    )
    spec = _refuse_spec_decode(args.base_url)
    identity = _producer_identity()

    corpus = Path(args.corpus).resolve()
    tokenizer_dir = Path(args.tokenizer or artifact_dir).resolve()
    token_ids = _tokenize(corpus, tokenizer_dir, args.max_tokens)
    contract = _calibration_contract(
        corpus, tokenizer_dir, token_ids,
        seqlen=args.seqlen, n_tokens_requested=args.max_tokens,
    )

    positions: list[dict[str, float] | None] = []
    for window in _windows(token_ids, args.seqlen):
        response = _post_completion(args.base_url, {
            "model": args.model, "prompt": window, "max_tokens": 1,
            "temperature": 0.0, "prompt_logprobs": args.top_k,
            "add_special_tokens": False,
        })
        for entry in response["choices"][0].get("prompt_logprobs") or []:
            # The first position of a window has no predecessor; vLLM returns
            # null there and it is not a scorable next-token event.
            if not entry:
                continue
            positions.append({
                key: (value["logprob"] if isinstance(value, dict) else value)
                for key, value in entry.items()
            })

    payload = {
        "schema": SERVED_DUMP_SCHEMA,
        "model": args.model,
        "artifact_dir": str(artifact_dir),
        "prompt_top_k": args.top_k,
        "seqlen": args.seqlen,
        "n_samples": len(_windows(token_ids, args.seqlen)),
        "n_positions": len(positions),
        "vocab_size": _vocab_size(artifact_dir),
        "spec_decode_detected": spec,
        "serve_fingerprint": manifest["serve_fingerprint"],
        "performance_stack_fingerprint": manifest.get(
            "performance_stack_fingerprint"
        ),
        "serve_manifest": manifest,
        "git_commit": identity["git_commit"],
        "gold_producer_identity": identity,
        "calibration_contract": contract,
        "calibration_contract_sha256": _canonical_sha256(contract),
        "positions": positions,
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"[gold] dumped {len(positions)} positions at top-{args.top_k} "
          f"-> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# kl
# ---------------------------------------------------------------------------
def _tail_logprob(row: Mapping[str, float], vocab_size: int) -> float:
    """Max-entropy estimate for a token the endpoint did not tabulate.

    The endpoint returns K entries; anything outside them shares the residual
    mass `1 - sum(exp(q_j))`.  Spreading it uniformly over the untabulated
    vocabulary is the maximum-entropy choice given what was observed, and it
    is bounded above by the K-th value, so it can never invent a KL term
    larger than the truncation itself allows.  Falling back to `min(row)` --
    what the ad-hoc script did -- systematically UNDERSTATES the divergence,
    because the K-th logprob is an upper bound on every value below it.
    """
    if vocab_size <= len(row):
        # Every token is tabulated; there is no untabulated set to estimate.
        return min(row.values())
    tabulated = sum(math.exp(value) for value in row.values())
    residual = 1.0 - tabulated
    # exp() round-trips are libm-dependent: a row whose true mass is exactly
    # 1.0 can re-exponentiate to 1.0 minus a few ulps, and log() of that
    # phantom residual is ~-36, not "no residual".  The only defensible floor
    # is the dtype's own rounding: summing len(row) values each <= 1 accrues
    # at most one ulp(1.0) of error per addition.
    if residual <= math.ulp(1.0) * len(row):
        return min(row.values())
    return min(math.log(residual / (vocab_size - len(row))), min(row.values()))


def _cmd_kl(args: argparse.Namespace) -> int:
    teacher_payload = json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    student_payload = json.loads(Path(args.student).read_text(encoding="utf-8"))
    for name, payload in (("teacher", teacher_payload), ("student", student_payload)):
        if payload.get("schema") != SERVED_DUMP_SCHEMA:
            raise SystemExit(f"REFUSING: {name} dump has schema "
                             f"{payload.get('schema')!r}")
    if teacher_payload["prompt_top_k"] != student_payload["prompt_top_k"]:
        raise SystemExit(
            "REFUSING: the two arms were dumped at different top-K "
            f"({teacher_payload['prompt_top_k']} vs "
            f"{student_payload['prompt_top_k']}); their truncation floors "
            "differ and the difference would not be the quantization"
        )
    if (teacher_payload["calibration_contract_sha256"]
            != student_payload["calibration_contract_sha256"]):
        raise SystemExit(
            "REFUSING: the two arms scored different text/geometry; "
            "KL requires both models to see identical inputs"
        )
    vocab_size = int(student_payload["vocab_size"])

    teacher = teacher_payload["positions"]
    student = student_payload["positions"]
    scored = min(len(teacher), len(student))

    all_kl: list[float] = []
    confident_kl: list[float] = []
    coverage: list[float] = []
    agree_all = 0
    agree_confident = 0
    for index in range(scored):
        p_row, q_row = teacher[index], student[index]
        if not p_row or not q_row:
            continue
        q_tail = _tail_logprob(q_row, vocab_size)
        divergence = sum(
            math.exp(lp) * (lp - q_row.get(token, q_tail))
            for token, lp in p_row.items()
        )
        mass = sum(math.exp(lp) for lp in p_row.values())
        coverage.append(min(mass, 1.0))
        all_kl.append(divergence)
        p_top = max(p_row, key=p_row.get)
        matched = p_top == max(q_row, key=q_row.get)
        agree_all += int(matched)
        if math.exp(p_row[p_top]) > CONFIDENT_TOP1_PROBABILITY:
            confident_kl.append(divergence)
            agree_confident += int(matched)
    if not all_kl:
        raise SystemExit("REFUSING: no scorable positions in common")

    identity = _producer_identity()
    record = {
        "schema": SERVED_KL_SCHEMA,
        "model": student_payload["artifact_dir"],
        "mode": "student",
        "score_positions": "all",
        # The served quantization is whatever the serve was launched with, as
        # recorded by serve_fingerprint from the server's own argv. It was a
        # hardcoded "gridbook" until that lane retired 2026-09-02; a hardcoded
        # lane name is not an attestation (see archive/gridbook_lane_2026-09-02/).
        "quantization": student_payload["serve_manifest"].get("quantization"),
        "prompt_top_k": student_payload["prompt_top_k"],
        "vocab_size": vocab_size,
        "n_samples": student_payload["n_samples"],
        "seqlen": student_payload["seqlen"],
        "n_positions": len(all_kl),
        "n_confident": len(confident_kl),
        "kl_mean": sum(all_kl) / len(all_kl),
        "kl_confident_mean": (
            sum(confident_kl) / len(confident_kl) if confident_kl else float("nan")
        ),
        "kl_max": max(all_kl),
        "top1_agreement_all": agree_all / len(all_kl),
        "top1_agreement_confident": (
            agree_confident / len(confident_kl) if confident_kl else float("nan")
        ),
        "topk_coverage_mean": sum(coverage) / len(coverage),
        "topk_coverage_min": min(coverage),
        "student_tail_model": "uniform_over_untabulated_vocabulary",
        "confident_top1_probability": CONFIDENT_TOP1_PROBABILITY,
        "spec_decode_detected": student_payload["spec_decode_detected"],
        "serve_fingerprint": student_payload["serve_fingerprint"],
        "serve_manifest": student_payload["serve_manifest"],
        "git_commit": identity["git_commit"],
        "gold_producer_identity": identity,
        "calibration_contract": student_payload["calibration_contract"],
        "calibration_contract_sha256": student_payload[
            "calibration_contract_sha256"
        ],
        "teacher_evidence": {
            "schema": TEACHER_ARM_SCHEMA,
            "artifact_dir": teacher_payload["artifact_dir"],
            "serve_fingerprint": teacher_payload["serve_fingerprint"],
            "spec_decode_detected": teacher_payload["spec_decode_detected"],
            "git_commit": teacher_payload["git_commit"],
            "prompt_top_k": teacher_payload["prompt_top_k"],
            "n_positions": teacher_payload["n_positions"],
            "dump_sha256": _sha256_file(Path(args.teacher)),
        },
    }
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({
        key: record[key] for key in (
            "n_positions", "n_confident", "kl_mean", "kl_confident_mean",
            "top1_agreement_confident", "topk_coverage_mean",
            "topk_coverage_min",
        )
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# ppl
# ---------------------------------------------------------------------------
def _cmd_ppl(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    manifest = _load_serve_manifest(artifact_dir)
    _check_manifest_describes_this_serve(
        manifest, base_url=args.base_url, served_model_name=args.model
    )
    spec = _refuse_spec_decode(args.base_url)
    identity = _producer_identity()

    corpus = Path(args.corpus).resolve()
    tokenizer_dir = Path(args.tokenizer or artifact_dir).resolve()
    token_ids = _tokenize(corpus, tokenizer_dir, args.max_tokens)
    contract = _calibration_contract(
        corpus, tokenizer_dir, token_ids,
        seqlen=args.seqlen, n_tokens_requested=args.max_tokens,
    )

    nlls: list[float] = []
    per_window: list[float] = []
    for window in _windows(token_ids, args.seqlen):
        response = _post_completion(args.base_url, {
            "model": args.model, "prompt": window, "max_tokens": 1,
            "temperature": 0.0, "prompt_logprobs": 0,
            "add_special_tokens": False,
        })
        window_nll: list[float] = []
        for entry, token_id in zip(
            response["choices"][0].get("prompt_logprobs") or [], window
        ):
            if not entry:
                continue                # first position: nothing predicts it
            value = entry.get(str(token_id))
            if value is None:
                continue
            window_nll.append(
                -(value["logprob"] if isinstance(value, dict) else value)
            )
        if not window_nll:
            continue
        nlls.extend(window_nll)
        per_window.append(sum(window_nll) / len(window_nll))
    if not nlls:
        raise SystemExit("REFUSING: no scored tokens")

    mean_nll = sum(nlls) / len(nlls)
    record = {
        "schema": SERVED_PPL_SCHEMA,
        "model": str(artifact_dir),
        "split": args.split,
        "n_tokens_requested": args.max_tokens,
        "n_tokens_scored": len(nlls),
        "n_samples": len(per_window),
        "seqlen": args.seqlen,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "per_chunk_mean_nll": per_window,
        "max_chunk_mean_nll": max(per_window),
        # As above: read the served quantization from the server's own argv.
        "quantization": manifest.get("quantization"),
        "spec_decode_detected": spec,
        "serve_fingerprint": manifest["serve_fingerprint"],
        "serve_manifest": manifest,
        "git_commit": identity["git_commit"],
        "gold_producer_identity": identity,
        "calibration_contract": contract,
        "calibration_contract_sha256": _canonical_sha256(contract),
    }
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({
        key: record[key]
        for key in ("n_tokens_scored", "n_samples", "mean_nll", "ppl",
                    "max_chunk_mean_nll")
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_serve_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--base-url", default="http://localhost:8000")
        target.add_argument("--model", required=True,
                            help="the --served-model-name of the live server")
        target.add_argument("--artifact-dir", required=True,
                            help="local artifact dir holding serve_manifest.json")
        target.add_argument("--corpus", required=True)
        target.add_argument("--tokenizer", default=None,
                            help="defaults to the artifact dir")
        target.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
        target.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        target.add_argument("--out", required=True)

    dump = sub.add_parser("dump", help="save per-position top-K prompt logprobs")
    add_serve_arguments(dump)
    dump.add_argument("--top-k", type=int, required=True,
                      help="must not exceed the server's --max-logprobs")
    dump.set_defaults(func=_cmd_dump)

    kl = sub.add_parser("kl", help="teacher/student dumps -> a gold.kl record")
    kl.add_argument("--teacher", required=True)
    kl.add_argument("--student", required=True)
    kl.add_argument("--out", required=True)
    kl.set_defaults(func=_cmd_kl)

    ppl = sub.add_parser("ppl", help="teacher-forced PPL -> a gold.ppl record")
    add_serve_arguments(ppl)
    ppl.add_argument("--split", default="wikitext-2-raw-v1/test")
    ppl.set_defaults(func=_cmd_ppl)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
