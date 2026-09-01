"""Automated validation harness for PrismaQuant shipped artifacts."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

from .layer_config import is_layer_config_meta_key
from .artifact_registry import (
    DEFAULT_REGISTRY_PATH,
    ArtifactRecord,
    ArtifactRegistry,
    canonical_layer_config_json,
    layer_config_sha256,
    new_record_id,
    utc_timestamp,
)


DEFAULT_CACHE_DIR = Path("/home/rob/dq-runs/prismaquant-validation-cache")
MMLU_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_VALIDATION_GRAPH_CONTEXT_ATTR = "_prismaquant_validation_graph_context"
_VALIDATION_GRAPH_CONTEXT_MISSING = object()
_VALIDATION_CUDA_GRAPH_REGISTRY = None


def _validation_cuda_graph_registry():
    global _VALIDATION_CUDA_GRAPH_REGISTRY
    if _VALIDATION_CUDA_GRAPH_REGISTRY is None:
        from .kl_measurement import CUDAGraphRegistry

        _VALIDATION_CUDA_GRAPH_REGISTRY = CUDAGraphRegistry(
            label="validation",
            max_entries=4,
            max_entries_env="PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE",
        )
    return _VALIDATION_CUDA_GRAPH_REGISTRY


def _validation_cuda_graph_enabled() -> bool:
    from .kl_measurement import _env_flag_enabled

    return _env_flag_enabled("PRISMAQUANT_VALIDATION_CUDA_GRAPHS", default=True)


def _validation_graph_context(model):
    return getattr(model, _VALIDATION_GRAPH_CONTEXT_ATTR, ("unperturbed",))


def _validation_cuda_graph_run(label: str, model, fn, *args, **kwargs):
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        pass
    return _validation_cuda_graph_registry().run(
        label,
        (id(model), _validation_graph_context(model)),
        fn,
        *args,
        enabled=_validation_cuda_graph_enabled(),
        device=device,
        keepalive=(model,),
        **kwargs,
    )


def validate_artifact(
    model_path: str,
    layer_config: dict | str | Path,
    *,
    cache_dir: Path,
    device: str = "cuda",
    dtype: str = "bf16",
    n_wikitext_tokens: int = 65536,
    n_mmlu_questions: int = 200,
    calib_seqlen: int = 512,
    calib_n_samples: int = 8,
    calib_split: str = "test",
    calib_seed: int = 42,
    progress: bool = True,
    _metric_backend: Callable[..., Mapping[str, Any]] | None = None,
) -> dict:
    """Return PPL, MMLU accuracy, End-KL, config hashes, and runtime."""
    started = time.monotonic()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_payload, _config_path = _load_layer_config_input(layer_config)
    config_sha = layer_config_sha256(config_payload)

    backend = _metric_backend or _env_metric_backend() or _compute_metrics
    raw_metrics = dict(
        backend(
            model_path=model_path,
            layer_config=config_payload,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            n_wikitext_tokens=int(n_wikitext_tokens),
            n_mmlu_questions=int(n_mmlu_questions),
            calib_seqlen=int(calib_seqlen),
            calib_n_samples=int(calib_n_samples),
            calib_split=str(calib_split),
            calib_seed=int(calib_seed),
            progress=bool(progress),
        )
    )

    metrics = {
        "ppl_wikitext": _finite_metric(raw_metrics, "ppl_wikitext"),
    }
    raw_end_kl = raw_metrics.get("end_kl")
    if raw_end_kl is None:
        raise KeyError("validation backend did not return 'end_kl'")
    end_kl_value = float(raw_end_kl)
    skip_end_kl = os.environ.get(
        "PRISMAQUANT_VALIDATION_SKIP_END_KL", "0"
    ) not in ("", "0", "false", "False")
    if math.isnan(end_kl_value) and skip_end_kl:
        metrics["end_kl"] = end_kl_value
    else:
        metrics["end_kl"] = _finite_metric(raw_metrics, "end_kl")
    # MMLU is skippable via n_mmlu_questions=0 → backend returns NaN.
    # Treat NaN as "metric intentionally omitted" so a partial survey is
    # still useful for the metrics that *were* requested.
    raw_mmlu = raw_metrics.get("ppl_mmlu_acc")
    if raw_mmlu is None:
        raise KeyError("validation backend did not return 'ppl_mmlu_acc'")
    mmlu_value = float(raw_mmlu)
    if math.isnan(mmlu_value) and int(n_mmlu_questions) <= 0:
        metrics["ppl_mmlu_acc"] = mmlu_value  # explicit-skip sentinel
    else:
        metrics["ppl_mmlu_acc"] = _finite_metric(raw_metrics, "ppl_mmlu_acc")
    metrics["model_sha"] = _sha256_model_reference(model_path)
    metrics["layer_config_sha"] = config_sha
    metrics["eval_seconds"] = float(time.monotonic() - started)
    # Metric-era marker (QC on review-batch): the end-KL eval split moved
    # from wikitext TRAIN to TEST on 2026-06-12 — pre-marker registry
    # records were measured on train and are NOT comparable at face value.
    # Records without 'eval_split' are the train era.
    metrics["eval_split"] = "test"
    metrics["metric_era"] = "2026-06-12-test-split"
    return metrics


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(f"validation backend did not return {key!r}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"validation metric {key!r} is not finite: {value!r}")
    return value


def _env_metric_backend() -> Callable[..., Mapping[str, Any]] | None:
    raw = os.environ.get("PRISMAQUANT_VALIDATION_FAKE_METRICS")
    if not raw:
        return None
    payload = json.loads(raw)

    def _backend(**_kwargs) -> Mapping[str, Any]:
        return payload

    return _backend


def _compute_metrics(
    *,
    model_path: str,
    layer_config: Mapping[str, Any],
    cache_dir: Path,
    device: str,
    dtype: str,
    n_wikitext_tokens: int,
    n_mmlu_questions: int,
    calib_seqlen: int,
    calib_n_samples: int,
    calib_split: str,
    calib_seed: int,
    progress: bool,
) -> dict:
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "validate_artifact requires torch, transformers, and datasets for "
            "the real backend; use _metric_backend or "
            "PRISMAQUANT_VALIDATION_FAKE_METRICS in lightweight tests"
        ) from exc

    torch_device = _resolve_torch_device(torch, device)
    torch_dtype = _torch_dtype(torch, dtype, torch_device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(cache_dir / "hf"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=str(cache_dir / "hf"),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(torch_device)
    model.eval()

    assignment = _layer_config_to_assignment(layer_config)
    cal_hash = hashlib.blake2b(
        canonical_layer_config_json(layer_config).encode("utf-8"),
        digest_size=16,
    ).hexdigest()

    with torch.no_grad():
        with _perturbed_model(model, assignment, cache_dir, cal_hash):
            ppl_wikitext = _wikitext_ppl(
                model=model,
                tokenizer=tokenizer,
                load_dataset=load_dataset,
                cache_dir=cache_dir,
                device=torch_device,
                n_tokens=n_wikitext_tokens,
                progress=progress,
            )
            ppl_mmlu_acc = _mmlu_accuracy(
                model=model,
                tokenizer=tokenizer,
                load_dataset=load_dataset,
                cache_dir=cache_dir,
                device=torch_device,
                n_questions=n_mmlu_questions,
                progress=progress,
            )
        if os.environ.get("PRISMAQUANT_VALIDATION_SKIP_END_KL", "0") not in ("", "0", "false", "False"):
            end_kl = float("nan")
        else:
            end_kl = _end_kl(
                model=model,
                tokenizer=tokenizer,
                load_dataset=load_dataset,
                assignment=assignment,
                cache_dir=cache_dir,
                device=torch_device,
                calib_seqlen=calib_seqlen,
                calib_n_samples=calib_n_samples,
                calib_split=calib_split,
                calib_seed=calib_seed,
                cal_hash=cal_hash,
                progress=progress,
            )
    return {
        "ppl_wikitext": float(ppl_wikitext),
        "ppl_mmlu_acc": float(ppl_mmlu_acc),
        "end_kl": float(end_kl),
    }


def _resolve_torch_device(torch, device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for validation but is not available")
    return torch.device(device)


def _torch_dtype(torch, dtype: str, device) -> Any:
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16 if device.type != "cpu" else torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if device.type != "cpu" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype!r}")


@contextlib.contextmanager
def _perturbed_model(
    model,
    assignment: Mapping[str, str],
    cache_dir: Path,
    cal_hash: str,
):
    active = _active_assignment(assignment)
    if not active:
        yield
        return
    from .perturbed_x_cache import PerturbedActivationCache

    prod_cache_obj = None
    prod_cache_path = os.environ.get("PRISMAQUANT_VALIDATION_PROD_CACHE")
    if prod_cache_path:
        import pickle as _pickle
        with open(prod_cache_path, "rb") as _fh:
            prod_cache_obj = _pickle.load(_fh)
        override_dir = os.environ.get("PRISMAQUANT_VALIDATION_PROD_CACHE_DIR")
        if override_dir and hasattr(prod_cache_obj, "relocate"):
            prod_cache_obj.relocate(override_dir)
        lru_gb = float(os.environ.get("PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB", "16"))
        if lru_gb > 0 and hasattr(prod_cache_obj, "enable_lru"):
            prod_cache_obj.enable_lru(int(lru_gb * 1024**3))

    hooks = PerturbedActivationCache(
        model,
        active,
        cache_dir / "validation_perturbed_hooks",
        input_rows=0,
        cal_hash=cal_hash,
        production_weight_cache=prod_cache_obj,
    )
    if hooks.missing:
        preview = ", ".join(hooks.missing[:8])
        suffix = " ..." if len(hooks.missing) > 8 else ""
        raise ValueError(
            "layer_config entries did not match model parameters: "
            f"{preview}{suffix}"
        )
    previous_context = getattr(
        model,
        _VALIDATION_GRAPH_CONTEXT_ATTR,
        _VALIDATION_GRAPH_CONTEXT_MISSING,
    )
    setattr(
        model,
        _VALIDATION_GRAPH_CONTEXT_ATTR,
        ("perturbed", tuple(sorted((str(k), str(v)) for k, v in active.items()))),
    )
    try:
        hooks.install()
        try:
            yield
        finally:
            hooks.remove()
    finally:
        if previous_context is _VALIDATION_GRAPH_CONTEXT_MISSING:
            try:
                delattr(model, _VALIDATION_GRAPH_CONTEXT_ATTR)
            except AttributeError:
                pass
        else:
            setattr(model, _VALIDATION_GRAPH_CONTEXT_ATTR, previous_context)


def _active_assignment(assignment: Mapping[str, str]) -> dict[str, str]:
    passthrough = {"BF16", "FP16", "FLOAT16", "FLOAT", "FP8_SOURCE"}
    return {
        name: fmt
        for name, fmt in assignment.items()
        if str(fmt).upper() not in passthrough
    }


def _wikitext_ppl(
    *,
    model,
    tokenizer,
    load_dataset,
    cache_dir: Path,
    device,
    n_tokens: int,
    progress: bool,
) -> float:
    input_ids = _load_wikitext_ids(
        tokenizer,
        load_dataset,
        cache_dir=cache_dir,
        split="test",
        n_tokens=max(int(n_tokens), 2),
    )
    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        raise ValueError("WikiText tokenization produced fewer than two tokens")

    # Chunked (non-overlapping-window) PPL: the token stream is split into
    # disjoint windows and each window is scored independently, so the first
    # token of every window sees no context from the previous window. This is
    # deliberately NOT strided/sliding-window evaluation. The env knob keeps
    # its historical name for compatibility but sets the window (chunk) size.
    max_window = 2048
    raw_window = os.environ.get("PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE")
    if raw_window:
        try:
            max_window = max(2, int(raw_window))
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE must be an integer"
            ) from exc
    window_len = min(_model_context_length(model, tokenizer), max_window, seq_len)
    window_len = max(int(window_len), 2)
    nll_sum = 0.0
    token_count = 0
    starts = range(0, seq_len, window_len)
    for begin in _progress_iter(starts, progress, "wikitext-ppl"):
        end = min(begin + window_len, seq_len)
        window = input_ids[:, begin:end].to(device)
        if window.size(1) < 2:
            break  # a 1-token remainder window has no prediction targets
        labels = window.clone()
        # The HF causal-LM loss is the mean NLL over the window's shifted
        # targets — (L - 1) of them, the first token has no prediction.
        # Weight each window by that true target count so the remainder
        # window is not overweighted by L / (L - 1).
        n_targets = int(window.size(1)) - 1

        def _loss_forward(input_ids, target_labels):
            return model(input_ids, labels=target_labels).loss

        # When output cloning is disabled this scalar is read immediately and
        # never held across another validation graph replay.
        loss = _validation_cuda_graph_run(
            "wikitext-ppl-loss",
            model,
            _loss_forward,
            window,
            labels,
        )
        nll_sum += float(loss.detach().float().item()) * float(n_targets)
        token_count += int(n_targets)
        del loss, labels, window
        if getattr(device, "type", None) == "cuda":
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        if end >= seq_len:
            break
    if token_count <= 0:
        raise ValueError("WikiText PPL accumulated zero target tokens")
    return float(math.exp(nll_sum / float(token_count)))


def _load_wikitext_ids(
    tokenizer,
    load_dataset,
    *,
    cache_dir: Path,
    split: str,
    n_tokens: int | None = None,
):
    last_error: Exception | None = None
    for dataset_name in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(
                dataset_name,
                "wikitext-2-raw-v1",
                split=split,
                cache_dir=str(cache_dir / "datasets"),
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    text = "\n\n".join(row["text"] for row in ds if row.get("text", "").strip())
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    if n_tokens is not None:
        ids = ids[:, : int(n_tokens)]
    return ids.contiguous()


def _mmlu_accuracy(
    *,
    model,
    tokenizer,
    load_dataset,
    cache_dir: Path,
    device,
    n_questions: int,
    progress: bool,
) -> float:
    if int(n_questions) <= 0:
        # Caller opted out of MMLU. Return NaN so downstream metric handling
        # treats it as missing instead of failing on a zero-question loop.
        return float("nan")
    rows = _load_mmlu_rows(load_dataset, cache_dir)
    questions = _select_diverse_mmlu(rows, int(n_questions))
    if not questions:
        raise ValueError("MMLU dataset yielded no usable multiple-choice rows")

    correct = 0
    for item in _progress_iter(questions, progress, "mmlu-acc"):
        prompt = _format_mmlu_prompt(item["question"], item["choices"])
        scores = [
            _choice_letter_nll(model, tokenizer, prompt, idx, device)
            for idx in range(len(item["choices"]))
        ]
        predicted = min(range(len(scores)), key=scores.__getitem__)
        correct += int(predicted == item["answer_index"])
    return float(correct) / float(len(questions))


def _load_mmlu_rows(load_dataset, cache_dir: Path):
    errors: list[str] = []
    for dataset_name in ("cais/mmlu", "lukaemon/mmlu"):
        for split in ("test", "validation", "dev"):
            try:
                return load_dataset(
                    dataset_name,
                    "all",
                    split=split,
                    cache_dir=str(cache_dir / "datasets"),
                )
            except Exception as exc:
                errors.append(f"{dataset_name}/{split}: {exc}")
    raise RuntimeError("could not load an MMLU dataset: " + " | ".join(errors[-3:]))


def _select_diverse_mmlu(rows, n_questions: int) -> list[dict]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        item = _normalise_mmlu_row(dict(row))
        if item is None:
            continue
        groups.setdefault(item["subject"], []).append(item)
    selected: list[dict] = []
    offsets = {subject: 0 for subject in groups}
    subjects = sorted(groups)
    while len(selected) < n_questions and subjects:
        advanced = False
        for subject in list(subjects):
            idx = offsets[subject]
            group = groups[subject]
            if idx >= len(group):
                subjects.remove(subject)
                continue
            selected.append(group[idx])
            offsets[subject] = idx + 1
            advanced = True
            if len(selected) >= n_questions:
                break
        if not advanced:
            break
    return selected


def _normalise_mmlu_row(row: Mapping[str, Any]) -> dict | None:
    question = row.get("question") or row.get("input") or row.get("prompt")
    if not isinstance(question, str) or not question.strip():
        return None

    choices = row.get("choices")
    if choices is None:
        choices = [row.get(letter) for letter in ("A", "B", "C", "D")]
    if not isinstance(choices, (list, tuple)) or len(choices) < 2:
        return None
    choices = [str(choice) for choice in choices if choice is not None]
    if len(choices) < 2 or len(choices) > len(MMLU_LETTERS):
        return None

    answer = row.get("answer", row.get("target", row.get("label")))
    answer_index = _answer_to_index(answer, choices)
    if answer_index is None or answer_index >= len(choices):
        return None

    subject = (
        row.get("subject")
        or row.get("category")
        or row.get("task")
        or row.get("subfield")
        or "all"
    )
    return {
        "subject": str(subject),
        "question": question.strip(),
        "choices": choices,
        "answer_index": int(answer_index),
    }


def _answer_to_index(answer: Any, choices: list[str]) -> int | None:
    if isinstance(answer, bool) or answer is None:
        return None
    if isinstance(answer, int):
        return int(answer)
    text = str(answer).strip()
    if text.isdigit():
        return int(text)
    upper = text.upper()
    if len(upper) == 1 and upper in MMLU_LETTERS:
        return MMLU_LETTERS.index(upper)
    for idx, choice in enumerate(choices):
        if text == choice or text.lower() == choice.lower():
            return idx
    return None


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    parts = [f"Question: {question}"]
    for idx, choice in enumerate(choices):
        parts.append(f"{MMLU_LETTERS[idx]}. {choice}")
    parts.append("Answer with the letter of the correct option.\nAnswer:")
    return "\n".join(parts)


def _choice_letter_nll(model, tokenizer, prompt: str, choice_idx: int, device) -> float:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    answer_ids = tokenizer(
        " " + MMLU_LETTERS[choice_idx],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    if int(answer_ids.numel()) == 0:
        answer_ids = tokenizer(
            MMLU_LETTERS[choice_idx],
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
    max_len = _model_context_length(model, tokenizer)
    max_prompt = max(1, max_len - int(answer_ids.numel()))
    if int(prompt_ids.numel()) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    input_ids = _cat_token_ids(prompt_ids, answer_ids).unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[:, : int(prompt_ids.numel())] = -100

    def _loss_forward(choice_input_ids, choice_labels):
        return model(choice_input_ids, labels=choice_labels).loss

    # When output cloning is disabled this scalar is consumed immediately for
    # the NLL value and is not retained across graph replays.
    loss = _validation_cuda_graph_run(
        "mmlu-choice-loss",
        model,
        _loss_forward,
        input_ids,
        labels,
    )
    return float(loss.detach().float().item()) * float(answer_ids.numel())


def _cat_token_ids(left, right):
    import torch

    return torch.cat([left, right], dim=0)


def _end_kl(
    *,
    model,
    tokenizer,
    load_dataset,
    assignment: Mapping[str, str],
    cache_dir: Path,
    device,
    calib_seqlen: int,
    calib_n_samples: int,
    calib_split: str,
    calib_seed: int,
    cal_hash: str,
    progress: bool,
) -> float:
    active = _active_assignment(assignment)
    if not active:
        return 0.0

    import torch
    from .kl_measurement import measure_assignment_kl

    calib_ids = _fixed_calib_ids(
        tokenizer,
        load_dataset,
        cache_dir=cache_dir,
        n_samples=int(calib_n_samples),
        seqlen=int(calib_seqlen),
        split=str(calib_split),
        seed=int(calib_seed),
    )
    ref_log_probs = []
    for i in _progress_iter(range(calib_ids.size(0)), progress, "end-kl-ref"):
        batch = calib_ids[i:i + 1].to(device)

        def _logits_forward(batch_ids):
            return model(batch_ids).logits[:, -1:, :].clone()

        # The logits are immediately transformed into a detached log-softmax
        # tensor before the next replay can overwrite the static graph output.
        logits = _validation_cuda_graph_run(
            "end-kl-ref-logits",
            model,
            _logits_forward,
            batch,
        )
        ref_log_probs.append(torch.log_softmax(logits.float(), dim=-1).detach())
    work_root = cache_dir / "end_kl"
    work_root.mkdir(parents=True, exist_ok=True)
    return float(
        measure_assignment_kl(
            model,
            active,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            # The reference above is built at the last token only
            # (logits[:, -1:, :]); the KL scope MUST match it. Left
            # unspecified, the scope resolves from PRISMAQUANT_FULL_SEQUENCE_KL
            # and a [1,1,V] teacher silently broadcasts against [1,T,V]
            # student log-probs (audit M7). measure_assignment_kl also
            # hard-fails on any teacher/student shape mismatch now.
            kl_scope="last_token",
        )
    )


def _fixed_calib_ids(
    tokenizer,
    load_dataset,
    *,
    cache_dir: Path,
    n_samples: int,
    seqlen: int,
    split: str = "test",
    seed: int = 42,
):
    import torch

    ids = _load_wikitext_ids(
        tokenizer,
        load_dataset,
        cache_dir=cache_dir,
        split=split,
        n_tokens=None,
    )[0]
    if int(ids.numel()) < seqlen + 1:
        repeats = math.ceil((seqlen + 1) / max(int(ids.numel()), 1))
        ids = ids.repeat(repeats)
    max_start = max(int(ids.numel()) - int(seqlen), 0)
    rng = random.Random(int(seed))
    if max_start >= n_samples:
        starts = rng.sample(range(max_start), n_samples)
    else:
        starts = [
            min(max_start, int(i * max_start / max(n_samples, 1)))
            for i in range(n_samples)
        ]
    return torch.stack([ids[s:s + seqlen] for s in starts], dim=0).contiguous()


def _model_context_length(model, tokenizer) -> int:
    cfg = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(cfg, name, None)
        if isinstance(value, int) and 1 < value < 10_000_000:
            return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 1 < value < 10_000_000:
        return value
    return 2048


def _progress_iter(iterable, progress: bool, desc: str):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc)


def _load_layer_config_input(layer_config: Mapping | str | Path) -> tuple[dict, Path | None]:
    if isinstance(layer_config, Mapping):
        return dict(layer_config), None
    path = Path(layer_config)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: layer_config is not a JSON object")
    return payload, path


def _layer_config_to_assignment(layer_config: Mapping[str, Any]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for name, entry in layer_config.items():
        if is_layer_config_meta_key(name):
            continue
        fmt = _entry_format_name(entry)
        if fmt is None:
            raise ValueError(f"could not map layer_config entry for {name!r}: {entry!r}")
        assignment[str(name)] = fmt
    return assignment


def _entry_format_name(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, int) and not isinstance(entry, bool):
        return {16: "BF16", 8: "INT8_W8A16", 4: "INT4_W4A16_g128"}.get(entry)
    if not isinstance(entry, Mapping):
        return None

    try:
        from . import format_registry as fr

        for name, spec in fr.REGISTRY.items():
            cfg = spec.autoround_config()
            keys = set(entry) | set(cfg)
            if all(entry.get(key) == cfg.get(key) for key in keys):
                return name
    except Exception:
        pass

    data_type = str(entry.get("data_type", "")).lower()
    bits = _optional_int(entry.get("bits"))
    group_size = _optional_int(entry.get("group_size"))
    act_bits = _optional_int(entry.get("act_bits"))

    if data_type == "float" and bits in {16, 32}:
        return "BF16"
    if data_type == "nv_fp" and bits == 4:
        return "NVFP4" if act_bits == 4 else "NVFP4A16"
    if data_type == "mx_fp":
        if bits == 4:
            return "MXFP4"
        if bits == 8:
            elt = str(entry.get("weight_element_dtype", "fp8_e4m3")).lower()
            if elt == "fp8_e5m2":
                return "MXFP8_E5M2" if act_bits == 8 else None
            return "MXFP8_E4M3" if act_bits == 8 else "MXFP8A16"
    if data_type == "int":
        if bits == 8:
            return "INT8_W8A16"
        if bits == 4:
            return "INT4_W4A16_g128"
    if data_type in {"fp8_e4m3", "fp8_e5m2"}:
        if group_size == 128 and act_bits == 16:
            return "FP8_SOURCE"
        return "FP8_E4M3" if data_type == "fp8_e4m3" else "FP8_E5M2"
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_model_reference(model_path: str) -> str:
    path = Path(model_path)
    h = hashlib.sha256()
    if path.is_file():
        _hash_file(h, path)
        return h.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            _hash_file(h, child)
        return h.hexdigest()
    h.update(f"hf:{model_path}".encode("utf-8"))
    return h.hexdigest()


def _hash_file(h: "hashlib._Hash", path: Path) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)


def _format_histogram_from_layer_config(layer_config: Mapping[str, Any]) -> dict:
    counts: dict[str, int] = {}
    for name, entry in layer_config.items():
        if is_layer_config_meta_key(name):
            continue
        fmt = _entry_format_name(entry) or _entry_label(entry)
        counts[fmt] = counts.get(fmt, 0) + 1
    return dict(sorted(counts.items()))


def _entry_label(entry: Any) -> str:
    if isinstance(entry, Mapping):
        data_type = entry.get("data_type", "unknown")
        bits = entry.get("bits", "?")
        return f"{data_type}:{bits}"
    return str(entry)


def _infer_target_bpp(path: Path | None) -> float:
    if path is None:
        return 0.0
    match = re.search(r"bpp[_-]([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else 0.0


def _main_validate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismaquant.validation_harness")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer-config", required=True)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-wikitext-tokens", type=int, default=65536)
    ap.add_argument("--n-mmlu-questions", type=int, default=200)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--calib-n-samples", type=int, default=8)
    ap.add_argument("--calib-split", default="test")
    ap.add_argument("--calib-seed", type=int, default=42)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    ap.add_argument("--artifact-path", default=None)
    ap.add_argument("--target-bpp", type=float, default=None)
    ap.add_argument("--achieved-bpp", type=float, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    metrics = validate_artifact(
        args.model,
        args.layer_config,
        cache_dir=Path(args.cache_dir),
        device=args.device,
        dtype=args.dtype,
        n_wikitext_tokens=args.n_wikitext_tokens,
        n_mmlu_questions=args.n_mmlu_questions,
        calib_seqlen=args.calib_seqlen,
        calib_n_samples=args.calib_n_samples,
        calib_split=args.calib_split,
        calib_seed=args.calib_seed,
        progress=args.progress,
    )
    output = dict(metrics)
    if args.register:
        layer_config, layer_config_path = _load_layer_config_input(args.layer_config)
        target_bpp = (
            float(args.target_bpp)
            if args.target_bpp is not None
            else _infer_target_bpp(layer_config_path)
        )
        achieved_bpp = (
            float(args.achieved_bpp)
            if args.achieved_bpp is not None
            else target_bpp
        )
        record = ArtifactRecord(
            record_id=new_record_id(),
            model_path=args.model,
            artifact_path=args.artifact_path,
            layer_config_sha=metrics["layer_config_sha"],
            layer_config_path=str(layer_config_path) if layer_config_path else None,
            target_bpp=target_bpp,
            achieved_bpp=achieved_bpp,
            format_histogram=_format_histogram_from_layer_config(layer_config),
            ppl_wikitext=metrics["ppl_wikitext"],
            ppl_mmlu_acc=metrics["ppl_mmlu_acc"],
            end_kl=metrics["end_kl"],
            eval_meta={
                "model_sha": metrics["model_sha"],
                "n_wikitext_tokens": args.n_wikitext_tokens,
                "n_mmlu_questions": args.n_mmlu_questions,
                "calib_seqlen": args.calib_seqlen,
                "calib_n_samples": args.calib_n_samples,
                "calib_split": args.calib_split,
                "calib_seed": args.calib_seed,
                "device": args.device,
                "dtype": args.dtype,
            },
            created_at=utc_timestamp(),
            notes=args.notes,
        )
        ArtifactRegistry(args.registry).add(record)
        output["record_id"] = record.record_id
        output["registry"] = args.registry
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _main_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismaquant.validation_harness compare")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--baseline-id", required=True)
    args = ap.parse_args(argv)
    result = ArtifactRegistry(args.registry).compare(args.candidate_id, args.baseline_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


def _main_list(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismaquant.validation_harness list")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    records = [record.to_dict() for record in ArtifactRegistry(args.registry).find_by_model(args.model)]
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "compare":
        return _main_compare(args[1:])
    if args and args[0] == "list":
        return _main_list(args[1:])
    return _main_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
