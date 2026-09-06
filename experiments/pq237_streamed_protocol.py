"""Immutable inputs for the bounded PQ237 streamed qualification.

Preparation tokenizes on CPU. Import and verification use only the standard
library; this module neither imports torch nor loads a model.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from collections.abc import Mapping


PROBES = 128
SEED = 237000
LENGTH = 64
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SCHEMA = "prismaquant.pq237.streamed_protocol.v1"
SCOPE = "descriptive_panel_only"
A4 = "TESSERA_E2M1_K1_R768"
A8 = "TESSERA_E4M3_K1_R896"
A16 = "TESSERA_BF16_K1_R896"
PLAN = {
    "model.layers.0.mlp.down_proj": (A8, A16),
    "model.layers.7.mlp.down_proj": (A16,),
    "model.layers.21.mlp.down_proj": (A4, A16),
}
ASSIGNMENTS = {
    f"L0{left}_L21{right}": dict(zip(PLAN, (left_fmt, A16, right_fmt)))
    for left, left_fmt in (("A8", A8), ("A16", A16))
    for right, right_fmt in (("A4", A4), ("A16", A16))
}
SPLIT_COUNTS = {"original_calibration": 2, "original_holdout": 4, "fresh_holdout": 32}
SELECTION = "first_32_distinct_top_level_articles_first_qualifying_paragraph_at_least_1024_characters_v1"
_TOP_HEADING = re.compile(r"^=\s*([^=]+?)\s*=$")
_HASH = re.compile(r"[0-9a-f]{64}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_texts() -> dict[str, list[str]]:
    path = Path(__file__).with_name("pq237_joint_aura_screen.py")
    assignments = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CALIBRATION", "HELDOUT"}:
                    assignments[target.id] = ast.literal_eval(node.value)
    return {"original_calibration": assignments["CALIBRATION"],
            "original_holdout": assignments["HELDOUT"]}


def _model_files(root: Path) -> dict:
    files = {str(path.relative_to(root)): {"sha256": _file_sha(path), "size": path.stat().st_size}
             for path in sorted(root.rglob("*")) if path.is_file()}
    if not files or "config.json" not in files or not any("tokenizer" in name for name in files):
        raise ValueError("model directory needs config, tokenizer and checkpoint files")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("model directory has no checkpoint weight files")
    return files


def _select_articles(text_rows) -> list[dict]:
    """Take each article's first qualifying paragraph in source order."""
    selected, seen_titles = [], set()
    pending = None
    for row_index, raw_text in enumerate(text_rows):
        text = str(raw_text).strip()
        if not text:
            continue
        heading = _TOP_HEADING.fullmatch(text)
        if heading:
            title = heading[1].strip()
            duplicate = title.casefold() in seen_titles
            seen_titles.add(title.casefold())
            pending = None if duplicate else (title, row_index)
            continue
        if pending is None:
            continue
        title, heading_row = pending
        if text.startswith("=") or len(text) < 1024:
            continue
        pending = None  # This article has supplied its first qualifying paragraph.
        selected.append({"text": text, "origin": {
            "title": title, "heading_row": heading_row, "paragraph_row": row_index}})
        if len(selected) == SPLIT_COUNTS["fresh_holdout"]:
            return selected
    raise ValueError(f"corpus has fewer than 32 qualifying distinct articles: found {len(selected)}")


def _arrow_texts(path: Path) -> list[str]:
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all().column("text").to_pylist()


def _make_protocol(model_path: Path, corpus_arrow: Path, tokenizer, text_rows) -> dict:
    model_path, corpus_arrow = model_path.resolve(), corpus_arrow.resolve()
    if corpus_arrow.parent.name != REVISION or corpus_arrow.name != "wikitext-validation.arrow":
        raise ValueError("corpus path must identify the pinned WikiText validation revision")
    splits = {}
    paragraphs = {name: [{"text": text, "origin": {"historical_index": i}}
                         for i, text in enumerate(texts)]
                  for name, texts in _historical_texts().items()}
    paragraphs["fresh_holdout"] = _select_articles(text_rows)
    for split, records in paragraphs.items():
        splits[split] = []
        for i, record in enumerate(records):
            token_ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
            if len(token_ids) < LENGTH:
                raise ValueError(f"{split}/{i}: text is shorter than the unpadded sequence length")
            splits[split].append({"sequence_id": f"{split}:{i:03d}", **record,
                                  "text_sha256": _sha(record["text"].encode()),
                                  "tokens": list(token_ids[:LENGTH])})
    return {
        "schema": SCHEMA, "scope": SCOPE,
        "scope_note": "Fixed descriptive sequence panel; no population confidence interval or served qualification.",
        "plan": {name: list(formats) for name, formats in PLAN.items()},
        "assignments": {name: dict(recipe) for name, recipe in ASSIGNMENTS.items()},
        "n_probes": PROBES, "seed_base": SEED,
        "sequence_length": LENGTH,
        "algorithm": {"producer_sha256": _file_sha(Path(__file__)),
                      "historical_screen_sha256": _file_sha(Path(__file__).with_name("pq237_joint_aura_screen.py")),
                      "fresh_selection": SELECTION, "tokenization": "first_64_no_special_tokens_no_padding"},
        "model": {"path": str(model_path), "files": _model_files(model_path)},
        "corpus": {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                   "split": "validation", "revision": REVISION,
                   "arrow_path": str(corpus_arrow), "arrow_sha256": _file_sha(corpus_arrow)},
        "splits": splits,
    }


def _validate_protocol(protocol: dict, *, model_path=None, corpus_arrow=None) -> None:
    expected = {"schema": SCHEMA, "scope": SCOPE, "n_probes": PROBES,
                "seed_base": SEED, "sequence_length": LENGTH,
                "plan": {name: list(formats) for name, formats in PLAN.items()},
                "assignments": ASSIGNMENTS}
    for field, value in expected.items():
        if protocol.get(field) != value or type(protocol.get(field)) is not type(value):
            raise ValueError(f"protocol {field} differs from the frozen experiment")
    algorithm = protocol["algorithm"]
    if algorithm != {"producer_sha256": _file_sha(Path(__file__)),
                     "historical_screen_sha256": _file_sha(Path(__file__).with_name("pq237_joint_aura_screen.py")),
                     "fresh_selection": SELECTION, "tokenization": "first_64_no_special_tokens_no_padding"}:
        raise ValueError("protocol algorithm/source identity mismatch")
    corpus = protocol["corpus"]
    for field, expected_value in {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                                  "split": "validation", "revision": REVISION}.items():
        if corpus.get(field) != expected_value:
            raise ValueError(f"protocol corpus {field} mismatch")
    arrow = Path(corpus_arrow if corpus_arrow is not None else corpus["arrow_path"])
    if _file_sha(arrow) != corpus["arrow_sha256"]:
        raise ValueError("protocol corpus Arrow hash mismatch")
    model = protocol["model"]
    root = Path(model_path if model_path is not None else model["path"])
    if _model_files(root) != model["files"]:
        raise ValueError("protocol model/tokenizer file identity mismatch")
    splits = protocol["splits"]
    if set(splits) != set(SPLIT_COUNTS):
        raise ValueError("protocol split names differ")
    historical = _historical_texts()
    seen_tokens, seen_ids, seen_titles = set(), set(), set()
    previous_paragraph_row = -1
    for split, count in SPLIT_COUNTS.items():
        rows = splits[split]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"protocol {split} must contain exactly {count} sequences")
        for i, row in enumerate(rows):
            sequence_id = row["sequence_id"]
            if sequence_id != f"{split}:{i:03d}" or sequence_id in seen_ids:
                raise ValueError("protocol sequence ID/order mismatch")
            seen_ids.add(sequence_id)
            tokens = row["tokens"]
            if (not isinstance(tokens, list) or len(tokens) != LENGTH
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("protocol tokens must be 64 nonnegative integer IDs")
            token_key = tuple(tokens)
            if token_key in seen_tokens:
                raise ValueError("duplicate token sequence within or across protocol splits")
            seen_tokens.add(token_key)
            text = row["text"]
            if not isinstance(text, str) or _sha(text.encode()) != row["text_sha256"]:
                raise ValueError("protocol paragraph hash mismatch")
            if split in historical:
                if text != historical[split][i] or row["origin"] != {"historical_index": i}:
                    raise ValueError("protocol original text/origin differs from historical screen")
            else:
                origin = row["origin"]
                title = origin["title"]
                heading_row, paragraph_row = origin["heading_row"], origin["paragraph_row"]
                if (not isinstance(title, str) or not title.strip() or title.casefold() in seen_titles
                        or type(heading_row) is not int or type(paragraph_row) is not int
                        or not 0 <= heading_row < paragraph_row or paragraph_row <= previous_paragraph_row
                        or len(text) < 1024):
                    raise ValueError("protocol fresh article origin/order/length mismatch")
                seen_titles.add(title.casefold())
                previous_paragraph_row = paragraph_row


def load_protocol(path, expected_sha256: str, *, model_path=None, corpus_arrow=None) -> dict:
    """Verify the frozen bytes and input files, allowing mapped input locations."""
    if not isinstance(expected_sha256, str) or _HASH.fullmatch(expected_sha256) is None:
        raise ValueError("expected protocol SHA-256 must be supplied independently")
    raw = Path(path).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("immutable protocol digest mismatch")
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate protocol JSON key {key}")
            result[key] = value
        return result
    try:
        protocol = json.loads(raw, object_pairs_hook=no_duplicates)
        _validate_protocol(protocol, model_path=model_path, corpus_arrow=corpus_arrow)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"incomplete protocol: {exc}") from exc
    return protocol


def paired_sequence_summary(rows_a: Mapping[str, float], rows_b: Mapping[str, float]) -> dict:
    """A minus B; a descriptive panel spread, never a population interval."""
    if not isinstance(rows_a, Mapping) or not isinstance(rows_b, Mapping):
        raise ValueError("paired sequences must be ordered mappings")
    ids = list(rows_a)
    if len(ids) < 2 or ids != list(rows_b) or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("paired sequences require at least two identical ordered sequence IDs")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for rows in (rows_a, rows_b) for value in rows.values()):
        raise ValueError("paired sequence values must be finite numbers")
    differences = [float(rows_a[key]) - float(rows_b[key]) for key in ids]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired sequence differences are nonfinite")
    return {"sequence_ids": ids, "n_sequences": len(ids), "difference_per_sequence": differences,
            "mean_difference": statistics.mean(differences),
            "descriptive_sequence_standard_error": statistics.stdev(differences) / math.sqrt(len(ids)),
            "scope": SCOPE,
            "uncertainty_note": "Descriptive SD of paired sequence differences divided by sqrt(n); no population confidence interval."}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus-arrow", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite immutable protocol: {args.out}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    protocol = _make_protocol(args.model, args.corpus_arrow, tokenizer, _arrow_texts(args.corpus_arrow))
    _validate_protocol(protocol)
    raw = (json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({"protocol": str(args.out), "sha256": _sha(raw),
                      "split_counts": SPLIT_COUNTS, "scope": SCOPE}), flush=True)


if __name__ == "__main__":
    main()
