"""Issue #210: `sensitivity_probe.stage_text_only` and
`perturbed_x_cache.stage_text_only_under_work_root` used to carry two
separately-typed copies of the same default strip-key list and staging
body. Both now delegate to one shared implementation
(`sensitivity_probe._stage_text_only_impl`), parameterised on the staging
root. This test pins the contract both public names still owe callers:
staging the SAME checkpoint through either name produces a byte-identical
`config.json`, differing only in where it lands.

Pre-fix status (recorded per AGENTS.md 14 / the issue's own ask to
establish this before promoting severity): on `main`, before this merge,
the two duplicated bodies already produced byte-identical output for this
checkpoint too -- verified by hand against the pre-merge functions with the
same fixture below. Neither duplicate's hardcoded fallback strip-key list
was reachable for this fixture: `detect_profile` resolves an unregistered
architecture to `DefaultProfile` (a real profile, not `None`), so both
paths took the *same* `profile.stage_text_only_strip_keys()` branch, not
the hardcoded literal. This test is therefore a pin, not a
failing-first regression -- see the issue and the commit for the fuller
answer to "could the two paths already disagree for a live profile",
which is about the hardcoded literal vs. profile-declared lists, not about
these two call sites disagreeing with each other.
"""
import json
from pathlib import Path

from prismaquant.sensitivity_probe import stage_text_only
from prismaquant.perturbed_x_cache import stage_text_only_under_work_root


def _write_synthetic_multimodal_moe_checkpoint(root: Path) -> Path:
    """A checkpoint no registered ModelProfile claims (unregistered
    model_type/architectures), carrying every key either duplicated
    staging function's hardcoded fallback list named, plus a MoE alias
    key and a text_config to lift -- so both branches of the staging body
    (strip-key removal, num_local_experts aliasing, text_config lifting,
    architectures rewrite) actually execute."""
    d = root / "synth_model"
    d.mkdir(parents=True)
    cfg = {
        "model_type": "pq210_synthetic_arch",
        "architectures": ["Pq210SyntheticForConditionalGeneration"],
        "vision_config": {"foo": "bar"},
        "audio_config": {"baz": 1},
        "speech_config": {"x": 2},
        "image_token_id": 100,
        "video_token_id": 101,
        "vision_start_token_id": 102,
        "vision_end_token_id": 103,
        "num_local_experts": 8,
        "text_config": {
            "model_type": "pq210_synthetic_text",
            "hidden_size": 2816,
            "num_hidden_layers": 40,
        },
    }
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors.index.json").write_text("{}")
    return d


def test_stage_text_only_and_under_work_root_agree_on_staged_config(tmp_path):
    src = _write_synthetic_multimodal_moe_checkpoint(tmp_path)

    staged_a = Path(stage_text_only(str(src)))
    staged_b = Path(
        stage_text_only_under_work_root(str(src), tmp_path / "work_root")
    )

    # Different homes -- that part of the contract is exactly what the two
    # public names still differ on.
    assert staged_a != staged_b
    assert staged_a.parent != (tmp_path / "work_root")
    assert staged_b.parent == (tmp_path / "work_root")

    # Same rule, same answer: byte-identical config.json apart from location.
    bytes_a = (staged_a / "config.json").read_bytes()
    bytes_b = (staged_b / "config.json").read_bytes()
    assert bytes_a == bytes_b

    cfg = json.loads(bytes_a)
    # Sanity: the shared body actually ran (didn't just echo the source).
    assert "vision_config" not in cfg
    assert "text_config" not in cfg
    assert cfg["hidden_size"] == 2816
    assert cfg["num_experts"] == 8
    assert cfg["architectures"] == ["Pq210SyntheticForCausalLM"]
