from __future__ import annotations

import json
from pathlib import Path

from prismaquant import pipeline


def _write_header_only_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {
        name: {
            "dtype": "BF16",
            "shape": shape,
            # The policy reads headers only. No tensor payload is needed for
            # this synthetic contract fixture.
            "data_offsets": [0, 0],
        }
        for name, shape in shapes.items()
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


def _model(
    root: Path,
    *,
    parameters: int | None,
    config: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(config or {
            "model_type": "dense_test",
            "architectures": ["DenseTestForCausalLM"],
        }),
        encoding="utf-8",
    )
    if parameters is not None:
        _write_header_only_safetensors(
            root / "model.safetensors",
            {"model.layers.0.weight": [parameters]},
        )
    return root


def test_hooks_is_admitted_only_for_proven_small_dense_checkpoint(tmp_path):
    model = _model(tmp_path / "small", parameters=34_999_999_999)

    policy = pipeline.frontier_materialization_policy(model)
    code, message = pipeline.check_frontier_materialization(model, "hooks")

    assert policy["parameters"] == 34_999_999_999
    assert policy["is_moe"] is False
    assert policy["requires_inplace"] is False
    assert code == 0
    assert "proven dense" in message


def test_hooks_refuses_checkpoint_at_35b_threshold(tmp_path):
    model = _model(
        tmp_path / "large",
        parameters=pipeline.FRONTIER_HOOKS_MAX_PARAMETERS,
    )

    code, message = pipeline.check_frontier_materialization(model, "hooks")

    assert code == 2
    assert "35,000,000,000" in message
    assert "inplace" in message


def test_hooks_refuses_small_moe_from_nested_config(tmp_path):
    model = _model(
        tmp_path / "moe",
        parameters=1_000,
        config={
            "model_type": "wrapper",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_experts": 256,
                "num_experts_per_tok": 8,
            },
        },
    )

    policy = pipeline.frontier_materialization_policy(model)
    code, message = pipeline.check_frontier_materialization(model, "hooks")

    assert policy["is_moe"] is True
    assert code == 2
    assert "routed experts" in message


def test_unknown_checkpoint_refuses_hooks_but_inplace_does_not_need_classification(
    tmp_path,
):
    model = _model(tmp_path / "unknown", parameters=None)

    hooks_code, hooks_message = pipeline.check_frontier_materialization(
        model, "hooks"
    )
    inplace_code, inplace_message = pipeline.check_frontier_materialization(
        model, "inplace"
    )

    assert hooks_code == 2
    assert "cannot prove" in hooks_message
    assert inplace_code == 0
    assert "memory-fit path" in inplace_message


def test_frontier_materialization_cli_is_fail_closed(tmp_path, capsys):
    model = _model(tmp_path / "large-cli", parameters=35_000_000_001)

    code = pipeline.main([
        "--check-frontier-materialization",
        str(model),
        "--frontier-materialization",
        "hooks",
    ])

    assert code == 2
    assert "[pipeline] ERROR:" in capsys.readouterr().out

