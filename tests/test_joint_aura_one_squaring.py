"""PQ #261 run-02: the streamed accumulator squared a signed projection with
``x ** 2`` (C ``pow``) while ``make_joint_aura_entry`` used ``x * x``; on glibc
they differ by one ulp for some inputs, and the exact checkpoint-reload
comparison of the two sample lists refused ("legacy/joint sample alignment
mismatch"). Both sites now call ``squared_signed``.
"""
import ast
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.joint_aura import (activation_identity, arithmetic_identity, identity_sha256,
                                    make_joint_aura_entry, squared_signed)

# Two signed totals recorded in joint-streamed-run-02 whose pow/mul squares
# differ by one ulp on this platform.
RECORDED = [0.09054819587618113, -0.022860320983454585]


def test_squared_signed_is_the_correctly_rounded_multiply():
    for total in RECORDED:
        assert squared_signed(total) == total * total


def test_row_squares_with_squared_signed():
    components = [{"weight": t, "activation": 0.0, "mixed": 0.0, "total": t} for t in RECORDED]
    arithmetic = arithmetic_identity(torch.float32)
    source_content = {"config": {"fixture": True}, "weight_map": {"fixture.weight": "fixture.weight"},
                      "shards": [{"path": "/fixture/synthetic.safetensors", "size": 1, "sha256": "a" * 64}]}
    source_model = {"schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic", "resolved_commit": None,
                    "content_sha256": canonical_json_sha256(source_content, where="one-squaring fixture"),
                    **source_content}
    probe = {"schema": "prismaquant.joint_aura.probes.v1", "seed_base": 0,
             "n_probes": len(RECORDED), "calibration_sha256": "c" * 64,
             "producer_source_sha256": "d" * 64, "source_model": source_model,
             "distribution": "rademacher", "normalization": "global_kl_fisher",
             "temperature": 1.0, "arithmetic": arithmetic}
    name, fmt = "model.layers.0.mlp.down_proj", "NVFP4"
    operator = {"schema": "prismaquant.joint_aura.operator.v1", "qname": name, "format": fmt,
                "probe_identity_sha256": identity_sha256(probe),
                "source_weight": {"content_sha256": "a" * 64, "shape": [64, 64],
                                  "dtype": "torch.float32", "logical_bytes": 16384},
                "rendered_weight": {"content_sha256": "b" * 64, "shape": [64, 64],
                                    "dtype": "torch.float32", "logical_bytes": 16384},
                "activation": activation_identity(fr.get_format(fmt), {}, name),
                "arithmetic": arithmetic}
    row = make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
                                signed_components=components)
    assert row["x2_per_probe"] == [squared_signed(t) for t in RECORDED]
    assert row["x2_per_probe"] == [t * t for t in RECORDED]


def test_streamed_accumulator_uses_the_shared_squaring():
    source = (Path(__file__).resolve().parents[1] / "prismaquant" / "aura_cost.py").read_text()
    tree = ast.parse(source)
    pow_squares = [node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
                   and isinstance(node.right, ast.Constant) and node.right.value == 2
                   and isinstance(node.left, ast.Subscript)
                   and isinstance(node.left.slice, ast.Constant) and node.left.slice.value == "total"]
    assert pow_squares == [], f"joint total squared with ** at lines {pow_squares}"
    assert "squared_signed(terms[\"total\"])" in source
