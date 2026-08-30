from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers import AutoConfig

import prismaquant.cost_streaming as cost_streaming
from prismaquant.model_profiles import detect_profile
from prismaquant.prismasnap import PRISMASNAP_ALGORITHM, PrismaSnapSearchConfig
import prismaquant.prismasnap_checkpoint as checkpoint
import prismaquant.prismasnap as prismasnap


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_worker_plan(
    root: Path,
    *,
    layers: list[int],
    layer_count: int,
    vector_prefix: str,
) -> dict[str, object]:
    root.mkdir()
    scales = {
        f"layer_{layer:05d}_{suffix}": torch.full(
            (16,), 1.0 + layer / 16.0, dtype=torch.float64
        )
        for layer in layers
        for suffix in ("input", "post", "updown")
    }
    save_file(scales, str(root / checkpoint.PLAN_SCALES), metadata={"format": "pt"})
    search = PrismaSnapSearchConfig(
        alphas=(0.0,),
        max_rounds=1,
        stage=False,
        polish=False,
        polish_top=0,
        polish_pool=0,
    ).as_dict()

    def tensor(layer: int, leaf: str) -> str:
        return f"model.layers.{layer}.{leaf}"

    def stats() -> dict[str, object]:
        return {
            "algorithm": PRISMASNAP_ALGORITHM,
            "error_baseline": 0.0,
            "error_final": 0.0,
            "improvement_fraction": 0.0,
            "groups": 1,
            "groups_moved": 0,
            "rounds": 1,
            "candidate_count": 1,
            "fell_back": False,
            "polish_pool": 0,
            "polished": 0,
            "variant": [],
        }

    seams: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    weight_map: dict[str, str] = {}
    tensor_rows: dict[str, dict[str, object]] = {}
    owner = "model.safetensors"
    for layer in range(layer_count):
        shapes = {
            "input_layernorm.weight": [16],
            "post_attention_layernorm.weight": [16],
            "self_attn.q_proj.weight": [16, 16],
            "self_attn.k_proj.weight": [16, 16],
            "self_attn.v_proj.weight": [16, 16],
            "mlp.gate_proj.weight": [16, 16],
            "mlp.up_proj.weight": [16, 16],
            "mlp.down_proj.weight": [16, 16],
        }
        for leaf, shape in shapes.items():
            name = tensor(layer, leaf)
            weight_map[name] = owner
            tensor_rows[name] = {"owner": owner, "shape": shape, "dtype": "BF16"}
    for layer in layers:
        input_vector = f"layer_{layer:05d}_input"
        post_vector = f"layer_{layer:05d}_post"
        updown_vector = f"layer_{layer:05d}_updown"
        input_norm = tensor(layer, "input_layernorm.weight")
        post_norm = tensor(layer, "post_attention_layernorm.weight")
        input_consumers = [
            tensor(layer, f"self_attn.{leaf}_proj.weight")
            for leaf in ("q", "k", "v")
        ]
        gate = tensor(layer, "mlp.gate_proj.weight")
        up = tensor(layer, "mlp.up_proj.weight")
        down = tensor(layer, "mlp.down_proj.weight")
        post_consumers = [gate, up]
        graph_sha = checkpoint._dense_plan_graph_sha256(
            layer=layer,
            input_norm=input_norm,
            input_offset=0.0,
            input_consumers=input_consumers,
            post_norm=post_norm,
            post_offset=0.0,
            post_consumers=post_consumers,
            gate=gate,
            up=up,
            down=down,
        )
        seams.extend(
            [
                {
                    "layer": layer,
                    "kind": "input_norm",
                    "vector": input_vector,
                    "norm": input_norm,
                    "norm_parameter_offset": 0.0,
                    "consumers": input_consumers,
                    "stats": stats(),
                    "graph_sha256": graph_sha,
                },
                {
                    "layer": layer,
                    "kind": "post_attention_norm",
                    "vector": post_vector,
                    "norm": post_norm,
                    "norm_parameter_offset": 0.0,
                    "consumers": post_consumers,
                    "stats": stats(),
                    "graph_sha256": graph_sha,
                },
                {
                    "layer": layer,
                    "kind": "up_down",
                    "vector": updown_vector,
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "stats": stats(),
                    "graph_sha256": graph_sha,
                },
            ]
        )
        for norm, vector, consumers in (
            (input_norm, input_vector, input_consumers),
            (post_norm, post_vector, post_consumers),
        ):
            transforms.append(
                {
                    "tensor": norm,
                    "vector": vector,
                    "operation": "affine_multiply",
                    "axis": 0,
                    "order": 0,
                    "parameter_offset": 0.0,
                }
            )
            transforms.extend(
                {
                    "tensor": name,
                    "vector": vector,
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                }
                for name in consumers
            )
        transforms.extend(
            [
                {
                    "tensor": up,
                    "vector": updown_vector,
                    "operation": "multiply",
                    "axis": 0,
                    "order": 1,
                },
                {
                    "tensor": down,
                    "vector": updown_vector,
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                },
            ]
        )
    tensor_metadata_unsigned = {
        "schema": checkpoint.TENSOR_METADATA_SCHEMA,
        "tensors": dict(sorted(tensor_rows.items())),
    }
    tensor_metadata = {
        **tensor_metadata_unsigned,
        "sha256": checkpoint.canonical_json_sha256(
            tensor_metadata_unsigned, where="synthetic tensor metadata"
        ),
    }
    plan: dict[str, object] = {
        "schema": checkpoint.PLAN_SCHEMA,
        "state": "PLANNED",
        "algorithm": PRISMASNAP_ALGORITHM,
        "producer": checkpoint._producer_identity(),
        "profile": "synthetic_dense",
        "source": {
            "identity": {
                "checkpoint_weight_map": dict(sorted(weight_map.items())),
            },
            "portable_identity": {
                "portable_content_sha256": "2" * 64,
            }
        },
        "probe": {"sha256": "1" * 64, "calib_hash": "same-calibration"},
        "model": {
            "hidden_size": 16,
            "layer_count": layer_count,
            "planned_layers": layers,
            "excluded_prefixes": ["model.visual.", "mtp."],
        },
        "search": search,
        "tensor_metadata": tensor_metadata,
        "tensor_metadata_binding": {
            "mode": "inline_full_header_scan",
            "manifest_sha256": None,
            "tensor_metadata_sha256": tensor_metadata["sha256"],
        },
        "scales": {
            "file": checkpoint.PLAN_SCALES,
            "sha256": checkpoint._sha256_file(root / checkpoint.PLAN_SCALES),
            "vectors": len(scales),
        },
        "seams": seams,
        "transforms": sorted(
            transforms, key=lambda row: (str(row["tensor"]), int(row["order"]))
        ),
        "verification": {
            "fp64_invariance_max_abs": 0.0,
            "threshold": 1e-10,
            "domain": "pre_cast_fp64_algebra",
            "required_bf16_fold_kl_max": 5e-4,
        },
    }
    plan["plan_sha256"] = checkpoint._plan_digest(plan)
    _json(root / checkpoint.PLAN_JSON, plan)
    return plan


def test_load_plan_rejects_manifest_and_scale_content_tampering(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"
    _write_worker_plan(
        manifest_dir, layers=[0], layer_count=1, vector_prefix="manifest"
    )
    manifest_path = manifest_dir / checkpoint.PLAN_JSON
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["search"]["alphas"] = [0.0, 0.5]
    _json(manifest_path, payload)
    with pytest.raises(RuntimeError, match="plan digest mismatch"):
        checkpoint.load_plan(manifest_dir)

    scale_dir = tmp_path / "scale"
    _write_worker_plan(scale_dir, layers=[0], layer_count=1, vector_prefix="scale")
    scale_path = scale_dir / checkpoint.PLAN_SCALES
    scale_path.write_bytes(scale_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="scale content digest mismatch"):
        checkpoint.load_plan(scale_dir)


def test_load_plan_rejects_rehashed_semantic_omission_and_cross_binding(
    tmp_path: Path,
) -> None:
    def rewrite(root: Path, mutate) -> None:
        path = root / checkpoint.PLAN_JSON
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        payload["plan_sha256"] = checkpoint._plan_digest(payload)
        _json(path, payload)

    extra = tmp_path / "extra-top"
    _write_worker_plan(extra, layers=[0], layer_count=1, vector_prefix="unused")
    rewrite(extra, lambda payload: payload.__setitem__("unexpected", True))
    with pytest.raises(RuntimeError, match="fields differ"):
        checkpoint.load_plan(extra)

    missing = tmp_path / "missing-seam"
    _write_worker_plan(missing, layers=[0], layer_count=1, vector_prefix="unused")
    rewrite(missing, lambda payload: payload["seams"].pop())
    with pytest.raises(RuntimeError, match="seam|scale-vector"):
        checkpoint.load_plan(missing)

    omitted_transform = tmp_path / "omitted-transform"
    _write_worker_plan(
        omitted_transform, layers=[0], layer_count=1, vector_prefix="unused"
    )
    rewrite(omitted_transform, lambda payload: payload["transforms"].pop())
    with pytest.raises(RuntimeError, match="not seam-derived"):
        checkpoint.load_plan(omitted_transform)

    graph = tmp_path / "cross-bound-graph"
    _write_worker_plan(graph, layers=[0], layer_count=1, vector_prefix="unused")

    def replace_graph(payload: dict[str, object]) -> None:
        for seam in payload["seams"]:
            seam["graph_sha256"] = "0" * 64

    rewrite(graph, replace_graph)
    with pytest.raises(RuntimeError, match="graph digest is not role-bound"):
        checkpoint.load_plan(graph)

    cross_bound = tmp_path / "cross-bound"
    _write_worker_plan(cross_bound, layers=[0], layer_count=2, vector_prefix="unused")

    def cross_bind(payload: dict[str, object]) -> None:
        seams = payload["seams"]
        row = next(item for item in seams if item["kind"] == "input_norm")
        old = row["consumers"][0]
        new = "model.layers.1.self_attn.q_proj.weight"
        row["consumers"][0] = new
        transform = next(
            item for item in payload["transforms"] if item["tensor"] == old
        )
        transform["tensor"] = new
        payload["transforms"].sort(
            key=lambda item: (str(item["tensor"]), int(item["order"]))
        )

    rewrite(cross_bound, cross_bind)
    with pytest.raises(RuntimeError, match="not bound to body layer 0"):
        checkpoint.load_plan(cross_bound)


def test_merge_plans_requires_identical_full_tensor_metadata(tmp_path: Path) -> None:
    left = tmp_path / "header-left"
    right = tmp_path / "header-right"
    _write_worker_plan(left, layers=[0], layer_count=2, vector_prefix="unused")
    _write_worker_plan(right, layers=[1], layer_count=2, vector_prefix="unused")
    path = right / checkpoint.PLAN_JSON
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["tensor_metadata"]
    metadata["tensors"]["model.layers.0.self_attn.q_proj.weight"]["shape"] = [17, 16]
    metadata["sha256"] = checkpoint.canonical_json_sha256(
        {"schema": metadata["schema"], "tensors": metadata["tensors"]},
        where="tampered worker tensor metadata",
    )
    payload["tensor_metadata_binding"]["tensor_metadata_sha256"] = metadata[
        "sha256"
    ]
    payload["plan_sha256"] = checkpoint._plan_digest(payload)
    _json(path, payload)
    checkpoint.load_plan(right)
    with pytest.raises(RuntimeError, match="disagree on tensor_metadata"):
        checkpoint.merge_plans([left, right], tmp_path / "header-merged")


def test_merge_plans_requires_an_exact_nonoverlapping_layer_union(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_worker_plan(left, layers=[0], layer_count=2, vector_prefix="left")
    _write_worker_plan(right, layers=[1], layer_count=2, vector_prefix="right")

    merged = checkpoint.merge_plans([left, right], tmp_path / "merged")
    assert merged["schema"] == checkpoint.PLAN_SET_SCHEMA
    assert merged["model"]["planned_layers"] == [0, 1]
    loaded, scales = checkpoint.load_plan(tmp_path / "merged")
    assert loaded == merged
    assert set(scales) == {
        "layer_00000_input",
        "layer_00000_post",
        "layer_00000_updown",
        "layer_00001_input",
        "layer_00001_post",
        "layer_00001_updown",
    }

    overlap_a = tmp_path / "overlap-a"
    overlap_b = tmp_path / "overlap-b"
    _write_worker_plan(overlap_a, layers=[0], layer_count=2, vector_prefix="oa")
    _write_worker_plan(overlap_b, layers=[0], layer_count=2, vector_prefix="ob")
    with pytest.raises(RuntimeError, match="overlap on layer 0"):
        checkpoint.merge_plans(
            [overlap_a, overlap_b], tmp_path / "overlap-output"
        )

    partial_a = tmp_path / "partial-a"
    partial_b = tmp_path / "partial-b"
    _write_worker_plan(partial_a, layers=[0], layer_count=3, vector_prefix="pa")
    _write_worker_plan(partial_b, layers=[2], layer_count=3, vector_prefix="pb")
    with pytest.raises(RuntimeError, match=r"coverage is not exact; missing=\[1\]"):
        checkpoint.merge_plans(
            [partial_a, partial_b], tmp_path / "partial-output"
        )


def test_merge_plans_resume_recovers_commit_and_rejects_order_equivocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "resume-left"
    right = tmp_path / "resume-right"
    _write_worker_plan(left, layers=[0], layer_count=2, vector_prefix="left")
    _write_worker_plan(right, layers=[1], layer_count=2, vector_prefix="right")
    output = tmp_path / "resume-merged-plan"
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    real_replace = checkpoint.os.replace
    interrupted = {"done": False}

    def interrupt_commit(source: str | Path, destination: str | Path) -> None:
        if (
            Path(source) == staging
            and Path(destination) == output
            and not interrupted["done"]
        ):
            interrupted["done"] = True
            raise RuntimeError("synthetic plan commit interruption")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", interrupt_commit)
    with pytest.raises(RuntimeError, match="synthetic plan commit interruption"):
        # Resume with no staging state is a valid first attempt.
        checkpoint.merge_plans([left, right], output, resume=True)
    assert staging.is_dir()
    assert not (staging / checkpoint.PLAN_MERGE_STATE_JSON).exists()

    monkeypatch.setattr(checkpoint.os, "replace", real_replace)
    merged = checkpoint.merge_plans([left, right], output, resume=True)
    assert merged["workers"] == [
        {
            "plan_sha256": checkpoint.load_plan(left)[0]["plan_sha256"],
            "layers": [0],
        },
        {
            "plan_sha256": checkpoint.load_plan(right)[0]["plan_sha256"],
            "layers": [1],
        },
    ]
    assert checkpoint.merge_plans([left, right], output, resume=True) == merged
    with pytest.raises(RuntimeError, match="ordered worker inputs"):
        checkpoint.merge_plans([right, left], output, resume=True)


def _tiny_planned_checkpoint(
    tmp_path: Path, *, layer_count: int = 1
) -> dict[str, object]:
    source = tmp_path / "source"
    source.mkdir()
    config: dict[str, object] = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": layer_count,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
    }
    _json(source / "config.json", config)
    (source / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    profile = detect_profile(str(source))
    assert profile.name == "qwen3_5_dense"

    recipe_shapes = {
        "model.embed_tokens.weight": (32, 16),
        "model.norm.weight": (16,),
        "lm_head.weight": (32, 16),
    }
    for layer in range(layer_count):
        recipe_shapes.update(
            {
                f"model.layers.{layer}.input_layernorm.weight": (16,),
                f"model.layers.{layer}.post_attention_layernorm.weight": (16,),
                f"model.layers.{layer}.self_attn.q_proj.weight": (16, 16),
                f"model.layers.{layer}.self_attn.k_proj.weight": (16, 16),
                f"model.layers.{layer}.self_attn.v_proj.weight": (16, 16),
                f"model.layers.{layer}.mlp.gate_proj.weight": (32, 16),
                f"model.layers.{layer}.mlp.up_proj.weight": (32, 16),
                f"model.layers.{layer}.mlp.down_proj.weight": (16, 32),
            }
        )
    generator = torch.Generator().manual_seed(314159)
    source_tensors: dict[str, torch.Tensor] = {}
    for recipe_name, shape in recipe_shapes.items():
        source_name = profile.source_tensor_name(recipe_name)
        if recipe_name.endswith("layernorm.weight") or recipe_name == "model.norm.weight":
            # Qwen3.5/Qwen3.8 stores gamma-1; zero is the identity norm
            # parameter, not one.
            value = torch.zeros(shape, dtype=torch.bfloat16)
        else:
            value = (torch.randn(shape, generator=generator) * 0.125).to(
                torch.bfloat16
            )
        source_tensors[source_name] = value

    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    first_keys = {
        key
        for key in source_tensors
        if "self_attn" in key or "input_layernorm" in key
    }
    first = {key: value for key, value in source_tensors.items() if key in first_keys}
    second = {key: value for key, value in source_tensors.items() if key not in first_keys}
    save_file(first, str(source / first_name), metadata={"format": "pt"})
    save_file(second, str(source / second_name), metadata={"format": "pt"})
    weight_map = {
        key: first_name if key in first_keys else second_name
        for key in sorted(source_tensors)
    }
    _json(
        source / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": sum(
                    value.numel() * value.element_size()
                    for value in source_tensors.values()
                )
            },
            "weight_map": weight_map,
        },
    )

    identity_config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    identity_config._commit_hash = "synthetic-revision"
    runner = SimpleNamespace(
        model=SimpleNamespace(config=identity_config),
        context=SimpleNamespace(
            weight_shard={
                key: source / shard for key, shard in weight_map.items()
            },
            weight_ckpt={key: key for key in weight_map},
        ),
    )
    identity = cost_streaming.build_streamed_model_identity(
        runner, str(source.resolve())
    )
    # Exercise the same validator the production planner consumes before the
    # identity is admitted into the synthetic plan.
    assert cost_streaming.validate_streamed_model_identity(
        identity, where="test source identity"
    ) == identity
    identity_path = tmp_path / "streamed_model_identity.json"
    _json(identity_path, identity)

    input_importance = np.linspace(0.5, 1.5, 16, dtype=np.float32)
    post_importance = np.linspace(1.5, 0.5, 16, dtype=np.float32)
    stats: dict[str, dict[str, object]] = {}
    for layer in range(layer_count):
        for leaf in ("q_proj", "k_proj", "v_proj"):
            stats[f"model.layers.{layer}.self_attn.{leaf}"] = {
                "act_sq_sum": input_importance.copy(),
                "in_features": 16,
                "out_features": 16,
            }
        for leaf in ("gate_proj", "up_proj"):
            stats[f"model.layers.{layer}.mlp.{leaf}"] = {
                "act_sq_sum": post_importance.copy(),
                "in_features": 16,
                "out_features": 32,
            }
        stats[f"model.layers.{layer}.mlp.down_proj"] = {
            "act_sq_sum": np.linspace(0.25, 2.0, 32, dtype=np.float32),
            "in_features": 32,
            "out_features": 16,
        }
    probe = {
        "stats": stats,
        "meta": {
            "model": str(source.resolve()),
            "calib_hash": "synthetic-calibration",
            "dataset": "synthetic",
            "nsamples": 2,
            "seqlen": 512,
            "dtype": "bf16",
            "device_map": "streaming-layerwise",
            "execution_device": "cuda:0",
            "calibration_modality": "text-only",
        },
    }
    probe_path = tmp_path / "probe.pkl"
    with probe_path.open("wb") as handle:
        pickle.dump(probe, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plan_dir = tmp_path / "plan"
    plan = checkpoint.plan_dense_checkpoint(
        source,
        probe_path,
        identity_path,
        plan_dir,
        device="cpu",
        search_config=PrismaSnapSearchConfig(
            alphas=(0.0,),
            max_rounds=1,
            stage=False,
            polish=False,
            polish_top=0,
            polish_pool=0,
            scale_rule="static_6",
        ),
    )
    assert plan["probe"]["sha256"] == checkpoint._sha256_file(probe_path)
    return {
        "source": source,
        "identity_path": identity_path,
        "probe_path": probe_path,
        "plan_dir": plan_dir,
        "plan": plan,
        "weight_map": weight_map,
        "source_tensors": source_tensors,
    }


def test_dense_plan_binds_full_headers_and_measured_consumer_order(
    tmp_path: Path,
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    plan = fixture["plan"]
    tensors = plan["tensor_metadata"]["tensors"]
    assert set(tensors) == set(fixture["weight_map"])
    for name, value in fixture["source_tensors"].items():
        assert tensors[name] == {
            "owner": fixture["weight_map"][name],
            "shape": list(value.shape),
            "dtype": "BF16",
        }
    input_seam = next(
        row for row in plan["seams"] if row["kind"] == "input_norm"
    )
    assert [name.rsplit(".", 2)[-2] for name in input_seam["consumers"]] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]


def test_bf16_projection_offset_one_handles_zero_and_collapsed_gamma() -> None:
    source = torch.tensor(
        [-1.0, -0.99609375, 0.0], dtype=torch.bfloat16
    )
    nominal = torch.tensor([4.0, 0.01, 1.25], dtype=torch.float64)
    projected, executed, counts = (
        checkpoint._project_bf16_norm_and_realize_inverse(
            source, nominal, parameter_offset=1.0
        )
    )
    assert torch.equal(
        projected,
        torch.tensor([-1.0, -0.99609375, 0.25], dtype=torch.bfloat16),
    )
    assert torch.equal(
        executed, torch.tensor([1.0, 1.0, 1.25], dtype=torch.float64)
    )
    assert counts == {
        "channels": 3,
        "candidate_realized_channels": 1,
        "source_zero_channels": 1,
        "projected_zero_channels": 1,
        "sign_mismatch_channels": 0,
        "nonfinite_channels": 0,
        "reapplication_mismatch_channels": 0,
    }
    replay = ((source.to(torch.float64) + 1.0) * executed - 1.0).to(
        torch.bfloat16
    )
    assert torch.equal(replay, projected)


def test_bf16_realized_plan_binds_parent_executes_identity_updown_and_materializes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISMAQUANT_CONTAINER_ROOTFS_SHA256", "a" * 64)
    fixture = _tiny_planned_checkpoint(tmp_path, layer_count=2)
    search = PrismaSnapSearchConfig()
    left = checkpoint.plan_dense_checkpoint(
        fixture["source"],
        fixture["probe_path"],
        fixture["identity_path"],
        tmp_path / "v2-left",
        device="cpu",
        layers=[0],
        search_config=search,
    )
    right = checkpoint.plan_dense_checkpoint(
        fixture["source"],
        fixture["probe_path"],
        fixture["identity_path"],
        tmp_path / "v2-right",
        device="cpu",
        layers=[1],
        search_config=search,
    )
    merged_dir = tmp_path / "v2-parent"
    parent = checkpoint.merge_plans(
        [tmp_path / "v2-left", tmp_path / "v2-right"], merged_dir
    )
    assert left["model"]["planned_layers"] == [0]
    assert right["model"]["planned_layers"] == [1]

    realized_dir = tmp_path / "v2-realized"
    realized = checkpoint.realize_bf16_plan(
        fixture["source"], merged_dir, realized_dir, device="cpu", resume=True
    )
    assert realized["schema"] == checkpoint.BF16_REALIZED_PLAN_SCHEMA
    assert realized["algorithm"] == checkpoint.PRISMASNAP_BF16_REALIZED_ALGORITHM
    assert realized["derivation"]["parent"]["plan_sha256"] == parent[
        "plan_sha256"
    ]
    assert all(
        row["executed_realized_channels"] == 0
        for row in realized["derivation"]["seams"]
        if row["kind"] == "up_down"
    )
    loaded, scales = checkpoint.load_plan(realized_dir)
    assert loaded == realized
    accepted = [
        row
        for row in realized["derivation"]["seams"]
        if row["kind"] != "up_down" and row["objective_accepted"]
    ]
    assert accepted, "synthetic default search must exercise one accepted v2 seam"
    transforms_by_vector: dict[str, list[dict[str, object]]] = {}
    for transform in realized["transforms"]:
        transforms_by_vector.setdefault(transform["vector"], []).append(transform)
    for row in realized["derivation"]["seams"]:
        if row["kind"] == "up_down":
            assert row["executed_groups_moved"] == 0
            assert torch.equal(
                scales[row["executed_vector"]],
                torch.ones_like(scales[row["executed_vector"]]),
            )
            assert row["executed_vector"] not in transforms_by_vector
        elif row["objective_accepted"]:
            seam = next(
                seam
                for seam in realized["seams"]
                if (seam["layer"], seam["kind"]) == (row["layer"], row["kind"])
            )
            assert row["executed_groups_moved"] > 0
            assert transforms_by_vector[row["projected_norm_vector"]] == [
                {
                    "tensor": seam["norm"],
                    "vector": row["projected_norm_vector"],
                    "operation": "replace_bf16",
                    "axis": 0,
                    "order": 0,
                }
            ]
            assert {
                transform["tensor"]
                for transform in transforms_by_vector[row["executed_vector"]]
            } == set(seam["consumers"])
        else:
            assert row["executed_groups_moved"] == 0
            assert row["executed_vector"] not in transforms_by_vector
            assert row["projected_norm_vector"] not in transforms_by_vector
    assert checkpoint.realize_bf16_plan(
        fixture["source"], merged_dir, realized_dir, device="cpu", resume=True
    ) == realized

    output = tmp_path / "v2-materialized"
    provenance = checkpoint.materialize_checkpoint(
        fixture["source"], realized_dir, output, device="cpu"
    )
    assert provenance["schema"] == checkpoint.PROVENANCE_SCHEMA_V2
    assert provenance["algorithm"] == checkpoint.PRISMASNAP_BF16_REALIZED_ALGORITHM
    assert provenance["derivation"] == realized["derivation"]
    execution_by_key = {
        (row["layer"], row["kind"]): row
        for row in realized["derivation"]["seams"]
    }
    for summary in provenance["seam_summary"]:
        execution = execution_by_key[(summary["layer"], summary["kind"])]
        assert summary["groups_moved"] == execution["executed_groups_moved"]
        assert summary["improvement_fraction"] == (
            0.0
            if summary["kind"] == "up_down"
            else execution["objective_improvement_fraction"]
        )
    changed_consumer = False
    seams_by_key = {
        (row["layer"], row["kind"]): row for row in realized["seams"]
    }
    for row in accepted:
        seam = seams_by_key[(row["layer"], row["kind"])]
        norm_name = seam["norm"]
        norm_owner = fixture["weight_map"][norm_name]
        with safe_open(str(output / norm_owner), framework="pt") as handle:
            assert torch.equal(
                handle.get_tensor(norm_name), scales[row["projected_norm_vector"]]
            )
        inverse = scales[row["executed_vector"]]
        for consumer_name in seam["consumers"]:
            expected = (
                fixture["source_tensors"][consumer_name].to(torch.float64)
                / inverse.view(1, -1)
            ).to(torch.bfloat16)
            owner = fixture["weight_map"][consumer_name]
            with safe_open(str(output / owner), framework="pt") as handle:
                observed = handle.get_tensor(consumer_name)
            assert torch.equal(observed, expected)
            changed_consumer |= not torch.equal(
                observed, fixture["source_tensors"][consumer_name]
            )
    assert changed_consumer
    from prismaquant.prismasnap_validation import (
        validate_prismasnap_provenance_payload,
    )

    validate_prismasnap_provenance_payload(
        provenance, require_verified=False, where="test BF16-realized provenance"
    )
    misleading = json.loads(json.dumps(provenance))
    updown_summary = next(
        row for row in misleading["seam_summary"] if row["kind"] == "up_down"
    )
    updown_summary["groups_moved"] = 1
    misleading["provenance_sha256"] = checkpoint.canonical_json_sha256(
        {
            key: value
            for key, value in misleading.items()
            if key != "provenance_sha256"
        },
        where="test misleading BF16-realized provenance",
    )
    with pytest.raises(RuntimeError, match="summary differs from execution"):
        validate_prismasnap_provenance_payload(
            misleading,
            require_verified=False,
            where="test misleading BF16-realized provenance",
        )

    shard_names = sorted(set(fixture["weight_map"].values()))
    part_a = tmp_path / "v2-part-a"
    part_b = tmp_path / "v2-part-b"
    checkpoint.materialize_checkpoint_part(
        fixture["source"],
        realized_dir,
        part_a,
        [shard_names[0]],
        device="cpu",
    )
    checkpoint.materialize_checkpoint_part(
        fixture["source"],
        realized_dir,
        part_b,
        [shard_names[1]],
        device="cpu",
    )
    collated = checkpoint.merge_checkpoint_parts(
        fixture["source"],
        realized_dir,
        [part_a, part_b],
        tmp_path / "v2-collated",
    )
    assert collated["schema"] == checkpoint.PROVENANCE_SCHEMA_V2
    assert collated["derivation"] == realized["derivation"]
    validate_prismasnap_provenance_payload(
        collated, require_verified=False, where="test BF16-realized collation"
    )

    monkeypatch.setattr(
        checkpoint,
        "_exact_executed_norm_objective",
        lambda *_args, **_kwargs: (1.0, 1.0),
    )
    identity_plan_dir = tmp_path / "v2-realized-all-identity"
    identity_plan = checkpoint.realize_bf16_plan(
        fixture["source"],
        merged_dir,
        identity_plan_dir,
        device="cpu",
        resume=True,
    )
    assert identity_plan["transforms"] == []
    assert all(
        row["executed_groups_moved"] == 0
        for row in identity_plan["derivation"]["seams"]
    )
    identity_output = tmp_path / "v2-materialized-all-identity"
    identity_provenance = checkpoint.materialize_checkpoint(
        fixture["source"], identity_plan_dir, identity_output, device="cpu"
    )
    assert identity_provenance["coverage"]["transformed_tensors"] == 0
    assert identity_provenance["coverage"]["materialized_changed_tensors"] == 0
    validate_prismasnap_provenance_payload(
        identity_provenance,
        require_verified=False,
        where="test all-identity BF16-realized provenance",
    )
    for tensor_name, expected in fixture["source_tensors"].items():
        with safe_open(
            str(identity_output / fixture["weight_map"][tensor_name]),
            framework="pt",
        ) as handle:
            assert torch.equal(handle.get_tensor(tensor_name), expected)

    tampered = tmp_path / "v2-tampered-parent"
    shutil.copytree(realized_dir, tampered)
    manifest_path = tampered / checkpoint.PLAN_JSON
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derivation"]["parent"]["plan_sha256"] = "0" * 64
    manifest["derivation"]["derivation_sha256"] = checkpoint._derivation_digest(
        manifest["derivation"]
    )
    manifest["plan_sha256"] = checkpoint._plan_digest(manifest)
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="reconstruct the bound parent"):
        checkpoint.load_plan(tampered)


def test_external_tensor_metadata_manifest_bypasses_worker_full_header_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    manifest_path = tmp_path / "tensor-metadata-manifest.json"
    manifest = checkpoint.scan_tensor_metadata_manifest(
        fixture["source"],
        fixture["identity_path"],
        manifest_path,
        resume=True,
    )
    assert checkpoint.scan_tensor_metadata_manifest(
        fixture["source"],
        fixture["identity_path"],
        manifest_path,
        resume=True,
    ) == manifest

    def refuse_inline_scan(_source) -> dict[str, object]:
        raise AssertionError("external manifest path performed a full header scan")

    monkeypatch.setattr(
        checkpoint, "_scan_checkpoint_tensor_metadata", refuse_inline_scan
    )
    plan = checkpoint.plan_dense_checkpoint(
        fixture["source"],
        fixture["probe_path"],
        fixture["identity_path"],
        tmp_path / "external-header-plan",
        device="cpu",
        tensor_metadata_manifest_path=manifest_path,
        search_config=PrismaSnapSearchConfig(
            alphas=(0.0,),
            max_rounds=1,
            stage=False,
            polish=False,
            polish_top=0,
            polish_pool=0,
        ),
    )
    assert plan["tensor_metadata"] == manifest["tensor_metadata"]
    assert plan["tensor_metadata_binding"] == {
        "mode": "external_manifest",
        "manifest_sha256": manifest["manifest_sha256"],
        "tensor_metadata_sha256": manifest["tensor_metadata"]["sha256"],
    }


def test_planner_rejects_probe_source_and_execution_contract_tampering(
    tmp_path: Path,
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    original = pickle.loads(Path(fixture["probe_path"]).read_bytes())
    wrong_model = tmp_path / "wrong-model"
    wrong_model.mkdir()
    mutations = (
        lambda meta: meta.__setitem__("model", str(wrong_model)),
        lambda meta: meta.__setitem__("dtype", "fp16"),
        lambda meta: meta.__setitem__("execution_device", "cpu"),
        lambda meta: meta.__setitem__("device_map", "cpu"),
        lambda meta: meta.__setitem__("nsamples", 0),
        lambda meta: meta.__setitem__("calibration_modality", "text"),
    )
    for ordinal, mutate in enumerate(mutations):
        payload = pickle.loads(pickle.dumps(original))
        mutate(payload["meta"])
        probe = tmp_path / f"tampered-probe-{ordinal}.pkl"
        with probe.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with pytest.raises(RuntimeError, match="not bound to the BF16"):
            checkpoint.plan_dense_checkpoint(
                fixture["source"],
                probe,
                fixture["identity_path"],
                tmp_path / f"tampered-plan-{ordinal}",
                device="cpu",
            )


def test_bind_legacy_text_probe_normalizes_only_modality_and_binds_plan(
    tmp_path: Path,
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    original_payload = pickle.loads(Path(fixture["probe_path"]).read_bytes())
    original_payload["meta"]["nsamples"] = 2
    original_payload["meta"]["seqlen"] = 512
    original_payload["meta"].pop("calibration_modality")
    legacy = tmp_path / "frozen-legacy-probe.pkl"
    with legacy.open("wb") as handle:
        pickle.dump(original_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    normalized = tmp_path / "bound-probe.pkl"
    receipt = checkpoint.bind_legacy_text_probe(
        fixture["source"],
        fixture["identity_path"],
        legacy,
        normalized,
        resume=True,
    )
    assert receipt["delta"] == {
        "field": "meta.calibration_modality",
        "before": "missing",
        "after": "text-only",
        "only_mutation": True,
    }
    normalized_payload = pickle.loads(normalized.read_bytes())
    assert normalized_payload["meta"]["calibration_modality"] == "text-only"
    normalized_payload["meta"].pop("calibration_modality")
    assert pickle.dumps(
        normalized_payload, protocol=pickle.HIGHEST_PROTOCOL
    ) == pickle.dumps(original_payload, protocol=pickle.HIGHEST_PROTOCOL)
    assert checkpoint.bind_legacy_text_probe(
        fixture["source"],
        fixture["identity_path"],
        legacy,
        normalized,
        resume=True,
    ) == receipt

    bound_plan = checkpoint.plan_dense_checkpoint(
        fixture["source"],
        normalized,
        fixture["identity_path"],
        tmp_path / "bound-plan",
        device="cpu",
        search_config=PrismaSnapSearchConfig(
            alphas=(0.0,),
            max_rounds=1,
            stage=False,
            polish=False,
            polish_top=0,
            polish_pool=0,
        ),
    )
    assert bound_plan["probe"]["calibration_modality"] == "text-only"
    assert bound_plan["probe"]["legacy_text_binding"]["binding_sha256"] == receipt[
        "binding_sha256"
    ]


def test_bind_legacy_text_probe_rejects_visual_stats_and_receipt_tampering(
    tmp_path: Path,
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    payload = pickle.loads(Path(fixture["probe_path"]).read_bytes())
    payload["meta"]["nsamples"] = 2
    payload["meta"]["seqlen"] = 512
    payload["meta"].pop("calibration_modality")
    first_row = next(iter(payload["stats"].values()))
    payload["stats"]["model.visual.blocks.0.proj"] = first_row
    visual = tmp_path / "visual-legacy-probe.pkl"
    with visual.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(RuntimeError, match="contains visual stats"):
        checkpoint.bind_legacy_text_probe(
            fixture["source"],
            fixture["identity_path"],
            visual,
            tmp_path / "visual-bound.pkl",
        )

    payload["stats"].pop("model.visual.blocks.0.proj")
    clean = tmp_path / "clean-legacy-probe.pkl"
    with clean.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    normalized = tmp_path / "tamper-bound.pkl"
    checkpoint.bind_legacy_text_probe(
        fixture["source"], fixture["identity_path"], clean, normalized
    )
    receipt_path = checkpoint._probe_binding_path(normalized)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["delta"]["after"] = "text"
    receipt["binding_sha256"] = checkpoint.canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "binding_sha256"},
        where="tampered legacy binding",
    )
    _json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="binding contract failed"):
        checkpoint.plan_dense_checkpoint(
            fixture["source"],
            normalized,
            fixture["identity_path"],
            tmp_path / "tampered-binding-plan",
            device="cpu",
        )


def test_production_entrypoints_require_cuda_and_attested_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        checkpoint.plan_dense_checkpoint(
            tmp_path / "absent-source",
            tmp_path / "absent-probe",
            tmp_path / "absent-identity",
            tmp_path / "absent-output",
            device="cpu",
            production=True,
        )
    monkeypatch.delenv("PRISMAQUANT_CONTAINER_ROOTFS_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="attested container"):
        checkpoint.plan_dense_checkpoint(
            tmp_path / "absent-source",
            tmp_path / "absent-probe",
            tmp_path / "absent-identity",
            tmp_path / "absent-output",
            device="cuda",
            production=True,
        )


def test_merge_checkpoint_parts_is_subject_to_the_production_execution_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2: the collating writer escaped the production contract.

    The merge executes on no device, so the half of the contract that binds it
    is container attestation, not the CUDA declaration. The gate must precede
    every other check, including the arity ValueError, so an unattested merge
    cannot even inspect the parts it was handed -- a single part dir is passed
    below precisely so a RuntimeError (not ValueError) proves the ordering.
    """

    monkeypatch.delenv("PRISMAQUANT_CONTAINER_ROOTFS_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="attested container"):
        checkpoint.merge_checkpoint_parts(
            tmp_path / "absent-source",
            tmp_path / "absent-plan",
            [tmp_path / "absent-part"],
            tmp_path / "absent-output",
            production=True,
        )


def test_merge_checkpoint_parts_takes_no_device() -> None:
    """The merge must not carry a device it never executes on.

    A `device` here could only be checked as a string, since nothing in the
    union ever reaches a device -- the assert-instead-of-attest shape principle
    14 refuses. Its absence from the signature is the fix, so it is asserted.
    """

    params = inspect.signature(checkpoint.merge_checkpoint_parts).parameters
    assert "device" not in params
    assert "production" in params


@pytest.mark.parametrize(
    "perturbed",
    [
        "prismaquant/prismasnap_contract.py",
        "prismaquant/prismasnap_moe_checkpoint.py",
        "prismaquant/prismasnap_moe.py",
    ],
)
def test_producer_identity_binds_the_moe_plan_stack_and_the_lane_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perturbed: str
) -> None:
    """Defect 3: edits to these three files must invalidate a producer receipt.

    Executes the real gate -- `_validate_materialization_plan`'s
    `producer != _producer_identity()` -- rather than asserting membership in
    the file list.
    """

    fixture = _tiny_checkpoint_parts(tmp_path)
    parts = list(fixture["parts"])
    baseline = checkpoint.merge_checkpoint_parts(
        fixture["source"],
        fixture["plan_dir"],
        parts,
        tmp_path / f"identity-baseline-{Path(perturbed).stem}",
    )
    assert baseline["state"] == "MATERIALIZED"

    original = checkpoint._sha256_file
    target = Path(perturbed).name

    def _perturbed(path):
        digest = original(path)
        if Path(path).name == target:
            return hashlib.sha256(digest.encode("utf-8")).hexdigest()
        return digest

    monkeypatch.setattr(checkpoint, "_sha256_file", _perturbed)
    with pytest.raises(RuntimeError, match="implementation/container identity differs"):
        checkpoint.merge_checkpoint_parts(
            fixture["source"],
            fixture["plan_dir"],
            parts,
            tmp_path / f"identity-perturbed-{Path(perturbed).stem}",
        )


def _tiny_checkpoint_parts(tmp_path: Path) -> dict[str, object]:
    fixture = _tiny_planned_checkpoint(tmp_path)
    source = fixture["source"]
    plan_dir = fixture["plan_dir"]
    shard_names = sorted(set(fixture["weight_map"].values()))
    partial_sources: list[Path] = []
    parts: list[Path] = []
    for ordinal, shard_name in enumerate(shard_names):
        partial = tmp_path / f"resume-partial-source-{ordinal}"
        partial.mkdir()
        for metadata_name in (
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ):
            shutil.copy2(source / metadata_name, partial / metadata_name)
        shutil.copy2(source / shard_name, partial / shard_name)
        part = tmp_path / f"resume-part-{ordinal}"
        checkpoint.materialize_checkpoint_part(
            partial,
            plan_dir,
            part,
            [shard_name],
            device="cpu",
        )
        partial_sources.append(partial)
        parts.append(part)
    metadata_only = tmp_path / "resume-metadata-only-source"
    metadata_only.mkdir()
    for metadata_name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        shutil.copy2(source / metadata_name, metadata_only / metadata_name)
    return {
        **fixture,
        "shard_names": shard_names,
        "partial_sources": partial_sources,
        "parts": parts,
        "metadata_only": metadata_only,
    }


def test_streaming_materialize_resume_no_clobber_and_exact_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    source = fixture["source"]
    plan_dir = fixture["plan_dir"]
    output = tmp_path / "snapped"
    real_atomic_json = checkpoint._atomic_json
    interrupted = {"done": False}

    def interrupt_after_first_receipt(path: Path, payload: object) -> None:
        real_atomic_json(path, payload)
        if path.parent.name == ".prismasnap-receipts" and not interrupted["done"]:
            interrupted["done"] = True
            raise RuntimeError("synthetic shard interruption")

    monkeypatch.setattr(checkpoint, "_atomic_json", interrupt_after_first_receipt)
    with pytest.raises(RuntimeError, match="synthetic shard interruption"):
        checkpoint.materialize_checkpoint(
            source, plan_dir, output, device="cpu", resume=False
        )
    temporary = output.with_name(output.name + ".prismasnap-incomplete")
    assert not output.exists()
    assert temporary.is_dir()
    assert len(list((temporary / ".prismasnap-receipts").glob("*.json"))) == 1

    # A different, internally valid plan cannot appropriate the prior plan's
    # partial shards during resume.
    other_plan = tmp_path / "other-plan"
    shutil.copytree(plan_dir, other_plan)
    other_manifest_path = other_plan / checkpoint.PLAN_JSON
    other_manifest = json.loads(other_manifest_path.read_text(encoding="utf-8"))
    other_manifest["search"]["max_rounds"] = 2
    other_manifest["plan_sha256"] = checkpoint._plan_digest(other_manifest)
    _json(other_manifest_path, other_manifest)
    with pytest.raises(RuntimeError, match="resume state belongs to different inputs"):
        checkpoint.materialize_checkpoint(
            source, other_plan, output, device="cpu", resume=True
        )

    monkeypatch.setattr(checkpoint, "_atomic_json", real_atomic_json)
    provenance = checkpoint.materialize_checkpoint(
        source, plan_dir, output, device="cpu", resume=True
    )
    assert output.is_dir()
    assert not temporary.exists()
    assert provenance["output"]["tensors"] == len(fixture["weight_map"])
    assert provenance["output"]["shards"] == 2
    assert provenance["coverage"]["materialized_changed_tensors"] == provenance[
        "coverage"
    ]["transformed_tensors"]
    assert provenance["coverage"]["materialized_changed_tensors"] > 0
    assert not (output / ".prismasnap-receipts").exists()
    assert not (output / "materialization_state.json").exists()
    assert (output / "tokenizer.json").read_text(encoding="utf-8") == "{}\n"

    output_index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert output_index["weight_map"] == fixture["weight_map"]
    for key, source_value in fixture["source_tensors"].items():
        shard = fixture["weight_map"][key]
        with safe_open(str(output / shard), framework="pt") as handle:
            materialized = handle.get_tensor(key)
        # The plan uses the no-op-only candidate set, so streaming still runs
        # every declared fold while providing an exact value oracle.
        assert torch.equal(materialized, source_value)

    provenance_bytes = (output / checkpoint.PROVENANCE_JSON).read_bytes()
    with pytest.raises(RuntimeError, match="output already exists"):
        checkpoint.materialize_checkpoint(
            source, plan_dir, output, device="cpu", resume=False
        )
    assert (output / checkpoint.PROVENANCE_JSON).read_bytes() == provenance_bytes


def test_checkpoint_parts_require_exact_union_and_collate_independent_bytes(
    tmp_path: Path,
) -> None:
    fixture = _tiny_planned_checkpoint(tmp_path)
    source = fixture["source"]
    plan_dir = fixture["plan_dir"]
    shard_names = sorted(set(fixture["weight_map"].values()))
    assert len(shard_names) == 2

    partial_sources: list[Path] = []
    for ordinal, shard_name in enumerate(shard_names):
        partial = tmp_path / f"partial-source-{ordinal}"
        partial.mkdir()
        for metadata_name in (
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ):
            shutil.copy2(source / metadata_name, partial / metadata_name)
        shutil.copy2(source / shard_name, partial / shard_name)
        partial_sources.append(partial)
    metadata_only = tmp_path / "metadata-only-source"
    metadata_only.mkdir()
    for metadata_name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        shutil.copy2(source / metadata_name, metadata_only / metadata_name)

    left = tmp_path / "part-left"
    right = tmp_path / "part-right"
    left_manifest = checkpoint.materialize_checkpoint_part(
        partial_sources[0], plan_dir, left, [shard_names[0]], device="cpu"
    )
    right_manifest = checkpoint.materialize_checkpoint_part(
        partial_sources[1], plan_dir, right, [shard_names[1]], device="cpu"
    )
    assert left_manifest["schema"] == checkpoint.PART_SCHEMA
    assert right_manifest["schema"] == checkpoint.PART_SCHEMA

    with pytest.raises(RuntimeError, match="overlap"):
        checkpoint.merge_checkpoint_parts(
            metadata_only, plan_dir, [left, left], tmp_path / "bad-merge"
        )

    output = tmp_path / "merged"
    provenance = checkpoint.merge_checkpoint_parts(
        metadata_only, plan_dir, [left, right], output
    )
    assert provenance["collation"]["exact_disjoint_shard_union"] is True
    assert provenance["coverage"]["materialized_changed_tensors"] == provenance[
        "coverage"
    ]["transformed_tensors"]
    for name in shard_names:
        part_shard = (left / name) if (left / name).exists() else (right / name)
        assert (output / name).stat().st_ino != part_shard.stat().st_ino
        assert checkpoint._sha256_file(output / name) == checkpoint._sha256_file(
            part_shard
        )


def test_checkpoint_part_merge_required_hardlinks_are_content_bound_and_resumable(
    tmp_path: Path,
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)
    output = tmp_path / "hardlinked-merge"
    provenance = checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        fixture["parts"],
        output,
        require_hardlinks=True,
        resume=True,
    )
    assert provenance["collation"]["shard_transfer_strategy"] == "hardlink_required"
    for name in fixture["shard_names"]:
        part_shard = next(path / name for path in fixture["parts"] if (path / name).is_file())
        assert output.joinpath(name).stat().st_dev == part_shard.stat().st_dev
        assert output.joinpath(name).stat().st_ino == part_shard.stat().st_ino
        assert checkpoint._sha256_file(output / name) == checkpoint._sha256_file(
            part_shard
        )
    assert checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        fixture["parts"],
        output,
        require_hardlinks=True,
        resume=True,
    ) == provenance


def test_checkpoint_part_merge_required_hardlink_refuses_without_copy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)

    def refuse_link(*_args, **_kwargs) -> None:
        raise OSError("synthetic cross-filesystem hardlink refusal")

    monkeypatch.setattr(checkpoint.os, "link", refuse_link)
    with pytest.raises(RuntimeError, match="required hardlink failed"):
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            fixture["parts"],
            tmp_path / "hardlink-refused",
            require_hardlinks=True,
            resume=True,
        )


def test_load_part_rejects_strict_receipt_and_count_tampering(
    tmp_path: Path,
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)
    part = fixture["parts"][0]
    original_manifest = (part / "part.json").read_bytes()
    plan = checkpoint.load_plan(fixture["plan_dir"])[0]
    source = checkpoint._Checkpoint(
        fixture["metadata_only"], require_all_shards=False
    )

    def write_tampered(mutator) -> None:
        payload = json.loads(original_manifest)
        mutator(payload)
        payload.pop("part_sha256")
        payload["part_sha256"] = checkpoint.canonical_json_sha256(
            payload, where="tampered test checkpoint part"
        )
        _json(part / "part.json", payload)

    mutations = [
        lambda payload: payload["shards"][0].__setitem__(
            "source_sha256", "0" * 64
        ),
        lambda payload: payload["shards"][0].__setitem__(
            "source_bytes", payload["shards"][0]["source_bytes"] + 1
        ),
        lambda payload: payload["shards"][0].__setitem__(
            "tensor_count", payload["shards"][0]["tensor_count"] + 1
        ),
        lambda payload: (
            payload["shards"][0].__setitem__(
                "changed_tensors",
                payload["shards"][0]["changed_tensors"] + 1,
            ),
            payload.__setitem__(
                "changed_tensors", payload["changed_tensors"] + 1
            ),
        ),
        lambda payload: payload["shards"][0].__setitem__("unexpected", True),
    ]
    for mutate in mutations:
        write_tampered(mutate)
        with pytest.raises(RuntimeError):
            checkpoint._load_part(part, plan=plan, source=source)
        (part / "part.json").write_bytes(original_manifest)

    shard_name = fixture["shard_names"][0]
    shard_path = part / shard_name
    original_shard = shard_path.read_bytes()
    shard_path.write_bytes(original_shard + b"tampered")
    with pytest.raises(RuntimeError, match="shard digest failed"):
        checkpoint._load_part(part, plan=plan, source=source)
    shard_path.write_bytes(original_shard)


def test_metadata_only_part_validation_rejects_rehashed_header_tampering(
    tmp_path: Path,
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)
    part = fixture["parts"][0]
    shard_name = fixture["shard_names"][0]
    shard_path = part / shard_name
    with safe_open(str(shard_path), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    victim = next(name for name, value in tensors.items() if value.ndim == 2)
    tensors[victim] = tensors[victim].reshape(-1)
    save_file(tensors, str(shard_path), metadata={"format": "pt"})

    manifest_path = part / "part.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = next(
        row for row in payload["shards"] if row["source_name"] == shard_name
    )
    receipt["output_bytes"] = shard_path.stat().st_size
    receipt["output_sha256"] = checkpoint._sha256_file(shard_path)
    payload.pop("part_sha256")
    payload["part_sha256"] = checkpoint.canonical_json_sha256(
        payload, where="rehashed header-tampered checkpoint part"
    )
    _json(manifest_path, payload)

    plan = checkpoint.load_plan(fixture["plan_dir"])[0]
    metadata_only = checkpoint._Checkpoint(
        fixture["metadata_only"], require_all_shards=False
    )
    with pytest.raises(RuntimeError, match="changed tensor metadata"):
        checkpoint._load_part(part, plan=plan, source=metadata_only)


def test_checkpoint_part_merge_resume_skips_verified_and_rejects_equivocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)
    parts = fixture["parts"]
    output = tmp_path / "resumed-collation"
    staging = output.with_name(output.name + ".prismasnap-incomplete")
    real_atomic_json = checkpoint._atomic_json
    interrupted = {"done": False}

    def interrupt_after_receipt(path: Path, payload: object) -> None:
        real_atomic_json(path, payload)
        if (
            path.parent.name == checkpoint.PART_MERGE_RECEIPTS_DIR
            and not interrupted["done"]
        ):
            interrupted["done"] = True
            raise RuntimeError("synthetic collation shard interruption")

    monkeypatch.setattr(checkpoint, "_atomic_json", interrupt_after_receipt)
    with pytest.raises(RuntimeError, match="synthetic collation shard interruption"):
        # Identical sealed-stage argv may always carry --resume, including on
        # the very first invocation when no staging directory exists yet.
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            parts,
            output,
            resume=True,
        )
    receipts_dir = staging / checkpoint.PART_MERGE_RECEIPTS_DIR
    verified_receipts = list(receipts_dir.glob("*.json"))
    assert len(verified_receipts) == 1
    verified_name = verified_receipts[0].name.removesuffix(".json")
    verified_inode = (staging / verified_name).stat().st_ino

    # A semantically identical but byte-reformatted part manifest changes its
    # durable file identity and cannot appropriate the existing staging state.
    part_manifest = parts[0] / "part.json"
    original_manifest = part_manifest.read_bytes()
    semantic_manifest = json.loads(original_manifest)
    part_manifest.write_text(
        json.dumps(semantic_manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(checkpoint, "_atomic_json", real_atomic_json)
    with pytest.raises(RuntimeError, match="belongs to different inputs"):
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            parts,
            output,
            resume=True,
        )
    part_manifest.write_bytes(original_manifest)

    real_copy = checkpoint._copy_file_durable
    copied_shards: list[str] = []

    def observe_copy(source: Path, destination: Path) -> None:
        if source.name in fixture["shard_names"]:
            copied_shards.append(source.name)
        real_copy(source, destination)

    monkeypatch.setattr(checkpoint, "_copy_file_durable", observe_copy)
    provenance = checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        parts,
        output,
        resume=True,
    )
    assert (output / verified_name).stat().st_ino == verified_inode
    assert verified_name not in copied_shards
    assert set(copied_shards) == set(fixture["shard_names"]) - {verified_name}
    assert provenance["collation"]["ordered_part_bindings_sha256"]
    assert checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        parts,
        output,
        resume=True,
    ) == provenance
    with pytest.raises(RuntimeError, match="different ordered parts"):
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            list(reversed(parts)),
            output,
            resume=True,
        )


def test_checkpoint_part_merge_recovers_commit_ready_and_committed_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _tiny_checkpoint_parts(tmp_path)
    parts = fixture["parts"]
    commit_ready_output = tmp_path / "commit-ready-collation"
    commit_ready_staging = commit_ready_output.with_name(
        commit_ready_output.name + ".prismasnap-incomplete"
    )
    real_replace = checkpoint.os.replace
    interrupted = {"done": False}

    def interrupt_before_rename(source: str | Path, destination: str | Path) -> None:
        if (
            Path(source) == commit_ready_staging
            and Path(destination) == commit_ready_output
            and not interrupted["done"]
        ):
            interrupted["done"] = True
            raise RuntimeError("synthetic commit-ready interruption")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", interrupt_before_rename)
    with pytest.raises(RuntimeError, match="synthetic commit-ready interruption"):
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            parts,
            commit_ready_output,
            resume=True,
        )
    assert commit_ready_staging.is_dir()
    assert (commit_ready_staging / checkpoint.PROVENANCE_JSON).is_file()
    assert not (commit_ready_staging / checkpoint.PART_MERGE_STATE_JSON).exists()

    monkeypatch.setattr(checkpoint.os, "replace", real_replace)
    commit_ready = checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        parts,
        commit_ready_output,
        resume=True,
    )
    assert commit_ready_output.is_dir()

    committed_output = tmp_path / "committed-window-collation"
    real_fsync_dir = checkpoint._fsync_dir
    committed_interrupted = {"done": False}

    def interrupt_after_rename(path: Path) -> None:
        real_fsync_dir(path)
        if (
            committed_output.is_dir()
            and Path(path) == committed_output.parent
            and not committed_interrupted["done"]
        ):
            committed_interrupted["done"] = True
            raise RuntimeError("synthetic committed-window interruption")

    monkeypatch.setattr(checkpoint, "_fsync_dir", interrupt_after_rename)
    with pytest.raises(RuntimeError, match="synthetic committed-window interruption"):
        checkpoint.merge_checkpoint_parts(
            fixture["metadata_only"],
            fixture["plan_dir"],
            parts,
            committed_output,
            resume=True,
        )
    assert committed_output.is_dir()
    monkeypatch.setattr(checkpoint, "_fsync_dir", real_fsync_dir)
    committed = checkpoint.merge_checkpoint_parts(
        fixture["metadata_only"],
        fixture["plan_dir"],
        parts,
        committed_output,
        resume=True,
    )
    assert committed == commit_ready


def _write_fp8_source(root: Path, *, paired: bool = True) -> torch.Tensor:
    """One indexed shard holding a block-scaled FP8 weight and a BF16 norm.

    Returns the fp64 reference value the checkpoint encodes, so the caller can
    assert the dequant against arithmetic computed outside the code under test
    rather than against the code's own output.
    """
    rows, cols = 4, 8
    block = (2, 4)
    root.mkdir(parents=True, exist_ok=True)
    codes = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    weight = (codes / 64.0).to(torch.float8_e4m3fn)
    # A distinct scale per block, so a transposed or mis-sized grid cannot
    # coincidentally reproduce the reference.
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    reference = weight.to(torch.float64) * (
        scale.to(torch.float64)
        .repeat_interleave(block[0], dim=0)
        .repeat_interleave(block[1], dim=1)
    )
    tensors = {
        "model.layers.0.attn.q_proj.weight": weight,
        "model.layers.0.input_layernorm.weight": torch.ones(cols, dtype=torch.bfloat16),
    }
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    if paired:
        tensors["model.layers.0.attn.q_proj.weight_scale_inv"] = scale
        weight_map["model.layers.0.attn.q_proj.weight_scale_inv"] = (
            "model-00001-of-00001.safetensors"
        )
    save_file(tensors, str(root / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (root / "config.json").write_text(
        json.dumps({
            "model_type": "test",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": list(block),
            },
        })
    )
    return reference


def test_load_bf16_dequants_a_block_scaled_fp8_source(tmp_path: Path) -> None:
    """The dequant must reproduce an independently computed reference.

    This is the test that catches a wrong block grid: a transposed or assumed
    128x128 grid is numel-compatible with the reshape and would mis-scale every
    block silently, so only comparing against arithmetic done outside the
    implementation proves anything.
    """
    root = tmp_path / "fp8-source"
    reference = _write_fp8_source(root)
    source = checkpoint._Checkpoint(root)
    key = "model.layers.0.attn.q_proj.weight"

    assert source.fp8_scale_key(key) == f"{key}_scale_inv"
    assert source.dequant_block() == (2, 4)

    loaded = source.load_bf16(key, torch.device("cpu"))
    assert loaded.dtype == torch.bfloat16
    torch.testing.assert_close(
        loaded.to(torch.float64), reference.to(torch.bfloat16).to(torch.float64)
    )

    # A BF16 tensor is returned untouched, so this path is a byte-identical
    # no-op on every checkpoint PrismaSnap already supported.
    norm_key = "model.layers.0.input_layernorm.weight"
    assert torch.equal(
        source.load_bf16(norm_key, torch.device("cpu")),
        source.load(norm_key, torch.device("cpu")),
    )


def test_load_bf16_refuses_fp8_with_no_paired_block_scale(tmp_path: Path) -> None:
    root = tmp_path / "fp8-unpaired"
    _write_fp8_source(root, paired=False)
    source = checkpoint._Checkpoint(root)
    with pytest.raises(RuntimeError, match="no weight_scale_inv"):
        source.load_bf16(
            "model.layers.0.attn.q_proj.weight", torch.device("cpu")
        )


def test_fp8_fold_matches_an_fp64_reference(tmp_path: Path) -> None:
    """Fold on the dequantized value, in BF16 -- never back into FP8.

    Letting ``apply_diagonal_transform`` restore the source dtype would round
    the folded value into float8_e4m3fn while its per-block scale stayed put:
    wrong values, no error. This asserts the folded result against an fp64
    reference instead, which is the only way to see that.
    """
    root = tmp_path / "fp8-fold"
    reference = _write_fp8_source(root)
    source = checkpoint._Checkpoint(root)
    value = source.load_bf16(
        "model.layers.0.attn.q_proj.weight", torch.device("cpu")
    )
    vector = torch.linspace(0.5, 2.0, value.shape[1], dtype=torch.float64)

    folded = prismasnap.apply_diagonal_transform(
        value, vector, "multiply", 1, output_dtype=torch.bfloat16
    )
    assert folded.dtype == torch.bfloat16
    expected = (reference.to(torch.bfloat16).to(torch.float64) * vector).to(
        torch.bfloat16
    )
    torch.testing.assert_close(folded, expected)


def test_uplift_preflight_refuses_before_writing_any_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An FP8-source materialization must fail closed at plan validation.

    The publishability refusals live downstream (``_copy_checkpoint_metadata``
    rewrites the config, ``_verify_output_census`` the dtype census), and
    reaching them means having already streamed the whole body. On a 284B FP8
    source that is hours and a full checkpoint of disk spent on a run that
    cannot commit. This pins the refusal to the front and proves the output
    tree -- committed *and* incomplete -- was never created.
    """
    fixture = _tiny_planned_checkpoint(tmp_path)
    monkeypatch.setattr(
        checkpoint,
        "_uplifted_source_tensors",
        lambda plan: {"model.layers.0.mlp.down_proj.weight"},
    )
    output = tmp_path / "materialized"
    with pytest.raises(RuntimeError, match="before any bytes are written"):
        checkpoint.materialize_checkpoint(
            fixture["source"], fixture["plan_dir"], output, device="cpu"
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".prismasnap-incomplete").exists()


def test_uplift_preflight_is_a_noop_on_a_bf16_census_and_names_operands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate keys off the plan's attested dtype census, and says why.

    A BF16 source must pass untouched -- the preflight is not allowed to be a
    blanket refusal. When it does fire it has to name the operands and the
    reason, so the message is actionable rather than "unsupported".
    """
    fixture = _tiny_planned_checkpoint(tmp_path)
    source = checkpoint._Checkpoint(Path(fixture["source"]))
    assert checkpoint._uplifted_source_tensors(fixture["plan"]) == set()
    checkpoint._preflight_uplift_publishability(fixture["plan"], source)

    target = sorted(checkpoint._transforms_by_tensor(fixture["plan"]))[0]
    monkeypatch.setattr(
        checkpoint, "_uplifted_source_tensors", lambda plan: {target}
    )
    with pytest.raises(RuntimeError) as excinfo:
        checkpoint._preflight_uplift_publishability(fixture["plan"], source)
    message = str(excinfo.value)
    assert target in message
    assert "quantization_config" in message
    assert "owed design work" in message
