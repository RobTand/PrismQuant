from __future__ import annotations

from pathlib import Path

import pytest

from prismaquant.rtx4090_graph_contract import (
    RTX4090_CUDAGRAPH_CAPTURE_SIZES,
    RTX4090_GRAPH_COMPILATION_CONFIG,
    RTX4090GraphContractError,
    compilation_config_json,
    create_compile_cache_preflight,
    validate_compile_cache_preflight,
    validate_rtx4090_graph_log,
)


_CACHE_ROOT = "/opt/pq-compile-cache/run-abc"
_CACHE = f"{_CACHE_ROOT}/f00dbabe/rank_0_0/backbone"
_SIZES = list(RTX4090_CUDAGRAPH_CAPTURE_SIZES)


def _log(*, mode: int = 3, graph_mode: str = "FULL_AND_PIECEWISE") -> str:
    mode_name = "VLLM_COMPILE" if mode == 3 else "NONE"
    return "\n".join(
        (
            "Initializing a V1 LLM engine (test) with config: "
            "max_seq_len=32768, "
            "compilation_config={'mode': "
            f"<CompilationMode.{mode_name}: {mode}>, "
            "'backend': 'inductor', "
            f"'cudagraph_mode': <CUDAGraphMode.{graph_mode}: (2, 1)>, "
            f"'cudagraph_capture_sizes': {_SIZES}, "
            "'max_cudagraph_capture_size': 64}",
            f"Using cache directory: {_CACHE} for vLLM's torch.compile",
            "Dynamo bytecode transform time: 10.60 s",
            "Compiling a graph for compile range (1, 32768) takes 47.50 s",
            "torch.compile took 64.13 s in total",
            "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): "
            "100%|##########| 7/7",
            "Capturing CUDA graphs (decode, FULL): 100%|##########| 7/7",
            "Graph capturing finished in 9 secs, took 1.73 GiB",
        )
    )


def test_compilation_config_is_the_exact_production_profile():
    assert RTX4090_GRAPH_COMPILATION_CONFIG == (
        '{"mode":3,"backend":"inductor",'
        '"cudagraph_mode":"FULL_AND_PIECEWISE",'
        '"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
    )
    assert compilation_config_json(cache_dir=_CACHE_ROOT) == (
        '{"mode":3,"backend":"inductor",'
        '"cudagraph_mode":"FULL_AND_PIECEWISE",'
        '"cudagraph_capture_sizes":[1,2,4,8,16,32,64],'
        f'"cache_dir":"{_CACHE_ROOT}"}}'
    )


@pytest.mark.parametrize("cache_dir", ("relative", "/", "/tmp/../bad"))
def test_compilation_config_rejects_non_dedicated_cache(cache_dir: str):
    with pytest.raises(ValueError, match="dedicated absolute"):
        compilation_config_json(cache_dir=cache_dir)


def test_positive_mode3_full_and_piecewise_receipt(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    receipt = validate_rtx4090_graph_log(
        path, expected_compile_cache=_CACHE
    )

    assert receipt["compilation_mode"] == 3
    assert receipt["compilation_backend"] == "inductor"
    assert receipt["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert receipt["capture_sizes"] == _SIZES
    assert receipt["max_model_len"] == 32768
    assert receipt["piecewise_capture_count"] == 7
    assert receipt["full_capture_count"] == 7
    assert receipt["configured_compile_cache_root"] == _CACHE
    assert receipt["compile_cache"] == _CACHE
    assert len(receipt["serve_log_sha256"]) == 64


@pytest.mark.parametrize(
    "replacement, message",
    (
        (
            "<CompilationMode.NONE: 0>",
            "forbidden compile/graph marker",
        ),
        (
            "<CUDAGraphMode.PIECEWISE: (1, 0)>",
            "mode-3 FULL_AND_PIECEWISE",
        ),
        ("[1, 2, 4, 8]", "capture sizes differ"),
    ),
)
def test_wrong_resolved_profile_fails(
    tmp_path: Path, replacement: str, message: str
):
    text = _log()
    if replacement.startswith("<CompilationMode"):
        text = text.replace("<CompilationMode.VLLM_COMPILE: 3>", replacement)
    elif replacement.startswith("<CUDAGraphMode"):
        text = text.replace(
            "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>", replacement
        )
    else:
        text = text.replace(str(_SIZES), replacement)
    path = tmp_path / "serve.log"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match=message):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)


@pytest.mark.parametrize(
    "marker",
    (
        "Inductor compilation was disabled by user settings",
        "Skipping CUDA graph capture",
        "torch._inductor.exc.InductorError: compile failed",
        "torch._dynamo hit config.recompile_limit",
        "Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE",
    ),
)
def test_disable_downgrade_and_fallback_markers_fail(
    tmp_path: Path, marker: str
):
    path = tmp_path / "serve.log"
    path.write_text(_log() + "\n" + marker, encoding="utf-8")

    with pytest.raises(
        RTX4090GraphContractError, match="forbidden compile/graph marker"
    ):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)


def test_cached_or_uncompiled_run_cannot_stand_in_for_fresh_compile(tmp_path: Path):
    path = tmp_path / "serve.log"
    text = _log().replace("Dynamo bytecode transform time: 10.60 s\n", "")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="fresh Dynamo/Inductor"):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)


@pytest.mark.parametrize(
    ("line", "message"),
    (
        (
            "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): "
            "100%|##########| 7/7\n",
            "PIECEWISE",
        ),
        (
            "Capturing CUDA graphs (decode, FULL): 100%|##########| 7/7\n",
            "FULL decode",
        ),
    ),
)
def test_both_graph_families_need_positive_completion(
    tmp_path: Path, line: str, message: str
):
    path = tmp_path / "serve.log"
    path.write_text(_log().replace(line, ""), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match=message):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)


def test_full_decode_must_capture_the_complete_requested_ladder(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(
        _log().replace(
            "Capturing CUDA graphs (decode, FULL): 100%|##########| 7/7",
            "Capturing CUDA graphs (decode, FULL): 100%|##########| 1/1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RTX4090GraphContractError, match="every requested FULL"):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)


def test_expected_cache_is_bound(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="fresh expected cache"):
        validate_rtx4090_graph_log(
            path, expected_compile_cache="/opt/pq-compile-cache/other"
        )


def test_compile_cache_can_be_bound_under_a_fresh_root(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    receipt = validate_rtx4090_graph_log(
        path, expected_compile_cache_root=_CACHE_ROOT
    )

    assert receipt["configured_compile_cache_root"] == _CACHE_ROOT
    assert receipt["compile_cache"] == _CACHE


def test_compile_cache_outside_fresh_root_fails(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="outside"):
        validate_rtx4090_graph_log(
            path, expected_compile_cache_root="/opt/other-cache-root"
        )


def test_exact_cache_and_root_are_mutually_exclusive(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="not both"):
        validate_rtx4090_graph_log(
            path,
            expected_compile_cache=_CACHE,
            expected_compile_cache_root=_CACHE_ROOT,
        )


def test_compile_cache_evidence_requires_an_expected_path(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log(), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="path/root"):
        validate_rtx4090_graph_log(path)


def test_compile_cache_preflight_binds_empty_root_to_post_tree(tmp_path: Path):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    nonce = "ab" * 16
    created = create_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )
    assert created["prelaunch_empty"] is True
    (cache / "fingerprint" / "rank_0_0").mkdir(parents=True)
    (cache / "fingerprint" / "rank_0_0" / "graph.py").write_text("compiled")

    validated = validate_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )
    assert validated["post_file_count"] == 1
    assert validated["post_total_bytes"] == len("compiled")
    assert len(validated["post_tree_sha256"]) == 64


def test_compile_cache_preflight_refuses_preexisting_root(tmp_path: Path):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    cache.mkdir()
    (cache / "stale").write_text("old")
    with pytest.raises(RTX4090GraphContractError, match="already exists"):
        create_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce="ab" * 16,
        )


def test_compile_cache_preflight_refuses_preexisting_receipt_without_creating_root(
    tmp_path: Path,
):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    receipt.write_text("stale", encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="already exists"):
        create_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce="ab" * 16,
        )

    assert not cache.exists()


def test_compile_cache_preflight_refuses_receipt_inside_cache_root(
    tmp_path: Path,
):
    cache = tmp_path / "fresh-cache"

    with pytest.raises(RTX4090GraphContractError, match="outside"):
        create_compile_cache_preflight(
            cache,
            cache / "preflight.json",
            configured_container_root="/compile-cache",
            session_nonce="ab" * 16,
        )

    assert not cache.exists()


def test_compile_cache_preflight_requires_nonempty_post_tree(tmp_path: Path):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    nonce = "ab" * 16
    create_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )

    with pytest.raises(RTX4090GraphContractError, match="stayed empty"):
        validate_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce=nonce,
        )


def test_compile_cache_preflight_refuses_replaced_root(tmp_path: Path):
    cache = tmp_path / "fresh-cache"
    displaced = tmp_path / "displaced-cache"
    receipt = tmp_path / "cache-preflight.json"
    nonce = "ab" * 16
    create_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )
    cache.rename(displaced)
    cache.mkdir()
    (cache / "compiled.py").write_text("compiled", encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="differs"):
        validate_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce=nonce,
        )


@pytest.mark.parametrize("nonce", ("ab" * 15, "AB" * 16, "not-hex"))
def test_compile_cache_preflight_requires_128_bit_lowercase_hex_nonce(
    tmp_path: Path, nonce: str,
):
    with pytest.raises(RTX4090GraphContractError, match="128-bit"):
        create_compile_cache_preflight(
            tmp_path / f"cache-{len(nonce)}",
            tmp_path / f"receipt-{len(nonce)}-{nonce[:2]}.json",
            configured_container_root="/compile-cache",
            session_nonce=nonce,
        )


@pytest.mark.parametrize("container_root", ("relative", "/", "/tmp/../bad"))
def test_compile_cache_preflight_requires_dedicated_container_root(
    tmp_path: Path, container_root: str,
):
    with pytest.raises(RTX4090GraphContractError, match="dedicated absolute"):
        create_compile_cache_preflight(
            tmp_path / "fresh-cache",
            tmp_path / "preflight.json",
            configured_container_root=container_root,
            session_nonce="ab" * 16,
        )


def test_compile_cache_preflight_refuses_symlink_member(tmp_path: Path):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    nonce = "cd" * 16
    create_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )
    target = tmp_path / "outside"
    target.write_text("compiled")
    (cache / "linked").symlink_to(target)
    with pytest.raises(RTX4090GraphContractError, match="symlink"):
        validate_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce=nonce,
        )


def test_compile_cache_preflight_refuses_replaced_symlink_receipt(
    tmp_path: Path,
):
    cache = tmp_path / "fresh-cache"
    receipt = tmp_path / "cache-preflight.json"
    saved = tmp_path / "saved-preflight.json"
    nonce = "cd" * 16
    create_compile_cache_preflight(
        cache,
        receipt,
        configured_container_root="/compile-cache",
        session_nonce=nonce,
    )
    receipt.replace(saved)
    receipt.symlink_to(saved)
    (cache / "compiled.py").write_text("compiled", encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="ordinary file"):
        validate_compile_cache_preflight(
            cache,
            receipt,
            configured_container_root="/compile-cache",
            session_nonce=nonce,
        )


def test_concatenated_engine_sessions_are_refused(tmp_path: Path):
    path = tmp_path / "serve.log"
    path.write_text(_log() + "\n" + _log(), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match="exactly one resolved"):
        validate_rtx4090_graph_log(
            path, expected_compile_cache_root=_CACHE_ROOT
        )


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    (
        ("'backend': 'inductor'", "'backend': ''", "explicit Inductor"),
        ("max_seq_len=32768", "max_seq_len=16384", "32K"),
    ),
)
def test_backend_and_resolved_context_are_exact(
    tmp_path: Path, needle: str, replacement: str, message: str
):
    path = tmp_path / "serve.log"
    path.write_text(_log().replace(needle, replacement), encoding="utf-8")

    with pytest.raises(RTX4090GraphContractError, match=message):
        validate_rtx4090_graph_log(path, expected_compile_cache=_CACHE)
