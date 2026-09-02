from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pytest

import prismaquant.dspark_matched_performance as perf
import prismaquant.dspark_matched_performance_collector as collector


def _source_root() -> Path:
    return Path(perf.__file__).resolve().parents[1]


def _policy_declaration(source: dict, tool: dict) -> dict:
    declared_at = "2026-08-13T00:00:00Z"
    payload = {
        "schema": collector.POLICY_DECLARATION_SCHEMA,
        "declared_at": declared_at,
        "policy": perf.release_policy(predeclared_at=declared_at),
        "source_snapshot": source,
        "tool": tool,
    }
    payload["declaration_sha256"] = perf.canonical_sha256(payload)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_tool_identity_is_source_closed_and_rejects_forgery() -> None:
    identity = perf.collector_tool_identity(
        source_root=_source_root(), git_commit="a" * 40
    )
    assert identity["name"] == perf.COLLECTOR_TOOL_NAME
    assert set(identity["source_files"]) == set(perf.COLLECTOR_SOURCE_PATHS)
    assert identity["collector_source_sha256"] == perf.file_sha256(
        _source_root() / "prismaquant/dspark_matched_performance_collector.py"
    )
    assert perf._validate_tool(identity, verify_local_source=True) == identity

    forged = deepcopy(identity)
    forged["source_files"][
        "prismaquant/dspark_matched_performance_collector.py"
    ] = "f" * 64
    forged["collector_source_sha256"] = "f" * 64
    forged["source_files_sha256"] = perf.canonical_sha256(
        dict(sorted(forged["source_files"].items()))
    )
    with pytest.raises(
        perf.DSparkMatchedPerformanceError, match="attester source"
    ):
        perf._validate_tool(forged, verify_local_source=True)

    renamed = deepcopy(identity)
    renamed["name"] = "validate_dspark_target_draft.py"
    with pytest.raises(
        perf.DSparkMatchedPerformanceError, match="source-closed"
    ):
        perf._validate_tool(renamed)


def test_policy_declaration_replays_file_source_and_digest(tmp_path: Path) -> None:
    source = {
        "schema": collector.SOURCE_SNAPSHOT_SCHEMA,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "closure_sha256": "c" * 64,
        "entry_count": 10,
    }
    tool = perf.collector_tool_identity(
        source_root=_source_root(), git_commit=source["commit"]
    )
    declaration = _policy_declaration(source, tool)
    path = tmp_path / "policy.json"
    _write_json(path, declaration)
    observed, digest = collector._load_policy_declaration(
        path, source_snapshot=source, tool=tool
    )
    assert observed == declaration
    assert digest == perf.file_sha256(path)

    declaration["policy"]["minimum_ready_mem_available_bytes"] -= 1
    declaration["declaration_sha256"] = perf.canonical_sha256(
        {
            key: value
            for key, value in declaration.items()
            if key != "declaration_sha256"
        }
    )
    _write_json(tmp_path / "tampered.json", declaration)
    with pytest.raises(collector.DSparkCollectorError, match="stale or foreign"):
        collector._load_policy_declaration(
            tmp_path / "tampered.json", source_snapshot=source, tool=tool
        )


def test_strict_json_and_no_clobber_reject_operator_substitution(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(collector.DSparkCollectorError, match="duplicate key"):
        collector._strict_json(duplicate)

    target = tmp_path / "evidence.json"
    collector._atomic_no_clobber(target, b"first\n")
    with pytest.raises(collector.DSparkCollectorError, match="replace existing"):
        collector._atomic_no_clobber(target, b"second\n")
    assert target.read_bytes() == b"first\n"

    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(collector.DSparkCollectorError, match="regular JSON"):
        collector._strict_json(link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_output = alias_parent / "nested" / "evidence.json"
    with pytest.raises(collector.DSparkCollectorError, match="non-canonical"):
        collector._atomic_no_clobber(aliased_output, b"unsafe\n")
    assert not (real_parent / "nested").exists()


def test_sampler_refuses_to_begin_after_vllm_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.serve_fingerprint as fingerprint

    monkeypatch.setattr(fingerprint, "find_server_pids", lambda: [101, 202])
    with pytest.raises(collector.DSparkCollectorError, match="before vLLM"):
        collector._assert_server_not_started()


def test_counter_snapshot_rejects_cross_model_and_reason_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = "\n".join(
        [
            'vllm:generation_tokens_total{engine="0",model_name="served"} 1000',
            'vllm:request_success_total{engine="0",finished_reason="length",model_name="served"} 8',
            'vllm:request_success_total{engine="0",finished_reason="stop",model_name="served"} 2',
            'vllm:request_success_total{engine="0",finished_reason="error",model_name="served"} 1',
        ]
    )
    monkeypatch.setattr(collector, "_http_text", lambda *_args: valid)
    assert collector._request_counter_snapshot("http://server", "served") == {
        "completed_requests": 11,
        "generation_tokens": 1000,
        "failed_requests": 1,
        "timed_out_requests": 0,
    }

    foreign = valid + (
        '\nvllm:generation_tokens_total{engine="0",model_name="other"} 1'
    )
    monkeypatch.setattr(collector, "_http_text", lambda *_args: foreign)
    with pytest.raises(collector.DSparkCollectorError, match="measured model"):
        collector._request_counter_snapshot("http://server", "served")

    unknown = valid + (
        '\nvllm:request_success_total{engine="0",finished_reason="mystery",'
        'model_name="served"} 1'
    )
    monkeypatch.setattr(collector, "_http_text", lambda *_args: unknown)
    with pytest.raises(collector.DSparkCollectorError, match="unknown finish"):
        collector._request_counter_snapshot("http://server", "served")


def test_counter_ledger_is_exact_integral_and_monotonic() -> None:
    before = {
        "completed_requests": 10,
        "generation_tokens": 100,
        "failed_requests": 0,
        "timed_out_requests": 0,
    }
    after = {
        "completed_requests": 18,
        "generation_tokens": 1124,
        "failed_requests": 0,
        "timed_out_requests": 0,
    }
    assert collector._counter_ledger(before, after)["delta"] == {
        "completed_requests": 8,
        "generation_tokens": 1024,
        "failed_requests": 0,
        "timed_out_requests": 0,
    }
    contaminated = dict(after, unrelated=1)
    with pytest.raises(collector.DSparkCollectorError, match="keys differ"):
        collector._counter_ledger(before, contaminated)
    regressed = dict(after, generation_tokens=99)
    with pytest.raises(collector.DSparkCollectorError, match="regressed"):
        collector._counter_ledger(before, regressed)


def test_fixed_workload_issues_only_the_builtin_eight_by_128(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict, int]] = []

    def fake_http(base_url, route, *, payload, timeout):
        calls.append((base_url, route, payload, timeout))
        return (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": f"answer-{len(calls)}"},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 128},
            },
            0.25,
        )

    monkeypatch.setattr(collector, "_http_json", fake_http)
    rows = collector._run_fixed_workload("http://server", "served")
    assert len(rows) == len(calls) == 8
    for index, (_base_url, route, payload, timeout) in enumerate(calls):
        assert route == "/v1/chat/completions"
        assert timeout == 600
        assert payload == {
            "model": "served",
            "messages": [
                {"role": "user", "content": collector.DSPARK_PROMPTS[index]}
            ],
            "temperature": 0,
            "max_tokens": 128,
            "ignore_eos": True,
            "stream": False,
        }
    assert all(row["completion_tokens"] == 128 for row in rows)

    def short_http(*_args, **_kwargs):
        return (
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "short"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 127},
            },
            0.1,
        )

    monkeypatch.setattr(collector, "_http_json", short_http)
    with pytest.raises(collector.DSparkCollectorError, match="128-token"):
        collector._run_fixed_workload("http://server", "served")


def test_real_vllm_kv_log_capacity_is_parsed_not_hand_authored(
    tmp_path: Path,
) -> None:
    log = tmp_path / "serve.log"
    log.write_text(
        "Initial free memory 8.0 GiB, reserved 1.6 GiB memory for KV Cache as "
        "specified by kv_cache_memory_bytes config\n"
        "GPU KV cache size: 306,490 tokens, Maximum concurrency for 262,144 "
        "tokens per request: 1.17x\n",
        encoding="utf-8",
    )
    graph = {"serve_log_sha256": perf.file_sha256(log)}
    capacity = collector._kv_capacity(log, graph=graph)
    assert capacity["capacity_tokens"] == 306_490
    assert capacity["concurrency_at_max_model_len"] == 1.17
    assert perf._validate_kv_capacity(capacity, graph_capture=graph) == capacity

    duplicated = tmp_path / "duplicate-kv.log"
    duplicated.write_text(log.read_text() * 2, encoding="utf-8")
    with pytest.raises(collector.DSparkCollectorError, match="one exact"):
        collector._kv_capacity(duplicated, graph=graph)

    false_concurrency = tmp_path / "false-concurrency.log"
    false_concurrency.write_text(
        log.read_text().replace("1.17x", "1.00x"), encoding="utf-8"
    )
    with pytest.raises(collector.DSparkCollectorError, match="full 256K"):
        collector._kv_capacity(false_concurrency, graph=graph)


def test_serve_log_snapshot_is_real_stable_bytes_and_no_clobber(
    tmp_path: Path,
) -> None:
    source = tmp_path / "serve.log"
    source.write_bytes(b"startup\ngraph captured\n")
    destination = tmp_path / "snapshots" / "serve.log"
    observed = collector._snapshot_log(source, destination)
    assert observed == destination
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(collector.DSparkCollectorError, match="replace existing"):
        collector._snapshot_log(source, destination)

    alias = tmp_path / "serve-link.log"
    alias.symlink_to(source)
    with pytest.raises(collector.DSparkCollectorError, match="regular file"):
        collector._snapshot_log(alias, tmp_path / "other.log")


def test_memory_evidence_distinguishes_oom_state_from_vllm_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "serve.log"
    log.write_text(
        "reserved 1.6 GiB memory for KV Cache as specified by "
        "kv_cache_memory_bytes config. If OOM'ed, check the difference.\n"
        "GPU KV cache size: 306,490 tokens, Maximum concurrency for 262,144 "
        "tokens per request: 1.17x\n",
        encoding="utf-8",
    )
    events = tmp_path / "memory.events"
    events.write_text("oom 0\noom_kill 0\n", encoding="ascii")
    monkeypatch.setattr(
        collector,
        "_cgroup_memory_events",
        lambda: (events, {"oom": 0, "oom_kill": 0}),
    )
    values = [120, 10, 9, 9, 10]
    samples = [
        {
            "sequence": index,
            "observed_at": f"2026-08-13T00:00:0{index}Z",
            "phase": phase,
            "mem_available_bytes": value * perf.GIB,
        }
        for index, (phase, value) in enumerate(zip(collector._PHASES, values))
    ]
    state = {
        "cgroup_memory_events_path": str(events),
        "cgroup_memory_events_before": {"oom": 0, "oom_kill": 0},
    }
    graph = {"serve_log_sha256": perf.file_sha256(log)}
    memory = collector._memory_evidence(
        samples=samples,
        state=state,
        graph=graph,
        log_path=log,
        server_alive_after=True,
    )
    assert memory["oom_events"] == 0
    assert memory["oom_kill_detected"] is False
    assert memory["watchdog_tripped"] is False

    log.write_text(log.read_text() + "CUDA out of memory.\n", encoding="utf-8")
    memory = collector._memory_evidence(
        samples=samples,
        state=state,
        graph={"serve_log_sha256": perf.file_sha256(log)},
        log_path=log,
        server_alive_after=False,
    )
    assert memory["oom_kill_detected"] is True
    assert memory["server_alive_after"] is False


def test_sampler_process_owns_continuous_phase_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "schema": collector.SOURCE_SNAPSHOT_SCHEMA,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "closure_sha256": "c" * 64,
        "entry_count": 1,
    }
    tool = {"identity": "test"}
    declaration = {"declared_at": "2026-08-13T00:00:00Z"}
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    events = tmp_path / "memory.events"
    events.write_text("oom 0\noom_kill 0\n", encoding="ascii")
    monkeypatch.setattr(
        collector,
        "_release_source_identity",
        lambda: (_source_root(), source, tool),
    )
    monkeypatch.setattr(
        collector,
        "_load_policy_declaration",
        lambda *_args, **_kwargs: (declaration, "d" * 64),
    )
    monkeypatch.setattr(collector, "_assert_server_not_started", lambda: None)
    monkeypatch.setattr(collector, "_read_mem_available", lambda: 120 * perf.GIB)
    monkeypatch.setattr(
        collector, "_cgroup_memory_events", lambda: (events, {"oom": 0, "oom_kill": 0})
    )
    monkeypatch.setattr(collector, "START_MEM_AVAILABLE_FLOOR_BYTES", 1)
    monkeypatch.setattr(collector, "MEMORY_SAMPLE_INTERVAL_MS", 50)

    state_dir = tmp_path / "sampler"
    state = collector.start_sampler(
        state_dir=state_dir, policy_declaration_path=policy
    )
    stopped = False
    try:
        for phase in ("ready", "warmup", "measured", "post"):
            after = collector._set_phase(state_dir, phase)
            collector._wait_phase_sample(
                state_dir, phase, after_sequence=after, timeout=2.0
            )
        rows = collector._stop_sampler(state_dir, state)
        stopped = True
        assert rows[0]["phase"] == "startup"
        assert rows[-1]["phase"] == "post"
        phases = [row["phase"] for row in rows]
        assert all(phase in phases for phase in collector._PHASES)
        assert [row["sequence"] for row in rows] == list(range(len(rows)))
        observed = [
            collector._parse_utc(row["observed_at"], where="test") for row in rows
        ]
        assert all(
            later > earlier for earlier, later in zip(observed, observed[1:])
        )
        with pytest.raises(collector.DSparkCollectorError, match="reuse"):
            collector.start_sampler(
                state_dir=state_dir, policy_declaration_path=policy
            )
    finally:
        if not stopped:
            try:
                collector._stop_sampler(state_dir, state)
            except Exception:
                pass
        os.waitpid(state["sampler_pid"], 0)


def test_sampler_state_rejects_replaced_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = state_dir / "samples.jsonl"
    ledger.write_text(
        '{"mem_available_bytes":1,"observed_at":"2026-08-13T00:00:00Z",'
        '"phase":"startup","sequence":0}\n',
        encoding="ascii",
    )
    ledger_stat = ledger.stat()
    state = {
        "schema": collector.SAMPLER_STATE_SCHEMA,
        "created_at": "2026-08-13T00:00:00Z",
        "source_snapshot": {},
        "tool": {},
        "policy_declaration": {},
        "policy_declaration_path": str(tmp_path / "policy.json"),
        "policy_declaration_file_sha256": "a" * 64,
        "sampler_pid": 123,
        "sampler_proc_start_ticks": 456,
        "sample_interval_ms": collector.MEMORY_SAMPLE_INTERVAL_MS,
        "sample_ledger_device": ledger_stat.st_dev,
        "sample_ledger_inode": ledger_stat.st_ino,
        "cgroup_memory_events_path": "/sys/fs/cgroup/memory.events",
        "cgroup_memory_events_before": {"oom": 0, "oom_kill": 0},
        "initial_mem_available_bytes": 120 * perf.GIB,
        "source_root_sha256": "b" * 64,
    }
    _write_json(state_dir / "state.json", state)
    replacement = state_dir / "replacement"
    replacement.write_text(ledger.read_text(), encoding="ascii")
    os.replace(replacement, ledger)
    monkeypatch.setattr(collector, "_proc_start_ticks", lambda _pid: 456)
    with pytest.raises(collector.DSparkCollectorError, match="identity changed"):
        collector._load_sampler_state(state_dir)


def test_cli_exposes_no_operator_authored_evidence_inputs() -> None:
    parser = collector._parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, collector.argparse._SubParsersAction)
    )
    collect = subparsers.choices["collect-arm"]
    destinations = {action.dest for action in collect._actions}
    assert not {
        "memory_json",
        "responses_json",
        "counters_json",
        "kv_json",
        "graph_json",
        "routes_json",
        "pre_manifest",
        "post_manifest",
    } & destinations
    assert {
        "serve_log",
        "serve_log_snapshot_out",
        "pre_manifest_out",
        "post_manifest_out",
        "output_json",
    } <= destinations


def test_release_source_boundary_rejects_an_unattested_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "prismaquant" / "collector.py"
    module.parent.mkdir()
    module.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr(collector, "__file__", str(module))
    with pytest.raises(collector.DSparkCollectorError, match="regular JSON"):
        collector._release_source_identity()


def test_timestamps_are_real_utc_not_operator_strings() -> None:
    observed = collector._parse_utc(collector._utc(), where="now")
    assert observed.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - observed).total_seconds()) < 2
