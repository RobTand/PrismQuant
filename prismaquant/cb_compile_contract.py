"""Fail-closed ``torch.compile`` contract for CB producer hot paths.

Generic CB production keeps its historical compatibility behavior: individual
callers may catch a compiler failure and use their bounded eager implementation.
Campaigns that set :data:`CB_COMPILE_FAIL_CLOSED_ENV` instead receive a
``fullgraph=True`` Inductor callable whose every invocation must dispatch a
compiled graph.  Returning from the original Python function without entering
the backend is treated as an eager fallback and is an error.

The small process-local proof session is intentionally execution evidence, not
a cache or a scheduler.  It counts strict helper attempts, successful compiled
dispatches, graph creations, and failures while the existing renderer owns all
tensor residency and prefetch decisions.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
import threading
from typing import Any


CB_COMPILE_FAIL_CLOSED_ENV = "PRISMAQUANT_CB_COMPILE_FAIL_CLOSED"
CB_COMPILE_EXECUTION_PROOF_SCHEMA = (
    "prismaquant.cb_compile_execution_proof.v1"
)

ENCODE_VQ_ARGMIN = "encode.vq_dist_argmin"
ENCODE_SCORE_MIN = "encode.score_min"
ENCODE_SCORE_ARGMIN = "encode.score_argmin"
ENCODE_SCORE_MIN_BATCHED = "encode.score_min_batched"
ENCODE_SCORE_MINARGMIN_BATCHED = "encode.score_minargmin_batched"
ATOM_CHUNK_BEST = "atom.chunk_best"
ATOM_BATCHED_CHUNK_BEST = "atom.batched_chunk_best"

CB_COMPILE_HELPERS = frozenset({
    ENCODE_VQ_ARGMIN,
    ENCODE_SCORE_MIN,
    ENCODE_SCORE_ARGMIN,
    ENCODE_SCORE_MIN_BATCHED,
    ENCODE_SCORE_MINARGMIN_BATCHED,
    ATOM_CHUNK_BEST,
    ATOM_BATCHED_CHUNK_BEST,
})

_POLICY = {
    "compiler": "torch.compile",
    "backend": "inductor",
    "dynamic": True,
    "fullgraph": True,
    "suppress_errors": False,
    "fallback": "refuse",
}
_COUNTER_FIELDS = (
    "attempted_calls",
    "cuda_calls",
    "compiled_dispatches",
    "graph_compiles",
    "compile_failures",
    "runtime_failures",
    "eager_fallbacks",
)


class CBCompileContractError(RuntimeError):
    """A strict CB helper did not execute through its compiled graph."""


def cb_compile_fail_closed() -> bool:
    """Return whether the shared strict producer compile contract is active."""
    return str(os.environ.get(CB_COMPILE_FAIL_CLOSED_ENV, "0")).strip().lower() not in {
        "", "0", "false", "no", "off",
    }


def _canonical_sha256(value: object) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _Counters:
    attempted_calls: int = 0
    cuda_calls: int = 0
    compiled_dispatches: int = 0
    graph_compiles: int = 0
    compile_failures: int = 0
    runtime_failures: int = 0
    eager_fallbacks: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in _COUNTER_FIELDS}


@dataclass
class _ProofSession:
    token: int
    helpers: dict[str, _Counters] = field(default_factory=dict)


_STATE_LOCK = threading.RLock()
_ACTIVE_SESSION: _ProofSession | None = None
_NEXT_TOKEN = 0


def _counter(helper: str) -> _Counters | None:
    if helper not in CB_COMPILE_HELPERS:
        raise CBCompileContractError(f"unknown CB compile helper {helper!r}")
    with _STATE_LOCK:
        if _ACTIVE_SESSION is None:
            return None
        return _ACTIVE_SESSION.helpers.setdefault(helper, _Counters())


def _record(helper: str, field_name: str) -> None:
    if field_name not in _COUNTER_FIELDS:
        raise AssertionError(f"unknown compile-proof counter {field_name!r}")
    with _STATE_LOCK:
        counters = _counter(helper)
        if counters is not None:
            setattr(counters, field_name, int(getattr(counters, field_name)) + 1)


def begin_cb_compile_execution_proof() -> int:
    """Open one process-local strict execution-proof interval."""
    global _ACTIVE_SESSION, _NEXT_TOKEN
    if not cb_compile_fail_closed():
        raise CBCompileContractError(
            f"{CB_COMPILE_FAIL_CLOSED_ENV}=1 is required before opening a "
            "strict CB compile proof"
        )
    with _STATE_LOCK:
        if _ACTIVE_SESSION is not None:
            raise CBCompileContractError(
                "a CB compile execution-proof session is already active"
            )
        _NEXT_TOKEN += 1
        _ACTIVE_SESSION = _ProofSession(token=_NEXT_TOKEN)
        return _NEXT_TOKEN


def abort_cb_compile_execution_proof(token: int) -> None:
    """Close ``token`` without issuing evidence after a failed render."""
    global _ACTIVE_SESSION
    with _STATE_LOCK:
        if _ACTIVE_SESSION is None:
            return
        if _ACTIVE_SESSION.token != int(token):
            raise CBCompileContractError(
                "CB compile proof abort token differs from the active session"
            )
        _ACTIVE_SESSION = None


def finish_cb_compile_execution_proof(token: int) -> dict[str, object]:
    """Close ``token`` and return a checksum-bound strict execution proof."""
    global _ACTIVE_SESSION
    with _STATE_LOCK:
        if _ACTIVE_SESSION is None or _ACTIVE_SESSION.token != int(token):
            raise CBCompileContractError(
                "CB compile proof finish token differs from the active session"
            )
        session = _ACTIVE_SESSION
        _ACTIVE_SESSION = None
        helpers = {
            name: session.helpers[name].as_dict()
            for name in sorted(session.helpers)
        }
    totals = {
        field_name: sum(row[field_name] for row in helpers.values())
        for field_name in _COUNTER_FIELDS
    }
    body: dict[str, object] = {
        "schema": CB_COMPILE_EXECUTION_PROOF_SCHEMA,
        "strict_setting": {CB_COMPILE_FAIL_CLOSED_ENV: "1"},
        "policy": dict(_POLICY),
        "helpers": helpers,
        "totals": totals,
    }
    proof = {**body, "proof_sha256": _canonical_sha256(body)}
    return validate_cb_compile_execution_proof(proof)


def _strict_nonnegative_int(value: object, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise CBCompileContractError(f"{where} must be a nonnegative integer")
    return int(value)


def validate_cb_compile_execution_proof(
    proof: Mapping[str, object],
    *,
    require_live_calls: bool = False,
    require_cuda_calls: bool = False,
    allowed_helper_prefixes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate a closed proof and optionally require live compiled work."""
    expected_keys = {
        "schema", "strict_setting", "policy", "helpers", "totals",
        "proof_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_keys:
        raise CBCompileContractError("CB compile execution proof shape is not closed")
    body = dict(proof)
    digest = body.pop("proof_sha256")
    if digest != _canonical_sha256(body):
        raise CBCompileContractError("CB compile execution proof checksum differs")
    if body["schema"] != CB_COMPILE_EXECUTION_PROOF_SCHEMA:
        raise CBCompileContractError("CB compile execution proof schema differs")
    if body["strict_setting"] != {CB_COMPILE_FAIL_CLOSED_ENV: "1"}:
        raise CBCompileContractError("CB compile strict setting differs")
    if body["policy"] != _POLICY:
        raise CBCompileContractError("CB compile policy is not strict fullgraph Inductor")
    raw_helpers = body["helpers"]
    raw_totals = body["totals"]
    if not isinstance(raw_helpers, Mapping) or not isinstance(raw_totals, Mapping):
        raise CBCompileContractError("CB compile proof counters are malformed")
    if set(raw_totals) != set(_COUNTER_FIELDS):
        raise CBCompileContractError("CB compile proof total counters differ")
    prefixes = tuple(str(item) for item in (allowed_helper_prefixes or ()))
    helpers: dict[str, dict[str, int]] = {}
    for raw_name, raw_row in sorted(raw_helpers.items()):
        name = str(raw_name)
        if name not in CB_COMPILE_HELPERS:
            raise CBCompileContractError(
                f"CB compile proof names unknown helper {name!r}"
            )
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            raise CBCompileContractError(
                f"CB compile helper {name!r} is outside the allowed route"
            )
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(_COUNTER_FIELDS):
            raise CBCompileContractError(
                f"CB compile helper {name!r} counter shape differs"
            )
        row = {
            field_name: _strict_nonnegative_int(
                raw_row[field_name], where=f"CB compile {name}.{field_name}"
            )
            for field_name in _COUNTER_FIELDS
        }
        if (
            row["compiled_dispatches"] != row["attempted_calls"]
            or row["compile_failures"]
            or row["runtime_failures"]
            or row["eager_fallbacks"]
        ):
            raise CBCompileContractError(
                f"CB compile helper {name!r} did not complete every call "
                "through a strict compiled graph"
            )
        if require_cuda_calls and row["cuda_calls"] != row["attempted_calls"]:
            raise CBCompileContractError(
                f"CB compile helper {name!r} did not execute exclusively on CUDA"
            )
        helpers[name] = row
    totals = {
        field_name: _strict_nonnegative_int(
            raw_totals[field_name], where=f"CB compile totals.{field_name}"
        )
        for field_name in _COUNTER_FIELDS
    }
    expected_totals = {
        field_name: sum(row[field_name] for row in helpers.values())
        for field_name in _COUNTER_FIELDS
    }
    if totals != expected_totals:
        raise CBCompileContractError("CB compile proof totals do not reconcile")
    if require_live_calls and totals["attempted_calls"] < 1:
        raise CBCompileContractError(
            "CB compile proof contains no live strict compiled calls"
        )
    return {
        **body,
        "helpers": helpers,
        "totals": totals,
        "proof_sha256": str(digest),
    }


def refuse_cb_compile_fallback(helper: str, *, reason: str) -> None:
    """Record and reject a route that would otherwise execute eagerly."""
    _record(helper, "attempted_calls")
    _record(helper, "eager_fallbacks")
    raise CBCompileContractError(
        f"strict CB compile helper {helper!r} refuses eager fallback: {reason}"
    )


def _inductor_backend(graph_module, example_inputs):
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend("inductor")(graph_module, example_inputs)


class _StrictCompiledCallable:
    """Callable guard proving each return crossed the strict backend."""

    def __init__(self, fn: Callable[..., Any], *, helper: str, dynamic: bool):
        if helper not in CB_COMPILE_HELPERS:
            raise CBCompileContractError(f"unknown CB compile helper {helper!r}")
        if dynamic is not True:
            raise CBCompileContractError("strict CB compile requires dynamic=True")
        self.helper = helper
        self._thread_state = threading.local()
        import torch

        if bool(getattr(torch._dynamo.config, "suppress_errors", False)):
            _record(helper, "compile_failures")
            raise CBCompileContractError(
                "strict CB compile requires torch._dynamo.config.suppress_errors=False"
            )
        try:
            self._compiled = torch.compile(
                fn,
                backend=self._backend,
                dynamic=True,
                fullgraph=True,
            )
        except Exception as exc:
            _record(helper, "compile_failures")
            raise CBCompileContractError(
                f"strict CB compile creation failed for {helper!r}"
            ) from exc

    def _backend(self, graph_module, example_inputs):
        try:
            compiled = _inductor_backend(graph_module, example_inputs)
        except Exception:
            _record(self.helper, "compile_failures")
            raise
        _record(self.helper, "graph_compiles")

        def execute(*args, **kwargs):
            result = compiled(*args, **kwargs)
            self._thread_state.dispatched = True
            _record(self.helper, "compiled_dispatches")
            return result

        return execute

    def __call__(self, *args, **kwargs):
        import torch

        _record(self.helper, "attempted_calls")
        tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        if tensor_args and all(bool(arg.is_cuda) for arg in tensor_args):
            _record(self.helper, "cuda_calls")
        if bool(getattr(torch._dynamo.config, "suppress_errors", False)):
            _record(self.helper, "compile_failures")
            raise CBCompileContractError(
                "strict CB compile refuses suppress_errors=True at execution"
            )
        previous = getattr(self._thread_state, "dispatched", None)
        self._thread_state.dispatched = False
        try:
            try:
                result = self._compiled(*args, **kwargs)
            except Exception as exc:
                _record(self.helper, "runtime_failures")
                if isinstance(exc, CBCompileContractError):
                    raise
                raise CBCompileContractError(
                    f"strict compiled CB helper {self.helper!r} failed at runtime"
                ) from exc
            dispatched = bool(self._thread_state.dispatched)
        finally:
            if previous is None:
                try:
                    del self._thread_state.dispatched
                except AttributeError:
                    pass
            else:
                self._thread_state.dispatched = previous
        if not dispatched:
            _record(self.helper, "eager_fallbacks")
            raise CBCompileContractError(
                f"strict compiled CB helper {self.helper!r} returned without "
                "a compiled backend dispatch"
            )
        return result


def compile_cb_callable(
    fn: Callable[..., Any],
    *,
    helper: str,
    dynamic: bool = True,
):
    """Compile ``fn`` under the shared strict no-fallback contract."""
    if not cb_compile_fail_closed():
        raise CBCompileContractError(
            f"{CB_COMPILE_FAIL_CLOSED_ENV}=1 is required for a strict callable"
        )
    return _StrictCompiledCallable(fn, helper=helper, dynamic=dynamic)
