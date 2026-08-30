"""Optional Dagster orchestration for immutable PrismaBuild actions.

Dagster is deliberately not imported at module import time.  The deterministic
graph contract and executor in this module work with the dependency-free
PrismaBuild core; :func:`build_dagster_definitions` imports Dagster only when a
deployment explicitly asks for native definitions.

Neither a Dagster run nor an asset materialization is result authority.  Every
cache hit, dependency, and successful SLURM resolution is accepted only after
``PrismaBuildCAS.lookup`` has revalidated the exact action receipt, producer
scope, and payload bytes.  Graph edges carry action keys and content digests,
never mutable filesystem paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Protocol

from . import prismabuild as pb
from . import prismabuild_slurm as ps


DAGSTER_ACTION_SPEC_SCHEMA_V1 = "prismaquant.prismabuild.dagster_action.v1"

_ACTION_SPEC_KEYS = frozenset(
    {
        "schema",
        "checkout_root",
        "resources",
        "placement",
        "dependencies",
        "retry",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "cpus",
        "memory_mib",
        "gpus",
        "constraint",
        "partition",
        "account",
        "qos",
        "time_limit",
    }
)
_PLACEMENT_KEYS = frozenset({"worker_id", "platform_key", "host_class"})
_DEPENDENCY_KEYS = frozenset(
    {"upstream_action_key", "input_id", "result_sha256", "result_bytes"}
)
_RETRY_KEYS = frozenset({"max_requeues", "poll_interval_seconds", "max_polls"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INPUT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,255}\Z")
_DEFINITION_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


class DagsterIntegrationError(pb.PrismaBuildError):
    """Base class for the optional Dagster integration."""


class DagsterUnavailableError(DagsterIntegrationError):
    """Dagster definitions were requested but the optional package is absent."""


class DagsterGraphError(DagsterIntegrationError, ValueError):
    """The explicit action DAG is malformed or not content-bound."""


class DagsterActionError(DagsterIntegrationError):
    """An orchestrated action did not end in a verified CAS result."""


def _exact_mapping(
    value: object, *, keys: frozenset[str], where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DagsterGraphError(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        raise DagsterGraphError(f"{where} keys must be strings")
    actual = set(value)
    if actual != set(keys):
        raise DagsterGraphError(
            f"{where} fields differ: missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )
    return value


def _text(value: object, *, where: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise DagsterGraphError(f"{where} has an invalid value")
    return value


def _nonnegative_integer(value: object, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise DagsterGraphError(f"{where} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise DagsterGraphError(f"{where} must be a positive integer")
    return value


def _positive_finite(value: object, *, where: str) -> float:
    if type(value) not in {int, float}:
        raise DagsterGraphError(f"{where} must be a positive finite number")
    normalized = float(value)
    if normalized <= 0 or not math.isfinite(normalized):
        raise DagsterGraphError(f"{where} must be a positive finite number")
    return normalized


def _absolute_path(value: object, *, where: str, root_ok: bool = False) -> Path:
    if type(value) is not str:
        raise DagsterGraphError(f"{where} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute() or (not root_ok and path == Path("/")):
        raise DagsterGraphError(f"{where} must be a non-root absolute path")
    return path


@dataclass(frozen=True)
class CASDependency:
    """One immutable upstream-result binding in the action graph."""

    upstream_action_key: str
    input_id: str
    result_sha256: str
    result_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "upstream_action_key",
            _text(
                self.upstream_action_key,
                where="dependency.upstream_action_key",
                pattern=_SHA256_RE,
            ),
        )
        object.__setattr__(
            self,
            "input_id",
            _text(self.input_id, where="dependency.input_id", pattern=_INPUT_ID_RE),
        )
        object.__setattr__(
            self,
            "result_sha256",
            _text(
                self.result_sha256,
                where="dependency.result_sha256",
                pattern=_SHA256_RE,
            ),
        )
        object.__setattr__(
            self,
            "result_bytes",
            _nonnegative_integer(
                self.result_bytes, where="dependency.result_bytes"
            ),
        )

    @classmethod
    def from_config(cls, value: object, *, where: str) -> "CASDependency":
        raw = _exact_mapping(value, keys=_DEPENDENCY_KEYS, where=where)
        return cls(
            upstream_action_key=raw["upstream_action_key"],  # type: ignore[arg-type]
            input_id=raw["input_id"],  # type: ignore[arg-type]
            result_sha256=raw["result_sha256"],  # type: ignore[arg-type]
            result_bytes=raw["result_bytes"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "upstream_action_key": self.upstream_action_key,
            "input_id": self.input_id,
            "result_sha256": self.result_sha256,
            "result_bytes": self.result_bytes,
        }


class ActionSpec:
    """A sealed action plus explicit SLURM placement and CAS dependencies.

    The action is retained as canonical JSON bytes so callers cannot mutate a
    graph's action identity after validation.
    """

    __slots__ = (
        "_action_json",
        "checkout_root",
        "resources",
        "placement",
        "dependencies",
        "max_requeues",
        "poll_interval_seconds",
        "max_polls",
    )

    def __init__(
        self,
        *,
        action: object,
        checkout_root: str | Path,
        resources: ps.SlurmResources,
        placement: ps.SlurmPlacement,
        dependencies: Sequence[CASDependency] = (),
        max_requeues: int = 0,
        poll_interval_seconds: float = 5.0,
        max_polls: int = 17280,
    ):
        normalized = pb.validate_action(action)
        self._action_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        path = Path(checkout_root)
        if not path.is_absolute() or path == Path("/"):
            raise DagsterGraphError("checkout_root must be a non-root absolute path")
        self.checkout_root = path
        if not isinstance(resources, ps.SlurmResources):
            raise DagsterGraphError("resources must be SlurmResources")
        if not isinstance(placement, ps.SlurmPlacement):
            raise DagsterGraphError("placement must be SlurmPlacement")
        self.resources = resources
        self.placement = placement
        deps = tuple(dependencies)
        if any(not isinstance(dep, CASDependency) for dep in deps):
            raise DagsterGraphError("dependencies must contain CASDependency values")
        if len({dep.upstream_action_key for dep in deps}) != len(deps):
            raise DagsterGraphError("dependencies must have unique upstream action keys")
        if len({dep.input_id for dep in deps}) != len(deps):
            raise DagsterGraphError("dependencies must bind unique downstream input ids")
        self.dependencies = tuple(sorted(deps, key=lambda dep: dep.upstream_action_key))
        self.max_requeues = _nonnegative_integer(
            max_requeues, where="retry.max_requeues"
        )
        self.poll_interval_seconds = _positive_finite(
            poll_interval_seconds, where="retry.poll_interval_seconds"
        )
        self.max_polls = _positive_integer(max_polls, where="retry.max_polls")
        pb.validate_worker_scope(
            normalized,
            platform_key=placement.platform_key,
            host_class=placement.host_class,
        )

    @property
    def action(self) -> dict[str, object]:
        value = json.loads(self._action_json)
        assert isinstance(value, dict)
        return value

    @property
    def action_key(self) -> str:
        value = self.action["action_key"]
        assert isinstance(value, str)
        return value

    @classmethod
    def from_config(cls, action: object, value: object) -> "ActionSpec":
        raw = _exact_mapping(value, keys=_ACTION_SPEC_KEYS, where="action spec")
        if raw["schema"] != DAGSTER_ACTION_SPEC_SCHEMA_V1:
            raise DagsterGraphError(
                f"action spec schema must be {DAGSTER_ACTION_SPEC_SCHEMA_V1!r}"
            )
        resources_raw = _exact_mapping(
            raw["resources"], keys=_RESOURCE_KEYS, where="action spec.resources"
        )
        placement_raw = _exact_mapping(
            raw["placement"], keys=_PLACEMENT_KEYS, where="action spec.placement"
        )
        retry_raw = _exact_mapping(
            raw["retry"], keys=_RETRY_KEYS, where="action spec.retry"
        )
        dependencies_raw = raw["dependencies"]
        if type(dependencies_raw) is not list:
            raise DagsterGraphError("action spec.dependencies must be an array")
        return cls(
            action=action,
            checkout_root=_absolute_path(
                raw["checkout_root"], where="action spec.checkout_root"
            ),
            resources=ps.SlurmResources(**resources_raw),  # type: ignore[arg-type]
            placement=ps.SlurmPlacement(**placement_raw),  # type: ignore[arg-type]
            dependencies=[
                CASDependency.from_config(
                    dependency, where=f"action spec.dependencies[{index}]"
                )
                for index, dependency in enumerate(dependencies_raw)
            ],
            max_requeues=retry_raw["max_requeues"],  # type: ignore[arg-type]
            poll_interval_seconds=retry_raw[  # type: ignore[arg-type]
                "poll_interval_seconds"
            ],
            max_polls=retry_raw["max_polls"],  # type: ignore[arg-type]
        )

    def as_config(self) -> dict[str, object]:
        return {
            "schema": DAGSTER_ACTION_SPEC_SCHEMA_V1,
            "checkout_root": str(self.checkout_root),
            "resources": {
                "cpus": self.resources.cpus,
                "memory_mib": self.resources.memory_mib,
                "gpus": self.resources.gpus,
                "constraint": self.resources.constraint,
                "partition": self.resources.partition,
                "account": self.resources.account,
                "qos": self.resources.qos,
                "time_limit": self.resources.time_limit,
            },
            "placement": {
                "worker_id": self.placement.worker_id,
                "platform_key": self.placement.platform_key,
                "host_class": self.placement.host_class,
            },
            "dependencies": [dependency.as_dict() for dependency in self.dependencies],
            "retry": {
                "max_requeues": self.max_requeues,
                "poll_interval_seconds": self.poll_interval_seconds,
                "max_polls": self.max_polls,
            },
        }


class ActionGraph:
    """Validated action graph with deterministic key-ordered topological order."""

    def __init__(self, specs: Sequence[ActionSpec], *, name: str = "prismabuild"):
        if _DEFINITION_NAME_RE.fullmatch(name) is None:
            raise DagsterGraphError("graph name is not a valid Dagster definition name")
        by_key: dict[str, ActionSpec] = {}
        for spec in specs:
            if not isinstance(spec, ActionSpec):
                raise DagsterGraphError("graph entries must be ActionSpec values")
            if spec.action_key in by_key:
                raise DagsterGraphError(f"duplicate action key: {spec.action_key}")
            by_key[spec.action_key] = spec
        if not by_key:
            raise DagsterGraphError("action graph must contain at least one action")
        for key, spec in by_key.items():
            inputs = {
                str(entry["id"]): entry
                for entry in spec.action["inputs"]  # type: ignore[union-attr]
            }
            for dependency in spec.dependencies:
                if dependency.upstream_action_key not in by_key:
                    raise DagsterGraphError(
                        f"action {key} depends on an unknown action key "
                        f"{dependency.upstream_action_key}"
                    )
                expected_input = {
                    "id": dependency.input_id,
                    "sha256": dependency.result_sha256,
                    "bytes": dependency.result_bytes,
                }
                if inputs.get(dependency.input_id) != expected_input:
                    raise DagsterGraphError(
                        f"action {key} dependency {dependency.upstream_action_key} "
                        "is not bound exactly in action.inputs"
                    )

        remaining = {key: len(spec.dependencies) for key, spec in by_key.items()}
        downstream: dict[str, list[str]] = {key: [] for key in by_key}
        for key, spec in by_key.items():
            for dependency in spec.dependencies:
                downstream[dependency.upstream_action_key].append(key)
        ready = sorted(key for key, count in remaining.items() if count == 0)
        ordered: list[str] = []
        while ready:
            key = ready.pop(0)
            ordered.append(key)
            for child in sorted(downstream[key]):
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(ordered) != len(by_key):
            raise DagsterGraphError("action graph contains a dependency cycle")
        self.name = name
        self._by_key = by_key
        self._ordered_keys = tuple(ordered)

    @property
    def ordered_specs(self) -> tuple[ActionSpec, ...]:
        return tuple(self._by_key[key] for key in self._ordered_keys)

    def spec(self, action_key: str) -> ActionSpec:
        try:
            return self._by_key[action_key]
        except KeyError as exc:
            raise DagsterGraphError(f"unknown action key: {action_key}") from exc


@dataclass(frozen=True)
class BoundActionResult:
    """Path-free value passed along Dagster edges after CAS verification."""

    action_key: str
    result_sha256: str
    result_bytes: int
    receipt_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "result_sha256": self.result_sha256,
            "result_bytes": self.result_bytes,
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_value(cls, value: object, *, where: str) -> "BoundActionResult":
        raw = _exact_mapping(
            value,
            keys=frozenset(
                {"action_key", "result_sha256", "result_bytes", "receipt_sha256"}
            ),
            where=where,
        )
        return cls(
            action_key=_text(
                raw["action_key"], where=f"{where}.action_key", pattern=_SHA256_RE
            ),
            result_sha256=_text(
                raw["result_sha256"],
                where=f"{where}.result_sha256",
                pattern=_SHA256_RE,
            ),
            result_bytes=_nonnegative_integer(
                raw["result_bytes"], where=f"{where}.result_bytes"
            ),
            receipt_sha256=_text(
                raw["receipt_sha256"],
                where=f"{where}.receipt_sha256",
                pattern=_SHA256_RE,
            ),
        )


def _bound_result(
    receipt: Mapping[str, object], *, expected_action_key: str
) -> BoundActionResult:
    result = receipt.get("result")
    if not isinstance(result, Mapping):
        raise DagsterActionError("verified CAS receipt has no result object")
    value = BoundActionResult(
        action_key=_text(
            receipt.get("action_key"),
            where="CAS receipt.action_key",
            pattern=_SHA256_RE,
        ),
        result_sha256=_text(
            result.get("sha256"),
            where="CAS receipt.result.sha256",
            pattern=_SHA256_RE,
        ),
        result_bytes=_nonnegative_integer(
            result.get("bytes"), where="CAS receipt.result.bytes"
        ),
        receipt_sha256=_text(
            receipt.get("receipt_sha256"),
            where="CAS receipt.receipt_sha256",
            pattern=_SHA256_RE,
        ),
    )
    if value.action_key != expected_action_key:
        raise DagsterActionError("CAS receipt action key differs from graph action")
    return value


@dataclass(frozen=True)
class DagsterResourceConfig:
    """Configuration for the optional Dagster-to-SLURM resource."""

    cas_root: Path
    log_root: Path
    worker_script: Path
    sbatch: Path = Path("/usr/bin/sbatch")
    squeue: Path = Path("/usr/bin/squeue")
    sacct: Path = Path("/usr/bin/sacct")
    scancel: Path = Path("/usr/bin/scancel")
    scontrol: Path = Path("/usr/bin/scontrol")
    command_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "cas_root",
            "log_root",
            "worker_script",
            "sbatch",
            "squeue",
            "sacct",
            "scancel",
            "scontrol",
        ):
            value = Path(getattr(self, name))
            if not value.is_absolute() or (
                name in {"cas_root", "log_root"} and value == Path("/")
            ):
                raise DagsterGraphError(f"{name} must be an absolute path")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "command_timeout_seconds",
            _positive_finite(
                self.command_timeout_seconds, where="command_timeout_seconds"
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DagsterResourceConfig":
        required = {"cas_root", "log_root", "worker_script"}
        optional = {
            "sbatch",
            "squeue",
            "sacct",
            "scancel",
            "scontrol",
            "command_timeout_seconds",
        }
        actual = set(value)
        if not required <= actual or not actual <= required | optional:
            raise DagsterGraphError(
                "Dagster resource config fields differ: "
                f"missing={sorted(required - actual)}, "
                f"extra={sorted(actual - (required | optional))}"
            )
        kwargs = dict(value)
        return cls(**kwargs)  # type: ignore[arg-type]

    def make_adapter(self) -> ps.SlurmAdapter:
        return ps.SlurmAdapter(
            cas_root=self.cas_root,
            log_root=self.log_root,
            worker_script=self.worker_script,
            sbatch=self.sbatch,
            squeue=self.squeue,
            sacct=self.sacct,
            scancel=self.scancel,
            scontrol=self.scontrol,
            command_timeout_seconds=self.command_timeout_seconds,
        )


class _Adapter(Protocol):
    def submit(
        self,
        action: object,
        *,
        checkout_root: str | Path,
        resources: ps.SlurmResources,
        placement: ps.SlurmPlacement,
        recompute: bool = False,
    ) -> ps.SlurmSubmission: ...

    def resolve(
        self, action: object, job_id: ps.SlurmJobId | str
    ) -> ps.SlurmResolution: ...

    def requeue(self, action: object, job_id: ps.SlurmJobId | str) -> bool: ...


class DagsterActionRunner:
    """Execute graph nodes while treating verified CAS state as sole truth."""

    def __init__(
        self,
        *,
        cas_root: str | Path,
        adapter: _Adapter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cas = pb.PrismaBuildCAS(cas_root)
        self.adapter = adapter
        self.sleep = sleep

    def _verified_result(self, spec: ActionSpec) -> BoundActionResult | None:
        receipt = self.cas.lookup(spec.action)
        if receipt is None:
            return None
        return _bound_result(receipt, expected_action_key=spec.action_key)

    def _verify_dependencies(
        self,
        graph: ActionGraph,
        spec: ActionSpec,
        upstream_values: Mapping[str, object] | None,
    ) -> None:
        expected_keys = {dependency.upstream_action_key for dependency in spec.dependencies}
        if upstream_values is not None and set(upstream_values) != expected_keys:
            raise DagsterActionError(
                f"action {spec.action_key} upstream values differ: "
                f"missing={sorted(expected_keys - set(upstream_values))}, "
                f"extra={sorted(set(upstream_values) - expected_keys)}"
            )
        for dependency in spec.dependencies:
            upstream = graph.spec(dependency.upstream_action_key)
            verified = self._verified_result(upstream)
            if verified is None:
                raise DagsterActionError(
                    f"upstream action {dependency.upstream_action_key} has no CAS receipt"
                )
            expected = BoundActionResult(
                action_key=dependency.upstream_action_key,
                result_sha256=dependency.result_sha256,
                result_bytes=dependency.result_bytes,
                receipt_sha256=verified.receipt_sha256,
            )
            observed = (
                BoundActionResult.from_value(
                    upstream_values[dependency.upstream_action_key],
                    where=f"upstream value {dependency.upstream_action_key}",
                )
                if upstream_values is not None
                else expected
            )
            if verified != expected or observed != expected:
                raise DagsterActionError(
                    f"upstream action {dependency.upstream_action_key} does not match "
                    "the dependency-bound CAS result"
                )

    def execute(
        self,
        graph: ActionGraph,
        action_key: str,
        *,
        upstream_values: Mapping[str, object] | None = None,
    ) -> BoundActionResult:
        spec = graph.spec(action_key)
        self._verify_dependencies(graph, spec, upstream_values)
        cached = self._verified_result(spec)
        if cached is not None:
            return cached

        submission = self.adapter.submit(
            spec.action,
            checkout_root=spec.checkout_root,
            resources=spec.resources,
            placement=spec.placement,
        )
        if submission.action_key != spec.action_key:
            raise DagsterActionError("SLURM submission returned a different action key")
        if submission.status == "cache_hit":
            cached = self._verified_result(spec)
            if cached is None:
                raise DagsterActionError(
                    "SLURM adapter reported cache_hit without a verified CAS receipt"
                )
            return cached
        if submission.status != "submitted" or submission.job_id is None:
            raise DagsterActionError("SLURM submission did not return a job id")

        polls = 0
        requeues = 0
        while True:
            resolution = self.adapter.resolve(spec.action, submission.job_id)
            if resolution.action_key != spec.action_key:
                raise DagsterActionError(
                    "SLURM resolution returned a different action key"
                )
            if resolution.status == "succeeded":
                # Do not trust the scheduler adapter's payload or receipt object:
                # read and verify the canonical CAS state independently again.
                result = self._verified_result(spec)
                if result is None:
                    raise DagsterActionError(
                        "orchestrator reported success without a verified CAS receipt"
                    )
                return result
            if resolution.status in {"pending", "running"}:
                polls += 1
                if polls >= spec.max_polls:
                    raise DagsterActionError(
                        f"action {spec.action_key} exceeded its explicit poll budget"
                    )
                self.sleep(spec.poll_interval_seconds)
                continue
            if requeues < spec.max_requeues:
                # Requeue the same SLURM allocation with the exact sealed action.
                # Dagster-level retries are intentionally disabled by the native
                # definition factory because a new submission could duplicate a
                # still-live allocation after an orchestrator process failure.
                requeued = self.adapter.requeue(spec.action, submission.job_id)
                if not requeued:
                    result = self._verified_result(spec)
                    if result is None:
                        raise DagsterActionError(
                            "SLURM declined requeue without a verified CAS receipt"
                        )
                    return result
                requeues += 1
                polls = 0
                continue
            raise DagsterActionError(
                f"action {spec.action_key} ended {resolution.status}: "
                f"{resolution.reason}"
            )


@dataclass(frozen=True)
class DagsterNodePlan:
    """Dependency-free description used to build native Dagster assets."""

    action_key: str
    asset_path: tuple[str, str]
    dependency_keys: tuple[str, ...]


def definition_plan(graph: ActionGraph) -> tuple[DagsterNodePlan, ...]:
    """Return deterministic native-definition inputs without importing Dagster."""

    return tuple(
        DagsterNodePlan(
            action_key=spec.action_key,
            asset_path=("prismabuild", spec.action_key),
            dependency_keys=tuple(
                dependency.upstream_action_key for dependency in spec.dependencies
            ),
        )
        for spec in graph.ordered_specs
    )


def _import_dagster() -> Any:
    try:
        return importlib.import_module("dagster")
    except ImportError as exc:
        raise DagsterUnavailableError(
            "Dagster is optional; install PrismaQuant's 'prismabuild' extra "
            "before requesting native definitions"
        ) from exc


def build_dagster_definitions(
    graph: ActionGraph,
    *,
    dagster_module: Any | None = None,
) -> Any:
    """Build Dagster assets and one asset job from an explicit action graph.

    The returned definitions require a ``prismabuild`` resource configured
    with ``cas_root``, ``log_root``, and ``worker_script``.  Every materialized
    asset is emitted only after :class:`DagsterActionRunner` independently
    verifies the CAS.  Dagster retry count is zero; bounded retries use
    ``SlurmAdapter.requeue`` on the original action/job identity.
    """

    dg = dagster_module if dagster_module is not None else _import_dagster()

    @dg.resource(
        config_schema={
            "cas_root": str,
            "log_root": str,
            "worker_script": str,
            "sbatch": dg.Field(str, default_value="/usr/bin/sbatch"),
            "squeue": dg.Field(str, default_value="/usr/bin/squeue"),
            "sacct": dg.Field(str, default_value="/usr/bin/sacct"),
            "scancel": dg.Field(str, default_value="/usr/bin/scancel"),
            "scontrol": dg.Field(str, default_value="/usr/bin/scontrol"),
            "command_timeout_seconds": dg.Field(float, default_value=30.0),
        }
    )
    def prismabuild_resource(init_context) -> DagsterActionRunner:
        config = DagsterResourceConfig.from_mapping(init_context.resource_config)
        return DagsterActionRunner(
            cas_root=config.cas_root,
            adapter=config.make_adapter(),
        )

    assets: list[Any] = []
    for node in definition_plan(graph):
        spec = graph.spec(node.action_key)
        asset_key = dg.AssetKey(list(node.asset_path))
        dependencies = [
            dg.AssetDep(dg.AssetKey(["prismabuild", key]))
            for key in node.dependency_keys
        ]

        def make_execute_asset(bound_spec: ActionSpec) -> Callable[..., object]:
            def execute_asset(context):
                runner = context.resources.prismabuild
                # Asset success is downstream of a verified receipt.  The
                # returned mapping is informational metadata, not a second
                # result store and not an input path for dependent actions.
                return runner.execute(graph, bound_spec.action_key).as_dict()

            execute_asset.__name__ = f"prismabuild_{bound_spec.action_key}"
            return execute_asset

        asset = dg.asset(
            key=asset_key,
            deps=dependencies,
            required_resource_keys={"prismabuild"},
            code_version=spec.action_key,
            retry_policy=dg.RetryPolicy(max_retries=0),
        )(make_execute_asset(spec))
        assets.append(asset)

    selection = dg.AssetSelection.assets(
        *[dg.AssetKey(list(node.asset_path)) for node in definition_plan(graph)]
    )
    job = dg.define_asset_job(f"{graph.name}_job", selection=selection)
    return dg.Definitions(
        assets=assets,
        jobs=[job],
        resources={"prismabuild": prismabuild_resource},
    )


__all__ = [
    "ActionGraph",
    "ActionSpec",
    "BoundActionResult",
    "CASDependency",
    "DAGSTER_ACTION_SPEC_SCHEMA_V1",
    "DagsterActionError",
    "DagsterActionRunner",
    "DagsterGraphError",
    "DagsterIntegrationError",
    "DagsterNodePlan",
    "DagsterResourceConfig",
    "DagsterUnavailableError",
    "build_dagster_definitions",
    "definition_plan",
]
