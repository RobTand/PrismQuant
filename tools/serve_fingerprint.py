#!/usr/bin/env python3
"""Serve fingerprint — mechanize the §7.4 reproducibility contract.

R15 (`docs/audits/architecture_re-vet_2026-07-30.md`). KL is bit-identical
*within* one docker session and drifts 4-8x *across* sessions: loading any CUDA
extension into the serving process shifts allocator addresses, activations get
different pointer alignments, and alignment-sensitive cuBLAS/CUTLASS heuristics
pick different kernels. On the 27B this reads as two bit-reproducible states,
conf-KL 0.01134 vs 0.01328 (+-17%), keyed purely on whether the gridbook `.so`
was resident during the dump. The rule ("A/B arms must have identical extension
residency; deltas under ~+-20% across differing stacks are not evidence") was
prose with nothing enforcing it.

This module makes the stack an object:

* `collect_manifest()` reads the **server's** address space
  (`/proc/<pid>/maps`) - it must be server-side, because the measuring client
  cannot see the server's residency, which is exactly why the drift stayed
  invisible for so long.
* `fingerprint()` = sha256 of the canonical JSON of the manifest **minus argv
  paths**, so two artifacts served the same way share a fingerprint (an A/B
  needs that) while a changed image / extension set / eager flag does not.

CLI (run inside the serving container, after READY):

    python3 -P /repo/tools/prismaquant_source_bootstrap.py \
        run-tool serve-fingerprint write \
        --out /dqruns/<run>/exported/serve_manifest.json --image vllm-node:latest

Stdlib only by construction: it must not import torch or vllm into the serving
container (an extra CUDA context on a 121 GiB unified pool is how boxes die),
so versions come from `importlib.metadata` and the GPU from NVML via
`nvidia-smi`.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

MANIFEST_SCHEMA = "prismaquant.serve_manifest/1"
MANIFEST_FILENAME = "serve_manifest.json"
GRIDBOOK_DISTRIBUTION_SCHEMA = (
    "prismaquant.installed_gridbook_distribution/2"
)
GRIDBOOK_IMPORT_ORIGIN_SCHEMA = "prismaquant.gridbook_import_origin/1"
GRIDBOOK_REPOSITORY = "https://github.com/RobTand/gridbook.git"
GOLD_PRODUCER_IDENTITY_SCHEMA = "prismaquant.gold_producer_identity/1"
MODELS_ENDPOINT_BINDING_SCHEMA = (
    "prismaquant.server_models_endpoint_binding/1"
)

_GOLD_PRODUCER_COMMON_FILES = (
    "prismaquant/gridbook_assignment.py",
    "prismaquant/gridbook_environment.py",
    "prismaquant/gridbook_runtime/gridbook_runtime_pin.json",
    "tools/dsv4_gridbook_contract.py",
    "tools/dsv4_wikitext_inputs.py",
    "tools/prepare_dsv4_wikitext_inputs.py",
    "tools/prismaquant_source_bootstrap.py",
    "tools/serve_fingerprint.py",
    "tools/spec_decode_guard.py",
)
_GOLD_PRODUCER_TOOL_FILES = {
    "measure_vllm_full_kl": (
        "tools/full_kl_teacher_payload.py",
        "tools/measure_vllm_full_kl.py",
    ),
    "measure_vllm_wikitext_ppl": (
        "tools/full_kl_teacher_payload.py",
        "tools/measure_vllm_wikitext_ppl.py",
    ),
}

_GRIDBOOK_SOURCE_SUFFIXES = frozenset({
    ".cu", ".cuh", ".h", ".hpp", ".json", ".py",
})
_REQUIRED_GRIDBOOK_SOURCE_FILES = frozenset({
    "gridbook/__init__.py",
    "gridbook/cuda_ext.py",
    "gridbook/plugin.py",
    "gridbook/runtime_contract.json",
    "gridbook/source_passthrough.py",
    "gridbook/fp8_source_w8a16.py",
    "gridbook/csrc/cb_gemv.cu",
    "gridbook/csrc/fp8_source_w8a16.cu",
    "gridbook/csrc/mxfp8_dense_gemm.cu",
})


def _gridbook_environment_allowlist() -> tuple[str, ...]:
    """Read the authoritative registry without importing ``prismaquant``.

    Importing a package submodule executes ``prismaquant.__init__``, which in
    turn imports torch.  This helper is intentionally used inside a serving
    container, so inspect the literal ``_var(...)`` declarations with the
    standard-library AST instead.  A changed registry shape fails closed
    rather than silently producing an incomplete process-environment proof.
    """
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_environment.py"
    )
    try:
        tree = ast.parse(
            registry_path.read_text(encoding="utf-8"),
            filename=str(registry_path),
        )
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise RuntimeError(
            f"cannot read Gridbook environment registry: {registry_path}"
        ) from exc

    assignments = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "GRIDBOOK_ENVIRONMENT_REGISTRY"
            for target in node.targets
        ):
            assignments.append(node.value)
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Tuple):
        raise RuntimeError(
            "Gridbook environment registry must be one literal tuple"
        )

    names: list[str] = []
    for entry in assignments[0].elts:
        if (
            not isinstance(entry, ast.Call)
            or not isinstance(entry.func, ast.Name)
            or entry.func.id != "_var"
            or not entry.args
            or not isinstance(entry.args[0], ast.Constant)
            or not isinstance(entry.args[0].value, str)
        ):
            raise RuntimeError(
                "Gridbook environment registry contains a non-literal name"
            )
        names.append(entry.args[0].value)
    if not names or len(names) != len(set(names)):
        raise RuntimeError(
            "Gridbook environment registry names must be non-empty and unique"
        )
    return tuple(sorted(names))


# Performance-sensitive variables that the release attestation is permitted to
# carry.  Values are read from the server processes' /proc entries, never from
# the short-lived ``docker exec`` process that writes the manifest.
SERVER_ENV_ALLOWLIST = (
    "PQ_GRIDBOOK_RUNTIME_COMMIT",
    "PQ_GRIDBOOK_RUNTIME_VERSION",
    # The serving process must not carry an explicit Python module-search
    # override.  ``server_environment_snapshot`` records only set values, so
    # validators prove affirmative absence by requiring this allowlisted name
    # to be missing from the exact process-environment projection.  The
    # short-lived fingerprint writer is bootstrapped from the verified /repo
    # snapshot with safe-path mode and no PYTHONPATH; it reads the independently
    # running server PIDs from /proc and is not one of them.
    "PYTHONPATH",
    # Python's safe-path mode prevents the empty-string/script-directory entry
    # from taking precedence over the exact VCS-installed Gridbook package.
    # The imported module origin is attested separately below.
    "PYTHONSAFEPATH",
    *_gridbook_environment_allowlist(),
)

#: The extensions whose residency moves the numbers (§7.4).
EXTENSION_PATTERN = re.compile(
    r"gridbook|prismaquant|pq_(?:cb|mxfp8|fp8_source)|flashinfer|"
    r"causal_conv1d|fla")

#: Packages whose version pins the numeric stack.
TRACKED_PACKAGES = (
    "vllm", "torch", "flashinfer-python", "gridbook", "prismaquant",
    "causal-conv1d", "flash-linear-attention", "transformers",
)

#: Keys excluded from the fingerprint: they identify the *run*, not the *stack*.
_FINGERPRINT_EXCLUDED = frozenset({
    "created", "launch_argv", "processes", "model", "container", "hostname",
    "serve_fingerprint", "schema", "served_model_name", "written_by",
    # Live PIDs define a session, not a numeric serving stack. Their stable
    # identities remain represented by ``processes``/``serve_session_id``.
    "measurement_parent_pid", "engine_descendant_pids",
})

_IN_PROCESS_OBSERVED_FIELDS = frozenset({
    "measurement_parent_pid",
    "engine_descendant_pids",
})

_PATH_PLACEHOLDER = "<path>"
_ARM_MODEL_PLACEHOLDER = "<arm-model>"


# ---------------------------------------------------------------------------
# Process inspection
# ---------------------------------------------------------------------------
def _read_cmdline(pid: str | int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def _read_process_name(pid: str | int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _process_parent_pid(pid: str | int) -> int | None:
    """Parent PID from ``/proc/<pid>/stat``, whose comm may contain spaces."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except Exception:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2:].split()
    try:
        parent = int(fields[1])  # field 4; fields starts at field 3 (state)
    except (IndexError, ValueError):
        return None
    return parent if parent >= 0 else None


def _read_process_ppid(
    pid: int,
    *,
    proc_root: str | os.PathLike = "/proc",
) -> int | None:
    """Read one kernel-authored ``PPid`` field fail-closed.

    A process may exit while ``/proc`` is being enumerated. Missing,
    malformed, duplicated, or negative fields are therefore treated as
    unreadable, never as proof that the process belongs to a measurement.
    """
    try:
        lines = (Path(proc_root) / str(pid) / "status").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except (OSError, UnicodeError):
        return None
    values: list[str] = []
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key == "PPid":
            values.append(value.strip())
    if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
        return None
    try:
        ppid = int(values[0])
    except ValueError:
        return None
    return ppid if ppid >= 0 else None


def _proc_pids(*, proc_root: str | os.PathLike = "/proc") -> list[int]:
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    return sorted(int(name) for name in names if name.isdigit())


def descendant_process_pids(
    parent_pid: int,
    *,
    proc_root: str | os.PathLike = "/proc",
) -> list[int]:
    """Transitive descendants proven by the live kernel PPid graph.

    No argv search participates in membership. An unrelated vLLM server in
    the same PID namespace can have a convincing process title, but cannot be
    admitted unless it descends from the measurement process.
    """
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, int):
        raise TypeError("parent_pid must be an integer PID")
    if parent_pid <= 0:
        raise ValueError("parent_pid must be positive")

    children: dict[int, list[int]] = {}
    for pid in _proc_pids(proc_root=proc_root):
        if pid == parent_pid:
            continue
        ppid = _read_process_ppid(pid, proc_root=proc_root)
        if ppid is None or ppid == pid:
            continue
        children.setdefault(ppid, []).append(pid)

    descendants: set[int] = set()
    pending = list(children.get(parent_pid, ()))
    while pending:
        pid = pending.pop()
        if pid == parent_pid or pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, ()))
    return sorted(descendants)


def argv_identifies_vllm_engine(argv: Sequence[str]) -> bool:
    """Whether argv explicitly identifies EngineCore/a vLLM engine.

    The common vLLM v1 title is ``VLLM::EngineCore``. Module-style launchers
    such as ``python -m vllm.v1.engine.core`` are accepted too, while a plain
    ``vllm serve`` front end is deliberately not an engine witness.
    """
    values = [str(value) for value in argv if str(value)]
    if not values:
        return False
    joined = " ".join(values)
    if re.search(
        r"(?:^|[^a-z0-9])engine[\s._:-]*core(?:$|[^a-z0-9])",
        joined,
        flags=re.IGNORECASE,
    ):
        return True
    has_vllm = re.search(
        r"(?:^|[^a-z0-9])vllm(?:$|[^a-z0-9])",
        joined,
        flags=re.IGNORECASE,
    ) is not None
    has_engine = re.search(
        r"(?:^|[^a-z0-9])engine(?:$|[^a-z0-9])",
        joined,
        flags=re.IGNORECASE,
    ) is not None
    return has_vllm and has_engine


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_process_environment(pid: str | int) -> dict[str, str] | None:
    if int(pid) == os.getpid():
        # /proc/self/environ is the exec-time snapshot on Linux and does not
        # reflect os.environ mutations used by in-process measurement tools or
        # their tests.  For the process doing the measurement, this mapping is
        # the actual live environment; server-side release snapshots inspect
        # different vLLM PIDs and always use /proc.
        return dict(os.environ)
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except Exception:
        return None
    result: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        if not separator:
            return None
        decoded_name = name.decode("utf-8", "strict")
        if decoded_name in result:
            return None
        result[decoded_name] = value.decode("utf-8", "strict")
    return result


def server_environment_snapshot(
    pids: Sequence[int],
    names: Sequence[str] = SERVER_ENV_ALLOWLIST,
) -> dict[str, Any]:
    """Exact allowlisted environment shared by every inspected server process."""
    allowlist = sorted(set(names))
    rows: list[dict[str, Any]] = []
    unreadable: list[int] = []
    for pid in sorted(set(pids)):
        try:
            environment = _read_process_environment(pid)
        except (UnicodeError, ValueError):
            environment = None
        if environment is None:
            unreadable.append(pid)
            continue
        selected = {
            name: environment[name] for name in allowlist if name in environment
        }
        rows.append(
            {
                "pid": pid,
                "values": selected,
                "sha256": _canonical_sha256(selected),
            }
        )
    distinct = {
        json.dumps(row["values"], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    consensus = dict(rows[0]["values"]) if rows and len(distinct) == 1 else None
    return {
        "schema": "prismaquant.server_process_environment/1",
        "allowlist": allowlist,
        "readable_pids": [row["pid"] for row in rows],
        "unreadable_pids": unreadable,
        "consistent": consensus is not None and not unreadable,
        "values": consensus,
        "processes": rows,
    }


def _process_start_time_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except Exception:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2:].split()
    # /proc/<pid>/stat field 22; ``fields`` starts at field 3.
    try:
        value = int(fields[19])
    except (IndexError, ValueError):
        return None
    return value if value >= 0 else None


def _readlink(path: str) -> str | None:
    try:
        return os.readlink(path)
    except Exception:
        return None


def host_identity() -> dict[str, Any]:
    """Stable host-boot identity, safe to persist without exposing machine-id."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip().lower()
    except Exception:
        boot_id = None
    try:
        machine_id = Path("/etc/machine-id").read_text().strip().lower()
    except Exception:
        machine_id = None
    return {
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "machine_id_sha256": (
            hashlib.sha256(machine_id.encode("ascii")).hexdigest()
            if machine_id
            else None
        ),
        "pid_namespace": _readlink("/proc/self/ns/pid"),
    }


def process_identities(pids: Sequence[int], *, boot_id: str | None) -> list[dict[str, Any]]:
    """PID-reuse-safe identities for the exact inspected process set."""
    rows: list[dict[str, Any]] = []
    for pid in sorted(set(pids)):
        argv = _read_cmdline(pid)
        start_time = _process_start_time_ticks(pid)
        namespace = _readlink(f"/proc/{pid}/ns/pid")
        executable = _readlink(f"/proc/{pid}/exe")
        if not argv or start_time is None or namespace is None or executable is None:
            rows.append(
                {
                    "pid": pid,
                    "argv": argv,
                    "cmdline": " ".join(argv),
                    "start_time_ticks": start_time,
                    "pid_namespace": namespace,
                    "executable": executable,
                    "identity_sha256": None,
                }
            )
            continue
        identity = {
            "pid": pid,
            "start_time_ticks": start_time,
            "pid_namespace": namespace,
            "executable": executable,
            "argv": argv,
        }
        rows.append(
            {
                "pid": pid,
                "argv": argv,
                "cmdline": " ".join(argv),
                "start_time_ticks": start_time,
                "pid_namespace": namespace,
                "executable": executable,
                "identity_sha256": process_identity_sha256(
                    identity, boot_id=boot_id
                ),
            }
        )
    return rows


def process_identity_sha256(
    process: Mapping[str, Any], *, boot_id: str | None
) -> str:
    return _canonical_sha256(
        {
            "boot_id": boot_id,
            "pid": process.get("pid"),
            "start_time_ticks": process.get("start_time_ticks"),
            "pid_namespace": process.get("pid_namespace"),
            "executable": process.get("executable"),
            "argv": process.get("argv"),
        }
    )


def serve_session_fingerprint(manifest: Mapping[str, Any]) -> str:
    processes = manifest.get("processes")
    process_hashes = [
        row.get("identity_sha256")
        for row in processes
        if isinstance(row, Mapping)
    ] if isinstance(processes, list) else []
    return _canonical_sha256(
        {
            "host_identity": manifest.get("host_identity"),
            "gpu_uuid": manifest.get("gpu_uuid"),
            "process_identity_sha256": sorted(process_hashes),
            "listener": manifest.get("listener_binding"),
        }
    )


def _looks_like_vllm_process(pid: int, pattern: str = "vllm") -> bool:
    joined = " ".join(_read_cmdline(pid))
    name = _read_process_name(pid)
    return bool(
        pattern.lower() in joined.lower()
        or "enginecore" in joined.lower()
        or "vllm" in name.lower()
        or "enginecore" in name.lower()
    )


def find_server_pids(pattern: str = "vllm") -> list[int]:
    """Every readable process whose argv looks like the vLLM server or engine.

    Both matter: on vLLM v1 the API front-end and the EngineCore worker are
    different processes, and it is the *engine* that has the kernels resident.
    """
    pids: list[int] = []
    try:
        entries = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    except Exception:
        return []
    for pid in entries:
        if _looks_like_vllm_process(pid, pattern):
            pids.append(pid)
    return pids


def find_in_process_server_pids(root_pid: int | None = None) -> list[int]:
    """Measurement process plus all of its live vLLM/EngineCore descendants.

    vLLM v1 constructs ``LLM`` in the measuring Python process but executes
    kernels in a spawned EngineCore.  A self-only maps/env snapshot therefore
    attests the wrong address space.  Restricting discovery to the current
    process tree avoids accidentally binding an unrelated server elsewhere in
    the same container or host.
    """
    root = os.getpid() if root_pid is None else int(root_pid)
    try:
        entries = sorted(int(value) for value in os.listdir("/proc") if value.isdigit())
    except Exception:
        return [root]
    children: dict[int, list[int]] = {}
    for pid in entries:
        parent = _process_parent_pid(pid)
        if parent is not None:
            children.setdefault(parent, []).append(pid)
    descendants: set[int] = set()
    pending = list(children.get(root, ()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, ()))
    selected = {root}
    selected.update(pid for pid in descendants if _looks_like_vllm_process(pid))
    return sorted(selected)


def _process_socket_inodes(pids: Sequence[int]) -> tuple[dict[str, set[int]], list[int]]:
    owners: dict[str, set[int]] = {}
    unreadable: list[int] = []
    for pid in sorted(set(pids)):
        directory = Path(f"/proc/{pid}/fd")
        try:
            entries = list(directory.iterdir())
        except Exception:
            unreadable.append(pid)
            continue
        for entry in entries:
            target = _readlink(str(entry))
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target or "")
            if match:
                owners.setdefault(match.group(1), set()).add(pid)
    return owners, unreadable


def _decode_proc_address(value: str, family: str) -> str | None:
    try:
        raw = bytes.fromhex(value)
        if family == "ipv4":
            if len(raw) != 4:
                return None
            return str(ipaddress.IPv4Address(raw[::-1]))
        if len(raw) != 16:
            return None
        # Linux exposes each 32-bit word in host byte order in /proc/net/tcp6.
        reordered = b"".join(
            raw[index:index + 4][::-1] for index in range(0, 16, 4)
        )
        return str(ipaddress.IPv6Address(reordered))
    except (ValueError, ipaddress.AddressValueError):
        return None


def process_tcp_listeners(pids: Sequence[int]) -> dict[str, Any]:
    """TCP LISTEN sockets actually held by the inspected process set."""
    owners, unreadable_pids = _process_socket_inodes(pids)
    rows: list[dict[str, Any]] = []
    sources = ((Path("/proc/net/tcp"), "ipv4"), (Path("/proc/net/tcp6"), "ipv6"))
    tables_readable = True
    for path, family in sources:
        try:
            lines = path.read_text(encoding="ascii").splitlines()[1:]
        except Exception:
            tables_readable = False
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            address_hex, separator, port_hex = fields[1].partition(":")
            inode = fields[9]
            if not separator or inode not in owners:
                continue
            address = _decode_proc_address(address_hex, family)
            try:
                port = int(port_hex, 16)
            except ValueError:
                continue
            if address is None or not 0 < port <= 65535:
                continue
            rows.append(
                {
                    "family": family,
                    "address": address,
                    "port": port,
                    "socket_inode": inode,
                    "pids": sorted(owners[inode]),
                }
            )
    rows.sort(key=lambda row: (row["family"], row["address"], row["port"], row["socket_inode"]))
    return {
        "schema": "prismaquant.server_tcp_listeners/1",
        "tables_readable": tables_readable,
        "unreadable_pids": unreadable_pids,
        "listeners": rows,
    }


def residency_scan(
    pids: Iterable[int | str],
) -> tuple[list[str], list[int], list[int]]:
    """`(basenames, readable_pids, unreadable_pids)` from `/proc/<pid>/maps`.

    The unreadable list is not bookkeeping: reading the maps of a root-owned
    container process from the host is denied, and that denial looks exactly
    like "no extensions are resident" — the false negative that would make two
    different stacks fingerprint identically. The caller records readability so
    an unverified scan can never masquerade as a verified empty one.
    """
    found: set[str] = set()
    readable: list[int] = []
    unreadable: list[int] = []
    for pid in pids:
        try:
            text = Path(f"/proc/{pid}/maps").read_text(errors="replace")
        except Exception:
            unreadable.append(int(pid))
            continue
        readable.append(int(pid))
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            path = parts[-1]
            if not path.startswith("/"):
                continue
            if ".so" not in path:
                continue
            if EXTENSION_PATTERN.search(path):
                found.add(os.path.basename(path))
    return sorted(found), readable, unreadable


def resident_extensions(pids: Iterable[int | str]) -> list[str]:
    """Sorted, de-duplicated basenames of the tracked `.so`s mapped by `pids`."""
    return residency_scan(pids)[0]


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions via metadata only — never imports the package."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def _file_identity(path: Path) -> dict[str, Any]:
    """Exact bytes of one installed/source file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _decode_record_sha256(value: str, *, path: str) -> str:
    prefix = "sha256="
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"Gridbook RECORD has no sha256 for {path}")
    encoded = value[len(prefix):]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as exc:
        raise ValueError(
            f"Gridbook RECORD has an invalid sha256 for {path}"
        ) from exc
    if len(raw) != hashlib.sha256().digest_size:
        raise ValueError(f"Gridbook RECORD has an invalid sha256 for {path}")
    return raw.hex()


def _distribution_file(
    distribution: importlib_metadata.Distribution,
    *,
    filename: str,
) -> tuple[str, Path]:
    matches = [
        item for item in (distribution.files or ())
        if item.name == filename and ".dist-info" in str(item.parent)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"installed Gridbook distribution must contain exactly one {filename}"
        )
    relative = str(matches[0])
    path = Path(distribution.locate_file(matches[0]))
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"installed Gridbook {filename} is missing or is a symlink"
        )
    return relative, path


def validate_gridbook_import_origin_identity(
    payload: Mapping[str, Any],
    *,
    expected_version: str,
) -> None:
    """Replay the path-independent structural part of an import-origin proof.

    The producer resolves symlinks and checks the live filesystem before it
    writes this record.  Consumers may run outside the serving container, so
    replay uses the recorded canonical absolute paths and their containment
    relation rather than trying to dereference container-only paths.
    """
    required = {
        "schema",
        "module_name",
        "imported_version",
        "distribution_package_root",
        "module_file",
        "module_search_locations",
        "identity_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("Gridbook import-origin identity is not closed")
    if (
        payload.get("schema") != GRIDBOOK_IMPORT_ORIGIN_SCHEMA
        or payload.get("module_name") != "gridbook"
        or payload.get("imported_version") != expected_version
    ):
        raise ValueError(
            "Gridbook imported module name/version differs from the distribution"
        )

    root_value = payload.get("distribution_package_root")
    file_value = payload.get("module_file")
    locations = payload.get("module_search_locations")
    if (
        not isinstance(root_value, str)
        or not isinstance(file_value, str)
        or not isinstance(locations, list)
        or not locations
        or any(not isinstance(value, str) for value in locations)
    ):
        raise ValueError("Gridbook import-origin paths are malformed")

    def canonical_absolute(value: str) -> Path:
        path = Path(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or str(path) != value
        ):
            raise ValueError(
                "Gridbook import-origin paths must be canonical and absolute"
            )
        return path

    root = canonical_absolute(root_value)
    module_file = canonical_absolute(file_value)
    search_locations = [canonical_absolute(value) for value in locations]
    try:
        module_file.relative_to(root)
        for location in search_locations:
            location.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "imported Gridbook module escapes the selected distribution package root"
        ) from exc
    if module_file != root / "__init__.py":
        raise ValueError(
            "imported Gridbook __file__ is not the selected distribution __init__.py"
        )
    if locations != sorted(set(locations)):
        raise ValueError(
            "Gridbook import-origin search locations are not canonical and unique"
        )
    unsigned = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    if payload.get("identity_sha256") != _canonical_sha256(unsigned):
        raise ValueError("Gridbook import-origin identity digest is stale")


def gridbook_import_origin_identity(
    distribution: importlib_metadata.Distribution,
    *,
    expected_version: str,
) -> dict[str, Any]:
    """Prove that ``import gridbook`` resolves inside ``distribution``.

    Metadata and imports have independent resolution rules.  In particular, a
    stale ``gridbook`` directory in the current working directory or on
    ``PYTHONPATH`` can be imported even while ``importlib.metadata`` selects the
    exact newly installed distribution.  Resolve both sides and require the
    imported ``__file__`` plus every package ``__path__`` entry to remain under
    the selected distribution's real package root.
    """
    init_items = [
        item for item in (distribution.files or ())
        if str(item) == "gridbook/__init__.py"
    ]
    if len(init_items) != 1:
        raise ValueError(
            "installed Gridbook distribution must contain exactly one "
            "gridbook/__init__.py"
        )
    installed_init = Path(distribution.locate_file(init_items[0]))
    if not installed_init.is_file() or installed_init.is_symlink():
        raise ValueError(
            "installed Gridbook distribution __init__.py is missing or is a symlink"
        )
    try:
        installed_init = installed_init.resolve(strict=True)
        package_root = installed_init.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "installed Gridbook distribution package root is unreadable"
        ) from exc

    try:
        module = importlib.import_module("gridbook")
    except Exception as exc:
        raise ValueError("installed Gridbook module cannot be imported") from exc
    imported_version = getattr(module, "__version__", None)
    if imported_version != expected_version:
        raise ValueError(
            f"imported Gridbook version {imported_version!r} differs from "
            f"installed distribution {expected_version!r}"
        )
    module_file_value = getattr(module, "__file__", None)
    module_path_value = getattr(module, "__path__", None)
    if not isinstance(module_file_value, str) or module_path_value is None:
        raise ValueError("imported Gridbook module has no concrete file/package path")
    try:
        module_file = Path(module_file_value).resolve(strict=True)
        search_locations = sorted({
            str(Path(value).resolve(strict=True)) for value in module_path_value
        })
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("imported Gridbook module paths are unreadable") from exc
    if not search_locations:
        raise ValueError("imported Gridbook module has an empty __path__")
    if module_file != installed_init:
        raise ValueError(
            "imported Gridbook __file__ does not equal the selected installed "
            "distribution's __init__.py (CWD/PYTHONPATH shadow suspected)"
        )
    try:
        module_file.relative_to(package_root)
        for location in map(Path, search_locations):
            location.relative_to(package_root)
    except ValueError as exc:
        raise ValueError(
            "imported Gridbook module escapes the selected installed "
            "distribution package root (CWD/PYTHONPATH shadow suspected)"
        ) from exc
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if not isinstance(spec_origin, str):
        raise ValueError("imported Gridbook module has no concrete spec origin")
    try:
        resolved_spec_origin = Path(spec_origin).resolve(strict=True)
    except OSError as exc:
        raise ValueError("imported Gridbook spec origin is unreadable") from exc
    if resolved_spec_origin != module_file:
        raise ValueError("imported Gridbook __spec__.origin differs from __file__")

    identity: dict[str, Any] = {
        "schema": GRIDBOOK_IMPORT_ORIGIN_SCHEMA,
        "module_name": "gridbook",
        "imported_version": imported_version,
        "distribution_package_root": str(package_root),
        "module_file": str(module_file),
        "module_search_locations": search_locations,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    validate_gridbook_import_origin_identity(
        identity, expected_version=expected_version
    )
    return identity


def _normalized_gridbook_distribution_pin(
    expected_pin: Mapping[str, str],
) -> dict[str, str]:
    """Return one closed VCS or wheel-backed Gridbook distribution pin."""

    base_keys = {"repository", "commit", "version"}
    keys = set(expected_pin) if isinstance(expected_pin, Mapping) else set()
    if (
        keys not in (base_keys, base_keys | {"wheel_sha256"})
        or expected_pin.get("repository") != GRIDBOOK_REPOSITORY
        or re.fullmatch(
            r"[0-9a-f]{40}", str(expected_pin.get("commit", ""))
        ) is None
        or not isinstance(expected_pin.get("version"), str)
        or not expected_pin.get("version")
        or (
            "wheel_sha256" in keys
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(expected_pin.get("wheel_sha256", "")),
            ) is None
        )
    ):
        raise ValueError("Gridbook distribution pin is not closed and exact")
    return {key: str(expected_pin[key]) for key in sorted(keys)}


def validate_gridbook_pep610_direct_url(
    direct_url: object,
    expected_pin: Mapping[str, str],
) -> str:
    """Validate PEP 610 identity for an exact VCS or release-wheel install.

    The commit remains part of both pin forms.  A wheel install additionally
    binds the independently verified wheel SHA-256; its installed source and
    metadata bytes are bound to RECORD by :func:`gridbook_distribution_provenance`.
    """

    pin = _normalized_gridbook_distribution_pin(expected_pin)
    if not isinstance(direct_url, Mapping):
        raise ValueError("installed Gridbook PEP 610 direct_url is not an object")
    direct_transport = direct_url.get("url")
    try:
        parsed_transport = urlsplit(direct_transport) if isinstance(
            direct_transport, str
        ) else None
    except ValueError:
        parsed_transport = None
    local_transport = (
        parsed_transport is not None
        and parsed_transport.scheme == "file"
        and parsed_transport.netloc in {"", "localhost"}
        and parsed_transport.path.startswith("/")
        and not parsed_transport.query
        and not parsed_transport.fragment
        and parsed_transport.username is None
        and parsed_transport.password is None
    )

    wheel_sha256 = pin.get("wheel_sha256")
    if wheel_sha256 is None:
        expected_vcs = {
            "vcs": "git",
            "requested_revision": pin["commit"],
            "commit_id": pin["commit"],
        }
        if (
            set(direct_url) != {"url", "vcs_info"}
            or direct_url.get("vcs_info") != expected_vcs
            or not (direct_transport == pin["repository"] or local_transport)
        ):
            raise ValueError(
                "installed Gridbook PEP 610 direct_url is not the exact "
                "pinned VCS commit"
            )
        return "vcs"

    archive_info = direct_url.get("archive_info")
    archive_keys = set(archive_info) if isinstance(
        archive_info, Mapping
    ) else set()
    hashes = archive_info.get("hashes") if isinstance(
        archive_info, Mapping
    ) else None
    wheel_name = (
        Path(unquote(parsed_transport.path)).name
        if parsed_transport is not None else ""
    )
    wheel_name_pattern = re.compile(
        rf"gridbook-{re.escape(pin['version'])}-[A-Za-z0-9_.+-]+[.]whl"
    )
    legacy_hash_valid = (
        "hash" not in archive_keys
        or archive_info.get("hash") == f"sha256={wheel_sha256}"
    ) if isinstance(archive_info, Mapping) else False
    if (
        set(direct_url) != {"url", "archive_info"}
        or not local_transport
        or wheel_name_pattern.fullmatch(wheel_name) is None
        or archive_keys not in ({"hashes"}, {"hash", "hashes"})
        or hashes != {"sha256": wheel_sha256}
        or not legacy_hash_valid
    ):
        raise ValueError(
            "installed Gridbook PEP 610 direct_url is not the exact pinned "
            "release wheel"
        )
    return "wheel"


def gridbook_distribution_provenance(
    expected_pin: Mapping[str, str],
) -> dict[str, Any]:
    """Attest the installed Gridbook package, not merely its version label.

    PEP 610 binds the install to either the exact external VCS revision or an
    independently pinned release-wheel digest.  RECORD then binds installed
    source/CUDA files and package metadata to their bytes.  Both are needed: a
    matching ``gridbook.__version__`` can be produced by unrelated code, while
    a truthful direct URL alone says nothing about post-install file mutation.
    """
    expected_pin = _normalized_gridbook_distribution_pin(expected_pin)

    try:
        distribution = importlib_metadata.distribution("gridbook")
    except Exception as exc:
        raise ValueError("the Gridbook distribution is not installed") from exc
    name = str(distribution.metadata.get("Name", "")).strip().lower()
    version = str(distribution.version)
    if name != "gridbook" or version != expected_pin["version"]:
        raise ValueError(
            "installed Gridbook name/version differs from the exact pin"
        )
    import_origin = gridbook_import_origin_identity(
        distribution, expected_version=version
    )

    direct_relative, direct_path = _distribution_file(
        distribution, filename="direct_url.json"
    )
    try:
        direct_url = json.loads(direct_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("installed Gridbook direct_url.json is unreadable") from exc
    validate_gridbook_pep610_direct_url(direct_url, expected_pin)

    record_relative, record_path = _distribution_file(
        distribution, filename="RECORD"
    )
    metadata_relative, metadata_path = _distribution_file(
        distribution, filename="METADATA"
    )
    record_rows: dict[str, tuple[str, str]] = {}
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 3 or not row[0] or row[0] in record_rows:
                    raise ValueError("malformed or duplicate RECORD row")
                record_rows[row[0]] = (row[1], row[2])
    except Exception as exc:
        raise ValueError("installed Gridbook RECORD is malformed") from exc

    source_items = [
        item for item in (distribution.files or ())
        if str(item).startswith("gridbook/")
        and "__pycache__" not in item.parts
        and Path(str(item)).suffix in _GRIDBOOK_SOURCE_SUFFIXES
    ]
    source_names = {str(item) for item in source_items}
    missing = sorted(_REQUIRED_GRIDBOOK_SOURCE_FILES - source_names)
    if missing:
        raise ValueError(
            f"installed Gridbook source/package-data closure is missing {missing}"
        )
    source_files: dict[str, dict[str, Any]] = {}
    for item in sorted(source_items, key=str):
        relative = str(item)
        path = Path(distribution.locate_file(item))
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"installed Gridbook source is missing or is a symlink: {relative}"
            )
        identity = _file_identity(path)
        record_hash, record_size = record_rows.get(relative, ("", ""))
        if (
            _decode_record_sha256(record_hash, path=relative)
            != identity["sha256"]
            or not record_size.isdigit()
            or int(record_size) != identity["bytes"]
        ):
            raise ValueError(
                f"installed Gridbook source differs from RECORD: {relative}"
            )
        source_files[relative] = identity

    direct_identity = _file_identity(direct_path)
    metadata_identity = _file_identity(metadata_path)
    for relative, identity in (
        (direct_relative, direct_identity),
        (metadata_relative, metadata_identity),
    ):
        record_hash, record_size = record_rows.get(relative, ("", ""))
        if (
            _decode_record_sha256(record_hash, path=relative)
            != identity["sha256"]
            or not record_size.isdigit()
            or int(record_size) != identity["bytes"]
        ):
            raise ValueError(
                f"installed Gridbook metadata differs from RECORD: {relative}"
            )

    return {
        "schema": GRIDBOOK_DISTRIBUTION_SCHEMA,
        "name": "gridbook",
        "repository": expected_pin["repository"],
        "version": version,
        "direct_url": direct_url,
        "direct_url_path": direct_relative,
        "direct_url_identity": direct_identity,
        "metadata_path": metadata_relative,
        "metadata_identity": metadata_identity,
        "record_path": record_relative,
        "record_identity": _file_identity(record_path),
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(source_files),
        "import_origin": import_origin,
    }


def gridbook_runtime_pin() -> dict[str, str] | None:
    """Immutable external Gridbook identity injected by the serve helper."""
    mapping = {
        "commit": "PQ_GRIDBOOK_RUNTIME_COMMIT",
        "version": "PQ_GRIDBOOK_RUNTIME_VERSION",
        "wheel_sha256": "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256",
    }
    value = {
        field: os.environ[name]
        for field, name in mapping.items()
        if os.environ.get(name)
    }
    return value or None


def git_provenance(repo: str | os.PathLike | None = None) -> dict[str, Any]:
    """Full producer commit plus an independently observed clean-tree bit."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    commit_override = os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    ).strip().lower()
    if commit_override and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_override
    ) is None:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or 64-hex commit"
        )
    dirty_text = os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_DIRTY", ""
    ).strip().lower()
    dirty_values = {
        "0": False, "false": False, "no": False,
        "1": True, "true": True, "yes": True,
    }
    if dirty_text and dirty_text not in dirty_values:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY must be one of "
            "0/1/false/true/no/yes"
        )

    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments], cwd=root, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
            ).stdout.strip()
        except Exception:
            return None

    observed = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    observed_dirty = None if status is None else bool(status)
    if (
        commit_override
        and observed is not None
        and commit_override != observed.lower()
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT contradicts the mounted checkout"
        )
    dirty_override = dirty_values[dirty_text] if dirty_text else None
    if (
        dirty_override is not None
        and observed_dirty is not None
        and dirty_override is not observed_dirty
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY contradicts the mounted checkout"
        )
    return {
        "commit": commit_override or observed,
        "tree": tree,
        "dirty": dirty_override if dirty_override is not None else observed_dirty,
    }


def git_commit(repo: str | os.PathLike | None = None) -> str | None:
    """Compatibility projection of :func:`git_provenance`."""
    return git_provenance(repo).get("commit")


def gold_producer_identity(measurement_tool: str) -> dict[str, Any]:
    """Bind a gold number to clean producer code and its exact source bytes."""
    tool_files = _GOLD_PRODUCER_TOOL_FILES.get(measurement_tool)
    if tool_files is None:
        raise ValueError(f"unknown gold measurement tool {measurement_tool!r}")
    provenance = git_provenance()
    commit = provenance.get("commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        raise ValueError("gold producer identity has no full git commit")
    if provenance.get("dirty") is not False:
        raise ValueError(
            "gold producer identity requires a proven clean PrismaQuant tree"
        )

    root = Path(__file__).resolve().parents[1]
    names = sorted(set(_GOLD_PRODUCER_COMMON_FILES + tuple(tool_files)))
    source_files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"gold producer source is missing or a symlink: {name}")
        source_files[name] = _file_identity(path)
    return {
        "schema": GOLD_PRODUCER_IDENTITY_SCHEMA,
        "measurement_tool": measurement_tool,
        "git_commit": commit,
        "git_tree": provenance.get("tree"),
        "git_dirty": False,
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(source_files),
    }


def artifact_binding(
    model_dir: str | os.PathLike,
    *,
    launch_model: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Bind a live server manifest to the exact mounted CB artifact."""
    root = Path(model_dir)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"served artifact path cannot be resolved: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"served artifact path is not a directory: {resolved_root}")
    if launch_model is not None:
        launch_path = Path(launch_model)
        if not launch_path.is_absolute():
            raise ValueError("serve launch model must be an absolute artifact path")
        try:
            resolved_launch = launch_path.resolve(strict=True)
            same_artifact = os.path.samefile(resolved_root, resolved_launch)
        except OSError as exc:
            raise ValueError(
                f"serve launch model cannot be resolved: {launch_model}"
            ) from exc
        if not same_artifact:
            raise ValueError(
                "--artifact-dir does not resolve to the artifact in the serve argv"
            )
    from prismaquant.shipcard import compute_model_sha

    quant_path = resolved_root / "quant_config.json"
    payload = json.loads(quant_path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    inventory = provenance.get("artifact_inventory") if isinstance(
        provenance, dict
    ) else None
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema")
        != "prismaquant.cb_export_artifact_inventory.v1"
        or inventory.get("scope") != "all_regular_files_recursive"
    ):
        raise ValueError("served artifact has no finalized recursive CB inventory")
    file_bytes = inventory.get("file_bytes")
    if not isinstance(file_bytes, dict) or not file_bytes:
        raise ValueError("served artifact inventory has no file ledger")
    observed: dict[str, int] = {}
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"served artifact contains symlink {path.relative_to(resolved_root)}"
            )
        if path.is_file():
            observed[path.relative_to(resolved_root).as_posix()] = int(path.stat().st_size)
    if observed != file_bytes or sum(observed.values()) != inventory.get(
        "export_directory_bytes"
    ):
        raise ValueError("served artifact files differ from finalized inventory")
    canonical = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema": "prismaquant.served_artifact_binding/1",
        "resolved_path": str(resolved_root),
        "launch_model": str(launch_model) if launch_model is not None else None,
        "model_sha": compute_model_sha(resolved_root),
        "artifact_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifact_bytes": sum(observed.values()),
    }


def gpu_identity() -> dict[str, Any]:
    """Stable GPU UUID, name, and driver without creating a CUDA context."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,uuid,driver_version",
             "--format=csv,noheader"],
            check=True, text=True, timeout=30,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip().splitlines()
    except Exception:
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver_version": None,
            "gpu_count": 0,
        }
    if not out:
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver_version": None,
            "gpu_count": 0,
        }
    first = [field.strip() for field in out[0].split(",")]
    return {
        "gpu_name": first[0] if first else None,
        "gpu_uuid": first[1] if len(first) > 1 else None,
        "driver_version": first[2] if len(first) > 2 else None,
        "gpu_count": len(out),
    }


# ---------------------------------------------------------------------------
# argv handling
# ---------------------------------------------------------------------------
def elide_argv_paths(argv: Sequence[str]) -> list[str]:
    """Replace every path-like token with `<path>`.

    This is what makes the fingerprint a property of the *stack* rather than of
    the run: arm A and arm B of an A/B name different artifact directories and
    different output files, and must still share a fingerprint, while
    `--enforce-eager`, `--kv-cache-dtype fp8` or a changed image must not.
    """
    out: list[str] = []
    for token in argv:
        if "/" in token or token.startswith("~"):
            out.append(_PATH_PLACEHOLDER)
        else:
            out.append(token)
    return out


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag:
            return argv[index + 1] if index + 1 < len(argv) else ""
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _serve_model(argv: Sequence[str]) -> str | None:
    explicit = _flag_value(argv, "--model")
    if explicit:
        return explicit
    for index, token in enumerate(argv):
        if token == "serve" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def normalize_performance_argv(argv: Sequence[str]) -> list[str]:
    """Canonical server argv with only arm artifact/name identity masked."""
    result: list[str] = []
    serve_index = next((index for index, value in enumerate(argv) if value == "serve"), None)
    model_positional = serve_index + 1 if serve_index is not None else None
    masked_value_flags = {"--model", "--served-model-name"}
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if model_positional is not None and index == model_positional:
            result.append(_ARM_MODEL_PLACEHOLDER)
            index += 1
            continue
        matched = next(
            (
                flag
                for flag in masked_value_flags
                if token == flag or token.startswith(flag + "=")
            ),
            None,
        )
        if matched is None:
            result.append(token)
            index += 1
            continue
        if token == matched:
            result.extend([matched, _ARM_MODEL_PLACEHOLDER])
            index += 2
        else:
            result.append(matched + "=" + _ARM_MODEL_PLACEHOLDER)
            index += 1
    return result


def performance_stack_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Release stack identity with candidate/baseline names intentionally absent."""
    listener = manifest.get("listener_binding")
    listener_stack = None
    if isinstance(listener, Mapping):
        listener_stack = {
            "base_url": listener.get("base_url"),
            "launch_host": listener.get("launch_host"),
            "launch_port": listener.get("launch_port"),
        }
    environment = manifest.get("server_process_environment")
    environment_values = environment.get("values") if isinstance(
        environment, Mapping
    ) else None
    return {
        "image": manifest.get("image"),
        # Host boot, container hostname/PID namespace, and process IDs define
        # live sessions, not the matched stack. Same-Spark equality is checked
        # explicitly from boot_id + physical GPU UUID across arms.
        "gpu_name": manifest.get("gpu_name"),
        "gpu_uuid": manifest.get("gpu_uuid"),
        "gpu_count": manifest.get("gpu_count"),
        "driver_version": manifest.get("driver_version"),
        "package_versions": manifest.get("package_versions"),
        "gridbook_runtime_pin": manifest.get("gridbook_runtime_pin"),
        "gridbook_distribution": manifest.get("gridbook_distribution"),
        "resident_extensions": manifest.get("resident_extensions"),
        "residency_readable": manifest.get("residency_readable"),
        "normalized_serve_argv": manifest.get("normalized_performance_argv"),
        "server_environment": environment_values,
        "listener": listener_stack,
    }


def performance_stack_fingerprint(manifest: Mapping[str, Any]) -> str:
    return _canonical_sha256(performance_stack_payload(manifest))


def _canonical_base_url(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("base URL must be an origin-only http(s) URL without credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("base URL has an invalid port") from exc
    host = parsed.hostname.lower()
    display_host = f"[{host}]" if ":" in host else host
    default_port = 443 if parsed.scheme == "https" else 80
    netloc = display_host if port == default_port else f"{display_host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", "")), host, port


def _models_endpoint_url(value: str) -> str:
    """Canonical ``/v1/models`` URL for an origin or OpenAI base URL."""
    parsed = urlsplit(value)
    if parsed.path.rstrip("/") not in {"", "/v1", "/v1/models"}:
        raise ValueError("models binding URL must be an origin, /v1, or /v1/models")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", parsed.query, parsed.fragment))
    canonical_origin, _host, _port = _canonical_base_url(origin)
    return canonical_origin + "/v1/models"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"models response repeats JSON key {key!r}")
        result[key] = value
    return result


def models_endpoint_binding_from_bytes(
    raw: bytes,
    *,
    request_url: str,
    expected_served_model: str,
) -> dict[str, Any]:
    """Bind the exact model-list bytes returned by one live server session."""
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("models endpoint returned an empty response")
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("models endpoint response exceeds the 16 MiB evidence limit")
    if not isinstance(expected_served_model, str) or not expected_served_model:
        raise ValueError("expected served model must be non-empty")
    try:
        payload = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"models response contains non-finite number {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("models endpoint did not return valid UTF-8 JSON") from exc
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("object") != "list"
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
    ):
        raise ValueError("models endpoint must return exactly one model card")
    row = rows[0]
    created = row.get("created")
    root = row.get("root")
    owned_by = row.get("owned_by")
    if (
        row.get("id") != expected_served_model
        or row.get("object") != "model"
        or isinstance(created, bool)
        or not isinstance(created, int)
        or created <= 0
        or not isinstance(root, str)
        or not root
        or not isinstance(owned_by, str)
        or not owned_by
    ):
        raise ValueError(
            "models endpoint card lacks the exact id/object/created/root/owner identity"
        )
    stable_model = {
        "id": expected_served_model,
        "object": "model",
        "owned_by": owned_by,
        "root": root,
        "max_model_len": row.get("max_model_len"),
    }
    canonical_identity = {
        "response_object": "list",
        "model_count": 1,
        "model": stable_model,
    }
    return {
        "schema": MODELS_ENDPOINT_BINDING_SCHEMA,
        "request_url": _models_endpoint_url(request_url),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "canonical_identity_sha256": _canonical_sha256(canonical_identity),
        **canonical_identity,
    }


def models_endpoint_binding_identity(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Path-independent projection shared by the server and validation client."""
    full_keys = {
        "schema", "request_url", "response_sha256", "response_bytes",
        "canonical_identity_sha256", "response_object", "model_count", "model",
    }
    identity_keys = {
        "schema", "canonical_identity_sha256", "response_object",
        "model_count", "model",
    }
    observed_keys = set(binding) if isinstance(binding, Mapping) else set()
    model = binding.get("model") if isinstance(binding, Mapping) else None
    response_sha = binding.get("response_sha256") if isinstance(
        binding, Mapping
    ) else None
    if (
        not isinstance(binding, Mapping)
        or observed_keys not in (full_keys, identity_keys)
        or binding.get("schema") != MODELS_ENDPOINT_BINDING_SCHEMA
        or binding.get("response_object") != "list"
        or binding.get("model_count") != 1
        or not isinstance(model, Mapping)
        or set(model) != {"id", "object", "owned_by", "root", "max_model_len"}
        or not isinstance(model.get("id"), str)
        or not model.get("id")
        or model.get("object") != "model"
        or not isinstance(model.get("owned_by"), str)
        or not model.get("owned_by")
        or not isinstance(model.get("root"), str)
        or not model.get("root")
        or (
            model.get("max_model_len") is not None
            and (
                isinstance(model.get("max_model_len"), bool)
                or not isinstance(model.get("max_model_len"), int)
                or model.get("max_model_len", 0) <= 0
            )
        )
    ):
        raise ValueError("models endpoint binding is malformed or non-canonical")
    if observed_keys == full_keys and (
        _models_endpoint_url(str(binding.get("request_url", "")))
        != binding.get("request_url")
        or not isinstance(response_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", response_sha) is None
        or isinstance(binding.get("response_bytes"), bool)
        or not isinstance(binding.get("response_bytes"), int)
        or binding.get("response_bytes", 0) <= 0
    ):
        raise ValueError("models endpoint observation is malformed or non-canonical")
    identity = {
        "response_object": "list",
        "model_count": 1,
        "model": dict(model),
    }
    canonical_sha = binding.get("canonical_identity_sha256")
    if (
        not isinstance(canonical_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", canonical_sha) is None
        or canonical_sha != _canonical_sha256(identity)
    ):
        raise ValueError("models endpoint canonical identity digest is stale")
    return {
        "schema": MODELS_ENDPOINT_BINDING_SCHEMA,
        "canonical_identity_sha256": canonical_sha,
        **identity,
    }


def query_models_endpoint_binding(
    base_url: str,
    *,
    expected_served_model: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Read and bind the exact `/v1/models` response from a live endpoint."""
    request_url = _models_endpoint_url(base_url)
    request = urllib.request.Request(
        request_url,
        method="GET",
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise ValueError(
                    f"GET {request_url} returned HTTP {response.status}"
                )
            raw = response.read(16 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"GET {request_url} failed: {exc}") from exc
    return models_endpoint_binding_from_bytes(
        raw,
        request_url=request_url,
        expected_served_model=expected_served_model,
    )


def listener_binding(
    launch_argv: Sequence[str],
    listeners: Mapping[str, Any],
    *,
    base_url: str | None,
) -> dict[str, Any] | None:
    if base_url is None:
        return None
    canonical_url, url_host, url_port = _canonical_base_url(base_url)
    launch_host = _flag_value(launch_argv, "--host") or "0.0.0.0"
    raw_port = _flag_value(launch_argv, "--port") or "8000"
    try:
        launch_port = int(raw_port)
    except ValueError as exc:
        raise ValueError("serve argv --port is not an integer") from exc
    if not 0 < launch_port <= 65535 or url_port != launch_port:
        raise ValueError("base URL port does not equal the server launch/listener port")
    rows = listeners.get("listeners") if isinstance(listeners, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("server TCP listener census is unavailable")
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("port") == launch_port
    ]
    if not matching:
        raise ValueError("no inspected server process owns the declared TCP listener")
    addresses = {str(row.get("address")) for row in matching}
    wildcard = {"0.0.0.0", "::"}
    if launch_host not in addresses and not (addresses & wildcard):
        raise ValueError("actual server listener address differs from --host")
    if url_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("release benchmark base URL must target the local server")
    return {
        "schema": "prismaquant.server_listener_binding/1",
        "base_url": canonical_url,
        "launch_host": launch_host,
        "launch_port": launch_port,
        "listeners": matching,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def collect_manifest(
    *,
    pids: Sequence[int] | None = None,
    launch_argv: Sequence[str] | None = None,
    image: str | None = None,
    source: str = "server",
    extra: Mapping[str, Any] | None = None,
    artifact_dir: str | os.PathLike | None = None,
    base_url: str | None = None,
    attestation_phase: str = "snapshot",
) -> dict[str, Any]:
    """Build the manifest for a live serving (or measuring) process."""
    if attestation_phase not in {"snapshot", "pre", "post"}:
        raise ValueError("attestation_phase must be snapshot, pre, or post")
    if pids is None:
        pids = find_server_pids()
    elif source == "server" and (
        base_url is not None or attestation_phase in {"pre", "post"}
    ):
        discovered = find_server_pids()
        if sorted(set(pids)) != discovered:
            raise ValueError(
                "server attestation must inspect the complete auto-discovered vLLM process set"
            )
    pids = sorted(set(pids))
    if not pids:
        if source == "server":
            raise ValueError("no live vLLM server processes were found")
        pids = [os.getpid()]

    if launch_argv is None:
        argv = None
        for pid in pids:
            candidate = _read_cmdline(pid)
            if candidate and any("serve" == token for token in candidate):
                argv = candidate
                break
        if argv is None:
            argv = _read_cmdline(pids[0]) or list(sys.argv)
        launch_argv = argv
    launch_argv = list(launch_argv)
    launch_model = _serve_model(launch_argv)

    enforce_eager = "--enforce-eager" in launch_argv or (
        "--enforce_eager" in launch_argv)
    extensions, readable_pids, unreadable_pids = residency_scan(pids)
    host = host_identity()
    processes = process_identities(pids, boot_id=host.get("boot_id"))
    process_environment = server_environment_snapshot(pids)
    listener_census = process_tcp_listeners(pids)
    bound_listener = listener_binding(
        launch_argv, listener_census, base_url=base_url
    )
    endpoint_models = (
        models_endpoint_binding_identity(query_models_endpoint_binding(
            base_url,
            expected_served_model=str(
                _flag_value(launch_argv, "--served-model-name") or ""
            ),
        ))
        if base_url is not None
        else None
    )
    gpu = gpu_identity()
    runtime_pin = gridbook_runtime_pin()
    gridbook_distribution = None
    if runtime_pin is not None:
        if set(runtime_pin) not in (
            {"commit", "version"},
            {"commit", "version", "wheel_sha256"},
        ):
            raise ValueError(
                "Gridbook runtime environment pin is partial; commit/version "
                "and optional wheel SHA-256 must form one closed pin"
            )
        gridbook_distribution = gridbook_distribution_provenance({
            "repository": GRIDBOOK_REPOSITORY,
            **runtime_pin,
        })
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "attestation_phase": attestation_phase,
        "source": source,
        "hostname": host["hostname"],
        "host_identity": host,
        "image": image or os.environ.get("PQ_SERVE_IMAGE"),
        "model": launch_model,
        "served_model_name": _flag_value(launch_argv, "--served-model-name"),
        "launch_argv": launch_argv,
        "launch_flags": elide_argv_paths(launch_argv),
        "normalized_performance_argv": normalize_performance_argv(launch_argv),
        "enforce_eager": bool(enforce_eager),
        "quantization": _flag_value(launch_argv, "--quantization"),
        "kv_cache_dtype": _flag_value(launch_argv, "--kv-cache-dtype"),
        "speculative_config": _flag_value(launch_argv, "--speculative-config"),
        "package_versions": package_versions(),
        "gridbook_runtime_pin": runtime_pin,
        "resident_extensions": extensions,
        # False whenever any inspected process's address space could not be
        # read (the host-side-of-a-container case): an unverified scan must not
        # fingerprint the same as a verified "nothing resident".
        "residency_readable": bool(readable_pids) and not unreadable_pids,
        "processes": processes,
        "server_process_environment": process_environment,
        # Compatibility field for existing endpoint/gold readers. Its source
        # is now the actual server processes rather than docker-exec.
        "pq_env": process_environment.get("values") or {},
        "listener_census": listener_census,
        "listener_binding": bound_listener,
        "models_endpoint_binding": endpoint_models,
    }
    if gridbook_distribution is not None:
        manifest["gridbook_distribution"] = gridbook_distribution
    manifest.update(gpu)
    if artifact_dir is not None:
        manifest["artifact_binding"] = artifact_binding(
            artifact_dir, launch_model=launch_model
        )
    if extra:
        annotations = dict(extra)
        if any(
            not isinstance(key, str)
            or not key
            or key.startswith("_")
            or key in manifest
            or key in _IN_PROCESS_OBSERVED_FIELDS
            for key in annotations
        ):
            raise ValueError(
                "extra annotations must use unique non-reserved public keys"
            )
        manifest.update(annotations)
    process_hashes = [row.get("identity_sha256") for row in processes]
    if any(not isinstance(value, str) for value in process_hashes):
        manifest["serve_session_id"] = None
    else:
        manifest["serve_session_id"] = serve_session_fingerprint(manifest)
    manifest["performance_stack_fingerprint"] = performance_stack_fingerprint(
        manifest
    )
    manifest["serve_fingerprint"] = fingerprint(manifest)
    return manifest


def fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest reduced to what defines the numeric stack."""
    return {
        key: value for key, value in manifest.items()
        if key not in _FINGERPRINT_EXCLUDED
    }


def fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = fingerprint_payload(manifest)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_differences(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> list[str]:
    """Fingerprint-relevant keys on which two manifests disagree."""
    if not left or not right:
        return []
    a, b = fingerprint_payload(left), fingerprint_payload(right)
    return sorted(
        key for key in set(a) | set(b) if a.get(key) != b.get(key)
    )


def self_manifest(
    *,
    image: str | None = None,
    extra: Mapping[str, Any] | None = None,
    artifact_dir: str | os.PathLike | None = None,
    require_engine_descendant: bool = False,
    gridbook_pin_attestation: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Manifest of this process and its complete descendant process tree.

    `tools/measure_vllm_full_kl.py` and `tools/measure_vllm_wikitext_ppl.py`
    construct their own `LLM`.  On vLLM v1 the Python process is the front end
    and a spawned EngineCore holds the kernels, so both address spaces are
    required for authoritative residency/environment evidence. Membership is
    ancestry-only, excluding unrelated vLLM processes in the same container.

    Set ``require_engine_descendant`` for release measurements that must prove
    a descendant whose argv identifies EngineCore/a vLLM engine was alive.
    """
    if not isinstance(require_engine_descendant, bool):
        raise TypeError("require_engine_descendant must be a bool")
    if gridbook_pin_attestation is not None:
        try:
            gridbook_pin_attestation = _normalized_gridbook_distribution_pin(
                gridbook_pin_attestation
            )
        except ValueError as exc:
            raise ValueError(
                "gridbook_pin_attestation is not an exact runtime pin"
            ) from exc
    parent_pid = os.getpid()
    descendants = descendant_process_pids(parent_pid)
    engine_descendants = [
        pid
        for pid in descendants
        if argv_identifies_vllm_engine(_read_cmdline(pid))
    ]
    if require_engine_descendant and not engine_descendants:
        raise ValueError(
            "in-process serve manifest found no live EngineCore/VLLM engine "
            "descendant of the measurement process"
        )

    manifest = collect_manifest(
        pids=[parent_pid, *descendants],
        launch_argv=list(sys.argv),
        image=image,
        source="in_process",
        extra=extra,
        artifact_dir=artifact_dir,
    )
    manifest["measurement_parent_pid"] = parent_pid
    manifest["engine_descendant_pids"] = engine_descendants
    if gridbook_pin_attestation is not None:
        expected_runtime_pin = {
            key: value for key, value in gridbook_pin_attestation.items()
            if key != "repository"
        }
        observed_runtime_pin = manifest.get("gridbook_runtime_pin")
        if (
            observed_runtime_pin is not None
            and observed_runtime_pin != expected_runtime_pin
        ):
            raise ValueError(
                "gridbook_pin_attestation differs from the live runtime environment"
            )
        manifest["gridbook_runtime_pin"] = expected_runtime_pin
        if "gridbook_distribution" not in manifest:
            manifest["gridbook_distribution"] = (
                gridbook_distribution_provenance(gridbook_pin_attestation)
            )
        manifest["performance_stack_fingerprint"] = (
            performance_stack_fingerprint(manifest)
        )
    # Keep the invariant explicit if the fingerprint projection evolves. Live
    # PIDs are excluded today; the full process identities bind the session.
    manifest["serve_fingerprint"] = fingerprint(manifest)
    return manifest


def load_manifest(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def find_manifest(model_dir: str | os.PathLike | None) -> Path | None:
    if not model_dir:
        return None
    candidate = Path(model_dir) / MANIFEST_FILENAME
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_write(args: argparse.Namespace) -> int:
    if args.pid is not None:
        raise ValueError(
            "--pid is not permitted for server attestations; inspect the complete vLLM process set"
        )
    # The DSv4 release container deliberately has no installed PrismaQuant.
    # When an exact transported root is present, prove the lazy shipcard import
    # used by artifact_binding resolves to that snapshot before inspecting or
    # writing any serving evidence. The outer launcher separately re-hashes the
    # complete snapshot closure immediately before this command.
    transported_root = os.environ.get("PQ_RUNTIME_PRISMAQUANT_ROOT")
    if transported_root is not None:
        root = Path(transported_root)
        if not root.is_absolute() or root.is_symlink():
            raise ValueError(
                "transported PrismaQuant root must be absolute and non-symlink"
            )
        root = root.resolve(strict=True)
        module = importlib.import_module("prismaquant.shipcard")
        module_file = getattr(module, "__file__", None)
        expected = (root / "prismaquant" / "shipcard.py").resolve(strict=True)
        if not isinstance(module_file, str) or Path(module_file).resolve(
            strict=True
        ) != expected:
            raise ValueError(
                "serve fingerprint shipcard import escapes the reviewed snapshot"
            )
    manifest = collect_manifest(
        pids=None,
        image=args.image,
        artifact_dir=args.artifact_dir,
        base_url=args.base_url,
        attestation_phase=args.attestation_phase,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[serve-manifest] {out} fingerprint="
          f"{manifest['serve_fingerprint'][:16]} "
          f"extensions={manifest['resident_extensions']}")
    if not manifest["residency_readable"]:
        print("[serve-manifest] WARN could not read every inspected process's "
              "/proc/<pid>/maps — the extension list is INCOMPLETE. Run this "
              "inside the serving container (docker exec), not on the host: "
              "an unreadable scan is not evidence that nothing is resident.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser(
        "write", help="write serve_manifest.json for the live server")
    p_write.add_argument("--out", required=True)
    p_write.add_argument("--image", default=None,
                         help="container image tag the server runs in")
    p_write.add_argument("--pid", type=int, default=None,
                         help="inspect only this pid (default: auto-discover "
                              "the vLLM server + engine processes)")
    p_write.add_argument(
        "--artifact-dir",
        default=None,
        help="exact mounted CB artifact served by this process",
    )
    p_write.add_argument(
        "--base-url",
        default=None,
        help="local benchmark origin bound to the actual server listener",
    )
    p_write.add_argument(
        "--attestation-phase",
        choices=("snapshot", "pre", "post"),
        default="snapshot",
        help="chronology role for this immutable server snapshot",
    )
    p_write.set_defaults(func=_cmd_write)

    p_show = sub.add_parser("show", help="pretty-print a manifest")
    p_show.add_argument("manifest")
    p_show.set_defaults(func=_cmd_show)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
