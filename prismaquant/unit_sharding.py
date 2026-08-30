"""Deterministic unit sharding for the render-bound production stages.

One render-bound stage run may be split across boxes by *unit* (parallel
producer v1). The split must be a pure function of the stage's own ordered
unit enumeration so that every shard, on every box, agrees on who renders
what without a coordinator handing out work items.

Three properties are load-bearing and are why this module exists instead of
an inline ``units[i::n]`` in each stage:

1. **Layer-contiguous.** Units are grouped into *atoms* by decoder-layer
   index and an atom is never split. Fused-sibling groups (q/k/v, gate/up)
   live inside one layer, and the NVFP4 joint fused-sibling global scale is
   computed as a max over the group members present in the run
   (``export_native_compressed._compute_nvfp4_joint_global``). A partition
   that split a fused group would hand its members different scales and the
   shard's rendered bytes would stop being the unsharded run's bytes. Layer
   atomicity is the invariant that keeps a sharded render bit-identical.
   Callers that know the real fused grouping should additionally assert it
   with :func:`assert_groups_within_atoms`.
2. **Balanced by exact source bytes.** The balance objective is the exact
   ``numel * element_size`` of each unit's source tensor, minimized over the
   worst shard by an exact contiguous-partition DP — not a unit count, not
   an estimate.
3. **Pure.** No RNG, no clock, no host state. ``(ordered units, N)`` in,
   partition + ``partition_hash`` out, so a merge tool can recompute the
   same partition from the stamp and gate completeness fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SCHEMA = "prismaquant.unit_sharding.v1"
SHARD_ENV = "PRISMAQUANT_UNIT_SHARD"

_SHARD_RE = re.compile(r"^(\d+)/(\d+)$")

# A decoder-layer atom is the qname prefix through the layer index. The
# container names are the explicit set the model profiles in this repo use;
# a unit that matches none of them becomes its own singleton atom, which is
# always a safe partition (it can only be over-split, never under-grouped)
# but gives the caller no fused-group guarantee — hence
# `assert_groups_within_atoms`.
_LAYER_CONTAINERS = ("layers", "blocks", "h")
_LAYER_RE = re.compile(
    r"^(?P<prefix>.*?\b(?:" + "|".join(_LAYER_CONTAINERS) + r")\.\d+)(?:\.|$)"
)


@dataclass(frozen=True)
class ShardSpec:
    """One shard of an ``index/count`` split."""

    index: int
    count: int

    @property
    def label(self) -> str:
        return f"{self.index}/{self.count}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


@dataclass(frozen=True)
class UnitPartition:
    """The full deterministic partition of one ordered unit enumeration."""

    units: tuple[tuple[str, int], ...]
    count: int
    shards: tuple[tuple[str, ...], ...]
    shard_bytes: tuple[int, ...]
    atom_keys: tuple[str, ...]
    partition_hash: str

    def units_for(self, spec: ShardSpec) -> tuple[str, ...]:
        if spec.count != self.count:
            raise ValueError(
                f"shard spec count {spec.count} does not match partition "
                f"count {self.count}"
            )
        return self.shards[spec.index]


def parse_shard_spec(value: str | None) -> ShardSpec | None:
    """Parse ``"i/N"``. ``None``/empty means "no shard" (the whole run).

    Malformed input is fatal, never a silent whole-run fallback: a driver
    that typos the split would otherwise render everything on every box and
    the merge would only find out afterwards.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _SHARD_RE.match(text)
    if match is None:
        raise ValueError(
            f"malformed {SHARD_ENV}={value!r}; expected 'i/N' with "
            "0 <= i < N (for example '0/2')"
        )
    index = int(match.group(1))
    count = int(match.group(2))
    if count < 1:
        raise ValueError(
            f"malformed {SHARD_ENV}={value!r}; shard count must be >= 1"
        )
    if index >= count:
        raise ValueError(
            f"malformed {SHARD_ENV}={value!r}; shard index must be < count"
        )
    return ShardSpec(index=index, count=count)


def resolve_shard_spec(
    env: Mapping[str, str] | None = None,
) -> ShardSpec | None:
    """Read :data:`SHARD_ENV` from ``env`` (default ``os.environ``)."""
    source = os.environ if env is None else env
    return parse_shard_spec(source.get(SHARD_ENV))


def atom_key(name: str) -> str:
    """Return the layer-atom key for a unit qname.

    ``model.layers.7.self_attn.q_proj`` -> ``model.layers.7``. A unit with no
    recognizable layer index is its own atom.
    """
    match = _LAYER_RE.match(str(name))
    if match is None:
        return str(name)
    return match.group("prefix")


def _normalize_units(
    units: Sequence[tuple[str, int]] | Sequence[Sequence],
) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in units:
        name, nbytes = str(item[0]), int(item[1])
        if nbytes < 0:
            raise ValueError(f"unit {name!r} has negative byte size {nbytes}")
        if name in seen:
            raise ValueError(f"duplicate unit in enumeration: {name!r}")
        seen.add(name)
        out.append((name, nbytes))
    return tuple(out)


def partition_hash(
    units: Sequence[tuple[str, int]],
    count: int,
) -> str:
    """SHA-256 over the exact partition inputs.

    The hash covers the FULL ordered enumeration (name + exact bytes) and the
    shard count, so any shard's stamp proves which enumeration it split and
    the merge tool can recompute the partition without the model.
    """
    normalized = _normalize_units(units)
    payload = json.dumps(
        {
            "schema": SCHEMA,
            "count": int(count),
            "units": [[name, nbytes] for name, nbytes in normalized],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contiguous_min_max_partition(
    weights: Sequence[int],
    count: int,
) -> list[int]:
    """Exact contiguous partition minimizing the heaviest part.

    Returns the ``count + 1`` boundary offsets. Deterministic tie-break: the
    smallest feasible split index wins, so the same inputs always yield the
    same boundaries on any host.
    """
    n = len(weights)
    prefix = [0] * (n + 1)
    for i, weight in enumerate(weights):
        prefix[i + 1] = prefix[i] + int(weight)

    # dp[k][i]: min possible heaviest part when the first i atoms are split
    # into k parts. Empty parts are legal (count may exceed atom count).
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(count + 1)]
    choice = [[0] * (n + 1) for _ in range(count + 1)]
    dp[0][0] = 0.0
    for k in range(1, count + 1):
        for i in range(0, n + 1):
            best = inf
            best_j = 0
            for j in range(0, i + 1):
                if dp[k - 1][j] == inf:
                    continue
                part = prefix[i] - prefix[j]
                candidate = dp[k - 1][j] if dp[k - 1][j] > part else float(part)
                if candidate < best:
                    best = candidate
                    best_j = j
            dp[k][i] = best
            choice[k][i] = best_j

    if dp[count][n] == inf:  # pragma: no cover - unreachable for count >= 1
        raise ValueError("no feasible contiguous partition")

    bounds = [0] * (count + 1)
    bounds[count] = n
    i = n
    for k in range(count, 0, -1):
        j = choice[k][i]
        bounds[k - 1] = j
        i = j
    return bounds


def partition_units(
    units: Sequence[tuple[str, int]],
    count: int,
) -> UnitPartition:
    """Split an ordered unit enumeration into ``count`` contiguous shards."""
    if int(count) < 1:
        raise ValueError(f"shard count must be >= 1, got {count}")
    normalized = _normalize_units(units)
    count = int(count)

    atom_order: list[str] = []
    atom_units: dict[str, list[str]] = {}
    atom_bytes: dict[str, int] = {}
    for name, nbytes in normalized:
        key = atom_key(name)
        if key not in atom_units:
            atom_order.append(key)
            atom_units[key] = []
            atom_bytes[key] = 0
        elif atom_order[-1] != key:
            raise ValueError(
                "unit enumeration is not layer-contiguous: atom "
                f"{key!r} reappears after {atom_order[-1]!r} at unit {name!r}"
            )
        atom_units[key].append(name)
        atom_bytes[key] += nbytes

    weights = [atom_bytes[key] for key in atom_order]
    bounds = _contiguous_min_max_partition(weights, count)

    shards: list[tuple[str, ...]] = []
    shard_bytes: list[int] = []
    for k in range(count):
        keys = atom_order[bounds[k]:bounds[k + 1]]
        names: list[str] = []
        total = 0
        for key in keys:
            names.extend(atom_units[key])
            total += atom_bytes[key]
        shards.append(tuple(names))
        shard_bytes.append(total)

    return UnitPartition(
        units=normalized,
        count=count,
        shards=tuple(shards),
        shard_bytes=tuple(shard_bytes),
        atom_keys=tuple(atom_order),
        partition_hash=partition_hash(normalized, count),
    )


def assert_groups_within_atoms(
    units: Sequence[str],
    group_key,
    *,
    where: str,
) -> None:
    """Refuse an enumeration whose co-rendering groups straddle atoms.

    ``group_key(name)`` returns the caller's real grouping identity (the
    fused-sibling group for the dense render path). A group split across two
    atoms could land on two boxes, and the members' shared NVFP4 global scale
    would be computed over different member sets — the shard's bytes would
    stop being the unsharded run's bytes. Fail closed instead.
    """
    atoms_by_group: dict[str, set[str]] = {}
    for name in units:
        key = group_key(name)
        if key is None:
            continue
        atoms_by_group.setdefault(str(key), set()).add(atom_key(name))
    straddling = sorted(
        group for group, atoms in atoms_by_group.items() if len(atoms) > 1
    )
    if straddling:
        raise ValueError(
            f"{where}: {len(straddling)} co-rendered group(s) straddle "
            f"layer atoms and cannot be unit-sharded; sample={straddling[:5]}"
        )


def shard_stamp(
    partition: UnitPartition,
    spec: ShardSpec,
) -> dict:
    """The provenance block every shard artifact carries."""
    if spec.count != partition.count:
        raise ValueError(
            f"shard spec {spec.label} does not match partition count "
            f"{partition.count}"
        )
    names = partition.shards[spec.index]
    return {
        "schema": SCHEMA,
        "shard": spec.label,
        "shard_index": int(spec.index),
        "shard_count": int(spec.count),
        "partition_hash": partition.partition_hash,
        "unit_names": list(names),
        "unit_count": len(names),
        "shard_bytes": int(partition.shard_bytes[spec.index]),
        "total_units": len(partition.units),
        "total_bytes": int(sum(nbytes for _, nbytes in partition.units)),
        # The complete ordered enumeration travels with every shard so the
        # merge tool can recompute the partition and gate completeness
        # without loading the model.
        "all_units": [[name, nbytes] for name, nbytes in partition.units],
    }


def owed_pairs_stamp(pairs) -> dict:
    """Record the exact ``(qname, format)`` entries this shard set out to render.

    The merge gate needs to know what each shard *owed* before it can call a
    missing entry missing. Reconstructing that downstream from an
    operator-supplied layer config would put the operator's file in the trust
    path: a config that disagrees with the one the shard actually rendered
    under would under-expect, and a unit dropped by the render would merge
    clean. So the shard states its own debt, at the moment it is fixed and
    from the same structure the render loop consumes.
    """
    canonical = sorted(
        {(str(qname), str(fmt).upper()) for qname, fmt in pairs}
    )
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return {
        "owed_pairs": [[qname, fmt] for qname, fmt in canonical],
        "owed_pair_count": len(canonical),
        "owed_pairs_sha256": hashlib.sha256(payload).hexdigest(),
    }


def owed_pairs_from_stamp(stamp: Mapping) -> set[tuple[str, str]] | None:
    """Read back an ``owed_pairs_stamp``, verifying its own digest.

    Returns ``None`` when the stamp predates the field, so a caller can fall
    back to reconstruction and say so.
    """
    raw = stamp.get("owed_pairs")
    if raw is None:
        return None
    pairs = sorted(
        {(str(item[0]), str(item[1]).upper()) for item in raw}
    )
    payload = json.dumps(
        [[qname, fmt] for qname, fmt in pairs], separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    recorded = str(stamp.get("owed_pairs_sha256", ""))
    if recorded and digest != recorded:
        raise ValueError(
            f"unit-shard stamp {stamp.get('shard')!r} owed_pairs do not match "
            f"owed_pairs_sha256 ({digest} != {recorded}); the stamp was "
            "edited after the render."
        )
    if int(stamp.get("owed_pair_count", len(pairs))) != len(pairs):
        raise ValueError(
            f"unit-shard stamp {stamp.get('shard')!r} owed_pair_count "
            f"{stamp.get('owed_pair_count')!r} disagrees with the "
            f"{len(pairs)} pairs it lists."
        )
    return set(pairs)


def host_identity() -> dict:
    """Who rendered this shard. Impure by design; provenance only.

    A distributed render is only as trustworthy as the claim that both boxes
    ran the same wheel on the same architecture, so the merged artifact has
    to be able to name what actually produced each shard's bytes.
    """
    import platform
    import socket

    identity: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    try:
        import torch

        identity["torch_version"] = str(torch.__version__)
        identity["torch_cuda_version"] = str(
            getattr(torch.version, "cuda", None)
        )
        if torch.cuda.is_available():
            identity["cuda_device_name"] = torch.cuda.get_device_name(0)
            identity["cuda_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(0)
            )
    except Exception:  # pragma: no cover - torch is always present in prod
        identity["torch_version"] = None
    return identity


def partition_from_stamp(stamp: Mapping) -> UnitPartition:
    """Rebuild the partition a shard stamp describes."""
    if str(stamp.get("schema")) != SCHEMA:
        raise ValueError(
            f"unexpected unit-shard stamp schema {stamp.get('schema')!r}"
        )
    all_units = stamp.get("all_units")
    if not isinstance(all_units, Sequence) or isinstance(all_units, (str, bytes)):
        raise ValueError("unit-shard stamp is missing its all_units enumeration")
    partition = partition_units(
        [(str(item[0]), int(item[1])) for item in all_units],
        int(stamp["shard_count"]),
    )
    if partition.partition_hash != str(stamp.get("partition_hash")):
        raise ValueError(
            "unit-shard stamp partition_hash does not match its own "
            "enumeration; the stamp is inconsistent"
        )
    return partition
