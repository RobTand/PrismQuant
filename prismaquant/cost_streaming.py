"""Shared layer-streaming execution adapter for cost stages.

This is intentionally a thin consumer of :class:`StreamingContext`.  It does
not own weights or maintain another cache: decoder residency, prefetch, and
unload all go through the existing streaming-model machinery.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Iterator

import torch

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)


STREAMED_MODEL_IDENTITY_SCHEMA = "prismaquant.streamed_model.identity.v1"
STREAMED_MODEL_IDENTITY_CACHE_SCHEMA = (
    "prismaquant.streamed_model.identity_cache.v1"
)
STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA = (
    "prismaquant.streamed_model.portable_content.v1"
)
_STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS = frozenset({
    "_name_or_path",
    "transformers_version",
})


@dataclass
class StreamedForwardBoundaries:
    """One exact source-model forward, cut at decoder-layer boundaries."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: object
    attention_mask: object
    activations_cpu: list[torch.Tensor]
    shared_pass_state: object


class StreamedCausalLM:
    """Causal-LM forward adapter over an existing ``StreamingContext``.

    ``pin_layer_for_qname`` keeps exactly one decoder layer installed while a
    caller temporarily mutates and restores a serving unit in it.  All other
    layers continue to stream through the context's cache.  This is the seam
    used by empirical expert KL; AURA additionally consumes the explicit
    boundary/isolated-layer methods for its streamed adjoint.
    """

    def __init__(
        self,
        context,
        profile,
        *,
        prefetch_lookahead: int = 2,
        require_prefetched_residency: bool = False,
    ):
        if type(require_prefetched_residency) is not bool:
            raise TypeError(
                "require_prefetched_residency must be a bool"
            )
        self.context = context
        self.model = context.model
        self.base_model = context.base_model
        self.layers = context.layers
        self.layers_prefix = str(context.layers_prefix)
        self.num_layers = int(context.num_layers)
        self.device = torch.device(context.device)
        self.dtype = context.dtype
        self.profile = profile
        self.prefetch_lookahead = max(0, int(prefetch_lookahead))
        self.require_prefetched_residency = require_prefetched_residency
        self._pinned_layer: int | None = None

    def layer_index_for_qname(self, qname: str) -> int:
        match = re.match(
            rf"^{re.escape(self.layers_prefix)}([0-9]+)(?:\.|$)",
            str(qname),
        )
        if match is None:
            raise RuntimeError(
                f"streamed cost unit {qname!r} is not under decoder prefix "
                f"{self.layers_prefix!r}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.num_layers:
            raise RuntimeError(
                f"streamed cost unit {qname!r} resolved invalid layer {layer}"
            )
        return layer

    @contextmanager
    def pin_layer(self, layer: int) -> Iterator[None]:
        layer = int(layer)
        if self._pinned_layer is not None:
            raise RuntimeError(
                f"streamed cost already pins layer {self._pinned_layer}; "
                f"cannot also pin layer {layer}"
            )
        self.context.install(layer)
        self._pinned_layer = layer
        try:
            yield
        finally:
            self._pinned_layer = None
            self.context.unload(layer)

    @contextmanager
    def pin_layer_for_qname(self, qname: str) -> Iterator[None]:
        with self.pin_layer(self.layer_index_for_qname(qname)):
            yield

    def _head(self):
        name = str(self.profile.lm_head_name())
        try:
            return self.model.get_submodule(name)
        except (AttributeError, KeyError):
            head = getattr(self.model, "lm_head", None)
            if head is None:
                raise RuntimeError(
                    f"streamed cost could not resolve profile lm_head {name!r}"
                )
            return head

    def _prepare(self, input_ids: torch.Tensor):
        ids = input_ids.to(self.device)
        position_ids = torch.arange(
            ids.size(-1), device=self.device
        ).unsqueeze(0)
        hidden = self.base_model.embed_tokens(ids).to(self.dtype)
        position_embeddings = _compute_position_embeddings(
            self.base_model, hidden, position_ids, self.profile
        )
        attention_mask = _compute_attention_mask(
            self.base_model, hidden, position_ids
        )
        hidden = self.profile.expand_hidden_for_layers(
            hidden, self.base_model
        )
        return ids, position_ids, hidden, position_embeddings, attention_mask

    def _call(self, layer: int, hidden: torch.Tensor, *, batch, pass_state):
        return _call_layer(
            self.layers[layer],
            hidden,
            position_embeddings=batch.position_embeddings,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            **self.profile.extra_layer_kwargs(input_ids=batch.input_ids),
            pass_state=pass_state,
        )

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.profile.collapse_hidden_after_layers(
            hidden, self.base_model
        )
        norm = _get_final_norm(self.base_model)
        if norm is not None:
            hidden = norm(hidden)
        return self._head()(hidden)

    def capture_boundaries(
        self, input_ids: torch.Tensor
    ) -> StreamedForwardBoundaries:
        """Stream a no-grad source forward and retain only boundary acts."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        batch.activations_cpu.append(hidden.detach().to("cpu"))
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                with torch.no_grad():
                    hidden = self._call(
                        layer, hidden, batch=batch, pass_state=pass_state
                    )
                batch.activations_cpu.append(hidden.detach().to("cpu"))
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        batch.shared_pass_state = self.profile.capture_forward_pass_state(
            pass_state
        )
        return batch

    def isolated_layer(
        self,
        batch: StreamedForwardBoundaries,
        layer: int,
        hidden: torch.Tensor,
        *,
        pass_state: dict | None,
    ) -> torch.Tensor:
        return self._call(layer, hidden, batch=batch, pass_state=pass_state)

    def tail_logits(
        self, batch: StreamedForwardBoundaries, hidden: torch.Tensor
    ) -> torch.Tensor:
        return self._finish(hidden)

    def schedule_reverse_prefetch(self, layer: int):
        """Prefetch the next layer in this runner's reverse traversal.

        This is the reverse-direction twin of the forward loops' explicit
        ``schedule_prefetch(layer + lookahead)`` call.  Residency remains
        owned by the existing :class:`StreamingContext` / ``LayerCache``;
        this method only supplies the traversal direction so reverse AURA
        can overlap the next source-layer read with the current layer's
        render and backward work.
        """
        target = int(layer) - self.prefetch_lookahead
        if self.prefetch_lookahead <= 0 or target < 0:
            return None
        return self.context.schedule_prefetch(target)

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any):
        """Run an end-to-end no-cache forward while streaming body layers."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                hidden = self._call(
                    layer, hidden, batch=batch, pass_state=pass_state
                )
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        return SimpleNamespace(logits=self.tail_logits(batch, hidden))

    def shutdown(self) -> None:
        self.context.shutdown()


def build_streamed_causal_lm(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offload_folder: str,
    profile,
    cache_headroom_gb: float | None = None,
    max_cache_slots: int | None = None,
    prefetch_workers: int | str | None = None,
    prefetch_min_available_gb: float | str | None = None,
    prefetch_lookahead: int = 2,
    require_prefetched_residency: bool = False,
    attn_implementation: str | None = None,
) -> StreamedCausalLM:
    """Build the repository's existing streaming context and wrap it."""
    from prismaquant.streaming_model import _build_streaming_context

    context = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=offload_folder,
        cache_headroom_gb=cache_headroom_gb,
        max_cache_slots=max_cache_slots,
        prefetch_workers=prefetch_workers,
        prefetch_min_available_gb=prefetch_min_available_gb,
        log_prefix="[cost-streaming]",
        attn_implementation=attn_implementation,
    )
    effective_lookahead = max(0, int(prefetch_lookahead))
    if context.max_cache_slots is not None:
        effective_lookahead = min(
            effective_lookahead,
            max(0, context.max_cache_slots - 1),
        )
    return StreamedCausalLM(
        context,
        profile,
        prefetch_lookahead=effective_lookahead,
        require_prefetched_residency=require_prefetched_residency,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _streamed_identity_stat_fingerprint(path: Path) -> dict[str, object]:
    """Return the mutation-sensitive local cache key for one source shard."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        # Unlike mtime, ctime cannot be restored with utime after a same-size
        # rewrite.  Matching all six fields makes a previously computed
        # content SHA safe to reuse without rereading a multi-hundred-GB
        # checkpoint.
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _local_checkpoint_shards(
    source_model: str | Path,
) -> tuple[dict[str, str] | None, list[Path] | None]:
    """Resolve the exact safetensors file set consumed by a local checkpoint.

    The streaming model omits auxiliary decoder namespaces it does not execute
    (DSv4's MTP towers are one example), while the exporter copies those
    tensors byte-verbatim.  The Hugging Face index is therefore the authority
    for complete source-byte coverage, not only ``context.weight_shard``.
    """
    root = Path(source_model)
    if not root.is_dir():
        return None, None
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"streamed model identity cannot read {index_path}"
            ) from exc
        weight_map = payload.get("weight_map") if isinstance(
            payload, dict
        ) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"streamed model identity requires a non-empty weight_map in "
                f"{index_path}"
            )
        canonical_map: dict[str, str] = {}
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise RuntimeError(
                    f"streamed model identity found an invalid tensor name in "
                    f"{index_path}"
                )
            if (
                not isinstance(shard_name, str)
                or not shard_name
                or Path(shard_name).name != shard_name
            ):
                raise RuntimeError(
                    f"streamed model identity found an unsafe shard name "
                    f"{shard_name!r} in {index_path}"
                )
            canonical_map[tensor_name] = shard_name
            shard_names.add(shard_name)
        shard_paths = sorted(
            ((root / name).resolve() for name in shard_names), key=str
        )
        missing = [str(path) for path in shard_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "streamed model identity checkpoint index references missing "
                f"shards: {missing[:8]}"
            )
        return dict(sorted(canonical_map.items())), shard_paths
    single = root / "model.safetensors"
    if single.is_file():
        return None, [single.resolve()]
    return None, None


SOURCE_CHECKPOINT_IDENTITY_SCHEMA = (
    "prismaquant.source_checkpoint.identity.v1"
)
# The non-shard files the standalone export reads from the checkpoint root:
# `config.json` through `stage_text_only`/`AutoConfig` (streaming_model.py:102
# and the exporter's skeleton build) and the index through
# `_local_checkpoint_shards` and export_native_compressed.py:6092.  A file that
# is absent contributes no row, so one appearing later is a different source.
SOURCE_CHECKPOINT_METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
)
# ... plus every `*.py` at the checkpoint root, because a `trust_remote_code`
# checkpoint executes modules from there to build the skeleton (MiniMax-M2
# ships `configuration_minimax_m2.py` and `modeling_minimax_m2.py`;
# DeepSeek-V4 uses the same pattern), discovered per call rather than named
# here.  Every one of
# them is bound, whether or not `auto_map` names it: binding a module nobody
# imports can cost a false refusal, naming only some can cost a false
# admission.
SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA = (
    "prismaquant.source_checkpoint.digest_cache.v1"
)


def _read_source_checkpoint_digest_cache(
    cache_path: Path,
) -> dict[str, dict[str, object]]:
    """Digests keyed by the six-field stat fingerprint of the file they cover.

    A corrupt or foreign cache is not an error: it simply reuses nothing, and
    every shard is hashed. The cache can only ever make the identity CHEAPER,
    never different -- the fingerprint it keys on includes ``ctime_ns``, which
    ``utime`` cannot restore after an in-place same-size rewrite.
    """
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA
    ):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    reusable: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        digest = str(entry.get("sha256", "")).lower()
        if (
            not isinstance(fingerprint, dict)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            continue
        reusable[canonical_fingerprint_key(fingerprint)] = {
            "fingerprint": fingerprint,
            "sha256": digest,
        }
    return reusable


def canonical_fingerprint_key(fingerprint: dict[str, object]) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def build_source_checkpoint_identity(
    source_model: str | Path,
    *,
    extra_shard_paths: object = (),
    digest_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Content identity of the exact safetensors byte set a run consumes.

    This is the runner-free half of :func:`build_streamed_model_identity`, for
    consumers that hold a checkpoint path rather than a live streaming runner
    -- the standalone compressed-tensors export resume cache is the first
    (PrismaQuant #340). It binds file CONTENT, so a same-size, same-header
    value edit is a different source, and the same bytes under a different
    directory are the same source.

    The weight bytes are not the whole source.
    :data:`SOURCE_CHECKPOINT_METADATA_FILES` names the non-shard files the
    export reads from the checkpoint root, and their sha256 is folded into
    the same ``content_sha256``: ``config.json`` decides the skeleton the
    payloads were quantized against -- the dtype map, ``tie_word_embeddings``,
    any ``quantization_config``, the layer counts -- and
    ``model.safetensors.index.json`` decides which shard each tensor is read
    from.  Every ``*.py`` at the checkpoint root joins them, because a
    ``trust_remote_code`` checkpoint executes modules from there to build the
    skeleton; all of them are bound rather than only those ``auto_map`` names,
    which can refuse falsely but never admit falsely.
    Editing any of them changes what a replayed ``layer_NNN.pt`` means while
    every shard byte stays identical.  They are kilobytes, so they are hashed
    on every call rather than cached.

    ``digest_cache_path`` makes the read happen once per (file, machine): a
    shard whose complete stat fingerprint still matches reuses its recorded
    digest instead of rereading that shard. Discovery, metadata hashing,
    digest-cache JSON handling and identity construction still run. Without
    this cache, every call hashes every shard.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    root = Path(source_model)
    _, indexed_shards = _local_checkpoint_shards(source_model)
    shard_paths = {Path(path).resolve() for path in (indexed_shards or ())}
    for path in extra_shard_paths or ():
        shard_paths.add(Path(path).resolve())
    missing = sorted(str(path) for path in shard_paths if not path.is_file())
    if missing:
        raise RuntimeError(
            f"source checkpoint identity references missing shards: "
            f"{missing[:8]}"
        )
    if not shard_paths:
        raise RuntimeError(
            f"source checkpoint identity found no safetensors shards under "
            f"{root}; refusing to stamp an unidentified source"
        )
    ordered = sorted(shard_paths, key=str)
    # Name each shard by its position INSIDE the checkpoint. A Hugging Face
    # snapshot dir holds symlinks into `blobs/`, so the resolved path is named
    # by an LFS hash; naming shards by that would make the SAME checkpoint
    # reached through a snapshot and through a plain directory two different
    # sources. Recover the in-checkpoint spelling from the directory listing.
    name_by_resolved: dict[str, str] = {}
    if root.is_dir():
        for entry in root.rglob("*.safetensors"):
            try:
                name_by_resolved.setdefault(
                    str(entry.resolve()), str(entry.relative_to(root))
                )
            except (OSError, ValueError):
                continue

    reusable = (
        _read_source_checkpoint_digest_cache(Path(digest_cache_path))
        if digest_cache_path is not None
        and Path(digest_cache_path).is_file()
        else {}
    )

    entries: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    for path in ordered:
        fingerprint = _streamed_identity_stat_fingerprint(path)
        cached = reusable.get(canonical_fingerprint_key(fingerprint))
        digest = str(cached["sha256"]) if cached is not None else None
        if digest is None:
            digest = _file_sha256(path)
            if _streamed_identity_stat_fingerprint(path) != fingerprint:
                raise RuntimeError(
                    f"source checkpoint shard changed while hashing: {path}"
                )
        entries.append({"fingerprint": fingerprint, "sha256": digest})
        # Relocating a checkpoint does not change its bytes, so the identity
        # is never keyed on the absolute path.
        name = name_by_resolved.get(str(path))
        if name is None:
            try:
                name = str(path.relative_to(root.resolve()))
            except ValueError:
                name = path.name
        shards.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    shards.sort(key=lambda row: (str(row["name"]), str(row["sha256"])))

    # The non-shard files the export reads.  Kilobytes each, so no digest
    # cache: hashing them costs less than deciding not to.
    metadata_names = list(SOURCE_CHECKPOINT_METADATA_FILES)
    if root.is_dir():
        # `trust_remote_code` checkpoints build their skeleton from modules at
        # the checkpoint root, so those are read bytes too.  All of them, not
        # only the ones `auto_map` names: over-binding refuses falsely, and
        # under-binding admits falsely.
        metadata_names += sorted(
            path.name for path in root.glob("*.py") if path.is_file()
        )
    metadata: list[dict[str, object]] = []
    for name in metadata_names:
        path = root / name
        if not path.is_file():
            continue
        fingerprint = _streamed_identity_stat_fingerprint(path)
        digest = _file_sha256(path)
        if _streamed_identity_stat_fingerprint(path) != fingerprint:
            raise RuntimeError(
                f"source checkpoint metadata changed while hashing: {path}"
            )
        metadata.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    metadata.sort(key=lambda row: str(row["name"]))

    identity = {
        "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
        "shards": shards,
        "metadata": metadata,
        "content_sha256": canonical_json_sha256(
            {
                "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
                "shards": shards,
                "metadata": metadata,
            },
            where="source checkpoint content identity",
        ),
    }

    if digest_cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        try:
            atomic_write_bytes(
                Path(digest_cache_path),
                json.dumps(
                    {
                        "schema": SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA,
                        "entries": entries,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            )
        except OSError:
            # The digest cache is an optimization. A read-only or full cache
            # directory costs a re-read next time; it never changes identity.
            pass
    return identity


def _read_streamed_model_identity_cache(
    cache_path: Path,
    *,
    source_model: str,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} is corrupt; "
            "refusing identity reuse"
        ) from exc
    if (
        not isinstance(cached, dict)
        or cached.get("schema") != STREAMED_MODEL_IDENTITY_CACHE_SCHEMA
        or cached.get("source") != str(source_model)
    ):
        raise RuntimeError(
            f"streamed model identity cache {cache_path} does not bind source "
            f"{source_model!r}"
        )
    identity = validate_streamed_model_identity(
        cached.get("identity"), where="streamed model identity cache"
    )
    return cached, identity


def build_streamed_model_identity(
    runner: StreamedCausalLM,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Hash the complete checkpoint backing a streamed cost run.

    End-to-end KL/adjoint values depend on every body/head/norm weight, so a
    path, index, or target-unit hash is insufficient.  This hashes each unique
    source shard exactly once and folds those digests together with the live
    checkpoint-key map and resolved config.  It is an initialization integrity
    pass, not a residency mechanism; decoder execution still uses the existing
    streaming cache.
    """
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    config = getattr(runner.model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    if "_name_or_path" in config_dict:
        # The multi-shard staging workaround loads the model from a fresh
        # mkdtemp copy each launch, so the loaded config's _name_or_path is
        # a random per-launch path; keying the identity on it makes every
        # relaunch refuse its own journal ("identity mismatch ... refusing
        # reuse or recompute"). The shard digests below are the real
        # identity — pin the path field to the caller's canonical source.
        config_dict["_name_or_path"] = str(source_model)
    mapping = {
        str(live): str(checkpoint)
        for live, checkpoint in sorted(runner.context.weight_ckpt.items())
    }
    runner_shard_paths = {
        Path(path).resolve()
        for path in runner.context.weight_shard.values()
    }
    checkpoint_weight_map, checkpoint_shard_paths = (
        _local_checkpoint_shards(source_model)
    )
    shard_paths = sorted(
        runner_shard_paths | set(checkpoint_shard_paths or ()), key=str
    )
    if not shard_paths:
        raise RuntimeError(
            "streamed model identity found no source checkpoint shards"
        )
    fingerprints = [
        _streamed_identity_stat_fingerprint(path) for path in shard_paths
    ]
    cache_path = Path(identity_cache_path) if identity_cache_path else None
    cached: dict[str, object] | None = None
    cached_identity: dict[str, object] | None = None
    if cache_path is not None and cache_path.is_file():
        cached, cached_identity = _read_streamed_model_identity_cache(
            cache_path, source_model=str(source_model)
        )
        if cached.get("fingerprints") == fingerprints:
            if (
                cached_identity.get("config") == canonical_json(
                    config_dict, where="streamed model config"
                )
                and cached_identity.get("weight_map") == mapping
                and cached_identity.get("checkpoint_weight_map")
                == checkpoint_weight_map
            ):
                return cached_identity

    # A schema-valid old cache may cover only the executable decoder shards.
    # Reuse each digest whose complete stat fingerprint still matches, and
    # hash only newly covered files (for DSv4 this upgrades 45 cached body
    # shards by reading the three MTP shards, rather than rereading 156 GB).
    reusable_sha: dict[str, str] = {}
    if cached is not None and cached_identity is not None:
        cached_fingerprints = cached.get("fingerprints")
        cached_shards = cached_identity.get("shards")
        if isinstance(cached_fingerprints, list) and isinstance(
            cached_shards, list
        ):
            cached_fp_by_path = {
                str(row.get("path")): row
                for row in cached_fingerprints
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            cached_shard_by_path = {
                str(row.get("path")): row
                for row in cached_shards
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            for fingerprint in fingerprints:
                path_key = str(fingerprint["path"])
                prior_fp = cached_fp_by_path.get(path_key)
                prior_shard = cached_shard_by_path.get(path_key)
                if (
                    prior_fp == fingerprint
                    and isinstance(prior_shard, dict)
                    and prior_shard.get("size") == fingerprint["size"]
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(prior_shard.get("sha256", "")).lower(),
                    )
                ):
                    reusable_sha[path_key] = str(
                        prior_shard["sha256"]
                    ).lower()

    shards: list[dict[str, object]] = []
    for path, fingerprint in zip(shard_paths, fingerprints, strict=True):
        path_key = str(path.resolve())
        digest = reusable_sha.get(path_key)
        if digest is None:
            digest = _file_sha256(path)
            if _streamed_identity_stat_fingerprint(path) != fingerprint:
                raise RuntimeError(
                    f"source checkpoint shard changed while hashing: {path}"
                )
        shards.append({
            "path": path_key,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    value_bearing = {
        "config": canonical_json(config_dict, where="streamed model config"),
        "weight_map": mapping,
        "shards": shards,
    }
    if checkpoint_weight_map is not None:
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    identity = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": str(source_model),
        "resolved_commit": getattr(config, "_commit_hash", None),
        "content_sha256": canonical_json_sha256(
            value_bearing, where="streamed model content identity"
        ),
        **value_bearing,
    }
    if cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        atomic_write_bytes(
            cache_path,
            json.dumps(
                {
                    "schema": STREAMED_MODEL_IDENTITY_CACHE_SCHEMA,
                    "source": str(source_model),
                    "fingerprints": fingerprints,
                    "identity": identity,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
    return identity


def validate_streamed_model_identity(
    identity: object, *, where: str
) -> dict[str, object]:
    """Require a value-bearing full-checkpoint identity, never a name stamp."""
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if not isinstance(identity, Mapping):
        raise RuntimeError(
            f"{where} requires a full streamed model identity object"
        )
    if identity.get("schema") != STREAMED_MODEL_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"{where} requires model identity schema "
            f"{STREAMED_MODEL_IDENTITY_SCHEMA!r}"
        )
    digest = str(identity.get("content_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(
            f"{where} requires exact model content_sha256"
        )
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"{where} requires source shard content identities")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise RuntimeError(f"{where} model shard {index} is malformed")
        shard_digest = str(shard.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", shard_digest) is None:
            raise RuntimeError(
                f"{where} model shard {index} lacks content sha256"
            )
    canonical = canonical_json(identity, where=f"{where} model identity")
    value_bearing = {
        "config": canonical.get("config"),
        "weight_map": canonical.get("weight_map"),
        "shards": canonical.get("shards"),
    }
    if "checkpoint_weight_map" in canonical:
        checkpoint_weight_map = canonical.get("checkpoint_weight_map")
        if not isinstance(checkpoint_weight_map, dict) or not all(
            isinstance(name, str)
            and name
            and isinstance(shard, str)
            and shard
            for name, shard in checkpoint_weight_map.items()
        ):
            raise RuntimeError(
                f"{where} model checkpoint_weight_map is malformed"
            )
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    expected = canonical_json_sha256(
        value_bearing, where=f"{where} model content identity"
    )
    if digest != expected:
        raise RuntimeError(
            f"{where} model content_sha256 does not match its source shard "
            "identity"
        )
    return canonical


def canonical_streamed_model_semantic_config(
    config: object,
    *,
    where: str = "streamed model config",
) -> dict[str, object]:
    """Return config semantics without host/runtime provenance fields.

    ``PretrainedConfig.to_dict()`` records the path it was loaded from and the
    installed Transformers version.  A text-only staging directory therefore
    makes two executions of the same checkpoint byte-distinct.  Those values
    remain in the v1 host-local identity for backward compatibility, but they
    cannot participate in a cross-host content join.  Strip them recursively
    so composed configs cannot reintroduce the same provenance below the top
    level.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json

    canonical = canonical_json(config, where=where)
    if not isinstance(canonical, dict):
        raise RuntimeError(f"{where} must be a JSON mapping")

    def _strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): _strip(item)
                for key, item in value.items()
                if key not in _STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    stripped = _strip(canonical)
    assert isinstance(stripped, dict)
    return stripped


def portable_streamed_model_content_identity(
    identity: object,
    *,
    where: str = "streamed model portable content identity",
) -> dict[str, object]:
    """Derive one path-neutral digest from a complete v1 local identity.

    The serialized v1 ``content_sha256`` intentionally remains unchanged: it
    binds absolute shard paths and the exact runtime config stored in an
    identity cache.  This additive projection is derivable from old caches and
    is the value suitable for comparing independently built caches on separate
    hosts.  Local cache validation must still happen before callers use it.
    """
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    canonical = validate_streamed_model_identity(identity, where=where)
    config = canonical_streamed_model_semantic_config(
        canonical.get("config"), where=f"{where} config",
    )

    def _string_map(value: object, *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise RuntimeError(f"{where} requires nonempty {field}")
        result: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            if (
                not isinstance(raw_key, str)
                or not raw_key
                or not isinstance(raw_value, str)
                or not raw_value
            ):
                raise RuntimeError(f"{where} {field} is malformed")
            result[raw_key] = raw_value
        return dict(sorted(result.items()))

    weight_map = _string_map(canonical.get("weight_map"), field="weight_map")
    checkpoint_weight_map = _string_map(
        canonical.get("checkpoint_weight_map"),
        field="checkpoint_weight_map",
    )
    raw_shards = canonical.get("shards")
    assert isinstance(raw_shards, list)  # validated above
    shards: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{where} shard {index} is malformed")
        path = raw.get("path")
        size = raw.get("size")
        sha256 = str(raw.get("sha256", "")).lower()
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"{where} shard {index} has no path")
        name = Path(path).name
        if (
            not name
            or name in seen_names
            or type(size) is not int
            or int(size) < 1
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise RuntimeError(
                f"{where} shard {index} has no unique basename/size/content"
            )
        seen_names.add(name)
        shards.append({"name": name, "size": int(size), "sha256": sha256})
    shards.sort(key=lambda row: str(row["name"]))

    body = {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "source_identity_schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "config": config,
        "weight_map": weight_map,
        "checkpoint_weight_map": checkpoint_weight_map,
        "shards": shards,
    }
    return {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "portable_content_sha256": canonical_json_sha256(
            body, where=where,
        ),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def compact_streamed_model_identity(
    identity: object,
    *,
    where: str = "streamed model identity",
) -> dict[str, object]:
    """Return the compact, value-bearing identity stored in an artifact.

    The full identity can be several MiB because it carries the complete
    tensor-to-shard map and one SHA-256 per source shard.  ``content_sha256``
    already binds those fields; the compact form retains coverage counts so a
    partial checkpoint identity cannot be mistaken for the full source.
    """

    canonical = validate_streamed_model_identity(identity, where=where)
    shards = canonical.get("shards")
    checkpoint_weight_map = canonical.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not isinstance(
        checkpoint_weight_map, dict
    ):
        raise RuntimeError(
            f"{where} does not attest a complete indexed checkpoint"
        )
    return {
        "schema": canonical.get("schema"),
        "content_sha256": canonical.get("content_sha256"),
        "resolved_commit": canonical.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def validate_cached_streamed_model_identity(
    source_model: str | Path,
    identity_cache_path: str | Path,
    *,
    require_complete_checkpoint: bool = True,
) -> dict[str, object]:
    """Validate a cached full-checkpoint identity without rereading weights.

    Exact per-shard SHA-256 values remain valid only while every mutation-
    sensitive stat fingerprint matches.  For a local indexed checkpoint the
    validator also requires coverage of the complete index shard set (not just
    the decoder shards loaded by a calibration runner) and binds the complete
    tensor-to-shard map.  This is the cheap, fail-closed handoff used before a
    large streaming export.
    """
    source = str(source_model)
    cache_path = Path(identity_cache_path)
    cached, identity = _read_streamed_model_identity_cache(
        cache_path, source_model=source
    )
    fingerprints = cached.get("fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} has no fingerprints"
        )
    fingerprint_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(fingerprints):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity cache fingerprint {index} is malformed"
            )
        path = str(Path(row["path"]).resolve())
        if path in fingerprint_by_path:
            raise RuntimeError(
                f"streamed model identity cache repeats shard path {path}"
            )
        fingerprint_by_path[path] = row

    identity_shards = identity.get("shards")
    assert isinstance(identity_shards, list)  # validated above
    shard_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(identity_shards):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity shard {index} has no exact path"
            )
        path = str(Path(row["path"]).resolve())
        if path in shard_by_path:
            raise RuntimeError(
                f"streamed model identity repeats shard path {path}"
            )
        shard_by_path[path] = row
    if set(shard_by_path) != set(fingerprint_by_path):
        raise RuntimeError(
            "streamed model identity cache fingerprint coverage differs from "
            "its value-bearing shard identity"
        )

    checkpoint_weight_map, checkpoint_paths = _local_checkpoint_shards(source)
    if require_complete_checkpoint:
        if checkpoint_paths is None:
            raise RuntimeError(
                "complete streamed model identity validation requires a local "
                "safetensors checkpoint"
            )
        expected_paths = {str(path.resolve()) for path in checkpoint_paths}
        if set(shard_by_path) != expected_paths:
            missing = sorted(expected_paths - set(shard_by_path))
            extra = sorted(set(shard_by_path) - expected_paths)
            raise RuntimeError(
                "streamed model identity does not cover the complete source "
                f"checkpoint: missing={missing[:8]}, extra={extra[:8]}"
            )
        if checkpoint_weight_map is not None and identity.get(
            "checkpoint_weight_map"
        ) != checkpoint_weight_map:
            raise RuntimeError(
                "streamed model identity tensor-to-shard map differs from the "
                "current source checkpoint index"
            )

    # The shard/index fingerprints above do not cover config.json.  Recreate
    # the same text-only Transformers config used by the streaming runner and
    # compare its semantic JSON to the config carried by the cached content
    # identity.  `_name_or_path` is host-local provenance and
    # `transformers_version` belongs to the separately pinned runtime image;
    # neither is model semantics.  All other fields must agree exactly, so a
    # same-shape change such as rope scaling cannot reuse old source hashes.
    try:
        from transformers import AutoConfig

        from prismaquant.sensitivity_probe import stage_text_only

        config_path = Path(source) / "config.json"
        config_before = _streamed_identity_stat_fingerprint(config_path)
        staged = stage_text_only(source)
        live_config = canonical_streamed_model_semantic_config(
            AutoConfig.from_pretrained(
                staged, trust_remote_code=True, local_files_only=True,
            ).to_dict(),
            where="live streamed model config",
        )
        cached_config = canonical_streamed_model_semantic_config(
            identity.get("config"), where="cached streamed model config",
        )
        if not isinstance(live_config, dict) or not isinstance(
            cached_config, dict
        ):
            raise TypeError("streamed model config is not a mapping")
        config_after = _streamed_identity_stat_fingerprint(config_path)
    except Exception as exc:
        raise RuntimeError(
            "streamed model identity cannot validate the live source config"
        ) from exc
    if config_before != config_after:
        raise RuntimeError(
            "streamed model identity source config changed while validating"
        )
    if live_config != cached_config:
        changed = sorted(
            key for key in set(live_config) | set(cached_config)
            if live_config.get(key) != cached_config.get(key)
        )
        raise RuntimeError(
            "streamed model identity live config differs from its cached "
            f"content identity: changed={changed[:12]}"
        )

    for path_key, expected in fingerprint_by_path.items():
        path = Path(path_key)
        if not path.is_file():
            raise RuntimeError(
                f"streamed model identity source shard is missing: {path}"
            )
        observed = _streamed_identity_stat_fingerprint(path)
        if observed != expected:
            raise RuntimeError(
                "streamed model identity source shard stat drifted; refusing "
                f"cached content SHA for {path}"
            )
        shard = shard_by_path[path_key]
        if shard.get("size") != observed["size"]:
            raise RuntimeError(
                f"streamed model identity shard size disagrees for {path}"
            )
    return identity
