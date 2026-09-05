"""Profile-owned routed-expert discovery shared by cost estimators.

Tensor rank is deliberately absent from the classification contract.  A
target is routed-expert state only when the resolved ``ModelProfile`` assigns
its qname to a packed-expert serving-format group.  That profile accessor
accepts both physical representations used by PrismaQuant: a packed parameter
(``...experts.gate_up_proj``) and an unpacked per-expert ``nn.Linear``
(``...experts.7.gate_proj``).
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
import torch.nn as nn


def resolve_routed_expert_profile(model: nn.Module, profile=None):
    """Resolve one profile for every routed-expert decision in a call."""
    if profile is None:
        from prismaquant.model_profiles import profile_from_model

        profile = profile_from_model(model)
    if profile is None:
        raise RuntimeError(
            "routed-expert classification could not resolve a ModelProfile"
        )
    return profile


def _profile_call(profile, accessor: str, *args):
    method = getattr(profile, accessor, None)
    if not callable(method):
        raise RuntimeError(
            f"profile {type(profile).__name__} cannot classify routed experts: "
            f"missing callable {accessor}()"
        )
    try:
        return method(*args)
    except Exception as exc:
        raise RuntimeError(
            f"profile {type(profile).__name__} could not determine routed-"
            f"expert membership via {accessor}()"
        ) from exc


def _projection_names(profile) -> frozenset[str]:
    """Every expert projection spelling the profile declares.

    The three accessors name different namespaces: live unpacked modules,
    packed-checkpoint projections, and vLLM's canonical FusedMoE scheme
    projections.  Their union is the only accepted vocabulary.
    """
    def declared_names(raw, *, where: str) -> tuple[str, ...]:
        if isinstance(raw, (str, bytes)):
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: expected a collection of projection names"
            )
        try:
            values = tuple(raw)
        except TypeError as exc:
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: expected a collection of projection names"
            ) from exc
        if any(not isinstance(name, str) or not name for name in values):
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: {values!r}"
            )
        return values

    packed = _profile_call(profile, "packed_expert_param_names")
    unpacked = _profile_call(profile, "unpacked_expert_projection_names")
    try:
        packed_names = declared_names(
            packed, where="packed_expert_param_names()"
        )
        names = set(packed_names)
        names.update(declared_names(
            unpacked, where="unpacked_expert_projection_names()"
        ))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"profile {type(profile).__name__} returned malformed routed-"
            "expert projection declarations"
        ) from exc
    for packed_name in packed_names:
        for accessor in (
            "packed_expert_projection_names",
            "vllm_fused_moe_scheme_projection_names",
        ):
            declared = _profile_call(profile, accessor, packed_name)
            names.update(declared_names(
                declared, where=f"{accessor}({packed_name!r})"
            ))
    if not names or any(not name or "." in name for name in names):
        raise RuntimeError(
            f"profile {type(profile).__name__} returned an invalid routed-"
            f"expert projection vocabulary: {sorted(names)!r}"
        )
    return frozenset(names)


@dataclass(frozen=True)
class RoutedExpertMatch:
    """One qname classified by a profile, independent of tensor rank."""

    qname: str
    profile_qname: str
    group_key: str
    projection_name: str
    regex_declared: bool


@dataclass(frozen=True)
class UnpackedExpertLinear:
    """A profile-declared routed expert represented by one ``nn.Linear``."""

    qname: str
    module: nn.Linear
    group_key: str
    unit_qname: str
    expert_id: int
    projection_name: str


class ProfileRoutedExpertClassifier:
    """Strict adapter around the existing ``ModelProfile`` accessors."""

    def __init__(self, profile):
        self.profile = profile
        self.projection_names = _projection_names(profile)
        raw_regex = _profile_call(profile, "per_expert_moe_regex")
        self.per_expert_pattern: re.Pattern[str] | None = None
        if raw_regex is not None:
            if not isinstance(raw_regex, str) or not raw_regex:
                raise RuntimeError(
                    f"profile {type(profile).__name__} returned a malformed "
                    "per_expert_moe_regex()"
                )
            body = raw_regex.removeprefix("re:")
            try:
                self.per_expert_pattern = re.compile(body)
            except re.error as exc:
                raise RuntimeError(
                    f"profile {type(profile).__name__} returned an invalid "
                    "per_expert_moe_regex()"
                ) from exc

    def _candidate_names(self, qname: str) -> tuple[str, ...]:
        recipe = _profile_call(
            self.profile, "live_to_recipe_name", str(qname)
        )
        if not isinstance(recipe, str) or not recipe:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} returned malformed "
                f"live_to_recipe_name({qname!r})={recipe!r}"
            )
        vllm = _profile_call(
            self.profile, "to_vllm_internal_name", recipe
        )
        if not isinstance(vllm, str) or not vllm:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} returned malformed "
                f"to_vllm_internal_name({recipe!r})={vllm!r}"
            )
        return tuple(dict.fromkeys((str(qname), recipe, vllm)))

    def classify(self, qname: str) -> RoutedExpertMatch | None:
        candidates = self._candidate_names(qname)
        grouped: list[tuple[str, str]] = []
        regex_declared = False
        for candidate in candidates:
            if (
                self.per_expert_pattern is not None
                and self.per_expert_pattern.fullmatch(candidate) is not None
            ):
                regex_declared = True
            group = _profile_call(
                self.profile, "packed_expert_format_group", candidate
            )
            if group is None:
                continue
            if not isinstance(group, str) or not group:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} returned malformed "
                    f"packed_expert_format_group({candidate!r})={group!r}"
                )
            grouped.append((candidate, group))

        if not grouped:
            if regex_declared:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} declares {qname!r} "
                    "as a per-expert Linear but cannot assign it a routed-"
                    "expert serving-format group"
                )
            return None

        group_keys = {group for _candidate, group in grouped}
        if len(group_keys) != 1:
            # `grouped` holds groups derived from the THREE spellings of ONE
            # Linear (live / recipe / vLLM-internal), and
            # `packed_expert_format_group` builds its key as
            # f"{parent}::__packed_format__:{projections}" from whichever
            # spelling it was handed. So on any model whose namespaces differ
            # by a prefix -- a multimodal wrapper contributes
            # `model.language_model.` on the checkpoint side and
            # `language_model.model.` on the vLLM side, against a bare
            # `model.` recipe name -- the raw keys ALWAYS differ and this gate
            # fires on a naming artifact rather than a declaration conflict.
            # Compare what actually carries meaning instead: the projection
            # grouping, and the structural position from `layers.` onward.
            # Those still catch a profile that groups one Linear two different
            # ways, or that points two spellings at different layers.
            def _meaning(key: str) -> tuple[str, str]:
                parent, _, grouping = key.partition("::")
                idx = parent.find("layers.")
                return (parent[idx:] if idx != -1 else parent), grouping

            if len({_meaning(group) for _candidate, group in grouped}) != 1:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} assigns conflicting "
                    f"routed-expert groups to {qname!r}: {sorted(group_keys)!r}"
                )
        profile_qname = grouped[0][0]
        projection = profile_qname.rsplit(".", 1)[-1]
        if projection not in self.projection_names:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} grouped {qname!r} as "
                f"a routed expert but projection {projection!r} is absent "
                "from its expert projection accessors"
            )
        return RoutedExpertMatch(
            qname=str(qname),
            profile_qname=profile_qname,
            # Deterministic: the first spelling that produced a group, not an
            # arbitrary element of a set. Every member of one experts module
            # walks the same candidate order, so siblings agree and union-find
            # still fuses them.
            group_key=grouped[0][1],
            projection_name=projection,
            regex_declared=regex_declared,
        )


def profile_declared_unpacked_expert_linears(
    model: nn.Module,
    profile=None,
) -> list[UnpackedExpertLinear]:
    """Return every routed expert represented by a per-expert Linear.

    Numeric expert-id parsing happens only *after* the profile classified the
    qname.  An unfamiliar representation is therefore a declaration error,
    never a name heuristic or a quiet empty result.
    """
    profile = resolve_routed_expert_profile(model, profile)
    classifier = ProfileRoutedExpertClassifier(profile)
    out: list[UnpackedExpertLinear] = []
    for qname, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = classifier.classify(qname)
        if match is None:
            continue
        parts = str(qname).split(".")
        if len(parts) < 3 or not parts[-2].isdigit():
            raise RuntimeError(
                f"profile {type(profile).__name__} classified Linear "
                f"{qname!r} as routed expert state but cannot determine its "
                "expert id from the declared per-expert representation"
            )
        if (
            classifier.per_expert_pattern is not None
            and not match.regex_declared
        ):
            raise RuntimeError(
                f"profile {type(profile).__name__} grouped Linear {qname!r} "
                "as routed expert state but its per_expert_moe_regex() does "
                "not recognize any mapped qname"
            )
        out.append(UnpackedExpertLinear(
            qname=str(qname),
            module=module,
            group_key=match.group_key,
            unit_qname=".".join(parts[:-2]),
            expert_id=int(parts[-2]),
            projection_name=parts[-1],
        ))
    return sorted(out, key=lambda member: member.qname)


@dataclass(frozen=True)
class PackedExpertProjection:
    """One profile-declared 2-D view of a live packed expert parameter.

    ``qname`` is the virtual per-expert module spelling in the live namespace,
    not a Tessera serving-group name. The producer still owns the projection
    from source units to its executed groups and wire containers.
    """

    qname: str
    packed_qname: str
    module_qname: str
    module: nn.Module
    param_name: str
    expert_id: int
    projection_name: str
    weight: torch.Tensor


def profile_declared_packed_expert_projections(
    model: nn.Module, profile=None,
) -> list[PackedExpertProjection]:
    """Split declared packed tensors using the native exporter's exact views.

    This shares source topology, not an encoder or a serving grammar. Views
    retain the live weight device/dtype and do not allocate a second cache.
    """
    from .export_native_compressed import _split_packed_expert_tensor
    from .measure_quant_cost import _enumerate_packed_experts

    profile = resolve_routed_expert_profile(model, profile)
    targets = set(profile_declared_routed_expert_targets(model, profile))
    result = []
    for packed_qname, packed, module_qname, module in _enumerate_packed_experts(
        model, targets, profile,
    ):
        param_name = packed_qname.rsplit(".", 1)[-1]
        for projection, weight in _split_packed_expert_tensor(packed, param_name, profile):
            for expert_id in range(int(weight.shape[0])):
                result.append(PackedExpertProjection(
                    qname=f"{module_qname}.{expert_id}.{projection}",
                    packed_qname=packed_qname, module_qname=module_qname,
                    module=module, param_name=param_name,
                    expert_id=expert_id, projection_name=projection,
                    weight=weight[expert_id],
                ))
    names = [member.qname for member in result]
    if len(set(names)) != len(names):
        raise RuntimeError("profile declares duplicate packed expert projection qnames")
    return sorted(result, key=lambda member: member.qname)


def profile_declared_routed_expert_targets(
    model: nn.Module,
    profile=None,
) -> list[str]:
    """All profile-declared routed targets, packed or unpacked."""
    profile = resolve_routed_expert_profile(model, profile)
    classifier = ProfileRoutedExpertClassifier(profile)
    out: set[str] = set()
    for qname, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if classifier.classify(qname) is not None:
                out.add(str(qname))
            continue
        for attr, _param in module.named_parameters(recurse=False):
            full = f"{qname}.{attr}" if qname else str(attr)
            if classifier.classify(full) is not None:
                out.add(full)
    return sorted(out)


__all__ = [
    "ProfileRoutedExpertClassifier",
    "RoutedExpertMatch",
    "UnpackedExpertLinear",
    "PackedExpertProjection",
    "profile_declared_packed_expert_projections",
    "profile_declared_routed_expert_targets",
    "profile_declared_unpacked_expert_linears",
    "resolve_routed_expert_profile",
]


def packed_expert_col_weights(col_weights, members_by_target, profile):
    """Per-expert imatrix vectors -> the ``(E, 1, in)`` stack entry each packed
    target needs, returned as a NEW mapping (the per-expert entries survive for
    the render identity, which is keyed by them).

    It lived in the codebook exporter until 2026-09-02 (as
    ``export_nvfp4_cb_streaming._packed_expert_col_weights``) and moved here
    when that lane was retired: the rule is about how a PACKED EXPERT stack
    pools its members' imatrix vectors, which is a routed-expert fact and not
    a property of any one container.

    A FUSED parent (``gate_up_proj`` = gate then up) has ONE input, so its two
    projections' vectors are two samples of the same per-column mean-square and
    are pooled by averaging. They are not identical in practice only because
    the probe caches each Linear's inputs separately under a row limit
    (DSv4-Flash layer 0: max |gate-up| ~ 0.3 against a vector norm ~ 4.6).
    Weighting one projection by the other's sample would be the actual error;
    per-row vectors cannot be expressed here at all, since the pack broadcasts
    ``(E, 1, in)`` across the whole stack."""
    out = dict(col_weights)
    for packed_qname, member_qnames in members_by_target.items():
        if packed_qname in out:
            continue
        projections = tuple(dict.fromkeys(
            projection for projection, _expert_id in member_qnames
        ))
        experts = sorted({e for _p, e in member_qnames})
        rows = []
        for e in experts:
            vecs = []
            for proj in projections:
                q = member_qnames[(proj, e)]
                if q not in col_weights:
                    raise ValueError(
                        f"{packed_qname}: CB stack member {q!r} has no "
                        "col_weights entry (no silent RTN)")
                vecs.append(torch.as_tensor(col_weights[q])
                            .reshape(-1).to(torch.float32))
            widths = {int(v.numel()) for v in vecs}
            if len(widths) != 1:
                raise ValueError(
                    f"{packed_qname}: expert {e} imatrix widths disagree "
                    f"across the fused projections {projections}: {widths}")
            rows.append(torch.stack(vecs).mean(dim=0) if len(vecs) > 1
                        else vecs[0])
        out[packed_qname] = torch.stack(rows).unsqueeze(1).contiguous()
    return out
