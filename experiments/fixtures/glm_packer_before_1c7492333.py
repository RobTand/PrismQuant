def _pack_per_expert_into_packed(
    out: dict[str, torch.Tensor],
    *,
    is_per_expert,
    parent_for_projection,
    projection_names_for,
    live_param_shape,
) -> int:
    """Stack per-expert checkpoint tensors into packed 3D live params.

    Some MoE checkpoints store each routed expert's projections separately
    on disk (``…experts.{i}.{proj}.weight``) while the live module exposes a
    single packed parameter per projection group (``…experts.gate_up_proj``,
    a ``[num_experts, …]`` tensor). The install resolver is keyed by the
    *live* parameter names, so the per-expert disk tensors never match and
    the slow fallback walks a non-existent ``experts.{i}`` submodule.

    This bridges the two layouts generically: every structural decision —
    which projections fuse into which packed param, and in what order —
    comes from the supplied callables, which the caller wires from the model
    profile's packed-experts spec. No architecture names appear here. The
    assembled tensor's shape is checked against the live parameter so a
    layout mismatch fails loud instead of silently mis-packing.

    Mutates ``out`` in place: removes consumed per-expert keys and inserts
    the packed keys. Returns the number of packed params produced (0 = the
    checkpoint isn't per-expert, or the live module isn't packed)."""
    # packed_full_name -> {expert_idx -> {projection -> tensor}}
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict))
    consumed: list[str] = []
    for key, t in out.items():
        name = key[:-len(".weight")] if key.endswith(".weight") else key
        if not is_per_expert(name):
            continue
        head, proj = name.rsplit(".", 1)           # head = …experts.{idx}
        experts_path, idx_str = head.rsplit(".", 1)
        if not idx_str.isdigit():
            continue
        parent = parent_for_projection(proj)
        if parent is None:
            continue
        packed_full = f"{experts_path}.{parent}"
        if live_param_shape(packed_full) is None:
            continue  # live module isn't packed for this group — leave as-is
        groups[packed_full][int(idx_str)][proj] = t
        consumed.append(key)
    produced = 0
    for packed_full, by_expert in groups.items():
        parent = packed_full.rsplit(".", 1)[1]
        order = tuple(projection_names_for(parent))
        n_experts = max(by_expert) + 1
        slabs: list[torch.Tensor] = []
        for i in range(n_experts):
            projs = by_expert.get(i)
            if projs is None or any(p not in projs for p in order):
                raise ValueError(
                    f"per-expert pack: {packed_full} missing expert {i} "
                    f"projection(s) {order}")
            if len(order) == 1:
                slabs.append(projs[order[0]])
            else:
                # Fuse projections along the output axis (the transformers
                # packed-FusedMoE convention), then stack experts on a new
                # leading axis. The shape check below is the safety net.
                slabs.append(torch.cat([projs[p] for p in order], dim=0))
        packed = torch.stack(slabs, dim=0).contiguous()
        target = live_param_shape(packed_full)
        if tuple(packed.shape) != tuple(target):
            raise ValueError(
                f"per-expert pack: assembled {packed_full} shape "
                f"{tuple(packed.shape)} != live param {tuple(target)}")
        out[packed_full] = packed
        produced += 1
    for key in consumed:
        out.pop(key, None)
    return produced
