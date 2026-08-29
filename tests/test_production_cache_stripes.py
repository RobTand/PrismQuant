from prismaquant.production_cache_stripes import plan_stripes


class _Profile:
    @staticmethod
    def fused_sibling_group(name):
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        return None

    @staticmethod
    def packed_expert_format_group(_name):
        return None


def _row(params, width):
    return {"n_params": params, "in_features": width}


def test_plan_is_disjoint_complete_and_layer_local():
    stats = {
        "model.layers.0.self_attn.q_proj": _row(20, 4),
        "model.layers.0.self_attn.k_proj": _row(10, 4),
        "model.layers.0.self_attn.v_proj": _row(10, 4),
        "model.layers.0.mlp.down_proj": _row(30, 8),
        "model.layers.1.mlp.down_proj": _row(60, 8),
        "lm_head": _row(50, 4),
        "mtp.fc": _row(8, 4),
        "mtp.layers.0.mlp.down_proj": _row(12, 4),
    }
    stripes = plan_stripes(stats, profile=_Profile(), n_stripes=2)
    owners = {
        name: stripe.index for stripe in stripes for name in stripe.qnames
    }
    assert set(owners) == set(stats)
    assert sum(len(stripe.qnames) for stripe in stripes) == len(stats)
    assert owners["model.layers.0.self_attn.q_proj"] == owners[
        "model.layers.0.self_attn.k_proj"
    ] == owners["model.layers.0.self_attn.v_proj"]
    assert owners["model.layers.0.mlp.down_proj"] == owners[
        "model.layers.0.self_attn.q_proj"
    ]
    assert owners["mtp.fc"] == owners["mtp.layers.0.mlp.down_proj"]


def test_plan_rejects_nonpositive_stripe_count():
    try:
        plan_stripes({"x": _row(1, 1)}, profile=_Profile(), n_stripes=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected a positive-stripe-count refusal")
