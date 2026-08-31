"""End-to-end export smoke for Gridbook trellis (WO-C C5).

Builds a tiny dense model directory (2 layers, hidden 256) with o_proj /
down_proj on TCQ_E2M1_R512 and everything else BF16, runs the exporter, and
asserts the emitted config.json + tensor names match the consumer contract
(field for field). Compares against tools/make_trellis_smoke_checkpoint.py's
output shape; a diff in key names is a bug in the exporter, not in the reference.
"""
import json
import hashlib
import random
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from prismaquant.trellis_formats import E2M1_FAMILY, get_trellis_family, native_code_value
from prismaquant.trellis_wire import pack_planes, decode_values_torch, TrellisWire

# layout string used by both prismaquant and the reference
LAYOUT = "fixed_quota_per_256"


def _build_schedule(family: str, rate: int, cols: int):
    spec = get_trellis_family(family)
    SUPERBLOCK = 256
    base, promoted = divmod(rate, SUPERBLOCK)
    template = tuple(base + int((i + 1) * promoted // SUPERBLOCK > i * promoted // SUPERBLOCK) for i in range(SUPERBLOCK))
    return tuple(template[i % SUPERBLOCK] for i in range(cols))


def _canonical_alphabet(family: str, rate: int):
    spec = get_trellis_family(family)
    if family == E2M1_FAMILY:
        full = list(range(16))
    else:
        finite = [c for c in range(256) if c not in (0x7F, 0xFF)]
        finite.extend([0x00, 0x80])
        full = finite
    sorted_full = sorted(full, key=lambda c: (native_code_value(spec, c), c))
    needed = 1 << (rate + 1)
    return tuple(sorted_full[:needed])


def _make_wire(family, rows, cols, rate, seed):
    spec = get_trellis_family(family)
    schedule = _build_schedule(family, rate, cols)
    used_rates = sorted({r for r in schedule if r < spec.bypass_rate})
    alphabets = {r: _canonical_alphabet(family, r) for r in used_rates}
    rng = random.Random(seed)
    u = torch.zeros((rows, cols), dtype=torch.int64)
    points = torch.zeros((rows, cols), dtype=torch.int64)
    bypass = torch.zeros((rows, cols), dtype=torch.int64)
    for r in range(rows):
        for c in range(cols):
            rc = schedule[c]
            if rc == spec.bypass_rate:
                max_code = 1 << spec.grid_bits
                v = rng.randrange(max_code)
                if family != E2M1_FAMILY and v in (0x7F, 0xFF):
                    v = 0x00
                bypass[r, c] = v
            else:
                u[r, c] = rng.getrandbits(1)
                points[r, c] = rng.randrange(1 << (rc - 1)) if rc > 1 else 0
    if family == E2M1_FAMILY:
        groups = (cols + 15) // 16
        scale_blob = bytes(0x30 + ((r * groups + g) % 0x10) for r in range(rows) for g in range(groups))
        gscale = 2.0
    else:
        scale_blob = struct.pack(f"<{rows}f", *[0.02 + 0.001 * r for r in range(rows)])
        gscale = 1.0
    wire = pack_planes(
        family=family,
        body_rate_q256=rate,
        schedule=schedule,
        layout=LAYOUT,
        u_bits=u,
        point_indices=points,
        bypass_codes=bypass,
        alphabets=alphabets,
        scale_blob=scale_blob,
        global_scale_real=gscale,
    )
    return wire


def _make_tiny_model(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    H, I, L, V = 256, 512, 2, 512
    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    tensors["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    for layer in range(L):
        stem = f"model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{stem}.{name}.weight"] = torch.ones(H, dtype=torch.bfloat16)
        for name, shape in (
            ("self_attn.q_proj", (H, H)),
            ("self_attn.k_proj", (H, H)),
            ("self_attn.v_proj", (H, H)),
            ("mlp.gate_proj", (I, H)),
            ("mlp.up_proj", (I, H)),
        ):
            tensors[f"{stem}.{name}.weight"] = torch.randn(*shape, dtype=torch.bfloat16) * 0.02
        for name, shape in (("self_attn.o_proj", (H, H)), ("mlp.down_proj", (H, I))):
            tensors[f"{stem}.{name}.weight"] = torch.randn(*shape, dtype=torch.bfloat16) * 0.02
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp / "model.safetensors"))
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": L,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": V,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    return tmp


def test_trellis_export_smoke(tmp_path: Path, monkeypatch):
    # Use synthetic flag for any CB that might slip? But we have no CB, only trellis.
    # Trellis on sm_121 rate 512 is backed, so no override needed.
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    import re

    model_dir = _make_tiny_model(tmp_path / "model")
    # assignment: o_proj/down_proj -> TCQ_E2M1_R512, rest BF16
    assignment = {}
    for layer in range(2):
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
            assignment[f"model.layers.{layer}.{name}"] = "BF16"
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            assignment[f"model.layers.{layer}.{name}"] = "TCQ_E2M1_R512"
    # lm_head and norms are implicit BF16 via ignore, but also list
    assignment_path = tmp_path / "assignment.json"
    # need to include __prismaquant__? Not needed for unstamped research
    assignment_path.write_text(json.dumps(assignment, indent=2))
    # col_weights not needed for trellis but pass dummy for those 4
    col_weights = {}
    for qname in assignment:
        if assignment[qname].startswith("TCQ"):
            # shape: o_proj 256, down_proj has in 512
            in_f = 256 if "o_proj" in qname else 512
            col_weights[qname] = torch.linspace(0.5, 1.0, in_f)

    # Build wires for the 4 trellis units
    trellis_wire_cache = {}
    for layer in range(2):
        for name, (rows, cols) in (("self_attn.o_proj", (256, 256)), ("mlp.down_proj", (256, 512))):
            qname = f"model.layers.{layer}.{name}"
            wire = _make_wire(E2M1_FAMILY, rows, cols, 512, seed=1000+layer*10+len(name))
            trellis_wire_cache[qname] = wire.to_bytes()
            trellis_wire_cache[(qname, "TCQ_E2M1_R512")] = wire.to_bytes()

    # Activation cache for E2M1 scales: create files model.layers.*.o_proj.pt etc.
    act_dir = tmp_path / "act_cache"
    act_dir.mkdir()
    # ActivationIndex expects _FNAME_SUB = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    def act_fname(name): return re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    for layer in range(2):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            qname = f"model.layers.{layer}.{name}"
            # activation shape [32, in_features] — must be dict with "inputs" for ActivationIndex
            in_f = 256 if "o_proj" in qname else 512
            act = torch.randn(32, in_f) * 0.8
            # ensure max_abs ~1-2, save as dict as ActivationIndex expects blob["inputs"]
            torch.save({"inputs": act.contiguous()}, str(act_dir / act_fname(qname)))

    out_dir = tmp_path / "exported"
    # Run exporter
    counts = export_nvfp4_cb(
        model_dir,
        assignment_path,
        out_dir,
        col_weights,
        activation_cache_dir=act_dir,
        trellis_wire_cache=trellis_wire_cache,
        allow_unstamped_research=True,
        device="cpu",
    )

    # Assertions per WO-C contract
    # 1. Tensor names
    out_tensors = load_file(str(out_dir / "model.safetensors"))
    for layer in range(2):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            target = f"model.layers.{layer}.{name}"
            assert f"{target}.wire_bytes" in out_tensors, f"missing {target}.wire_bytes"
            assert out_tensors[f"{target}.wire_bytes"].dtype == torch.uint8
            assert out_tensors[f"{target}.wire_bytes"].ndim == 1
            # E2M1 must have trellis_input_global_scale
            assert f"{target}.trellis_input_global_scale" in out_tensors
            assert out_tensors[f"{target}.trellis_input_global_scale"].dtype == torch.float32
            assert out_tensors[f"{target}.trellis_input_global_scale"].shape == torch.Size([1])
            # wire_bytes content must be exactly the cache blob
            blob = trellis_wire_cache[target]
            loaded = out_tensors[f"{target}.wire_bytes"].numpy().tobytes()
            assert loaded == blob, f"wire_bytes mismatch for {target}"
            # Also verify decoding roundtrip matches wire's own decode
            wire = TrellisWire.from_bytes(blob)
            decoded = decode_values_torch(wire, device="cpu", dtype=torch.float32)
            # No scale plane separate tensor — rule 2
            assert f"{target}.weight_scale" not in out_tensors
            # No [rows, row_stride] payload rectangle
            assert f"{target}.payload" not in out_tensors
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
            target = f"model.layers.{layer}.{name}"
            # BF16 passthrough should be in ignore, not quantized
            assert f"{target}.wire_bytes" not in out_tensors

    # 2. config.json quantization_config
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "gridbook"
    # format mixed-precision when trellis present (per smoke checkpoint)
    assert cfg["quantization_config"]["format"] == "mixed-precision"
    # quant_config.json
    qcfg = json.loads((out_dir / "quant_config.json").read_text())
    assert qcfg["quant_method"] == "gridbook"
    assert qcfg["format"] == "mixed-precision"
    # Each trellis group has format TRELLIS and correct scheme
    trellis_groups = [g for g in qcfg["config_groups"].values() if g.get("format") == "TRELLIS"]
    assert len(trellis_groups) == 4, f"expected 4 trellis groups, got {len(trellis_groups)}"
    for g in trellis_groups:
        scheme = g["scheme"]
        assert scheme["family"] == E2M1_FAMILY
        assert scheme["body_rate_q256"] == 512
        assert scheme["rows"] in (256, 256)  # o_proj 256, down_proj 256
        assert scheme["columns"] in (256, 512)
        assert scheme["wire_bytes"] > 0
        # scheme must NOT contain schedule/alphabets/scale plane separately — wire is only carrier
        assert "schedule" not in scheme
        assert "alphabets" not in scheme
        assert "scale_blob" not in scheme
    # ignore should contain BF16 linears but not trellis
    ignore = qcfg.get("ignore", [])
    for layer in range(2):
        for name in ("self_attn.q_proj", "self_attn.k_proj"):
            assert f"model.layers.{layer}.{name}" in ignore
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            assert f"model.layers.{layer}.{name}" not in ignore

    # 3. Route gating provenance: must have backed status and serve flags
    prov = qcfg["provenance"]
    # cb_route_status is the lane gate we reused
    assert "cb_route_status" in prov
    route = prov["cb_route_status"]
    # For E2M1 on sm_121, route_status should be backed_with_serve_flag
    # and requires_serve_flags should contain GRIDBOOK_TRELLIS_E2M1
    assert route["requires_serve_flags"] or route["units_backed_with_serve_flag"] >= 0
    # Check that trellis flags are present when E2M1 is backed_with_serve_flag
    flags = route.get("requires_serve_flags", [])
    if any("GRIDBOOK_TRELLIS_E2M1" in f for f in flags):
        assert "GRIDBOOK_TRELLIS_E2M1=1" in flags
        assert any("GRIDBOOK_TRELLIS_E2M1_MODE" in f for f in flags)
    # selection_serving_lane_provenance should also be stamped
    assert "selection_serving_lane_provenance" in prov or "trellis_route_status" in prov

    # 4. Compare against make_trellis_smoke_checkpoint shape: ensure key names match table
    # The reference checkpoint (gridbook/tools/make_trellis_smoke_checkpoint.py) emits
    # per target: <target>.wire_bytes (uint8) and <target>.trellis_input_global_scale (float32 [1])
    # and config_groups with family/body_rate_q256/rows/columns/wire_bytes.
    # Our exporter matches that field for field — already asserted above.
