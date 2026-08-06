import pytest
from harness.byte_accounting import ledger_row, bytes_uniform, bytes_split, bpp

def test_ledger_examples_equal_bytes():
    # Acceptance requires 3 hand-checked examples identical bytes (shared-scale model)
    cases = [
        ("gate_proj", 32, [30, 34]),
        ("gate_proj", 33, [31, 33, 33, 35]),
        ("down_proj", 36, [34, 34, 38, 38]),
    ]
    for proj, uk, ks in cases:
        row = ledger_row(proj, uk, ks, per_group_scale=False)
        assert row["equal"], f"ledger not equal {row}"
        assert row["delta_bytes"] == 0

def test_bytes_uniform_values():
    # Hand compute gate 2048x4096 K32: index 2048*16*4*32=4194304 +8192=4202496
    assert bytes_uniform("gate_proj", 32) == 4194304 + 8192
    assert bytes_uniform("up_proj", 32) == 4194304 + 8192
    # down 4096x2048 K32: index 4096*8*4*32=4194304 +16384=4210688
    assert bytes_uniform("down_proj", 32) == 4194304 + 16384

def test_split_shared_scale_equality():
    # G=2 gate, uniform K32 vs split [30,34] avg 32
    u = bytes_uniform("gate_proj", 32)
    s = bytes_split("gate_proj", [30, 34], per_group_scale=False)
    assert s == u
    # G=4
    u4 = bytes_uniform("gate_proj", 33)
    s4 = bytes_split("gate_proj", [31, 33, 33, 35], per_group_scale=False)
    assert s4 == u4

def test_per_group_scale_overhead():
    # With per-group scale, split costs extra (G-1)*4*out
    u = bytes_uniform("gate_proj", 32, per_group_scale=True, groups=2)
    # gate uniform with G=2 per-group scale already includes G*scale, but our uniform for split comparison uses G scale?
    # Check overhead: shared vs per-group for split
    shared = bytes_split("gate_proj", [30, 34], per_group_scale=False)
    per_group = bytes_split("gate_proj", [30, 34], per_group_scale=True)
    assert per_group - shared == 8192  # (2-1)*8192

def test_bpp_reasonable():
    # gate K32 bpp = 8*bytes/params ; params 8M, bytes 4.2M -> ~4.0 bpw?
    bp = bpp(bytes_uniform("gate_proj", 32), "gate_proj")
    assert 3.5 < bp < 5.0
