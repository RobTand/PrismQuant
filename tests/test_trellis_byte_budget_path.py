"""``UNWIRED_LINKS`` #2 and #6, exercised where the ledger names them.

Both entries name a site inside a *run*, not a helper:

  * #2 ``allocator.py:3369-3386`` -- "the allocator dies inside the Pareto
    sweep, before layer_config.json is written".
  * #6 ``footprint.py:1183`` -- "the byte-budget (``--target-disk-gb``) path
    has its own registry lookup".

The wiring commit's own tests call ``build_trellis_menu`` and
``footprint.assignment_artifact_bytes`` directly. That proves the two
functions, and it is worth proving; it does not prove the *run*. Between a
built menu and a priced selection sit the Pareto sweep, the super-item
expansion, ``promote_serving_units``, the merged
``{**accounting_stats, **fixed_stats, **stats}`` view the footprint pricer is
handed, and the byte-budget bisection -- any of which could drop, copy or
recompute the exact bytes without either direct-call test noticing. So this
drives the real ``allocator.main()`` with ``--target-disk-gb`` and a trellis
rung in the menu.

The expectation is computed INDEPENDENTLY, from the format name plus the
manifest, through ``trellis_footprint.trellis_tensor_payload_breakdown`` --
never from the ``Candidate`` the allocator built. A test whose expected value
is read off the object under test cannot catch the failure this whole seam
exists to prevent: a byte count that is plausible and wrong. Byte-budget
undershoot is invisible, so the closed form a ``FormatSpec`` would have
returned is asserted to be a DIFFERENT number, and the one the run reports is
asserted to be the descriptor's.
"""
from __future__ import annotations

import json
import math
import pickle
import struct
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import footprint as fp
from prismaquant import trellis_menu as tm
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown
from prismaquant.trellis_formats import (
    E4M3_FAMILY,
    LAYOUT_TIGHT_OFFSETS,
    SUPERBLOCK_WEIGHTS,
    get_trellis_family,
    native_code_value,
    parse_trellis_format_name,
)
from prismaquant.trellis_rate_surface import uniform_column_schedule

# o_proj only: never a fused sibling and never a packed expert, so this test
# stays clear of UNWIRED_LINKS #3/#4/#5 (super-item aggregation and rank),
# which other work owns. What is under test here is byte accounting.
_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
_OUT = 512
_IN = SUPERBLOCK_WEIGHTS          # one whole superblock; a short final block
_NPARAMS = _OUT * _IN             # is legal on the wire but the campaign's
_OVERHEAD_RESERVE = 512           # to declare, so the seam refuses it
_FLOOR_TENSORS = {
    "model.embed_tokens.weight": ("BF16", (512, 64)),
    "lm_head.weight": ("BF16", (512, 64)),
    "model.norm.weight": ("BF16", (64,)),
}

PROFILE = "trellis_research_sm121"
COST_MODE = "aura"
CURRENCY = "aura-adjoint"
_ANCHORS = [
    {"q256": 512, "dloss": 4.0e-3, "stderr": 1e-4},
    {"q256": 1024, "dloss": 1.0e-3, "stderr": 1e-4},
    {"q256": 1536, "dloss": 3.0e-4, "stderr": 2e-4},
]


def _alphabet(rate: int) -> list[int]:
    """A NaN-free E4M3 alphabet ordered by decoded value then code."""
    spec = get_trellis_family(E4M3_FAMILY)
    required = 1 << (rate + 1)
    codes = [code for code in range(256) if code not in (0x7F, 0xFF)]
    if required > len(codes):
        codes.extend((0x00, 0x80))
        required = len(codes)
    codes.sort(key=lambda code: (native_code_value(spec, code), code))
    start = (len(codes) - required) // 2
    return list(codes[start:start + required])


_ALPHABETS = {rate: _alphabet(rate) for rate in range(1, 8)}


def _write_safetensors(path, tensors):
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for dim in shape:
            nbytes *= dim
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)


def _fixture(tmp_path):
    """Synthetic bf16 checkpoint + probe/cost pickles + a surface manifest."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    for name in _NAMES:
        tensors[f"{name}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)

    stats = {
        name: {"h_trace": 1.0 + 0.1 * i, "n_params": _NPARAMS,
               "in_features": _IN, "out_features": _OUT}
        for i, name in enumerate(_NAMES)
    }
    probe_p = tmp_path / "probe.pkl"
    probe_p.write_bytes(pickle.dumps(
        {"stats": stats, "meta": {"model": str(model_dir)}}))

    cost_p = tmp_path / "cost.pkl"
    cost_p.write_bytes(pickle.dumps({
        "costs": {
            name: {
                "NVFP4": {"weight_mse": 1e-4, "output_mse": 1e-4,
                          "output_mse_measured": True,
                          "predicted_dloss": 1e-4},
                "FP8_E4M3": {"weight_mse": 1e-6, "output_mse": 1e-6,
                             "output_mse_measured": True,
                             "predicted_dloss": 1e-6},
            }
            for name in _NAMES
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
        # The run's objective is ATTESTED from the cost table (re-vet R2); the
        # seam refuses an unstamped one rather than defaulting.
        "provenance": {"cost_mode": COST_MODE},
    }))

    manifest_p = tmp_path / "surface.json"
    manifest_p.write_text(json.dumps({
        "schema": tm.TRELLIS_SURFACE_MANIFEST_SCHEMA,
        "cost_mode": COST_MODE,
        "currency": CURRENCY,
        "target_profile": PROFILE,
        "activation_contract": "W8A16",
        "layout": LAYOUT_TIGHT_OFFSETS,
        "rungs_per_unit": 6,
        "provenance": {"encoder": "stage6", "encode_tier": "max"},
        "anchors": {
            name: {
                "family": E4M3_FAMILY,
                "alphabets": {str(r): codes
                              for r, codes in _ALPHABETS.items()},
                "points": _ANCHORS,
            }
            for name in _NAMES
        },
    }))
    return model_dir, probe_p, cost_p, manifest_p, stats


def _descriptor_bytes(fmt: str) -> int:
    """Exact serialized bytes for one rung, derived from the MANIFEST.

    Independent of the ``Candidate`` the allocator built: the closed
    ``TCQ_<grid>_R<q256>`` name yields (family, rate), the manifest yields the
    layout and the alphabets, and a real per-column schedule yields the body
    stride, the block offsets and the alphabet directory. This is the only
    number this file will accept for a trellis row.
    """
    parsed = parse_trellis_format_name(fmt)
    assert parsed is not None, fmt
    family, rate = parsed
    schedule = uniform_column_schedule(_IN, rate, family=E4M3_FAMILY)
    used = sorted({code for code in schedule if code < family.bypass_rate})
    breakdown = trellis_tensor_payload_breakdown(
        (_OUT, _IN),
        family=E4M3_FAMILY,
        body_rate_q256=rate,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={code: _ALPHABETS[code] for code in used},
    )
    return int(breakdown["total_bytes"])


def _body_rate_closed_form_bytes(fmt: str) -> int:
    """What a ``(name, shape)`` closed form would have returned.

    The body rate alone, with none of the side information the wire actually
    carries -- the shape of "plausible and wrong" this seam exists to refuse.
    """
    _family, rate = parse_trellis_format_name(fmt)
    return math.ceil(_NPARAMS * (rate / 256.0) / 8.0)


def _stub_solver(fmt_for_target, seen):
    """A ``solve_with_promotion`` that returns a chosen format per target.

    It also records the menu it was handed, so the test can name a rung that
    really is on offer without trusting anything the menu says about bytes.
    """
    def solve(stats, candidates, target_bits, format_specs, format_rank,
              bit_precision, **kw):
        seen.update({
            name: sorted(c.fmt for c in menu)
            for name, menu in candidates.items()
        })
        fmt = fmt_for_target(float(target_bits))
        assign = {name: fmt for name in candidates}
        total_params = sum(stats[name]["n_params"] for name in assign)
        bits = 0.0
        for name in assign:
            cand = next(c for c in candidates[name] if c.fmt == fmt)
            bits += 8.0 * cand.memory_bytes
        achieved = bits / max(total_params, 1)
        diag = kw.get("diagnostics")
        if diag is not None:
            diag.update({"feasible": True, "achieved_bits": achieved,
                         "predicted_dloss": None, "evals": 1})
        return assign, achieved
    return solve


def _run(monkeypatch, tmp_path, probe_p, cost_p, manifest_p, *, disk_gb,
         fmt_for_target, pareto="4.6,8.2"):
    seen: dict[str, list[str]] = {}
    monkeypatch.setenv(tm.TRELLIS_SURFACE_ENV, str(manifest_p))
    monkeypatch.setattr(alloc, "solve_with_promotion",
                        _stub_solver(fmt_for_target, seen))
    layer_config = tmp_path / "layer_config.json"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", pareto,
        "--target-disk-gb", repr(disk_gb),
        "--layer-config", str(layer_config),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--artifact-overhead-reserve-bytes", str(_OVERHEAD_RESERVE),
        "--allow-default-profile",
    ])
    alloc.main()
    return (
        json.loads((tmp_path / "selection.json").read_text()),
        json.loads(layer_config.read_text()),
        seen,
    )


def _a_rung_on_offer(seen) -> str:
    """The sparsest trellis rung the run's own menu offered, by name only.

    Sparsest so that every Pareto target in this fixture can afford it and
    the run exercises the pricing path at more than one budget. Only the
    NAME is taken from the menu; the bytes come from the manifest.
    """
    rungs = sorted(
        (fmt for fmt in seen[_NAMES[0]] if fmt.startswith("TCQ_")),
        key=lambda fmt: parse_trellis_format_name(fmt)[1],
    )
    assert rungs, (
        "the seam put no trellis rung in the menu, so this file is testing "
        f"nothing; menu was {seen[_NAMES[0]]}"
    )
    return rungs[0]


# ---------------------------------------------------------------------------
# UNWIRED_LINKS #2 and #6, in one run
# ---------------------------------------------------------------------------
def test_the_byte_budget_run_prices_a_trellis_rung_at_descriptor_bytes(
        monkeypatch, tmp_path):
    """The whole path: Pareto sweep (#2) and byte-budget pricing (#6).

    Entry #2's failure was death INSIDE the Pareto sweep, before
    ``layer_config.json`` was written; entry #6's was an independent registry
    lookup in the ``--target-disk-gb`` path. Both are asserted here by the run
    completing AND by the bytes it reports being the descriptor's, not a
    closed form over ``(name, shape)``.
    """
    model_dir, probe_p, cost_p, manifest_p, stats = _fixture(tmp_path)

    # Pass one: learn which rungs the manifest densified to, without letting
    # the run's own byte arithmetic near the expectation.
    probe = _run(
        monkeypatch, tmp_path, probe_p, cost_p, manifest_p,
        disk_gb=1.0, fmt_for_target=lambda _t: "NVFP4",
    )[2]
    rung = _a_rung_on_offer(probe)
    exact = _descriptor_bytes(rung)
    floor = fp.floor_bytes_for_model(str(model_dir), _NAMES, stats)["floor_bytes"]
    expected_payload = floor + len(_NAMES) * exact

    selection, layer_cfg, seen = _run(
        monkeypatch, tmp_path, probe_p, cost_p, manifest_p,
        disk_gb=(expected_payload + _OVERHEAD_RESERVE + 10_000) / fp.GB,
        fmt_for_target=lambda _target: rung,
    )

    # #6: every priced grid rung went through
    # footprint.assignment_artifact_bytes and came back at the descriptor's
    # bytes -- at both budgets, so this is the pricing path and not one
    # lucky row.
    assert len(selection["grid"]) == 2, selection["grid"]
    for row in selection["grid"]:
        assert row["tensor_payload_gb"] * fp.GB == pytest.approx(
            float(expected_payload), abs=1.0
        ), (
            f"target {row['target_bits']}: the byte-budget path did not "
            "price the rung at its descriptor bytes"
        )

    # ...and the closed form a FormatSpec would have returned is a different
    # number, so the assertion above is not satisfiable by accident. The wire
    # carries an 88-byte header, a nibble schedule plane, block offsets, an
    # alphabet directory and a per-row scale plane on top of the body.
    closed_form = _body_rate_closed_form_bytes(rung)
    assert exact > closed_form, (
        f"{rung}: descriptor bytes {exact} are not above the body-rate closed "
        f"form {closed_form}; the fixture has stopped distinguishing the two "
        "numbers this test exists to tell apart"
    )
    assert selection["grid"][0]["tensor_payload_gb"] * fp.GB != pytest.approx(
        float(floor + len(_NAMES) * closed_form), abs=1.0)

    # #2: the sweep survived and layer_config.json was written, carrying the
    # closed TCQ spelling that layer_config.canonicalize_format round-trips --
    # which is what makes the exporter's own refusal reachable at all.
    assert layer_cfg[_NAMES[0]] == rung
    from prismaquant.layer_config import canonicalize_format
    assert canonicalize_format(layer_cfg[_NAMES[0]]) == rung


def test_the_merged_stats_view_the_pricer_sees_keeps_the_exact_bytes(
        monkeypatch, tmp_path):
    """``_footprint_scalars`` hands the pricer a MERGED stats view.

    ``{**accounting_stats, **fixed_stats, **stats}`` copies dict references,
    so the seam's ``_memory_bytes_by_format`` write survives. That is true by
    construction today and invisible if it ever stops being: a copy that
    dropped the map would leave ``footprint`` refusing, and a copy that
    dropped only the TCQ ROWS would leave it pricing a closed form. Pin the
    map's presence on the object the pricer is actually handed.
    """
    _model_dir, probe_p, cost_p, manifest_p, _stats = _fixture(tmp_path)
    handed: list[dict] = []
    real = fp.assignment_artifact_bytes

    def spy(assignment, view, **kw):
        handed.append(view)
        return real(assignment, view, **kw)

    monkeypatch.setattr(fp, "assignment_artifact_bytes", spy)
    probe = _run(
        monkeypatch, tmp_path, probe_p, cost_p, manifest_p,
        disk_gb=1.0, fmt_for_target=lambda _t: "NVFP4",
    )[2]
    rung = _a_rung_on_offer(probe)

    assert handed, "the byte-budget selector never called the shared pricer"
    view = handed[-1]
    recorded = view[_NAMES[0]].get("_memory_bytes_by_format") or {}
    assert recorded.get(rung) == _descriptor_bytes(rung), (
        "the stats view handed to footprint.assignment_artifact_bytes carries "
        f"no descriptor bytes for {rung}: {sorted(recorded)}"
    )


def test_a_trellis_row_the_seam_never_priced_is_refused_not_estimated(
        monkeypatch, tmp_path):
    """The refusal, on the run's own accounting path.

    A stats entry the seam did not write has no honest byte answer. The
    allocator's exact payload filter must say so rather than fall through to
    a registry lookup -- and the message must name the map, because that is
    the fix.
    """
    _model_dir, probe_p, cost_p, manifest_p, _stats = _fixture(tmp_path)
    probe = _run(
        monkeypatch, tmp_path, probe_p, cost_p, manifest_p,
        disk_gb=1.0, fmt_for_target=lambda _t: "NVFP4",
    )[2]
    rung = _a_rung_on_offer(probe)

    unpriced = {name: {"n_params": _NPARAMS, "in_features": _IN,
                       "out_features": _OUT}
                for name in _NAMES}
    with pytest.raises(ValueError, match="_memory_bytes_by_format"):
        fp.assignment_artifact_bytes(
            {name: rung for name in _NAMES},
            unpriced,
            source_total_bytes=10_000_000,
            source_manifest=None,
            regime="bf16",
        )
