"""The default lane's executed-activation list is derived, not cited.

RobTand/prismaquant#163: ``lane_specs/compressed_tensors.json``'s
``served_activation_quantization.executes`` is the authority every A-side
caller reaches (via ``aqua_activation_cost.resolve_executed_activation_formats``),
but its derivation was prose citing call sites at line numbers. A refactor
that moves a cited line, or a scheme-table edit that adds/drops
``input_activations`` (MXFP4 is one field away from flipping), silently kept
the old A-side price -- and a wrong value here lets a mispriced allocation
through the DP with no refusal anywhere.

The mechanical version this pins, exactly the preflight the lane spec's own
``if_this_changes`` note describes:

* the EXPECTED set is derived from the producer table that owns it
  (``export_native_compressed.FORMAT_SCHEME``: the canonical formats whose
  scheme carries ``input_activations``);
* the EMITTED set is read off an actually emitted ``quantization_config``
  (``build_quantization_config``): the formats whose ``config_groups`` carry
  ``input_activations``;
* ``require_compressed_executes_derived_from_scheme`` refuses any drift
  between the two and the lane spec, and the export preflight runs it before
  any GPU render.

What is pinned is the rule, not the roster: every expected set below is
rebuilt from ``FORMAT_SCHEME`` inside the test, so a scheme-table edit that
adds/drops ``input_activations`` fails here instead of silently keeping the
old A-side price.
"""
from __future__ import annotations

from copy import deepcopy

import pytest


def _lane_executes():
    from prismaquant.lane_spec import load_lane_spec

    spec = load_lane_spec("compressed_tensors")
    assert spec.served_activation_quantization is not None
    return set(spec.served_activation_quantization.executes)


def test_scheme_table_derivation_equals_the_lane_spec():
    """The lane spec's list is what the scheme table implies -- rebuilt here
    from the table, not restated, so the roster cannot go stale quietly."""
    import prismaquant.export_native_compressed as enc
    from prismaquant.export_native_compressed import (
        derive_executed_activation_formats,
    )

    expected = {
        enc._canonical_export_format(fmt)
        for fmt, scheme in enc.FORMAT_SCHEME.items()
        if "input_activations" in scheme
    }
    assert derive_executed_activation_formats() == expected
    assert _lane_executes() == expected
    # The legacy alias is a spelling, not a rung: it must not leak into the
    # derivation as a distinct entry.
    assert "MXFP8" not in derive_executed_activation_formats()
    assert derive_executed_activation_formats() <= enc.EXPORTABLE_FORMATS


def test_emitted_config_groups_carry_input_activations_for_the_lane_spec():
    """The file's own 'stronger version': read the emitted
    ``quantization_config`` and assert the lane spec's list equals the set of
    ``config_groups`` carrying ``input_activations``.

    One unit per emittable format plus a BF16 passthrough, through the real
    ``build_quantization_config``, so the emission path (catchall grouping,
    target building, the FP8_SOURCE carve-out) is what is being read, not the
    table it was built from.
    """
    import prismaquant.export_native_compressed as enc
    from prismaquant.export_native_compressed import (
        executed_activation_formats_in_quantization_config,
    )

    emittable = sorted(enc.EXPORTABLE_FORMATS - enc.CONTAINER_PASSTHROUGH_FORMATS)
    assert emittable, "no emittable format left to read the emission from"
    assignment = {
        f"model.layers.{i}.mlp.down_proj": fmt
        for i, fmt in enumerate(emittable)
    }
    qc = enc.build_quantization_config(assignment, {"lm_head"})
    assert qc["quant_method"] == "compressed-tensors"
    emitted = executed_activation_formats_in_quantization_config(qc)
    assert emitted == _lane_executes()
    # MXFP4's scheme carries no `input_activations`, so its group must be the
    # one without -- the dense-vs-fused distinction the lane spec warns about
    # is exactly this: one format's group answering differently from the rest.
    groups_with_a_side = {
        name for name, group in qc["config_groups"].items()
        if "input_activations" in group
    }
    assert groups_with_a_side, "no emitted group carries input_activations"
    assert len(groups_with_a_side) == len(qc["config_groups"]) - 1


def test_a_scheme_that_gains_input_activations_is_refused(monkeypatch):
    """MXFP4 is one field away from flipping: if its scheme gains
    ``input_activations`` the lane spec must change with it, loudly."""
    import prismaquant.export_native_compressed as enc
    from prismaquant.export_native_compressed import (
        require_compressed_executes_derived_from_scheme,
    )

    flipped = deepcopy(enc.MXFP4_SCHEME)
    flipped["input_activations"] = deepcopy(enc.NVFP4_SCHEME["input_activations"])
    monkeypatch.setitem(enc.FORMAT_SCHEME, "MXFP4", flipped)
    with pytest.raises(RuntimeError, match="MXFP4"):
        require_compressed_executes_derived_from_scheme()


def test_a_scheme_that_loses_input_activations_is_refused(monkeypatch):
    """The other direction: silently keeping a stale A-side price for a rung
    the runtime no longer executes is the DSv4 mispricing, not thrift."""
    import prismaquant.export_native_compressed as enc
    from prismaquant.export_native_compressed import (
        require_compressed_executes_derived_from_scheme,
    )

    narrowed = deepcopy(enc.NVFP4_SCHEME)
    del narrowed["input_activations"]
    monkeypatch.setitem(enc.FORMAT_SCHEME, "NVFP4", narrowed)
    with pytest.raises(RuntimeError, match="NVFP4"):
        require_compressed_executes_derived_from_scheme()


def test_an_unmatchable_emitted_group_is_refused():
    """Fail closed on the emission side too: a ``config_group`` no scheme in
    the producer table explains is drift, not a clean bill."""
    from prismaquant.export_native_compressed import (
        executed_activation_formats_in_quantization_config,
    )

    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "format": "something-new",
                "weights": {"num_bits": 4},
                "input_activations": {"num_bits": 4},
                "targets": ["re:^model[.]layers[.]0[.]mlp[.]down_proj$"],
            },
        },
    }
    with pytest.raises(RuntimeError, match="group_0"):
        executed_activation_formats_in_quantization_config(qc)
