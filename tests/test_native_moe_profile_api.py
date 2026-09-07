"""Exercise boundary validation with the real registered LFM profile."""
from types import SimpleNamespace

import pytest
import torch

from prismaquant.model_profiles import Lfm2MoeProfile, profile_from_model
from prismaquant.native_moe_panel import captured_moe_boundary


def test_capture_uses_real_lfm_profile_property_before_target_validation():
    model = SimpleNamespace(config=SimpleNamespace(model_type="lfm2_moe", architectures=["Lfm2MoeForCausalLM"],
                                                   _attn_implementation="eager"),
        _prismaquant_pretrained_initialization_contract={"schema": "prismaquant.pretrained_initialization.v1",
            "scope": "checkpoint_missing_state", "status": "completed", "transformers_version": "fixture"})
    assert isinstance(profile_from_model(model), Lfm2MoeProfile)
    # An unrelated target must be rejected by the actual profile check, not
    # crash while calling a string-valued property. No fake profile is injected.
    with pytest.raises(ValueError, match="not the source model's declared LFM experts"):
        captured_moe_boundary(torch.nn.Identity(), (), {}, torch.tensor([[0, 0], [0, 1]]),
            unit="model.layers.2.feed_forward.experts", source_model=model, prefill_rows=2,
            calibration_receipt={"schema": "prismaquant.calibration_input.v1", "shape": [1, 2]}, producer_source={})
