"""GLM attention pins cover checkpoint and converted live spellings."""
import pytest
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


@pytest.mark.parametrize('projection', ['f_a_proj', 'f_b_proj'])
@pytest.mark.parametrize('suffix', ['', '.weight'])
def test_kda_forget_projection_stays_pinned_after_checkpoint_conversion(projection, suffix):
    profile = Glm5NextProfile()
    original = f'model.language_model.layers.0.self_attn.{projection}{suffix}'
    live = f'model.language_model.layers.0.self_attn.forget_gate.{projection}{suffix}'
    assert profile.is_pinned_name(original)
    assert profile.is_pinned_name(live), live
