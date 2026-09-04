"""Recipe provenance records every encoder setting the producer publishes."""
from __future__ import annotations

import json

from tessera import export as texport

from prismaquant.tessera_hessian import encoder_recipe


def _producer_source(**settings):
    identity = {field: "" if field.endswith("sha256") else 0
                for field in texport.HESSIAN_IDENTITY}
    return texport.ActivationSource(hessians={}, provenance=identity, **settings)


def _published_settings(source):
    # The producer owns normalization and the setting roster. Capture identity
    # and explanatory prose are the only config fields that are not a recipe.
    config = source.config_block()
    return {key: value for key, value in config.items()
            if key not in {"hessian", "note"}}


def test_default_recipe_records_all_producer_config_settings():
    source = _producer_source()
    expected = _published_settings(source)

    recipe = encoder_recipe()

    assert recipe == expected
    # In particular, an unset trailing objective and a disabled sweep are
    # actual settings, not missing values that provenance may omit.
    assert recipe["refit_objective_trailing"] == expected["refit_objective_trailing"]
    assert recipe["refit_gauss_seidel"] == expected["refit_gauss_seidel"]
    assert json.loads(json.dumps(recipe)) == expected


def test_recipe_preserves_producer_normalized_per_plane_settings(monkeypatch):
    source = _producer_source(
        refit_objective_trailing=dict(texport.DEFAULT_REFIT_OBJECTIVE),
        refit_gauss_seidel={"lut16": True},
    )
    expected = _published_settings(source)
    # Model a producer default change without changing the installed defaults
    # or executing an encode. The recipe must read the object it constructs.
    monkeypatch.setattr(texport, "ActivationSource", lambda **kwargs: source)

    recipe = encoder_recipe()

    assert recipe == expected
    assert type(recipe["refit_objective_trailing"]) is dict
    assert type(recipe["refit_gauss_seidel"]) is dict
    assert json.loads(json.dumps(recipe)) == expected
