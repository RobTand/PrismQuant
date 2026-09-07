"""Runtime loader drops and component prefixes are not dispatch rewrites."""
from types import SimpleNamespace

import pytest

from prismaquant.model_profiles import vllm_registry
from prismaquant.model_profiles.validate import _check_name_remap


def check(monkeypatch, mapping, rewrite):
    monkeypatch.setattr(vllm_registry, "vllm_class_for_architecture", lambda _: object)
    monkeypatch.setattr(vllm_registry, "hf_to_vllm_prefix_map_from_class", lambda _: mapping)
    return _check_name_remap(SimpleNamespace(
        vllm_architecture_class=lambda: "Example", to_vllm_internal_name=rewrite,
    ))


def test_loader_drop_is_excluded_and_reported(monkeypatch):
    seen = []
    def rewrite(name):
        seen.append(name)
        return name.replace("model.", "body.", 1)
    result = check(monkeypatch, {"mtp.": None, "model.": "body."}, rewrite)
    assert result.ok
    assert seen == ["model.x.y"]
    assert "1 prefix rewrites agree" in result.detail
    assert "1 loader-only drops excluded" in result.detail


@pytest.mark.parametrize("source", ["model.vision_tower", "model.vision_tower."])
def test_probe_starts_a_real_component(monkeypatch, source):
    seen = []
    def rewrite(name):
        seen.append(name)
        return name.replace("model.vision_tower.", "vision.", 1)
    result = check(monkeypatch, {source: "vision."}, rewrite)
    assert result.ok
    assert seen == ["model.vision_tower.x.y"]


def test_incorrect_rewrite_still_fails(monkeypatch):
    result = check(monkeypatch, {"model.": "body."}, lambda name: name)
    assert not result.ok
    assert "expected prefix 'body.'" in result.detail
