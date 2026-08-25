from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CHECK_INSTALLED = REPO / ".github" / "scripts" / "check_installed.py"


def _call_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def test_installed_gate_enumerates_every_packaged_serving_profile() -> None:
    tree = ast.parse(CHECK_INSTALLED.read_text(encoding="utf-8"))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]

    assert any(
        isinstance(loop.iter, ast.Name)
        and loop.iter.id == "profile_names"
        for loop in loops
    )
    assert any(
        isinstance(node, ast.Call)
        and _call_path(node.func) == ("sp", "serving_profile_names")
        for node in ast.walk(tree)
    )


def test_installed_gate_smokes_the_two_host_campaign_cli() -> None:
    tree = ast.parse(CHECK_INSTALLED.read_text(encoding="utf-8"))
    argv_literals = [
        tuple(
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
    ]

    assert any(
        "prismaquant.rtx4090_two_host_campaign" in values
        and "--help" in values
        for values in argv_literals
    )
