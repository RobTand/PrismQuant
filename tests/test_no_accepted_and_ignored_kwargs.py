"""A public entry point must not accept an argument it ignores.

``build_candidates`` carried ``cost_mode`` for months after its only reader --
the trellis seam's currency gate -- was archived (#125, `a212b73`). A caller
passing ``cost_mode="aura"`` got no error and no effect: it could state an
intent the callee had stopped honouring, and nothing anywhere said so. That is
the same failure shape as a gate that names a check it does not run.

So the property, not the name. Pinning ``"cost_mode" not in signature`` would
be the roster again -- true today, silent on the next dead kwarg. This walks
each entry point's own AST and asks whether every parameter it declares is
actually LOAD-ed somewhere in the body. A parameter that is only ever bound is
accepted and ignored.

Deliberately narrow in two ways, because a repo-wide version of this rule would
be noise: it covers a named list of allocator entry points whose kwargs encode
caller intent, and it exempts the conventional sinks below. Adding an entry
point here is cheap; the cost of a false positive is that someone deletes a
real interface parameter to make a test pass.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import allocator_solver as asolver

#: Entry points whose keyword arguments are how a caller states an intent. If
#: one of these accepts a name it never reads, the intent goes nowhere.
ENTRY_POINTS = [
    (ac, "build_candidates"),
    (asolver, "promote_serving_units"),
    (asolver, "solve_allocation"),
]

#: Parameters that are legitimately bound and not read in the body.
EXEMPT = {
    "self", "cls",
    # *args / **kwargs forwarding is checked by the call it forwards to.
    "args", "kwargs",
}


def _declared_and_read(fn) -> tuple[set[str], set[str]]:
    # dedent: a nested function's source carries its enclosing indentation.
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), type(node)

    a = node.args
    declared = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    if a.vararg:
        declared.add(a.vararg.arg)
    if a.kwarg:
        declared.add(a.kwarg.arg)

    read = {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return declared - EXEMPT, read


@pytest.mark.parametrize("module,name", ENTRY_POINTS,
                         ids=[f"{m.__name__.split('.')[-1]}.{n}"
                              for m, n in ENTRY_POINTS])
def test_every_declared_parameter_is_read(module, name):
    fn = getattr(module, name)
    declared, read = _declared_and_read(fn)

    ignored = sorted(declared - read)
    assert not ignored, (
        f"{module.__name__}.{name} accepts {ignored} and never reads "
        f"{'them' if len(ignored) > 1 else 'it'}. A caller can state an intent "
        "the callee has stopped honouring, and nothing says so. Either read the "
        "parameter or delete it -- do not leave it as a courtesy."
    )


def test_the_check_can_actually_fail():
    """Anti-vacuity: the walker must catch a parameter that is only bound.

    Without this, a bug in ``_declared_and_read`` that returned an empty
    ``declared`` would make every case above pass and say nothing.
    """
    def sample(used, ignored=None):      # noqa: ARG001 - the point of the test
        return used

    declared, read = _declared_and_read(sample)

    assert declared == {"used", "ignored"}
    assert sorted(declared - read) == ["ignored"]
