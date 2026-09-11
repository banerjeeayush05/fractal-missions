"""Imports the `m2` package must never have (decision §12; PRD §10, §11)."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "m2"
SOURCES = sorted(PACKAGE.rglob("*.py"))

# scipy.optimize is permitted in tests/ only, for M2.8's recovery check. PyTorch, Warp, PETSc and
# any second autodiff framework are banned outright: one autodiff system, so a gradient bug has one
# place to be. ViennaPS/ViennaRay are GPL-3.0 and must never be linked, imported or vendored.
FORBIDDEN_IN_PACKAGE = ("scipy.optimize", "torch", "warp", "petsc4py", "tensorflow",
                        "viennaps", "viennals", "viennaray")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_sources_found():
    assert len(SOURCES) >= 8


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_package_imports_nothing_forbidden(path):
    imported = _imported_modules(path)
    for banned in FORBIDDEN_IN_PACKAGE:
        hits = {m for m in imported if m == banned or m.startswith(banned + ".")}
        assert not hits, f"{path.name} imports {hits}: forbidden in the m2 package (decision §12, §10)"


def test_optimiser_is_available_to_tests():
    """The M2.8 recovery check may use scipy.optimize from tests/ — just not from the package."""
    from scipy.optimize import minimize  # noqa: F401
