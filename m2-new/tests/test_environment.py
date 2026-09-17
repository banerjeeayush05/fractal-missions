"""Stage 1 gate — the environment is what the PRD requires.

Nothing here tests physics; there is none yet. These check the two environment properties
that are silent when wrong: fp64, and the one-autodiff-system rule. Both are cheap now and
expensive to discover later, because neither announces itself as a failure — fp64 announces
itself as a tolerance nobody can meet, and a second autodiff framework announces itself as a
gradient bug with two places to hide.

The import-boundary test walks the package source with `ast` rather than importing it. At
stage 1 the package is one file and the test is nearly vacuous; it is written now so that it
already has teeth on the day stage 3 adds the first module that could violate it.
"""

import ast
import pathlib

import jax
import jax.numpy as jnp

import geocore

PACKAGE_ROOT = pathlib.Path(geocore.__file__).parent

# Repo CLAUDE.md: one autodiff system, JAX. A gradient bug must have one place to be.
FORBIDDEN_ANYWHERE = (
    "torch", "tensorflow", "warp", "petsc4py", "mitsuba", "optix", "embree",
    "viennaps", "viennals", "viennaray",     # also GPL-3.0 — never link, import or vendor
)

# PRD §10 / decision §12: permitted inside tests/ for M2.8, never imported by the package.
FORBIDDEN_IN_PACKAGE = ("scipy.optimize",)

# Decision A11: scikit-image is V13's independent reference ONLY, never in the differentiated path.
FORBIDDEN_IN_PACKAGE += ("skimage",)


def _imported_module_names(path: pathlib.Path) -> set[str]:
    """Every module name a source file imports, without executing it."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _package_sources() -> list[pathlib.Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_fp64_is_enabled_by_importing_the_package():
    """PRD §7.5. `import geocore` must be sufficient; a caller must not have to remember a flag."""
    assert jax.config.read("jax_enable_x64") is True
    assert jnp.asarray(1.0).dtype == jnp.float64
    assert jnp.asarray([1.0, 2.0]).sum().dtype == jnp.float64


def test_arithmetic_stays_in_fp64():
    """A default that is fp64 but degrades under an operation would be worse than no default,
    because it would pass the check above and still lose the precision V15 needs."""
    x = jnp.linspace(0.0, 1.0, 8)
    assert (x * x).dtype == jnp.float64
    assert jnp.sqrt(x + 1.0).dtype == jnp.float64
    assert jax.grad(lambda t: jnp.sum(t**2))(x).dtype == jnp.float64


def test_package_imports_no_second_autodiff_system():
    """Repo CLAUDE.md. Checked at the source level so it holds for modules that are never
    imported by the test suite as well as those that are."""
    offenders = {}
    for source in _package_sources():
        hits = {
            name for name in _imported_module_names(source)
            if name.split(".")[0] in FORBIDDEN_ANYWHERE
        }
        if hits:
            offenders[source.relative_to(PACKAGE_ROOT)] = sorted(hits)
    assert not offenders, f"forbidden framework imported by the geocore package: {offenders}"


def test_package_does_not_import_scipy_optimize():
    """PRD §10, decision §12: permitted in tests/ for M2.8, never in the package. An optimiser
    inside the package is how "the optimiser converged" becomes gradient evidence (§11)."""
    offenders = {}
    for source in _package_sources():
        hits = {
            name for name in _imported_module_names(source)
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IN_PACKAGE)
        }
        if hits:
            offenders[source.relative_to(PACKAGE_ROOT)] = sorted(hits)
    assert not offenders, f"scipy.optimize imported by the geocore package: {offenders}"


def test_the_import_boundary_check_can_actually_fail(tmp_path):
    """The stage-1 instance of the rule behind V19: a check that has never been shown to fail
    has not been verified. At stage 1 the package is one file, so both tests above would pass
    against a scanner that always returns nothing."""
    sabotaged = tmp_path / "sabotaged.py"
    sabotaged.write_text("import torch\nfrom scipy.optimize import minimize\n")
    found = _imported_module_names(sabotaged)
    assert "torch" in found
    assert "scipy.optimize" in found
    assert "scipy.optimize.minimize" in found
