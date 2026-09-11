"""PRD §7.3: every declared parameter must produce a nonzero gradient on at least one objective.

A parameter captured in a closure instead of read from the PyTree gets an exactly-zero
gradient column. Liveness is therefore checked per scalar component, with an exact ``!= 0``
test (OPEN_QUESTIONS B11): a tolerance would only hide the failure this exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from m2.verification.gradcheck import CheckResult

PyTree = Any


def component_names(params: PyTree) -> list[str]:
    """Human-readable name for each scalar component, in ravel_pytree order."""
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        base = jax.tree_util.keystr(path)
        size = int(np.size(leaf))
        names += [base] if size == 1 and np.ndim(leaf) == 0 else [f"{base}[{i}]" for i in range(size)]
    return names


def check_parameters_live(objectives: Mapping[str, Callable[[PyTree], Any]], params: PyTree) -> CheckResult:
    if not objectives:
        raise ValueError("at least one objective is required")
    names = component_names(params)
    flat_grads = {name: np.asarray(ravel_pytree(jax.grad(J)(params))[0]) for name, J in objectives.items()}
    stacked = np.stack(list(flat_grads.values()))  # (n_objectives, n_components)
    live = np.any(stacked != 0.0, axis=0)
    dead = [names[i] for i in np.flatnonzero(~live)]
    max_abs = {names[i]: float(np.max(np.abs(stacked[:, i]))) for i in range(len(names))}
    passed = not dead
    msg = ("every parameter has a nonzero gradient on at least one objective" if passed else
           f"zero gradient on every objective (captured in a closure?): {dead}")
    return CheckResult("§7.3", passed, msg, {"max_abs_grad": max_abs, "dead": dead,
                                             "objectives": list(objectives)})
