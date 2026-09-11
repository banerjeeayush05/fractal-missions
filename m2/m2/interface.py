"""The velocity contract M3 builds against (PRD §5.5, §9 item 6).

Defined once in ``m2.schema`` (§5.0) and re-exported here with a version (OPEN_QUESTIONS B9).
v0.2 (decision §3) added `run_seed` and `stage_index`, made `positions` closest-point projections
in a fixed-capacity padded set, and added `weights` for smooth band membership.
v0.3 (decision A16/B14) adds `cell_id` — a stable, opaque, flattened grid index that seeds the RNG —
and `n_active`. Status stays "provisional" until the M3 owner signs off; do not freeze at M2.3
without it. Also bumped for the parameter-`scale` convention of decision B20, which M3 should adopt
for its own declarations.

Rules the M2 solver must obey (§5.5), whatever the version:
- at most one velocity-model call per RK stage;
- never assume two calls with the same inputs return the same value;
- seed from ``(run_seed, step_index, stage_index, cell_id)`` — never global, never stateful, never
  hashed from ``time``. Draws are independent per RK stage by default; a model may choose to share
  them, and M2 does not make that choice for it.
- treat ``cell_id`` as opaque: seed with it, never index geometry with it.
- expect common random numbers to break across a change of grid spacing, since ids mean different
  things at different dx.
- ``weights`` is zero for padded entries, so a model must produce a **finite** speed for them
  (``0 * NaN`` is NaN) and M2 ignores whatever it returns there. Never ``stop_gradient`` the
  weights: they are part of the forward map and their derivative is real.
- ``n_active`` says how many entries are real, for sizing K and for seeing overflow coming.

Changing ``VelocityRequest`` or ``VelocityModel`` changes ``contract_fingerprint()``. The
contract test then fails until ``CONTRACT_VERSION`` and ``CONTRACT_FINGERPRINT`` are bumped in
the same commit, so a contract change can never slip through unreviewed.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields

from m2.schema import PyTree, VelocityModel, VelocityRequest, check_velocity_output

__all__ = [
    "CONTRACT_FINGERPRINT",
    "CONTRACT_STATUS",
    "CONTRACT_VERSION",
    "PyTree",
    "VelocityModel",
    "VelocityRequest",
    "check_velocity_output",
    "contract_fingerprint",
]

CONTRACT_VERSION = "0.3.0"
CONTRACT_STATUS = "provisional"
CONTRACT_FINGERPRINT = "fcd41feff65512d353fe3b0b0186cc70d32ccae28b43b521f5347242595f765b"


def contract_fingerprint() -> str:
    """SHA-256 over the request's field names and annotations plus the model signature."""
    parts = [f"{f.name}:{f.type}" for f in fields(VelocityRequest)]
    parts.append(f"frozen:{VelocityRequest.__dataclass_params__.frozen}")
    parts.append("model:" + ",".join(_qualified(a) for a in VelocityModel.__args__))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _qualified(obj: object) -> str:
    qualname = getattr(obj, "__qualname__", None)
    return f"{obj.__module__}.{qualname}" if qualname else repr(obj)
