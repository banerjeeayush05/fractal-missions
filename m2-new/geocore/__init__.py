"""M2 — differentiable geometry core.

Level-set interface evolution in 2D and 3D, in JAX, with a gradient that is verified rather
than assumed. M2 does not compute the surface velocity; it consumes one through a frozen
contract so that M2 and M3 can fail independently.

fp64 is enabled HERE, at package import, before any array can be created.

PRD §7.5 and §11 ("do not start in fp32"). The reason it lives at import rather than in a
setup function a caller remembers to call: the failure mode of fp32 is not an error, it is a
tolerance that quietly cannot be met. V15 requires jax.jvp and jax.vjp to agree to 1e-10
relative, and fp32 carries about 7 decimal digits — the check would fail, someone would
"fix" it by loosening the tolerance to 1e-6, and the suite would go green having stopped
testing anything. Enabling it at import makes `import geocore` sufficient and forgetting it
impossible.

This module deliberately imports nothing from the rest of the package. Anything imported
here would be imported by every consumer of every submodule, and a cycle introduced at this
level is painful to unpick later.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
