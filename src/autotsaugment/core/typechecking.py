"""Opt-in runtime shape checks and the shared array type aliases.

Set ``AUTOTSAUGMENT_TYPECHECK=1`` before import to enable the checks. Decorate batch-level
methods only: one check per batch is cheap, one check per row is not.
"""

import os
from collections.abc import Callable
from typing import Final

from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor

TYPECHECK_ENV: Final = "AUTOTSAUGMENT_TYPECHECK"

# --- array aliases -------------------------------------------------------------------------
# Dimension names: n = series or windows, k = lockstep environments, a = action dimension,
# d = benchmark datasets, q = quantile levels, t = time steps, h = horizon steps, c = context.

SeriesBatch = Float[Tensor, "n t"]
SeriesBatchK = Float[Tensor, "k n t"]
Action = Float[Tensor, "a"]
ActionBatch = Float[Tensor, "k a"]
Context = Float[Tensor, "n c"]
Target = Float[Tensor, "n h"]
Quantiles = Float[Tensor, "n h q"]
QuantileLevels = Float[Tensor, "q"]
DatasetIndex = Int64[Tensor, "n"]
PerWindow = Float[Tensor, "n"]
PerDataset = Float[Tensor, "d"]
PerDatasetK = Float[Tensor, "k d"]
PerEnv = Float[Tensor, "k"]
DoneBatch = Bool[Tensor, "k"]
Scalar = Float[Tensor, ""]


def typecheck_enabled() -> bool:
    """Return True when runtime shape checks are switched on through the environment."""
    return os.environ.get(TYPECHECK_ENV, "0").strip().lower() not in {"", "0", "false"}


def checked[F: Callable[..., object]](func: F) -> F:
    """Wrap ``func`` with jaxtyping and beartype when the checks are enabled.

    Args:
        func: A function or method whose annotations use the aliases of this module.

    Returns:
        The wrapped function when checks are enabled, ``func`` itself otherwise.
    """
    if not typecheck_enabled():
        return func
    return jaxtyped(typechecker=beartype)(func)
