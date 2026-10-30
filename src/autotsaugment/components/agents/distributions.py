"""The squashed Gaussian of the GRPO policy: sample, log-probability and entropy.

A pre-squash value ``u`` has a Gaussian distribution with location ``loc`` and scale ``scale``.
``tanh`` and an affine map put it in the action box,
``action = low + (tanh(u) + 1) / 2 * (high - low)``, the map of
:func:`~autotsaugment.components.agents.torchrl.networks.squash`. The log-probability of the
action is the Gaussian log-density of ``u`` minus the log-determinant of this map, as in
:class:`torchrl.modules.TanhNormal`. The functions take the pre-squash value, so they never
invert ``tanh``, and they make no host synchronization.
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor

from autotsaugment.core.typechecking import Action, checked

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
"""The constant term of the Gaussian log-density."""

PerDimension = Float[Tensor, "*batch a"]
"""A value per action dimension, with any leading batch dimensions."""

PerSample = Float[Tensor, "*batch"]
"""A value per sample: the sum over the action dimensions."""


def sample_pre_squash(loc: Tensor, scale: Tensor, generator: torch.Generator) -> Tensor:
    """Return ``loc + scale * noise`` with standard normal noise from ``generator``.

    Args:
        loc: Location of the Gaussian, shape ``(..., a)``.
        scale: Scale of the Gaussian, same shape, positive.
        generator: Generator on the device of ``loc``. It owns the randomness of the draw.
    """
    noise = torch.randn(loc.shape, generator=generator, device=loc.device, dtype=loc.dtype)
    return loc + scale * noise


@checked
def log_prob(
    pre_squash: PerDimension, loc: PerDimension, scale: PerDimension, low: Action, high: Action
) -> PerSample:
    """Return the log-probability of the squashed action of ``pre_squash``.

    ``sum_dims [log N(u; loc, scale) - log(1 - tanh(u)^2) - log((high - low) / 2)]``, with the
    stable form ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2 u))``.

    Args:
        pre_squash: The pre-squash value u, shape ``(..., a)``.
        loc: Location of the Gaussian, same shape.
        scale: Scale of the Gaussian, same shape, positive.
        low: Lower bound of the box, shape ``(a,)``.
        high: Upper bound of the box, shape ``(a,)``.

    Returns:
        Shape ``(...)``, the sum over the action dimensions.
    """
    normal = -0.5 * ((pre_squash - loc) / scale) ** 2 - torch.log(scale) - LOG_SQRT_2PI
    softplus = torch.nn.functional.softplus(-2.0 * pre_squash)
    log_tanh_derivative = 2.0 * (math.log(2.0) - pre_squash - softplus)
    log_half_width = torch.log((high - low) / 2.0)
    return (normal - log_tanh_derivative - log_half_width).sum(dim=-1)


@checked
def gaussian_entropy(scale: PerDimension) -> PerSample:
    """Return the entropy of the pre-squash Gaussian, summed over the action dimensions.

    ``sum_dims [0.5 + 0.5 log(2 pi) + log scale]``. The squash changes the entropy of the action,
    but this value is a cheap measure of the exploration width.

    Args:
        scale: Scale of the Gaussian, shape ``(..., a)``, positive.

    Returns:
        Shape ``(...)``.
    """
    return (0.5 + LOG_SQRT_2PI + torch.log(scale)).sum(dim=-1)
