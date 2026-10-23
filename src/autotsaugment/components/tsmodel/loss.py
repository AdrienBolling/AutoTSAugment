"""Pinball loss of the Timestral head, in the normalized space of the model.

The model forecasts horizon patch ``j`` in the frame ``(loc_j, scale_j)``: the causal
statistics of the series up to the end of the patch before it. Patch 0 uses the frame of the
last context patch, the anchor of :meth:`timestral.TimestralModel.predict_block`. During
training the true horizon values are known, so the frames of the later patches come from the
true series (teacher forcing). :mod:`autotsaugment.components.tsmodel.timestral` explains
why the loss uses this space.
"""

import torch
from timestral.scaler import apply_affine_asinh_masked, causal_patch_loc_scale
from torch import Tensor

from autotsaugment.core.metrics import pinball_loss


def horizon_frames(
    window: Tensor,
    observed: Tensor,
    *,
    context_patches: int,
    patch_size: int,
    minimum_scale: float,
) -> tuple[Tensor, Tensor]:
    """Return the frame of every horizon step of a full window.

    Args:
        window: Context and horizon values, shape ``(n, t)``, ``t`` a multiple of
            ``patch_size``. Unobserved values must be finite (for example 0).
        observed: Observation mask, shape ``(n, t)``.
        context_patches: Patches of the context, L. The horizon is the rest of the window.
        patch_size: Time steps per patch.
        minimum_scale: Lower bound of the scale.

    Returns:
        ``(loc, scale)``, each shape ``(n, t - L * patch_size)``. Horizon patch ``j`` holds
        the causal statistics of the window up to the end of patch ``L + j - 1``.
    """
    loc, scale = causal_patch_loc_scale(
        window, observed, patch_size, minimum_scale, stabilize_with_global=False
    )
    start = (context_patches - 1) * patch_size
    stop = window.shape[-1] - patch_size
    return loc[:, start:stop], scale[:, start:stop]


def normalized_pinball_loss(
    quantiles: Tensor,
    window: Tensor,
    observed: Tensor,
    levels: Tensor,
    *,
    context_patches: int,
    patch_size: int,
    minimum_scale: float,
    use_asinh: bool,
) -> Tensor:
    """Return the mean pinball loss of normalized quantiles against a full window.

    Args:
        quantiles: Normalized quantiles of the horizon, shape ``(n, h, q)``.
        window: Context and horizon values, shape ``(n, L * patch_size + h)``. Unobserved
            values must be finite.
        observed: Observation mask of ``window``.
        levels: Quantile levels, shape ``(q,)``.
        context_patches: Patches of the context, L.
        patch_size: Time steps per patch.
        minimum_scale: Lower bound of the scale.
        use_asinh: Whether the model squashes normalized values with ``asinh``.

    Returns:
        The mean loss over the observed horizon steps and the levels, a 0-d tensor.
    """
    loc, scale = horizon_frames(
        window,
        observed,
        context_patches=context_patches,
        patch_size=patch_size,
        minimum_scale=minimum_scale,
    )
    start = context_patches * patch_size
    target_observed = observed[:, start:]
    target = apply_affine_asinh_masked(
        window[:, start:], loc, scale, target_observed, use_asinh=use_asinh
    )
    target = torch.where(target_observed, target, torch.full_like(target, float("nan")))
    loss = pinball_loss(quantiles, target, levels).sum()
    count = target_observed.sum() * levels.numel()
    return loss / count.clamp_min(1)
