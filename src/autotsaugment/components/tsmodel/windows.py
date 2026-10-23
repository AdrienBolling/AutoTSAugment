"""Training and forecast windows, cut on the device.

The functions in this module run on the device of their input and never synchronize with
the host.
"""

import torch
from torch import Tensor


def random_windows(series: Tensor, length: int, *, generator: torch.Generator) -> Tensor:
    """Cut one random window from each series, then shuffle the windows.

    The generator draws the start of each window first, then the row order.

    Args:
        series: Series, shape ``(n, t)``, with ``t >= length``.
        length: Steps per window, context plus horizon.
        generator: Generator on the device of ``series``.

    Returns:
        Windows, shape ``(n, length)``. Each row is a contiguous slice of one series.

    Raises:
        ValueError: If the series are shorter than ``length``.
    """
    n, steps = series.shape
    if steps < length:
        msg = f"series length {steps} is shorter than the window length {length}"
        raise ValueError(msg)
    device = series.device
    starts = torch.randint(0, steps - length + 1, (n,), generator=generator, device=device)
    index = starts.unsqueeze(1) + torch.arange(length, device=device)
    windows = torch.gather(series, 1, index)
    order = torch.randperm(n, generator=generator, device=device)
    return windows[order]


def pad_context(context: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Left-pad a context with missing values to a multiple of ``patch_size``.

    Args:
        context: Context, shape ``(n, c)``. A non-finite value marks a missing value.
        patch_size: Time steps per patch.

    Returns:
        ``(inputs, observed)``, each shape ``(n, c')`` with ``c'`` the next multiple of
        ``patch_size``. ``inputs`` holds 0 where ``observed`` is False.
    """
    pad = (-context.shape[-1]) % patch_size
    if pad:
        context = torch.nn.functional.pad(context, (pad, 0), value=float("nan"))
    observed = torch.isfinite(context)
    inputs = torch.where(observed, context, torch.zeros_like(context))
    return inputs, observed
