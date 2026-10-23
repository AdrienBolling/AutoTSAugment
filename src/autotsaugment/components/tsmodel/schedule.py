"""Learning-rate schedule of one episode: linear warmup, then cosine decay.

The schedule spans ``episode_epochs * steps_per_epoch`` optimizer steps. It runs on the host
and changes only the Python learning rate of the optimizer, so it never touches the device.
"""

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step: int, *, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    """Return the learning-rate factor of an optimizer step.

    Args:
        step: Index of the optimizer step, from 0.
        total_steps: Steps of the episode. Later steps keep the final factor.
        warmup_steps: Steps of the linear warmup, at the start of the episode.
        min_ratio: Factor at the end of the cosine decay, in ``[0, 1]``.

    Returns:
        ``(step + 1) / warmup_steps`` during the warmup, then a cosine decay from 1 to
        ``min_ratio`` over the remaining steps.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: Optimizer, *, total_steps: int, warmup_fraction: float, min_ratio: float
) -> LambdaLR:
    """Return a scheduler that applies :func:`warmup_cosine` to every parameter group.

    Call ``step()`` on the result after each optimizer step.

    Args:
        optimizer: The optimizer; its learning rates are the peak values.
        total_steps: Optimizer steps of the episode.
        warmup_fraction: Share of ``total_steps`` used for the warmup, in ``[0, 1)``.
        min_ratio: Factor at the end of the decay.
    """
    warmup_steps = round(warmup_fraction * total_steps)
    factor = partial(
        warmup_cosine, total_steps=total_steps, warmup_steps=warmup_steps, min_ratio=min_ratio
    )
    return LambdaLR(optimizer, factor)
