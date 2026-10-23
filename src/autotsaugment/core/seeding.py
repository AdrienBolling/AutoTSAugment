"""Seed derivation and generator construction.

Every component receives a seed derived from the root seed and its own name, so two
components never share a random stream and a run is reproducible per component.
"""

import hashlib
import random

import numpy as np
import torch

MAX_SEED = 2**31 - 1


def derive_seed(root: int, *keys: str | int) -> int:
    """Return a seed derived from ``root`` and a path of keys.

    Args:
        root: The root seed of the run.
        *keys: Names or indices that identify the consumer, for example ``("env", 3)``.

    Returns:
        A seed in ``[0, 2**31 - 1]``, stable across processes and platforms.
    """
    text = ":".join([str(root), *(str(key) for key in keys)])
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "little") % (MAX_SEED + 1)


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Return a torch generator on ``device`` seeded with ``seed``."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_everything(seed: int) -> None:
    """Seed the global random streams of Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  # reason: third-party code may still use the legacy global stream
    torch.manual_seed(seed)
