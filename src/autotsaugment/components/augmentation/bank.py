"""The KernelSynth kernel bank: 33 Gaussian-process kernels, ported to torch.

This module ports ``KERNEL_BANK`` of ``scripts/kernel-synth.py`` in
`chronos-forecasting <https://github.com/amazon-science/chronos-forecasting>`_ (Ansari et al.
2024, Apache License 2.0). The original bank has 33 kernels: 21 ``ExpSineSquared``, 3
``DotProduct``, 3 ``RBF``, 3 ``RationalQuadratic``, 2 ``WhiteKernel`` and 1
``ConstantKernel``. The original evaluates scikit-learn kernels on the grid
``X = linspace(0, 1, LENGTH)`` with ``LENGTH = 1024``. This port uses the same grid,
``linspace(0, 1, series_length)``, the same order and the same comments, and the
scikit-learn formulas, with ``r = |x - x'|``:

- ``ExpSineSquared(length_scale=1, periodicity=p) = exp(-2 sin^2(pi r / p))``
- ``DotProduct(sigma_0) = sigma_0^2 + x x'``
- ``RBF(length_scale=l) = exp(-r^2 / (2 l^2))``
- ``RationalQuadratic(alpha, length_scale=1) = (1 + r^2 / (2 alpha))^(-alpha)``
- ``WhiteKernel(noise_level) = noise_level * [x == x']``
- ``ConstantKernel() = 1``

Periods are stored in time steps. The periodicity in grid units is
``period_steps / series_length``, as in the original (``24 / LENGTH``). Thus a different
``series_length`` keeps the same period in steps. The grid step is
``1 / (series_length - 1)``, so the exact period is ``period_steps * (series_length - 1) /
series_length`` steps: 0.1% short of the nominal period at 1024 steps, as in the original.
The other kernels keep the original grid units: their scale relative to the series length
does not change with ``series_length``.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


class KernelKind(StrEnum):
    """The six scikit-learn kernel families of the bank."""

    PERIODIC = "periodic"
    LINEAR = "linear"
    RBF = "rbf"
    RATIONAL_QUADRATIC = "rq"
    WHITE = "white"
    CONSTANT = "constant"


@dataclass(frozen=True)
class BankKernel:
    """One kernel of the bank.

    Attributes:
        kind: The kernel family.
        parameter: The one free hyperparameter of the family: the period in time steps
            (periodic), ``sigma_0`` (linear), ``length_scale`` (rbf), ``alpha`` (rq),
            ``noise_level`` (white). Ignored for the constant kernel.
        comment: The comment of the original bank entry, empty when it has none.
    """

    kind: KernelKind
    parameter: float = 1.0
    comment: str = ""

    @property
    def short_name(self) -> str:
        """Return a readable name, for example ``periodic_24`` or ``rbf_0.1``."""
        if self.kind is KernelKind.CONSTANT:
            return "constant"
        value = int(self.parameter) if self.parameter.is_integer() else self.parameter
        return f"{self.kind.value}_{value}"

    def gram(self, grid: Tensor) -> Tensor:
        """Return the Gram matrix of the kernel on ``grid``.

        Args:
            grid: The input points, shape ``(t,)``, equally spaced from 0 to 1.

        Returns:
            The matrix ``k(grid[i], grid[j])``, shape ``(t, t)``, with the dtype of ``grid``.
        """
        length = grid.shape[0]
        distance = (grid[:, None] - grid[None, :]).abs()
        match self.kind:
            case KernelKind.PERIODIC:
                periodicity = self.parameter / length
                return torch.exp(-2.0 * torch.sin(math.pi * distance / periodicity) ** 2)
            case KernelKind.LINEAR:
                return self.parameter**2 + grid[:, None] * grid[None, :]
            case KernelKind.RBF:
                return torch.exp(-(distance**2) / (2.0 * self.parameter**2))
            case KernelKind.RATIONAL_QUADRATIC:
                return (1.0 + distance**2 / (2.0 * self.parameter)) ** (-self.parameter)
            case KernelKind.WHITE:
                return self.parameter * torch.eye(length, dtype=grid.dtype, device=grid.device)
            case KernelKind.CONSTANT:
                return torch.ones((length, length), dtype=grid.dtype, device=grid.device)


_P = KernelKind.PERIODIC

KERNEL_BANK: tuple[BankKernel, ...] = (
    BankKernel(_P, 24, "H"),
    BankKernel(_P, 48, "0.5H"),
    BankKernel(_P, 96, "0.25H"),
    BankKernel(_P, 24 * 7, "H"),
    BankKernel(_P, 48 * 7, "0.5H"),
    BankKernel(_P, 96 * 7, "0.25H"),
    BankKernel(_P, 7, "D"),
    BankKernel(_P, 14, "0.5D"),
    BankKernel(_P, 30, "D"),
    BankKernel(_P, 60, "0.5D"),
    BankKernel(_P, 365, "D"),
    BankKernel(_P, 365 * 2, "0.5D"),
    BankKernel(_P, 4, "W"),
    BankKernel(_P, 26, "W"),
    BankKernel(_P, 52, "W"),
    BankKernel(_P, 4, "M"),
    BankKernel(_P, 6, "M"),
    BankKernel(_P, 12, "M"),
    BankKernel(_P, 4, "Q"),
    BankKernel(_P, 4 * 10, "Q"),
    BankKernel(_P, 10, "Y"),
    BankKernel(KernelKind.LINEAR, 0.0),
    BankKernel(KernelKind.LINEAR, 1.0),
    BankKernel(KernelKind.LINEAR, 10.0),
    BankKernel(KernelKind.RBF, 0.1),
    BankKernel(KernelKind.RBF, 1.0),
    BankKernel(KernelKind.RBF, 10.0),
    BankKernel(KernelKind.RATIONAL_QUADRATIC, 0.1),
    BankKernel(KernelKind.RATIONAL_QUADRATIC, 1.0),
    BankKernel(KernelKind.RATIONAL_QUADRATIC, 10.0),
    BankKernel(KernelKind.WHITE, 0.1),
    BankKernel(KernelKind.WHITE, 1.0),
    BankKernel(KernelKind.CONSTANT),
)
"""The kernels of the original bank, in the original order, with the original comments."""

BANK_SIZE = len(KERNEL_BANK)
"""Number of kernels in the bank."""

KERNEL_NAMES: tuple[str, ...] = tuple(
    f"{index:02d}_{kernel.short_name}" for index, kernel in enumerate(KERNEL_BANK)
)
"""Unique kernel names: the bank index and the short name, for example ``00_periodic_24``."""


def grid(series_length: int, *, device: torch.device | str = "cpu") -> Tensor:
    """Return the input grid ``linspace(0, 1, series_length)`` in float64."""
    return torch.linspace(0.0, 1.0, series_length, dtype=torch.float64, device=device)


def gram_matrices(
    series_length: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the Gram matrix of every bank kernel on the grid.

    The matrices are computed in float64 and then cast to ``dtype``.

    Args:
        series_length: Number of grid points, t.
        device: Device of the result.
        dtype: Dtype of the result.

    Returns:
        The stacked matrices, shape ``(BANK_SIZE, t, t)``.
    """
    points = grid(series_length, device=device)
    return torch.stack([kernel.gram(points).to(dtype) for kernel in KERNEL_BANK])
