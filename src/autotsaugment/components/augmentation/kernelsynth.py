"""KernelSynth: Gaussian-process series from composed kernels, sampled in batches on the device.

This module ports the generator of ``scripts/kernel-synth.py`` in
`chronos-forecasting <https://github.com/amazon-science/chronos-forecasting>`_ (Ansari et al.
2024, Apache License 2.0). The original makes one series as follows:

1. Draw a count ``m`` uniformly in ``1..max_kernels``.
2. Draw ``m`` kernels from the bank, uniformly and with replacement.
3. Compose them left to right with ``functools.reduce``. Each step is ``+`` or ``*`` with equal
   probability.
4. Draw one sample of the zero-mean Gaussian process with this kernel on
   ``linspace(0, 1, LENGTH)``.

Here the action sets the three distributions of steps 1 to 3. The action is a box of
``BANK_SIZE + max_kernels + 1`` logits in ``[-logit_bound, logit_bound]``, 39 with the
defaults (33 kernels, 5 counts, 1 operator):

- ``kernel/<index>_<name>``: the kernel probabilities are ``softmax(action[:BANK_SIZE])``.
- ``count/<j>``: the count probabilities are ``softmax`` of the next ``max_kernels`` logits.
- ``op/mul``: the probability of ``*`` is ``sigmoid`` of the last logit.

The zero action gives the original recipe: uniform kernels, uniform count, ``P(*) = 0.5``.

The sampler never synchronizes with the host. It draws all counts, kernels, operators and
standard normal vectors first. Then it processes the series in chunks:

1. Compose the Gram matrix of each series as element-wise sums and products of the
   precomputed bank matrices. A position beyond the count is an identity step.
2. Add ``jitter * s`` to the diagonal, where ``s`` is the mean of the diagonal of the composed
   matrix (the mean prior variance of the series). A relative jitter keeps the same
   signal-to-jitter ratio for every kernel. The variance of the composed kernels spans more than
   five orders of magnitude. With the uniform action at 1024 steps, an absolute jitter of 1e-4
   fails for 7% of the series in float32; the relative jitter of 1e-4 failed for none of
   16384 series.
3. Factorize in float32 with ``torch.linalg.cholesky_ex`` and multiply the factor by the
   standard normal vector.

Then ``max_jitter_retries`` retry passes run, each with 10 times the jitter of the pass before.
The number of failed series is not known on the host, so each retry pass factorizes a fixed
batch of ``chunk_size`` series, the failed series first, and replaces only the failed series
with ``torch.where``. A series that still fails after the last pass gets an independent
Gaussian sample with the marginal variance of its kernel (the diagonal of its Gram matrix).
This fallback keeps every output finite at the cost of one element-wise operation. An
eigendecomposition fallback is not possible: ``torch.linalg.eigh`` synchronizes on CUDA to
check its convergence flag.

On CUDA, the Gram composition runs through :func:`torch.compile`, which fuses the bank lookups
and the element-wise steps into one kernel. On other devices it runs eagerly.
"""

from typing import NamedTuple

import torch
from pydantic import Field
from torch import Tensor

from autotsaugment.components.augmentation.bank import BANK_SIZE, KERNEL_NAMES, gram_matrices
from autotsaugment.core.augmentation import AugmentationConfig, AugmentationMethod
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.typechecking import Action, ActionBatch, SeriesBatchK, checked

JITTER_GROWTH = 10.0
"""Factor of the jitter from one factorization pass to the next."""

MAX_KERNELS_BOUND = 8
"""Largest valid ``max_kernels``. See :class:`KernelSynthConfig`."""


class KernelSynthConfig(AugmentationConfig):
    """Config of :class:`KernelSynth`.

    Attributes:
        series_length: Length of every series, in time steps.
        max_kernels: Largest number of bank kernels in one composed kernel, in ``1..8``. The
            bound keeps the composed Gram matrix inside the float32 range: a product of
            ``m`` copies of ``DotProduct(sigma_0=10)`` has entries near ``101**m``, which
            overflows at ``m = 20``. At ``m = 8`` the entries stay near ``1e16``.
        logit_bound: Bound of every action logit. The action box is
            ``[-logit_bound, logit_bound]``.
        jitter: Jitter of the first factorization pass, relative to the mean diagonal of the
            composed Gram matrix.
        max_jitter_retries: Number of retry passes after the first pass. Each pass multiplies
            the jitter by 10.
        chunk_size: Number of series in one batched Cholesky call, and in one retry pass.
    """

    series_length: int = Field(default=1024, ge=2)
    max_kernels: int = Field(default=5, ge=1, le=MAX_KERNELS_BOUND)
    logit_bound: float = Field(default=4.0, gt=0.0)
    jitter: float = Field(default=1e-4, gt=0.0)
    max_jitter_retries: int = Field(default=3, ge=0)
    chunk_size: int = Field(default=256, ge=1)


class KernelStructure(NamedTuple):
    """The random structure of a batch of composed kernels.

    Attributes:
        counts: Number of bank kernels per series, in ``1..max_kernels``, shape ``(k, n)``.
        kernels: Bank index per position, shape ``(k, n, max_kernels)``. Positions at or
            beyond the count are drawn but not used.
        mul: True where the step from position ``j`` to ``j + 1`` is a product, False where it
            is a sum, shape ``(k, n, max_kernels - 1)``.
    """

    counts: Tensor
    kernels: Tensor
    mul: Tensor


class Generated(NamedTuple):
    """Series of :meth:`KernelSynth._generate` and the factorization stage of each series.

    Attributes:
        series: The series, shape ``(r, series_length)``.
        stage: The pass that made each series, shape ``(r,)``: 0 for the first Cholesky pass,
            ``j`` for retry ``j``, and ``max_jitter_retries + 1`` for the independent fallback.
    """

    series: Tensor
    stage: Tensor


def compose(bank: Tensor, kernels: Tensor, mul: Tensor, counts: Tensor) -> Tensor:
    """Compose bank entries left to right with element-wise sums and products.

    Args:
        bank: One entry per bank kernel, shape ``(BANK_SIZE, *s)``, for example the Gram
            matrices ``(BANK_SIZE, t, t)`` or their diagonals ``(BANK_SIZE, t)``.
        kernels: Bank index per position, shape ``(r, m)``.
        mul: Operator per step, True for a product, shape ``(r, m - 1)``.
        counts: Number of used positions per row, in ``1..m``, shape ``(r,)``.

    Returns:
        The composed entries, shape ``(r, *s)``.
    """
    expand = (-1,) + (1,) * (bank.ndim - 1)
    result = bank[kernels[:, 0]]
    for position in range(1, kernels.shape[1]):
        term = bank[kernels[:, position]]
        active = (counts > position).view(expand)
        product = mul[:, position - 1].view(expand)
        result = torch.where(active, torch.where(product, result * term, result + term), result)
    return result


def compose_gram(
    bank: Tensor, kernels: Tensor, mul: Tensor, counts: Tensor, jitter: Tensor
) -> Tensor:
    """Compose Gram matrices and add a jitter to their diagonal.

    Args:
        bank: Gram matrices of the bank, shape ``(BANK_SIZE, t, t)``.
        kernels: Bank index per position, shape ``(r, m)``.
        mul: Operator per step, True for a product, shape ``(r, m - 1)``.
        counts: Number of used positions per row, shape ``(r,)``.
        jitter: Value added to the diagonal of each matrix, shape ``(r,)``.

    Returns:
        The jittered Gram matrices, shape ``(r, t, t)``.
    """
    gram = compose(bank, kernels, mul, counts)
    steps = torch.arange(gram.shape[-1], device=gram.device)
    diagonal = steps[:, None] == steps[None, :]
    return torch.where(diagonal, gram + jitter[:, None, None], gram)


class KernelSynth(AugmentationMethod[KernelSynthConfig]):
    """KernelSynth with an action over the kernel, count and operator distributions.

    See the module docstring for the recipe, the action layout and the sampler.

    Args:
        config: The method config.
        device: Device of the bank, the action space and every generated series.
    """

    def __init__(self, *, config: KernelSynthConfig, device: torch.device | str) -> None:
        self.config = config
        self._device = torch.device(device)
        self._bank = gram_matrices(config.series_length, device=self._device)
        self._bank_diagonal = self._bank.diagonal(dim1=-2, dim2=-1).clone()
        names = (
            *(f"kernel/{name}" for name in KERNEL_NAMES),
            *(f"count/{count}" for count in range(1, config.max_kernels + 1)),
            "op/mul",
        )
        bound = torch.full((len(names),), config.logit_bound, device=self._device)
        self._space = BoxSpace(low=-bound, high=bound.clone(), names=names)
        self._compose_gram = (
            torch.compile(compose_gram) if self._device.type == "cuda" else compose_gram
        )
        self._last_stage: Tensor | None = None

    @property
    def action_space(self) -> BoxSpace:
        """Return the box of kernel, count and operator logits."""
        return self._space

    @property
    def device(self) -> torch.device:
        """Return the device of the bank and of the series."""
        return self._device

    @property
    def last_stage(self) -> Tensor | None:
        """Return the factorization stage of each series of the last :meth:`sample` call.

        The tensor has shape ``(k, n)`` and stays on the device. Its values are the
        ``stage`` values of :class:`Generated`. It is None before the first call.
        """
        return self._last_stage

    def uniform_action(self) -> Action:
        """Return zeros: uniform kernels, uniform count and ``P(*) = 0.5``, as in Chronos."""
        return torch.zeros(self._space.dim, device=self._device)

    def probabilities(self, action: ActionBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Return the three distributions that a batch of actions sets.

        The action is clipped to the box first.

        Args:
            action: Actions, shape ``(k, a)``.

        Returns:
            The kernel probabilities ``(k, BANK_SIZE)``, the count probabilities
            ``(k, max_kernels)`` for the counts ``1..max_kernels``, and ``P(*)`` ``(k,)``.
        """
        if action.shape[-1] != self._space.dim:
            msg = (
                f"expected actions of dimension {self._space.dim}, got shape {tuple(action.shape)}"
            )
            raise ValueError(msg)
        bound = self.config.logit_bound
        logits = action.to(dtype=torch.float32).clamp(-bound, bound)
        split = BANK_SIZE + self.config.max_kernels
        kernel = torch.softmax(logits[:, :BANK_SIZE], dim=-1)
        count = torch.softmax(logits[:, BANK_SIZE:split], dim=-1)
        return kernel, count, torch.sigmoid(logits[:, split])

    @checked
    def sample(self, action: ActionBatch, n: int, *, generator: torch.Generator) -> SeriesBatchK:
        """Return ``n`` series per action, with no host synchronization.

        Args:
            action: One action per lockstep environment, shape ``(k, a)``, on :attr:`device`.
            n: Number of series per environment, at least 1.
            generator: Generator on :attr:`device` that owns all randomness of the call.

        Returns:
            Series of shape ``(k, n, series_length)``, float32, finite, on :attr:`device`.

        Raises:
            ValueError: If ``n`` is below 1 or the action dimension is wrong.
        """
        if n < 1:
            msg = f"n must be at least 1, got {n}"
            raise ValueError(msg)
        structure = self._draw_structure(action, n, generator=generator)
        k, m = action.shape[0], self.config.max_kernels
        rows = k * n
        noise = torch.randn((rows, self.series_length), generator=generator, device=self._device)
        generated = self._generate(
            structure.kernels.reshape(rows, m),
            structure.mul.reshape(rows, m - 1),
            structure.counts.reshape(rows),
            noise,
        )
        self._last_stage = generated.stage.view(k, n)
        return generated.series.view(k, n, self.series_length)

    def _draw_structure(
        self, action: ActionBatch, n: int, *, generator: torch.Generator
    ) -> KernelStructure:
        """Draw the counts, kernels and operators of ``n`` series per action.

        Each draw inverts the cumulative distribution of a uniform variable on the device.

        Args:
            action: Actions, shape ``(k, a)``.
            n: Number of series per environment.
            generator: Generator on :attr:`device`.

        Returns:
            The drawn structure, see :class:`KernelStructure`.
        """
        kernel_p, count_p, mul_p = self.probabilities(action)
        k, m = action.shape[0], self.config.max_kernels
        count_u = torch.rand((k, n), generator=generator, device=self._device)
        kernel_u = torch.rand((k, n * m), generator=generator, device=self._device)
        mul_u = torch.rand((k, n, m - 1), generator=generator, device=self._device)
        counts = _inverse_cdf(count_p, count_u) + 1
        kernels = _inverse_cdf(kernel_p, kernel_u).view(k, n, m)
        return KernelStructure(counts=counts, kernels=kernels, mul=mul_u < mul_p[:, None, None])

    def _generate(self, kernels: Tensor, mul: Tensor, counts: Tensor, noise: Tensor) -> Generated:
        """Return one Gaussian-process series per row of a flat structure.

        Args:
            kernels: Bank index per position, shape ``(r, max_kernels)``.
            mul: Operator per step, True for a product, shape ``(r, max_kernels - 1)``.
            counts: Number of used positions per row, shape ``(r,)``.
            noise: Standard normal draws, shape ``(r, series_length)``.

        Returns:
            The series and the stage that made each of them.
        """
        rows, chunk = counts.shape[0], self.config.chunk_size
        variance = compose(self._bank_diagonal, kernels, mul, counts)
        jitter = self.config.jitter * variance.mean(dim=-1)
        series = torch.empty_like(noise)
        failed = torch.empty(rows, dtype=torch.bool, device=self._device)
        for start in range(0, rows, chunk):
            part = slice(start, min(start + chunk, rows))
            series[part], failed[part] = self._factor_and_draw(
                kernels[part], mul[part], counts[part], jitter[part], noise[part]
            )
        stage = torch.zeros(rows, dtype=torch.int64, device=self._device)
        capacity = min(chunk, rows)
        for attempt in range(1, self.config.max_jitter_retries + 1):
            # Failed rows first; the stable sort keeps the call deterministic.
            order = torch.sort(failed.to(torch.uint8), descending=True, stable=True).indices
            batch = order[:capacity]
            draws, still_failed = self._factor_and_draw(
                kernels[batch],
                mul[batch],
                counts[batch],
                jitter[batch] * JITTER_GROWTH**attempt,
                noise[batch],
            )
            was_failed = failed[batch]
            fixed = was_failed & ~still_failed
            series.index_copy_(0, batch, torch.where(fixed[:, None], draws, series[batch]))
            failed.index_copy_(0, batch, was_failed & still_failed)
            stage.index_copy_(0, batch, torch.where(fixed, attempt, stage[batch]))
        independent = variance.clamp(min=0.0).sqrt() * noise
        series = torch.where(failed[:, None], independent, series)
        stage = torch.where(failed, self.config.max_jitter_retries + 1, stage)
        return Generated(series=series, stage=stage)

    def _factor_and_draw(
        self, kernels: Tensor, mul: Tensor, counts: Tensor, jitter: Tensor, noise: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the series of one batch and a flag per series that failed to factorize."""
        gram = self._compose_gram(self._bank, kernels, mul, counts, jitter)
        factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
        del gram  # free the Gram matrices before the product
        draws = (factor @ noise.unsqueeze(-1)).squeeze(-1)
        failed = (info != 0) | ~torch.isfinite(draws).all(dim=-1)
        return draws, failed

    def describe(self, action: Action) -> dict[str, float]:
        """Return the kernel, count and operator probabilities of one action, by action name.

        This method synchronizes with the host.
        """
        kernel, count, mul = self.probabilities(action.reshape(1, -1))
        values = torch.cat([kernel[0], count[0], mul]).detach().cpu().tolist()
        return dict(zip(self._space.names, values, strict=True))


def _inverse_cdf(probabilities: Tensor, uniform: Tensor) -> Tensor:
    """Return category indices for uniform draws, batched over the first dimension.

    Args:
        probabilities: One distribution per row, shape ``(k, c)``.
        uniform: Draws in ``[0, 1)``, shape ``(k, d)``.

    Returns:
        Indices in ``0..c-1``, shape ``(k, d)``, int64.
    """
    cdf = probabilities.cumsum(dim=-1)
    index = torch.searchsorted(cdf, uniform, right=True)
    return index.clamp_(max=probabilities.shape[-1] - 1)
