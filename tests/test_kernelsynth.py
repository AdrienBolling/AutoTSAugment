import functools
import logging
import time
from pathlib import Path

import hydra.utils
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from autotsaugment.components.augmentation.bank import BANK_SIZE, KERNEL_NAMES, gram_matrices, grid
from autotsaugment.components.augmentation.kernelsynth import (
    KernelSynth,
    KernelSynthConfig,
    compose,
    compose_gram,
)

LOGGER = logging.getLogger(__name__)
LENGTH = 64
# A one-hot action with the default bound 4.0 puts only 0.9894 of the mass on the kernel
# (33 kernels). The one-hot tests use a larger bound, so the off-target mass is negligible.
ONE_HOT_BOUND = 20.0
# 0.999 quantile of the chi-square distribution with 32 degrees of freedom.
CHI2_32_999 = 62.49


def make(device: torch.device | str = "cpu", **overrides: float) -> KernelSynth:
    config = KernelSynthConfig.model_validate({"series_length": LENGTH, **overrides})
    return KernelSynth(config=config, device=device)


def one_hot(method: KernelSynth, kernel: str, count: int = 1) -> Tensor:
    bound = method.config.logit_bound
    action = torch.full((method.action_space.dim,), -bound, device=method.device)
    action[KERNEL_NAMES.index(kernel)] = bound
    action[BANK_SIZE + count - 1] = bound
    action[-1] = 0.0
    return action


def seeded(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


# --- composition ---------------------------------------------------------------------------


def test_compose_matches_a_left_to_right_reduce():
    bank = gram_matrices(8, dtype=torch.float64)
    generator = seeded(0)
    rows = 200
    kernels = torch.randint(0, BANK_SIZE, (rows, 5), generator=generator)
    mul = torch.rand((rows, 4), generator=generator) < 0.5
    counts = torch.randint(1, 6, (rows,), generator=generator)
    got = compose(bank, kernels, mul, counts)
    for row in range(rows):
        used = [bank[index] for index in kernels[row, : counts[row]].tolist()]
        steps = zip(mul[row].tolist(), used[1:], strict=False)
        expected = functools.reduce(
            lambda acc, step: acc * step[1] if step[0] else acc + step[1], steps, used[0]
        )
        torch.testing.assert_close(got[row], expected)
    diagonal = compose(bank.diagonal(dim1=-2, dim2=-1), kernels, mul, counts)
    torch.testing.assert_close(diagonal, got.diagonal(dim1=-2, dim2=-1))


def test_compose_gram_adds_the_jitter_to_the_diagonal_only():
    bank = gram_matrices(8)
    kernels = torch.tensor([[25, 3], [32, 0]])
    mul = torch.tensor([[True], [False]])
    counts = torch.tensor([2, 1])
    jitter = torch.tensor([0.5, 2.0])
    plain = compose(bank, kernels, mul, counts)
    jittered = compose_gram(bank, kernels, mul, counts, jitter)
    torch.testing.assert_close(jittered - plain, jitter[:, None, None] * torch.eye(8))


# --- action space and describe -------------------------------------------------------------


def test_action_space_layout():
    method = make()
    space = method.action_space
    assert space.dim == BANK_SIZE + 5 + 1 == 39
    assert space.names[0] == "kernel/00_periodic_24"
    assert space.names[21] == "kernel/21_linear_0"
    assert space.names[BANK_SIZE : BANK_SIZE + 5] == tuple(f"count/{j}" for j in range(1, 6))
    assert space.names[-1] == "op/mul"
    assert len(set(space.names)) == space.dim
    torch.testing.assert_close(space.low, torch.full((39,), -4.0))
    torch.testing.assert_close(space.high, torch.full((39,), 4.0))


@pytest.mark.parametrize("max_kernels", [1, 3])
def test_max_kernels_sets_the_count_dimensions(max_kernels: int):
    method = make(max_kernels=max_kernels)
    assert method.action_space.dim == BANK_SIZE + max_kernels + 1
    assert method.action_space.names[-2] == f"count/{max_kernels}"
    series = method.sample(method.uniform_action().expand(2, -1), 4, generator=seeded(0))
    assert series.shape == (2, 4, LENGTH)
    assert torch.isfinite(series).all()


def test_hydra_builds_the_method_from_the_config_file():
    path = Path(__file__).resolve().parents[1] / "configs/env/augmentation/kernelsynth.yaml"
    node = OmegaConf.load(path)
    node.device = "cpu"
    node.config.series_length = LENGTH
    method = hydra.utils.instantiate(node)
    assert isinstance(method, KernelSynth)
    assert method.config == KernelSynthConfig(series_length=LENGTH)


def test_uniform_action_is_zero_and_inside_the_box():
    method = make()
    action = method.uniform_action()
    assert torch.equal(action, torch.zeros(method.action_space.dim))
    assert method.action_space.contains(action[None])


def test_describe_the_uniform_action():
    method = make()
    stats = method.describe(method.uniform_action())
    assert tuple(stats) == method.action_space.names
    for name in KERNEL_NAMES:
        assert stats[f"kernel/{name}"] == pytest.approx(1 / BANK_SIZE)
    for count in range(1, 6):
        assert stats[f"count/{count}"] == pytest.approx(0.2)
    assert stats["op/mul"] == pytest.approx(0.5)


def test_describe_clips_the_action_to_the_box():
    method = make()
    action = torch.zeros(method.action_space.dim)
    action[0] = 100.0
    clipped = action.clamp(max=4.0)
    assert method.describe(action) == method.describe(clipped)


def test_wrong_inputs_raise():
    method = make()
    with pytest.raises(ValueError, match="dimension"):
        method.sample(torch.zeros(1, 7), 2, generator=seeded(0))
    with pytest.raises(ValueError, match="at least 1"):
        method.sample(method.uniform_action()[None], 0, generator=seeded(0))


# --- sample ---------------------------------------------------------------------------------


def test_sample_shapes_and_finite_values(device: torch.device, generator: torch.Generator):
    method = make(device)
    action = method.action_space.sample(3, generator=generator)
    series = method.sample(action, 5, generator=generator)
    assert series.shape == (3, 5, LENGTH)
    assert series.dtype == torch.float32
    assert series.device == device
    assert torch.isfinite(series).all()
    assert method.last_stage is not None
    assert method.last_stage.shape == (3, 5)
    assert (method.last_stage == 0).all()


def test_sample_is_deterministic_per_seed():
    method = make()
    action = method.uniform_action().expand(2, -1)
    first = method.sample(action, 16, generator=seeded(7))
    again = method.sample(action, 16, generator=seeded(7))
    other = method.sample(action, 16, generator=seeded(8))
    torch.testing.assert_close(first, again, rtol=0.0, atol=0.0)
    assert not torch.allclose(first, other)


def test_chunk_size_does_not_change_the_series():
    action = make().uniform_action().expand(2, -1)
    small = make(chunk_size=7).sample(action, 20, generator=seeded(3))
    large = make(chunk_size=256).sample(action, 20, generator=seeded(3))
    torch.testing.assert_close(small, large)


# --- distributions of the structure -------------------------------------------------------


def test_uniform_action_counts_are_uniform():
    method = make()
    n = 20000
    structure = method._draw_structure(method.uniform_action()[None], n, generator=seeded(0))
    frequency = torch.bincount(structure.counts.flatten(), minlength=6)[1:] / n
    standard_error = (0.2 * 0.8 / n) ** 0.5
    assert structure.counts.min() >= 1
    assert structure.counts.max() <= 5
    assert ((frequency - 0.2).abs() < 3 * standard_error).all(), frequency


def test_uniform_action_kernels_are_uniform():
    method = make()
    n = 20000
    structure = method._draw_structure(method.uniform_action()[None], n, generator=seeded(1))
    observed = torch.bincount(structure.kernels.flatten(), minlength=BANK_SIZE).double()
    expected = structure.kernels.numel() / BANK_SIZE
    chi2 = ((observed - expected) ** 2 / expected).sum()
    assert observed.numel() == BANK_SIZE
    assert chi2 < CHI2_32_999


def test_uniform_action_operators_are_fair():
    method = make()
    n = 20000
    structure = method._draw_structure(method.uniform_action()[None], n, generator=seeded(2))
    share = structure.mul.double().mean()
    standard_error = (0.25 / structure.mul.numel()) ** 0.5
    assert abs(share - 0.5) < 3 * standard_error


def test_action_sets_the_operator_probability():
    method = make()
    action = method.uniform_action()[None].clone()
    action[0, -1] = 2.0
    structure = method._draw_structure(action, 20000, generator=seeded(3))
    assert structure.mul.double().mean() == pytest.approx(
        torch.sigmoid(torch.tensor(2.0)), abs=0.01
    )


# --- one-hot actions -----------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["31_white_1", "32_constant", "17_periodic_12", "21_linear_0"])
def test_one_hot_action_uses_only_that_kernel(kernel: str):
    method = make(logit_bound=ONE_HOT_BOUND)
    action = one_hot(method, kernel)
    stats = method.describe(action)
    assert stats[f"kernel/{kernel}"] >= 0.99
    assert stats["count/1"] >= 0.99
    structure = method._draw_structure(action[None], 2000, generator=seeded(4))
    assert (structure.counts == 1).all()
    assert (structure.kernels[..., 0] == KERNEL_NAMES.index(kernel)).all()


def test_white_kernel_gives_uncorrelated_steps():
    method = make(logit_bound=ONE_HOT_BOUND)
    series = method.sample(one_hot(method, "31_white_1")[None], 1000, generator=seeded(5))[0]
    lag1 = (series[:, :-1] * series[:, 1:]).sum() / (series[:, :-1] ** 2).sum()
    assert abs(lag1) < 0.02
    assert series.var() == pytest.approx(1.0, abs=0.05)


def test_constant_kernel_gives_constant_series():
    method = make(logit_bound=ONE_HOT_BOUND)
    series = method.sample(one_hot(method, "32_constant")[None], 1000, generator=seeded(6))[0]
    assert series.var(dim=1).mean() < 1e-3
    assert series.mean(dim=1).var() == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize(
    ("kernel", "period"), [("12_periodic_4", 4), ("06_periodic_7", 7), ("17_periodic_12", 12)]
)
def test_periodic_kernel_repeats_after_its_period(kernel: str, period: int):
    method = make(logit_bound=ONE_HOT_BOUND)
    series = method.sample(one_hot(method, kernel)[None], 500, generator=seeded(7))[0]
    at_period = ((series[:, period:] - series[:, :-period]) ** 2).mean()
    half = period // 2
    at_half_period = ((series[:, half:] - series[:, :-half]) ** 2).mean()
    # The exact period is period * 63 / 64 steps, so the lag-period correlation is 0.995.
    assert at_period < 0.03
    assert at_half_period > 1.0


def test_linear_kernel_gives_lines_through_the_origin():
    method = make(logit_bound=ONE_HOT_BOUND)
    series = method.sample(one_hot(method, "21_linear_0")[None], 500, generator=seeded(8))[0]
    second_difference = series[:, 2:] - 2 * series[:, 1:-1] + series[:, :-2]
    assert second_difference.pow(2).mean().sqrt() < 0.03
    assert series[:, 0].pow(2).mean().sqrt() < 0.02
    assert series[:, -1].var() == pytest.approx(1.0, abs=0.2)


# --- jitter retries and fallback -----------------------------------------------------------


def test_retries_repair_failed_rows_in_fixed_batches():
    # A rank-1 Gram matrix with a 1e-9 relative jitter fails in float32; 1e-1 succeeds.
    method = make(logit_bound=ONE_HOT_BOUND, jitter=1e-9, max_jitter_retries=8, chunk_size=16)
    action = one_hot(method, "21_linear_0")[None]
    series = method.sample(action, 40, generator=seeded(9))
    stage = method.last_stage
    assert stage is not None
    assert torch.isfinite(series).all()
    assert (stage >= 1).all()
    assert (stage <= 8).all()
    assert (torch.bincount(stage.flatten()) <= 16).all()
    for attempt in stage.unique().tolist():
        direct = make(
            logit_bound=ONE_HOT_BOUND, jitter=1e-9 * 10.0**attempt, max_jitter_retries=0
        ).sample(action, 40, generator=seeded(9))
        rows = stage[0] == attempt
        torch.testing.assert_close(series[0, rows], direct[0, rows])


def test_fallback_gives_independent_draws_with_the_kernel_variance():
    method = make(logit_bound=ONE_HOT_BOUND, jitter=1e-9, max_jitter_retries=0)
    series = method.sample(one_hot(method, "22_linear_1")[None], 2000, generator=seeded(10))[0]
    assert method.last_stage is not None
    assert (method.last_stage == 1).all()
    assert torch.isfinite(series).all()
    variance = 1.0 + grid(LENGTH).float() ** 2  # DotProduct(sigma_0=1) diagonal
    torch.testing.assert_close(series.var(dim=0), variance, rtol=0.1, atol=0.0)
    lag1 = (series[:, :-1] * series[:, 1:]).mean() / series.pow(2).mean()
    assert abs(lag1) < 0.02


# --- GPU -------------------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.gpu
@needs_cuda
def test_gpu_compiled_composition_matches_eager():
    method = make("cuda")
    generator = seeded(0, "cuda")
    kernels = torch.randint(0, BANK_SIZE, (32, 5), generator=generator, device="cuda")
    mul = torch.rand((32, 4), generator=generator, device="cuda") < 0.5
    counts = torch.randint(1, 6, (32,), generator=generator, device="cuda")
    jitter = torch.rand(32, generator=generator, device="cuda")
    compiled = method._compose_gram(method._bank, kernels, mul, counts, jitter)
    eager = compose_gram(method._bank, kernels, mul, counts, jitter)
    torch.testing.assert_close(compiled, eager)
    del method, compiled, eager
    torch.cuda.empty_cache()


@pytest.mark.gpu
@needs_cuda
@pytest.mark.filterwarnings("ignore:Synchronization debug mode:UserWarning")
def test_gpu_throughput_memory_and_no_host_sync():
    method = KernelSynth(config=KernelSynthConfig(), device="cuda")
    generator = seeded(0, "cuda")
    action = method.uniform_action().expand(4, -1).clone()
    try:
        method.sample(action, 1024, generator=generator)  # warm-up: compiles the composition
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        series = method.sample(action, 1024, generator=generator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        LOGGER.info("KernelSynth 4 x 1024 x 1024: %.3f s, peak %.2f GiB", elapsed, peak_gib)
        assert series.shape == (4, 1024, 1024)
        assert torch.isfinite(series).all()
        assert elapsed < 2.0
        assert peak_gib < 4.0
        torch.cuda.set_sync_debug_mode("error")
        try:
            method.sample(action, 1024, generator=generator)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert method.last_stage is not None
        assert (method.last_stage == 0).all()
    finally:
        del method
        torch.cuda.empty_cache()
