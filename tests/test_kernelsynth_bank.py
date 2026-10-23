import pytest
import torch

from autotsaugment.components.augmentation.bank import (
    BANK_SIZE,
    KERNEL_BANK,
    KERNEL_NAMES,
    KernelKind,
    gram_matrices,
    grid,
)

LENGTH = 64


def test_bank_layout():
    assert BANK_SIZE == 33
    kinds = [kernel.kind for kernel in KERNEL_BANK]
    assert kinds.count(KernelKind.PERIODIC) == 21
    assert kinds[21:] == [KernelKind.LINEAR] * 3 + [KernelKind.RBF] * 3 + [
        KernelKind.RATIONAL_QUADRATIC
    ] * 3 + [KernelKind.WHITE] * 2 + [KernelKind.CONSTANT]
    assert KERNEL_NAMES[0] == "00_periodic_24"
    assert KERNEL_NAMES[21] == "21_linear_0"
    assert KERNEL_NAMES[24] == "24_rbf_0.1"
    assert KERNEL_NAMES[32] == "32_constant"
    assert len(set(KERNEL_NAMES)) == BANK_SIZE


def test_bank_matches_scikit_learn():
    kernels = pytest.importorskip("sklearn.gaussian_process.kernels")
    periods = (24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 4, 6, 12, 4, 40, 10)
    reference = [kernels.ExpSineSquared(periodicity=p / LENGTH) for p in periods]
    reference += [kernels.DotProduct(sigma_0=s) for s in (0.0, 1.0, 10.0)]
    reference += [kernels.RBF(length_scale=s) for s in (0.1, 1.0, 10.0)]
    reference += [kernels.RationalQuadratic(alpha=a) for a in (0.1, 1.0, 10.0)]
    reference += [kernels.WhiteKernel(noise_level=s) for s in (0.1, 1.0)]
    reference += [kernels.ConstantKernel()]
    points = grid(LENGTH).numpy()[:, None]
    expected = torch.stack([torch.from_numpy(kernel(points)) for kernel in reference])
    torch.testing.assert_close(gram_matrices(LENGTH, dtype=torch.float64), expected)


@pytest.mark.parametrize("length", [64, 256])
def test_periods_are_in_time_steps(length: int):
    bank = gram_matrices(length, dtype=torch.float64)
    for index, kernel in enumerate(KERNEL_BANK):
        period = int(kernel.parameter)
        if kernel.kind is KernelKind.PERIODIC and 2 * period < length:
            assert bank[index, 0, period] > 0.99, KERNEL_NAMES[index]
            assert bank[index, 0, period // 2] < 0.2, KERNEL_NAMES[index]


def test_gram_matrices_are_symmetric_and_on_the_requested_dtype():
    bank = gram_matrices(16)
    assert bank.shape == (BANK_SIZE, 16, 16)
    assert bank.dtype == torch.float32
    torch.testing.assert_close(bank, bank.mT)
