import dataclasses

import pytest
import torch

from autotsaugment.core.benchmark import WindowSet


def make(n_per: int = 5, d: int = 3) -> WindowSet:
    n = n_per * d
    return WindowSet(
        context=torch.arange(n * 4, dtype=torch.float32).view(n, 4),
        target=torch.zeros(n, 2),
        dataset_index=torch.arange(d).repeat_interleave(n_per),
        naive_loss_sum=torch.ones(n),
        abs_sum=torch.ones(n),
        dataset_names=tuple(f"d{i}" for i in range(d)),
        domains=("Energy",) * d,
        quantile_levels=(0.1, 0.5, 0.9),
    )


def test_subset_keeps_at_most_per_dataset(generator: torch.Generator):
    ws = make()
    sub = ws.subset(2, generator=generator)
    assert sub.n_windows == 6
    counts = torch.bincount(sub.dataset_index, minlength=3)
    torch.testing.assert_close(counts, torch.tensor([2, 2, 2]))
    assert sub.dataset_names == ws.dataset_names
    full = ws.subset(100, generator=generator)
    assert full.n_windows == ws.n_windows


def test_select_and_properties():
    ws = make()
    sel = ws.select(torch.tensor([0, 14]))
    assert sel.n_windows == 2
    assert sel.context_length == 4
    assert sel.horizon == 2
    assert sel.n_datasets == 3
    torch.testing.assert_close(sel.dataset_index, torch.tensor([0, 2]))


def test_quantile_levels_are_carried(generator: torch.Generator):
    ws = make()
    assert ws.quantile_levels == (0.1, 0.5, 0.9)
    assert ws.to("cpu").quantile_levels == ws.quantile_levels
    assert ws.select(torch.tensor([1, 2])).quantile_levels == ws.quantile_levels
    assert ws.subset(2, generator=generator).quantile_levels == ws.quantile_levels
    with pytest.raises(ValueError, match="at least one level"):
        dataclasses.replace(ws, quantile_levels=())


def test_rows_by_dataset_is_computed_once():
    ws = make(n_per=4, d=3)
    rows = ws.rows_by_dataset
    assert len(rows) == 3
    for dataset, values in enumerate(rows):
        torch.testing.assert_close(values, torch.arange(4) + 4 * dataset)
    assert ws.rows_by_dataset is rows
    assert ws.to("cpu").rows_by_dataset is not rows
    empty = make(n_per=2, d=2).select(torch.tensor([0, 1]))
    assert [part.numel() for part in empty.rows_by_dataset] == [2, 0]
