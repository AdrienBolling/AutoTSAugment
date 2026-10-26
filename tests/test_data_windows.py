import dataclasses
import math
from fractions import Fraction

import pytest
import torch

from autotsaugment.components.data.arrow import SeriesArrays
from autotsaugment.components.data.windows import (
    COMPARISON,
    REWARD,
    SplitBounds,
    SplitWindows,
    TooFewWindowsError,
    WindowSpec,
    assemble_window_set,
    build_dataset_windows,
    candidate_windows,
    check_window_counts,
    cut_windows,
    forward_fill,
    naive_scores,
    split_bounds,
    subsample,
    usable_ends,
    valid_windows,
)
from autotsaugment.core.seeding import make_generator

NAN = float("nan")
SPEC = WindowSpec(
    context_length=8, horizon=4, season=2, stride=4, max_context_length=12, max_horizon=6
)
LEVELS = (0.1, 0.5, 0.9)
LEVEL_TENSOR = torch.tensor(LEVELS)


def make_series(*rows: list[float]) -> SeriesArrays:
    values = torch.tensor([value for row in rows for value in row], dtype=torch.float32)
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lengths, 0)])
    return SeriesArrays(values=values, offsets=offsets, freq="H")


def bounds_of(start: list[int], end: list[int], context_start: list[int]) -> SplitBounds:
    return SplitBounds(
        start=torch.tensor(start), end=torch.tensor(end), context_start=torch.tensor(context_start)
    )


@pytest.mark.parametrize("cross", [True, False])
def test_split_bounds_use_exact_integer_floors(cross: bool):
    lengths = torch.tensor([10, 20, 51, 100, 212, 17420])
    expected_test = [int(Fraction(9, 10) * t) for t in lengths.tolist()]
    torch.testing.assert_close(usable_ends(lengths), torch.tensor(expected_test))
    bounds = split_bounds(lengths, 0.85, comparison_context_may_cross=cross)
    expected_reward = [int(Fraction(17, 20) * t) for t in expected_test]
    reward, comparison = bounds[REWARD], bounds[COMPARISON]
    torch.testing.assert_close(reward.end, torch.tensor(expected_reward))
    assert reward.start.tolist() == reward.context_start.tolist() == [0] * 6
    torch.testing.assert_close(comparison.start, reward.end)
    torch.testing.assert_close(comparison.end, torch.tensor(expected_test))
    expected_context = [0] * 6 if cross else expected_reward
    assert comparison.context_start.tolist() == expected_context
    # 0.7 * 90 is 62.99999999999999 in floating point; the exact floor is 63.
    assert math.floor(0.7 * 90) == 62
    short = split_bounds(torch.tensor([100]), 0.7, comparison_context_may_cross=cross)
    assert short[REWARD].end.item() == 63


def test_candidate_windows_are_anchored_at_the_split_end():
    bounds = bounds_of([0, 3], [20, 9], [0, 3])
    series, horizon_start = candidate_windows(bounds, SPEC)
    # series 0: span 20 - 4 - 2 = 14 -> 4 candidates; series 1: span 0 -> 1 candidate
    assert series.tolist() == [0, 0, 0, 0, 1]
    assert horizon_start.tolist() == [4, 8, 12, 16, 5]
    assert torch.all(horizon_start + SPEC.horizon <= bounds.end[series])
    assert torch.all(horizon_start - bounds.context_start[series] >= SPEC.season)


def test_candidate_windows_with_a_crossing_context():
    # horizons in [10, 15): only t = 11 fits; it needs a context that starts before 10
    strict, _ = candidate_windows(bounds_of([10], [15], [10]), SPEC)
    crossing, horizon_start = candidate_windows(bounds_of([10], [15], [0]), SPEC)
    assert strict.numel() == 0
    assert crossing.tolist() == [0]
    assert horizon_start.tolist() == [11]


def test_candidate_windows_skip_short_splits():
    series, horizon_start = candidate_windows(bounds_of([0], [5], [0]), SPEC)
    assert series.numel() == 0
    assert horizon_start.numel() == 0


def test_valid_windows_drop_nan_targets_and_sparse_contexts():
    row = [float(t) for t in range(24)]
    for t in range(12, 16):  # the whole target of the window at t = 12
        row[t] = NAN
    sparse = [NAN] * 23 + [1.0]
    data = make_series(row, sparse)
    series, horizon_start = valid_windows(data, bounds_of([0, 0], [24, 24], [0, 0]), SPEC)
    assert series.tolist() == [0, 0, 0, 0]
    assert horizon_start.tolist() == [4, 8, 16, 20]  # t = 12 dropped, series 1 dropped


def test_cut_windows_pads_and_never_reads_before_the_context_start():
    data = make_series([float(t) for t in range(30)])
    start = torch.tensor([10])
    context, target = cut_windows(data.values, torch.tensor([0]), torch.tensor([14]), start, SPEC)
    assert context.shape == (1, 12)
    assert target.shape == (1, 6)
    # context [max(10, 14 - 8), 14) = [10, 14), right-aligned; the rest is NaN
    assert torch.isnan(context[0, :8]).all()
    assert context[0, 8:].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert target[0, :4].tolist() == [14.0, 15.0, 16.0, 17.0]
    assert torch.isnan(target[0, 4:]).all()


def test_forward_fill_keeps_leading_nan():
    context = torch.tensor([[NAN, 1.0, NAN, NAN, 4.0, NAN], [2.0, NAN, 3.0, NAN, NAN, NAN]])
    filled = forward_fill(context)
    expected = torch.tensor([[NAN, 1.0, 1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 3.0, 3.0, 3.0, 3.0]])
    torch.testing.assert_close(filled, expected, equal_nan=True)


def test_naive_scores_match_a_hand_computation():
    context = torch.tensor([[NAN, NAN, 1.0, 2.0, NAN, 4.0]])
    target = torch.tensor([[3.0, NAN, 5.0, NAN]])
    loss, abs_sum = naive_scores(context, target, LEVEL_TENSOR, 2)
    # filled context [.., 1, 2, 2, 4]; season 2 -> forecast [2, 4, 2, 4]
    forecast = torch.tensor([2.0, 4.0, 2.0, 4.0])
    expected = 0.0
    for step, y in enumerate([3.0, NAN, 5.0, NAN]):
        if math.isnan(y):
            continue
        error = y - forecast[step].item()
        expected += sum(max(q * error, (q - 1) * error) for q in LEVEL_TENSOR.tolist())
    assert loss.item() == pytest.approx(expected)
    assert abs_sum.item() == pytest.approx(8.0)


def test_subsample_is_sorted_seeded_and_bounded():
    a = subsample(100, 10, make_generator("cpu", 1))
    b = subsample(100, 10, make_generator("cpu", 1))
    c = subsample(100, 10, make_generator("cpu", 2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.numel() == 10
    assert torch.all(a[1:] > a[:-1])
    assert subsample(5, 10, make_generator("cpu", 1)).tolist() == [0, 1, 2, 3, 4]


def _build(
    data: SeriesArrays,
    dataset_id: str,
    per_split: int = 3,
    *,
    cross: bool = True,
    min_windows: int = 1,
) -> dict[str, SplitWindows]:
    return build_dataset_windows(
        data,
        SPEC,
        dataset_id=dataset_id,
        reward_fraction=0.5,
        comparison_context_may_cross=cross,
        windows_per_split={REWARD: per_split, COMPARISON: per_split},
        min_windows=min_windows,
        levels=LEVELS,
        seed=0,
    )


@pytest.mark.parametrize("cross", [True, False])
def test_build_dataset_windows_respects_the_split_rules(cross: bool):
    rows = [[float(t) for t in range(length)] for length in (200, 150, 120)]
    data = make_series(*rows)
    windows = _build(data, "x/H", per_split=1000, cross=cross)
    test_start = usable_ends(data.lengths)
    reward_end = test_start // 2
    context_reads_reward = False
    for split, part in windows.items():
        assert part.n_windows > 0
        horizon_lo = torch.zeros_like(reward_end) if split == REWARD else reward_end
        horizon_hi = reward_end if split == REWARD else test_start
        context_lo = (
            reward_end if split == COMPARISON and not cross else torch.zeros_like(reward_end)
        )
        for row in range(part.n_windows):
            series = int(part.series[row])
            context = part.context[row][torch.isfinite(part.context[row])]
            target = part.target[row][torch.isfinite(part.target[row])]
            assert int(target.min()) >= int(horizon_lo[series])
            assert int(target.max()) < int(horizon_hi[series])
            assert int(context.min()) >= int(context_lo[series])
            assert int(context.max()) < int(target.min())
            if split == COMPARISON and int(context.min()) < int(reward_end[series]):
                context_reads_reward = True
    assert context_reads_reward == cross


def test_build_dataset_windows_subsample_depends_on_the_dataset_id():
    data = make_series(*[[float(t) for t in range(400)] for _ in range(4)])
    first = _build(data, "a/H")
    again = _build(data, "a/H")
    other = _build(data, "b/H")
    for split in (REWARD, COMPARISON):
        assert first[split].n_windows == 3
        assert torch.equal(first[split].horizon_start, again[split].horizon_start)
        assert torch.equal(first[split].series, again[split].series)
    picked = (first[REWARD].series * 1000 + first[REWARD].horizon_start).tolist()
    picked_other = (other[REWARD].series * 1000 + other[REWARD].horizon_start).tolist()
    assert picked != picked_other


def test_build_dataset_windows_raises_on_an_empty_split():
    data = make_series([float(t) for t in range(10)])
    with pytest.raises(
        TooFewWindowsError,
        match=r"x/H: too few windows in the reward split \(reward 0, comparison 1",
    ):
        _build(data, "x/H")


def test_build_dataset_windows_raises_below_the_minimum():
    # length 60: test_start 54, reward_end 27; comparison horizons start in [27, 50]
    data = make_series([float(t) for t in range(60)])
    assert _build(data, "x/H", min_windows=6)[COMPARISON].n_windows == 3
    with pytest.raises(
        TooFewWindowsError, match=r"\(reward 6, comparison 6, min_windows_per_split 7\)"
    ):
        _build(data, "x/H", min_windows=7)


def test_check_window_counts_names_every_short_split():
    check_window_counts("x/H", {REWARD: 5, COMPARISON: 5}, 5)
    with pytest.raises(
        TooFewWindowsError, match="x/H: too few windows in the reward and comparison"
    ):
        check_window_counts("x/H", {REWARD: 1, COMPARISON: 2}, 5)


def test_split_windows_dict_round_trip():
    data = make_series([float(t) for t in range(200)])
    part = _build(data, "x/H")[REWARD]
    assert part.quantile_levels == LEVELS
    again = SplitWindows.from_dict(part.as_dict(), quantile_levels=part.quantile_levels)
    for name, tensor in part.as_dict().items():
        torch.testing.assert_close(getattr(again, name), tensor, equal_nan=True)
    assert again.quantile_levels == LEVELS
    with pytest.raises(KeyError, match="missing window fields"):
        SplitWindows.from_dict({"context": part.context}, quantile_levels=LEVELS)


def test_assemble_window_set_carries_the_levels():
    data = make_series([float(t) for t in range(200)])
    parts = [_build(data, "a/H")[REWARD], _build(data, "b/H")[REWARD]]
    window_set = assemble_window_set(
        parts,
        dataset_names=("a/H", "b/H"),
        domains=("Energy", "Energy"),
        device=torch.device("cpu"),
    )
    assert window_set.quantile_levels == LEVELS
    other = dataclasses.replace(parts[1], quantile_levels=(0.25, 0.5, 0.75))
    with pytest.raises(ValueError, match="same quantile levels"):
        assemble_window_set(
            [parts[0], other],
            dataset_names=("a/H", "b/H"),
            domains=("Energy", "Energy"),
            device=torch.device("cpu"),
        )


def test_window_spec_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="exceeds the padded width"):
        WindowSpec(
            context_length=16, horizon=4, season=2, stride=4, max_context_length=8, max_horizon=4
        )
    with pytest.raises(ValueError, match="at least 1"):
        WindowSpec(
            context_length=8, horizon=4, season=2, stride=0, max_context_length=8, max_horizon=4
        )
