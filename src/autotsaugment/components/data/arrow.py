"""Read GIFT-Eval Arrow shards into flat univariate series, with pyarrow only.

The shards are Hugging Face ``datasets`` files: Arrow IPC streams with one row per item and
the columns ``item_id``, ``start``, ``freq``, ``target`` and sometimes
``past_feat_dynamic_real`` (ignored). ``target`` is ``list<float>`` for a univariate item and
a list of such lists (``fixed_size_list`` or ``list``) for a multivariate item. Each variate
becomes its own series, in item order, then variate order. NaN marks a missing value.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from pyarrow import ipc
from torch import Tensor


@dataclass(frozen=True)
class SeriesArrays:
    """The univariate series of one dataset, stored flat.

    Attributes:
        values: Values of all series, concatenated, float32, shape ``(total,)``. NaN marks a
            missing value.
        offsets: Start of each series in ``values``, int64, shape ``(n_series + 1,)``. Series
            ``i`` is ``values[offsets[i]:offsets[i + 1]]``.
        freq: The pandas frequency alias of the ``freq`` column, as stored.
    """

    values: Tensor
    offsets: Tensor
    freq: str

    @property
    def n_series(self) -> int:
        """Return the number of series."""
        return int(self.offsets.numel()) - 1

    @property
    def lengths(self) -> Tensor:
        """Return the length of each series, int64, shape ``(n_series,)``."""
        return self.offsets[1:] - self.offsets[:-1]

    def series(self, index: int) -> Tensor:
        """Return series ``index`` as a view of ``values``."""
        return self.values[int(self.offsets[index]) : int(self.offsets[index + 1])]


def read_table(path: Path) -> pa.Table:
    """Read one Arrow shard, as an IPC stream or, if that fails, as an IPC file."""
    with pa.OSFile(str(path)) as source:
        try:
            return ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_file(source).read_all()


def _is_list(kind: pa.DataType) -> bool:
    return (
        pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind)
    )


def _flatten_target(column: pa.ChunkedArray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the flat float32 values and the length of each variate of a ``target`` column."""
    kind = column.type
    if not _is_list(kind):
        msg = f"{path}: column 'target' has type {kind}, expected a list"
        raise TypeError(msg)
    nested = _is_list(kind.value_type)
    values: list[np.ndarray] = [np.empty(0, dtype=np.float32)]
    lengths: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    for chunk in column.chunks:
        variates = chunk.flatten() if nested else chunk
        counts = variates.value_lengths()
        if counts.null_count:
            msg = f"{path}: column 'target' has null series"
            raise ValueError(msg)
        flat = variates.flatten().cast(pa.float32()).fill_null(float("nan"))
        values.append(flat.to_numpy(zero_copy_only=False))
        lengths.append(counts.to_numpy(zero_copy_only=False))
    flat_values = np.concatenate(values).astype(np.float32, copy=False)
    return flat_values, np.concatenate(lengths).astype(np.int64, copy=False)


def _read_freq(table: pa.Table, path: Path) -> str:
    if "freq" not in table.column_names:
        msg = f"{path}: no 'freq' column; columns: {table.column_names}"
        raise ValueError(msg)
    frequencies = {freq for freq in table.column("freq").unique().to_pylist() if freq}
    if len(frequencies) != 1:
        msg = f"{path}: expected one frequency, found {sorted(frequencies)}"
        raise ValueError(msg)
    return frequencies.pop()


def read_series(paths: Sequence[Path]) -> SeriesArrays:
    """Read the shards of one dataset into flat univariate series.

    Args:
        paths: The shard files of the dataset, in order.

    Returns:
        The series of all shards, in shard order.

    Raises:
        ValueError: No shard is given, a shard has no ``target`` or ``freq`` column, the
            shards hold several frequencies, or a target is null.
        TypeError: The ``target`` column is not a list.
    """
    if not paths:
        msg = "read_series needs at least one shard"
        raise ValueError(msg)
    values: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    frequencies: set[str] = set()
    for path in paths:
        table = read_table(path)
        if "target" not in table.column_names:
            msg = f"{path}: no 'target' column; columns: {table.column_names}"
            raise ValueError(msg)
        flat, counts = _flatten_target(table.column("target"), path)
        values.append(flat)
        lengths.append(counts)
        frequencies.add(_read_freq(table, path))
    if len(frequencies) != 1:
        msg = f"the shards {list(paths)} hold several frequencies: {sorted(frequencies)}"
        raise ValueError(msg)
    counts = np.concatenate(lengths)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return SeriesArrays(
        values=torch.from_numpy(np.concatenate(values)),
        offsets=torch.from_numpy(offsets),
        freq=frequencies.pop(),
    )
