"""Cache of the window sets of each dataset id.

One file per dataset id: ``root/cache/<name>__<freq>__<sha1>.pt``. The SHA-1 hashes every
field that changes the windows of that dataset (see
``GiftEvalBenchmarkConfig.cache_fields``) and :data:`CACHE_VERSION`. A change of any of them
gives a new file name, so a stale file is never read. The file holds a dict of plain tensors
and strings, written with ``torch.save`` and read with ``torch.load(weights_only=True)``.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import torch

from autotsaugment.components.data.windows import SPLITS, SplitWindows

CACHE_DIR: Final = "cache"
CACHE_VERSION: Final = 1
FORMAT_ALLOWANCE_BYTES: Final = 64 * 1024  # torch.save adds about 6 kB to the tensor bytes


class CacheFormatError(ValueError):
    """A cache file does not hold the expected payload."""


def cache_key(fields: Mapping[str, object]) -> str:
    """Return the SHA-1 hex digest of ``fields`` and :data:`CACHE_VERSION`.

    Args:
        fields: JSON-serializable values that determine the windows.
    """
    text = json.dumps({"version": CACHE_VERSION, **fields}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()


def cache_path(root: Path, stem: str, key: str) -> Path:
    """Return ``root/cache/<stem>__<key>.pt``, with ``stem = "<name>__<freq>"``."""
    return root / CACHE_DIR / f"{stem}__{key}.pt"


def cache_file_bytes(windows: Mapping[str, SplitWindows]) -> int:
    """Return an upper bound of the size of the cache file of ``windows``, in bytes.

    The bound is the byte size of every tensor plus :data:`FORMAT_ALLOWANCE_BYTES` for the
    file format.
    """
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for split in SPLITS
        for tensor in windows[split].as_dict().values()
    )
    return tensor_bytes + FORMAT_ALLOWANCE_BYTES


def save_windows(path: Path, dataset_id: str, windows: Mapping[str, SplitWindows]) -> None:
    """Write the windows of one dataset id, atomically.

    Args:
        path: The cache file.
        dataset_id: The id ``"<name>/<freq>"``, stored for inspection.
        windows: The windows of each split.
    """
    payload = {
        "version": CACHE_VERSION,
        "dataset_id": dataset_id,
        "splits": tuple(SPLITS),
        **{split: windows[split].as_dict() for split in SPLITS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    torch.save(payload, partial)
    partial.replace(path)


def load_windows(path: Path) -> dict[str, SplitWindows]:
    """Read the windows of one dataset id.

    Args:
        path: The cache file.

    Returns:
        The windows of each split, on the CPU.

    Raises:
        CacheFormatError: The file does not hold a payload of :data:`CACHE_VERSION`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        msg = f"{path} is not a window cache of version {CACHE_VERSION}; delete it to rebuild"
        raise CacheFormatError(msg)
    try:
        return {split: SplitWindows.from_dict(payload[split]) for split in SPLITS}
    except KeyError as error:
        msg = f"{path} is incomplete ({error}); delete it to rebuild"
        raise CacheFormatError(msg) from error
