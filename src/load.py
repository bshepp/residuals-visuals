"""
Memory-mapped .npy loading and scale-aware cropping.

The crop window is defined in scale=2 coordinates and grows linearly with
upsampling factor. We slice through the mmap so we never load the full array
into memory — for a scale=16 file (24000x6000 float64, ~1.1 GB), we read only
the ~80 MB crop region from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

import config


def crop_window_for_scale(scale: int) -> tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) for a residual at given scale."""
    cy = config.CROP_CY_S2 * scale // 2
    cx = config.CROP_CX_S2 * scale // 2
    half = config.CROP_SIZE_S2 * scale // 4
    return cy - half, cy + half, cx - half, cx + half


def load_crop(path: Path | str, scale: int) -> np.ndarray:
    """
    Memory-map the .npy and return the cropped region as a regular ndarray.
    Only the crop region is materialized in memory.
    """
    arr = np.load(path, mmap_mode="r")
    r0, r1, c0, c1 = crop_window_for_scale(scale)
    # Bounds-clamp in case a smaller-than-expected file ever appears
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(arr.shape[0], r1)
    c1 = min(arr.shape[1], c1)
    return np.asarray(arr[r0:r1, c0:c1])


def load_thumb(path: Path | str, scale: int, thumb_size: int = config.THUMB_SIZE) -> np.ndarray:
    """
    Load the crop and downsample to (thumb_size, thumb_size) via order-1 zoom.
    Returns float32 to halve memory in the cache.
    """
    crop = load_crop(path, scale)
    # zoom factor goes from crop size down to thumb size
    zy = thumb_size / crop.shape[0]
    zx = thumb_size / crop.shape[1]
    thumb = zoom(crop, (zy, zx), order=1, prefilter=False)
    # Pad/trim if zoom rounding gave +/- 1 px
    if thumb.shape != (thumb_size, thumb_size):
        out = np.zeros((thumb_size, thumb_size), dtype=np.float32)
        h = min(thumb.shape[0], thumb_size)
        w = min(thumb.shape[1], thumb_size)
        out[:h, :w] = thumb[:h, :w]
        thumb = out
    return thumb.astype(np.float32, copy=False)
