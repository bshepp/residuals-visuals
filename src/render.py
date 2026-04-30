"""
Shared rendering helpers: percentile-clipped symmetric normalization and
RdBu_r colormap application. Matches the RESIDUALS visualization conventions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config


def symmetric_vmax(arr: np.ndarray, percentile: float = config.PERCENTILE_CLIP) -> float:
    """99th-pctile of |arr|, with floor at 1e-9 to avoid divide-by-zero."""
    vmax = float(np.percentile(np.abs(arr), percentile))
    return vmax if vmax > 1e-9 else 1.0


def to_rgb(
    arr: np.ndarray,
    cmap: str = config.RESIDUAL_CMAP,
    vmax: float | None = None,
) -> np.ndarray:
    """Map a 2D residual array to RGB float32 in [0,1] using symmetric normalization."""
    if vmax is None:
        vmax = symmetric_vmax(arr)
    norm = np.clip((arr + vmax) / (2 * vmax), 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(norm)[:, :, :3]
    return rgb.astype(np.float32, copy=False)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0,1] to uint8 [0,255]."""
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def hillshade(dem: np.ndarray, az_deg: float = 315, alt_deg: float = 45) -> np.ndarray:
    """Standard 315° NW illumination hillshade, normalized to [0,1]."""
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    dy, dx = np.gradient(dem.astype(np.float64))
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    s = np.cos(alt) * np.sin(slope) * np.cos(az - aspect) + np.sin(alt) * np.cos(slope)
    return ((s - s.min()) / (s.max() - s.min() + 1e-12)).astype(np.float32)
