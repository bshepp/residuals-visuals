"""
Compact 40-dim signature vector per residual.

Components:
  [0:32]  Radial-averaged log-power FFT spectrum (32 bins)   - texture/scale fingerprint
  [32]    mean(arr)
  [33]    std(arr)
  [34]    skew  (3rd standardized moment)
  [35]    kurtosis (excess, 4th standardized moment)
  [36]    p50(|arr|)
  [37]    p99(|arr|)
  [38]    Sobel edge density (mean |grad|)
  [39]    spatial autocorrelation at lag 1 (mean of horizontal+vertical)

Designed to cluster algorithms by visual character without sensitivity to
overall amplitude scale (which varies wildly between e.g. wavelet level 5 and
gaussian sigma 2).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel

SIGNATURE_DIM = 40


def _radial_log_spectrum(arr: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Radial-averaged log power spectrum, normalized to unit sum."""
    h, w = arr.shape
    # 2D FFT, magnitude, log-scaled, fftshift
    f = np.fft.fftshift(np.abs(np.fft.fft2(arr)))
    f = np.log1p(f)
    # Radial bins
    cy, cx = h / 2, w / 2
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - cy, xx - cx)
    r_max = min(cy, cx)
    # Bin by integer radius into n_bins
    edges = np.linspace(0, r_max, n_bins + 1)
    spectrum = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.any():
            spectrum[i] = float(f[mask].mean())
    # Normalize so the spectrum is amplitude-invariant
    s = spectrum.sum()
    if s > 0:
        spectrum /= s
    return spectrum


def signature(arr: np.ndarray) -> np.ndarray:
    """Compute the 40-dim signature for a 2D residual array."""
    a = arr.astype(np.float32, copy=False)
    flat = a.ravel()

    spec = _radial_log_spectrum(a, n_bins=32)

    mean = float(flat.mean())
    std = float(flat.std()) + 1e-12
    z = (flat - mean) / std
    skew = float((z**3).mean())
    kurt = float((z**4).mean() - 3.0)

    abs_a = np.abs(flat)
    p50 = float(np.median(abs_a))
    p99 = float(np.percentile(abs_a, 99))

    gx = sobel(a, axis=1, mode="reflect")
    gy = sobel(a, axis=0, mode="reflect")
    edge_density = float(np.hypot(gx, gy).mean())

    # Lag-1 spatial autocorrelation (Pearson) in horizontal + vertical
    if a.shape[1] > 1:
        ah = ((a[:, :-1] - mean) * (a[:, 1:] - mean)).mean() / (std**2)
    else:
        ah = 0.0
    if a.shape[0] > 1:
        av = ((a[:-1, :] - mean) * (a[1:, :] - mean)).mean() / (std**2)
    else:
        av = 0.0
    autocorr = float((ah + av) / 2)

    out = np.empty(SIGNATURE_DIM, dtype=np.float32)
    out[0:32] = spec
    out[32:40] = [mean, std, skew, kurt, p50, p99, edge_density, autocorr]
    return out
