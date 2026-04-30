"""
Parameter sweep videos. For each family with a clear scalar parameter axis,
walk through the parameter values, render each as a frame, and write an MP4.

Frames between adjacent parameter values are linearly interpolated to give
a smooth scrub feel (24 fps, 5-second loops).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import zarr

import config
from src.catalog import load_catalog
from src.render import symmetric_vmax, to_uint8


@dataclass
class Sweep:
    """One parameter sweep video definition."""

    name: str
    title: str
    decomp_family: str
    sweep_param: str  # the parameter being swept
    fixed_params: dict  # all other decomp params held at these values
    upsamp_method: str = "bicubic"
    upsamp_scale: int = 2
    fps: int = 24
    duration_per_step_sec: float = 0.7
    hold_first_last_sec: float = 0.4


# 6 sweeps spanning different algorithm flavors
SWEEPS: list[Sweep] = [
    Sweep(
        name="gaussian_sigma",
        title="gaussian: σ ∈ {2, 5, 10, 20, 50, 100}",
        decomp_family="gaussian",
        sweep_param="sigma",
        fixed_params={},
    ),
    Sweep(
        name="aniso_diff_iterations",
        title="anisotropic_diffusion: iterations ∈ {5, 10, 20, 50}  (κ=50, γ=0.1)",
        decomp_family="anisotropic_diffusion",
        sweep_param="iterations",
        fixed_params={"kappa": 50, "gamma": 0.1},
    ),
    Sweep(
        name="wavelet_bior_level",
        title="wavelet_biorthogonal: level ∈ {1, 2, 3, 4, 5}  (bior3.5)",
        decomp_family="wavelet_biorthogonal",
        sweep_param="level",
        fixed_params={"wavelet": "bior3.5"},
    ),
    Sweep(
        name="morph_size",
        title="morphological opening: size ∈ {5, 10, 20, 50, 100}",
        decomp_family="morphological",
        sweep_param="size",
        fixed_params={"operation": "opening"},
    ),
    Sweep(
        name="bilateral_sigma_space",
        title="bilateral: σ_space ∈ {25, 50, 75, 100, 150}  (d=9, σ_color=75)",
        decomp_family="bilateral",
        sweep_param="sigma_space",
        fixed_params={"d": 9, "sigma_color": 75},
    ),
    Sweep(
        name="dog_sigma_high",
        title="DoG: σ_high ∈ {5, 10, 20, 50, 100}  (σ_low=2)",
        decomp_family="dog",
        sweep_param="sigma_high",
        fixed_params={"sigma_low": 2},
    ),
]


def _select_frames(catalog, sweep: Sweep) -> list[tuple[float | int | str, str]]:
    """Find the catalog rows for a sweep, sorted by parameter value."""
    df = catalog[
        (catalog["decomp_family"] == sweep.decomp_family)
        & (catalog["upsamp_method"] == sweep.upsamp_method)
        & (catalog["scale"] == sweep.upsamp_scale)
    ]
    rows = []
    for _, row in df.iterrows():
        params = row["decomp_params"]
        if not all(params.get(k) == v for k, v in sweep.fixed_params.items()):
            continue
        if sweep.sweep_param not in params:
            continue
        rows.append((params[sweep.sweep_param], row["filename"]))
    rows.sort(key=lambda r: (isinstance(r[0], str), r[0]))
    return rows


def _interp_frames(thumbs: list[np.ndarray], n_between: int) -> list[np.ndarray]:
    """Linearly interpolate n_between frames between each adjacent pair."""
    out = []
    for i, t in enumerate(thumbs):
        out.append(t)
        if i < len(thumbs) - 1:
            for k in range(1, n_between + 1):
                alpha = k / (n_between + 1)
                out.append((1 - alpha) * t + alpha * thumbs[i + 1])
    return out


def render_sweep_video(catalog, thumbs_arr: zarr.Array, sweep: Sweep, output_path: Path) -> None:
    keyframes = _select_frames(catalog, sweep)
    if not keyframes:
        print(f"  SKIP {sweep.name}: no matching files")
        return

    print(f"  {sweep.name}: {len(keyframes)} keyframes")
    # Map filenames -> row_idx -> thumbnails
    name_to_idx = dict(zip(catalog["filename"], catalog.index))
    key_thumbs = []
    for value, filename in keyframes:
        idx = name_to_idx.get(filename)
        if idx is None:
            print(f"    missing: {filename}")
            continue
        key_thumbs.append(np.asarray(thumbs_arr[int(idx)]))

    if len(key_thumbs) < 2:
        print(f"  SKIP {sweep.name}: <2 frames")
        return

    # Use a SHARED vmax across the sweep (so changes feel like the same scene)
    stacked = np.stack(key_thumbs)
    shared_vmax = float(np.percentile(np.abs(stacked), config.PERCENTILE_CLIP))
    if shared_vmax < 1e-9:
        shared_vmax = 1.0

    # Tween between keyframes
    n_between = max(1, int(round(sweep.duration_per_step_sec * sweep.fps)) - 1)
    sequence = _interp_frames(key_thumbs, n_between)

    # Hold first and last
    n_hold = int(round(sweep.hold_first_last_sec * sweep.fps))
    full = [sequence[0]] * n_hold + sequence + [sequence[-1]] * n_hold

    # Render frames
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(config.RESIDUAL_CMAP)
    norm_frames = []
    for f in full:
        norm = np.clip((f + shared_vmax) / (2 * shared_vmax), 0, 1)
        rgb = cmap(norm)[:, :, :3]
        norm_frames.append(to_uint8(rgb))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=sweep.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=8,
    )
    try:
        for frame in norm_frames:
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"    wrote {output_path} ({len(norm_frames)} frames @ {sweep.fps} fps)")
