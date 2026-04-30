"""
UMAP atlas: embed all 39,731 residuals into 2D using their 40-dim signatures,
then render as a thumbnail mosaic colored by decomposition family category.

Two outputs:
  - umap_scatter.png  (small, every point a colored dot)
  - umap_atlas.png    (large, every point is its 64x64 thumbnail in a packed grid)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import to_rgb as mpl_to_rgb

import config
from src.catalog import load_catalog
from src.render import symmetric_vmax

# Map decomp_family -> category color (loosely follows RESIDUALS' categorization)
FAMILY_CATEGORY = {
    "gaussian": "classical",
    "gaussian_anisotropic": "classical",
    "uniform": "classical",
    "median": "edge_preserving",
    "bilateral": "edge_preserving",
    "guided": "edge_preserving",
    "anisotropic_diffusion": "edge_preserving",
    "wavelet_dwt": "wavelet",
    "wavelet_biorthogonal": "wavelet",
    "wavelet_reverse_biorthogonal": "wavelet",
    "morphological": "morphological",
    "morphological_square": "morphological",
    "morphological_diamond": "morphological",
    "morphological_rect": "morphological",
    "morphological_ellipse": "morphological",
    "morphological_gradient": "morphological",
    "tophat": "morphological",
    "tophat_combined": "morphological",
    "rolling_ball": "morphological",
    "polynomial": "trend_removal",
    "polynomial_high": "trend_removal",
    "local_polynomial": "trend_removal",
    "dog": "multiscale",
    "dog_multiscale": "multiscale",
    "log": "multiscale",
}

CATEGORY_COLORS = {
    "classical": "#4a7bd1",
    "edge_preserving": "#d14a7b",
    "wavelet": "#7bd14a",
    "morphological": "#d1a04a",
    "trend_removal": "#9a4ad1",
    "multiscale": "#4ad1c8",
}


def fit_umap(signatures: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Fit UMAP on the 40-dim signature matrix, return Nx2 coords."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(signatures)


def render_scatter(coords: np.ndarray, families: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12), facecolor="black")
    ax.set_facecolor("black")
    for category, color in CATEGORY_COLORS.items():
        mask = families.map(FAMILY_CATEGORY).eq(category).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            c=color,
            alpha=0.5,
            label=f"{category} (n={int(mask.sum())})",
            linewidths=0,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", frameon=False, labelcolor="white", fontsize=10)
    ax.set_title(
        f"UMAP of {len(coords):,} residuals  ·  40-dim signature (radial FFT + moments)",
        color="white",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  wrote {output_path}")


def _grid_pack(coords: np.ndarray, cell_size: int, canvas_cells: int = 200) -> np.ndarray:
    """
    Snap UMAP coords to a `canvas_cells x canvas_cells` grid; on collisions, keep
    the first arrival (stable to a deterministic sort by index). Returns
    integer (row, col) for each input point, or -1 if dropped due to collision.
    """
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = cmax - cmin + 1e-9
    norm = (coords - cmin) / span  # in [0,1]
    cells = np.clip((norm * canvas_cells).astype(int), 0, canvas_cells - 1)

    occupied = np.full((canvas_cells, canvas_cells), -1, dtype=np.int64)
    placement = np.full(len(coords), -1, dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        if occupied[r, c] == -1:
            occupied[r, c] = i
            placement[i] = r * canvas_cells + c
        else:
            # Try a tiny spiral search for an empty cell
            placed = False
            for radius in range(1, 6):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < canvas_cells and 0 <= cc < canvas_cells and occupied[rr, cc] == -1:
                            occupied[rr, cc] = i
                            placement[i] = rr * canvas_cells + cc
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
    return placement


def render_atlas(
    coords: np.ndarray,
    families: pd.Series,
    thumbs: zarr.Array,
    output_path: Path,
    thumb_px: int = 64,
    canvas_cells: int = 220,
    border_px: int = 2,
) -> None:
    """
    Render a thumbnail mosaic: each point becomes its 64x64 thumbnail at its
    UMAP grid position, with a colored border indicating its category.
    """
    placements = _grid_pack(coords, cell_size=thumb_px, canvas_cells=canvas_cells)

    cell_total = thumb_px + 2 * border_px
    canvas_h = canvas_cells * cell_total
    canvas_w = canvas_cells * cell_total
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    cmap = plt.get_cmap(config.RESIDUAL_CMAP)
    placed = 0
    for i, p in enumerate(placements):
        if p == -1:
            continue
        cell_r, cell_c = divmod(int(p), canvas_cells)
        # Resize-on-the-fly: cached thumbs are 256x256, downsample to thumb_px
        t = np.asarray(thumbs[i])
        if t.shape[0] != thumb_px:
            stride = t.shape[0] // thumb_px
            t = t[::stride, ::stride][:thumb_px, :thumb_px]
        vmax = symmetric_vmax(t)
        norm = np.clip((t + vmax) / (2 * vmax), 0, 1)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

        # Category border
        family = families.iloc[i]
        category = FAMILY_CATEGORY.get(family, "classical")
        border = (np.array(mpl_to_rgb(CATEGORY_COLORS[category])) * 255).astype(np.uint8)

        y0 = cell_r * cell_total
        x0 = cell_c * cell_total
        canvas[y0 : y0 + cell_total, x0 : x0 + cell_total] = border  # fill border
        canvas[y0 + border_px : y0 + border_px + thumb_px, x0 + border_px : x0 + border_px + thumb_px] = rgb
        placed += 1

    print(f"  placed {placed}/{len(coords)} thumbnails (rest collided)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(str(output_path), canvas)
    print(f"  wrote {output_path} ({canvas_w}x{canvas_h})")
