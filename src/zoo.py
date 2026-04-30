"""
Algorithm zoo grid: one canonical exemplar per of 24 decomposition families,
all at bicubic_scale2, rendered as a labeled mosaic poster.

Reads from cache/thumbnails.zarr (populated by scripts/04_compute_signatures.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

import config
from src.catalog import load_catalog
from src.render import hillshade, symmetric_vmax, to_uint8

# Canonical exemplar per family (validated to exist in DATA_DIR).
ZOO_EXEMPLARS: list[tuple[str, str]] = [
    ("gaussian σ=10",                "gaussian_sigma10___bicubic_scale2.npy"),
    ("gauss-aniso σx=σy=10",         "gaussian_anisotropic_sigma_x10_sigma_y10___bicubic_scale2.npy"),
    ("bilateral d=9 σc=σs=75",       "bilateral_d9_sigma_color75_sigma_space75___bicubic_scale2.npy"),
    ("median size=5",                "median_size5___bicubic_scale2.npy"),
    ("uniform size=10",              "uniform_size10___bicubic_scale2.npy"),
    ("morph open size=20",           "morphological_operationaverage_size20___bicubic_scale2.npy"),
    ("morph-square s=10",            "morphological_square_operationopening_size10___bicubic_scale2.npy"),
    ("morph-diamond r=10",           "morphological_diamond_operationopening_radius10___bicubic_scale2.npy"),
    ("morph-rect 20×5",              "morphological_rect_height5_operationopening_width20___bicubic_scale2.npy"),
    ("morph-ellipse 20×10",          "morphological_ellipse_height10_operationopening_width20___bicubic_scale2.npy"),
    ("morph-gradient s=5",           "morphological_gradient_shapedisk_size5___bicubic_scale2.npy"),
    ("tophat-white s=20",            "tophat_modewhite_size20___bicubic_scale2.npy"),
    ("tophat-combined s=20",         "tophat_combined_size20___bicubic_scale2.npy"),
    ("rolling-ball r=50",            "rolling_ball_radius50___bicubic_scale2.npy"),
    ("polynomial deg=2",             "polynomial_degree2___bicubic_scale2.npy"),
    ("polynomial-high deg=4",        "polynomial_high_degree4___bicubic_scale2.npy"),
    ("local-poly w=51 deg=2",        "local_polynomial_degree2_window_size51___bicubic_scale2.npy"),
    ("DoG σ=2,10",                   "dog_sigma_high10_sigma_low2___bicubic_scale2.npy"),
    ("DoG-multi 4 scales",           "dog_multiscale_base_sigma1.0_n_scales4_sigma_ratio1.6___bicubic_scale2.npy"),
    ("LoG σ=5",                      "log_sigma5___bicubic_scale2.npy"),
    ("guided r=8 ε=0.01",            "guided_eps0.01_radius8___bicubic_scale2.npy"),
    ("aniso-diff κ=50 i=10",         "anisotropic_diffusion_gamma0.1_iterations10_kappa50___bicubic_scale2.npy"),
    ("wavelet-DWT db4 L=3",          "wavelet_dwt_level3_waveletdb4___bicubic_scale2.npy"),
    ("wavelet-bior 3.5 L=3",         "wavelet_biorthogonal_level3_waveletbior3.5___bicubic_scale2.npy"),
]


def _get_thumb(filename: str, catalog, thumbs: zarr.Array) -> np.ndarray:
    """Look up the cached thumbnail by filename."""
    matches = catalog.index[catalog["filename"] == filename]
    if len(matches) == 0:
        raise KeyError(filename)
    return np.asarray(thumbs[int(matches[0])])


def render_zoo(output_path: Path) -> None:
    catalog = load_catalog(config.CATALOG_PATH)
    if not config.THUMBNAILS_ZARR.exists():
        raise FileNotFoundError(
            f"Thumbnails cache not found at {config.THUMBNAILS_ZARR}. "
            "Run scripts/04_compute_signatures.py first."
        )
    thumbs = zarr.open(str(config.THUMBNAILS_ZARR), mode="r")

    # Hillshade panel from the source DEM (cropped + downsampled to thumb size)
    dem = np.load(config.SOURCE_DEM)
    hs = hillshade(dem)
    # DEM is half the scale=2 array in each dim
    cy = config.CROP_CY_S2 // 2
    cx = config.CROP_CX_S2 // 2
    half = config.CROP_SIZE_S2 // 4
    hs_crop = hs[cy - half : cy + half, cx - half : cx + half]
    from scipy.ndimage import zoom

    factor = config.THUMB_SIZE / hs_crop.shape[0]
    hs_thumb = zoom(hs_crop, factor, order=1, prefilter=False)[: config.THUMB_SIZE, : config.THUMB_SIZE]

    # Layout: 5 cols x 5 rows = 25 panels (1 hillshade + 24 algorithms)
    cols, rows = 5, 5
    fig_w, fig_h = cols * 3.0, rows * 3.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    axes[0].imshow(hs_thumb, cmap="gray", origin="upper", aspect="equal")
    axes[0].set_title("hillshade (DEM)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    missing = []
    for i, (label, filename) in enumerate(ZOO_EXEMPLARS, start=1):
        ax = axes[i]
        try:
            thumb = _get_thumb(filename, catalog, thumbs)
        except KeyError:
            missing.append(filename)
            ax.set_title(f"MISSING\n{label}", fontsize=8, color="red")
            ax.axis("off")
            continue
        vmax = symmetric_vmax(thumb)
        ax.imshow(thumb, cmap=config.RESIDUAL_CMAP, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    for j in range(len(ZOO_EXEMPLARS) + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "RESIDUALS: Algorithm Zoo  ·  same Fairfield County terrain seen by 24 decomposition algorithms\n"
        f"crop center=({config.CROP_CY_S2},{config.CROP_CX_S2}) size={config.CROP_SIZE_S2}² @ scale=2  ·  "
        f"upsampler fixed at bicubic ·  cmap=RdBu_r p99-clipped",
        fontsize=11,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    if missing:
        print(f"WARNING: {len(missing)} exemplars missing from cache: {missing}")
    print(f"Saved {output_path}")
