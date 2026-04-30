"""
Overview: hillshade + 24 representative residuals (one per decomposition family),
all at bicubic_scale2. Used to choose the crop center for zoo / sweeps / atlas.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEM_PATH = Path("F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy")
RESULTS_DIR = Path("D:/DIVERGE_exhaustive/results")
OUT_DIR = Path(__file__).parent

# 24 family exemplars at bicubic_scale2 (defaults-ish params)
SAMPLES = [
    ("gaussian", "gaussian_sigma10___bicubic_scale2.npy"),
    ("gauss-aniso", "gaussian_anisotropic_sigma_x10_sigma_y10___bicubic_scale2.npy"),
    ("bilateral", "bilateral_d9_sigma_color75_sigma_space75___bicubic_scale2.npy"),
    ("median", "median_size5___bicubic_scale2.npy"),
    ("uniform", "uniform_size10___bicubic_scale2.npy"),
    ("morph-disk", "morphological_operationaverage_size20___bicubic_scale2.npy"),
    ("morph-square", "morphological_square_operationopening_size10___bicubic_scale2.npy"),
    ("morph-diamond", "morphological_diamond_operationopening_radius10___bicubic_scale2.npy"),
    ("morph-rect", "morphological_rect_height5_operationopening_width20___bicubic_scale2.npy"),
    ("morph-ellipse", "morphological_ellipse_height10_operationopening_width20___bicubic_scale2.npy"),
    ("morph-grad", "morphological_gradient_shapedisk_size5___bicubic_scale2.npy"),
    ("tophat-w", "tophat_modewhite_size20___bicubic_scale2.npy"),
    ("tophat-comb", "tophat_combined_size20___bicubic_scale2.npy"),
    ("rolling-ball", "rolling_ball_radius50___bicubic_scale2.npy"),
    ("poly", "polynomial_degree2___bicubic_scale2.npy"),
    ("poly-high", "polynomial_high_degree4___bicubic_scale2.npy"),
    ("local-poly", "local_polynomial_degree2_window_size51___bicubic_scale2.npy"),
    ("dog", "dog_sigma_high10_sigma_low2___bicubic_scale2.npy"),
    ("dog-multi", "dog_multiscale_base_sigma1.0_n_scales4_sigma_ratio1.6___bicubic_scale2.npy"),
    ("log", "log_sigma5___bicubic_scale2.npy"),
    ("guided", "guided_eps0.01_radius8___bicubic_scale2.npy"),
    ("aniso-diff", "anisotropic_diffusion_gamma0.1_iterations10_kappa50___bicubic_scale2.npy"),
    ("wavelet-db4", "wavelet_dwt_level3_waveletdb4___bicubic_scale2.npy"),
    ("wavelet-bior", "wavelet_biorthogonal_level3_waveletbior3.5___bicubic_scale2.npy"),
]


def hillshade(dem: np.ndarray, az_deg: float = 315, alt_deg: float = 45) -> np.ndarray:
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = np.cos(alt) * np.sin(slope) * np.cos(az - aspect) + np.sin(alt) * np.cos(slope)
    return (shaded - shaded.min()) / (shaded.max() - shaded.min() + 1e-12)


def main() -> None:
    dem = np.load(DEM_PATH)
    hs = hillshade(dem)

    # Grid: 5 cols x 5 rows = 25 panels (1 hillshade + 24 residuals)
    cols, rows = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 4.5))
    axes = axes.flatten()

    axes[0].imshow(hs, cmap="gray", origin="upper", aspect="equal")
    axes[0].set_title("hillshade", fontsize=8)
    # Row markers every 200 rows
    for r in range(0, dem.shape[0], 200):
        axes[0].axhline(r, color="cyan", lw=0.4, alpha=0.6)
        axes[0].text(2, r + 8, str(r), color="cyan", fontsize=5)
    axes[0].axis("off")

    missing = []
    for i, (label, name) in enumerate(SAMPLES, start=1):
        ax = axes[i]
        path = RESULTS_DIR / name
        if not path.exists():
            missing.append(name)
            ax.set_title(f"MISSING\n{label}", fontsize=7, color="red")
            ax.axis("off")
            continue
        # Memory-map and downsample for display (residuals are 3000x750)
        arr = np.load(path, mmap_mode="r")
        thumb = np.asarray(arr[::4, ::1])  # 750x750 display
        vmax = float(np.percentile(np.abs(thumb), 99)) or 1.0
        ax.imshow(thumb, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    for j in range(len(SAMPLES) + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Fairfield County DEM (1500×375) + 24 decomposition residuals @ bicubic_scale2",
        fontsize=10,
    )
    plt.tight_layout()
    out = OUT_DIR / "02_dem_overview_full.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")
    if missing:
        print("Missing:", missing)


if __name__ == "__main__":
    main()
