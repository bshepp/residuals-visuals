"""
Validate the chosen crop window:
  scale=2 coordinates: center=(row 1600, col 400), size=640x640
  -> in DEM (1.5x scale) coords: center=(800, 200), size=320x320
  Dodges the left-edge upsampling artifact zone (~cols 0-30 at scale=2).
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

DEM_PATH = Path("F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy")
RESULTS_DIR = Path("D:/DIVERGE_exhaustive/results")
OUT_DIR = Path(__file__).parent

# crop spec at scale=2
CY2, CX2 = 1600, 400
SZ2 = 640
HALF = SZ2 // 2

SAMPLES = [
    ("hillshade", None),
    ("gaussian", "gaussian_sigma10___bicubic_scale2.npy"),
    ("bilateral", "bilateral_d9_sigma_color75_sigma_space75___bicubic_scale2.npy"),
    ("morph-disk", "morphological_operationaverage_size20___bicubic_scale2.npy"),
    ("morph-rect", "morphological_rect_height5_operationopening_width20___bicubic_scale2.npy"),
    ("tophat-w", "tophat_modewhite_size20___bicubic_scale2.npy"),
    ("rolling-ball", "rolling_ball_radius50___bicubic_scale2.npy"),
    ("dog-multi", "dog_multiscale_base_sigma1.0_n_scales4_sigma_ratio1.6___bicubic_scale2.npy"),
    ("aniso-diff", "anisotropic_diffusion_gamma0.1_iterations10_kappa50___bicubic_scale2.npy"),
    ("wavelet-bior", "wavelet_biorthogonal_level3_waveletbior3.5___bicubic_scale2.npy"),
]


def hillshade(dem, az_deg=315, alt_deg=45):
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    s = np.cos(alt) * np.sin(slope) * np.cos(az - aspect) + np.sin(alt) * np.cos(slope)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def crop_at_scale(arr: np.ndarray, scale: int) -> np.ndarray:
    """Crop using scale-aware coordinates derived from the scale=2 spec."""
    # scale=2 -> arr is 3000x750. scale=4 -> 6000x1500. etc.
    cy = CY2 * scale // 2
    cx = CX2 * scale // 2
    half = HALF * scale // 2
    return np.asarray(arr[cy - half : cy + half, cx - half : cx + half])


def main() -> None:
    dem = np.load(DEM_PATH)
    hs = hillshade(dem)
    # DEM is half the scale=2 array, so its crop is half the size
    cy_dem, cx_dem, half_dem = CY2 // 2, CX2 // 2, HALF // 2
    hs_crop = hs[cy_dem - half_dem : cy_dem + half_dem, cx_dem - half_dem : cx_dem + half_dem]

    cols = 5
    rows = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = axes.flatten()

    # Top-left: full hillshade with crop rectangle highlighted
    axes[0].imshow(hs, cmap="gray", origin="upper", aspect="equal")
    rect = patches.Rectangle(
        (cx_dem - half_dem, cy_dem - half_dem),
        2 * half_dem,
        2 * half_dem,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"DEM (full) + crop\ncenter=({cy_dem},{cx_dem}), size={2 * half_dem}", fontsize=8)
    axes[0].axis("off")

    # Top-right of first row: cropped hillshade
    axes[1].imshow(hs_crop, cmap="gray", origin="upper", aspect="equal")
    axes[1].set_title("hillshade (cropped)", fontsize=8)
    axes[1].axis("off")

    # Remaining: cropped residuals
    for i, (label, name) in enumerate(SAMPLES[1:], start=2):
        ax = axes[i]
        path = RESULTS_DIR / name
        if not path.exists():
            ax.set_title(f"MISSING {label}", fontsize=7, color="red")
            ax.axis("off")
            continue
        arr = np.load(path, mmap_mode="r")
        c = crop_at_scale(arr, scale=2)
        vmax = float(np.percentile(np.abs(c), 99)) or 1.0
        ax.imshow(c, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    for j in range(len(SAMPLES) + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Crop preview: scale=2 center=({CY2},{CX2}), size={SZ2}x{SZ2}", fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / "03_crop_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
