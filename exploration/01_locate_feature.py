"""
One-off exploration: render the source DEM as hillshade and a few sample
residuals so we can pick a feature-rich crop region for the zoo / sweeps / atlas.

Source DEM: F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy
Residuals are at 2x (3000x750), 4x (6000x1500), 8x (12000x3000), 16x (24000x6000).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEM_PATH = Path("F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy")
RESULTS_DIR = Path("D:/DIVERGE_exhaustive/results")
OUT_DIR = Path(__file__).parent

SAMPLES = [
    "gaussian_sigma10___bicubic_scale2.npy",
    "bilateral_d9_sigmacolor75_sigmaspace75___bicubic_scale2.npy",
    "morphological_opening_size20___bicubic_scale2.npy",
    "tophat_size20_modewhite___bicubic_scale2.npy",
    "dog_sigmalow2_sigmahigh10___bicubic_scale2.npy",
    "anisotropic_diffusion_gamma0.1_iterations10_kappa50___bicubic_scale2.npy",
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
    print(f"DEM shape: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}] ft")

    hs = hillshade(dem)

    n_samples = len(SAMPLES)
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(2.5 * (n_samples + 1), 10))

    axes[0].imshow(hs, cmap="gray", origin="upper", aspect="equal")
    axes[0].set_title(f"Hillshade\n{dem.shape}", fontsize=8)
    for r in range(0, dem.shape[0], 200):
        axes[0].axhline(r, color="cyan", lw=0.3, alpha=0.5)
        axes[0].text(5, r + 5, str(r), color="cyan", fontsize=6)
    axes[0].axis("off")

    for ax, name in zip(axes[1:], SAMPLES):
        path = RESULTS_DIR / name
        if not path.exists():
            ax.set_title(f"MISSING\n{name[:30]}", fontsize=7)
            ax.axis("off")
            continue
        arr = np.load(path, mmap_mode="r")
        # Match DEM aspect — these are 2x upsampled (3000x750), downsample for display
        thumb = arr[::4, ::1]
        vmax = np.percentile(np.abs(thumb), 99)
        ax.imshow(thumb, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")
        ax.set_title(name.split("___")[0][:25], fontsize=7)
        for r in range(0, thumb.shape[0], 200):
            ax.axhline(r, color="cyan", lw=0.3, alpha=0.5)
        ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / "01_dem_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")

    # Now render gradient magnitude to find the most "feature-dense" region.
    dy, dx = np.gradient(dem.astype(np.float64))
    grad = np.hypot(dx, dy)
    # Row-wise feature density: sum of strong-gradient pixels per row band.
    band = 100
    n_bands = dem.shape[0] // band
    densities = []
    for b in range(n_bands):
        chunk = grad[b * band : (b + 1) * band]
        densities.append(float(np.percentile(chunk, 95)))
    print("\nFeature density per 100-row band (95th-pct gradient):")
    for b, d in enumerate(densities):
        bar = "#" * int(d * 4)
        print(f"  rows {b * band:4d}-{(b + 1) * band:4d}: {d:6.3f}  {bar}")

    best_band = int(np.argmax(densities))
    best_row = best_band * band + band // 2
    print(f"\nMost feature-dense region centered at row ~{best_row}")


if __name__ == "__main__":
    main()
