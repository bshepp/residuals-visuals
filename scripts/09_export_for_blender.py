"""
Export 3D UMAP coords + per-point category color to a CSV that
the Blender script can ingest without needing pandas/numpy.

Usage: python scripts/09_export_for_blender.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.atlas import CATEGORY_COLORS, FAMILY_CATEGORY  # noqa: E402
from src.catalog import load_catalog  # noqa: E402

UMAP_3D_COORDS_PATH = config.CACHE_DIR / "umap_coords_3d.parquet"
BLENDER_DATA_PATH = config.CACHE_DIR / "blender_points.csv"


def main() -> None:
    if not UMAP_3D_COORDS_PATH.exists():
        print(f"ERROR: {UMAP_3D_COORDS_PATH} not found. Run 07_compute_umap_3d.py first.")
        sys.exit(1)

    catalog = load_catalog(config.CATALOG_PATH).reset_index(drop=True)
    coords_df = pd.read_parquet(UMAP_3D_COORDS_PATH)
    merged = catalog.merge(coords_df[["filename", "u", "v", "w"]], on="filename", how="inner")

    coords = merged[["u", "v", "w"]].to_numpy()
    # Center on origin, scale to fit roughly within ±10 units
    center = coords.mean(axis=0)
    coords = coords - center
    span = np.percentile(np.linalg.norm(coords, axis=1), 99)
    coords = coords / span * 10.0

    rgbs = []
    for fam in merged["decomp_family"]:
        cat = FAMILY_CATEGORY.get(fam, "classical")
        rgbs.append(to_rgb(CATEGORY_COLORS[cat]))
    rgbs = np.array(rgbs, dtype=np.float32)

    out = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "r": rgbs[:, 0],
        "g": rgbs[:, 1],
        "b": rgbs[:, 2],
        "category": merged["decomp_family"].map(FAMILY_CATEGORY),
        "family": merged["decomp_family"],
    })
    out.to_csv(BLENDER_DATA_PATH, index=False)
    print(f"Wrote {BLENDER_DATA_PATH}  ({len(out):,} points)")
    print(f"  bounds: x∈[{out['x'].min():.2f},{out['x'].max():.2f}] "
          f"y∈[{out['y'].min():.2f},{out['y'].max():.2f}] "
          f"z∈[{out['z'].min():.2f},{out['z'].max():.2f}]")
    print()
    print("Per-category centroids (useful for camera path planning):")
    for cat, grp in out.groupby("category"):
        cx, cy, cz = grp[["x", "y", "z"]].mean()
        print(f"  {cat:18s}  n={len(grp):5d}  centroid=({cx:+.2f}, {cy:+.2f}, {cz:+.2f})")


if __name__ == "__main__":
    main()
