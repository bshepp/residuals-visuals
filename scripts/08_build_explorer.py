"""
Build the interactive 3D Atlas Explorer (self-contained HTML).

Usage: python scripts/08_build_explorer.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.catalog import load_catalog  # noqa: E402
from src.explorer import build_explorer  # noqa: E402

UMAP_3D_COORDS_PATH = config.CACHE_DIR / "umap_coords_3d.parquet"


def main() -> None:
    if not UMAP_3D_COORDS_PATH.exists():
        print(f"ERROR: 3D UMAP coords not found at {UMAP_3D_COORDS_PATH}.")
        print("Run scripts/07_compute_umap_3d.py first.")
        sys.exit(1)

    catalog = load_catalog(config.CATALOG_PATH).reset_index(drop=True)
    coords_df = pd.read_parquet(UMAP_3D_COORDS_PATH)

    merged = catalog.merge(coords_df[["filename", "u", "v", "w"]], on="filename", how="inner")
    print(f"Merged: {len(merged):,} points")

    coords = merged[["u", "v", "w"]].to_numpy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.ATLAS_DIR / f"atlas_explorer_{timestamp}.html"
    build_explorer(coords, merged, out)


if __name__ == "__main__":
    main()
