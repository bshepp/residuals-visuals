"""
Phase B4: render the UMAP scatter plot and the thumbnail-mosaic atlas.

Usage: python scripts/06_render_atlas.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.atlas import render_atlas, render_scatter  # noqa: E402
from src.catalog import load_catalog  # noqa: E402

UMAP_COORDS_PATH = config.CACHE_DIR / "umap_coords.parquet"


def main() -> None:
    if not UMAP_COORDS_PATH.exists():
        print(f"ERROR: UMAP coords not found. Run 05_compute_umap.py first.")
        sys.exit(1)

    catalog = load_catalog(config.CATALOG_PATH).reset_index(drop=True)
    coords_df = pd.read_parquet(UMAP_COORDS_PATH)

    # Align rows by filename (catalog is the master index for thumbs)
    merged = catalog.merge(coords_df[["filename", "u", "v"]], on="filename", how="inner")
    merged = merged.sort_values("row_idx" if "row_idx" in merged.columns else "filename").reset_index(drop=True)
    coords = merged[["u", "v"]].to_numpy(dtype=np.float32)
    families = merged["decomp_family"]

    thumbs = zarr.open(str(config.THUMBNAILS_ZARR), mode="r")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) lightweight scatter
    render_scatter(coords, families, config.ATLAS_DIR / f"umap_scatter_{timestamp}.png")

    # 2) full thumbnail mosaic atlas
    render_atlas(
        coords,
        families,
        thumbs,
        config.ATLAS_DIR / f"umap_atlas_{timestamp}.png",
        thumb_px=64,
        canvas_cells=220,
        border_px=2,
    )


if __name__ == "__main__":
    main()
