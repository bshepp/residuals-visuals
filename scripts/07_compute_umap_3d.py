"""
Refit UMAP into 3D for the interactive atlas explorer.

Usage: python scripts/07_compute_umap_3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.signatures import SIGNATURE_DIM  # noqa: E402

UMAP_3D_COORDS_PATH = config.CACHE_DIR / "umap_coords_3d.parquet"


def main() -> None:
    sigs_path = config.SIGNATURES_PATH if config.SIGNATURES_PATH.exists() else config.CHECKPOINT_PATH
    if not sigs_path.exists():
        print("ERROR: signatures not found. Run 04_compute_signatures.py first.")
        sys.exit(1)

    df = pd.read_parquet(sigs_path)
    sig_cols = [f"s{i:02d}" for i in range(SIGNATURE_DIM)]
    X = df[sig_cols].to_numpy(dtype=np.float32)
    print(f"Signatures: {X.shape}")

    print("Scaling...")
    X_scaled = RobustScaler().fit_transform(X)

    import umap

    print("Fitting 3D UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=3,
        metric="euclidean",
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(X_scaled.astype(np.float32))
    print(f"  coords: {coords.shape}")

    out = pd.DataFrame({
        "filename": df["filename"],
        "row_idx": df["row_idx"],
        "u": coords[:, 0],
        "v": coords[:, 1],
        "w": coords[:, 2],
    })
    out.to_parquet(UMAP_3D_COORDS_PATH, engine="pyarrow", index=False)
    print(f"Wrote {UMAP_3D_COORDS_PATH}")


if __name__ == "__main__":
    main()
