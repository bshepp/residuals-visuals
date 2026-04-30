"""
Phase B3: fit UMAP on the cached signatures.

Usage: python scripts/05_compute_umap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.atlas import fit_umap  # noqa: E402
from src.signatures import SIGNATURE_DIM  # noqa: E402

UMAP_COORDS_PATH = config.CACHE_DIR / "umap_coords.parquet"


def main() -> None:
    if not config.SIGNATURES_PATH.exists():
        # Fall back to the checkpoint if the final hasn't been promoted
        if config.CHECKPOINT_PATH.exists():
            sigs_path = config.CHECKPOINT_PATH
            print(f"Using checkpoint: {sigs_path}")
        else:
            print(f"ERROR: signatures not found. Run 04_compute_signatures.py first.")
            sys.exit(1)
    else:
        sigs_path = config.SIGNATURES_PATH

    df = pd.read_parquet(sigs_path)
    sig_cols = [f"s{i:02d}" for i in range(SIGNATURE_DIM)]
    X = df[sig_cols].to_numpy(dtype=np.float32)
    print(f"Signatures: {X.shape}")

    # Robust scaling so heavy-tailed stats (kurt, p99) don't dominate
    print("Scaling...")
    X_scaled = RobustScaler().fit_transform(X)

    print("Fitting UMAP...")
    coords = fit_umap(X_scaled.astype(np.float32))
    print(f"  coords: {coords.shape}")

    out = pd.DataFrame({
        "filename": df["filename"],
        "row_idx": df["row_idx"],
        "u": coords[:, 0],
        "v": coords[:, 1],
    })
    out.to_parquet(UMAP_COORDS_PATH, engine="pyarrow", index=False)
    print(f"Wrote {UMAP_COORDS_PATH}")


if __name__ == "__main__":
    main()
