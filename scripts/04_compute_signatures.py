"""
Phase A — heaviest step. One-time scan of all 39,731 residual .npy files.

For each file:
  1. mmap-load, slice the crop window (only the crop region is read from disk)
  2. downsample to 256x256 -> store in cache/thumbnails.zarr
  3. compute 40-dim signature -> append to cache/signatures.parquet

Throttling (keeps machine responsive):
  - Single thread; no multiprocessing
  - Process priority BELOW_NORMAL on Windows (psutil)
  - Sleep config.THROTTLE_SLEEP_MS between files
  - Checkpoint every config.CHECKPOINT_EVERY files
  - Restartable: skips files already in the checkpoint

Estimated time at 50ms throttle: ~3-4 hours for 39,731 files.
Estimated cache size: ~10 GB (thumbnails) + ~10 MB (signatures).

Usage:
  python scripts/04_compute_signatures.py
  python scripts/04_compute_signatures.py --no-throttle   # full speed (~1 hr)
  python scripts/04_compute_signatures.py --limit 500     # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.catalog import load_catalog  # noqa: E402
from src.load import load_thumb  # noqa: E402
from src.signatures import SIGNATURE_DIM, signature  # noqa: E402


def lower_priority() -> None:
    """Drop to BELOW_NORMAL priority + low I/O priority on Windows."""
    try:
        import psutil

        p = psutil.Process(os.getpid())
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)  # POSIX nice value
        # I/O priority (Windows-specific)
        if hasattr(psutil, "IOPRIO_LOW"):
            try:
                p.ionice(psutil.IOPRIO_LOW)
            except (AttributeError, OSError):
                pass
        print("  process priority lowered")
    except Exception as e:
        print(f"  could not lower priority: {e}")


def open_thumb_store(n_files: int, thumb_size: int) -> zarr.Array:
    """Open or create the thumbnails zarr array (n_files, thumb_size, thumb_size, float32)."""
    config.THUMBNAILS_ZARR.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(
        str(config.THUMBNAILS_ZARR),
        mode="a",
        shape=(n_files, thumb_size, thumb_size),
        chunks=(64, thumb_size, thumb_size),
        dtype="float32",
    )
    return store


def load_existing_checkpoint() -> set[str]:
    """Return the set of filenames already processed."""
    if not config.CHECKPOINT_PATH.exists():
        return set()
    df = pd.read_parquet(config.CHECKPOINT_PATH)
    return set(df["filename"].tolist())


def append_checkpoint(rows: list[dict]) -> None:
    """Append signature rows to the checkpoint parquet (atomic via rewrite)."""
    df_new = pd.DataFrame(rows)
    if config.CHECKPOINT_PATH.exists():
        df_old = pd.read_parquet(config.CHECKPOINT_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    tmp = config.CHECKPOINT_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(config.CHECKPOINT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-throttle", action="store_true", help="Disable inter-file sleep")
    parser.add_argument("--no-priority-drop", action="store_true", help="Keep normal priority")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (smoke test)")
    args = parser.parse_args()

    if not config.CATALOG_PATH.exists():
        print(f"ERROR: catalog not found at {config.CATALOG_PATH}")
        print("Run scripts/01_build_catalog.py first.")
        sys.exit(1)

    catalog = load_catalog(config.CATALOG_PATH)
    if args.limit:
        catalog = catalog.head(args.limit).copy()
    n_files = len(catalog)
    print(f"Catalog: {n_files:,} files")

    if not args.no_priority_drop:
        lower_priority()

    sleep_s = 0.0 if args.no_throttle else (config.THROTTLE_SLEEP_MS / 1000.0)
    print(f"Throttle sleep: {sleep_s * 1000:.0f} ms/file")

    # Open thumbnail store sized to the full catalog
    thumb_store = open_thumb_store(n_files, config.THUMB_SIZE)

    # Build a stable index map from filename -> row position in zarr/dataframe
    catalog = catalog.reset_index(drop=True)
    catalog["row_idx"] = catalog.index

    done = load_existing_checkpoint()
    print(f"Already in checkpoint: {len(done):,}")
    todo = catalog[~catalog["filename"].isin(done)].reset_index(drop=True)
    print(f"To process: {len(todo):,}")

    if len(todo) == 0:
        print("Nothing to do. Run scripts/05_compute_umap.py next.")
        return

    pending: list[dict] = []
    t0 = time.time()
    pbar = tqdm(total=len(todo), unit="file", smoothing=0.05)

    for _, row in todo.iterrows():
        filename = row["filename"]
        path = config.DATA_DIR / filename
        try:
            thumb = load_thumb(path, scale=int(row["scale"]), thumb_size=config.THUMB_SIZE)
            sig = signature(thumb)
            thumb_store[int(row["row_idx"])] = thumb
            sig_dict = {f"s{i:02d}": float(sig[i]) for i in range(SIGNATURE_DIM)}
            pending.append({"filename": filename, "row_idx": int(row["row_idx"]), **sig_dict})
        except Exception as e:
            pbar.write(f"  ERR {filename}: {e}")

        pbar.update(1)
        if sleep_s:
            time.sleep(sleep_s)

        if len(pending) >= config.CHECKPOINT_EVERY:
            append_checkpoint(pending)
            pending = []

    if pending:
        append_checkpoint(pending)

    pbar.close()
    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min")

    # Promote the checkpoint to the final signatures parquet
    df = pd.read_parquet(config.CHECKPOINT_PATH)
    df.to_parquet(config.SIGNATURES_PATH, engine="pyarrow", index=False)
    print(f"Wrote {config.SIGNATURES_PATH} ({len(df):,} rows)")
    print(f"Thumbnails at {config.THUMBNAILS_ZARR}")


if __name__ == "__main__":
    main()
