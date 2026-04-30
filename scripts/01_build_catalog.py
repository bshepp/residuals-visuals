"""
Phase 0: parse all 39,731 .npy filenames into a structured parquet catalog.

Usage: python scripts/01_build_catalog.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# allow `import config` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.catalog import build_catalog, load_catalog, save_catalog  # noqa: E402


def main() -> None:
    print(f"Scanning {config.DATA_DIR}...")
    t0 = time.time()
    df = build_catalog(config.DATA_DIR)
    elapsed = time.time() - t0
    print(f"  found {len(df):,} files in {elapsed:.1f}s")

    save_catalog(df, config.CATALOG_PATH)
    print(f"  wrote {config.CATALOG_PATH}")

    # Sanity check
    df2 = load_catalog(config.CATALOG_PATH)
    print()
    print("Family breakdown:")
    counts = df2["decomp_family"].value_counts().sort_index()
    for family, count in counts.items():
        print(f"  {family:35s} {count:>5,}")
    print()
    print("Upsampling method breakdown:")
    counts = df2["upsamp_method"].value_counts().sort_index()
    for method, count in counts.items():
        print(f"  {method:25s} {count:>5,}")
    print()
    print("Scale breakdown:")
    counts = df2["scale"].value_counts().sort_index()
    for scale, count in counts.items():
        print(f"  scale={scale:<2}  {count:>5,}")
    print()
    print(f"Total: {len(df2):,} entries")
    total_bytes = int(df2["size_bytes"].sum())
    print(f"Total size: {total_bytes / 1024**4:.2f} TiB")


if __name__ == "__main__":
    main()
