"""
Phase B2: render parameter-sweep MP4 videos from the cached thumbnails.

Usage: python scripts/03_build_sweeps.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.catalog import load_catalog  # noqa: E402
from src.sweep import SWEEPS, render_sweep_video  # noqa: E402


def main() -> None:
    if not config.THUMBNAILS_ZARR.exists():
        print(f"ERROR: {config.THUMBNAILS_ZARR} not found. Run 04_compute_signatures.py first.")
        sys.exit(1)

    catalog = load_catalog(config.CATALOG_PATH).reset_index(drop=True)
    thumbs = zarr.open(str(config.THUMBNAILS_ZARR), mode="r")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.SWEEPS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep in SWEEPS:
        out = out_dir / f"{sweep.name}.mp4"
        render_sweep_video(catalog, thumbs, sweep, out)

    print(f"\nDone. Videos in {out_dir}")


if __name__ == "__main__":
    main()
