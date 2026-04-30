"""
Project-wide paths and the locked-in crop spec.

The crop spec is expressed in scale=2 coordinates and scales linearly with
upsampling factor. The thumbnail is always 256x256 regardless of source scale.
"""

from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path("D:/DIVERGE_exhaustive/results")
SOURCE_DEM = Path("F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy")
DOCS_DIR = Path("D:/DIVERGE_exhaustive/documentation")

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
ZOO_DIR = OUTPUT_DIR / "zoo"
SWEEPS_DIR = OUTPUT_DIR / "sweeps"
ATLAS_DIR = OUTPUT_DIR / "atlas"

CATALOG_PATH = CACHE_DIR / "catalog.parquet"
THUMBNAILS_ZARR = CACHE_DIR / "thumbnails.zarr"
SIGNATURES_PATH = CACHE_DIR / "signatures.parquet"
CHECKPOINT_PATH = CACHE_DIR / "signatures_checkpoint.parquet"

# --- crop spec (locked in 2026-04-25 after exploration) ---
# Coordinates are in the scale=2 array (3000 rows x 750 cols).
CROP_CY_S2 = 1600
CROP_CX_S2 = 400
CROP_SIZE_S2 = 640

THUMB_SIZE = 256  # all thumbnails normalize to this

# --- visualization conventions ---
# Inherited from RESIDUALS for visual continuity:
#   residuals -> RdBu_r, symmetric, 99th-pctile clip
#   divergence -> hot
#   meta-divergence -> viridis
RESIDUAL_CMAP = "RdBu_r"
PERCENTILE_CLIP = 99

# --- throttle settings for Phase A ---
THROTTLE_SLEEP_MS = 50  # sleep between files
CHECKPOINT_EVERY = 200  # save signature progress every N files
