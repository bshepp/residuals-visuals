"""
Build a HuggingFace Datasets-loadable package at package/hf/.

Schema (single 'train' split):
  filename       : string         original .npy filename
  decomp_family  : string         e.g. 'gaussian', 'wavelet_biorthogonal'
  decomp_params  : string (json)  e.g. '{"sigma": 10}'
  upsamp_method  : string         e.g. 'bicubic', 'sinc_hamming'
  upsamp_params  : string (json)
  scale          : int32
  category       : string         decomp meta-category (classical / wavelet / ...)
  signature      : list<float32>  40-dim signature vector
  umap_2d        : list<float32>  [u, v]
  umap_3d        : list<float32>  [u, v, w]
  image          : binary (PNG)   256x256 RdBu_r-rendered residual

Output: package/hf/data/train-00000-of-NNNNN.parquet (sharded ~500 MB each)
        package/hf/README.md  (dataset card)
        package/hf/.gitattributes  (LFS routing for parquet)

Usage: python scripts/11_package_hf.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.atlas import FAMILY_CATEGORY  # noqa: E402
from src.catalog import load_catalog  # noqa: E402
from src.signatures import SIGNATURE_DIM  # noqa: E402

PKG_DIR = config.PROJECT_ROOT / "package" / "hf"
DATA_DIR = PKG_DIR / "data"
SHARD_TARGET_BYTES = 500 * 1024 * 1024  # ~500 MB per parquet shard


def render_thumbnail_png(arr: np.ndarray, vmax: float, cmap) -> bytes:
    """Render a 2D float32 residual to a PNG (RGB)."""
    norm = np.clip((arr + vmax) / (2 * vmax), 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_table_from_rows(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=pa.schema([
        ("filename", pa.string()),
        ("decomp_family", pa.string()),
        ("decomp_params", pa.string()),
        ("upsamp_method", pa.string()),
        ("upsamp_params", pa.string()),
        ("scale", pa.int32()),
        ("category", pa.string()),
        ("signature", pa.list_(pa.float32(), SIGNATURE_DIM)),
        ("umap_2d", pa.list_(pa.float32(), 2)),
        ("umap_3d", pa.list_(pa.float32(), 3)),
        ("image", pa.binary()),
    ]))


def write_dataset_card(path: Path) -> None:
    card = """---
license: apache-2.0
language:
- en
pretty_name: RESIDUALS — LiDAR DEM residual fingerprints
tags:
- lidar
- dem
- digital-elevation-model
- signal-processing
- wavelet
- morphological-filtering
- remote-sensing
- archaeology
- algorithm-fingerprinting
- umap
size_categories:
- 10K<n<100K
task_categories:
- image-classification
- feature-extraction
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# RESIDUALS — LiDAR DEM residual fingerprints

39,716 residual images extracted by applying **593 distinct decomposition configurations × 25 upsampling methods** to a single Fairfield County, Ohio LiDAR-derived Digital Elevation Model (1500×375 at 3.33 ft/px). Each row pairs a 256×256 PNG of the residual (rendered with the standard `RdBu_r` colormap, 99th-percentile symmetric clipping) with the algorithm and parameters that produced it, plus a 40-dim signature vector and pre-computed 2D/3D UMAP coordinates.

The companion source-code project is [RESIDUALS](https://github.com/bshepp/RESIDUALS); the rendered atlas / sweep videos / Blender flythrough that visualize this dataset are produced by the [residuals-visuals](https://github.com/bshepp/residuals-visuals) project.

## Why this dataset is interesting

- **Algorithm classification benchmark**: predict which decomposition algorithm produced a residual (24 family classes, or 593 fine-grained classes). A non-trivial visual task — many algorithm families produce visually similar outputs at certain parameter settings.
- **Algorithm fingerprinting / signal-processing forensics**: the included 40-dim signatures (radial-FFT power spectrum + statistical moments) cluster algorithm families clearly in UMAP space — useful as a reference for inverse-problem and provenance work.
- **Same scene, all algorithms**: every residual is a different mathematical lens on the *same* underlying terrain, making this an unusually clean substrate for studying what each algorithm preserves vs. destroys.
- **Reproducible**: source DEM hashes + RESIDUALS source code = full regeneration of the original 4.28 TB float64 outputs.

## Schema

| Column | Type | Description |
|---|---|---|
| `filename` | string | Original `.npy` filename in the source RESIDUALS exhaustive run |
| `decomp_family` | string | e.g. `gaussian`, `wavelet_biorthogonal`, `morphological_rect`, `anisotropic_diffusion` |
| `decomp_params` | string (JSON) | e.g. `{"sigma": 10}` or `{"wavelet": "bior3.5", "level": 3}` |
| `upsamp_method` | string | e.g. `bicubic`, `lanczos`, `fft_zeropad`, `sinc_hamming` |
| `upsamp_params` | string (JSON) | e.g. `{"scale": 2, "kernel_size": 8}` |
| `scale` | int32 | Upsampling factor: 2, 3, 4, 8, or 16 |
| `category` | string | Meta-category: `classical`, `edge_preserving`, `wavelet`, `morphological`, `trend_removal`, `multiscale` |
| `signature` | list<float32>[40] | 32-bin radial-FFT log-power spectrum + 8 statistical moments (mean, std, skew, kurtosis, p50\\|abs\\|, p99\\|abs\\|, edge density, lag-1 autocorrelation) |
| `umap_2d` | list<float32>[2] | 2D UMAP embedding of the signature |
| `umap_3d` | list<float32>[3] | 3D UMAP embedding of the signature |
| `image` | binary (PNG) | 256×256 PNG of the residual, RdBu_r colormap, 99th-pctile symmetric clip |

## Source data

- **DEM**: 1500×375 array, 3.33 ft/px resolution, derived from LiDAR tiles `BS000600.las` through `BS000603.las` covering ~1 mi² in Fairfield County, Ohio. Z range: 808.5–1034.4 ft. CRS: Ohio State Plane South (EPSG:3735).
- **Residual crop**: 640×640 region at scale=2 coordinates `(row=1600, col=400)`, normalized to 256×256. Region was selected for high feature density (sweeping diagonal stream channel, branching drainage, V-shaped linear feature, distinct ridge/embankment patterns).
- 15 of the original 39,731 files are excluded due to division-by-zero edge cases in signature computation (all `bicubic_16x_scale=16` outputs).

## Splits

Single `train` split with 39,716 rows. The dataset is small enough that downstream users typically define their own splits (e.g. by `decomp_family` for held-out generalization, or random for IID evaluation).

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("bshepp/diverge-residuals")
print(ds)
print(ds["train"][0]["decomp_family"], ds["train"][0]["decomp_params"])
ds["train"][0]["image"]  # PIL.Image.Image, 256x256
```

## Citation

If you use this dataset, please cite:

- This dataset (BibTeX assigned on publish)
- The [Zenodo deposit](https://zenodo.org/record/...) (DOI on publish) — contains additional artifacts
- The [RESIDUALS source code](https://github.com/bshepp/RESIDUALS)

## License

Apache 2.0 — same as the source RESIDUALS project.

## Limitations and ethics

- **Single scene**: all 39,716 samples come from the same DEM. Models trained here may not generalize to other terrains. Treat as a fingerprinting benchmark, not a generic remote-sensing pretraining set.
- **Class imbalance**: family counts range from 201 (`polynomial`) to 5,346 (`wavelet`). Use stratified splits or balanced sampling if classification accuracy matters.
- **Known artifact**: the leftmost ~30 columns of the source DEM exhibit upsampling-boundary effects. The crop window dodges this entirely.
- **No PII / sensitive content**: terrain residuals only. The source LiDAR is publicly-available Ohio state data.
"""
    path.write_text(card, encoding="utf-8")
    print(f"  [ok]  README.md (dataset card)")


def main() -> None:
    if not config.SIGNATURES_PATH.exists() or not config.THUMBNAILS_ZARR.exists():
        print("ERROR: signatures or thumbnails cache not found. Run earlier scripts first.")
        sys.exit(1)

    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)
    DATA_DIR.mkdir(parents=True)

    # Use load_catalog (not pd.read_parquet) so decomp_params / upsamp_params are dicts, not str
    catalog = load_catalog(config.CATALOG_PATH)
    sigs = pd.read_parquet(config.SIGNATURES_PATH)
    coords_2d = pd.read_parquet(config.CACHE_DIR / "umap_coords.parquet")
    coords_3d = pd.read_parquet(config.CACHE_DIR / "umap_coords_3d.parquet")
    thumbs = zarr.open(str(config.THUMBNAILS_ZARR), mode="r")

    # Merge everything — `sigs` has filename + row_idx + signature; only it has the
    # subset of 39,716 successfully-processed files. Use it as the master.
    sig_cols = [f"s{i:02d}" for i in range(SIGNATURE_DIM)]
    df = sigs[["filename", "row_idx", *sig_cols]].copy()
    df = df.merge(catalog[["filename", "decomp_family", "decomp_params", "upsamp_method", "upsamp_params", "scale"]], on="filename", how="left")
    df = df.merge(coords_2d[["filename", "u", "v"]].rename(columns={"u": "u2", "v": "v2"}), on="filename", how="left")
    df = df.merge(coords_3d[["filename", "u", "v", "w"]], on="filename", how="left")

    df["category"] = df["decomp_family"].map(FAMILY_CATEGORY).fillna("classical")
    df["decomp_params"] = df["decomp_params"].apply(lambda d: json.dumps(d) if isinstance(d, dict) else "{}")
    df["upsamp_params"] = df["upsamp_params"].apply(lambda d: json.dumps(d) if isinstance(d, dict) else "{}")
    df = df.sort_values("row_idx").reset_index(drop=True)
    print(f"Master frame: {len(df):,} rows")

    cmap = plt.get_cmap(config.RESIDUAL_CMAP)

    # Stream rows -> shards
    rows: list[dict] = []
    bytes_in_shard = 0
    shard_idx = 0
    written_shards = []

    pbar = tqdm(total=len(df), unit="row")
    for _, row in df.iterrows():
        thumb = np.asarray(thumbs[int(row["row_idx"])])
        vmax = float(np.percentile(np.abs(thumb), config.PERCENTILE_CLIP)) or 1.0
        png_bytes = render_thumbnail_png(thumb, vmax, cmap)

        sig_vec = np.array([row[c] for c in sig_cols], dtype=np.float32).tolist()
        rows.append({
            "filename": row["filename"],
            "decomp_family": row["decomp_family"],
            "decomp_params": row["decomp_params"],
            "upsamp_method": row["upsamp_method"],
            "upsamp_params": row["upsamp_params"],
            "scale": int(row["scale"]),
            "category": row["category"],
            "signature": sig_vec,
            "umap_2d": [float(row["u2"]), float(row["v2"])],
            "umap_3d": [float(row["u"]), float(row["v"]), float(row["w"])],
            "image": png_bytes,
        })
        bytes_in_shard += len(png_bytes) + 200  # rough overhead

        if bytes_in_shard >= SHARD_TARGET_BYTES:
            tbl = build_table_from_rows(rows)
            shard_path = DATA_DIR / f"train-{shard_idx:05d}.parquet"
            pq.write_table(tbl, shard_path, compression="zstd", compression_level=10)
            written_shards.append(shard_path)
            print(f"\n  wrote {shard_path.name} ({shard_path.stat().st_size / 1024 / 1024:.1f} MB)")
            rows = []
            bytes_in_shard = 0
            shard_idx += 1
        pbar.update(1)
    pbar.close()

    if rows:
        tbl = build_table_from_rows(rows)
        shard_path = DATA_DIR / f"train-{shard_idx:05d}.parquet"
        pq.write_table(tbl, shard_path, compression="zstd", compression_level=10)
        written_shards.append(shard_path)
        print(f"  wrote {shard_path.name} ({shard_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Re-name shards to the standard "train-NNNNN-of-MMMMM" pattern HF prefers
    n_total = len(written_shards)
    for i, p in enumerate(written_shards):
        new_name = DATA_DIR / f"train-{i:05d}-of-{n_total:05d}.parquet"
        if new_name != p:
            p.rename(new_name)

    write_dataset_card(PKG_DIR / "README.md")

    # .gitattributes — route parquet through Git LFS
    (PKG_DIR / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.png filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    total_bytes = sum(p.stat().st_size for p in PKG_DIR.rglob("*") if p.is_file())
    print()
    print(f"HF package built: {PKG_DIR}")
    print(f"Shards:           {n_total}")
    print(f"Total size:       {total_bytes / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
