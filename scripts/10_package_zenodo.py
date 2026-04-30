"""
Build the Zenodo deposit package at package/zenodo/.

Zenodo records have a 50 GB hard limit. Our package targets ~6-8 GB:
  - catalog + signatures + UMAP coords (~25 MB)
  - thumbnails.zarr (zipped, ~5-7 GB)
  - prior-art docs + method parameters + hashes (~5 MB)
  - rendered visualizations (zoo grid, atlas, sweeps) (~50 MB)
  - source code snapshots (text, <1 MB)
  - README + LICENSE + Zenodo metadata.json

After running this, the package/zenodo/ directory can be uploaded to Zenodo
either through the web UI (drag-and-drop) or via REST API (see scripts/14_deploy_zenodo.py).

Usage: python scripts/10_package_zenodo.py
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

PKG_DIR = config.PROJECT_ROOT / "package" / "zenodo"
RESIDUALS_DIR = Path("F:/science-projects/RESIDUALS")
EXHAUSTIVE_DOCS = Path("D:/DIVERGE_exhaustive/documentation")


def copy_or_skip(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        print(f"  [skip] {label}: {src} missing")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    size_mb = (sum(p.stat().st_size for p in (dst.rglob("*") if dst.is_dir() else [dst]) if p.is_file()) / 1024 / 1024)
    print(f"  [ok]  {label}: {dst.name}  ({size_mb:.1f} MB)")


def zip_directory(src_dir: Path, zip_path: Path, label: str, compresslevel: int = 6) -> None:
    if not src_dir.exists():
        print(f"  [skip] {label}: {src_dir} missing")
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir.parent))
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  [ok]  {label}: {zip_path.name}  ({size_mb:.1f} MB)")


def write_metadata_json(path: Path) -> None:
    meta = {
        "metadata": {
            "title": "RESIDUALS: 39,731-residual exhaustive parameter sweep of decomposition × upsampling methods on a Fairfield County, Ohio LiDAR DEM",
            "upload_type": "dataset",
            "description": (
                "<p>This deposit accompanies the RESIDUALS prior-art exhaustive run. "
                "It contains a structured catalog of 39,731 residual outputs produced by "
                "applying 593 distinct decomposition configurations (gaussian, bilateral, "
                "wavelet, morphological, top-hat, polynomial, DoG, LoG, anisotropic diffusion, "
                "rolling-ball, guided, and others) crossed with 25 upsampling methods "
                "(bicubic, lanczos, B-spline, FFT zero-pad, sinc-windowed, edge-directed, etc.) "
                "applied to a 1500×375 Digital Elevation Model derived from LiDAR over "
                "Fairfield County, Ohio.</p>"
                "<p><strong>Contents:</strong></p>"
                "<ul>"
                "<li><code>data/catalog.parquet</code> — 39,731 entries (decomposition family, "
                "parameters, upsampling method, scale, file hash).</li>"
                "<li><code>data/signatures.parquet</code> — 40-dim signature vector per residual "
                "(32-bin radial-FFT log-power spectrum + 8 statistical moments).</li>"
                "<li><code>data/umap_coords_2d.parquet</code>, <code>umap_coords_3d.parquet</code> "
                "— UMAP embeddings of the signatures (n=39,716; 15 div-by-zero edge cases dropped).</li>"
                "<li><code>data/thumbnails.zarr.zip</code> — 39,716 cropped 256×256 float32 thumbnails "
                "centered on a feature-rich region of the DEM.</li>"
                "<li><code>documentation/prior_art.md</code>, <code>method_parameters.json</code>, "
                "<code>hashes.csv</code> — full method parameter spaces and SHA-256 hashes of "
                "all 39,731 outputs (proof-of-existence for the full 4.28 TB dataset).</li>"
                "<li><code>visualizations/</code> — algorithm zoo grid, UMAP scatter, parameter "
                "sweep videos, atlas mosaic preview.</li>"
                "</ul>"
                "<p>The full 4.28 TB raw output set is not deposited here (Zenodo size limits) "
                "but is fully reproducible from the source DEM + the RESIDUALS code "
                "(<a href='https://github.com/bshepp/RESIDUALS'>github.com/bshepp/RESIDUALS</a>). "
                "All file hashes are included for verification.</p>"
                "<p>This deposit serves a dual purpose: (1) prior-art documentation to prevent "
                "exclusive claims on these specific signal-processing combinations applied to "
                "DEMs, and (2) a reusable benchmark for algorithm-fingerprinting and "
                "feature-extraction research.</p>"
            ),
            "creators": [
                {"name": "Sheppard, Brian", "affiliation": "Independent researcher", "orcid": ""}
            ],
            "keywords": [
                "lidar",
                "digital elevation model",
                "DEM",
                "signal decomposition",
                "wavelet",
                "morphological filtering",
                "anisotropic diffusion",
                "super-resolution",
                "upsampling",
                "feature detection",
                "archaeology",
                "remote sensing",
                "UMAP",
                "prior art",
                "Fairfield County Ohio",
            ],
            "license": "Apache-2.0",
            "access_right": "open",
            "language": "eng",
            "communities": [],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/bshepp/RESIDUALS",
                    "relation": "isSupplementTo",
                    "scheme": "url",
                    "resource_type": "software",
                }
            ],
            "version": "1.0.0",
            "notes": (
                "Source DEM resolution: 3.33 ft/px. DEM bounds (Ohio State Plane South, ft): "
                "x_min=2000000, y_min=600000.02. Z range: 808.5–1034.4 ft. "
                "Source LiDAR tiles: BS000600.las, BS000601.las, BS000602.las, BS000603.las."
            ),
        }
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [ok]  metadata.json")


def write_readme(path: Path) -> None:
    readme = """# RESIDUALS: Exhaustive Decomposition × Upsampling Sweep on a LiDAR DEM

Companion deposit to the [RESIDUALS](https://github.com/bshepp/RESIDUALS) prior-art exhaustive run.

## What's in this deposit

| Path | Description |
|------|-------------|
| `data/catalog.parquet` | 39,731 entries with parsed decomposition family + params, upsampling method + params, scale, file hash |
| `data/signatures.parquet` | 40-dim signature vector per residual (32-bin radial-FFT log-power spectrum + 8 statistical moments) |
| `data/umap_coords_2d.parquet` | 2D UMAP embedding of signatures |
| `data/umap_coords_3d.parquet` | 3D UMAP embedding of signatures |
| `data/thumbnails.zarr.zip` | 39,716 zarr-stored 256×256 float32 thumbnails of cropped residuals (raw values) |
| `documentation/prior_art.md` | Method parameter spaces, total combinations, hashes (markdown table) |
| `documentation/method_parameters.json` | Machine-readable method parameter specification |
| `documentation/hashes.csv` | SHA-256 hash of every one of the 39,731 outputs |
| `visualizations/algorithm_zoo.png` | 24-panel poster: same terrain through 24 different decomposition algorithms |
| `visualizations/umap_scatter.png` | 2D UMAP scatter colored by algorithm category |
| `visualizations/umap_atlas_preview.png` | Thumbnail mosaic at UMAP positions (downscaled preview) |
| `visualizations/sweeps/*.mp4` | Parameter sweep flythrough videos (gaussian σ, wavelet level, etc.) |

## Source data not in this deposit

The full 4.28 TB raw output set is **not** included due to Zenodo size limits.
It is fully reproducible from:

  1. The source DEM: `fairfield_sample_1.5ft.npy` (1500×375, derived from
     LiDAR tiles BS000600–BS000603.las)
  2. The RESIDUALS code: https://github.com/bshepp/RESIDUALS
     (run `python run_exhaustive.py --dem fairfield_sample_1.5ft.npy`)

`documentation/hashes.csv` contains SHA-256 of all 39,731 outputs as proof-of-existence
and verification.

## Loading the data

```python
import pandas as pd
import zarr

catalog = pd.read_parquet('data/catalog.parquet')
sigs    = pd.read_parquet('data/signatures.parquet')
coords  = pd.read_parquet('data/umap_coords_3d.parquet')

# Thumbnails: unzip first, then:
thumbs = zarr.open('data/thumbnails.zarr', mode='r')
img = thumbs[42]   # 256x256 float32 residual
```

## Citation

If you use this deposit, please cite both:

1. This Zenodo record (DOI assigned on publication)
2. The RESIDUALS source code repository

## License

Apache 2.0 — same as the RESIDUALS source project.

## Contact

bshepp@gmail.com
"""
    path.write_text(readme, encoding="utf-8")
    print(f"  [ok]  README.md")


def write_hashes_csv(out_path: Path) -> None:
    """Extract SHA-256 hashes from the exhaustive_results JSON if present."""
    candidates = sorted(EXHAUSTIVE_DOCS.glob("exhaustive_results_*.json"), reverse=True)
    if not candidates:
        print(f"  [skip] hashes.csv: no exhaustive_results_*.json found")
        return
    src = candidates[0]
    print(f"  [..]  reading hashes from {src.name}")
    with open(src) as f:
        data = json.load(f)

    rows = []

    def _walk(obj, depth=0):
        if depth > 6:
            return
        if isinstance(obj, dict):
            if "hash" in obj or "sha256" in obj:
                h = obj.get("sha256") or obj.get("hash")
                fn = obj.get("filename") or obj.get("file") or obj.get("name") or ""
                rows.append((fn, h))
            for v in obj.values():
                _walk(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v, depth + 1)

    _walk(data)

    if not rows:
        print(f"  [skip] hashes.csv: no hash entries found in {src.name}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("filename,sha256\n")
        for fn, h in rows:
            f.write(f"{fn},{h}\n")
    print(f"  [ok]  hashes.csv  ({len(rows):,} entries)")


def main() -> None:
    if PKG_DIR.exists():
        print(f"Removing existing {PKG_DIR}")
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)

    print(f"Building Zenodo package at {PKG_DIR}")
    print()
    print("Top-level metadata + README:")
    write_readme(PKG_DIR / "README.md")
    write_metadata_json(PKG_DIR / "metadata.json")
    copy_or_skip(RESIDUALS_DIR / "LICENSE", PKG_DIR / "LICENSE", "LICENSE")

    print()
    print("Tabular data:")
    copy_or_skip(config.CATALOG_PATH, PKG_DIR / "data" / "catalog.parquet", "catalog")
    copy_or_skip(config.SIGNATURES_PATH, PKG_DIR / "data" / "signatures.parquet", "signatures")
    copy_or_skip(config.CACHE_DIR / "umap_coords.parquet", PKG_DIR / "data" / "umap_coords_2d.parquet", "umap_2d")
    copy_or_skip(config.CACHE_DIR / "umap_coords_3d.parquet", PKG_DIR / "data" / "umap_coords_3d.parquet", "umap_3d")

    print()
    print("Thumbnails (large — this step takes minutes):")
    zip_directory(config.THUMBNAILS_ZARR, PKG_DIR / "data" / "thumbnails.zarr.zip", "thumbnails.zarr.zip", compresslevel=6)

    print()
    print("Documentation:")
    candidates = sorted(EXHAUSTIVE_DOCS.glob("PRIOR_ART_*.md"), reverse=True)
    if candidates:
        copy_or_skip(candidates[0], PKG_DIR / "documentation" / "prior_art.md", "prior_art.md")
    candidates = sorted(EXHAUSTIVE_DOCS.glob("method_parameters_*.json"), reverse=True)
    if candidates:
        copy_or_skip(candidates[0], PKG_DIR / "documentation" / "method_parameters.json", "method_parameters.json")
    write_hashes_csv(PKG_DIR / "documentation" / "hashes.csv")

    print()
    print("Visualizations:")
    zoo_pngs = sorted((config.OUTPUT_DIR / "zoo").glob("algorithm_zoo_*.png"), reverse=True)
    if zoo_pngs:
        copy_or_skip(zoo_pngs[0], PKG_DIR / "visualizations" / "algorithm_zoo.png", "zoo grid")
    scatter_pngs = sorted((config.OUTPUT_DIR / "atlas").glob("umap_scatter_*.png"), reverse=True)
    if scatter_pngs:
        copy_or_skip(scatter_pngs[0], PKG_DIR / "visualizations" / "umap_scatter.png", "umap scatter")
    atlas_4x = sorted((config.OUTPUT_DIR / "atlas").glob("umap_atlas_*_4x.png"), reverse=True)
    if atlas_4x:
        copy_or_skip(atlas_4x[0], PKG_DIR / "visualizations" / "umap_atlas_preview.png", "atlas preview")
    flythrough = sorted((config.OUTPUT_DIR / "atlas").glob("atlas_flythrough_*.mp4"), reverse=True)
    if flythrough:
        copy_or_skip(flythrough[0], PKG_DIR / "visualizations" / "atlas_flythrough.mp4", "flythrough")
    sweep_dirs = sorted((config.OUTPUT_DIR / "sweeps").iterdir(), reverse=True) if (config.OUTPUT_DIR / "sweeps").exists() else []
    if sweep_dirs:
        for mp4 in sweep_dirs[0].glob("*.mp4"):
            copy_or_skip(mp4, PKG_DIR / "visualizations" / "sweeps" / mp4.name, f"sweep:{mp4.stem}")

    print()
    print("Source code snapshot:")
    src_root = config.PROJECT_ROOT / "src"
    scripts_root = config.PROJECT_ROOT / "scripts"
    blender_root = config.PROJECT_ROOT / "blender"
    for src_dir, label in [(src_root, "residuals-visuals/src"), (scripts_root, "residuals-visuals/scripts"), (blender_root, "residuals-visuals/blender")]:
        for f in src_dir.glob("*.py"):
            copy_or_skip(f, PKG_DIR / "code" / src_dir.name / f.name, f"{label}/{f.name}")
    for cf in ("README.md", "config.py", "requirements.txt", "pyproject.toml"):
        candidate = config.PROJECT_ROOT / cf
        if candidate.exists():
            copy_or_skip(candidate, PKG_DIR / "code" / "residuals-visuals" / cf, f"residuals-visuals/{cf}")

    # Final size summary
    total = sum(p.stat().st_size for p in PKG_DIR.rglob("*") if p.is_file())
    print()
    print(f"Package built: {PKG_DIR}")
    print(f"Total size:    {total / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
