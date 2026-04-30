# residuals-visuals

Artistic and educational visualizations of the **RESIDUALS exhaustive parameter sweep** — 39,731 residual arrays produced by running 593 distinct decomposition configurations × 25 upsampling configurations on a Fairfield County, Ohio LiDAR DEM.

- **Project page**: https://residuals.briansheppard.com
- **Dataset (HF)**: https://huggingface.co/datasets/bshepp/residuals-fingerprints
- **Archive (Zenodo, DOI)**: https://doi.org/10.5281/zenodo.19903273
- **Source pipeline**: https://github.com/bshepp/RESIDUALS

This project is **standalone** — it does not import from RESIDUALS. It only consumes the `.npy` outputs (4.28 TB) at `D:/DIVERGE_exhaustive/results/` and the source DEM at `F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy`.

## What it makes

1. **Algorithm zoo grid** — single high-res poster: same terrain crop seen through ~24 different decomposition algorithms, side by side. Educational + striking.
2. **Parameter sweep videos** — short MP4 loops scrubbing through one parameter axis per algorithm family (gaussian σ, anisotropic_diffusion iterations, wavelet level, etc.).
3. **UMAP atlas** — all 39,731 residuals embedded into 2D by signature (FFT radial spectrum + statistical moments), rendered as a thumbnail mosaic colored by decomposition category.

## Crop

All visuals use a fixed **640×640 crop at scale=2 coordinates**, centered at `(row=1600, col=400)`. This window contains a sweeping diagonal stream channel, branching drainage, a sharp V-shaped linear feature, and dramatic ridge/embankment patterns. It dodges the left-edge upsampling artifact band.

The crop scales linearly with upsampling factor — at scale=4 it's 1280×1280, at scale=8 it's 2560×2560, etc. All thumbnails normalize to 256×256 in the cache.

## Pipeline

| Phase | Script | What | Cost |
|-------|--------|------|------|
| 0 | `scripts/01_build_catalog.py` | Parse all 39,731 filenames into a parquet table (decomp family, params, upsamp method, scale, path). | seconds |
| A | `scripts/04_compute_signatures.py` | Throttled scan of all 4 TB. For each `.npy`: mmap-load → crop → 256×256 thumbnail + 64-dim signature vector. Writes `cache/thumbnails.zarr` (~10 GB) + `cache/signatures.parquet` (~10 MB). Single-threaded, low priority, 50 ms inter-file sleep, checkpointed every 200 files. **Run once, overnight.** | 3–4 hr (throttled) |
| B1 | `scripts/02_build_zoo.py` | Render the 24-panel zoo grid from cache. | < 1 min |
| B2 | `scripts/03_build_sweeps.py` | Render 6 parameter-sweep MP4 videos from cache. | ~5 min |
| B3 | `scripts/05_compute_umap.py` | Fit UMAP on signatures. | ~2 min |
| B4 | `scripts/06_render_atlas.py` | Plot all 39,731 thumbnails at their UMAP positions. | ~5 min |

## Install

```bash
cd F:/science-projects/residuals-visuals  # local path, may differ on your system
pip install -r requirements.txt
```

## Run

```bash
# Once:
python scripts/01_build_catalog.py
python scripts/04_compute_signatures.py        # heavy, throttled, overnight

# Then iteratively:
python scripts/02_build_zoo.py
python scripts/03_build_sweeps.py
python scripts/05_compute_umap.py
python scripts/06_render_atlas.py
```

Outputs land in `output/zoo/`, `output/sweeps/`, `output/atlas/`.

## Data sources

- **Residuals**: `D:/DIVERGE_exhaustive/results/` (4.28 TB, 39,731 `.npy` files)
- **Source DEM**: `F:/science-projects/RESIDUALS/data/test_dems/fairfield_sample_1.5ft.npy` (1500×375, 3.33 ft/px, Fairfield County OH)
- **Method documentation**: `D:/DIVERGE_exhaustive/documentation/PRIOR_ART_20260105_134622.md`

## License

Apache 2.0 — same as the source RESIDUALS project.
