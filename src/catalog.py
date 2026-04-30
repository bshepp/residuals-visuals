"""
Parse the 39,731 .npy filenames into a structured catalog.

Filename format: {decomp_name}_{paramN}_{paramN+1}...___{upsamp_name}_scale{N}.npy

We extract:
  - decomp_family   (e.g. "gaussian", "wavelet_biorthogonal", "morphological_rect")
  - decomp_params   (dict of name -> value)
  - upsamp_method   (e.g. "bicubic", "lanczos", "sinc_hamming")
  - upsamp_params   (dict, including 'scale')
  - scale           (int — for crop sizing)
  - path            (relative to DATA_DIR)
  - size_bytes      (file size on disk)

Family detection uses a known-prefix list because some families have
underscore-separated multi-word names (e.g. "wavelet_biorthogonal", "rolling_ball").
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import config

# Order matters: longest prefixes first so "wavelet_biorthogonal" wins over "wavelet_dwt".
DECOMP_FAMILIES = [
    "wavelet_reverse_biorthogonal",
    "wavelet_biorthogonal",
    "wavelet_dwt",
    "morphological_gradient",
    "morphological_diamond",
    "morphological_ellipse",
    "morphological_square",
    "morphological_rect",
    "morphological",
    "anisotropic_diffusion",
    "gaussian_anisotropic",
    "dog_multiscale",
    "tophat_combined",
    "rolling_ball",
    "local_polynomial",
    "polynomial_high",
    "polynomial",
    "bilateral",
    "gaussian",
    "uniform",
    "median",
    "tophat",
    "guided",
    "dog",
    "log",
]

UPSAMP_METHODS = [
    "bicubic_3x",
    "bicubic_16x",
    "bicubic",
    "lanczos",
    "bspline",
    "fft_zeropad",
    "nearest",
    "bilinear",
    "quadratic",
    "quartic",
    "quintic",
    "area",
    "linear_exact",
    "sinc_hamming",
    "sinc_blackman",
    "cubic_catmull_rom",
    "cubic_mitchell",
    "edge_directed",
    "regularized",
]


def _detect_family(prefix: str, families: list[str]) -> str | None:
    for f in families:
        if prefix == f or prefix.startswith(f + "_"):
            return f
    return None


_PARAM_RE = re.compile(r"([a-z_]+?)([-+]?[0-9]*\.?[0-9]+|[A-Za-z][A-Za-z0-9.]*)$")


def _parse_params(tail: str, family: str) -> dict[str, str | float]:
    """
    Parse a param tail like 'd9_sigma_color75_sigma_space75' into a dict.

    The decomposition methods use multi-word param names with underscores
    (e.g. 'sigma_color', 'n_scales'), so a naive split on '_' is wrong.
    Strategy: greedily match known param names from a per-family list, then
    extract their values.
    """
    if not tail:
        return {}

    # Known params per family (from the prior-art doc)
    family_params = {
        "gaussian": ["sigma"],
        "gaussian_anisotropic": ["sigma_x", "sigma_y"],
        "bilateral": ["d", "sigma_color", "sigma_space"],
        "wavelet_dwt": ["wavelet", "level"],
        "wavelet_biorthogonal": ["wavelet", "level"],
        "wavelet_reverse_biorthogonal": ["wavelet", "level"],
        "morphological": ["operation", "size"],
        "morphological_square": ["operation", "size"],
        "morphological_rect": ["operation", "width", "height"],
        "morphological_diamond": ["operation", "radius"],
        "morphological_ellipse": ["operation", "width", "height"],
        "morphological_gradient": ["size", "shape"],
        "tophat": ["size", "mode"],
        "tophat_combined": ["size"],
        "polynomial": ["degree"],
        "polynomial_high": ["degree"],
        "local_polynomial": ["window_size", "degree"],
        "anisotropic_diffusion": ["iterations", "kappa", "gamma"],
        "rolling_ball": ["radius"],
        "uniform": ["size"],
        "median": ["size"],
        "guided": ["radius", "eps"],
        "dog": ["sigma_low", "sigma_high"],
        "dog_multiscale": ["sigma_ratio", "n_scales", "base_sigma"],
        "log": ["sigma"],
    }
    expected = family_params.get(family, [])
    out: dict[str, str | float] = {}
    remaining = tail
    # Try greedy matching: for each expected param, find "{name}{value}" prefix.
    # Params can appear in any order, so we loop until nothing matches.
    progress = True
    while remaining and progress:
        progress = False
        for pname in expected:
            for sep in ("_" + pname, pname):
                idx = remaining.find(sep)
                if idx == -1:
                    continue
                # value starts right after the param name
                vstart = idx + len(sep)
                # value runs to next '_' followed by another known param name, or end
                vend = len(remaining)
                for other in expected:
                    if other == pname:
                        continue
                    candidate = remaining.find("_" + other, vstart)
                    if candidate != -1 and candidate < vend:
                        vend = candidate
                value = remaining[vstart:vend]
                if not value:
                    continue
                # cast numeric values
                try:
                    out[pname] = float(value)
                    if out[pname].is_integer():
                        out[pname] = int(out[pname])
                except ValueError:
                    out[pname] = value
                # remove matched span (preserves remaining params)
                remaining = (remaining[:idx] + remaining[vend:]).strip("_")
                progress = True
                break
            if progress:
                break
    return out


def _parse_upsamp(part: str) -> tuple[str, dict[str, str | float], int]:
    """
    Parse the right side of '___', e.g. 'sinc_hamming_kernel_size16_scale8'
    -> ('sinc_hamming', {'kernel_size': 16, 'scale': 8}, 8)
    """
    method = None
    for m in UPSAMP_METHODS:
        if part == m or part.startswith(m + "_") or part.startswith(m):
            # need to ensure exact prefix match (avoid 'bicubic' matching 'bicubic_3x')
            if part == m or part[len(m)] == "_":
                method = m
                break
    if method is None:
        # Last resort: split on '_scale'
        method = part.rsplit("_scale", 1)[0]
    tail = part[len(method) :].lstrip("_")

    # Extract scale
    scale_match = re.search(r"scale(\d+)", tail)
    scale = int(scale_match.group(1)) if scale_match else 1

    params: dict[str, str | float] = {"scale": scale}
    # Other upsamp params: kernel_size (sinc), lambda_reg (regularized)
    ks = re.search(r"kernel_size(\d+)", tail)
    if ks:
        params["kernel_size"] = int(ks.group(1))
    lr = re.search(r"lambda_reg([0-9.]+)", tail)
    if lr:
        params["lambda_reg"] = float(lr.group(1))

    return method, params, scale


def parse_filename(name: str) -> dict | None:
    """Parse one filename into a row dict, or None if it doesn't match the format."""
    if not name.endswith(".npy") or "___" not in name:
        return None
    stem = name[:-4]
    decomp_part, upsamp_part = stem.split("___", 1)

    family = _detect_family(decomp_part, DECOMP_FAMILIES)
    if family is None:
        return None

    decomp_tail = decomp_part[len(family) :].lstrip("_")
    decomp_params = _parse_params(decomp_tail, family)

    upsamp_method, upsamp_params, scale = _parse_upsamp(upsamp_part)

    return {
        "filename": name,
        "decomp_family": family,
        "decomp_params": decomp_params,
        "upsamp_method": upsamp_method,
        "upsamp_params": upsamp_params,
        "scale": scale,
    }


def build_catalog(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Walk DATA_DIR, parse every .npy filename, return a DataFrame."""
    rows = []
    skipped = 0
    for entry in data_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".npy":
            continue
        parsed = parse_filename(entry.name)
        if parsed is None:
            skipped += 1
            continue
        parsed["size_bytes"] = entry.stat().st_size
        rows.append(parsed)

    df = pd.DataFrame(rows)
    df = df.sort_values(["decomp_family", "filename"]).reset_index(drop=True)
    if skipped:
        print(f"  skipped {skipped} files that did not match the filename format")
    return df


def save_catalog(df: pd.DataFrame, path: Path = config.CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert dict columns to JSON strings for parquet compatibility
    out = df.copy()
    out["decomp_params"] = out["decomp_params"].apply(lambda d: str(d))
    out["upsamp_params"] = out["upsamp_params"].apply(lambda d: str(d))
    out.to_parquet(path, engine="pyarrow", index=False)


def load_catalog(path: Path = config.CATALOG_PATH) -> pd.DataFrame:
    import ast

    df = pd.read_parquet(path, engine="pyarrow")
    df["decomp_params"] = df["decomp_params"].apply(ast.literal_eval)
    df["upsamp_params"] = df["upsamp_params"].apply(ast.literal_eval)
    return df
