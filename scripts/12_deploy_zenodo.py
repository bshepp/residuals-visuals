"""
Deploy the Zenodo package via the REST API.

Zenodo has no official CLI. This script:
  1. Creates a new draft deposition using the metadata in package/zenodo/metadata.json
  2. Uploads every file in package/zenodo/ (excluding metadata.json itself)
  3. Returns the draft URL — you log in and click "Publish" manually

Required: a Zenodo personal access token with `deposit:write` and `deposit:actions` scopes.

  Get one at:   https://zenodo.org/account/settings/applications/tokens/new/
  (or sandbox:  https://sandbox.zenodo.org/account/settings/applications/tokens/new/)

Usage:
  ZENODO_TOKEN=xxxxx python scripts/12_deploy_zenodo.py            # production
  ZENODO_TOKEN=xxxxx python scripts/12_deploy_zenodo.py --sandbox  # test against sandbox first

The script is idempotent on file uploads (skips files already present in the draft).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

PKG_DIR = config.PROJECT_ROOT / "package" / "zenodo"
DEPLOY_DIR = config.PROJECT_ROOT / "package" / "zenodo_deploy"


def prepare_flat_layout() -> Path:
    """
    Zenodo deposit files are flat (no subdirectories). Build a sibling
    directory `zenodo_deploy/` where each subdir of PKG_DIR is collapsed:
      - small bundles (code/, documentation/, visualizations/) -> single .zip
      - data/ files copied as-is (each is already independently useful)
      - top-level files (README.md, LICENSE) copied as-is
    """
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True)

    print(f"Preparing flat deploy layout at {DEPLOY_DIR}...")

    # 1. Top-level files copied as-is
    for top in PKG_DIR.iterdir():
        if top.is_file() and top.name != "metadata.json":
            shutil.copy2(top, DEPLOY_DIR / top.name)
            print(f"  copy: {top.name}")

    # 2. data/ subdir: each file at deploy root
    data_dir = PKG_DIR / "data"
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, DEPLOY_DIR / f.name)
                print(f"  copy: {f.name}")

    # 3. Bundle the rest into zips
    for sub_name in ("code", "documentation", "visualizations"):
        sub = PKG_DIR / sub_name
        if not sub.exists():
            continue
        zpath = DEPLOY_DIR / f"{sub_name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sub.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(sub.parent))
        print(f"  bundle: {zpath.name} ({zpath.stat().st_size / 1024:.0f} KB)")

    total = sum(p.stat().st_size for p in DEPLOY_DIR.iterdir() if p.is_file())
    print(f"Deploy layout: {len(list(DEPLOY_DIR.iterdir()))} files, {total / 1024**3:.2f} GiB")
    return DEPLOY_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org instead of zenodo.org")
    p.add_argument("--deposition-id", type=int, default=None, help="resume an existing draft instead of creating a new one")
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN environment variable.")
        print("Get one at https://zenodo.org/account/settings/applications/tokens/new/")
        sys.exit(1)

    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    headers = {"Authorization": f"Bearer {token}"}

    metadata_path = PKG_DIR / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} missing. Run scripts/10_package_zenodo.py first.")
        sys.exit(1)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Build (or reuse) the flat deploy layout
    if DEPLOY_DIR.exists() and any(DEPLOY_DIR.iterdir()):
        print(f"Reusing existing deploy layout at {DEPLOY_DIR}")
        deploy_dir = DEPLOY_DIR
    else:
        deploy_dir = prepare_flat_layout()

    # 1. Create or resume draft
    if args.deposition_id:
        print(f"Resuming deposition {args.deposition_id}")
        r = requests.get(f"{base}/deposit/depositions/{args.deposition_id}", headers=headers, timeout=30)
        r.raise_for_status()
        deposition = r.json()
    else:
        print(f"Creating new deposition on {base}")
        r = requests.post(f"{base}/deposit/depositions", headers=headers, json={}, timeout=30)
        r.raise_for_status()
        deposition = r.json()
        # Apply metadata
        r = requests.put(
            f"{base}/deposit/depositions/{deposition['id']}",
            headers=headers,
            json=metadata,
            timeout=30,
        )
        r.raise_for_status()
        deposition = r.json()

    print(f"Deposition ID: {deposition['id']}")
    print(f"Draft URL:     {deposition['links']['html']}")
    bucket_url = deposition["links"]["bucket"]
    print(f"Bucket URL:    {bucket_url}")

    # 2. List existing files in the draft (so we can skip already-uploaded ones)
    r = requests.get(f"{base}/deposit/depositions/{deposition['id']}/files", headers=headers, timeout=30)
    r.raise_for_status()
    existing = {f["filename"]: f for f in r.json()}
    print(f"Existing files in draft: {len(existing)}")

    # 3. Walk the FLAT deploy dir, upload each file
    files = sorted([p_ for p_ in deploy_dir.iterdir() if p_.is_file()])
    total_bytes = sum(p_.stat().st_size for p_ in files)
    print(f"Files to upload: {len(files)} ({total_bytes / 1024**3:.2f} GiB)")

    for fp in files:
        key = fp.name
        size = fp.stat().st_size
        if key in existing and existing[key].get("filesize") == size:
            print(f"  skip (already uploaded): {key}")
            continue

        print(f"  uploading {key} ({size / 1024 / 1024:.1f} MB)...", flush=True)

        # Wrap file in a tqdm reader so we get progress without using
        # transfer-encoding: chunked (which Zenodo's bucket API rejects).
        with open(fp, "rb") as raw:
            with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar:
                # CallbackIOWrapper passes a real seekable file with known length
                from tqdm.utils import CallbackIOWrapper
                wrapped = CallbackIOWrapper(pbar.update, raw, "read")
                r = requests.put(
                    f"{bucket_url}/{key}",
                    data=wrapped,
                    headers={**headers, "Content-Length": str(size)},
                    timeout=None,
                )
        if r.status_code not in (200, 201):
            print(f"    FAILED ({r.status_code}): {r.text[:300]}")
            sys.exit(1)
        print(f"    ok", flush=True)

    print()
    print(f"All files uploaded.")
    print(f"Visit {deposition['links']['html']} to review and click PUBLISH.")
    print(f"(After publishing, the DOI is permanent. Stay in draft to continue editing.)")


if __name__ == "__main__":
    main()
