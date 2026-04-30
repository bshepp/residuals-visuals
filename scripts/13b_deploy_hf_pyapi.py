"""
Robust HF deployment using huggingface_hub Python API directly.
Uploads each file individually with retry + progress, more debuggable than `hf upload`.

Usage: python scripts/13b_deploy_hf_pyapi.py --repo bshepp/residuals-fingerprints
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

PKG_DIR = config.PROJECT_ROOT / "package" / "hf"


def upload_one(api: HfApi, repo_id: str, local: Path, remote: str, max_retries: int = 3) -> None:
    size_mb = local.stat().st_size / 1024 / 1024
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"  uploading {remote} ({size_mb:.1f} MB) attempt {attempt}/{max_retries}...", flush=True)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {remote}",
            )
            elapsed = time.time() - t0
            mbps = size_mb / elapsed if elapsed > 0 else 0
            print(f"    ok ({elapsed:.1f}s, {mbps:.1f} MB/s)", flush=True)
            return
        except Exception as e:
            print(f"    attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == max_retries:
                raise
            time.sleep(5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not PKG_DIR.exists():
        print(f"ERROR: {PKG_DIR} missing.")
        sys.exit(1)

    api = HfApi()

    # Ensure repo exists
    print(f"Ensuring repo {args.repo} exists...")
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    # Check what's already uploaded so we can skip
    try:
        existing_files = set(api.list_repo_files(repo_id=args.repo, repo_type="dataset"))
        existing_files = {f for f in existing_files if f != ".gitattributes"}
        print(f"Existing files in repo: {len(existing_files)}")
    except Exception:
        existing_files = set()

    # Order: README + .gitattributes first (small), then parquet shards
    small = []
    large = []
    for fp in sorted(PKG_DIR.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(PKG_DIR)).replace("\\", "/")
        if rel == ".gitattributes":
            continue  # already on repo from create_repo
        (large if fp.stat().st_size > 50 * 1024 * 1024 else small).append((fp, rel))

    print(f"Files to upload: {len(small)} small + {len(large)} large = {len(small) + len(large)}")

    for fp, rel in small + large:
        if rel in existing_files:
            print(f"  skip (already on repo): {rel}")
            continue
        upload_one(api, args.repo, fp, rel)

    print()
    print(f"Done. View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
