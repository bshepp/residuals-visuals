"""
Deploy the HF dataset package via the `hf` CLI (already authenticated).

Steps:
  1. Create the dataset repo (idempotent — does nothing if it exists).
  2. Upload everything in package/hf/ to the repo root.

Usage:
  python scripts/13_deploy_hf.py --repo bshepp/diverge-residuals
  python scripts/13_deploy_hf.py --repo bshepp/diverge-residuals --private  # private first, flip to public later via web
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

PKG_DIR = config.PROJECT_ROOT / "package" / "hf"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help='HF dataset repo id, e.g. "bshepp/diverge-residuals"')
    p.add_argument("--private", action="store_true", help="create as private dataset (default: public)")
    args = p.parse_args()

    if not PKG_DIR.exists():
        print(f"ERROR: {PKG_DIR} missing. Run scripts/11_package_hf.py first.")
        sys.exit(1)

    # 1. Create the repo (no-op if it already exists)
    create_cmd = ["hf", "repo", "create", args.repo, "--type", "dataset", "--exist-ok"]
    if args.private:
        create_cmd.append("--private")
    run(create_cmd)

    # 2. Upload the entire package directory.
    # `hf upload` syntax: hf upload <repo_id> <local_path> [<path_in_repo>] --repo-type dataset
    # Passing "." for path_in_repo uploads contents of local_path to repo root.
    upload_cmd = [
        "hf", "upload", args.repo, str(PKG_DIR), ".",
        "--repo-type", "dataset",
        "--commit-message", "Initial upload: residuals fingerprints dataset",
    ]
    rc = run(upload_cmd)
    if rc.returncode != 0:
        print()
        print(f"Upload exited non-zero ({rc.returncode}). Common causes:")
        print(" - LFS not initialized: run `git lfs install` once on this machine")
        print(" - Token lacks write scope: re-run `hf auth login` and choose 'write' scope")
        sys.exit(rc.returncode)

    print()
    print(f"Done. View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
