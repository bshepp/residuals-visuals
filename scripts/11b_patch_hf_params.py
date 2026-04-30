"""
Patch the existing HF parquet shards to fix decomp_params / upsamp_params,
which got written as '{}' due to a type-check bug in 11_package_hf.py.

Reads each shard, replaces those two columns with proper JSON strings derived
from src.catalog.load_catalog(), and rewrites the shard.

Usage: python scripts/11b_patch_hf_params.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.catalog import load_catalog  # noqa: E402

DATA_DIR = config.PROJECT_ROOT / "package" / "hf" / "data"


def main() -> None:
    catalog = load_catalog(config.CATALOG_PATH)
    name_to_decomp = dict(zip(catalog["filename"], catalog["decomp_params"]))
    name_to_upsamp = dict(zip(catalog["filename"], catalog["upsamp_params"]))

    shards = sorted(DATA_DIR.glob("train-*.parquet"))
    print(f"Patching {len(shards)} shards in {DATA_DIR}")

    for shard_path in shards:
        print(f"  {shard_path.name} ...", end=" ", flush=True)
        tbl = pq.read_table(shard_path)
        filenames = tbl.column("filename").to_pylist()

        new_decomp = [json.dumps(name_to_decomp.get(fn, {})) for fn in filenames]
        new_upsamp = [json.dumps(name_to_upsamp.get(fn, {})) for fn in filenames]

        tbl = tbl.set_column(
            tbl.schema.get_field_index("decomp_params"),
            pa.field("decomp_params", pa.string()),
            pa.array(new_decomp, type=pa.string()),
        )
        tbl = tbl.set_column(
            tbl.schema.get_field_index("upsamp_params"),
            pa.field("upsamp_params", pa.string()),
            pa.array(new_upsamp, type=pa.string()),
        )

        # Rewrite atomically: write to .tmp, replace
        tmp = shard_path.with_suffix(".parquet.tmp")
        pq.write_table(tbl, tmp, compression="zstd", compression_level=10)
        tmp.replace(shard_path)
        print(f"done ({tbl.num_rows} rows)")

    # Sanity check first shard
    t = pq.read_table(shards[0])
    print()
    print("Verification — first shard, row 0:")
    r0 = t.to_pylist()[0]
    print(f"  filename:      {r0['filename']}")
    print(f"  decomp_family: {r0['decomp_family']}")
    print(f"  decomp_params: {r0['decomp_params']}")
    print(f"  upsamp_method: {r0['upsamp_method']}")
    print(f"  upsamp_params: {r0['upsamp_params']}")


if __name__ == "__main__":
    main()
