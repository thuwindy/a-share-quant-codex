from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import apply_stock_metadata_to_frame, normalize_local_history_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge stock-level metadata from a local CSV into the canonical A-share daily dataset."
    )
    parser.add_argument("--existing-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--metadata-path", required=True, help="CSV with at least a code column.")
    parser.add_argument("--output-path", default=None, help="Defaults to overwriting --existing-path.")
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Overwrite existing non-empty metadata instead of only filling missing/Unknown fields.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    existing_path = Path(args.existing_path)
    metadata_path = Path(args.metadata_path)
    if not existing_path.exists():
        raise FileNotFoundError(f"{existing_path} does not exist.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} does not exist.")

    metadata = pd.read_csv(metadata_path)
    if "code" not in metadata.columns:
        raise ValueError("Metadata CSV must contain a `code` column.")
    metadata["code"] = metadata["code"].astype(str)
    keep_cols = ["code", "industry", "name", "list_status", "list_date", "delist_date"]
    metadata = metadata[[col for col in keep_cols if col in metadata.columns]].drop_duplicates(subset=["code"], keep="first")

    target_path = Path(args.output_path or existing_path)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    total_rows = 0
    for chunk in pd.read_csv(existing_path, chunksize=args.chunk_size):
        normalized = normalize_local_history_frame(chunk)
        enriched = apply_stock_metadata_to_frame(
            normalized,
            metadata=metadata,
            overwrite_unknown_only=not args.overwrite_all,
        )
        enriched.to_csv(temp_path, index=False, mode="a", header=not wrote_any)
        wrote_any = True
        total_rows += len(enriched)

    if target_path.exists():
        target_path.unlink()
    temp_path.rename(target_path)
    print(f"[OK] merged metadata for {total_rows} rows -> {target_path}")
