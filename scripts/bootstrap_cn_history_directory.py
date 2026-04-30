from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.cn_history_dir import list_cn_history_files, normalize_cn_history_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a directory of per-stock CN daily CSV files into the repo's canonical daily schema."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing one CSV per stock.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data" / "a_share_daily.csv"),
        help="Output CSV path in the canonical schema.",
    )
    parser.add_argument("--start-date", default=None, help="Optional start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional end date in YYYY-MM-DD.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for how many stock files to process; useful for smoke tests.",
    )
    parser.add_argument(
        "--code-prefix",
        default=None,
        help="Optional prefix filter, e.g. 000 or 60, to build a narrower starter universe.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    files = list_cn_history_files(args.source_dir)
    if args.code_prefix:
        files = [path for path in files if path.stem.startswith(args.code_prefix)]
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError("No source CSV files matched the requested filters.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    for idx, path in enumerate(files, start=1):
        frame = normalize_cn_history_file(path, start=args.start_date, end=args.end_date)
        if frame.empty:
            continue
        frame.to_csv(output_path, index=False, mode="a", header=not output_path.exists())
        total_rows += len(frame)
        if idx % 100 == 0 or idx == len(files):
            print(f"[PROGRESS] files={idx}/{len(files)} rows={total_rows} latest={path.name}")

    print(f"[OK] wrote canonical daily dataset to {output_path} rows={total_rows} files={len(files)}")
