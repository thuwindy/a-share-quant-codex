from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.feature_panels import attach_premium_feature_panels
from ashare_quant.data.tushare_sync import load_local_history_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align premium Tushare raw tables into the local daily panel.")
    parser.add_argument("--daily-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--flow-path", default=str(ROOT / "data" / "tushare_flow_daily.csv"))
    parser.add_argument("--chip-path", default=str(ROOT / "data" / "tushare_chip_daily.csv"))
    parser.add_argument("--report-rc-path", default=str(ROOT / "data" / "tushare_report_rc_events.csv"))
    parser.add_argument("--hk-hold-path", default=str(ROOT / "data" / "tushare_hk_hold_daily.csv"))
    parser.add_argument("--limit-sentiment-path", default=str(ROOT / "data" / "tushare_limit_sentiment_daily.csv"))
    parser.add_argument("--output-path", default=str(ROOT / "data" / "a_share_daily_premium.csv"))
    return parser


def _load_optional(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source)


def main() -> None:
    args = build_parser().parse_args()
    daily = load_local_history_csv(args.daily_path)
    flow = _load_optional(args.flow_path)
    chip = _load_optional(args.chip_path)
    report_rc = _load_optional(args.report_rc_path)
    hk_hold = _load_optional(args.hk_hold_path)
    limit_sentiment = _load_optional(args.limit_sentiment_path)

    aligned = attach_premium_feature_panels(
        daily,
        moneyflow_df=flow,
        chip_df=chip,
        report_rc_df=report_rc,
        hk_hold_df=hk_hold,
        limit_sentiment_df=limit_sentiment,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    print(f"[OK] wrote aligned feature panel with {len(aligned)} rows -> {output_path}")


if __name__ == "__main__":
    main()
