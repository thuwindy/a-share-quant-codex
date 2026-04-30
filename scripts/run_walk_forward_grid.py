from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.walk_forward import (
    build_walk_forward_summary_markdown,
    run_walk_forward_analysis,
)
from ashare_quant.data.research_slice import build_research_slice


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare walk-forward performance across multiple universe sizes and train windows."
    )
    parser.add_argument("--input-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--max-codes-list",
        default="200,500",
        help="Comma-separated universe sizes. Use 0 to keep the full eligible universe.",
    )
    parser.add_argument(
        "--train-years-list",
        default="2,3,5",
        help="Comma-separated train-window lengths in years.",
    )
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument(
        "--suffixes",
        default="SH,SZ",
        help="Comma-separated market suffixes to keep in the research slice.",
    )
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest.json"))
    parser.add_argument("--output-prefix", default="walk_forward")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    suffixes = tuple(part.strip().upper() for part in args.suffixes.split(",") if part.strip())
    universe_sizes = _parse_int_list(args.max_codes_list)
    train_years_list = _parse_int_list(args.train_years_list)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    case_rows: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    factor_frames: list[pd.DataFrame] = []
    slice_profiles: list[dict] = []

    for max_codes in universe_sizes:
        universe_tag = "all" if max_codes <= 0 else str(max_codes)
        slice_path = ROOT / "data" / f"{args.output_prefix}_slice_{universe_tag}.csv"
        profile = build_research_slice(
            input_path=args.input_path,
            output_path=slice_path,
            start_date=args.start_date,
            end_date=args.end_date,
            include_suffixes=suffixes,
            max_codes=max_codes,
        )
        slice_profiles.append({"universe_tag": universe_tag, **profile.__dict__})

        for train_years in train_years_list:
            summary, folds_df, factor_stats_df = run_walk_forward_analysis(
                data_path=slice_path,
                research_config_path=args.research_config,
                backtest_config_path=args.backtest_config,
                train_years=train_years,
                test_years=args.test_years,
                step_years=args.step_years,
                data_adjust=args.adjust,
            )
            case_summary = {
                "universe_tag": universe_tag,
                "max_codes": max_codes,
                "rows": profile.rows,
                "unique_codes": profile.unique_codes,
                **summary,
            }
            case_rows.append(case_summary)

            folds_df = folds_df.copy()
            folds_df["universe_tag"] = universe_tag
            folds_df["max_codes"] = max_codes
            folds_df["train_years"] = train_years
            fold_frames.append(folds_df)

            factor_stats_df = factor_stats_df.copy()
            factor_stats_df["universe_tag"] = universe_tag
            factor_stats_df["max_codes"] = max_codes
            factor_stats_df["train_years"] = train_years
            factor_frames.append(factor_stats_df)

            print(
                json.dumps(
                    {
                        "universe_tag": universe_tag,
                        "train_years": train_years,
                        "mean_sharpe": summary["mean_sharpe"],
                        "mean_annual_return": summary["mean_annual_return"],
                        "best_factor": summary["best_factor"],
                    },
                    ensure_ascii=False,
                )
            )

    case_df = pd.DataFrame(case_rows).sort_values(["mean_sharpe", "mean_annual_return"], ascending=False)
    folds_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    factor_df = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    profile_df = pd.DataFrame(slice_profiles)

    case_path = out_dir / f"{args.output_prefix}_cases.csv"
    fold_path = out_dir / f"{args.output_prefix}_folds.csv"
    factor_path = out_dir / f"{args.output_prefix}_factors.csv"
    profile_path = out_dir / f"{args.output_prefix}_slice_profiles.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary.md"

    case_df.to_csv(case_path, index=False)
    folds_df.to_csv(fold_path, index=False)
    factor_df.to_csv(factor_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    summary_path.write_text(
        build_walk_forward_summary_markdown(case_summaries=case_df, factor_stats=factor_df),
        encoding="utf-8",
    )

    print(f"[OK] walk-forward outputs saved to {out_dir}")
