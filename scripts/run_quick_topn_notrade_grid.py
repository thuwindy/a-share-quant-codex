from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.quick_grid import parse_float_grid, parse_int_grid, run_topn_notrade_quick_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quick grid search for Top-N and no-trade-band thresholds (T23)."
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity_stress_15bps.json"))
    parser.add_argument("--adjust", choices=("none", "qfq", "hfq"), default="qfq")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "quick_grid"))
    parser.add_argument("--output-prefix", default="quick_grid")
    parser.add_argument("--top-n-grid", default="20,30,40")
    parser.add_argument("--weight-change-grid", default="0.03,0.05")
    parser.add_argument("--rank-change-grid", default="8,15,20")
    parser.add_argument("--sleeve-grid", default="")
    parser.add_argument("--max-markdown-rows", type=int, default=20)
    return parser


def _build_markdown(
    *,
    summary_df,
    payload: dict,
    max_rows: int,
    generated_at: str,
) -> str:
    show_df = summary_df.head(max_rows).copy()
    return "\n".join(
        [
            "# Quick Grid Summary (T23)",
            "",
            f"- generated_at: `{generated_at}`",
            f"- data_path: `{payload.get('data_path', '')}`",
            f"- research_config: `{payload.get('research_config_path', '')}`",
            f"- backtest_config: `{payload.get('backtest_config_path', '')}`",
            f"- rows: `{payload.get('rows', 0)}`",
            "",
            "## Grid Definition",
            "",
            f"- top_n: `{payload.get('grid', {}).get('top_n', [])}`",
            f"- sleeve_count: `{payload.get('grid', {}).get('sleeve_count', [])}`",
            f"- weight_change_threshold: `{payload.get('grid', {}).get('weight_change_threshold', [])}`",
            f"- rank_change_threshold: `{payload.get('grid', {}).get('rank_change_threshold', [])}`",
            "",
            "## Top Results",
            "",
            show_df.to_markdown(index=False) if not show_df.empty else "_No rows._",
            "",
            "## Recommended Overrides",
            "",
            f"- `{json.dumps(payload.get('recommended_overrides', {}), ensure_ascii=False)}`",
            "",
        ]
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    top_n_grid = parse_int_grid(args.top_n_grid, fallback=[20, 30, 40])
    weight_grid = parse_float_grid(args.weight_change_grid, fallback=[0.03, 0.05])
    rank_grid = parse_int_grid(args.rank_change_grid, fallback=[8, 15, 20])
    sleeve_grid = parse_int_grid(args.sleeve_grid, fallback=[]) if str(args.sleeve_grid).strip() else None

    summary_df, payload = run_topn_notrade_quick_grid(
        data_path=args.data_path,
        research_config_path=args.research_config,
        backtest_config_path=args.backtest_config,
        data_adjust=args.adjust,
        top_n_grid=top_n_grid,
        weight_change_grid=weight_grid,
        rank_change_grid=rank_grid,
        sleeve_grid=sleeve_grid,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = out_dir / f"{args.output_prefix}_summary.csv"
    json_path = out_dir / f"{args.output_prefix}_summary.json"
    best_path = out_dir / f"{args.output_prefix}_best_overrides.json"
    md_path = out_dir / f"{args.output_prefix}_summary.md"

    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                **payload,
                "rows_detail": summary_df.to_dict(orient="records"),
                "generated_at": generated_at,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    best_path.write_text(
        json.dumps(payload.get("recommended_overrides", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(
        _build_markdown(summary_df=summary_df, payload=payload, max_rows=max(args.max_markdown_rows, 1), generated_at=generated_at),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "rows": int(len(summary_df)),
                "best": payload.get("best", {}),
                "recommended_overrides": payload.get("recommended_overrides", {}),
                "summary_csv_path": str(csv_path),
                "summary_json_path": str(json_path),
                "best_overrides_path": str(best_path),
                "summary_md_path": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"[OK] quick grid outputs saved to {out_dir}")
