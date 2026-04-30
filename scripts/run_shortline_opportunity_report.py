from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_sync import sync_tushare_limit_sentiment_daily


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build under-20 shortline opportunity cards after market close.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--premium-dir", default=str(ROOT / "data" / "premium_v22"))
    parser.add_argument("--limit-file", default="tushare_limit_sentiment_daily.csv")
    parser.add_argument("--config", default=str(ROOT / "configs" / "research_shortline_opportunity.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "shortline_opportunities"))
    parser.add_argument("--output-prefix", default="shortline_opportunity")
    parser.add_argument("--prediction-date", default="")
    parser.add_argument("--sync-limit-data", action="store_true")
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    return parser


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_numeric(series: pd.Series | object, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(default)


def _parse_up_stat_streak(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("")
    streak = text.str.extract(r"^\s*(\d+)")[0]
    return pd.to_numeric(streak, errors="coerce").fillna(0.0)


def _format_hhmmss(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    digits = digits.zfill(6)[-6:]
    return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"


def _latest_trading_dates(df: pd.DataFrame, end_date: pd.Timestamp, lookback: int) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique().tolist())
    dates = [pd.Timestamp(item) for item in dates if pd.Timestamp(item) <= end_date]
    return dates[-lookback:] if len(dates) > lookback else dates


def _infer_latest_date_from_csv(path: Path) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=["date"], chunksize=300_000, low_memory=False):
        ts = pd.to_datetime(chunk["date"], errors="coerce").max()
        if pd.isna(ts):
            continue
        latest = ts if latest is None or ts > latest else latest
    if latest is None:
        raise ValueError(f"Could not infer latest date from {path}.")
    return pd.Timestamp(latest)


def _load_recent_daily_panel(path: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    usecols = ["date", "code", "close", "industry", "amount", "market_cap", "name"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=300_000, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.loc[chunk["date"].ge(start_date)].copy()
        if chunk.empty:
            continue
        pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["code"] = out["code"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def _normalize_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.0).clip(0.0, 1.0)


def _turnover_quality(turnover: pd.Series) -> pd.Series:
    # Prefer moderate activity; too cold lacks participation, too hot risks exhaustion.
    distance = (turnover - 8.0).abs()
    quality = 1.0 - (distance / 12.0)
    return quality.clip(0.0, 1.0)


def _float_mv_quality(float_mv: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    low = float(cfg.get("strong_float_mv_min", 5_000_000_000.0))
    high = float(cfg.get("strong_float_mv_max", 40_000_000_000.0))
    out = pd.Series(0.35, index=float_mv.index, dtype=float)
    out = out.where(~float_mv.between(low, high, inclusive="both"), 1.0)
    out = out.where(~float_mv.between(high, high * 3, inclusive="right"), 0.65)
    out = out.where(~float_mv.lt(low), 0.55)
    return out.clip(0.0, 1.0)


def _recent_move_quality(move4: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    high = float(cfg.get("high_recent_move_threshold", 0.2))
    base = 1.0 - ((move4 - 0.10).abs() / 0.20)
    base = base.clip(0.0, 1.0)
    return np.where(move4 > high, np.maximum(base * 0.65, 0.15), base)


def _seal_quality(open_times: pd.Series) -> pd.Series:
    out = pd.Series(1.0, index=open_times.index, dtype=float)
    out = out.where(~open_times.eq(1), 0.8)
    out = out.where(~open_times.eq(2), 0.65)
    out = out.where(~open_times.ge(3), 0.45)
    return out.clip(0.0, 1.0)


def _suggest_band(row: pd.Series, cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    close_px = float(row["close"])
    open_times = float(row.get("open_times", 0.0) or 0.0)
    recent_move = float(row["recent_4d_return"])
    grade = str(row["grade"])
    style = str(row["style"])

    low_pct = 0.0
    high_pct = 0.03
    if open_times > 0:
        low_pct, high_pct = -0.015, 0.015
    if recent_move > float(cfg.get("high_recent_move_threshold", 0.2)):
        high_pct = min(high_pct, 0.01)

    if style == "稳健":
        low_pct = max(low_pct, -0.005)
        high_pct = min(high_pct, 0.02)

    buy_low = close_px * (1.0 + low_pct)
    buy_high = close_px * (1.0 + high_pct)

    if grade == "A" and style == "进攻":
        target_low = close_px * (1.0 + float(cfg.get("aggressive_target_pct_low", 0.07)))
        target_high = close_px * (1.0 + float(cfg.get("aggressive_target_pct_high", 0.10)))
    elif grade in {"A", "B"}:
        target_low = close_px * (1.0 + float(cfg.get("steady_target_pct_low", 0.05)))
        target_high = close_px * (1.0 + float(cfg.get("steady_target_pct_high", 0.08)))
    else:
        target_low = close_px * (1.0 + float(cfg.get("watch_target_pct_low", 0.03)))
        target_high = close_px * (1.0 + float(cfg.get("watch_target_pct_high", 0.05)))

    stop_pct = float(cfg.get("tight_stop_loss_pct", 0.035)) if open_times > 0 else float(cfg.get("base_stop_loss_pct", 0.05))
    stop_line = buy_low * (1.0 - stop_pct)

    if grade == "A" and style == "稳健":
        position = "不超过总仓位 15%"
    elif grade == "A":
        position = "不超过总仓位 10%"
    elif grade == "B":
        position = "不超过总仓位 8%"
    else:
        position = "不超过总仓位 5%"

    return (
        f"{buy_low:.2f} ~ {buy_high:.2f}",
        f"{target_low:.2f} ~ {target_high:.2f}",
        f"跌破 {stop_line:.2f} 止损",
        position,
    )


def _build_logic(row: pd.Series) -> str:
    board = int(row["current_streak"])
    open_count = int(row["open_board_count_20d"])
    fd_amt = float(row["fd_amount"]) / 100000000.0
    turnover = float(row["turnover_ratio"])
    move_text = str(row["recent_4d_path"])
    first_time = str(row["first_time_fmt"])
    if first_time == "-":
        first_time = "盘中"
    return (
        f"{row['industry']}方向活跃，{board}连板，近20日炸板{open_count}次，"
        f"封单金额约{fd_amt:.2f}亿，今日换手{turnover:.2f}%，"
        f"近4日走势 {move_text}。若次日竞价强于预期且不高开过多，可优先观察。"
    )


def _render_markdown(selection_date: str, cards: pd.DataFrame, summary: dict[str, Any], output_json: Path) -> str:
    lines = [
        "# 精选短线机会（明日可关注）",
        "",
        f"- selection_date: `{selection_date}`",
        f"- source: `{output_json}`",
        f"- selected_count: `{summary.get('selected_count', 0)}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "- ranking: `按 shortline_score 从高到低排序`",
        "",
    ]
    for row in cards.itertuples(index=False):
        lines.extend(
            [
                f"## {row.name}",
                "",
                f"- 排名: `{int(row.rank)}`",
                f"- 评分: `{row.shortline_score:.4f}`",
                f"- 代码: `{row.code}`",
                f"- 收盘价: `{row.close:.2f}`",
                f"- 近一个月连板次数: `{int(row.limit_up_count_20d)}`",
                f"- 当前连板高度: `{int(row.current_streak)}`",
                f"- 炸板次数: `{int(row.open_board_count_20d)}`",
                f"- 换手率: `{row.turnover_ratio:.2f}%`",
                f"- 封单金额: `{row.fd_amount / 100000000.0:.2f}亿`",
                f"- 流通市值: `{row.float_mv / 100000000.0:.2f}亿`",
                f"- 近4日涨幅走势: `{row.recent_4d_path}`",
                f"- 明日买入区间: `{row.buy_range}`",
                f"- 目标价: `{row.target_price}`",
                f"- 止损线: `{row.stop_loss}`",
                f"- 建议仓位: `{row.suggested_position}`",
                f"- 档位: `{row.grade}` / `{row.style}`",
                f"- 操作逻辑: {row.operation_logic}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = load_json(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    premium_dir = Path(args.premium_dir)
    premium_dir.mkdir(parents=True, exist_ok=True)
    limit_path = premium_dir / args.limit_file

    latest_daily_date = _infer_latest_date_from_csv(data_path)

    if args.sync_limit_data:
        sync_start = max((latest_daily_date - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), "2019-01-01")
        sync_tushare_limit_sentiment_daily(
            existing_path=limit_path,
            start_date=sync_start,
            end_date=latest_daily_date.strftime("%Y-%m-%d"),
            http_url=args.http_url,
            proxy_url=args.proxy_url,
            bypass_system_proxy=args.bypass_system_proxy,
            pause_seconds=args.pause_seconds,
        )

    limit_df = pd.read_csv(limit_path, low_memory=False)
    limit_df["date"] = pd.to_datetime(limit_df["date"], errors="coerce")
    selection_date = pd.Timestamp(args.prediction_date) if args.prediction_date else pd.Timestamp(limit_df["date"].max())
    if pd.isna(selection_date):
        raise ValueError("No valid selection date could be inferred from limit sentiment data.")

    lookback = int(cfg.get("lookback_trading_days", 20))
    daily = _load_recent_daily_panel(data_path, selection_date - pd.Timedelta(days=90))
    trade_dates = _latest_trading_dates(daily, selection_date, lookback)
    if not trade_dates:
        raise ValueError("No recent trading dates available for shortline report.")

    daily_sel = daily.loc[daily["date"].isin(trade_dates), ["date", "code", "close", "industry", "amount", "market_cap", "name"]].copy()
    latest_daily = daily_sel.loc[daily_sel["date"] == selection_date].copy()
    if latest_daily.empty:
        raise ValueError(f"Daily master does not contain selection date {selection_date.date()}.")

    limit_df["close"] = _safe_numeric(limit_df.get("close"))
    limit_df["turnover_ratio"] = _safe_numeric(limit_df.get("turnover_ratio"))
    limit_df["fd_amount"] = _safe_numeric(limit_df.get("fd_amount"))
    limit_df["float_mv"] = _safe_numeric(limit_df.get("float_mv"))
    limit_df["amount"] = _safe_numeric(limit_df.get("amount"))
    limit_df["open_times"] = _safe_numeric(limit_df.get("open_times"))
    limit_df["current_streak"] = _parse_up_stat_streak(limit_df.get("up_stat", pd.Series(dtype=str)))
    limit_df["limit_flag"] = limit_df.get("limit", pd.Series("", index=limit_df.index)).astype("string").str.upper()
    history = limit_df.loc[limit_df["date"].isin(trade_dates)].copy()
    history = history.loc[history["close"].le(float(cfg.get("max_price", 20.0)))]
    if bool(cfg.get("prefer_limit_up_only", True)):
        history = history.loc[history["limit_flag"].eq("U")]

    grouped = (
        history.groupby("code", as_index=False)
        .agg(
            limit_up_count_20d=("limit_flag", lambda s: int((s == "U").sum())),
            open_board_count_20d=("open_times", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        )
    )

    latest_limit = history.loc[history["date"] == selection_date].copy()
    latest_limit = latest_limit.merge(grouped, on="code", how="left")
    latest_limit = latest_limit.merge(
        latest_daily[["code", "close", "industry", "amount", "market_cap", "name"]],
        on="code",
        how="left",
        suffixes=("_limit", ""),
    )
    latest_limit["close"] = _safe_numeric(latest_limit["close"]).where(lambda s: s.gt(0), _safe_numeric(latest_limit["close_limit"]))
    latest_limit["industry"] = latest_limit["industry"].fillna(latest_limit.get("industry_limit"))
    latest_limit["name"] = latest_limit["name"].fillna(latest_limit.get("name_limit"))

    recent_close = (
        daily_sel.loc[daily_sel["code"].isin(latest_limit["code"])]
        .sort_values(["code", "date"])
        .groupby("code")["close"]
        .apply(lambda s: list(pd.to_numeric(s, errors="coerce").dropna().tail(4).round(2)))
        .rename("recent_close_path")
        .reset_index()
    )
    latest_limit = latest_limit.merge(recent_close, on="code", how="left")
    latest_limit["recent_4d_return"] = latest_limit["recent_close_path"].apply(
        lambda items: ((items[-1] / items[0]) - 1.0) if isinstance(items, list) and len(items) >= 2 and items[0] else 0.0
    )
    latest_limit["recent_4d_path"] = latest_limit["recent_close_path"].apply(
        lambda items: "->".join(f"{float(v):.2f}" for v in items) if isinstance(items, list) and items else "-"
    )

    latest_limit = latest_limit.loc[
        latest_limit["close"].le(float(cfg.get("max_price", 20.0)))
        & latest_limit["turnover_ratio"].ge(float(cfg.get("min_turnover_ratio", 1.0)))
        & latest_limit["amount"].ge(float(cfg.get("min_amount", 80_000_000.0)))
        & latest_limit["fd_amount"].ge(float(cfg.get("min_fd_amount", 30_000_000.0)))
    ].copy()
    if not bool(cfg.get("allow_open_board", True)):
        latest_limit = latest_limit.loc[latest_limit["open_times"].le(0)]
    if latest_limit.empty:
        raise ValueError(f"No under-20 shortline candidates matched the filters on {selection_date.date()}.")

    latest_limit["fd_score"] = _normalize_rank(latest_limit["fd_amount"], ascending=True)
    latest_limit["streak_score"] = (latest_limit["current_streak"].clip(lower=0.0, upper=5.0) / 5.0).fillna(0.0)
    latest_limit["turnover_score"] = _turnover_quality(latest_limit["turnover_ratio"])
    latest_limit["float_mv_score"] = _float_mv_quality(latest_limit["float_mv"], cfg)
    latest_limit["recent_move_score"] = pd.Series(_recent_move_quality(latest_limit["recent_4d_return"], cfg), index=latest_limit.index, dtype=float)
    latest_limit["seal_score"] = _seal_quality(latest_limit["open_times"])

    latest_limit["shortline_score"] = (
        float(cfg.get("fd_amount_weight", 0.3)) * latest_limit["fd_score"]
        + float(cfg.get("streak_weight", 0.2)) * latest_limit["streak_score"]
        + float(cfg.get("turnover_weight", 0.15)) * latest_limit["turnover_score"]
        + float(cfg.get("float_mv_weight", 0.15)) * latest_limit["float_mv_score"]
        + float(cfg.get("recent_move_weight", 0.1)) * latest_limit["recent_move_score"]
        + float(cfg.get("seal_weight", 0.1)) * latest_limit["seal_score"]
    )

    a_threshold = float(cfg.get("a_grade_threshold", 0.72))
    b_threshold = float(cfg.get("b_grade_threshold", 0.58))
    latest_limit["grade"] = np.where(
        latest_limit["shortline_score"] >= a_threshold,
        "A",
        np.where(latest_limit["shortline_score"] >= b_threshold, "B", "C"),
    )
    latest_limit["style"] = np.where(
        (latest_limit["float_mv"].between(float(cfg.get("strong_float_mv_min", 5_000_000_000.0)), float(cfg.get("strong_float_mv_max", 40_000_000_000.0))))
        & (latest_limit["turnover_ratio"] <= 8.0)
        & (latest_limit["open_times"] <= 0),
        "稳健",
        np.where(
            (latest_limit["turnover_ratio"] >= 8.0) | (latest_limit["current_streak"] >= 2.0),
            "进攻",
            "观察",
        ),
    )
    latest_limit["risk_level"] = np.where(latest_limit["style"] == "稳健", "偏低", np.where(latest_limit["grade"] == "A", "中等", "偏高"))
    latest_limit["first_time_fmt"] = latest_limit.get("first_time", pd.Series("-", index=latest_limit.index)).apply(_format_hhmmss)

    latest_limit = latest_limit.sort_values(["shortline_score", "fd_amount", "current_streak"], ascending=[False, False, False]).reset_index(drop=True)
    latest_limit["rank"] = range(1, len(latest_limit) + 1)
    latest_limit = latest_limit.head(int(cfg.get("top_n", 12))).copy()

    bands = latest_limit.apply(lambda row: _suggest_band(row, cfg), axis=1)
    latest_limit["buy_range"] = bands.apply(lambda item: item[0])
    latest_limit["target_price"] = bands.apply(lambda item: item[1])
    latest_limit["stop_loss"] = bands.apply(lambda item: item[2])
    latest_limit["suggested_position"] = bands.apply(lambda item: item[3])
    latest_limit["operation_logic"] = latest_limit.apply(_build_logic, axis=1)

    prefix = f"{args.output_prefix}_{selection_date.strftime('%Y%m%d')}"
    csv_path = output_dir / f"{prefix}_cards.csv"
    json_path = output_dir / f"{prefix}_cards.json"
    md_path = output_dir / f"{prefix}_report.md"

    export_cols = [
        "rank",
        "code",
        "name",
        "close",
        "current_streak",
        "limit_up_count_20d",
        "open_board_count_20d",
        "turnover_ratio",
        "fd_amount",
        "float_mv",
        "recent_4d_path",
        "buy_range",
        "target_price",
        "stop_loss",
        "suggested_position",
        "grade",
        "style",
        "risk_level",
        "shortline_score",
        "industry",
        "operation_logic",
    ]
    latest_limit[export_cols].to_csv(csv_path, index=False)
    payload = {
        "summary": {
            "selection_date": selection_date.strftime("%Y-%m-%d"),
            "selected_count": int(len(latest_limit)),
            "candidate_count": int(len(history.loc[history["date"] == selection_date])),
            "data_latest_date": latest_daily_date.strftime("%Y-%m-%d"),
            "limit_latest_date": pd.Timestamp(limit_df["date"].max()).strftime("%Y-%m-%d"),
            "report_type": "shortline_opportunity_under20",
        },
        "cards": json.loads(latest_limit[export_cols].to_json(orient="records", force_ascii=False)),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(selection_date.strftime("%Y-%m-%d"), latest_limit[export_cols], payload["summary"], json_path), encoding="utf-8")

    print(
        json.dumps(
            {
                "selection_date": selection_date.strftime("%Y-%m-%d"),
                "selected_count": int(len(latest_limit)),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "report_path": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
