from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyAssessment:
    classification: str
    analyst_view: str
    trader_view: str
    key_risks: list[str]
    exposure_warning: str


def summarize_portfolio_exposure(picks: pd.DataFrame) -> dict[str, object]:
    if picks is None or picks.empty:
        return {
            "industry_weights": {},
            "size_distribution": {},
            "top_industry_weight": 0.0,
            "top_two_industry_weight": 0.0,
            "avg_market_cap": 0.0,
            "exposure_warning": "No active positions.",
        }

    weights = pd.to_numeric(picks.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    weight_total = float(weights.sum()) or 1.0
    normalized = weights / weight_total
    industry = picks.get("industry", pd.Series("Unknown", index=picks.index)).fillna("Unknown").astype(str)
    industry_weights = (
        pd.DataFrame({"industry": industry, "weight": normalized})
        .groupby("industry", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )
    market_cap = pd.to_numeric(picks.get("market_cap", pd.Series(pd.NA, index=picks.index)), errors="coerce")
    size_bucket = pd.Series("Unknown", index=picks.index, dtype="object")
    valid = market_cap.notna()
    if valid.sum() >= 3:
        rank_pct = market_cap.loc[valid].rank(method="first", pct=True)
        buckets = pd.cut(
            rank_pct,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["Small", "Mid", "Large"],
            include_lowest=True,
        )
        size_bucket.loc[valid] = buckets.astype(str)
    size_distribution = (
        pd.DataFrame({"size_bucket": size_bucket, "weight": normalized})
        .groupby("size_bucket", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )

    top_industry_weight = float(industry_weights["weight"].iloc[0]) if not industry_weights.empty else 0.0
    top_two_industry_weight = float(industry_weights["weight"].head(2).sum()) if not industry_weights.empty else 0.0
    exposure_warning = "Industry/style concentration looks acceptable."
    if top_industry_weight >= 0.35 or top_two_industry_weight >= 0.60:
        exposure_warning = "This portfolio behaves more like a style basket than a diversified alpha book."

    return {
        "industry_weights": {
            str(getattr(row, "industry")): float(getattr(row, "weight"))
            for row in industry_weights.itertuples(index=False)
        },
        "size_distribution": {
            str(getattr(row, "size_bucket")): float(getattr(row, "weight"))
            for row in size_distribution.itertuples(index=False)
        },
        "top_industry_weight": top_industry_weight,
        "top_two_industry_weight": top_two_industry_weight,
        "avg_market_cap": float(market_cap.mean()) if market_cap.notna().any() else 0.0,
        "exposure_warning": exposure_warning,
    }


def classify_strategy(metrics: dict, exposure_summary: dict[str, object] | None = None) -> StrategyAssessment:
    exposure_summary = exposure_summary or {}
    gross_return = float(metrics.get("gross_annual_return", 0.0))
    net_return = float(metrics.get("annual_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    avg_turnover = float(metrics.get("avg_turnover", 0.0))
    cost_drag = float(metrics.get("after_cost_return_drag", 0.0))

    key_risks: list[str] = []
    if gross_return > 0.0 and net_return <= 0.0:
        key_risks.append("Signal is positive before cost but not strong enough to survive trading friction.")
    if cost_drag >= 0.05:
        key_risks.append("Cost drag is too high relative to gross alpha.")
    if avg_turnover >= 0.25:
        key_risks.append("Turnover is still high for a daily-K production prototype.")
    if max_drawdown <= -0.35:
        key_risks.append("Drawdown is too deep for a production-oriented long-only book.")
    if exposure_summary.get("top_industry_weight", 0.0) >= 0.35:
        key_risks.append("Industry concentration is elevated.")

    if net_return > 0.0 and sharpe > 0.3 and cost_drag < 0.05 and avg_turnover < 0.25 and max_drawdown > -0.35:
        classification = "tradable prototype"
        analyst_view = "This system can support a focused buy-list and now has a credible deployable portfolio layer."
        trader_view = "Alpha appears thick enough to survive costs under the current assumptions."
    else:
        classification = "research candidate engine / monitor only"
        analyst_view = "The system is useful for narrowing the research universe and surfacing explainable style baskets."
        trader_view = "The signal is still too thin after costs; execution and turnover dominate the outcome."

    exposure_warning = str(exposure_summary.get("exposure_warning", ""))
    return StrategyAssessment(
        classification=classification,
        analyst_view=analyst_view,
        trader_view=trader_view,
        key_risks=key_risks,
        exposure_warning=exposure_warning,
    )


def assessment_payload(metrics: dict, picks: pd.DataFrame | None = None) -> dict[str, object]:
    exposure = summarize_portfolio_exposure(picks if picks is not None else pd.DataFrame())
    assessment = classify_strategy(metrics=metrics, exposure_summary=exposure)
    payload = asdict(assessment)
    payload["exposure_summary"] = exposure
    return payload
