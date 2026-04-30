from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from ashare_quant.data.research_slice import ResearchSliceProfile


def extract_display_weights(weight_payload: dict) -> dict[str, float]:
    if not weight_payload:
        return {}
    if all(isinstance(value, (int, float)) for value in weight_payload.values()):
        return {str(key): float(value) for key, value in weight_payload.items()}

    horizon_models = weight_payload.get("horizon_models", {})
    if horizon_models:
        if len(horizon_models) == 1:
            only = next(iter(horizon_models.values()))
            weights = only.get("weights", {})
            if isinstance(weights, dict):
                return {str(key): float(value) for key, value in weights.items() if isinstance(value, (int, float))}
        flat: dict[str, float] = {}
        final_horizon_weights = weight_payload.get("final_horizon_weights", {})
        for horizon, model in horizon_models.items():
            horizon_scale = float(final_horizon_weights.get(str(horizon), 1.0 / max(len(horizon_models), 1)))
            for factor, value in (model.get("weights", {}) or {}).items():
                if isinstance(value, (int, float)):
                    flat[str(factor)] = flat.get(str(factor), 0.0) + float(value) * horizon_scale
        if flat:
            return flat

    final_horizon_weights = weight_payload.get("final_horizon_weights", {})
    if isinstance(final_horizon_weights, dict):
        return {
            f"horizon_{key}": float(value)
            for key, value in final_horizon_weights.items()
            if isinstance(value, (int, float))
        }
    return {}


def sorted_factor_weights(weights: dict[str, float]) -> list[tuple[str, float]]:
    display_weights = extract_display_weights(weights)
    return sorted(display_weights.items(), key=lambda item: abs(item[1]), reverse=True)


def build_research_summary_markdown(
    profile: ResearchSliceProfile,
    metrics: dict,
    factor_weights: dict[str, float],
    slice_path: str | Path,
    adjust_mode: str,
) -> str:
    ranked_weights = sorted_factor_weights(factor_weights)
    top_lines = "\n".join(
        f"- `{name}`: {value:.4f}" for name, value in ranked_weights[: min(5, len(ranked_weights))]
    )
    markdown = dedent(
        f"""
        # Real Daily Research Summary

        ## Dataset

        - slice path: `{slice_path}`
        - date range: `{profile.start_date}` to `{profile.end_date}`
        - rows: `{profile.rows}`
        - unique codes: `{profile.unique_codes}`
        - avg rows per code: `{profile.avg_rows_per_code:.1f}`
        - suspended ratio: `{profile.suspended_ratio:.2%}`
        - ST ratio: `{profile.st_ratio:.2%}`
        - unknown industry ratio: `{profile.unknown_industry_ratio:.2%}`
        - price adjustment: `{adjust_mode}`

        ## Backtest Metrics

        - gross_annual_return: `{metrics.get("gross_annual_return", 0.0):.4f}`
        - annual_return: `{metrics.get("annual_return", 0.0):.4f}`
        - annual_volatility: `{metrics.get("annual_volatility", 0.0):.4f}`
        - sharpe: `{metrics.get("sharpe", 0.0):.4f}`
        - max_drawdown: `{metrics.get("max_drawdown", 0.0):.4f}`
        - hit_rate: `{metrics.get("hit_rate", 0.0):.4f}`
        - avg_turnover: `{metrics.get("avg_turnover", 0.0):.4f}`
        - after_cost_return_drag: `{metrics.get("after_cost_return_drag", 0.0):.4f}`
        - strategy_classification: `{metrics.get("strategy_assessment", {}).get("classification", "n/a")}`

        ## Factor Weights

        {top_lines or "- no factor weights"}

        ## LLM Notes

        - This summary is generated from executable outputs, not from a language-model estimate of returns.
        - The report now distinguishes research-helper value from tradable-strategy value; do not treat the observation pool as a direct buy list.
        - The current real-data workflow still lacks richer point-in-time fundamentals, execution adapters, and production monitoring.

        ## Prompt Seed

        ```text
        Read this real-data A-share research summary and critique it like a quant PM.
        Focus on:
        1. whether the universe construction is reasonable,
        2. whether the factor exposures look intuitive,
        3. whether the Sharpe / drawdown / turnover tradeoff looks credible,
        4. what the biggest data-quality and backtest-assumption risks are,
        5. which three next experiments are highest impact.
        Do not restate the metrics; interpret them.
        ```
        """
    ).strip()
    return markdown + "\n"


def write_research_summary(
    output_path: str | Path,
    profile: ResearchSliceProfile,
    metrics: dict,
    factor_weights: dict[str, float],
    slice_path: str | Path,
    adjust_mode: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_research_summary_markdown(
            profile=profile,
            metrics=metrics,
            factor_weights=factor_weights,
            slice_path=slice_path,
            adjust_mode=adjust_mode,
        ),
        encoding="utf-8",
    )
    return output_path


def write_research_summary_json(
    output_path: str | Path,
    profile: ResearchSliceProfile,
    metrics: dict,
    factor_weights: dict[str, float],
    slice_path: str | Path,
    adjust_mode: str,
) -> Path:
    payload = {
        "slice_path": str(slice_path),
        "adjust_mode": adjust_mode,
        "profile": profile.__dict__,
        "metrics": metrics,
        "factor_weights": extract_display_weights(factor_weights),
        "raw_factor_payload": factor_weights,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
