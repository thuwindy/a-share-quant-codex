from __future__ import annotations

import numpy as np
import pandas as pd


def _annualized_return_from_equity(equity: pd.Series, annual_days: int) -> float:
    if equity.empty:
        return 0.0
    final_equity = float(pd.to_numeric(equity.iloc[-1], errors="coerce"))
    if not np.isfinite(final_equity):
        return 0.0
    if final_equity <= 0.0:
        return -1.0
    return float(final_equity ** (annual_days / max(len(equity), 1)) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = pd.Series(-1.0, index=equity.index, dtype=float)
    valid = peak.gt(0) & equity.notna()
    drawdown.loc[valid] = equity.loc[valid] / peak.loc[valid] - 1.0
    return max(float(drawdown.min()), -1.0)


def payoff_ratio(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return 0.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if wins.empty or losses.empty:
        return 0.0
    avg_win = float(wins.mean())
    avg_loss = abs(float(losses.mean()))
    if avg_loss <= 0:
        return 0.0
    return avg_win / avg_loss


def summarize_returns(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    annual_days: int = 252,
    gross_returns: pd.Series | None = None,
    concentration: pd.Series | None = None,
    cost: pd.Series | None = None,
) -> dict:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    ann_return = _annualized_return_from_equity(equity, annual_days=annual_days)
    ann_vol = float(returns.std(ddof=0) * np.sqrt(annual_days))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(equity)
    hit_rate = float((returns > 0).mean())
    avg_turnover = float(turnover.mean()) if turnover is not None and len(turnover) else 0.0
    payload = {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "hit_rate": hit_rate,
        "payoff_ratio": payoff_ratio(returns),
        "avg_turnover": avg_turnover,
    }
    if gross_returns is not None and len(gross_returns):
        gross_returns = gross_returns.fillna(0.0)
        gross_equity = (1.0 + gross_returns).cumprod()
        payload["gross_annual_return"] = _annualized_return_from_equity(gross_equity, annual_days=annual_days)
        payload["gross_sharpe"] = (
            payload["gross_annual_return"] / float(gross_returns.std(ddof=0) * np.sqrt(annual_days))
            if float(gross_returns.std(ddof=0)) > 0
            else 0.0
        )
        payload["after_cost_return_drag"] = payload["gross_annual_return"] - payload["annual_return"]
    if concentration is not None and len(concentration):
        payload["avg_max_position_weight"] = float(concentration.mean())
    if cost is not None and len(cost):
        payload["avg_cost"] = float(cost.mean())
    return payload
