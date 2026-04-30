from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .cost import liquidity_aware_transaction_cost, transaction_cost
from .metrics import summarize_returns


@dataclass
class BacktestConfig:
    commission: float = 0.0003
    slippage: float = 0.0005
    sell_tax: float = 0.001
    annual_trading_days: int = 252
    signal_time: str = "close"
    execution_price: str = "close"
    execution_lag: int = 1
    holding_window: int = 5
    sleeve_count: int = 1
    use_liquidity_aware_cost: bool = False
    portfolio_notional: float = 10_000_000.0
    slippage_adv_coef: float = 0.1
    min_adv20: float = 1_000_000.0
    max_participation: float = 0.25
    max_slippage: float = 0.02
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    trailing_stop_pct: float = 0.0


class DailyBacktester:
    """A lightweight daily research backtester with explicit signal/execution alignment."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    @staticmethod
    def _execution_tradeability_maps(sl: pd.DataFrame) -> tuple[dict[str, bool], dict[str, bool]]:
        amount = pd.to_numeric(sl["amount"], errors="coerce").fillna(0.0)
        can_buy = sl["can_buy"].fillna(True).astype(bool) & amount.gt(0.0)
        can_sell = sl["can_sell"].fillna(True).astype(bool) & amount.gt(0.0)
        return dict(zip(sl["code"], can_buy)), dict(zip(sl["code"], can_sell))

    @staticmethod
    def _constrained_target_weights(
        old_weights: dict[str, float],
        requested_weights: dict[str, float],
        can_buy_map: dict[str, bool],
        can_sell_map: dict[str, bool],
    ) -> tuple[dict[str, float], int, int]:
        current = {code: float(weight) for code, weight in old_weights.items() if float(weight) > 1e-12}
        blocked_buys = 0
        blocked_sells = 0

        for code, old_weight in list(current.items()):
            requested_weight = float(requested_weights.get(code, 0.0))
            if requested_weight >= old_weight - 1e-12:
                continue
            if can_sell_map.get(code, False):
                if requested_weight <= 1e-12:
                    current.pop(code, None)
                else:
                    current[code] = requested_weight
            else:
                blocked_sells += 1

        available_cash = max(1.0 - sum(current.values()), 0.0)
        buy_candidates: list[tuple[str, float, float]] = []
        for code, requested_weight in requested_weights.items():
            old_weight = float(old_weights.get(code, 0.0))
            requested_weight = float(requested_weight)
            if requested_weight > old_weight + 1e-12:
                buy_candidates.append((code, old_weight, requested_weight))
        buy_candidates.sort(key=lambda item: item[2] - item[1], reverse=True)

        for code, old_weight, requested_weight in buy_candidates:
            if not can_buy_map.get(code, False):
                blocked_buys += 1
                continue
            desired_increase = requested_weight - old_weight
            allowed_increase = min(desired_increase, available_cash)
            if allowed_increase <= 1e-12:
                blocked_buys += 1
                continue
            current[code] = old_weight + allowed_increase
            available_cash -= allowed_increase

        return current, blocked_buys, blocked_sells

    @staticmethod
    def _weights_for_execution_date(targets: pd.DataFrame) -> Dict[pd.Timestamp, dict[int, dict[str, float]]]:
        mapping: Dict[pd.Timestamp, dict[int, dict[str, float]]] = {}
        if targets is None or targets.empty:
            return mapping
        if "execution_date" not in targets.columns:
            if "date" not in targets.columns:
                return mapping
            grouped = targets.groupby("date")
            for date, sl in grouped:
                mapping[pd.Timestamp(date)] = {0: dict(zip(sl["code"], sl["target_weight"]))}
            return mapping

        for execution_date, sl in targets.groupby("execution_date"):
            sleeve_map: dict[int, dict[str, float]] = {}
            for sleeve, sleeve_df in sl.groupby("sleeve" if "sleeve" in sl.columns else lambda _: 0):
                sleeve_map[int(sleeve)] = dict(zip(sleeve_df["code"], sleeve_df["target_weight"]))
            mapping[pd.Timestamp(execution_date)] = sleeve_map
        return mapping

    def run(self, panel: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        panel = panel.sort_values(["date", "code"]).copy()
        close_col = "research_close" if "research_close" in panel.columns else "close"
        if "research_open" in panel.columns:
            open_col = "research_open"
        elif "open" in panel.columns:
            open_col = "open"
        else:
            open_col = close_col
        if "research_vwap" in panel.columns:
            vwap_col = "research_vwap"
        else:
            vwap_col = None
        prev_close = panel.groupby("code")[close_col].shift(1)
        open_px = pd.to_numeric(panel[open_col], errors="coerce")
        close_px = pd.to_numeric(panel[close_col], errors="coerce")
        if vwap_col is not None:
            vwap_px = pd.to_numeric(panel[vwap_col], errors="coerce")
        else:
            vwap_px = pd.concat([open_px, close_px], axis=1).mean(axis=1)
        prev_close = pd.to_numeric(prev_close, errors="coerce")
        panel["ret_close_close"] = close_px.div(prev_close.replace(0.0, np.nan)) - 1.0
        panel["ret_open_close"] = close_px.div(open_px.replace(0.0, np.nan)) - 1.0
        panel["ret_gap_open"] = open_px.div(prev_close.replace(0.0, np.nan)) - 1.0
        panel["ret_vwap_close"] = close_px.div(vwap_px.replace(0.0, np.nan)) - 1.0
        panel["ret_gap_vwap"] = vwap_px.div(prev_close.replace(0.0, np.nan)) - 1.0
        panel["ret_1"] = panel["ret_close_close"].fillna(0.0)
        panel["adv20"] = (
            panel.groupby("code")["amount"]
            .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=5).mean())
            .fillna(pd.to_numeric(panel["amount"], errors="coerce"))
        )

        execution_map = self._weights_for_execution_date(targets)
        dates = sorted(panel["date"].drop_duplicates())
        sleeve_count = max(int(self.config.sleeve_count), 1)
        current_sleeve_weights: dict[int, dict[str, float]] = {sleeve: {} for sleeve in range(sleeve_count)}
        current_sleeve_state: dict[int, dict[str, dict[str, float]]] = {sleeve: {} for sleeve in range(sleeve_count)}
        prior_close_map: dict[str, float] = {}

        def _forced_exit_reason(position_state: dict[str, float], prior_close: float | None) -> str:
            if prior_close is None or not np.isfinite(prior_close):
                return ""
            entry_close = float(position_state.get("entry_close", np.nan))
            peak_close = float(position_state.get("peak_close", np.nan))
            if not np.isfinite(entry_close) or entry_close <= 0:
                return ""
            return_since_entry = prior_close / entry_close - 1.0
            if float(self.config.stop_loss_pct) > 0 and return_since_entry <= -float(self.config.stop_loss_pct):
                return "stop_loss"
            if float(self.config.take_profit_pct) > 0 and return_since_entry >= float(self.config.take_profit_pct):
                return "take_profit"
            if (
                float(self.config.trailing_stop_pct) > 0
                and np.isfinite(peak_close)
                and peak_close > 0
                and prior_close / peak_close - 1.0 <= -float(self.config.trailing_stop_pct)
            ):
                return "trailing_stop"
            return ""

        records = []
        for date, sl in panel.groupby("date"):
            total_weights: dict[str, float] = {}
            for sleeve_weights in current_sleeve_weights.values():
                for code, weight in sleeve_weights.items():
                    total_weights[code] = total_weights.get(code, 0.0) + weight / sleeve_count

            cost = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            blocked_buys = 0
            blocked_sells = 0
            forced_exit_events: list[dict[str, object]] = []
            effective_targets = {int(k): dict(v) for k, v in execution_map.get(pd.Timestamp(date), {}).items()}
            for sleeve in range(sleeve_count):
                sleeve_weights = current_sleeve_weights.get(sleeve, {})
                sleeve_state = current_sleeve_state.get(sleeve, {})
                for code, old_weight in sleeve_weights.items():
                    if float(old_weight) <= 1e-12:
                        continue
                    reason = _forced_exit_reason(sleeve_state.get(code, {}), prior_close_map.get(code))
                    if not reason:
                        continue
                    base = effective_targets.get(sleeve, sleeve_weights.copy())
                    base[str(code)] = 0.0
                    effective_targets[sleeve] = base
                    forced_exit_events.append(
                        {
                            "sleeve": sleeve,
                            "code": str(code),
                            "reason": reason,
                        }
                    )
            executed_today = bool(effective_targets)

            if self.config.execution_price == "open":
                gap_map = dict(zip(sl["code"], sl["ret_gap_open"].fillna(0.0)))
                overnight_ret = sum(total_weights.get(code, 0.0) * gap_map.get(code, 0.0) for code in total_weights)
            elif self.config.execution_price == "vwap":
                gap_map = dict(zip(sl["code"], sl["ret_gap_vwap"].fillna(0.0)))
                overnight_ret = sum(total_weights.get(code, 0.0) * gap_map.get(code, 0.0) for code in total_weights)
            else:
                overnight_ret = 0.0

            if executed_today:
                adv20_map = dict(zip(sl["code"], sl["adv20"]))
                can_buy_map, can_sell_map = self._execution_tradeability_maps(sl)
                close_map_today = dict(zip(sl["code"], close_px.loc[sl.index].fillna(np.nan)))
                for sleeve, new_weights in effective_targets.items():
                    old_weights = current_sleeve_weights.get(sleeve, {})
                    constrained_weights, sleeve_blocked_buys, sleeve_blocked_sells = self._constrained_target_weights(
                        old_weights=old_weights,
                        requested_weights=new_weights,
                        can_buy_map=can_buy_map,
                        can_sell_map=can_sell_map,
                    )
                    blocked_buys += sleeve_blocked_buys
                    blocked_sells += sleeve_blocked_sells
                    names = set(old_weights) | set(constrained_weights)
                    deltas = {name: float(constrained_weights.get(name, 0.0) - old_weights.get(name, 0.0)) for name in names}
                    if self.config.use_liquidity_aware_cost:
                        sleeve_cost, sleeve_buy, sleeve_sell = liquidity_aware_transaction_cost(
                            deltas,
                            adv20_map=adv20_map,
                            commission=self.config.commission,
                            base_slippage=self.config.slippage,
                            sell_tax=self.config.sell_tax,
                            portfolio_notional=self.config.portfolio_notional / sleeve_count,
                            slippage_adv_coef=self.config.slippage_adv_coef,
                            min_adv20=self.config.min_adv20,
                            max_participation=self.config.max_participation,
                            max_slippage=self.config.max_slippage,
                        )
                    else:
                        sleeve_buy = sum(max(delta, 0.0) for delta in deltas.values())
                        sleeve_sell = sum(max(-delta, 0.0) for delta in deltas.values())
                        sleeve_cost = transaction_cost(
                            sleeve_buy,
                            sleeve_sell,
                            commission=self.config.commission,
                            slippage=self.config.slippage,
                            sell_tax=self.config.sell_tax,
                        )
                    buy_turnover += sleeve_buy / sleeve_count
                    sell_turnover += sleeve_sell / sleeve_count
                    cost += sleeve_cost / sleeve_count
                    current_sleeve_weights[sleeve] = constrained_weights
                    old_state = current_sleeve_state.get(sleeve, {})
                    refreshed_state: dict[str, dict[str, float]] = {}
                    for code, weight in constrained_weights.items():
                        if float(weight) <= 1e-12:
                            continue
                        close_ref = float(close_map_today.get(code, np.nan))
                        prev_state = old_state.get(code, {})
                        if code in old_state:
                            peak_close = float(prev_state.get("peak_close", close_ref))
                            if np.isfinite(close_ref):
                                peak_close = max(peak_close, close_ref)
                            refreshed_state[code] = {
                                "entry_close": float(prev_state.get("entry_close", close_ref)),
                                "peak_close": peak_close,
                            }
                        else:
                            refreshed_state[code] = {
                                "entry_close": close_ref,
                                "peak_close": close_ref,
                            }
                    current_sleeve_state[sleeve] = refreshed_state

            total_post_weights: dict[str, float] = {}
            for sleeve_weights in current_sleeve_weights.values():
                for code, weight in sleeve_weights.items():
                    total_post_weights[code] = total_post_weights.get(code, 0.0) + weight / sleeve_count

            if self.config.execution_price == "open":
                intraday_map = dict(zip(sl["code"], sl["ret_open_close"].fillna(0.0)))
                intraday_ret = sum(total_post_weights.get(code, 0.0) * intraday_map.get(code, 0.0) for code in total_post_weights)
                gross_ret = (1.0 + overnight_ret) * (1.0 + intraday_ret) - 1.0
                net_ret = (1.0 + overnight_ret) * max(1.0 - cost, 0.0) * (1.0 + intraday_ret) - 1.0
            elif self.config.execution_price == "vwap":
                intraday_map = dict(zip(sl["code"], sl["ret_vwap_close"].fillna(0.0)))
                intraday_ret = sum(total_post_weights.get(code, 0.0) * intraday_map.get(code, 0.0) for code in total_post_weights)
                gross_ret = (1.0 + overnight_ret) * (1.0 + intraday_ret) - 1.0
                net_ret = (1.0 + overnight_ret) * max(1.0 - cost, 0.0) * (1.0 + intraday_ret) - 1.0
            else:
                ret_map = dict(zip(sl["code"], sl["ret_1"]))
                gross_ret = sum(total_weights.get(code, 0.0) * ret_map.get(code, 0.0) for code in total_weights)
                net_ret = gross_ret - cost

            concentration = max(total_post_weights.values()) if total_post_weights else 0.0
            records.append(
                {
                    "date": date,
                    "gross_return": gross_ret,
                    "cost": cost,
                    "net_return": net_ret,
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "turnover": buy_turnover + sell_turnover,
                    "n_holdings": len(total_post_weights),
                    "max_position_weight": concentration,
                    "blocked_buys": blocked_buys,
                    "blocked_sells": blocked_sells,
                    "forced_exit_count": len(forced_exit_events),
                }
            )
            prior_close_map = dict(zip(sl["code"], close_px.loc[sl.index].fillna(np.nan)))

        result = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        result["gross_equity"] = (1.0 + result["gross_return"]).cumprod()
        result["equity"] = (1.0 + result["net_return"]).cumprod()
        metrics = summarize_returns(
            result["net_return"],
            turnover=result["turnover"],
            gross_returns=result["gross_return"],
            annual_days=self.config.annual_trading_days,
            concentration=result["max_position_weight"],
            cost=result["cost"],
        )
        metrics["signal_time"] = self.config.signal_time
        metrics["execution_time"] = f"t+{self.config.execution_lag} {self.config.execution_price}"
        metrics["holding_window"] = self.config.holding_window
        metrics["sleeve_count"] = sleeve_count
        metrics["avg_blocked_buys"] = float(result["blocked_buys"].mean()) if len(result) else 0.0
        metrics["avg_blocked_sells"] = float(result["blocked_sells"].mean()) if len(result) else 0.0
        metrics["avg_forced_exit_count"] = float(result["forced_exit_count"].mean()) if len(result) else 0.0
        metrics["stop_loss_pct"] = float(self.config.stop_loss_pct)
        metrics["take_profit_pct"] = float(self.config.take_profit_pct)
        metrics["trailing_stop_pct"] = float(self.config.trailing_stop_pct)
        return result, metrics
