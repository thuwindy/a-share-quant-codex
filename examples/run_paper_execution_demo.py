from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.execution.base import Order
from ashare_quant.execution.paper import PaperExecutionAdapter


def build_buy_orders_from_targets(
    target_weights: pd.DataFrame,
    price_snapshot: pd.Series,
    portfolio_value: float,
) -> list[Order]:
    orders: list[Order] = []
    for row in target_weights.itertuples(index=False):
        price = float(price_snapshot.loc[row.code])
        quantity = portfolio_value * float(row.target_weight) / price
        if quantity <= 0:
            continue
        orders.append(
            Order(
                trading_day=row.date,
                code=row.code,
                side="BUY",
                quantity=quantity,
                reference_price=price,
            )
        )
    return orders


if __name__ == "__main__":
    target_path = ROOT / "outputs" / "target_weights.csv"
    if not target_path.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "run_mock_backtest.py")], check=True)

    targets = pd.read_csv(target_path, parse_dates=["date"])
    if targets.empty:
        raise ValueError("No target weights found in outputs/target_weights.csv.")

    first_day = targets["date"].min()
    day_targets = targets.loc[targets["date"] == first_day].copy()

    panel = pd.read_csv(ROOT / "data" / "mock_daily.csv", parse_dates=["date"])
    snapshot = panel.loc[panel["date"] == first_day, ["code", "close"]].set_index("code")["close"]

    adapter = PaperExecutionAdapter(initial_cash=1_000_000.0)
    deployable_value = adapter.get_cash() / (1.0 + adapter.config.slippage + adapter.config.commission) * 0.999
    fills = adapter.submit_orders(build_buy_orders_from_targets(day_targets, snapshot, portfolio_value=deployable_value))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([fill.__dict__ for fill in fills]).to_csv(out_dir / "paper_fills.csv", index=False)
    (out_dir / "paper_positions.json").write_text(
        json.dumps(adapter.get_positions(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] generated {len(fills)} paper fills for {first_day.date()} with cash left {adapter.get_cash():.2f}")
