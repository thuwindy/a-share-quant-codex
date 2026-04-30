from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.backtest.cost import transaction_cost
from ashare_quant.execution.base import Order
from ashare_quant.execution.paper import PaperExecutionAdapter, PaperExecutionConfig


@dataclass
class CheckRow:
    check: str
    status: str
    value: float | int | str
    threshold: str
    note: str

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "note": self.note,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run T22: commission/slippage/sell-tax consistency checks.")
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_liquidity_stress_15bps.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "data_quality"))
    parser.add_argument("--output-prefix", default="t22_cost_model_consistency")
    parser.add_argument("--price", type=float, default=10.0)
    parser.add_argument("--buy-qty", type=float, default=1234.0)
    parser.add_argument("--sell-qty", type=float, default=567.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _md(rows: list[CheckRow]) -> str:
    lines = [
        "# T22 Cost Model Consistency",
        "",
        "| check | status | value | threshold | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {check} | {status} | {value} | {threshold} | {note} |".format(
                check=row.check,
                status=row.status,
                value=row.value,
                threshold=row.threshold,
                note=row.note.replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.backtest_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Backtest config not found: {cfg_path}")

    cfg = _load_json(cfg_path)
    commission = float(cfg.get("commission", 0.0003))
    slippage = float(cfg.get("slippage", 0.0005))
    sell_tax = float(cfg.get("sell_tax", 0.001))
    price = float(args.price)
    buy_qty = float(args.buy_qty)
    sell_qty = float(args.sell_qty)
    tolerance = float(args.tolerance)

    checks: list[CheckRow] = [
        CheckRow("commission", "INFO", commission, "-", "From backtest config."),
        CheckRow("slippage", "INFO", slippage, "-", "From backtest config."),
        CheckRow("sell_tax", "INFO", sell_tax, "-", "From backtest config."),
        CheckRow("use_liquidity_aware_cost", "INFO", bool(cfg.get("use_liquidity_aware_cost", False)), "-", "Reported only; T22 checks fixed commission/slippage/sell-tax semantics."),
    ]

    adapter = PaperExecutionAdapter(
        initial_cash=10_000_000.0,
        config=PaperExecutionConfig(commission=commission, slippage=slippage, sell_tax=sell_tax),
    )
    initial_cash = adapter.get_cash()

    buy_order = Order(trading_day="2024-01-02", code="000001.SZ", side="BUY", quantity=buy_qty, reference_price=price)
    buy_fill = adapter.submit_orders([buy_order])[0]
    buy_notional = buy_qty * price
    expected_buy_cost = buy_notional * (commission + slippage)
    buy_cash_delta = initial_cash - adapter.get_cash()
    expected_buy_cash_delta = buy_notional + expected_buy_cost

    checks.append(
        CheckRow(
            "paper_buy_fill_price_reference",
            _status(abs(float(buy_fill.fill_price) - price) <= tolerance),
            f"fill_price={buy_fill.fill_price:.12f}",
            f"== {price:.12f}",
            "Paper fill price should be reference price; slippage is charged via explicit cost.",
        )
    )
    checks.append(
        CheckRow(
            "paper_buy_cost_formula",
            _status(abs(float(buy_fill.cost) - expected_buy_cost) <= tolerance),
            f"observed={float(buy_fill.cost):.12f}, expected={expected_buy_cost:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "BUY cost must equal notional * (commission + slippage).",
        )
    )
    checks.append(
        CheckRow(
            "paper_buy_cash_delta",
            _status(abs(float(buy_cash_delta) - expected_buy_cash_delta) <= tolerance),
            f"observed={float(buy_cash_delta):.12f}, expected={expected_buy_cash_delta:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "BUY cash impact = notional + buy_cost.",
        )
    )

    cash_after_buy = adapter.get_cash()
    sell_order = Order(trading_day="2024-01-03", code="000001.SZ", side="SELL", quantity=sell_qty, reference_price=price)
    sell_fill = adapter.submit_orders([sell_order])[0]
    sell_notional = sell_qty * price
    expected_sell_cost = sell_notional * (commission + slippage + sell_tax)
    sell_cash_delta = adapter.get_cash() - cash_after_buy
    expected_sell_cash_delta = sell_notional - expected_sell_cost

    checks.append(
        CheckRow(
            "paper_sell_fill_price_reference",
            _status(abs(float(sell_fill.fill_price) - price) <= tolerance),
            f"fill_price={sell_fill.fill_price:.12f}",
            f"== {price:.12f}",
            "Paper fill price should be reference price; slippage is charged via explicit cost.",
        )
    )
    checks.append(
        CheckRow(
            "paper_sell_cost_formula",
            _status(abs(float(sell_fill.cost) - expected_sell_cost) <= tolerance),
            f"observed={float(sell_fill.cost):.12f}, expected={expected_sell_cost:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "SELL cost must equal notional * (commission + slippage + sell_tax).",
        )
    )
    checks.append(
        CheckRow(
            "paper_sell_cash_delta",
            _status(abs(float(sell_cash_delta) - expected_sell_cash_delta) <= tolerance),
            f"observed={float(sell_cash_delta):.12f}, expected={expected_sell_cash_delta:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "SELL cash impact = notional - sell_cost.",
        )
    )

    fixed_buy_cost = transaction_cost(
        buy_turnover=buy_notional,
        sell_turnover=0.0,
        commission=commission,
        slippage=slippage,
        sell_tax=sell_tax,
    )
    fixed_sell_cost = transaction_cost(
        buy_turnover=0.0,
        sell_turnover=sell_notional,
        commission=commission,
        slippage=slippage,
        sell_tax=sell_tax,
    )
    fixed_roundtrip = transaction_cost(
        buy_turnover=buy_notional,
        sell_turnover=sell_notional,
        commission=commission,
        slippage=slippage,
        sell_tax=sell_tax,
    )
    observed_roundtrip = float(buy_fill.cost) + float(sell_fill.cost)

    checks.append(
        CheckRow(
            "formula_vs_transaction_cost_buy",
            _status(abs(expected_buy_cost - fixed_buy_cost) <= tolerance),
            f"expected={expected_buy_cost:.12f}, transaction_cost={fixed_buy_cost:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "BUY explicit formula equals shared transaction_cost utility.",
        )
    )
    checks.append(
        CheckRow(
            "formula_vs_transaction_cost_sell",
            _status(abs(expected_sell_cost - fixed_sell_cost) <= tolerance),
            f"expected={expected_sell_cost:.12f}, transaction_cost={fixed_sell_cost:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "SELL explicit formula equals shared transaction_cost utility.",
        )
    )
    checks.append(
        CheckRow(
            "paper_roundtrip_vs_transaction_cost",
            _status(abs(observed_roundtrip - fixed_roundtrip) <= tolerance),
            f"observed={observed_roundtrip:.12f}, transaction_cost={fixed_roundtrip:.12f}",
            f"abs_diff<={tolerance:.1e}",
            "Paper adapter total cost equals shared transaction_cost for same notional turnovers.",
        )
    )

    status_rank = {"FAIL": 2, "PASS": 1, "INFO": 0}
    overall_rank = max(status_rank.get(row.status, 0) for row in checks)
    overall_status = "FAIL" if overall_rank >= 2 else "PASS"

    checks_csv_path = out_dir / f"{args.output_prefix}_checks.csv"
    checks_md_path = out_dir / f"{args.output_prefix}_checks.md"
    summary_json_path = out_dir / f"{args.output_prefix}_summary.json"

    pd.DataFrame([row.as_dict() for row in checks]).to_csv(checks_csv_path, index=False)
    checks_md_path.write_text(_md(checks), encoding="utf-8")
    summary = {
        "overall_status": overall_status,
        "backtest_config": str(cfg_path),
        "commission": commission,
        "slippage": slippage,
        "sell_tax": sell_tax,
        "price": price,
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "checks_csv_path": str(checks_csv_path),
        "checks_md_path": str(checks_md_path),
        "run_date": str(date.today()),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] T22 cost model consistency outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
