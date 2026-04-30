from __future__ import annotations

from typing import Any

from ashare_quant.execution.base import Order
from ashare_quant.execution.live import LiveExecutionConfig
from ashare_quant.execution.paper import PaperExecutionAdapter, PaperExecutionConfig


def _require_fastapi():
    from fastapi import Body, FastAPI, HTTPException

    return FastAPI, HTTPException, Body


def create_app(
    *,
    initial_cash: float = 1_000_000.0,
    commission: float = 0.0003,
    slippage: float = 0.0005,
    sell_tax: float = 0.001,
):
    FastAPI, HTTPException, Body = _require_fastapi()

    app = FastAPI(title="A-share Quant Trade API", version="0.1.0")
    app.state.paper_adapter = PaperExecutionAdapter(
        initial_cash=initial_cash,
        config=PaperExecutionConfig(
            commission=commission,
            slippage=slippage,
            sell_tax=sell_tax,
        ),
    )
    app.state.live_config = LiveExecutionConfig()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "mode": "paper"}

    @app.get("/paper/state")
    def paper_state() -> dict[str, Any]:
        adapter = app.state.paper_adapter
        return {
            "cash": adapter.get_cash(),
            "positions": adapter.get_positions(),
            "fills": [fill.__dict__ for fill in adapter.get_fills()],
        }

    @app.post("/paper/orders")
    def submit_paper_orders(requests: list[dict[str, Any]] = Body(...)) -> dict[str, Any]:
        adapter = app.state.paper_adapter
        try:
            fills = adapter.submit_orders(
                [
                    Order(
                        trading_day=str(req["trading_day"]),
                        code=str(req["code"]),
                        side=str(req["side"]).upper(),
                        quantity=float(req["quantity"]),
                        reference_price=float(req["reference_price"]),
                    )
                    for req in requests
                ]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "submitted": len(requests),
            "fills": [fill.__dict__ for fill in fills],
            "cash": adapter.get_cash(),
            "positions": adapter.get_positions(),
        }

    @app.get("/paper/fills")
    def paper_fills(trading_day: str | None = None) -> dict[str, Any]:
        adapter = app.state.paper_adapter
        fills = adapter.get_fills(trading_day=trading_day)
        return {"count": len(fills), "fills": [fill.__dict__ for fill in fills]}

    @app.get("/broker/capabilities")
    def broker_capabilities() -> dict[str, Any]:
        return {
            "paper": {
                "enabled": True,
                "submit_orders": True,
                "query_positions": True,
                "query_fills": True,
            },
            "live": {
                "enabled": False,
                "broker_name": app.state.live_config.broker_name,
                "note": "Concrete QMT/掘金/券商适配器尚未接入，此处保留统一接口位。",
            },
        }

    return app
