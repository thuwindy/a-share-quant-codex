from __future__ import annotations

import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from ashare_quant.service.trade_api import create_app


class TradeApiTest(unittest.TestCase):
    def test_paper_order_flow(self) -> None:
        client = TestClient(create_app(initial_cash=100_000.0))
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        order_resp = client.post(
            "/paper/orders",
            json=[
                {
                    "trading_day": "2026-04-03",
                    "code": "000001.SZ",
                    "side": "BUY",
                    "quantity": 100.0,
                    "reference_price": 10.0,
                }
            ],
        )
        self.assertEqual(order_resp.status_code, 200)
        state_resp = client.get("/paper/state")
        self.assertEqual(state_resp.status_code, 200)
        payload = state_resp.json()
        self.assertIn("000001.SZ", payload["positions"])
        caps_resp = client.get("/broker/capabilities")
        self.assertEqual(caps_resp.status_code, 200)
        self.assertFalse(caps_resp.json()["live"]["enabled"])


if __name__ == "__main__":
    unittest.main()
