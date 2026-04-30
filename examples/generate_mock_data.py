from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def make_mock_daily_panel(n_stocks: int = 30, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    industries = ["Bank", "TMT", "Consumer", "Industrial", "Healthcare"]

    meta = pd.DataFrame(
        {
            "code": codes,
            "industry": rng.choice(industries, size=n_stocks),
            "base_mcap": np.exp(rng.normal(23.0, 1.0, size=n_stocks)),
            "alpha_strength": rng.normal(0.0, 1.0, size=n_stocks),
        }
    )
    meta.loc[meta.index[:2], "alpha_strength"] += 1.5

    rows = []
    industry_map = {k: i for i, k in enumerate(industries)}
    industry_shock = rng.normal(0.0, 0.003, size=(n_days, len(industries)))
    market_shock = rng.normal(0.0003, 0.01, size=n_days)

    for _, row in meta.iterrows():
        code = row["code"]
        industry = row["industry"]
        price = float(rng.uniform(8.0, 60.0))
        mcap = float(row["base_mcap"])
        alpha_strength = float(row["alpha_strength"])

        prev_rets: list[float] = []
        for t, date in enumerate(dates):
            ind_ret = industry_shock[t, industry_map[industry]]
            momentum_term = np.mean(prev_rets[-20:]) if len(prev_rets) >= 5 else 0.0
            quality_term = alpha_strength * 0.0015
            epsilon = rng.normal(0.0, 0.012)
            ret = market_shock[t] + ind_ret + 0.35 * momentum_term + quality_term + epsilon
            ret = float(np.clip(ret, -0.095, 0.095))
            open_px = price * (1.0 + rng.normal(0.0, 0.004))
            close_px = price * (1.0 + ret)
            high_px = max(open_px, close_px) * (1.0 + abs(rng.normal(0.0, 0.01)))
            low_px = min(open_px, close_px) * (1.0 - abs(rng.normal(0.0, 0.01)))
            volume = float(rng.integers(2_000_000, 20_000_000))
            amount = float(volume * (open_px + close_px) / 2.0)
            mcap = float(max(1e8, mcap * (1.0 + ret) * (1.0 + rng.normal(0.0, 0.002))))

            is_st = bool(code in {codes[-1], codes[-2]})
            is_suspended = bool(rng.random() < 0.01)
            can_buy = not is_suspended and abs(ret) < 0.094
            can_sell = not is_suspended and abs(ret) < 0.094

            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": round(open_px, 4),
                    "high": round(high_px, 4),
                    "low": round(low_px, 4),
                    "close": round(close_px, 4),
                    "volume": volume,
                    "amount": round(amount, 2),
                    "market_cap": round(mcap, 2),
                    "industry": industry,
                    "is_st": is_st,
                    "is_suspended": is_suspended,
                    "can_buy": can_buy,
                    "can_sell": can_sell,
                }
            )
            price = close_px
            prev_rets.append(ret)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)
    df = make_mock_daily_panel()
    out_path = out_dir / "mock_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
