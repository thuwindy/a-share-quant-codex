from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.pipeline import run_research_pipeline
from ashare_quant.analysis.research_report import extract_display_weights


if __name__ == "__main__":
    data_path = ROOT / "data" / "mock_daily.csv"
    if not data_path.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "examples" / "generate_mock_data.py")], check=True)

    result, metrics, score_details, targets = run_research_pipeline(
        data_path=data_path,
        research_config_path=ROOT / "configs" / "research.json",
        backtest_config_path=ROOT / "configs" / "backtest.json",
    )
    weights = extract_display_weights(score_details)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    result.to_csv(out_dir / "equity_curve.csv", index=False)
    (out_dir / "mock_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "factor_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "score_details.json").write_text(json.dumps(score_details, ensure_ascii=False, indent=2), encoding="utf-8")
    targets.to_csv(out_dir / "target_weights.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(result["date"], result["equity"])
    plt.title("Mock Equity Curve")
    plt.xlabel("date")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("[OK] outputs saved to", out_dir)
