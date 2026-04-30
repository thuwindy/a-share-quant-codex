---
name: a-share-factor-research
description: Use when asked to add, modify, or evaluate a stock-selection factor for the A-share quant research system. Do not use for pure infrastructure tasks or UI-only changes.
---

1. Read `AGENTS.md`, `docs/study_guide.md`, and `docs/architecture.md`.
2. Clarify the factor category: technical, fundamental, analyst, microstructure, or alternative data.
3. Implement the factor in `src/ashare_quant/factors/`.
4. Add or update neutralization if the factor should be size / industry neutral.
5. Ensure the label alignment is leakage-safe.
6. Run `python examples/run_mock_backtest.py`.
7. Summarize:
   - factor intuition
   - factor formula
   - expected horizon
   - whether IC / Sharpe improved on the mock pipeline
8. Add tests when practical.
