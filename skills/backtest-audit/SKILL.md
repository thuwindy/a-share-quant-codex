---
name: backtest-audit
description: Use when asked to audit the backtest engine, portfolio construction, transaction costs, or A-share trading constraints. Do not use for unrelated factor ideation tasks.
---

1. Read `AGENTS.md` and `docs/architecture.md`.
2. Inspect for:
   - lookahead bias
   - label / signal misalignment
   - unrealistic execution assumptions
   - missing A-share constraints
   - missing transaction costs
3. Prefer high-confidence, minimal-risk fixes.
4. Add or update tests in `tests/`.
5. Run:
   - `python examples/run_mock_backtest.py`
   - `python -m unittest discover -s tests`
6. Write a short audit summary with risks that remain.
