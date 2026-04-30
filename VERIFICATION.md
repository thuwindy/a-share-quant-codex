# 验证记录

我在当前环境里实际跑过以下命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python examples/generate_mock_data.py
.venv/bin/python examples/run_mock_backtest.py
.venv/bin/python examples/run_real_daily_backtest.py --data-path data/mock_daily.csv --adjust none
.venv/bin/python examples/run_paper_execution_demo.py
.venv/bin/python scripts/bootstrap_cn_history_directory.py --source-dir /path/to/your/cn_daily_csv_dir --output-path data/a_share_daily_smoke.csv --max-files 40 --start-date 2021-01-01 --end-date 2026-03-20
.venv/bin/python examples/run_real_daily_backtest.py --data-path data/a_share_daily_smoke.csv --adjust qfq
.venv/bin/python scripts/bootstrap_cn_history_directory.py --source-dir /path/to/your/cn_daily_csv_dir --output-path data/a_share_daily.csv --end-date 2026-03-20
.venv/bin/python scripts/run_real_research_workflow.py --input-path data/a_share_daily.csv --slice-path data/research_slice_400.csv --start-date 2019-01-01 --end-date 2026-03-20 --max-codes 400 --adjust qfq --output-prefix workflow_real
.venv/bin/python scripts/run_real_research_workflow.py --input-path data/a_share_daily.csv --slice-path data/research_slice_200.csv --start-date 2019-01-01 --end-date 2026-03-20 --max-codes 200 --adjust qfq --output-prefix workflow_real
.venv/bin/python -m unittest discover -s tests
```

## 结果

- mock 数据成功生成
- 回测成功运行
- 真实日 K 入口在 `mock_daily.csv` 上成功跑通
- 真实目录型历史数据已成功标准化为 `data/a_share_daily.csv`
- 真实目录型历史数据 smoke test 已成功跑通
- 真实研究 workflow 已在 200 / 400 只高流动性股票样本上跑通
- paper execution demo 成功生成 mock fills
- 输出已写入 `outputs/`
- 单元测试通过

## 当前 demo 指标（来自 mock 数据）

- annual_return: 0.24616581818369698
- annual_volatility: 0.18179040709477623
- sharpe: 1.354118856532175
- max_drawdown: -0.06288571139177779
- hit_rate: 0.4791666666666667
- avg_turnover: 0.19791666666666666

说明：这些数值只用于验证系统链路打通，不代表真实 A 股实盘表现。
