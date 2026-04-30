# A股量化系统（Codex 协作版）

这是一个“课程项目级 / 研究原型级”的 A 股量化系统骨架，目标不是立刻实盘，而是把**学习资料、研究流程、A 股约束、Codex 协作方式**放进同一个可演进仓库里。

## 你会得到什么

- 一份学习梳理：`docs/study_guide.md`
- 一份系统设计：`docs/architecture.md`
- 一份 Codex 使用说明：`docs/codex_workflow.md`
- 一套可运行原型：
  - 模拟 A 股日频数据生成
  - 因子计算
  - 因子中性化
  - 标签生成
  - 因子加权打分
  - 长-only 组合构建
  - 简单回测与绩效评估
- 一套 Codex 项目配置：
  - `AGENTS.md`
  - `skills/` 下的 3 个项目技能

## 仓库结构

```text
.
├── AGENTS.md
├── configs/
├── data/
├── docs/
├── examples/
├── outputs/
├── skills/
├── src/ashare_quant/
└── tests/
```

## 设计原则

1. **先研究、后交易**：先验证因子和组合逻辑，再谈执行。
2. **面向 A 股约束**：T+1、ST、停牌、涨跌停、换手与冲击成本必须进入系统。
3. **因子视角为主**：以截面选股 + 风险约束 + 回测验证为主线。
4. **LLM 负责研究协作，不替代数值计算**：让 Codex 负责代码、文档、测试、流程自动化；让 Python 负责计算与回测。
5. **从原型到生产渐进升级**：当前是 research-grade，后续可替换为真实数据、真实撮合、真实风控。

## 快速开始

### 1. 进入仓库

```bash
cd a_share_quant_codex
```

### 2. 安装依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 3. 生成模拟数据

```bash
.venv/bin/python examples/generate_mock_data.py
```

### 4. 运行示例回测

```bash
.venv/bin/python examples/run_mock_backtest.py
```

### 5. 运行测试

```bash
.venv/bin/python -m unittest discover -s tests
```

## 关键输出

运行后会生成：

- `outputs/mock_metrics.json`
- `outputs/equity_curve.csv`
- `outputs/equity_curve.png`
- `outputs/factor_weights.json`
- `outputs/target_weights.csv`

## 生产化骨架新增项

当前仓库还不是实盘系统，但已经新增两条可扩展接口：

- `src/ashare_quant/data/base.py`：统一的 `MarketDataSource` 抽象，`CSVDataSource` 继续可用
- `src/ashare_quant/data/tushare_adapter.py`：Tushare 日 K 增量抓取与统一 schema 归一化
- `src/ashare_quant/execution/`：`ExecutionAdapter` 与 `PaperExecutionAdapter`

可运行的 paper execution 示例：

```bash
.venv/bin/python examples/run_paper_execution_demo.py
```

如果你要做“回测之后继续跑动态模拟盘”，现在可以直接用收盘后 paper monitor：

```bash
.venv/bin/python scripts/run_paper_monitor.py \
  --monitor-dir outputs/daily_monitor_auto \
  --slice-path data/daily_monitor_slice.csv \
  --adjust qfq \
  --paper-config configs/paper_trade_guardrails.json \
  --backtest-config configs/backtest_liquidity.json \
  --state-path outputs/paper_monitor_auto/paper_state.json \
  --output-dir outputs/paper_monitor_auto
```

它和单次回测的区别是：

- 会先执行上一交易日挂着的模拟订单
- 会按最新收盘价更新 paper NAV、现金、持仓和回撤
- 会把 drawdown / 单日亏损 / regime 风险状态一起纳入风控
- 只为下一交易日生成新的待执行订单，而不是把 daily monitor 直接当买入清单

## 真实日 K 工作流

如果你已经有一份本地十年历史数据，并且只想用 Tushare 做 2026-03-20 之后的周度增量更新，推荐顺序是：

1. 如果你的历史数据是“每只股票一个 CSV”的目录，先执行：

```bash
.venv/bin/python scripts/bootstrap_cn_history_directory.py --source-dir /path/to/your/cn_daily_csv_dir --output-path data/a_share_daily.csv
```

2. 把基座历史文件放到 `data/a_share_daily.csv`
3. 设置环境变量 `TUSHARE_TOKEN`
4. 如果你需要走自定义 HTTP 入口，再额外设置 `TUSHARE_HTTP_URL`
5. 每周五收盘后或周末运行：

```bash
.venv/bin/python scripts/update_tushare_daily_dataset.py --existing-path data/a_share_daily.csv
```

6. 跑真实日 K 研究回测：

```bash
.venv/bin/python examples/run_real_daily_backtest.py --data-path data/a_share_daily.csv --adjust qfq
```

如果你想直接用“研究 baseline -> 更接近实盘原型”的默认配置，推荐：

```bash
.venv/bin/python examples/run_real_daily_backtest.py --data-path data/a_share_daily_industry.csv --adjust qfq --research-config configs/recommended_default_config.json --backtest-config configs/backtest_liquidity.json
```

当前默认配置已经从“monitor 优先”切到“更接近实盘的低换手原型”，核心是：

- `stable6` 因子集，而不是默认全开 `all12`
- 只用 `3d/5d` 短周期标签，对齐 `5d` 持有
- 更强的 no-trade band
- monitor / observation pool 与 tradable strategy / deployable prototype 两层分离

保留的旧配置在：

- `configs/research_monitor_legacy.json`

如果你想直接试“研究机器人当前效果”，推荐用全量主库先切研究样本，再一次性跑回测和 LLM-ready 摘要：

```bash
.venv/bin/python scripts/run_real_research_workflow.py --input-path data/a_share_daily.csv --slice-path data/research_slice.csv --start-date 2019-01-01 --max-codes 500 --adjust qfq
```

说明：

- `qfq` 更适合长区间日 K 因子研究
- 如果你的本地历史文件本身就是未复权原始行情，可以先用 `--adjust none`
- 更新脚本默认只从本地文件里的**最后一个交易日之后**开始增量拉取，不会重下整段十年历史
- `--max-codes 0` 表示保留整个符合条件的股票池，而不是只取高流动性的前 N 只
- 如果官方默认入口不通，可以设置 `TUSHARE_HTTP_URL=http://8.136.22.187:8010/`，或在脚本上显式传 `--http-url`
- 如果第三方要求走标准代理，也可以设置 `TUSHARE_PROXY_URL=http://x.x.x.x:xxxx`，或在脚本上显式传 `--proxy-url`
- 如果你的 macOS 系统代理把 Tushare 请求带偏了，可以额外传 `--bypass-system-proxy`，或设置 `TUSHARE_BYPASS_SYSTEM_PROXY=1`

先做最小连通性检查：

```bash
.venv/bin/python scripts/check_tushare_connection.py --bypass-system-proxy
```

生成最新一期候选股：

```bash
.venv/bin/python scripts/generate_latest_picks.py --data-path data/research_slice.csv --adjust qfq --output-prefix latest_picks
```

做 walk-forward 对比：

```bash
.venv/bin/python scripts/run_walk_forward_grid.py --input-path data/a_share_daily.csv --start-date 2019-01-01 --max-codes-list 200,500 --train-years-list 2,3 --adjust qfq --output-prefix walk_forward_expanded
```

做 baseline 到“可实盘原型”的 ablation：

```bash
.venv/bin/python scripts/run_ablation_suite.py --data-path data/a_share_daily_industry.csv --adjust qfq --output-dir outputs/ablation_suite
```

这会输出：

- `outputs/ablation_suite/ablation_summary.csv`
- `outputs/ablation_suite/ablation_summary.md`
- 每个 scenario 各自的 `equity_curve / ic_series / quantile_summary / industry_performance / size_performance / market_state_performance / construction_grid / cost_sensitivity`
- `recommended_default_config.json` 与双结论摘要（分析师视角 / 交易员视角）

如果你手头已经有 `code -> industry/name/...` 的本地映射表，也可以先不等 Tushare，直接回填元数据：

```bash
.venv/bin/python scripts/enrich_metadata_from_csv.py --existing-path data/a_share_daily.csv --metadata-path data/stock_metadata.csv
```

## 如何与 Codex 一起工作

先读：

- `AGENTS.md`
- `docs/codex_workflow.md`
- `skills/`

推荐任务示例：

- “给系统新增一个质量因子，并做 size / industry neutralization。”
- “使用 subagents：一个检查未来函数泄漏，一个审计成本模型，一个补测试。”
- “把当前回测从等权 Top-N 升级成风险预算约束组合。”
- “为 T+1 / 停牌 / 涨跌停规则写回测约束测试。”


## 新增文档

- `PROJECT_HANDOFF.md`：项目交接说明
- `docs/05_course_alignment.md`：和课程项目要求的对齐方式
- `docs/06_codex_rebuild_steps.md`：让 Codex 按步骤重建/扩展仓库
- `docs/07_production_roadmap.md`：从研究原型到实盘机器人的升级路线
- `docs/08_neurips_report_outline.md`：最终报告提纲
- `docs/09_real_data_adapter_plan.md`：真实数据与执行接口接入计划
- `docs/10_tushare_daily_workflow.md`：本地历史 + Tushare 周度增量更新工作流

## 当前原型的边界

- 目前使用**模拟数据**，方便本地复现实验流程。
- 回测是**研究原型**，不是逐笔撮合引擎。
- 执行层还没有接实盘接口。
- 风险模型目前只做了轻量版中性化；生产版应引入更完整的 Barra 风格风险框架。
- `daily_monitor` 默认输出的是 observation pool + tradable strategy snapshot，不应直接当作无条件买入清单。

## 下一步建议

1. 接入真实 A 股数据源（Tushare / AkShare / Wind / 聚宽 / 米筐等）
2. 增加行业约束、风格暴露约束、换手约束
3. 增加分层回测、Walk-forward、滚动训练与稳定性分析
4. 用 Codex 技能把“因子开发 → 回测 → 报告输出”流程自动化
