# 给 Codex 的复现步骤

这份文档的目标不是让 Codex“理解金融理论”，而是让它**按可执行步骤重建本仓库**。

## 0. 目标产物

最终让 Codex 重建出一个仓库，至少包含：

- `README.md`
- `AGENTS.md`
- `skills/`
- `src/ashare_quant/`
- `examples/`
- `tests/`
- `docs/`
- `configs/`

并且满足：

- 能生成 mock 数据
- 能跑一遍最小回测
- 能输出 metrics
- 能通过单元测试

## 1. 第一次进入仓库时给 Codex 的 prompt

```text
Read README.md, AGENTS.md, docs/study_guide.md, docs/architecture.md, and docs/05_course_alignment.md first.
Then summarize the repository, identify the minimal runnable path, and propose an implementation plan with exactly 8 steps.
```

## 2. 从零开始重建时的顺序

### Step 1. 建仓库骨架

要求 Codex 创建目录：

```text
configs/
docs/
examples/
outputs/
skills/
src/ashare_quant/
tests/
```

并写：

- `README.md`
- `AGENTS.md`
- `requirements.txt`
- `pyproject.toml`

### Step 2. 先写文档，再写代码

让 Codex 先写：

- `docs/study_guide.md`
- `docs/architecture.md`
- `docs/codex_workflow.md`

这样做的好处是：

- 后续每个实现任务都有统一上下文
- Codex 更不容易写偏

### Step 3. 实现数据层

先让 Codex 只做：

- `CSVDataSource`
- `apply_basic_universe_filters`

要求：

- 输入是日频 CSV
- 至少支持 `date/ticker/close/volume/industry/market_cap/is_st/is_suspended/can_buy/can_sell`
- 股票池过滤必须显式考虑 ST、停牌、低流动性或缺失值

### Step 4. 实现因子层

先只做轻量可解释因子：

- 20 日动量
- 5 日反转
- 20 日波动率
- 20 日流动性
- 一个简单质量代理因子

要求：

- 函数拆小
- 输出列名清晰
- 后续能接中性化

### Step 5. 实现标签与中性化

让 Codex 接着实现：

- `add_forward_return_label`
- `neutralize_by_size_and_industry`

要求：

- 标签 horizon 可配置
- 中性化要支持因子列批量处理
- 保证按截面做处理，而不是跨时间泄漏

### Step 6. 实现模型与组合

先做最稳妥版本：

- IC 加权线性打分
- Top-N 等权 long-only 组合

不要一开始就让 Codex 上复杂优化器。

### Step 7. 实现最小回测

让 Codex 实现：

- 日频持仓收益回放
- 佣金
- 印花税
- 滑点
- 基础换手统计

要求：

- 所有指标由 Python 计算
- 回测逻辑要和标签、调仓频率对齐

### Step 8. 补 examples 与 tests

最后再让 Codex 写：

- `examples/generate_mock_data.py`
- `examples/run_mock_backtest.py`
- `tests/test_metrics.py`
- `tests/test_neutralize.py`

并要求它运行：

```bash
.venv/bin/python examples/generate_mock_data.py
.venv/bin/python examples/run_mock_backtest.py
.venv/bin/python -m unittest discover -s tests
```

## 3. 最实用的任务拆法

### 任务 A：只做一个模块

```text
Read AGENTS.md first. Only implement the CSV data adapter and the basic A-share universe filter. Do not touch any other modules. Add minimal tests if needed.
```

### 任务 B：只做因子

```text
Use the a-share-factor-research skill. Add one new interpretable factor to src/ashare_quant/factors/. Keep the change minimal, add neutralized output support, and summarize the economic rationale.
```

### 任务 C：只做审计

```text
Use the backtest-audit skill. Audit label alignment, future leakage, transaction cost assumptions, and A-share tradeability assumptions. Make only high-confidence edits.
```

### 任务 D：只做报告

```text
Use the research-report skill. Read outputs/, summarize metrics, list limitations, and draft a report-ready method and experiment section.
```

## 4. 什么时候用 subagents

只有在任务天然可以并行时才用。例如：

```text
Use subagents.
- Agent 1 audits lookahead bias and label alignment.
- Agent 2 reviews A-share tradeability rules, including ST, suspension, and limit-up/down assumptions.
- Agent 3 improves tests and CI-style run commands.
Return one merged action plan first, then implement the approved subset.
```

适合并行的任务：

- 审计不同模块
- 同时补测试与文档
- 大规模代码探索

不适合并行的任务：

- 同时改同一个核心函数
- 还没确定架构就开很多子代理

## 5. 给 Codex 的约束写法

下面这种 prompt 往往更稳定：

```text
Make the smallest reliable change.
Preserve the runnable demo.
Do not invent unavailable data fields.
Do not claim performance improvements unless they are produced by executable code.
```

## 6. 如何避免 Codex 跑偏

### 不要这样说

```text
Build a production trading robot.
```

太大、太虚，Codex 会发散。

### 建议这样说

```text
Implement the next research-grade step toward a production trading robot: add a real-data adapter interface, keep the mock demo working, and document the contract in docs/.
```

## 7. 最终复现验收标准

只要 Codex 重建后的仓库满足下面 6 条，就说明基本成功：

1. 能读 mock 数据
2. 能生成因子与标签
3. 能构建组合
4. 能跑回测
5. 能输出指标
6. 能通过测试

## 8. 再往下怎么推进

重建完研究骨架后，再按这个顺序继续：

1. 真实数据接入
2. 分层回测与 walk-forward
3. 树模型 / 排序模型
4. 组合优化与风险约束
5. 执行适配层
6. 调度、监控、告警
