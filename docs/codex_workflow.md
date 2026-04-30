# 如何用 Codex 推进这个 A 股量化仓库

## 1. 为什么这个仓库适合 Codex

因为量化研究里有大量重复但结构化的工作：

- 实现新因子
- 做中性化
- 检查未来函数
- 跑回测
- 比较输出
- 补文档和测试

这些都非常适合交给 Codex。

## 2. 推荐使用方式

### App / IDE / CLI 都可以

如果你已经习惯 VS Code / Cursor，就直接在 IDE 里用。
如果想把复杂任务并行拆开，Codex App / CLI 更适合。

## 3. 启动顺序

### CLI

```bash
npm i -g @openai/codex
codex
```

### 进入仓库后第一句建议

```text
Read README.md, AGENTS.md, docs/study_guide.md and docs/architecture.md first. Then summarize the repository and propose the next three highest-impact tasks.
```

## 4. 最有效的 Prompt 模板

### 4.1 新增因子

```text
Use the a-share-factor-research skill. Add a new cross-sectional factor called analyst_revision_proxy. Keep A-share constraints, add neutralization, write tests, run the mock backtest, and summarize whether the factor improves IC or Sharpe.
```

### 4.2 审计未来函数

```text
Use the backtest-audit skill. Audit the repository for lookahead bias, execution-date leakage, benchmark misuse, and unrealistic turnover assumptions. Make only high-confidence fixes and explain each one.
```

### 4.3 并行子代理

```text
Use subagents.
- Agent 1: inspect factor leakage and label alignment.
- Agent 2: review transaction costs and A-share constraints.
- Agent 3: improve tests and docs.
Then merge the results into one implementation plan.
```

### 4.4 生成答辩材料

```text
Use the research-report skill. Read outputs/, summarize metrics, explain the strategy logic, list weaknesses, and draft a project-report section in docs/.
```

## 5. 这个仓库里 Codex 应遵守的规则

1. 先读 `AGENTS.md`
2. 不要跳过测试
3. 不要直接把 LLM 输出当作数值结论
4. 任何改动都尽量留下可复验命令
5. 尽量以小 PR / 小提交推进

## 6. 推荐的迭代节奏

### 版本 0：课程项目

- 日频
- 因子选股
- 简单回测
- 基础风控

### 版本 1：研究平台

- 多因子
- 滚动训练
- 分层分析
- 自动报告

### 版本 2：仿生产

- 真实数据
- 更真实撮合
- 行业/风格风险约束
- 调度与实验管理

## 7. 适合交给 Codex 的任务清单

### 高适配

- 新模块脚手架
- 代码重构
- 单测补齐
- 研究报告整理
- 实验编排
- docs / README / 配置清理

### 中适配

- 因子公式草案
- 指标扩展
- 风控逻辑升级

### 低适配（需人工强审）

- 直接决定最终交易参数
- 根据少量样本给出投资结论
- 以“看起来有效”替代严格样本外验证

## 8. 最后的原则

让 Codex 做：

- 重复工作
- 结构化工作
- 可验证工作

不要让 Codex 代替：

- 市场判断
- 数值真值
- 风险背书
