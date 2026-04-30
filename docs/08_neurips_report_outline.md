# NeurIPS 风格最终报告提纲

下面给你一个可以直接写成最终课程报告的结构。

## 标题建议

**Codex-Assisted A-Share Quant Research and Trading Robot: A Production-Oriented Multi-Factor System with Agentic Research Workflows**

## Abstract

建议写 4 句话：

1. 研究背景：A 股量化研究需要同时处理因子有效性与真实交易约束
2. 方法：提出一个 A 股多因子研究系统，并用 Codex 作为研究协作层
3. 实验：在日频截面选股任务上验证 baseline 与增强模块
4. 结论：系统在研究效率与可复验性上有效，但生产化仍依赖真实数据与执行层

## 1. Introduction

写清 3 件事：

- 为什么 A 股和美股不能直接套同一套交易假设
- 为什么量化项目不能只写一个模型
- 为什么 Agent / Codex 在这里的位置是“研究自动化”，不是直接预测器

## 2. Related Work

建议分 4 小节：

### 2.1 Multi-factor investing and A-share stock selection

- Fama-French / Carhart
- A 股因子研究

### 2.2 Financial machine learning

- 标签设计
- 排序学习
- walk-forward 与过拟合控制

### 2.3 Automated feature/factor mining

- OpenFE / OpenFE++
- AutoAlpha
- AlphaGen / QuantaAlpha

### 2.4 LLM agents for finance and quantitative research

- Alpha-GPT
- FactorMAD
- TradingAgents / HedgeAgents
- LLM + Agent 工程化工作流

## 3. Problem Formulation

这里定义：

- 标的：A 股股票池
- 输入：行情、行业、市值、可交易性、基本面等
- 输出：股票横截面得分或目标权重
- 目标：成本后风险调整收益最大化或稳定提升

## 4. System Design

### 4.1 Architecture

写总架构图：

```text
Data -> Factor -> Label -> Model -> Portfolio -> Backtest -> Report
                     ^
                     |
                  Codex layer
```

### 4.2 A-share constraints

专门写：

- T+1
- ST
- 停牌
- 涨跌停
- 佣金/印花税/滑点

### 4.3 Codex collaboration layer

写：

- `AGENTS.md`
- skills
- subagents
- 为什么这些机制提高研究一致性

## 5. Method

### 5.1 Factors

介绍当前使用的因子与经济含义。

### 5.2 Neutralization and preprocessing

- winsorize / z-score
- size / industry neutralization

### 5.3 Labels

- 未来 5 日收益
- 可扩展标签

### 5.4 Ranking model

- IC 加权线性打分基线
- 可扩展到树模型或排序模型

### 5.5 Portfolio construction and backtest

- Top-N long-only
- 调仓周期
- 成本与约束

## 6. Experiments

### 6.1 Experimental setup

- 股票池
- 时间区间
- 调仓频率
- 数据切分
- 指标

### 6.2 Baselines

- 随机/简单基线
- 单因子
- 多因子 IC 加权
- 你的增强版

### 6.3 Main results

展示：

- 年化收益
- 波动率
- Sharpe
- 最大回撤
- 换手
- IC / ICIR

### 6.4 Ablation study

至少做：

- 有无中性化
- 有无成本
- 单因子 vs 多因子
- 线性 vs 非线性模型

### 6.5 Failure cases and limitations

写清楚：

- mock 数据局限
- 因子衰减
- 风险模型不完整
- 还未接实盘接口

## 7. From Research Prototype to Production

这个章节会很加分。

写：

- 真正上线还缺哪些模块
- 为什么研究结果不能直接映射成实盘收益
- 你的 production roadmap 是什么

## 8. Use of AI Tools

建议单独成节。

可写：

- 使用 Codex/LLM 的范围
- 哪些由可执行代码验证
- 哪些决策由人工复核

## 9. Conclusion

结论建议强调：

- 系统价值在“研究自动化 + 工程可复验”
- A 股规则对系统设计有实质影响
- Codex 能提高研发效率，但不能替代严谨回测和风控

## Appendix 建议放什么

- 数据字段定义
- 额外图表
- 更多因子公式
- 提示词模板
- AI 使用声明全文
