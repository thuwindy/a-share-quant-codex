# 课程项目对齐说明

## 建议项目题目

**Codex-Assisted A-Share Quant Research and Trading Robot: From Factor Mining to Production-Oriented Deployment**

中文可写成：

**面向 A 股的 Codex 协作式量化研究与交易机器人：从因子挖掘到生产化落地**

## 为什么这个题目适合课程

这份项目同时覆盖了课程网页里最核心的三条方向：

1. **Factors for Chinese A-share Stocks**
   - 以 A 股为研究对象
   - 做多因子选股、因子中性化、IC/分层/组合回测
2. **LLM for Factor Mining / LLM Financial Agents**
   - 不把大模型直接当作预测器
   - 而是让 Codex 成为“研究工程师 / 研究操作系统”
   - 用 AGENTS.md、skills、subagents 固化研发流程
3. **Production-oriented system design**
   - 项目不仅停留在 notebook
   - 给出从研究原型到可部署机器人的工程升级路径

## 本项目的研究问题

### Q1. 如何构建一个面向 A 股的研究级量化系统？

回答方式：

- 用统一的数据契约接日频行情、行业、市值、ST/停牌/涨跌停标记
- 用因子层 + 标签层 + 模型层 + 组合层 + 回测层组成研究链路
- 用 A 股约束修正回测假设

### Q2. 如何让 Codex 真正帮助量化研究，而不是只会“生成代码”？

回答方式：

- 用 `AGENTS.md` 写仓库长期规则
- 用 `skills/` 固化高频任务：新增因子、回测审计、生成报告
- 用 subagents 并行做“泄漏审计 / 成本审计 / 测试补齐”

### Q3. 如何把课程项目继续推进到“可投入使用”的量化机器人？

回答方式：

- 给出真实数据接入方案
- 给出执行适配层、风控层、监控层、调度层设计
- 给出研究 -> 仿生产 -> 实盘灰度三阶段路线图

## 你在报告里可以怎么写创新点

下面是**比较稳妥、易写、也比较容易拿分**的创新表达，不必夸大：

### 创新点 1：研究流程创新

不是只比较几个模型，而是把：

- A 股规则
- 因子研究
- 回测验证
- Codex 协作

放进同一个可复验仓库。

### 创新点 2：Agent 使用位置合理

不是让 LLM 直接给出买卖结论，而是让它：

- 生成因子实现草案
- 审计未来函数
- 补测试
- 汇总实验
- 生成报告草稿

这比“直接用 LLM 预测涨跌”更符合课程里对 Agent 的技术定位。

### 创新点 3：A 股落地导向

项目明确把下列现实问题纳入系统：

- T+1
- ST 与停牌
- 涨跌停导致不可买卖
- 印花税、佣金、滑点
- 行业/市值中性化

## Baseline 设计建议

报告里至少准备 3 个 baseline：

1. **等权随机/简单基线**
   - 等权持有股票池
   - 或者简单市值排序
2. **单因子基线**
   - 动量因子
   - 反转因子
3. **多因子线性组合基线**
   - 当前仓库提供的 IC 加权线性 ranker

然后再展示你的增强版：

- 更强的因子库
- 更好的中性化
- 更好的标签
- 更好的模型
- 更真实的交易约束

## 最好准备的实验章节

### 1. 因子有效性

- RankIC / ICIR
- 分层收益
- 因子稳定性

### 2. 组合有效性

- 年化收益
- 波动率
- Sharpe
- 最大回撤
- 换手

### 3. 消融实验

至少做以下消融：

- 不做中性化 vs 做中性化
- 不考虑成本 vs 考虑成本
- 单因子 vs 多因子
- 简单线性模型 vs 树模型/排序模型

### 4. A 股约束影响分析

展示：

- 停牌/涨跌停/印花税纳入后，绩效如何变化
- 换手约束对结果的影响

### 5. Codex 贡献分析

报告里专门写一小节：

- 哪些模块由 Codex 协助完成
- 哪些地方由人工审查
- 哪些结果必须由代码而不是语言模型计算

## 报告里的合规披露

### 开源代码使用

如果你后续接入了第三方库或改写了公开仓库：

- 在 related work / implementation details 里明确说明来源
- 明确写“哪些部分是原始代码，哪些部分是你改的”

### AI 使用披露

建议单独放一个小节：

**Use of AI Tools**

可写：

- We used Codex / LLMs for code scaffolding, documentation drafting, experiment organization, and report polishing.
- All quantitative results, backtests, and metrics were produced and verified by executable Python code.
- Final implementation choices and result interpretation were reviewed by the authors.

## 最终展示建议

答辩 5-10 分钟时，不要平均讲所有模块。建议只讲 4 件事：

1. 问题：为什么不是“随便跑一个模型”就行
2. 系统：A 股研究系统 + Codex 协作层
3. 结果：核心指标与对比
4. 落地：怎样升级成可投入使用的机器人

## 一句话定位

这份项目最好的定位不是：

> “我用大模型炒股。”

而是：

> “我做了一套面向 A 股的量化研究与交易机器人骨架，并让 Codex 成为可靠的研究协作层。”
