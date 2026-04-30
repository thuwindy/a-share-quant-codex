# V2.2 Final Strategy Memo

## 1. Production Baseline

当前生产候选基线为 `V2.2 baseline`，这是在最近窗口 `2024-01-02` 到 `2026-03-30`、`T+1 VWAP`、`15bps` 滑点、严格 A 股交易约束下，表现最稳且最诚实的版本。

核心配置：

- 持有周期：`40d`
- 标签周期：`20d / 40d`
- 执行口径：`signal at t close -> execute at t+1 vwap`
- 股票池：高流动性 `500` 股票母池
- 组合构建：`Top 40`
- 权重方式：`rank`
- 换手阻尼：`weight_change_threshold = 0.05`
- 行业约束：`strict industry cap = 15%`
- 风险惩罚：`volatility_20_neu`、`turnover_20_neu`
- 主 alpha：`momentum_20_neu`、`momentum_60_neu`

同窗回测结果：

- Gross Annual Return：`51.76%`
- Net Annual Return：`48.19%`
- Sharpe：`1.9632`
- Max Drawdown：`-16.35%`
- Avg Turnover：`4.08%`
- Cost Drag：`3.57%`

结论：

- 这是当前仓库中最适合挂到动态模拟盘的默认版本。
- 在当前真实执行假设下，它已经明显优于所有短周期量价方案，也优于已测试的 premium 变体。
- 默认生产配置应继续使用 `configs/research_v2_2_mid40_vwap_indcap_top40.json`。

## 2. The Graveyard

以下因子或策略路线，已经在本项目的真实压力测试中被明确证伪或降级，不应再作为默认主策略复活。

### `reversal_5_neu`

死因：短线反转收益高度依赖隔夜跳空和开盘情绪释放。在 `T+1 open / vwap + 15bps` 下，毛收益被执行和摩擦成本快速吃穿。

### `intraday_range_10_neu`

死因：更像短期噪音捕手，不适合作为日 K 中线 alpha。在真实执行口径下，无法稳定贡献 after-cost 收益。

### `gap_5_neu`

死因：信号过于贴近隔夜与早盘博弈，日线级别无法稳定提取可交易 alpha。

### `analyst_revision_score`

死因：在 `2024-2026` 的 A 股极端行情中，卖方修正显著滞后。作为主打分因子时没有形成增益，反而轻微拖累 Sharpe。

### `overhead_resistance` 作为硬 veto

死因：会误杀右尾收益最强的趋势票。做动量时，过强的筹码阻力 veto 会砍掉真正的主升浪和强力解套行情。

### `stable6 + 3d/5d`

死因：短周期量价反转在 `T+1 VWAP + 15bps` 下被彻底打穿，不具备继续作为主策略优化的价值。

## 3. Premium Arsenal

10000 积分的高级数据并没有浪费。它们目前最好的定位不是“强行顶替 V2.2”，而是成为未来研究和盯盘的副武器。

### `smart_money_inflow_20`

当前定位：观察指标 / 次级增强候选

结论：

- 有真实信息量
- 能稳定拿到正权重
- 但在当前 recent-window 下，还不足以让 `V2.3` 打败 `V2.2`

建议用法：

- 作为 observation layer 的解释因子
- 作为后续 `V2.2+` 的次级增强候选
- 优先观察是否能改善选股胜率，而不是强行追求更高年化

### `overhead_resistance`

当前定位：观察指标 / 新仓门禁候选

结论：

- 不适合作为全局硬 veto
- 不适合作为过强 penalty
- 更适合做 `strict entry, loose exit` 的新仓入场过滤器

建议用法：

- 仅对新开仓位生效
- 老仓只看动量衰竭，不因筹码阻力提前砍掉

### `analyst_revision_score`

当前定位：储备因子，不进入默认生产主链

结论：

- 这一窗口下没有形成稳定增益
- 可保留在数据中台中，供未来低频基本面轮动实验使用
- 不应出现在当前实盘原型默认因子集中

## 4. Next Horizons

当前日 K 中线主策略已经逼近“在现有约束下的性价比最优”。下一阶段不应继续无节制堆因子，而应把重心转向以下四条战线。

### 4.1 实盘滑点统计

目标：

- 用 paper trading 的真实成交近似，校准当前 `15bps` 假设
- 每日记录 `signal price / assumed VWAP / simulated fill / realized slippage`

意义：

- 如果真实 paper 滑点长期低于 `15bps`，当前 V2.2 的真实可部署性会更强
- 如果更高，应提前收紧仓位或下调预期

### 4.2 Regime Gate 二次打磨

目标：

- 不让风控等于“不交易”
- 研究市场状态差时，是否应做“降新仓、保老仓”而不是粗暴缩总暴露

意义：

- 把当前 `long-only` 中线引擎从“会赚钱”进一步推向“更会活下来”

### 4.3 低频风格轮动

目标：

- 不再从短线量价里挤 alpha
- 转向更低频的风格切换、行业轮动、红利 vs 硬科技切换研究

意义：

- 当前 V2.2 的胜利，本质上已经证明“中线趋势 + 严约束”比短线博弈更适合这套系统

### 4.4 Premium 数据的保守增强

目标：

- 继续保留 `smart_money_inflow_20`
- 对 `overhead_resistance` 做更细的 entry-only 试验
- 暂停 `analyst_revision_score` 的主链使用

意义：

- 让高级数据先做“雷达”和“副武器”
- 避免为了证明 10000 积分有价值而强行把它们塞进主策略

## Final Decision

最终投决结论如下：

- 默认生产候选策略：`V2.2 baseline`
- 当前策略定位：`tradable prototype`，适合继续挂接 paper trading
- Premium 数据定位：`data moat + research arsenal`，不是当前默认收益引擎
- 近期不再继续挖短周期日 K alpha
- 下一阶段重心：`paper trading`, `slippage calibration`, `regime refinement`, `low-frequency rotation`

一句话结案：

> 当前最强的，不是最复杂的，而是最诚实、最克制、最能穿越真实执行摩擦的那一版。
