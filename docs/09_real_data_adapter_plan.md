# 真实数据与实盘接口接入计划

## 1. 当前数据契约

当前最小研究链路只要求一张日频表，核心字段包括：

- `date`
- `ticker`
- `open/high/low/close`
- `volume`
- `industry`
- `market_cap`
- `is_st`
- `is_suspended`
- `can_buy`
- `can_sell`

## 2. 第一阶段建议接什么数据

先只接足够支撑研究的字段，不要一口气追求完美：

### 行情与可交易性

- 复权价格
- 成交量 / 成交额
- ST / 停牌标记
- 涨跌停可买卖标记

### 元数据

- 行业分类
- 市值
- 上市日期
- 是否退市

### 基本面

- PB / PE / PS / PCF
- ROE / ROA / 毛利率 / 净利率
- 营收增长 / 净利增长

## 3. 数据源选择策略

### 低成本起步

- AkShare
- Tushare

适合：

- 课程项目
- 本地研究
- 原型验证

### 研究增强

- 聚宽 / 米筐 / Wind / 东方财富等研究平台或数据终端

适合：

- 更完整的基本面和预期数据
- 更高质量的历史回测

### 实盘阶段

- 券商柜台 / 交易接口 / OMS / PMS

适合：

- 真实订单生命周期管理
- 账户与持仓回传

## 4. 建议的数据适配器接口

当前仓库已经新增了最小 `MarketDataSource` 抽象，`CSVDataSource` 继续保留为默认研究适配器。

未来可以统一成下面的接口：

```python
class MarketDataSource:
    def load_daily_bars(self, start: str, end: str) -> pd.DataFrame:
        ...

    def load_fundamentals(self, start: str, end: str) -> pd.DataFrame:
        ...

    def load_tradeability_flags(self, start: str, end: str) -> pd.DataFrame:
        ...
```

## 5. 实盘执行接口建议

当前仓库已经补了 `ExecutionAdapter` 抽象和 `PaperExecutionAdapter`，可以先记录 mock orders / fills。

研究系统与执行系统不要耦死。建议定义一个 execution adapter：

```python
class ExecutionAdapter:
    def get_positions(self):
        ...

    def get_cash(self):
        ...

    def submit_orders(self, orders):
        ...

    def get_fills(self, trading_day: str):
        ...
```

## 6. 先做 paper trading

不要直接实盘。先做：

1. 盘后跑信号
2. 生成目标权重
3. 通过 mock execution adapter 记录“理论成交”
4. 校验与回测的一致性
5. 再切模拟盘 / 小资金实盘

## 7. 接口接入顺序

最推荐的顺序：

1. `CSVDataSource` 保持不动
2. 新增 `TushareDataSource` 或 `AkshareDataSource`
3. 新增 `ExecutionAdapter` 抽象层
4. 新增 `PaperExecutionAdapter`
5. 最后才接真实 broker adapter

## 8. 和 Codex 协作的方式

给 Codex 的好 prompt 例子：

```text
Add a new data adapter interface for real A-share daily bars. Keep CSVDataSource working. Do not implement any vendor-specific authentication yet. Document the schema and add a mock adapter test.
```

```text
Add an ExecutionAdapter abstraction and a PaperExecutionAdapter that records orders and mock fills. Do not connect to any live broker. Preserve the research pipeline.
```

## 9. 当前仓库里的 Tushare 日 K 落地方式

这部分现在已经不是规划，而是已实现的工作流：

1. 本地十年历史数据放在 `data/a_share_daily.csv`
2. `scripts/update_tushare_daily_dataset.py` 只从本地最后日期之后做增量更新
3. `examples/run_real_daily_backtest.py` 直接读取本地 CSV 跑日 K 研究回测
4. `CSVDataSource` 额外支持 `none / qfq / hfq` 三种加载模式

如果你只想把 Tushare 用在每周更新，而不是回补十年历史，这就是当前最推荐的接法。
