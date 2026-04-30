# Tushare 日 K 接入工作流

## 目标

这套工作流的目标不是“把十年历史都从 Tushare 重新下载一遍”，而是：

1. 以你已经找到的本地十年历史数据为基座
2. 只用 Tushare 补 **2026-03-20** 之后的新日 K
3. 每周五收盘后或周末做一次增量更新
4. 用统一 schema 直接接本仓库的研究与回测链路

## 推荐的文件放置

默认约定：

- 本地历史文件：`data/a_share_daily.csv`
- 更新后仍写回：`data/a_share_daily.csv`

如果你不想覆盖原文件，可以给脚本传 `--output-path`。

如果你的原始历史数据是“每只股票一个 CSV”的目录形式，可以先执行：

```bash
.venv/bin/python scripts/bootstrap_cn_history_directory.py --source-dir /path/to/your/cn_daily_csv_dir --output-path data/a_share_daily.csv
```

这样会先把原始目录统一成仓库的标准 schema，再交给后续的 Tushare 周更脚本。

## 本地历史文件需要的最小字段

为了直接接当前研究管线，建议历史文件至少包含：

- `date`
- `code`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`

更完整、最推荐的字段是：

- `market_cap`
- `industry`
- `is_st`
- `is_suspended`
- `can_buy`
- `can_sell`
- `adj_factor`

脚本会自动兼容一些常见别名：

- `ts_code -> code`
- `ticker -> code`
- `trade_date -> date`
- `vol -> volume`

## Tushare 在这个仓库里怎么用

### 1. 增量更新

脚本：

```bash
.venv/bin/python scripts/update_tushare_daily_dataset.py --existing-path data/a_share_daily.csv
```

如果你需要改走自定义 HTTP 入口，也可以：

```bash
.venv/bin/python scripts/update_tushare_daily_dataset.py --existing-path data/a_share_daily.csv --http-url http://8.136.22.187:8010/
```

默认行为：

- 自动读取本地文件里最后一个日期
- 从下一天开始拉取 Tushare 日 K
- 只追加新增日期
- 按 `date + code` 去重
- 尽量保留本地已有的行业、名称和历史元数据

### 2. 真实日 K 回测

```bash
.venv/bin/python examples/run_real_daily_backtest.py --data-path data/a_share_daily.csv --adjust qfq
```

`--adjust` 选项：

- `none`：不复权
- `qfq`：前复权，适合长期日 K 因子研究
- `hfq`：后复权

## 权限与字段说明

这套实现优先使用：

- `trade_cal`
- `daily`
- `daily_basic`
- `adj_factor`
- `stock_basic`
- `stk_limit`
- `stock_st`

其中：

- `daily` 用来补新增日 K
- `daily_basic` 用来补 `market_cap`
- `adj_factor` 用来支持 `qfq/hfq`
- `stock_basic` 用来补 `industry`
- `stk_limit` 用来推导 `can_buy/can_sell`
- `stock_st` 用来标记 `is_st`

如果你的 Tushare 账号权限不足，脚本会对可选接口给出 warning，并尽量保留已有本地字段，不会直接重拉整段历史。

如果你使用的是非默认入口，可以设置：

```bash
export TUSHARE_HTTP_URL=http://8.136.22.187:8010/
```

如果对方提供的是“标准代理地址”，也可以设置：

```bash
export TUSHARE_PROXY_URL=http://x.x.x.x:xxxx
```

如果你的机器开着系统代理，而 Tushare 被错误转发到本地无效端口，也可以设置：

```bash
export TUSHARE_BYPASS_SYSTEM_PROXY=1
```

先做最小连通性检查：

```bash
.venv/bin/python scripts/check_tushare_connection.py --bypass-system-proxy
```

## 为什么建议周五收盘后或周末更新

你说的是“每周五更新一次”，这个节奏是合理的。建议实际更新时间放在：

- 周五 17:30 以后
- 或者周六

这样更稳，原因是：

- 当日 `daily` 和 `daily_basic` 更可能已经完整
- 你不需要在交易时段里处理不完整日线
- 周更节奏也更适合日 K 因子研究，不会额外浪费接口调用

## 当前实现的边界

已经做到：

- 不重下整段十年历史
- 只做新增日期增量更新
- 支持本地 CSV 加载 `qfq/hfq`
- 可直接跑仓库里的日 K 研究链路

还没有做到：

- 自动把“只有 OHLCV 的旧历史文件”补齐成完整 `market_cap / industry / ST` 历史真值
- 真实账户执行
- 调度器与告警

所以如果你的十年历史文件只有价格，没有市值/行业/交易约束字段，建议先确认一下字段情况，再决定是否做一次性 bootstrap 清洗。
