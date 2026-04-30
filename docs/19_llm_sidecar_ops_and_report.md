# LLM Sidecar Ops And Report

这层只做旁路增强，不改变主策略、风控、选股、推送主链路。

## 环境变量

使用 OpenAI 兼容接口。先复制 `.env.example`，再在本地未跟踪的 `.env` 或 shell 环境中填写真实值：

```bash
cp .env.example .env
# edit .env locally; never commit real keys
```

兼容备用变量：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`。`LLM_RETRIES` 用于中转站间歇超时时重试，默认 1 次。

## 运维审查

```bash
.venv/bin/python scripts/llm_daily_ops_review.py
```

输出：

- `outputs/llm_ops_review/llm_daily_ops_review_YYYYMMDD.json`
- `outputs/llm_ops_review/llm_daily_ops_review_YYYYMMDD.md`

它读取 cron 日志、最新产物日期、PushPlus 状态、preflight 状态、Tushare 状态，并输出：

- 系统是否正常
- 哪条链路异常
- 是否需要修复
- 建议命令
- 风险等级

如果没有配置 LLM key/base，或 LLM 调用失败，会自动降级到本地模板摘要。

## 日报解释渲染

```bash
.venv/bin/python scripts/llm_report_renderer.py
```

输出：

- `outputs/llm_report_renderer/llm_report_YYYYMMDD.json`
- `outputs/llm_report_renderer/llm_report_YYYYMMDD.md`

硬校验：

- 主策略、弹性池、短线、风控日期必须一致，否则不调用 LLM。
- LLM 输出里的数字必须能在结构化 JSON 输入里找到，否则丢弃 LLM 文本并回退模板。
- 输出超长会自动回退模板。
- 任意异常都不会阻塞现有晚报主链路。

## 当前使用边界

LLM 只负责解释和运维摘要，不负责：

- 改主分
- 改排序
- 改仓位
- 改风控
- 自动执行修复命令

如果后续要接入 PushPlus 正式推送，应先让这两个脚本连续旁路运行数日，确认不会产生错误日期、虚构数字或超长内容。
