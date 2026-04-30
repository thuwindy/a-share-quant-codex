# Quant Agent TODO

Status legend: `[x] done`, `[ ] pending`, `[-] in progress`

## Phase 0: Contract & Orchestration

- [x] T00 Define 10-agent unified contract (`configs/agents_manifest.yaml`)
- [x] T01 Define DAG orchestration template (`configs/agent_orchestration_template.yaml`)
- [x] T02 Add runnable workflow script (`scripts/run_agent_workflow_template.sh`)
- [x] T03 Add risk governor baseline (`scripts/run_risk_governor.py`, `configs/risk_governor.json`)
- [x] T04 Add premium incremental + progress logs (`scripts/sync_tushare_premium_tables.py`)

## Phase 1: Data (A1) & Analysis (A2)

- [x] T10 Verify daily sync consistency and row-level dedupe on `(date, code)`  
  evidence: `outputs/data_quality/t10_daily_sync_consistency_checks.csv`
- [x] T11 Verify bool/date dtype normalization and no DtypeWarning in core path  
  evidence: `outputs/data_quality/t11_dtype_schema_checks.csv`, `outputs/daily_monitor_t11_check/t11_check_20260401_report.md`
- [x] T12 Add data quality report (missing ratio, duplicate ratio, unknown industry ratio)  
  evidence: `outputs/data_quality/daily_quality_20260401_summary.csv`, `outputs/data_quality/industry_backfill_20260401.json`
- [x] T13 Validate point-in-time event alignment (`ann_date <= date`)  
  evidence: `outputs/data_quality/t13_point_in_time_checks.csv`, `outputs/data_quality/t13_point_in_time_summary.json`
- [x] T14 Define storage split plan: sqlite/dev, mysql|clickhouse/prod  
  evidence: `docs/12_storage_split_plan.md`

## Phase 2: Strategy (A3) & Backtest (A4)

- [x] T20 Freeze baseline config for stable production default  
  evidence: `configs/research_production_default.json`, `outputs/ops_logs/post_close_20260401_234723.log`
- [x] T21 Validate signal/label/execution alignment under close/open/vwap variants  
  evidence: `outputs/data_quality/t21_execution_alignment_checks.csv`, `outputs/data_quality/t21_execution_alignment_summary.json`, `scripts/check_execution_alignment_variants.py`, `tests/test_label_builder.py`, `tests/test_engine.py`
- [x] T22 Ensure cost model includes commission/slippage/sell-tax consistently  
  evidence: `outputs/data_quality/t22_cost_model_consistency_checks.csv`, `outputs/data_quality/t22_cost_model_consistency_summary.json`, `scripts/check_cost_model_consistency.py`, `tests/test_paper_execution.py`
- [x] T23 Add top-N and no-trade-band quick grid runner  
  evidence: `scripts/run_quick_topn_notrade_grid.py`, `src/ashare_quant/analysis/quick_grid.py`, `outputs/quick_grid/quick_grid_smoke_20260402_summary.csv`, `outputs/quick_grid/quick_grid_smoke_20260402_best_overrides.json`
- [x] T24 Add strategy change log (what changed, why changed, impact)  
  evidence: `src/ashare_quant/analysis/audit_report.py`, `outputs/ablation_suite_smoke_t2431/strategy_change_log.csv`, `outputs/ablation_suite_smoke_t2431/ablation_summary.md`

## Phase 3: Eval (A5) & Visualize (A6)

- [x] T30 Standardize metric pack output (`annual_return`, `sharpe`, `max_drawdown`, `turnover`, `cost_drag`)  
  evidence: `outputs/daily_monitor_auto/daily_monitor_20260401_metric_pack.json`, `outputs/data_quality/t22_cost_model_consistency_checks.csv`, `scripts/build_standard_metric_pack.py`, `src/ashare_quant/analysis/metric_pack.py`, `outputs/ops_logs/post_close_20260402_003931.log`
- [x] T31 Add leakage/bias checklist in every experiment summary  
  evidence: `src/ashare_quant/analysis/audit_report.py`, `outputs/ablation_suite_smoke_t2431/leakage_bias_checklist.csv`, `outputs/ablation_suite_smoke_t2431/ablation_summary.md`
- [x] T32 Add daily dual-board report checklist (main + under20)  
  evidence: `scripts/build_dual_board_report.py`, `outputs/daily_monitor_dual/dual_monitor_20260401_report.md`
- [x] T33 Add risk-governor summary section into daily markdown  
  evidence: `scripts/refresh_daily_monitor_report.py`, `outputs/daily_monitor_auto/daily_monitor_20260401_report.md`, `outputs/ops_logs/t32_t33_smoke_20260403.txt`

## Phase 4: Utils (A7), Live (A8), Risk (A9), Orchestrator (A10)

- [x] T40 Wire workflow toggles for fast/standard/full modes  
  evidence: `scripts/run_agent_workflow_template.sh` (`WORKFLOW_MODE`), `outputs/ops_logs/post_close_20260401_233903.log`
- [x] T41 Add retry + timeout policy for premium sync tasks  
  evidence: `scripts/sync_tushare_premium_tables.py` (`--task-max-retries`, `--task-retry-sleep-seconds`, `--degrade-report-rc-on-error`), `outputs/premium_sync_smoke/*`
- [x] T42 Add paper-trade gate: block when A9 status is `BLOCK`  
  evidence: `outputs/paper_monitor_auto/paper_gate_20260401.json`, `outputs/paper_monitor_auto/paper_gate_allow.flag`, `outputs/ops_logs/post_close_20260402_005107.log`
- [x] T43 Add cloud cron deployment recipe (daily post-close run + log rotation)  
  evidence: `scripts/install_cloud_cron.sh`, `scripts/rotate_ops_logs.sh`, `docs/13_cloud_cron_recipe.md`, `outputs/ops_logs/t43_cron_dry_run_20260402.txt`
- [x] T44 Add failure notification hook (placeholder: log/event, later webhook)  
  evidence: `scripts/notify_workflow_failure.py`, `outputs/ops_logs/post_close_last_error.json`, `outputs/ops_logs/alerts/failure_event_20260403_194517.json`, `docs/14_reliability_and_tuning_hooks.md`

## Phase 5: Readability & LLM Integration

- [x] T34 Upgrade daily reports into desk-style dashboard markdown  
  evidence: `scripts/run_daily_monitor.py`, `scripts/build_dual_board_report.py`, `outputs/daily_monitor_auto/daily_monitor_20260401_report.md`, `outputs/daily_monitor_dual/dual_monitor_20260401_report.md`
- [x] T35 Define stable LLM attachment plan for 10-agent workflow  
  evidence: `docs/15_llm_agent_integration_plan.md`

## One-by-one execution order (recommended)

1. Real cloud rerun on server with latest workflow
2. Add webhook/IM notification for A7
3. If needed, start phase-1 LLM attachment on A6 only
