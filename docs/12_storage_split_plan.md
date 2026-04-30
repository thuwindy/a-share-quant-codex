# T14 Storage Split Plan (Dev / Prod)

## Goal

Define a clear storage split so the quant workflow can run fast locally and scale safely in production.

## Environment Split

| Environment | Default Store | Use Case | Notes |
| --- | --- | --- | --- |
| Local dev / single-user | SQLite + CSV/Parquet | Fast debugging, replay, one-machine daily run | Zero-ops, easiest onboarding |
| Small team staging | MySQL + object storage (Parquet) | Shared run states, shared configs, integration tests | Keep schema strict and auditable |
| Production research/data-plane | ClickHouse + object storage | Large panel/history, factor/event joins, analytics queries | Partition by date, sort by `(date, code)` |
| Production control-plane | MySQL | Workflow state, run checkpoints, strategy configs, paper/live orders | Transaction-safe and easy CRUD |

## What Goes Where

### 1) SQLite (dev default)

- `ops_state.db`: workflow checkpoints, daily run status, lightweight cache.
- `paper_state.db`: paper-trade positions/orders for local replay.
- Rule: only for local/manual mode, not authoritative in team/prod.

### 2) MySQL (team/prod control-plane)

- `workflow_runs`: run id, status, started_at, finished_at, log_path.
- `risk_gate_results`: date, status(PASS/WARN/BLOCK), triggered checks.
- `paper_orders`: staged/executed/cancelled order lifecycle.
- `strategy_configs`: active config version and change history.
- Rule: use MySQL for durable state that needs ACID and concurrent writes.

### 3) ClickHouse (prod data-plane)

- `daily_bar` (large historical panel; partition by month).
- `premium_flow_daily`, `premium_chip_daily`, `premium_hk_hold_daily`, `premium_limit_sentiment_daily`.
- `feature_panel` / `factor_panel` / `signal_table` / `fill_table` / `metrics_daily`.
- Rule: high-volume analytical reads/writes, append + partition pruning.

## Canonical Keys & Partitions

- Canonical key for panel tables: `(date, code)`.
- ClickHouse partition: `toYYYYMM(date)`.
- Order key recommendation:
  - data/factor tables: `ORDER BY (date, code)`.
  - run/event tables: `ORDER BY (run_date, run_id)`.

## Data Contract & Time Alignment Constraints

- Event visibility must satisfy `event_date <= date` (point-in-time).
- Factor/label use adjusted price; tradability/execution checks use raw price.
- Store `signal_time`, `execution_time`, `holding_window` in signal/metrics outputs.

## Retention & Backup

- Raw premium snapshots: keep 180 days hot + monthly archive.
- Daily bars/factors/signals/metrics: keep full history.
- MySQL control-plane: daily backup + 7-day binlog retention.

## Migration Path

1. Keep current local workflow on CSV + SQLite.
2. Add MySQL for workflow/risk/paper state persistence.
3. Mirror large panel/factor outputs into ClickHouse.
4. Switch read path for analytics/report from CSV to ClickHouse views.

## Acceptance Criteria (T14 Done)

- Dev mode can run end-to-end without MySQL/ClickHouse dependency.
- Control-plane tables are mapped to MySQL (schema documented).
- Data-plane large tables are mapped to ClickHouse (keys/partitions documented).
- Contract remains consistent with `configs/agents_manifest.yaml`.
