from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight check and auto-repair for evening brief artifacts.")
    parser.add_argument("--python-bin", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--master-data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--main-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_auto"))
    parser.add_argument("--elastic-monitor-dir", default=str(ROOT / "outputs" / "daily_monitor_under20_elastic"))
    parser.add_argument("--shortline-dir", default=str(ROOT / "outputs" / "shortline_opportunities"))
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--main-slice-path", default=str(ROOT / "data" / "daily_monitor_main_slice.csv"))
    parser.add_argument("--elastic-slice-path", default=str(ROOT / "data" / "daily_monitor_elastic_slice.csv"))
    parser.add_argument("--main-research-config", default=str(ROOT / "configs" / "research_production_default.json"))
    parser.add_argument("--elastic-research-config", default=str(ROOT / "configs" / "research_under20_elastic_top20.json"))
    parser.add_argument("--backtest-config", default=str(ROOT / "configs" / "backtest_production_managed_15bps.json"))
    parser.add_argument("--shortline-config", default=str(ROOT / "configs" / "research_shortline_opportunity.json"))
    parser.add_argument("--premium-dir", default=str(ROOT / "data" / "premium_v22"))
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--sync-shortline-limit-data", action="store_true")
    parser.add_argument("--elastic-start-date", default="2023-01-01")
    parser.add_argument("--elastic-max-codes", type=int, default=200)
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _latest_date(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        step = 4096
        data = b""
        pos = size
        while pos > 0 and b"\n" not in data:
            delta = min(step, pos)
            pos -= delta
            fh.seek(pos)
            data = fh.read(delta) + data
        lines = [line for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].split(",", 1)[0].strip()


def _match_date_path(patterns: list[str], date_key: str) -> Path | None:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern))
    matched = [path for path in found if date_key in path.name]
    if not matched:
        return None
    return sorted(matched)[-1]


def _run(cmd: list[str], *, cwd: Path, dry_run: bool, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "stdout": "", "stderr": "", "dry_run": True}
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _artifact_state(root: Path, date_key: str) -> dict[str, Path | None]:
    return {
        "main": _match_date_path(
            [str(root / "outputs" / "daily_monitor_auto" / "daily_monitor_*_picks.json")],
            date_key,
        ),
        "elastic": _match_date_path(
            [str(root / "outputs" / "daily_monitor_under20_elastic" / "under20_elastic_*_picks.json")],
            date_key,
        ),
        "shortline": _match_date_path(
            [str(root / "outputs" / "shortline_opportunities" / "shortline_opportunity_*_cards.json")],
            date_key,
        ),
        "risk": _match_date_path(
            [str(root / "outputs" / "risk_governor" / "risk_gate_*.json")],
            date_key,
        ),
    }


def _write_empty_shortline_artifact(output_dir: Path, target_date: str, reason: str, *, dry_run: bool) -> dict[str, Any]:
    date_key = target_date.replace("-", "")
    json_path = output_dir / f"shortline_opportunity_{date_key}_cards.json"
    csv_path = output_dir / f"shortline_opportunity_{date_key}_cards.csv"
    report_path = output_dir / f"shortline_opportunity_{date_key}_report.md"
    payload = {
        "summary": {
            "selection_date": target_date,
            "selected_count": 0,
            "candidate_count": 0,
            "data_latest_date": target_date,
            "limit_latest_date": target_date,
            "report_type": "shortline_opportunity_under20",
            "status": "no_candidates",
            "reason": reason,
        },
        "cards": [],
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path.write_text("rank,code,name,close,shortline_score,risk_level,operation_logic\n", encoding="utf-8")
        report_path.write_text(f"# 精选短线机会 {target_date}\n\n{reason}\n", encoding="utf-8")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "reason": reason,
        "dry_run": dry_run,
    }


def main() -> None:
    args = build_parser().parse_args()
    python_bin = str(args.python_bin)
    target_date = _latest_date(Path(args.master_data_path))
    if not target_date:
        raise ValueError(f"failed to resolve latest date from master data: {args.master_data_path}")
    date_key = target_date.replace("-", "")

    result: dict[str, Any] = {
        "target_date": target_date,
        "date_key": date_key,
        "repair_steps": [],
    }
    state = _artifact_state(ROOT, date_key)
    result["before"] = {key: str(value) if value else "" for key, value in state.items()}

    missing = [name for name, value in state.items() if value is None]
    if "main" in missing or "elastic" in missing:
        step = _run(
            [
                "bash",
                str(ROOT / "scripts" / "run_split_daily_monitors.sh"),
            ],
            cwd=ROOT,
            dry_run=bool(args.dry_run),
            extra_env={"TARGET_DATE": target_date, "FORCE_REBUILD_SLICES": "1"},
        )
        result["repair_steps"].append(
            {
                "step": "run_split_daily_monitors",
                **step,
            }
        )
    state = _artifact_state(ROOT, date_key)
    missing = [name for name, value in state.items() if value is None]

    if "main" in missing:
        step = _run(
            [
                python_bin,
                str(ROOT / "scripts" / "run_daily_monitor.py"),
                "--skip-update",
                "--skip-backtest",
                "--data-path",
                str(args.master_data_path),
                "--slice-path",
                str(args.main_slice_path),
                "--end-date",
                target_date,
                "--adjust",
                str(args.adjust),
                "--research-config",
                str(args.main_research_config),
                "--backtest-config",
                str(args.backtest_config),
                "--output-dir",
                str(args.main_monitor_dir),
            ],
            cwd=ROOT,
            dry_run=bool(args.dry_run),
        )
        result["repair_steps"].append({"step": "rebuild_main", **step})
        state = _artifact_state(ROOT, date_key)
        missing = [name for name, value in state.items() if value is None]

    if "elastic" in missing:
        step = _run(
            [
                python_bin,
                str(ROOT / "scripts" / "build_monitor_observation_output.py"),
                "--input-path",
                str(args.master_data_path),
                "--slice-path",
                str(args.elastic_slice_path),
                "--start-date",
                str(args.elastic_start_date),
                "--end-date",
                target_date,
                "--max-codes",
                str(args.elastic_max_codes),
                "--research-config",
                str(args.elastic_research_config),
                "--output-dir",
                str(args.elastic_monitor_dir),
                "--output-prefix",
                "under20_elastic",
                "--prediction-date",
                target_date,
                "--adjust",
                str(args.adjust),
            ],
            cwd=ROOT,
            dry_run=bool(args.dry_run),
        )
        result["repair_steps"].append({"step": "rebuild_elastic", **step})
        state = _artifact_state(ROOT, date_key)
        missing = [name for name, value in state.items() if value is None]

    if "shortline" in missing:
        shortline_cmd = [
            python_bin,
            str(ROOT / "scripts" / "run_shortline_opportunity_report.py"),
            "--data-path",
            str(args.master_data_path),
            "--premium-dir",
            str(args.premium_dir),
            "--config",
            str(args.shortline_config),
            "--output-dir",
            str(args.shortline_dir),
            "--prediction-date",
            target_date,
        ]
        if bool(args.sync_shortline_limit_data):
            shortline_cmd.append("--sync-limit-data")
        if bool(args.bypass_system_proxy):
            shortline_cmd.append("--bypass-system-proxy")
        step = _run(
            shortline_cmd,
            cwd=ROOT,
            dry_run=bool(args.dry_run),
        )
        result["repair_steps"].append({"step": "rebuild_shortline", **step})
        state = _artifact_state(ROOT, date_key)
        missing = [name for name, value in state.items() if value is None]
        no_candidate_text = f"No under-20 shortline candidates matched the filters on {target_date}."
        combined_output = f"{step.get('stdout', '')}\n{step.get('stderr', '')}"
        if "shortline" in missing and "No under-20 shortline candidates matched the filters" in combined_output:
            empty_step = _write_empty_shortline_artifact(
                Path(args.shortline_dir),
                target_date,
                no_candidate_text,
                dry_run=bool(args.dry_run),
            )
            result["repair_steps"].append({"step": "write_empty_shortline", **empty_step})
            state = _artifact_state(ROOT, date_key)
            missing = [name for name, value in state.items() if value is None]

    if "risk" in missing and state.get("main") is not None:
        risk_step = _run(
            [
                python_bin,
                str(ROOT / "scripts" / "run_risk_governor.py"),
                "--monitor-dir",
                str(args.main_monitor_dir),
                "--output-dir",
                str(args.risk_dir),
            ],
            cwd=ROOT,
            dry_run=bool(args.dry_run),
        )
        report_step = _run(
            [
                python_bin,
                str(ROOT / "scripts" / "refresh_daily_monitor_report.py"),
                "--monitor-dir",
                str(args.main_monitor_dir),
                "--risk-dir",
                str(args.risk_dir),
            ],
            cwd=ROOT,
            dry_run=bool(args.dry_run),
        )
        result["repair_steps"].append({"step": "rebuild_risk", **risk_step})
        result["repair_steps"].append({"step": "refresh_main_report", **report_step})

    state = _artifact_state(ROOT, date_key)
    result["after"] = {key: str(value) if value else "" for key, value in state.items()}
    result["missing_after"] = [name for name, value in state.items() if value is None]
    result["ok"] = not result["missing_after"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["missing_after"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
