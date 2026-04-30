from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.execution.paper_gate import (
    evaluate_paper_gate,
    latest_risk_gate_json,
    load_risk_gate_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate risk-governor output and decide whether paper-live step is allowed (T42)."
    )
    parser.add_argument("--risk-json", default="")
    parser.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk_governor"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "paper_monitor_auto"))
    parser.add_argument("--output-prefix", default="paper_gate")
    parser.add_argument("--block-statuses", default="BLOCK", help="Comma-separated statuses that should block paper-live.")
    parser.add_argument("--allow-flag-path", default="", help="Optional path to write allow flag (1 allow, 0 block).")
    parser.add_argument("--exit-on-block", action="store_true", help="Exit with code 20 when gate blocks paper-live.")
    return parser


def _build_markdown(payload: dict) -> str:
    reasons = payload.get("block_reasons", [])
    lines = [
        "# Paper Trade Gate",
        "",
        f"- selection_date: `{payload.get('selection_date', '')}`",
        f"- allow_paper_live: `{payload.get('allow_paper_live', False)}`",
        f"- risk_status: `{payload.get('risk_status', '')}`",
        f"- block_statuses: `{', '.join(payload.get('block_statuses', []))}`",
        f"- risk_json_path: `{payload.get('risk_json_path', '')}`",
        f"- note: `{payload.get('note', '')}`",
        "",
        "## Block Reasons",
        "",
    ]
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _resolve_output_name(prefix: str, selection_date: str) -> str:
    if selection_date:
        return f"{prefix}_{selection_date.replace('-', '')}"
    return f"{prefix}_{date.today().strftime('%Y%m%d')}"


if __name__ == "__main__":
    args = build_parser().parse_args()

    block_statuses = [token.strip() for token in str(args.block_statuses).split(",") if token.strip()]

    risk_json_path: Path | None = None
    payload: dict
    try:
        risk_json_path = Path(args.risk_json) if args.risk_json else latest_risk_gate_json(args.risk_dir)
        summary = load_risk_gate_summary(risk_json_path)
        decision = evaluate_paper_gate(summary, risk_json_path=risk_json_path, block_statuses=block_statuses)
        payload = decision.as_dict()
    except FileNotFoundError:
        payload = {
            "selection_date": "",
            "allow_paper_live": False,
            "risk_status": "MISSING",
            "block_statuses": [str(s).upper() for s in block_statuses],
            "block_reasons": ["missing_risk_gate"],
            "risk_json_path": "",
            "note": "paper_live_blocked_missing_risk_gate",
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _resolve_output_name(args.output_prefix, payload.get("selection_date", ""))
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    flag_path = Path(args.allow_flag_path) if args.allow_flag_path else (out_dir / "paper_gate_allow.flag")
    flag_path.write_text("1" if payload.get("allow_paper_live", False) else "0", encoding="utf-8")

    print(
        json.dumps(
            {
                "allow_paper_live": payload.get("allow_paper_live", False),
                "risk_status": payload.get("risk_status", ""),
                "selection_date": payload.get("selection_date", ""),
                "block_reasons": payload.get("block_reasons", []),
                "risk_json_path": str(risk_json_path) if risk_json_path is not None else "",
                "gate_json_path": str(json_path),
                "gate_md_path": str(md_path),
                "allow_flag_path": str(flag_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] paper trade gate outputs saved to {out_dir}")

    if args.exit_on_block and not bool(payload.get("allow_paper_live", False)):
        raise SystemExit(20)
