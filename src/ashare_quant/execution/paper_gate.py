from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperGateDecision:
    selection_date: str
    allow_paper_live: bool
    risk_status: str
    block_statuses: list[str]
    block_reasons: list[str]
    risk_json_path: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def latest_risk_gate_json(risk_dir: str | Path) -> Path:
    files = sorted(Path(risk_dir).glob("risk_gate_*.json"))
    if not files:
        raise FileNotFoundError(f"No risk gate JSON found under {risk_dir}.")
    return files[-1]


def load_risk_gate_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_paper_gate(
    summary: dict[str, Any],
    *,
    risk_json_path: str | Path,
    block_statuses: list[str] | tuple[str, ...] = ("BLOCK",),
) -> PaperGateDecision:
    normalized_blocks = [str(s).upper() for s in block_statuses]
    status = str(summary.get("status", "UNKNOWN")).upper()
    checks = list(summary.get("checks", []))
    failed_block_checks = [
        str(item.get("name", ""))
        for item in checks
        if (not bool(item.get("ok", False))) and str(item.get("severity", "")).lower() == "block"
    ]
    allow = status not in normalized_blocks
    note = (
        "paper_live_allowed"
        if allow
        else "paper_live_blocked_by_risk_governor"
    )
    return PaperGateDecision(
        selection_date=str(summary.get("selection_date", "")),
        allow_paper_live=bool(allow),
        risk_status=status,
        block_statuses=normalized_blocks,
        block_reasons=failed_block_checks,
        risk_json_path=str(risk_json_path),
        note=note,
    )

