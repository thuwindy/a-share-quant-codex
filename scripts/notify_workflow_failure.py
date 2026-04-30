from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import urllib.request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit workflow failure notification event (T44 placeholder; optional webhook)."
    )
    parser.add_argument("--workflow-name", default="post_close_daily_agents")
    parser.add_argument("--task-id", default="unknown")
    parser.add_argument("--exit-code", type=int, default=1)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--error-message", default="")
    parser.add_argument("--output-dir", default="outputs/ops_logs/alerts")
    parser.add_argument("--output-prefix", default="failure_event")
    parser.add_argument("--webhook-url", default="")
    parser.add_argument("--channel", choices=("generic", "wecom", "feishu"), default="generic")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser


def _build_generic_payload(event: dict) -> dict:
    return event


def _build_wecom_payload(event: dict) -> dict:
    content = (
        f"## Workflow Failure\n"
        f"> workflow: `{event['workflow_name']}`\n"
        f"> task: `{event['task_id']}`\n"
        f"> exit_code: `{event['exit_code']}`\n"
        f"> host: `{event['host']}`\n"
        f"> log_path: `{event['log_path']}`\n"
        f"> error: `{event['error_message']}`"
    )
    return {"msgtype": "markdown", "markdown": {"content": content}}


def _build_feishu_payload(event: dict) -> dict:
    text = (
        f"Workflow Failure\n"
        f"workflow={event['workflow_name']}\n"
        f"task={event['task_id']}\n"
        f"exit_code={event['exit_code']}\n"
        f"host={event['host']}\n"
        f"log_path={event['log_path']}\n"
        f"error={event['error_message']}"
    )
    return {"msg_type": "text", "content": {"text": text}}


def _render_payload(event: dict, channel: str) -> dict:
    channel_key = str(channel or "generic").lower()
    if channel_key == "wecom":
        return _build_wecom_payload(event)
    if channel_key == "feishu":
        return _build_feishu_payload(event)
    return _build_generic_payload(event)


def _post_webhook(url: str, payload: dict, timeout_seconds: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=max(timeout_seconds, 0.1)) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return {"ok": True, "status_code": int(resp.status), "response_body_preview": body[:500]}


if __name__ == "__main__":
    args = build_parser().parse_args()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "event_type": "workflow_failure",
        "workflow_name": str(args.workflow_name),
        "task_id": str(args.task_id),
        "exit_code": int(args.exit_code),
        "error_message": str(args.error_message),
        "log_path": str(args.log_path),
        "host": socket.gethostname(),
        "created_at": timestamp,
    }

    webhook_url = str(args.webhook_url).strip()
    rendered_payload = _render_payload(event, args.channel)
    if webhook_url:
        try:
            event["webhook_result"] = _post_webhook(webhook_url, rendered_payload, timeout_seconds=float(args.timeout_seconds))
        except Exception as exc:  # pragma: no cover - best-effort notify path
            event["webhook_result"] = {"ok": False, "error": str(exc)}
    else:
        event["webhook_result"] = {"ok": False, "error": "webhook_url_not_configured"}
    event["channel"] = str(args.channel)
    event["rendered_payload"] = rendered_payload

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.output_prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Workflow Failure Notification",
                "",
                f"- workflow_name: `{event['workflow_name']}`",
                f"- task_id: `{event['task_id']}`",
                f"- exit_code: `{event['exit_code']}`",
                f"- created_at: `{event['created_at']}`",
                f"- host: `{event['host']}`",
                f"- log_path: `{event['log_path']}`",
                f"- error_message: `{event['error_message']}`",
                f"- webhook_result: `{json.dumps(event['webhook_result'], ensure_ascii=False)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "event_type": event["event_type"],
                "workflow_name": event["workflow_name"],
                "task_id": event["task_id"],
                "exit_code": event["exit_code"],
                "json_path": str(json_path),
                "md_path": str(md_path),
                "webhook_result": event["webhook_result"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] failure notification event written to {out_dir}")
