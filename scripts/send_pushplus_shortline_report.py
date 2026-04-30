from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send latest shortline opportunity cards to PushPlus.")
    parser.add_argument("--report-dir", default=str(ROOT / "outputs" / "shortline_opportunities"))
    parser.add_argument("--cards-json", default="")
    parser.add_argument("--token", default=os.environ.get("PUSHPLUS_TOKEN", ""))
    parser.add_argument("--title-prefix", default="精选短线机会")
    parser.add_argument("--template", choices=("markdown", "html", "txt"), default="markdown")
    parser.add_argument("--topic", default=os.environ.get("PUSHPLUS_TOPIC", ""))
    parser.add_argument("--channel", default=os.environ.get("PUSHPLUS_CHANNEL", "wechat"))
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _latest_cards_json(report_dir: Path) -> Path:
    candidates = sorted(glob.glob(str(report_dir / "shortline_opportunity_*_cards.json")))
    if not candidates:
        raise FileNotFoundError(f"No shortline cards json found under {report_dir}.")
    return Path(candidates[-1])


def _load_payload(cards_json: Path) -> dict[str, Any]:
    return json.loads(cards_json.read_text(encoding="utf-8"))


def _build_markdown_message(payload: dict[str, Any], title_prefix: str) -> tuple[str, str]:
    summary = payload.get("summary", {})
    cards = payload.get("cards", [])
    selection_date = str(summary.get("selection_date", ""))
    title = f"{title_prefix} {selection_date}".strip()
    lines = [
        f"# {title}",
        "",
        f"- 选股日期：`{selection_date}`",
        f"- 入选数量：`{summary.get('selected_count', len(cards))}`",
        f"- 候选数量：`{summary.get('candidate_count', 0)}`",
        f"- 报告类型：`{summary.get('report_type', 'shortline_opportunity_under20')}`",
        "",
    ]
    for card in cards[:5]:
        lines.extend(
            [
                f"## {card.get('rank', '-')}. {card.get('name', '-')}",
                "",
                f"- 代码：`{card.get('code', '-')}`",
                f"- 评分：`{float(card.get('shortline_score', 0.0)):.4f}`",
                f"- 收盘价：`{float(card.get('close', 0.0)):.2f}`",
                f"- 近一个月连板次数：`{int(card.get('limit_up_count_20d', 0))}`",
                f"- 当前连板高度：`{int(card.get('current_streak', 0))}`",
                f"- 炸板次数：`{int(card.get('open_board_count_20d', 0))}`",
                f"- 换手率：`{float(card.get('turnover_ratio', 0.0)):.2f}%`",
                f"- 封单金额：`{float(card.get('fd_amount', 0.0)) / 100000000.0:.2f}亿`",
                f"- 流通市值：`{float(card.get('float_mv', 0.0)) / 100000000.0:.2f}亿`",
                f"- 近4日走势：`{card.get('recent_4d_path', '-')}`",
                f"- 明日买入区间：`{card.get('buy_range', '-')}`",
                f"- 目标价：`{card.get('target_price', '-')}`",
                f"- 止损线：`{card.get('stop_loss', '-')}`",
                f"- 建议仓位：`{card.get('suggested_position', '-')}`",
                f"- 档位：`{card.get('grade', '-')}` / `{card.get('style', '-')}` / 风险 `{card.get('risk_level', '-')}`",
                f"- 操作逻辑：{card.get('operation_logic', '-')}",
                "",
            ]
        )
    return title, "\n".join(lines).strip() + "\n"


def _post_pushplus(token: str, title: str, content: str, template: str, topic: str, channel: str, timeout_seconds: float) -> dict[str, Any]:
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": template,
        "channel": channel,
    }
    if topic:
        payload["topic"] = topic
    response = httpx.post("https://www.pushplus.plus/send", json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def main() -> None:
    args = build_parser().parse_args()
    cards_json = Path(args.cards_json) if args.cards_json else _latest_cards_json(Path(args.report_dir))
    payload = _load_payload(cards_json)
    title, content = _build_markdown_message(payload, args.title_prefix)

    result = {
        "cards_json": str(cards_json),
        "title": title,
        "template": args.template,
        "channel": args.channel,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        result["preview"] = content[:2000]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    token = str(args.token).strip()
    if not token:
        raise ValueError("PUSHPLUS_TOKEN is required.")
    result["pushplus_result"] = _post_pushplus(
        token=token,
        title=title,
        content=content,
        template=args.template,
        topic=str(args.topic).strip(),
        channel=str(args.channel).strip(),
        timeout_seconds=float(args.timeout_seconds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
