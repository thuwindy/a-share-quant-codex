from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]


def latest_match(patterns: list[str]) -> Path | None:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern))
    if not found:
        return None

    def _score(path: Path) -> tuple[str, str]:
        match = re.search(r"(20\d{6})", path.name)
        return (match.group(1) if match else "00000000", str(path))

    return sorted(found, key=_score)[-1]


def match_date_path(patterns: list[str], date_key: str) -> Path | None:
    if not date_key:
        return None
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(item) for item in glob.glob(pattern))
    matched = [path for path in found if date_key in path.name]
    return sorted(matched)[-1] if matched else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def path_date_key(path: Path | None) -> str:
    if path is None:
        return ""
    match = re.search(r"(20\d{6})", path.name)
    return match.group(1) if match else ""


def payload_date_key(payload: dict[str, Any], path: Path | None) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    raw = str(summary.get("selection_date") or payload.get("selection_date") or "").strip()
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", raw):
        return raw.replace("-", "")
    return path_date_key(path)


def latest_master_date(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        chunk = 4096
        data = b""
        pos = size
        while pos > 0 and b"\n" not in data:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
    lines = [line for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].split(",", 1)[0].strip()


def tail_text(path: Path, *, max_bytes: int = 80_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="ignore")


def normalize_chat_endpoint(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("missing LLM base url")
    if base.endswith("/chat/completions"):
        return base
    if base == "https://api.deepseek.com" or base == "https://api.deepseek.com/":
        return f"{base.rstrip('/')}/chat/completions"
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def resolve_llm_settings(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    return {
        "api_key": api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        "base_url": base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "",
        "model": model or os.environ.get("LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
    }


def call_openai_compatible_chat(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    timeout_seconds: float = 30.0,
    retries: int | None = None,
    retry_sleep_seconds: float = 2.0,
) -> str:
    if not api_key:
        raise ValueError("missing LLM api key; set LLM_API_KEY or DEEPSEEK_API_KEY")
    endpoint = normalize_chat_endpoint(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "api.deepseek.com" in endpoint and os.environ.get("LLM_THINKING", "disabled").lower() != "enabled":
        payload["thinking"] = {"type": "disabled"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    max_retries = int(os.environ.get("LLM_RETRIES", "1")) if retries is None else max(0, int(retries))
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(max(0.0, retry_sleep_seconds))
    else:
        raise last_error or ValueError("LLM request failed")
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response has no choices")
    content = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("LLM response content is empty")
    return content


def compact_json(data: Any, *, limit: int = 20_000) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
