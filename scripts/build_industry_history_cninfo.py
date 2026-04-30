from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import py_mini_racer
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from akshare.datasets import get_ths_js
from ashare_quant.data.industry_history import (
    build_industry_events_from_frames,
    canonical_code_to_symbol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch CNInfo industry-change history for all codes in the canonical A-share CSV."
    )
    parser.add_argument("--input-path", default=str(ROOT / "data" / "a_share_daily.csv"))
    parser.add_argument("--output-path", default=str(ROOT / "data" / "industry_events_cninfo.csv"))
    parser.add_argument("--start-date", default="19900101")
    parser.add_argument("--end-date", default="20260320")
    parser.add_argument("--industry-level", choices=("门类", "次类", "大类", "中类"), default="大类")
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=1, help="Use >1 to fetch industry history concurrently.")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 to fetch all codes.")
    return parser


def _clear_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def _load_cninfo_js() -> str:
    setting_file_path = get_ths_js("cninfo.js")
    return Path(setting_file_path).read_text(encoding="utf-8")


def _build_cninfo_accept_enckey() -> str:
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(_load_cninfo_js())
    return js_code.call("getResCode1")


def _build_cninfo_headers(mcode: str) -> dict[str, str]:
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Length": "0",
        "Host": "webapi.cninfo.com.cn",
        "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Pragma": "no-cache",
        "Proxy-Connection": "keep-alive",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/93.0.4577.63 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }


def _fetch_cninfo_change(symbol: str, start_date: str, end_date: str, headers: dict[str, str]) -> pd.DataFrame:
    url = "https://webapi.cninfo.com.cn/api/stock/p_stock2110"
    params = {
        "scode": symbol,
        "sdate": "-".join([start_date[:4], start_date[4:6], start_date[6:]]),
        "edate": "-".join([end_date[:4], end_date[4:6], end_date[6:]]),
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    data_json = response.json()
    records = data_json.get("records", [])
    if not records:
        return pd.DataFrame()
    temp_df = pd.DataFrame(records)
    cols_map = {
        "ORGNAME": "机构名称",
        "SECCODE": "证券代码",
        "SECNAME": "新证券简称",
        "VARYDATE": "变更日期",
        "F001V": "分类标准编码",
        "F002V": "分类标准",
        "F003V": "行业编码",
        "F004V": "行业门类",
        "F005V": "行业次类",
        "F006V": "行业大类",
        "F007V": "行业中类",
        "F008C": "最新记录标识",
    }
    temp_df.rename(columns=cols_map, inplace=True)
    temp_df["变更日期"] = pd.to_datetime(temp_df["变更日期"], errors="coerce").dt.date
    if "最新记录标识" in temp_df.columns:
        temp_df = temp_df.drop(columns=["最新记录标识"])
    return temp_df


if __name__ == "__main__":
    args = build_parser().parse_args()
    _clear_proxy_env()

    code_series = pd.read_csv(args.input_path, usecols=["code"], low_memory=False)["code"].dropna().astype(str).drop_duplicates()
    codes = code_series.tolist()
    if args.limit > 0:
        codes = codes[: args.limit]

    mcode = _build_cninfo_accept_enckey()
    headers = _build_cninfo_headers(mcode)

    def fetch_one(canonical_code: str) -> tuple[str, pd.DataFrame | None, str | None]:
        symbol = canonical_code_to_symbol(canonical_code)
        try:
            frame = _fetch_cninfo_change(
                symbol=symbol,
                headers=headers,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            return canonical_code, frame, None
        except Exception as exc:  # noqa: BLE001
            return canonical_code, None, str(exc)

    frames: list[tuple[str, pd.DataFrame]] = []
    errors: list[dict[str, str]] = []
    if args.max_workers <= 1:
        for idx, canonical_code in enumerate(codes, start=1):
            code, frame, error = fetch_one(canonical_code)
            if frame is not None:
                frames.append((code, frame))
            if error is not None:
                errors.append({"code": code, "error": error})
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
            if idx % 100 == 0:
                print(json.dumps({"processed": idx, "total": len(codes), "errors": len(errors)}, ensure_ascii=False))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(fetch_one, code): code for code in codes}
            processed = 0
            for future in as_completed(future_map):
                processed += 1
                code, frame, error = future.result()
                if frame is not None:
                    frames.append((code, frame))
                if error is not None:
                    errors.append({"code": code, "error": error})
                if processed % 100 == 0 or processed == len(codes):
                    print(
                        json.dumps(
                            {"processed": processed, "total": len(codes), "errors": len(errors)},
                            ensure_ascii=False,
                        )
                    )

    event_df, summary = build_industry_events_from_frames(frames=frames, industry_level=args.industry_level)
    summary_payload = {
        **summary.__dict__,
        "codes_total": len(codes),
        "codes_failed": len(errors),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(output_path, index=False)
    if errors:
        (output_path.parent / f"{output_path.stem}_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"[OK] industry events saved to {output_path}")
