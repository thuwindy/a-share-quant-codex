from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.data.tushare_adapter import TushareDataSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Unknown/missing industry values for the canonical daily A-share CSV "
            "using layered sources (in-file known history -> Tushare stock_basic -> exchange bucket fallback)."
        )
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "a_share_daily_industry.csv"))
    parser.add_argument("--output-path", default=None, help="Defaults to overwriting --data-path.")
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument("--unknown-industry-tokens", default="unknown,unk,其他,other,nan,none,null")
    parser.add_argument("--metadata-source", choices=("tushare", "none"), default="tushare")
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--bypass-system-proxy", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--enable-unclassified-bucket",
        action="store_true",
        help=(
            "Fill still-unresolved rows with synthetic bucket labels such as Legacy_Unclassified_SH. "
            "This is useful for neutralization stability and data-quality control."
        ),
    )
    parser.add_argument("--unclassified-prefix", default="Legacy_Unclassified_")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path. Defaults to outputs/data_quality/industry_backfill_YYYYMMDD.json.",
    )
    return parser


def _unknown_tokens(raw: str) -> set[str]:
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


def _is_unknown(series: pd.Series, unknown_tokens: set[str]) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(unknown_tokens)


def _valid_map_from_frame(frame: pd.DataFrame, unknown_tokens: set[str]) -> dict[str, str]:
    if frame.empty:
        return {}
    work = frame.copy()
    work["code"] = work["code"].astype(str)
    work["industry"] = work["industry"].astype("string").str.strip()
    valid = ~_is_unknown(work["industry"], unknown_tokens)
    work = work.loc[valid, ["code", "industry"]].drop_duplicates(subset=["code"], keep="first")
    return dict(zip(work["code"], work["industry"]))


def _build_known_map(
    *,
    data_path: Path,
    chunk_size: int,
    unknown_tokens: set[str],
) -> dict[str, str]:
    known_map: dict[str, str] = {}
    for chunk in pd.read_csv(data_path, usecols=["code", "industry"], chunksize=chunk_size, low_memory=False):
        mask = ~_is_unknown(chunk["industry"], unknown_tokens)
        if not mask.any():
            continue
        part = chunk.loc[mask, ["code", "industry"]].copy()
        part["code"] = part["code"].astype(str)
        part["industry"] = part["industry"].astype("string").str.strip()
        for code, industry in zip(part["code"], part["industry"]):
            if code not in known_map and isinstance(industry, str) and industry:
                known_map[code] = industry
    return known_map


def _build_tushare_metadata_map(
    *,
    unknown_tokens: set[str],
    http_url: str | None,
    proxy_url: str | None,
    bypass_system_proxy: bool,
    pause_seconds: float,
) -> dict[str, str]:
    source = TushareDataSource(
        http_url=http_url,
        proxy_url=proxy_url,
        bypass_system_proxy=bypass_system_proxy,
        pause_seconds=pause_seconds,
    )
    metadata = source.load_stock_metadata()
    return _valid_map_from_frame(metadata[["code", "industry"]], unknown_tokens)


def _bucket_from_code(series: pd.Series, prefix: str) -> pd.Series:
    suffix = series.astype(str).str.extract(r"\.([A-Za-z]+)$")[0].fillna("UNK").str.upper()
    return prefix + suffix


def _default_report_path() -> Path:
    stamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    return ROOT / "outputs" / "data_quality" / f"industry_backfill_{stamp}.json"


if __name__ == "__main__":
    args = build_parser().parse_args()
    unknown_tokens = _unknown_tokens(args.unknown_industry_tokens)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist.")

    output_path = Path(args.output_path) if args.output_path else data_path
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_path) if args.report_path else _default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] building code-level known industry map from local history ...")
    known_map = _build_known_map(
        data_path=data_path,
        chunk_size=args.chunk_size,
        unknown_tokens=unknown_tokens,
    )
    print(f"[INFO] local known map size: {len(known_map)} codes")

    metadata_map: dict[str, str] = {}
    if args.metadata_source == "tushare":
        print("[INFO] fetching Tushare stock_basic metadata map ...")
        metadata_map = _build_tushare_metadata_map(
            unknown_tokens=unknown_tokens,
            http_url=args.http_url,
            proxy_url=args.proxy_url,
            bypass_system_proxy=args.bypass_system_proxy,
            pause_seconds=args.pause_seconds,
        )
        print(f"[INFO] Tushare metadata map size: {len(metadata_map)} codes")

    rows_total = 0
    unknown_before = 0
    unknown_after = 0
    source_counts = {
        "already_known": 0,
        "fill_from_local_known": 0,
        "fill_from_tushare_stock_basic": 0,
        "fill_from_unclassified_bucket": 0,
        "still_unknown": 0,
    }
    unresolved_codes: set[str] = set()

    wrote_any = False
    chunk_idx = 0
    for chunk in pd.read_csv(data_path, chunksize=args.chunk_size, low_memory=False):
        chunk_idx += 1
        if "industry" not in chunk.columns or "code" not in chunk.columns:
            raise ValueError("Input CSV must contain `code` and `industry` columns.")

        code_series = chunk["code"].astype(str)
        before_mask = _is_unknown(chunk["industry"], unknown_tokens)
        unknown_before += int(before_mask.sum())
        rows_total += len(chunk)

        source_col = pd.Series("already_known", index=chunk.index, dtype="string")
        source_col.loc[before_mask] = "still_unknown"

        if before_mask.any():
            local_fill = code_series.map(known_map)
            local_mask = before_mask & local_fill.notna()
            if local_mask.any():
                chunk.loc[local_mask, "industry"] = local_fill.loc[local_mask]
                source_col.loc[local_mask] = "fill_from_local_known"

        mid_mask = _is_unknown(chunk["industry"], unknown_tokens)
        if mid_mask.any() and metadata_map:
            metadata_fill = code_series.map(metadata_map)
            metadata_mask = mid_mask & metadata_fill.notna()
            if metadata_mask.any():
                chunk.loc[metadata_mask, "industry"] = metadata_fill.loc[metadata_mask]
                source_col.loc[metadata_mask] = "fill_from_tushare_stock_basic"

        end_mask = _is_unknown(chunk["industry"], unknown_tokens)
        if end_mask.any() and args.enable_unclassified_bucket:
            bucket = _bucket_from_code(code_series, args.unclassified_prefix)
            chunk.loc[end_mask, "industry"] = bucket.loc[end_mask]
            source_col.loc[end_mask] = "fill_from_unclassified_bucket"

        final_mask = _is_unknown(chunk["industry"], unknown_tokens)
        unknown_after += int(final_mask.sum())
        if final_mask.any():
            unresolved_codes.update(code_series.loc[final_mask].unique().tolist())
            source_col.loc[final_mask] = "still_unknown"

        for key in source_counts:
            source_counts[key] += int((source_col == key).sum())

        chunk.to_csv(temp_path, index=False, mode="a", header=not wrote_any)
        wrote_any = True

        if chunk_idx % 20 == 0:
            print(
                json.dumps(
                    {
                        "chunk_idx": chunk_idx,
                        "rows_processed": rows_total,
                        "unknown_before_so_far": unknown_before,
                        "unknown_after_so_far": unknown_after,
                    },
                    ensure_ascii=False,
                )
            )

    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)

    report = {
        "data_path": str(data_path.resolve()),
        "output_path": str(output_path.resolve()),
        "rows_total": int(rows_total),
        "unknown_before": int(unknown_before),
        "unknown_after": int(unknown_after),
        "unknown_ratio_before": float(unknown_before / rows_total) if rows_total else 0.0,
        "unknown_ratio_after": float(unknown_after / rows_total) if rows_total else 0.0,
        "known_map_codes": int(len(known_map)),
        "metadata_source": args.metadata_source,
        "metadata_map_codes": int(len(metadata_map)),
        "enable_unclassified_bucket": bool(args.enable_unclassified_bucket),
        "source_counts": source_counts,
        "unresolved_codes_count": int(len(unresolved_codes)),
        "unresolved_codes_sample": sorted(unresolved_codes)[:50],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] industry backfill completed -> {output_path}")
    print(f"[OK] report saved -> {report_path}")
