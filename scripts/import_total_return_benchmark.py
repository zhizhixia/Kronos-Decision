"""把经核验的沪深300全收益 CSV 注册到本地 Qlib 数据目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

INSTRUMENT = "CSIH00300"
INDEX_CODE = "H00300"
FEATURES = ("open", "high", "low", "close", "volume", "factor", "vwap")
TRUSTED_SOURCE_HOSTS = ("csindex.com.cn", "eastmoney.com")
AKSHARE_SOURCE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=2.H00300"


def import_total_return(
    csv_path: str | Path,
    provider_uri: str | Path,
    source: str = "manual_csv",
    source_url: str | None = None,
) -> dict[str, str | int]:
    """校验 CSV、写入 Qlib 二进制特征与来源哈希，返回导入摘要。"""
    source_path = Path(csv_path)
    bars = _load_bars(source_path)
    content_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return _register_total_return(bars, provider_uri, source, source_path.name, content_hash, source_url)


def import_from_akshare(provider_uri: str | Path) -> dict[str, str | int]:
    """下载 AkShare 的 csiH00300 并注册为可审计的 Qlib 全收益基准。"""
    raw = _download_akshare()
    bars = _normalize_bars(raw)
    content_hash = hashlib.sha256(raw.to_csv(index=False).encode("utf-8")).hexdigest()
    return _register_total_return(bars, provider_uri, "akshare_csiH00300", "akshare:csiH00300", content_hash, AKSHARE_SOURCE_URL)


def _register_total_return(
    bars: pd.DataFrame,
    provider_uri: str | Path,
    source: str,
    source_file: str,
    content_hash: str,
    source_url: str | None,
) -> dict[str, str | int]:
    """将已规范化的全收益日线注册到同一 Qlib 日历。"""
    provider = Path(provider_uri)
    calendar = _read_calendar(provider)
    aligned = _align_to_calendar(bars, calendar)
    _write_features(provider, aligned, calendar)
    _upsert_instrument(provider, aligned.index.min(), aligned.index.max())
    metadata = {
        "instrument": INSTRUMENT,
        "index_code": INDEX_CODE,
        "is_total_return": True,
        "source": source,
        "source_file": source_file,
        "source_url": source_url,
        "source_retrieved_at": datetime.now().isoformat(),
        "provenance_verified": _is_verifiable_source_url(source_url),
        "coverage_start": aligned.index.min().date().isoformat(),
        "coverage_end": aligned.index.max().date().isoformat(),
        "content_hash": content_hash,
        "imported_at": datetime.now().isoformat(),
    }
    _atomic_json(provider / "benchmarks" / f"{INDEX_CODE}.json", metadata)
    return {"instrument": INSTRUMENT, "rows": len(aligned), "content_hash": content_hash}


def _is_verifiable_source_url(source_url: str | None) -> bool:
    """仅接受可审计的 HTTPS 中证或东方财富 H00300 来源。"""
    try:
        parsed = urlparse(str(source_url))
        host = parsed.hostname or ""
        trusted = any(host == domain or host.endswith(f".{domain}") for domain in TRUSTED_SOURCE_HOSTS)
        return parsed.scheme == "https" and trusted and INDEX_CODE.lower() in str(source_url).lower()
    except (TypeError, ValueError):
        return False


def _read_calendar(provider: Path) -> pd.DatetimeIndex:
    path = provider / "calendars" / "day.txt"
    if not path.exists():
        raise ValueError("Qlib 日历不存在，无法导入全收益基准。")
    calendar = pd.DatetimeIndex(pd.to_datetime(path.read_text(encoding="utf-8").splitlines())).normalize()
    if calendar.empty or not calendar.is_monotonic_increasing:
        raise ValueError("Qlib 日历为空或顺序异常。")
    return calendar


def _load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"全收益 CSV 不存在：{path}")
    bars = pd.read_csv(path, encoding="utf-8-sig")
    return _normalize_bars(bars)


def _download_akshare() -> pd.DataFrame:
    """下载中证沪深300全收益指数；网络或接口失败必须显式报错。"""
    try:
        import akshare as ak
        bars = ak.stock_zh_index_daily_em(symbol="csiH00300", start_date="20050101")
    except Exception as exc:
        raise RuntimeError("AkShare 无法获取 csiH00300；请改用 --csv 导入经核验数据。") from exc
    if bars is None or bars.empty:
        raise RuntimeError("AkShare 未返回 csiH00300；请改用 --csv 导入经核验数据。")
    return bars


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """规范英文或中文列名，并校验全收益日线的 OHLC 数据契约。"""
    aliases = {"日期": "date", "收盘": "close", "收盘价": "close", "开盘": "open", "开盘价": "open", "最高": "high", "最高价": "high", "最低": "low", "最低价": "low"}
    bars = bars.rename(columns={column: aliases.get(str(column).strip(), str(column).strip().lower()) for column in bars.columns})
    if not {"date", "close"}.issubset(bars.columns):
        raise ValueError("全收益 CSV 必须至少包含 date 和 close 列。")
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars.get(column, bars["close"]), errors="coerce")
    clean = bars.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if clean.empty or clean["date"].duplicated().any() or (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("全收益 CSV 存在空值、重复日期或非正价格。")
    if not (clean["high"] >= clean[["open", "close", "low"]].max(axis=1)).all() or not (clean["low"] <= clean[["open", "close", "high"]].min(axis=1)).all():
        raise ValueError("全收益 CSV 存在 OHLC 不变量错误。")
    return clean.set_index("date")[["open", "high", "low", "close"]]


def _align_to_calendar(bars: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    start, end = bars.index.min(), bars.index.max()
    expected = calendar[(calendar >= start) & (calendar <= end)]
    missing = expected.difference(bars.index)
    if expected.empty or not missing.empty:
        preview = ",".join(day.date().isoformat() for day in missing[:3])
        raise ValueError(f"全收益 CSV 与 Qlib 日历不连续，缺失日期：{preview or '覆盖范围为空'}")
    return bars.reindex(expected)


def _write_features(provider: Path, bars: pd.DataFrame, calendar: pd.DatetimeIndex) -> None:
    features_dir = provider / "features" / INSTRUMENT
    features_dir.mkdir(parents=True, exist_ok=True)
    start_index = int(calendar.get_loc(bars.index.min()))
    frame = bars.assign(volume=1.0, factor=1.0, vwap=bars["close"])
    for field in FEATURES:
        values = frame[field].to_numpy(dtype=np.float32)
        payload = np.hstack([np.float32(start_index), values]).astype("<f").tobytes()
        _atomic_bytes(features_dir / f"{field}.day.bin", payload)


def _upsert_instrument(provider: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    path = provider / "instruments" / "all.txt"
    if not path.exists():
        raise ValueError("Qlib instruments/all.txt 不存在，无法注册全收益基准。")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(f"{INSTRUMENT}\t")]
    lines.append(f"{INSTRUMENT}\t{start.date().isoformat()}\t{end.date().isoformat()}")
    _atomic_text(path, "\n".join(lines) + "\n")


def _atomic_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="导入沪深300全收益指数到 Qlib")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="经核验的 H00300 日线 CSV（date,close[,open,high,low]）")
    input_group.add_argument("--download-akshare", action="store_true", help="使用 AkShare 下载 csiH00300")
    parser.add_argument("--provider-uri", default="qlib_data/cn_data")
    parser.add_argument("--source", default="manual_csv", help="数据来源标识，例如 csindex_export")
    parser.add_argument("--source-url", help="手工 CSV 的可审计 HTTPS H00300 来源 URL")
    args = parser.parse_args()
    try:
        if args.csv and not _is_verifiable_source_url(args.source_url):
            raise ValueError("手工 CSV 必须提供受信 HTTPS 且包含 H00300 的 --source-url。")
        result = import_from_akshare(args.provider_uri) if args.download_akshare else import_total_return(args.csv, args.provider_uri, args.source, args.source_url)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
