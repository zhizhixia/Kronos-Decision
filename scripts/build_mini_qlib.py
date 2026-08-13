"""从本地 CSV 缓存构建小型 Qlib 数据目录（集成测试与研究冒烟用）。"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "factor", "vwap"]


def build(cache_dir: str | Path, output: str | Path, codes: list[str] | None = None) -> int:
    """读取缓存 CSV，生成 calendar.txt、instruments 与 day.<field>.bin。"""
    cache = Path(cache_dir)
    target = Path(output)
    if target.exists():
        shutil.rmtree(target)
    (target / "features").mkdir(parents=True, exist_ok=True)
    (target / "calendars").mkdir(parents=True, exist_ok=True)
    (target / "instruments").mkdir(parents=True, exist_ok=True)
    frames = {}
    for csv_path in sorted(cache.glob("*.csv")):
        code = csv_path.stem
        if codes and code not in codes:
            continue
        if code == "pool_hs300":
            continue
        bars = pd.read_csv(csv_path, parse_dates=["date"])
        if len(bars) < 100:
            continue
        bars = bars.set_index("date")[FIELDS[:5]].copy()
        bars["factor"] = 1.0
        bars["vwap"] = bars["close"]
        frames[code] = bars
    if not frames:
        raise RuntimeError("没有可用缓存数据。")
    calendar = sorted(set().union(*[frame.index for frame in frames.values()]))
    calendar = pd.DatetimeIndex(calendar).sort_values()
    (target / "calendars" / "day.txt").write_text("\n".join(day.strftime("%Y-%m-%d") for day in calendar), encoding="utf-8")
    instrument_lines = []
    for code in sorted(frames):
        qlib_code = _to_qlib_code(code)
        frame = frames[code].reindex(calendar)
        features_dir = target / "features" / qlib_code
        features_dir.mkdir(parents=True, exist_ok=True)
        for field in FIELDS:
            values = frame[field].to_numpy(dtype=np.float32)
            payload = np.hstack([np.float32(0), values]).astype("<f")
            (features_dir / f"{field}.day.bin").write_bytes(payload.tobytes())
        instrument_lines.append(f"{qlib_code}\t{calendar[0].strftime('%Y-%m-%d')}\t{calendar[-1].strftime('%Y-%m-%d')}")
    benchmark = _synthetic_benchmark(frames, calendar)
    bench_dir = target / "features" / "SH000300"
    bench_dir.mkdir(parents=True, exist_ok=True)
    for field in ("open", "high", "low", "close", "volume", "factor", "vwap"):
        values = benchmark[field].to_numpy(dtype=np.float32)
        payload = np.hstack([np.float32(0), values]).astype("<f")
        (bench_dir / f"{field}.day.bin").write_bytes(payload.tobytes())
    instrument_lines.append(f"SH000300\t{calendar[0].strftime('%Y-%m-%d')}\t{calendar[-1].strftime('%Y-%m-%d')}")
    (target / "instruments" / "all.txt").write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")
    # csi300.txt 只含真实股票成员，不含基准指数，避免基准被当作股票预测
    stock_lines = [line for line in instrument_lines if not line.startswith("SH000300\t")]
    (target / "instruments" / "csi300.txt").write_text("\n".join(stock_lines) + "\n", encoding="utf-8")
    return len(frames)


def _synthetic_benchmark(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """三只股票等权合成指数（仅用于小型集成测试，非真实沪深300）。"""
    closes = pd.DataFrame({code: frames[code]["close"] for code in sorted(frames)}).reindex(calendar)
    daily = closes.pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    price = 1000.0 * (1 + daily).cumprod()
    benchmark = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0, "factor": 1.0, "vwap": price})
    return benchmark


def _to_qlib_code(code: str) -> str:
    cleaned = str(code).zfill(6)
    market = "SH" if cleaned.startswith(("5", "6", "9")) else "SZ"
    return f"{market}{cleaned}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建小型 Qlib 数据目录")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default="qlib_data/cn_data")
    parser.add_argument("--codes", default=None, help="逗号分隔的股票代码")
    args = parser.parse_args()
    codes = args.codes.split(",") if args.codes else None
    count = build(args.cache_dir, args.output, codes)
    print(f"built {count} stocks -> {args.output}")
