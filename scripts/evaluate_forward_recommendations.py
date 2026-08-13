"""从本地建议台账和显式价格 CSV 生成独立前瞻观察工件。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.forward_ledger import evaluate_forward_ledger, forward_ledger_report
from portfolio.store import PortfolioStore


def main() -> int:
    parser = argparse.ArgumentParser(description="评估已满20日的本地前瞻建议台账")
    parser.add_argument("--price-csv", required=True, help="宽表价格 CSV：date 列加六码股票代码列")
    parser.add_argument("--database", default="data/user/kronos.db")
    parser.add_argument("--portfolio-id", default="default")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--out-dir", default="artifacts/forward")
    args = parser.parse_args()
    prices = pd.read_csv(args.price_csv)
    observations = evaluate_forward_ledger(PortfolioStore(args.database).list_recommendations(args.portfolio_id, 10000), prices, args.horizon)
    report = forward_ledger_report(observations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    _atomic_csv(out_dir / f"{stamp}.csv", observations)
    _atomic_json(out_dir / f"{stamp}.json", report)
    print(json.dumps({"status": report["status"], "observations": len(observations), "output": str(out_dir / stamp)}, ensure_ascii=False))
    return 0


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        frame.to_csv(handle, index=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


if __name__ == "__main__":
    raise SystemExit(main())
