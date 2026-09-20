"""每日建议批量入口：对自选股票生成 v2 五态报告、保存建议历史并输出 CSV。

供 Windows 计划任务每日收盘后调用（在仓库根目录运行）：

    $env:PYTHONPATH = ".\\vendor;."
    python scripts/daily_recommendations.py --stocks "600519,000001"

回补历史锚点时加 --as-of（引擎会截取该日之前的数据并生成对应建议）：

    $env:PYTHONPATH = "."
    python scripts/daily_recommendations.py --stocks "600519,000001" --as-of "2026-08-12"

输出默认写到 artifacts/daily/YYYY-MM-DD.csv，只保存建议，不产生任何真实交易。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_STOCKS = ["600519", "000001", "600036"]


def main() -> int:
    parser = argparse.ArgumentParser(description="每日批量生成五态建议")
    parser.add_argument("--stocks", default=None, help="逗号分隔的股票代码")
    parser.add_argument("--out", default=None, help="CSV 输出路径（默认 artifacts/daily/YYYY-MM-DD.csv）")
    parser.add_argument("--portfolio-id", default="default")
    parser.add_argument("--as-of", dest="as_of", default=None, help="回补历史锚点 YYYY-MM-DD；缺省为最新完整交易日")
    args = parser.parse_args()

    from decision.engine import DecisionEngine
    from portfolio.store import PortfolioStore

    stocks = [code.strip() for code in args.stocks.split(",")] if args.stocks else DEFAULT_STOCKS
    engine = DecisionEngine()
    store = PortfolioStore()
    rows = []
    for code in stocks:
        try:
            report = engine.decision_report_v2(code, args.portfolio_id, as_of=args.as_of)
            recommendation = report["recommendation"]
            eligibility = report.get("eligibility") or {}
            rows.append({
                "stock_code": code,
                "action": recommendation["action"],
                "legacy_signal": recommendation["legacy_signal"],
                "reason_codes": "|".join(recommendation["reason_codes"]),
                "gate_passed": bool(report["evidence_gate"].get("passed")),
                "eligible": bool(eligibility.get("eligible", True)),
                "data_as_of": report["data_provenance"].get("as_of"),
                "generated_at": report["generated_at"],
            })
            try:
                store.save_recommendation(args.portfolio_id, code, recommendation["action"], report["horizons"], "gate-v1" if report["evidence_gate"].get("checks", {}).get("VERSION_MATCH") else "no-evidence", report["generated_at"], report["data_provenance"].get("as_of"), bool(report["evidence_gate"].get("passed")))
            except Exception:
                pass  # 建议历史写入失败不阻断批量任务
        except Exception as exc:
            rows.append({"stock_code": code, "action": "ERROR", "legacy_signal": "HOLD", "reason_codes": f"EXCEPTION:{exc}", "gate_passed": False, "eligible": False, "generated_at": datetime.now().isoformat()})
    frame = pd.DataFrame(rows)
    stamp = args.as_of.replace("-", "") if args.as_of else datetime.now().strftime("%Y-%m-%d")
    out = Path(args.out) if args.out else Path("artifacts/daily") / f"{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=out.parent, suffix=".tmp") as handle:
        frame.to_csv(handle, index=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, out)
    print(json.dumps({"date": datetime.now().strftime("%Y-%m-%d"), "count": int(len(frame)), "output": str(out), "actions": frame["action"].value_counts().to_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
