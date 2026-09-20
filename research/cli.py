"""研究协议驱动的有界离线评价入口。

CLI 不联网、不下载模型。缺少冻结协议或真实输入文件时直接失败，不能用
fixture 自动冒充样本外实验。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Sequence

import pandas as pd

from research.backtest import BacktestEngine, ExecutionRules
from research.evaluation import evaluate_backtest
from research.protocol import ProtocolError, ResearchProtocol, load_protocol


class ResearchCliError(ValueError):
    """CLI 参数、输入或产物错误。"""


def evaluate_command(args: argparse.Namespace) -> int:
    """执行一次冻结协议绑定的有界离线评价。"""
    protocol = load_protocol(args.protocol)
    if not protocol.is_frozen:
        raise ResearchCliError("研究协议必须先冻结，不能用可变协议运行评价")
    prices = _read_csv(args.prices, "prices")
    signals = _read_csv(args.signals, "signals")
    prices = _bind_single_stock_code(prices, signals)
    requested_as_of = args.as_of or protocol.data_visible_boundary
    effective_as_of = _validate_protocol_inputs(protocol, prices, signals, requested_as_of)
    rules = ExecutionRules(
        version=str(protocol.metadata.get("execution_rules_version", "daily-v1")),
        commission_rate=_cost_value(protocol.costs, ("commission_rate", "commission"), 0.0003),
        stamp_duty_rate=_cost_value(protocol.costs, ("stamp_duty_rate", "stamp_duty"), 0.001),
        slippage_bps=float(protocol.costs.get("slippage_bps", 5.0)),
        lot_size=int(protocol.metadata.get("lot_size", 100)),
        t_plus_one=bool(protocol.metadata.get("t_plus_one", True)),
    )
    result = BacktestEngine(rules).run(prices, signals, float(args.initial_cash), as_of=effective_as_of)
    evaluation = evaluate_backtest(result)
    output = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.version,
        "protocol_hash": protocol.canonical_hash,
        "execution_rules_version": rules.version,
        "as_of": effective_as_of,
        "experiment_status": "COMPLETED" if evaluation.valid else "FAILED",
        "evaluation": {
            "metrics": dict(evaluation.metrics),
            "sample_counts": dict(evaluation.sample_counts),
            "reasons": list(evaluation.reasons),
        },
        "fills": [asdict(fill) | {"execution_date": fill.execution_date.isoformat()} for fill in result.fills],
        "failures": [asdict(fill) | {"execution_date": fill.execution_date.isoformat()} for fill in result.failures],
        "rules": asdict(rules),
    }
    _atomic_json(Path(args.output), output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造研究 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="Kronos 冻结协议离线研究入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="运行有界回测评价")
    evaluate.add_argument("--protocol", required=True, type=Path)
    evaluate.add_argument("--prices", required=True, type=Path)
    evaluate.add_argument("--signals", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--initial-cash", type=float, default=1_000_000.0)
    evaluate.add_argument("--as-of", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行 CLI 并将用户错误转换为非零退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return evaluate_command(args)
        raise ResearchCliError(f"不支持的命令：{args.command}")
    except (OSError, ProtocolError, ResearchCliError, ValueError) as exc:
        parser.error(str(exc))
    return 2


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    """读取研究输入并拒绝目录、空文件和无法解析的 CSV。"""
    if not path.is_file():
        raise ResearchCliError(f"{name} 文件不存在：{path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ResearchCliError(f"无法读取 {name}：{path}") from exc
    if frame.empty:
        raise ResearchCliError(f"{name} 不能为空：{path}")
    return frame


def _bind_single_stock_code(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """为单证券价格文件绑定信号中的唯一代码，拒绝含糊的多证券输入。"""
    if "code" in prices.columns:
        return prices
    if "code" not in signals.columns:
        raise ResearchCliError("prices 缺少 code，signals 也没有可推断的证券代码")
    codes = tuple(dict.fromkeys(str(value).strip() for value in signals["code"].dropna()))
    if len(codes) != 1 or not codes[0]:
        raise ResearchCliError("prices 缺少 code，signals 必须只包含一个证券代码")
    bound = prices.copy()
    bound.insert(1 if "date" in bound.columns else 0, "code", codes[0])
    return bound


def _validate_protocol_inputs(
    protocol: ResearchProtocol,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    as_of: str,
) -> str:
    """把 CSV 输入绑定到冻结协议的时点、窗口、证券和股票池范围。"""
    if "date" not in prices.columns:
        raise ResearchCliError("prices 缺少 date，无法校验测试窗口")
    if "signal_date" not in signals.columns:
        raise ResearchCliError("signals 缺少 signal_date，无法校验测试窗口")
    price_codes = tuple(prices["code"].dropna().astype(str))
    signal_codes = tuple(signals["code"].dropna().astype(str)) if "code" in signals else ()
    if not signal_codes:
        raise ResearchCliError("signals 缺少可校验的证券代码")
    pool_values: list[object] = []
    for frame in (prices, signals):
        for column in ("stock_pool_version", "pool_version"):
            if column in frame.columns:
                pool_values.extend(frame[column].dropna().tolist())
    try:
        return protocol.validate_evaluation_inputs(
            as_of=as_of,
            price_dates=tuple(prices["date"]),
            signal_dates=tuple(signals["signal_date"]),
            security_codes=price_codes + signal_codes,
            stock_pool_versions=tuple(pool_values),
        )
    except ProtocolError as exc:
        raise ResearchCliError(str(exc)) from exc


def _cost_value(costs: object, keys: tuple[str, ...], default: float) -> float:
    """读取兼容旧协议命名的成本字段，不把空值当成零成本。"""
    if not isinstance(costs, Mapping):
        return default
    for key in keys:
        value = costs.get(key)
        if value is not None:
            return float(value)
    return default


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """以同目录临时文件原子写入研究结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ResearchCliError", "build_parser", "evaluate_command", "main"]
