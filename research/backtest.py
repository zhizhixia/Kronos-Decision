"""可独立核对的日线回测执行规则。

本模块只接受显式行情和信号，不取数、不调用模型、不访问经纪商。信号在
收盘后产生，最早只能在下一交易日开盘执行；成交量、停牌和涨跌停等未知
字段不会被默认为可成交。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Iterable, Mapping, Sequence

import pandas as pd


class ExecutionRuleError(ValueError):
    """行情、信号或执行规则违反可审计约束。"""


@dataclass(frozen=True)
class ExecutionRules:
    """日线执行规则的冻结版本。"""

    version: str = "daily-v1"
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001
    slippage_bps: float = 5.0
    lot_size: int = 100
    t_plus_one: bool = True
    allow_corporate_actions: bool = False
    money_precision: int = 8

    def __post_init__(self) -> None:
        for name in ("commission_rate", "stamp_duty_rate", "slippage_bps"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ExecutionRuleError(f"{name} 必须是非负有限数")
        if isinstance(self.lot_size, bool) or self.lot_size <= 0:
            raise ExecutionRuleError("lot_size 必须是正整数")
        if isinstance(self.money_precision, bool) or not 0 <= self.money_precision <= 12:
            raise ExecutionRuleError("money_precision 必须在 0 到 12 之间")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ExecutionRuleError("执行规则版本不能为空")


@dataclass(frozen=True)
class TradeFill:
    """一笔实际或失败的模拟成交。"""

    execution_date: pd.Timestamp
    code: str
    side: str
    requested_shares: int
    filled_shares: int
    price: float | None
    gross_value: float
    fees: float
    status: str
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    """回测结果及其全部成交记录。"""

    equity_curve: pd.DataFrame
    fills: tuple[TradeFill, ...]
    failures: tuple[TradeFill, ...]
    final_cash: float
    final_positions: Mapping[str, int]
    rules_version: str


class BacktestEngine:
    """执行不使用未来价格、未来成交量的有界日线回测。"""

    def __init__(self, rules: ExecutionRules | None = None) -> None:
        self.rules = rules or ExecutionRules()

    def run(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float,
        *,
        as_of: pd.Timestamp | str | None = None,
    ) -> BacktestResult:
        """按下一交易日开盘执行信号并返回完整账本。"""
        if not isfinite(float(initial_cash)) or float(initial_cash) <= 0:
            raise ExecutionRuleError("initial_cash 必须是正有限数")
        prepared_bars = self._prepare_bars(bars, as_of)
        prepared_signals = self._prepare_signals(signals, prepared_bars)
        return self._simulate(prepared_bars, prepared_signals, float(initial_cash))

    def _prepare_bars(
        self, bars: pd.DataFrame, as_of: pd.Timestamp | str | None
    ) -> pd.DataFrame:
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            raise ExecutionRuleError("bars 不能为空")
        required = {"date", "code", "open", "close"}
        missing = required.difference(bars.columns)
        if missing:
            raise ExecutionRuleError(f"bars 缺少字段：{sorted(missing)}")
        frame = bars.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ExecutionRuleError("bars.date 含无效日期")
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
            frame = frame.loc[frame["date"] <= cutoff]
        frame["code"] = frame["code"].astype(str)
        for column in ("open", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any() or (frame[column] <= 0).any():
                raise ExecutionRuleError(f"bars.{column} 含缺失或非正价格")
        execution_volume_columns = [
            column for column in ("execution_volume", "open_volume") if column in frame.columns
        ]
        if not execution_volume_columns:
            raise ExecutionRuleError(
                "缺少执行时点成交量（execution_volume/open_volume），不能判断开盘可成交数量"
            )
        volume_column = execution_volume_columns[0]
        if len(execution_volume_columns) == 2:
            first = pd.to_numeric(frame[execution_volume_columns[0]], errors="coerce")
            second = pd.to_numeric(frame[execution_volume_columns[1]], errors="coerce")
            if not first.equals(second):
                raise ExecutionRuleError("execution_volume 与 open_volume 不一致")
        frame["_volume"] = pd.to_numeric(frame[volume_column], errors="coerce")
        if frame["_volume"].isna().any() or (frame["_volume"] < 0).any():
            raise ExecutionRuleError("执行时点成交量含缺失或负值")
        for column in ("can_trade", "suspended", "limit_up", "limit_down"):
            if column not in frame.columns:
                frame[column] = pd.NA
            else:
                frame[column] = frame[column].map(self._normalise_status_value)
        frame = frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        if frame.empty:
            raise ExecutionRuleError("as_of 之前没有可用行情")
        self._reject_unsupported_actions(frame)
        return frame

    def _prepare_signals(self, signals: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(signals, pd.DataFrame) or signals.empty:
            return pd.DataFrame(columns=["signal_date", "code", "action", "target_shares", "target_weight"])
        required = {"signal_date", "code", "action"}
        missing = required.difference(signals.columns)
        if missing:
            raise ExecutionRuleError(f"signals 缺少字段：{sorted(missing)}")
        frame = signals.copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        if frame["signal_date"].isna().any():
            raise ExecutionRuleError("signals.signal_date 含无效日期")
        frame["code"] = frame["code"].astype(str)
        frame["action"] = frame["action"].astype(str).str.upper()
        if not frame["action"].isin({"BUY", "SELL", "HOLD"}).all():
            raise ExecutionRuleError("signals.action 只能是 BUY/SELL/HOLD")
        if "target_shares" not in frame.columns:
            frame["target_shares"] = pd.NA
        if "target_weight" not in frame.columns:
            frame["target_weight"] = pd.NA
        for column in ("target_shares", "target_weight"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (frame["target_shares"].dropna() < 0).any():
            raise ExecutionRuleError("target_shares 不能为负")
        if ((frame["target_weight"].dropna() < 0) | (frame["target_weight"].dropna() > 1)).any():
            raise ExecutionRuleError("target_weight 必须在 0 到 1 之间")
        if not frame["code"].isin(set(bars["code"])).all():
            raise ExecutionRuleError("signals 含不在行情中的证券")
        return frame.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)

    def _simulate(
        self, bars: pd.DataFrame, signals: pd.DataFrame, initial_cash: float
    ) -> BacktestResult:
        cash = initial_cash
        positions: dict[str, int] = {}
        lots: dict[str, list[tuple[pd.Timestamp, int]]] = {}
        fills: list[TradeFill] = []
        failures: list[TradeFill] = []
        bars_by_code = {code: group.reset_index(drop=True) for code, group in bars.groupby("code")}
        orders, scheduling_failures = self._scheduled_orders(signals, bars_by_code)
        equity_rows: list[dict[str, object]] = []
        for date_value in sorted(bars["date"].unique()):
            date = pd.Timestamp(date_value)
            for order in orders.get(date, []):
                fill, cash = self._execute_order(order, date, bars_by_code, positions, lots, cash)
                (fills if fill.status in {"FILLED", "PARTIAL"} else failures).append(fill)
            market_value = 0.0
            for code, group in bars_by_code.items():
                row = group.loc[group["date"] == date]
                if row.empty:
                    if positions.get(code, 0) > 0:
                        raise ExecutionRuleError(
                            f"持仓 {code} 在 {date.date()} 缺少行情，回测不可评价"
                        )
                    continue
                market_value += positions.get(code, 0) * float(row.iloc[-1]["close"])
            equity_rows.append(
                {"date": date, "cash": cash, "market_value": market_value, "equity": cash + market_value}
            )
        equity_curve = pd.DataFrame(equity_rows)
        return BacktestResult(
            equity_curve=equity_curve,
            fills=tuple(fills),
            failures=tuple(scheduling_failures + failures),
            final_cash=cash,
            final_positions=dict(positions),
            rules_version=self.rules.version,
        )

    def _scheduled_orders(
        self, signals: pd.DataFrame, bars_by_code: Mapping[str, pd.DataFrame]
    ) -> tuple[dict[pd.Timestamp, list[dict[str, object]]], list[TradeFill]]:
        scheduled: dict[pd.Timestamp, list[dict[str, object]]] = {}
        failures: list[TradeFill] = []
        for signal in signals.itertuples(index=False):
            code = str(signal.code)
            group = bars_by_code[code]
            candidates = group.loc[group["date"] > signal.signal_date, "date"]
            if candidates.empty:
                failures.append(
                    self._failed_fill(
                        pd.Timestamp(signal.signal_date),
                        code,
                        str(signal.action),
                        self._signal_requested_shares(signal),
                        "NO_NEXT_BAR",
                    )
                )
                continue
            execution_date = pd.Timestamp(candidates.iloc[0])
            scheduled.setdefault(execution_date, []).append(
                {
                    "code": code,
                    "action": str(signal.action),
                    "target_shares": signal.target_shares,
                    "target_weight": signal.target_weight,
                }
            )
        return scheduled, failures

    def _signal_requested_shares(self, signal: object) -> int:
        """估算排程失败记录中的原始申报数量。"""
        target = getattr(signal, "target_shares", None)
        if target is None or pd.isna(target):
            return self.rules.lot_size
        return max(int(float(target)), 0)

    def _execute_order(
        self,
        order: Mapping[str, object],
        date: pd.Timestamp,
        bars_by_code: Mapping[str, pd.DataFrame],
        positions: dict[str, int],
        lots: dict[str, list[tuple[pd.Timestamp, int]]],
        cash: float,
    ) -> tuple[TradeFill, float]:
        code = str(order["code"])
        row = bars_by_code[code].loc[bars_by_code[code]["date"] == date]
        if row.empty:
            return self._failed_fill(date, code, str(order["action"]), 0, "NO_EXECUTION_BAR"), cash
        bar = row.iloc[0]
        side = str(order["action"])
        requested = self._requested_shares(order, bar, positions.get(code, 0), cash)
        original_requested = requested
        if side == "HOLD" or requested <= 0:
            return self._failed_fill(date, code, side, max(requested, 0), "NO_ORDER"), cash
        reason = self._trade_block_reason(bar, side)
        if reason:
            return self._failed_fill(date, code, side, requested, reason), cash
        if side == "SELL":
            requested = min(requested, self._sellable_shares(code, date, positions, lots))
            if requested <= 0:
                return self._failed_fill(date, code, side, original_requested, "T_PLUS_ONE_OR_EMPTY"), cash
        available = int(floor(float(bar["_volume"]) / self.rules.lot_size) * self.rules.lot_size)
        requested = min(requested, available)
        if requested <= 0:
            return self._failed_fill(date, code, side, requested, "NO_EXECUTABLE_VOLUME"), cash
        price = self._execution_price(float(bar["open"]), side)
        gross = price * requested
        fees = self._fees(gross, side)
        if side == "BUY":
            affordable = self._affordable_shares(cash, price)
            requested = min(requested, affordable)
            if requested <= 0:
                return self._failed_fill(date, code, side, requested, "INSUFFICIENT_CASH"), cash
            gross = price * requested
            fees = self._fees(gross, side)
            cash = round(cash - gross - fees, self.rules.money_precision)
            positions[code] = positions.get(code, 0) + requested
            lots.setdefault(code, []).append((date, requested))
        else:
            cash = round(cash + gross - fees, self.rules.money_precision)
            positions[code] = positions.get(code, 0) - requested
            self._consume_lots(code, date, requested, lots)
        status = "FILLED" if requested == original_requested else "PARTIAL"
        fill = TradeFill(date, code, side, original_requested, requested, price, gross, fees, status, "OK")
        return fill, cash

    def _requested_shares(
        self, order: Mapping[str, object], bar: pd.Series, current: int, cash: float
    ) -> int:
        target_shares = order.get("target_shares")
        target_weight = order.get("target_weight")
        if target_shares is not None and pd.notna(target_shares):
            target = int(floor(float(target_shares) / self.rules.lot_size) * self.rules.lot_size)
            return max(target - current, 0) if order["action"] == "BUY" else max(current - target, 0)
        if target_weight is None or pd.isna(target_weight):
            return self.rules.lot_size
        execution_price = self._execution_price(float(bar["open"]), str(order["action"]))
        estimated_equity = cash + current * execution_price
        target = int(
            floor(
                (estimated_equity * float(target_weight))
                / execution_price
                / self.rules.lot_size
            )
            * self.rules.lot_size
        )
        return max(target - current, 0) if order["action"] == "BUY" else max(current - target, 0)

    def _trade_block_reason(self, bar: pd.Series, side: str) -> str:
        flags = {
            column: self._known_status_value(bar.get(column, pd.NA))
            for column in ("can_trade", "suspended", "limit_up", "limit_down")
        }
        if any(value is None for value in flags.values()):
            return "UNKNOWN_TRADING_STATUS"
        if flags["can_trade"] is False or flags["suspended"] is True:
            return "SUSPENDED_OR_UNKNOWN"
        if side == "BUY" and flags["limit_up"] is True:
            return "LIMIT_UP"
        if side == "SELL" and flags["limit_down"] is True:
            return "LIMIT_DOWN"
        return ""

    @staticmethod
    def _normalise_status_value(value: object) -> object:
        """把交易状态规范为布尔值；缺失或未知值保留为不可交易。"""
        if value is None:
            return pd.NA
        try:
            if bool(pd.isna(value)):
                return pd.NA
        except (TypeError, ValueError):
            return pd.NA
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return pd.NA

    @staticmethod
    def _known_status_value(value: object) -> bool | None:
        """读取已规范化状态；未知值返回 None 而不是默认放行。"""
        if value is pd.NA or value is None:
            return None
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            return None
        if isinstance(value, bool):
            return value
        if type(value).__name__ in {"bool", "bool_"}:
            return bool(value)
        return None

    def _sellable_shares(self, code: str, date: pd.Timestamp, positions: Mapping[str, int], lots: Mapping[str, list[tuple[pd.Timestamp, int]]]) -> int:
        if not self.rules.t_plus_one:
            return int(positions.get(code, 0))
        return sum(shares for buy_date, shares in lots.get(code, []) if buy_date < date)

    def _consume_lots(self, code: str, date: pd.Timestamp, shares: int, lots: dict[str, list[tuple[pd.Timestamp, int]]]) -> None:
        remaining = shares
        updated: list[tuple[pd.Timestamp, int]] = []
        for buy_date, lot_shares in lots.get(code, []):
            if buy_date >= date or remaining <= 0:
                updated.append((buy_date, lot_shares))
                continue
            used = min(lot_shares, remaining)
            remaining -= used
            if lot_shares > used:
                updated.append((buy_date, lot_shares - used))
        lots[code] = updated

    def _affordable_shares(self, cash: float, price: float) -> int:
        estimate = int(floor(cash / (price * (1 + self.rules.commission_rate)) / self.rules.lot_size) * self.rules.lot_size)
        while estimate > 0 and price * estimate + self._fees(price * estimate, "BUY") > cash:
            estimate -= self.rules.lot_size
        return max(estimate, 0)

    def _execution_price(self, open_price: float, side: str) -> float:
        slippage = self.rules.slippage_bps / 10000
        return open_price * (1 + slippage if side == "BUY" else 1 - slippage)

    def _fees(self, gross: float, side: str) -> float:
        stamp = self.rules.stamp_duty_rate if side == "SELL" else 0.0
        return round(gross * (self.rules.commission_rate + stamp), self.rules.money_precision)

    def _failed_fill(self, date: pd.Timestamp, code: str, side: str, requested: int, reason: str) -> TradeFill:
        return TradeFill(date, code, side, int(requested), 0, None, 0.0, 0.0, "FAILED", reason)

    def _reject_unsupported_actions(self, bars: pd.DataFrame) -> None:
        for column in ("dividend", "split_factor"):
            if column not in bars.columns:
                continue
            values = pd.to_numeric(bars[column], errors="coerce")
            if values.isna().any():
                raise ExecutionRuleError(f"{column} 含缺失，不能处理公司行动")
            neutral = values == (1 if column == "split_factor" else 0)
            if not neutral.all():
                raise ExecutionRuleError(f"暂不支持非中性公司行动：{column}")


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_cash: float,
    rules: ExecutionRules | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """函数式兼容入口，仍使用同一套冻结执行规则。"""
    return BacktestEngine(rules).run(bars, signals, initial_cash=initial_cash, as_of=as_of)


__all__ = ["BacktestEngine", "BacktestResult", "ExecutionRuleError", "ExecutionRules", "TradeFill", "run_backtest"]
