"""市场数据统一契约与可重放元数据。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


HARD_QUALITY_FLAGS = {
    "CALENDAR_QLIB_OUTDATED",
    "CALENDAR_WEEKDAY_FALLBACK",
}


def has_hard_quality_flags(flags: tuple[str, ...] | list[str]) -> bool:
    """判断质量旗标是否足以阻断正式建议。"""
    normalized = {str(flag) for flag in flags}
    return any(flag.startswith("ERROR_") for flag in normalized) or bool(
        normalized & HARD_QUALITY_FLAGS
    )


@dataclass(frozen=True)
class MarketDataBundle:
    """所有日线数据源都返回的统一数据包。"""

    bars: pd.DataFrame
    source: str
    retrieved_at: datetime
    as_of: pd.Timestamp
    latest_complete_session: pd.Timestamp
    adjustment: str
    calendar_version: str
    universe_version: str | None
    is_stale: bool
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    fallback_chain: tuple[str, ...] = field(default_factory=tuple)
    content_hash: str = ""
    # 绑定不可变 CSV 快照；空值仅保留旧版构造兼容，不代表可审计输入。
    snapshot_id: str = ""

    @property
    def has_hard_quality_error(self) -> bool:
        """是否含有禁止产生正式建议的数据质量错误。"""
        return has_hard_quality_flags(self.quality_flags)


@dataclass(frozen=True)
class InstrumentState:
    """记录证券在一个时点可见的身份、状态和来源。"""

    code: str
    name: str
    security_status: str
    effective_date: pd.Timestamp
    available_at: pd.Timestamp
    listed_date: pd.Timestamp | None = None
    delisted_date: pd.Timestamp | None = None
    source: str = ""
    source_revision: str = ""
    corporate_action_version: str = ""
    exchange: str = ""
    sector: str = ""
    suspended: bool | None = None
    risk_warning: bool | None = None
    rule_version: str = ""

    @property
    def is_status_known(self) -> bool:
        """返回状态是否为明确的非空文本。"""
        return isinstance(self.security_status, str) and bool(self.security_status.strip())

    @property
    def is_active(self) -> bool:
        """返回状态是否明确表示仍在正常交易。"""
        if not isinstance(self.security_status, str):
            return False
        if self.suspended is True or self.risk_warning is True:
            return False
        return self.security_status.strip().upper() in {
            "NORMAL",
            "ACTIVE",
            "TRADING",
            "LISTED",
            "正常",
            "正常交易",
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    """一个股票池在指定时点的可审计快照。"""

    universe: str
    scope: str
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    universe_version: str
    source: str
    instruments: tuple[InstrumentState, ...] = field(default_factory=tuple)
    history_available: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def active_codes(self) -> tuple[str, ...]:
        """只返回有明确正常交易状态的证券代码。"""
        return tuple(item.code for item in self.instruments if item.is_active)

    @property
    def usable(self) -> bool:
        """返回当前快照是否足以支持股票池消费。"""
        return bool(self.instruments) and not self.reasons and all(
            item.is_status_known and item.is_active for item in self.instruments
        )

    @property
    def is_historical(self) -> bool:
        """返回该股票池是否来自明确的历史时点资料。"""
        return self.scope == "history" and self.history_available

    @property
    def fetched_at(self) -> pd.Timestamp:
        """返回获取/可见时间的兼容名称。"""
        return self.available_at



# 方案中的正式名称；旧代码继续使用 MarketDataBundle。
MarketSnapshot = MarketDataBundle


def bars_hash(bars: pd.DataFrame) -> str:
    """计算与行顺序、数值均绑定的稳定内容哈希。"""
    canonical = bars.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
