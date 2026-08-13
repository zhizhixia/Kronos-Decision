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

    @property
    def has_hard_quality_error(self) -> bool:
        """是否含有禁止产生正式建议的数据质量错误。"""
        return has_hard_quality_flags(self.quality_flags)


def bars_hash(bars: pd.DataFrame) -> str:
    """计算与行顺序、数值均绑定的稳定内容哈希。"""
    canonical = bars.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
