"""正式研究基准解析：全收益指数不可用时必须失败闭合。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

PRICE_INDEX_INSTRUMENT = "SH000300"
TOTAL_RETURN_INSTRUMENTS = ("CSIH00300", "H00300")
METADATA_PATH = Path("benchmarks") / "H00300.json"


@dataclass(frozen=True)
class BenchmarkSpec:
    """一次评估使用的基准及其可审计覆盖范围。"""

    instrument: str
    is_total_return: bool
    source: str
    coverage_start: str | None
    coverage_end: str | None
    reason_codes: tuple[str, ...] = ()
    content_hash: str | None = None
    source_url: str | None = None
    source_retrieved_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """转换为 manifest 可持久化的基准来源信息。"""
        return asdict(self)


def resolve_benchmark(
    provider_uri: str | Path,
    required_start: str,
    required_end: str,
) -> BenchmarkSpec:
    """仅在 Qlib 全收益数据和来源元数据完整时启用 H00300。"""
    provider = Path(provider_uri)
    ranges = _instrument_ranges(provider / "instruments" / "all.txt")
    metadata = _read_metadata(provider / METADATA_PATH)
    for instrument in TOTAL_RETURN_INSTRUMENTS:
        coverage = ranges.get(instrument)
        if coverage and _covers(coverage, required_start, required_end) and _is_total_return(metadata, instrument):
            return BenchmarkSpec(
                instrument, True, str(metadata.get("source", "csv_import")),
                coverage[0], coverage[1], (), str(metadata.get("content_hash") or "") or None,
                str(metadata.get("source_url") or "") or None,
                str(metadata.get("source_retrieved_at") or "") or None,
            )
        if coverage and _covers(coverage, required_start, required_end) and _matches_total_return_identity(metadata, instrument):
            return BenchmarkSpec(PRICE_INDEX_INSTRUMENT, False, "qlib_price_index", None, None, ("TOTAL_RETURN_BENCHMARK_PROVENANCE_UNVERIFIED",))
    reason = "TOTAL_RETURN_BENCHMARK_UNAVAILABLE"
    return BenchmarkSpec(PRICE_INDEX_INSTRUMENT, False, "qlib_price_index", None, None, (reason,))


def _instrument_ranges(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    result: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            result[parts[0]] = (parts[1], parts[2])
    return result


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _covers(coverage: tuple[str, str], required_start: str, required_end: str) -> bool:
    try:
        start, end = (_date(value) for value in coverage)
        return start <= _date(required_start) and end >= _date(required_end)
    except ValueError:
        return False


def _date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _is_total_return(metadata: dict[str, Any], instrument: str) -> bool:
    return _matches_total_return_identity(metadata, instrument) and bool(metadata.get("provenance_verified")) and _has_source_metadata(metadata)


def _matches_total_return_identity(metadata: dict[str, Any], instrument: str) -> bool:
    return bool(metadata.get("is_total_return")) and metadata.get("instrument") == instrument and metadata.get("index_code") == "H00300"


def _has_source_metadata(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("source_url")) and bool(metadata.get("source_retrieved_at"))
