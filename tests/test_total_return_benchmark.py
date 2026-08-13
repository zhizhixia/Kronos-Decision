"""沪深300全收益基准注册与正式协议数据依赖测试。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluation.benchmark import PRICE_INDEX_INSTRUMENT, resolve_benchmark
from scripts.import_total_return_benchmark import INSTRUMENT, import_from_akshare, import_total_return


def _provider(tmp_path) -> tuple:
    provider = tmp_path / "cn_data"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    dates = pd.bdate_range("2024-01-02", periods=4)
    (provider / "calendars" / "day.txt").write_text("\n".join(day.date().isoformat() for day in dates), encoding="utf-8")
    (provider / "instruments" / "all.txt").write_text("SH000300\t2024-01-02\t2024-01-05\n", encoding="utf-8")
    return provider, dates


def _csv(path, dates: pd.DatetimeIndex, drop_last: bool = False) -> None:
    values = pd.DataFrame({"date": dates[:-1] if drop_last else dates, "close": range(1000, 1000 + len(dates) - int(drop_last))})
    values.to_csv(path, index=False)


def test_price_benchmark_remains_failure_closed_without_registered_total_return(tmp_path) -> None:
    provider, _ = _provider(tmp_path)

    spec = resolve_benchmark(provider, "2024-01-02", "2024-01-05")

    assert spec.instrument == PRICE_INDEX_INSTRUMENT
    assert not spec.is_total_return
    assert spec.reason_codes == ("TOTAL_RETURN_BENCHMARK_UNAVAILABLE",)


def test_imported_total_return_csv_registers_auditable_qlib_benchmark(tmp_path) -> None:
    provider, dates = _provider(tmp_path)
    source = tmp_path / "h00300.csv"
    _csv(source, dates)

    result = import_total_return(source, provider, source="csindex_export", source_url="https://www.csindex.com.cn/indices/H00300")
    spec = resolve_benchmark(provider, "2024-01-02", "2024-01-05")
    metadata = json.loads((provider / "benchmarks" / "H00300.json").read_text(encoding="utf-8"))

    assert result["instrument"] == INSTRUMENT
    assert result["rows"] == 4
    assert spec.instrument == INSTRUMENT and spec.is_total_return
    assert spec.source == "csindex_export"
    assert spec.content_hash == metadata["content_hash"]
    assert spec.source_url == "https://www.csindex.com.cn/indices/H00300"
    assert metadata["provenance_verified"] is True
    assert (provider / "features" / INSTRUMENT / "close.day.bin").exists()
    assert not resolve_benchmark(provider, "2011-01-01", "2024-01-05").is_total_return


def test_akshare_import_uses_same_auditable_registration(tmp_path, monkeypatch) -> None:
    provider, dates = _provider(tmp_path)
    bars = pd.DataFrame({"date": dates, "open": [1000, 1001, 1002, 1003], "high": [1000, 1001, 1002, 1003], "low": [1000, 1001, 1002, 1003], "close": [1000, 1001, 1002, 1003]})
    monkeypatch.setattr("scripts.import_total_return_benchmark._download_akshare", lambda: bars)

    import_from_akshare(provider)
    spec = resolve_benchmark(provider, "2024-01-02", "2024-01-05")

    assert spec.is_total_return
    assert spec.source == "akshare_csiH00300"
    assert spec.source_url == "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=2.H00300"


def test_import_rejects_total_return_csv_with_calendar_gap(tmp_path) -> None:
    provider, dates = _provider(tmp_path)
    source = tmp_path / "h00300-gap.csv"
    pd.DataFrame({"date": [dates[0], dates[2], dates[3]], "close": [1000, 1002, 1003]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="不连续"):
        import_total_return(source, provider)


def test_unverified_manual_csv_cannot_enable_formal_total_return_benchmark(tmp_path) -> None:
    provider, dates = _provider(tmp_path)
    source = tmp_path / "h00300.csv"
    _csv(source, dates)

    import_total_return(source, provider, source="arbitrary_label")
    spec = resolve_benchmark(provider, "2024-01-02", "2024-01-05")

    assert spec.instrument == PRICE_INDEX_INSTRUMENT
    assert not spec.is_total_return
    assert spec.reason_codes == ("TOTAL_RETURN_BENCHMARK_PROVENANCE_UNVERIFIED",)


def test_untrusted_https_source_cannot_enable_formal_total_return_benchmark(tmp_path) -> None:
    provider, dates = _provider(tmp_path)
    source = tmp_path / "h00300.csv"
    _csv(source, dates)

    import_total_return(source, provider, source_url="https://example.com/H00300.csv")
    spec = resolve_benchmark(provider, "2024-01-02", "2024-01-05")

    assert not spec.is_total_return
    assert spec.reason_codes == ("TOTAL_RETURN_BENCHMARK_PROVENANCE_UNVERIFIED",)
