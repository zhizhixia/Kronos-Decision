"""研究 CLI 的冻结协议和产物边界。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from research.cli import main
from tests.test_research_protocol import _make_protocol


def test_evaluate_cli_requires_frozen_protocol_and_writes_bound_result(tmp_path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _make_protocol().freeze().save(protocol_path)
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "code": ["600519"] * 3,
            "open": [10.0, 10.0, 11.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [1000, 1000, 1000],
            "execution_volume": [1000, 1000, 1000],
            "can_trade": [True] * 3,
            "suspended": [False] * 3,
            "limit_up": [False] * 3,
            "limit_down": [False] * 3,
        }
    ).to_csv(prices_path, index=False)
    signals_path = tmp_path / "signals.csv"
    pd.DataFrame(
        {"signal_date": ["2024-01-02"], "code": ["600519"], "action": ["BUY"], "target_shares": [100]}
    ).to_csv(signals_path, index=False)
    output = tmp_path / "result.json"

    assert main(
        [
            "evaluate",
            "--protocol",
            str(protocol_path),
            "--prices",
            str(prices_path),
            "--signals",
            str(signals_path),
            "--output",
            str(output),
            "--initial-cash",
            "2000",
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol_hash"] == _make_protocol().freeze().canonical_hash
    assert payload["experiment_status"] == "COMPLETED"
    assert payload["rules"]["version"] == "daily-v1"


def test_evaluate_cli_binds_single_stock_price_file(tmp_path) -> None:
    """单证券本地价格文件可由唯一信号代码绑定，避免隐式多证券混用。"""
    protocol_path = tmp_path / "protocol.json"
    _make_protocol().freeze().save(protocol_path)
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [10.0, 10.0, 11.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [1000, 1000, 1000],
            "execution_volume": [1000, 1000, 1000],
            "can_trade": [True] * 3,
            "suspended": [False] * 3,
            "limit_up": [False] * 3,
            "limit_down": [False] * 3,
        }
    ).to_csv(prices_path, index=False)
    signals_path = tmp_path / "signals.csv"
    pd.DataFrame(
        {"signal_date": ["2024-01-02"], "code": ["600519"], "action": ["BUY"], "target_shares": [100]}
    ).to_csv(signals_path, index=False)
    output = tmp_path / "result.json"
    assert main(
        [
            "evaluate",
            "--protocol",
            str(protocol_path),
            "--prices",
            str(prices_path),
            "--signals",
            str(signals_path),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2024-06-30"


def test_evaluate_cli_rejects_as_of_override_outside_frozen_boundary(tmp_path) -> None:
    """--as-of 不能把冻结协议的可见边界改成另一个时点。"""
    protocol_path = tmp_path / "protocol.json"
    _make_protocol().freeze().save(protocol_path)
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "code": ["600519"] * 3,
            "open": [10.0, 10.0, 11.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [1000] * 3,
            "execution_volume": [1000] * 3,
            "can_trade": [True] * 3,
            "suspended": [False] * 3,
            "limit_up": [False] * 3,
            "limit_down": [False] * 3,
        }
    ).to_csv(prices_path, index=False)
    signals_path = tmp_path / "signals.csv"
    pd.DataFrame(
        {"signal_date": ["2024-01-02"], "code": ["600519"], "action": ["BUY"], "target_shares": [100]}
    ).to_csv(signals_path, index=False)
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--protocol",
                str(protocol_path),
                "--prices",
                str(prices_path),
                "--signals",
                str(signals_path),
                "--output",
                str(output),
                "--as-of",
                "2024-06-29",
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


def test_evaluate_cli_rejects_rows_beyond_visible_boundary(tmp_path) -> None:
    """输入含可见边界之后的数据时不能用协议哈希输出 COMPLETED。"""
    protocol_path = tmp_path / "protocol.json"
    _make_protocol().freeze().save(protocol_path)
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-06-28", "2024-07-01"],
            "code": ["600519", "600519"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1000, 1000],
            "execution_volume": [1000, 1000],
            "can_trade": [True, True],
            "suspended": [False, False],
            "limit_up": [False, False],
            "limit_down": [False, False],
        }
    ).to_csv(prices_path, index=False)
    signals_path = tmp_path / "signals.csv"
    pd.DataFrame(
        {"signal_date": ["2024-06-28"], "code": ["600519"], "action": ["BUY"], "target_shares": [100]}
    ).to_csv(signals_path, index=False)
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--protocol",
                str(protocol_path),
                "--prices",
                str(prices_path),
                "--signals",
                str(signals_path),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()
