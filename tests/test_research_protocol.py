"""研究协议的独立边界测试。"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json

import pytest

from research.protocol import (
    FrozenProtocolError,
    ProtocolIntegrityError,
    ProtocolValidationError,
    ResearchProtocol,
)


def _make_protocol(**overrides: object) -> ResearchProtocol:
    """构造测试使用的合法协议，不调用被测对象生成预期值。"""
    values: dict[str, object] = {
        "protocol_id": "baseline-protocol",
        "version": "1.0",
        "decision_as_of": "2024-06-30",
        "data_visible_boundary": "2024-06-30",
        "training_start": "2020-01-01",
        "training_end": "2022-12-31",
        "calibration_start": "2023-01-01",
        "calibration_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
        "horizons": (5, 10, 20),
        "costs": {"commission": 0.001, "slippage": 0.002},
        "budget": 12,
        "security_scope": ("A_SHARE",),
        "stock_pool_version": "pool-2024-01",
        "benchmark": "equal-weight",
        "primary_metric": "net_excess_return",
        "strategy_id": "baseline",
        "strategy_version": "s1",
        "model_version": "none",
        "data_version": "snapshot-1",
    }
    values.update(overrides)
    return ResearchProtocol(**values)


def test_decision_time_cannot_be_later_than_visible_boundary() -> None:
    """决策时点晚于数据边界时必须拒绝。"""
    with pytest.raises(ProtocolValidationError, match="不能晚于"):
        _make_protocol(
            decision_as_of="2024-07-01",
            data_visible_boundary="2024-06-30",
        )


def test_test_window_cannot_overlap_training_or_calibration() -> None:
    """测试窗口与训练或校准相交时必须拒绝。"""
    with pytest.raises(ProtocolValidationError, match="时间窗口重叠"):
        _make_protocol(test_start="2022-12-31")
    with pytest.raises(ProtocolValidationError, match="时间窗口重叠"):
        _make_protocol(test_start="2023-12-31")


def test_supported_horizons_are_exactly_five_ten_and_twenty() -> None:
    """不支持的周期和重复周期不能进入协议。"""
    with pytest.raises(ProtocolValidationError, match="仅支持"):
        _make_protocol(horizons=(5, 15))
    with pytest.raises(ProtocolValidationError, match="不能重复"):
        _make_protocol(horizons=(5, 5))


def test_costs_and_budget_must_be_non_negative() -> None:
    """成本和预算的负数必须在构造时失败。"""
    with pytest.raises(ProtocolValidationError, match="非负"):
        _make_protocol(costs={"commission": -0.001})
    with pytest.raises(ProtocolValidationError, match="非负"):
        _make_protocol(budget=-1)


def test_default_aliases_do_not_conflict_with_budget_defaults() -> None:
    """省略默认 budget 时使用 experiment/trial budget 别名应正常工作。"""
    protocol = ResearchProtocol(
        protocol_id="alias-protocol",
        decision_as_of="2024-06-30",
        data_visible_until="2024-06-30",
        train_start="2020-01-01",
        train_end="2022-12-31",
        calibration_start="2023-01-01",
        calibration_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-06-30",
        experiment_budget=7,
        pool_version="pool-1",
    )
    assert protocol.experiment_budget == 7.0
    assert protocol.stock_pool_version == "pool-1"

    trial_protocol = ResearchProtocol(
        protocol_id="trial-protocol",
        decision_as_of="2024-06-30",
        data_visible_until="2024-06-30",
        train_start="2020-01-01",
        train_end="2022-12-31",
        calibration_start="2023-01-01",
        calibration_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-06-30",
        trial_budget=3,
    )
    assert trial_protocol.budget == 3.0


def test_freeze_is_immutable_and_revision_is_a_new_version(tmp_path) -> None:
    """冻结对象不能原地改写，修订必须产生新版本和新哈希。"""
    original = _make_protocol()
    frozen = original.freeze()
    assert frozen.frozen is True
    assert original.frozen is False
    with pytest.raises((AttributeError, TypeError)):
        frozen.budget = 99
    with pytest.raises(TypeError):
        frozen.costs["commission"] = 0.9

    revised = frozen.revise(costs={"commission": 0.003, "slippage": 0.002})
    assert revised.version == "1.1"
    assert revised.parent_version == "1.0"
    assert revised.frozen is False
    assert revised.costs["commission"] == 0.003
    assert frozen.costs["commission"] == 0.001
    assert revised.canonical_hash != frozen.canonical_hash
    aliased_revision = frozen.revise(experiment_budget=21)
    assert aliased_revision.budget == 21.0
    history_path = tmp_path / "protocol-history.json"
    frozen.save(history_path)
    with pytest.raises(FrozenProtocolError):
        revised.save(history_path)
    with pytest.raises(FrozenProtocolError):
        frozen.revise(version="1.0", budget=1)


def test_canonical_hash_is_saved_and_detects_tampering(tmp_path) -> None:
    """保存的规范哈希可独立复算，篡改 JSON 时读回失败。"""
    protocol = _make_protocol().freeze()
    path = tmp_path / "protocol.json"
    protocol.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_payload = dict(payload)
    expected_payload.pop("canonical_hash")
    expected = sha256(
        json.dumps(
            expected_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert payload["canonical_hash"] == expected
    assert ResearchProtocol.load(path).canonical_hash == expected

    payload["budget"] = 13.0
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ProtocolIntegrityError):
        ResearchProtocol.load(path)


def test_date_objects_are_accepted_without_pandas() -> None:
    """标准库 date 输入可完成协议校验。"""
    protocol = _make_protocol(
        decision_as_of=date(2024, 6, 30),
        data_visible_boundary=date(2024, 6, 30),
    )
    assert protocol.decision_as_of == "2024-06-30"


def test_frozen_protocol_rejects_evaluation_boundary_and_scope_overrides() -> None:
    """冻结协议的实际评价输入不能越过时点、测试窗口或证券范围。"""
    protocol = _make_protocol(security_scope=("600519",)).freeze()
    common = {
        "as_of": "2024-06-30",
        "price_dates": ("2024-06-28",),
        "signal_dates": ("2024-06-28",),
        "security_codes": ("600519",),
    }
    assert protocol.validate_evaluation_inputs(**common) == "2024-06-30"

    with pytest.raises(ProtocolValidationError, match="data_visible_boundary"):
        protocol.validate_evaluation_inputs(**{**common, "as_of": "2024-06-29"})
    with pytest.raises(ProtocolValidationError, match="证券不在"):
        protocol.validate_evaluation_inputs(**{**common, "security_codes": ("000001",)})
    with pytest.raises(ProtocolValidationError, match="data_visible_boundary"):
        protocol.validate_evaluation_inputs(
            **{**common, "price_dates": ("2024-07-01",)}
        )
