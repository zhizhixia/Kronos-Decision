"""研究注册表的独立生命周期和持久化测试。"""

from __future__ import annotations

import json

import pytest

from research.registry import (
    ApprovalRequiredError,
    BindingMismatchError,
    ELIGIBLE,
    ExperimentRecord,
    ImmutableExperimentError,
    RESEARCH,
    RETIRED,
    RUN_FAILED,
    ResearchRegistry,
    SHADOW,
    SUSPENDED,
)


_BINDING = {
    "strategy_id": "baseline",
    "strategy_version": "strategy-1",
    "model_version": "model-0",
    "data_version": "snapshot-1",
    "period": "10",
    "stock_pool_version": "pool-2024",
    "protocol_version": "protocol-1.0",
}


def _record(experiment_id: str = "exp-1") -> ExperimentRecord:
    """构造带完整证据绑定的成功实验。"""
    return ExperimentRecord(
        experiment_id=experiment_id,
        evidence_binding=_BINDING,
        result={"metric": 0.12},
    )


def test_register_persists_success_and_failure_records(tmp_path) -> None:
    """失败实验也必须写入真实临时 JSON，并可被新实例读回。"""
    path = tmp_path / "registry.json"
    registry = ResearchRegistry(path)
    failed = registry.register(
        {
            "experiment_id": "failed-exp",
            "run_status": RUN_FAILED,
            "error": "数据缺口",
            "result": {},
        }
    )
    assert failed.status == RESEARCH
    assert failed.run_status == RUN_FAILED
    assert failed.error == "数据缺口"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["experiments"]["failed-exp"]["current"]["run_status"] == RUN_FAILED
    reloaded = ResearchRegistry(path)
    assert reloaded.get("failed-exp").error == "数据缺口"


def test_ordinary_register_and_update_cannot_grant_eligibility(tmp_path) -> None:
    """普通写接口不能从研究状态直接跳到 ELIGIBLE。"""
    registry = ResearchRegistry(tmp_path / "registry.json")
    with pytest.raises(ApprovalRequiredError):
        registry.register({"experiment_id": "direct", "status": ELIGIBLE})

    registry.register(_record())
    with pytest.raises(ApprovalRequiredError):
        registry.update("exp-1", status=ELIGIBLE)
    assert registry.get("exp-1").status == RESEARCH


def test_shadow_requires_explicit_approval_and_exact_binding(tmp_path) -> None:
    """SHADOW 只有在 token、方法和逐字段绑定都匹配时才可获批。"""
    registry = ResearchRegistry(tmp_path / "registry.json")
    registry.register(_record())
    shadow = registry.enter_shadow("exp-1", reason="等待样本外观察")
    assert shadow.status == SHADOW

    wrong_binding = dict(_BINDING)
    wrong_binding["model_version"] = "model-2"
    with pytest.raises(BindingMismatchError):
        registry.approve_eligibility(
            "exp-1",
            approval_token="owner-approval",
            approval_method="manual-review",
            evidence_binding=wrong_binding,
        )
    assert registry.get("exp-1").status == SHADOW

    eligible = registry.approve_eligibility(
        "exp-1",
        approval_token="owner-approval",
        approval_method="manual-review",
        evidence_binding=_BINDING,
    )
    assert eligible.status == ELIGIBLE
    assert registry.is_eligible("exp-1", _BINDING) is True
    wrong_period = dict(_BINDING)
    wrong_period["period"] = "20"
    assert registry.is_eligible("exp-1", wrong_period) is False


def test_binding_revision_drops_old_qualification(tmp_path) -> None:
    """版本或股票池改变时新快照不能继承原证据资格。"""
    registry = ResearchRegistry(tmp_path / "registry.json")
    registry.register(_record())
    registry.enter_shadow("exp-1")
    registry.approve(
        "exp-1",
        token="token-1",
        method="review-1",
    )
    with pytest.raises(ImmutableExperimentError):
        registry.update("exp-1", model_version="model-2")

    research_registry = ResearchRegistry(tmp_path / "research-registry.json")
    research_registry.register(_record("research-exp"))
    changed = research_registry.update("research-exp", model_version="model-2")
    assert changed.status == RESEARCH
    assert changed.evidence_binding is not None
    assert changed.evidence_binding.model_version == "model-2"
    assert research_registry.is_eligible("research-exp", _BINDING) is False
    assert len(research_registry.history("research-exp")) == 2


def test_suspend_reinstate_retire_keep_all_history(tmp_path) -> None:
    """暂停、重新审批恢复和退役都追加快照而不删除历史。"""
    registry = ResearchRegistry(tmp_path / "registry.json")
    registry.register(_record())
    registry.approve(
        "exp-1",
        approval_token="token-1",
        approval_method="manual-review",
    )
    suspended = registry.suspend("exp-1", reason="漂移待复核")
    assert suspended.status == SUSPENDED

    with pytest.raises(ApprovalRequiredError):
        registry.reinstate("exp-1")
    reinstated = registry.reinstate(
        "exp-1",
        reason="复核通过",
        approval_token="token-2",
        approval_method="manual-review-2",
    )
    assert reinstated.status == ELIGIBLE

    retired = registry.retire("exp-1", reason="实验结束")
    assert retired.status == RETIRED
    assert len(registry.history("exp-1")) == 5
    events = registry.history_events("exp-1")
    assert [event["event"] for event in events] == [
        "REGISTERED",
        "APPROVED_ELIGIBLE",
        "SUSPENDED",
        "REINSTATED_ELIGIBLE",
        "RETIRED",
    ]

    reopened = ResearchRegistry(tmp_path / "registry.json")
    assert reopened.get("exp-1").status == RETIRED
    assert len(reopened.history("exp-1")) == 5
