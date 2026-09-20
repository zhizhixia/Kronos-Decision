"""研究核心契约哈希与状态反例。"""
from __future__ import annotations

import pytest

from research.contracts import ForecastArtifact, ResearchContractError, StrategySpec, ValidationEvidence


def test_research_contract_hashes_are_stable_and_bind_versions() -> None:
    strategy = StrategySpec("baseline", "1", (5, 10, 20), "u1", "CLOSE", "NEXT_OPEN", {"max_weight": 0.1})
    assert strategy.content_hash
    artifact = ForecastArtifact("a1", "s1", "2024-01-02", "m1", "t1", {"temperature": 1}, 7, ({"horizon": 5},), "RESEARCH_ONLY")
    assert artifact.content_hash
    evidence = ValidationEvidence("e1", "p1", "s1", "m1", {"rank_ic": 0.1}, "sample", ("未接入公告",), None, "RESEARCH_ONLY")
    assert evidence.content_hash


def test_research_contract_rejects_invalid_actionable_status() -> None:
    with pytest.raises(ResearchContractError):
        StrategySpec("bad", "1", (3,), "u1", "CLOSE", "NEXT_OPEN", {})
    with pytest.raises(ResearchContractError):
        ForecastArtifact("a1", "s1", "2024-01-02", "m1", "t1", {}, 1, (), "QUALIFIED")
