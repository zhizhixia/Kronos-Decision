"""报告跨文件发布状态机测试。"""
from __future__ import annotations

import pytest

from decision.storage import ReportStorage, ReportStorageError


def _payload() -> dict[str, object]:
    return {"schema_version": "2.0", "run_status": "SUCCEEDED", "action_permission": "NONE", "data_provenance": {"snapshot_id": "s1"}}


def test_report_publication_transitions_are_fail_closed(tmp_path) -> None:
    storage = ReportStorage(tmp_path / "reports.sqlite")
    pending = storage.save_report(
        report_id="r1",
        payload=_payload(),
        snapshot_id="s1",
        evaluation_id="none",
        model_revision="baseline",
        config_hash="c1",
        publication_status="PENDING",
    )
    assert storage.get_published("r1") is None
    published = storage.mark_published("r1", pending.version)
    assert published.publication_status == "PUBLISHED"
    with pytest.raises(ReportStorageError):
        storage.mark_failed("r1", pending.version)
