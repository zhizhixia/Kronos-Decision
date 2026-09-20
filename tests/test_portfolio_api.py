"""持仓预览、确认和重启读回 API 测试。"""
from __future__ import annotations


def _csv() -> str:
    return "code,quantity,available_quantity,avg_cost,as_of,source,name\n600519,1000,900,100.5,2024-01-02,manual,贵州茅台\n"


def test_portfolio_preview_has_no_write_side_effect() -> None:
    from webui.app import app

    app_client = app.test_client()
    response = app_client.post(
        "/api/v2/portfolio/preview",
        json={"csv_content": _csv()},
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["positions"][0]["available_quantity"] == 900


def test_portfolio_confirm_requires_explicit_confirmation_and_reads_back() -> None:
    from webui.app import app

    app_client = app.test_client()
    content = _csv()
    preview = app_client.post(
        "/api/v2/portfolio/preview",
        json={"csv_content": content},
        headers={"Origin": "http://127.0.0.1:7070"},
    ).get_json()
    rejected = app_client.post(
        "/api/v2/portfolio/confirm",
        json={
            "csv_content": content,
            "preview_id": preview["preview_id"],
            "content_hash": preview["content_hash"],
            "cash": 10000,
            "as_of": "2024-01-02",
            "available_at": "2024-01-02T16:00:00+08:00",
            "source": "manual",
            "confirmed": False,
        },
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert rejected.status_code == 400

    confirmed = app_client.post(
        "/api/v2/portfolio/confirm",
        json={
            "csv_content": content,
            "preview_id": preview["preview_id"],
            "content_hash": preview["content_hash"],
            "cash": 10000,
            "as_of": "2024-01-02",
            "available_at": "2024-01-02T16:00:00+08:00",
            "source": "manual",
            "confirmed": True,
        },
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert confirmed.status_code == 200
    snapshot = confirmed.get_json()["snapshot"]
    restored = app_client.get(f"/api/v2/portfolio/snapshots/{snapshot['snapshot_id']}")
    assert restored.status_code == 200
    assert restored.get_json()["snapshot"]["content_hash"] == snapshot["content_hash"]
