"""组合页 UI 契约测试。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from webui.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"

V2_TEMPLATES = [WEBUI_DIR / "templates" / "base_v2.html", WEBUI_DIR / "templates" / "portfolio.html"]
V2_STATIC = [
    WEBUI_DIR / "static" / "workbench.css",
    WEBUI_DIR / "static" / "workbench.js",
    WEBUI_DIR / "static" / "portfolio.js",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _scan_for_external_links(text: str) -> list[str]:
    """返回文本中疑似 http(s) 外链或 Google Fonts 的片段（SVG 命名空间除外）。"""
    matches = []
    for pattern in (
        r"https?://[^\s\"'`<>)\x0b]+",
        r"googleapis\.com",
        r"gstatic\.com",
        r"fonts\.googleapis",
    ):
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return [m for m in matches if "w3.org/2000/svg" not in m]


def _starts_with_bom(path: Path) -> bool:
    raw = path.read_bytes()
    return raw.startswith(b"\xef\xbb\xbf")


@pytest.fixture
def client():
    return app.test_client()


def test_portfolio_page_returns_200_with_shared_assets(client) -> None:
    response = client.get("/portfolio")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 继承 base_v2.html
    assert 'href="/static/workbench.css"' in text
    assert 'src="/static/workbench.js"' in text

    # 组合页专属脚本
    assert 'src="/static/portfolio.js"' in text

    # 关键中文内容
    assert "组合" in text
    assert "现金" in text
    assert "持仓数量" in text
    assert "风险档案" in text
    assert "组合版本" in text
    assert "组合资料" in text
    assert "持仓" in text
    assert "约束上限" in text
    assert "模拟调仓" in text
    assert "保存修改" in text
    assert "放弃修改" in text
    assert "行业约束将在模拟调仓时验证" in text

    # 不虚构市值盈亏
    assert "市值" not in text
    assert "盈亏" not in text
    assert "收益率" not in text

    # 无确认下单 / 券商控件
    assert "确认下单" not in text
    assert "券商" not in text
    assert "交易密码" not in text

    # 表单元素存在
    assert 'id="profile-name"' in text
    assert 'name="risk-profile"' in text
    assert 'value="conservative"' in text
    assert 'value="balanced"' in text
    assert 'value="aggressive"' in text
    assert 'id="cash"' in text
    assert 'id="holdings-body"' in text
    assert 'id="constraints-grid"' in text
    assert 'id="rebalance-btn"' in text


def test_portfolio_static_assets_serve_200(client) -> None:
    for path in ("/static/portfolio.js", "/static/workbench.css", "/static/workbench.js"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} 应返回 200"


def test_portfolio_v2_templates_and_static_have_no_external_links() -> None:
    for path in V2_TEMPLATES + V2_STATIC:
        assert path.exists(), f"{path} 必须存在"
        bad = _scan_for_external_links(_read_text(path))
        assert not bad, f"{path} 发现外部链接：{bad}"


def test_portfolio_files_are_utf8_without_bom() -> None:
    for path in V2_TEMPLATES + V2_STATIC:
        assert path.exists(), f"{path} 必须存在"
        assert not _starts_with_bom(path), f"{path} 包含 UTF-8 BOM"
        # 尝试用 UTF-8 读取，失败会抛 UnicodeDecodeError
        _read_text(path)


def test_workbench_js_exposes_generic_request_and_put_helpers() -> None:
    """workbench.js 应收敛为通用 requestJson 并公开 putJson。"""
    text = _read_text(WEBUI_DIR / "static" / "workbench.js")
    assert "function requestJson" in text, "workbench.js 应定义 requestJson"
    assert "function putJson" in text, "workbench.js 应定义 putJson"
    assert "function postJson" in text, "postJson 行为保持兼容"
    assert "function getJson" in text, "getJson 行为保持兼容"
    assert "requestJson: requestJson" in text, "requestJson 应导出"
    assert "putJson: putJson" in text, "putJson 应导出"
    # 不再重复实现 fetch/AbortController，统一委托 requestJson
    assert text.count("new AbortController()") == 1


def test_portfolio_save_uses_put_while_rebalance_uses_post() -> None:
    """保存走 PUT，模拟调仓走 POST——通过静态断言锁死调用路径。"""
    text = _read_text(WEBUI_DIR / "static" / "portfolio.js")
    assert 'WB.putJson("/api/v2/portfolio"' in text, "保存应使用 putJson"
    assert 'WB.postJson("/api/v2/portfolio", payload)' not in text, "保存不应再用 postJson"
    assert 'WB.postJson("/api/v2/portfolio/rebalance"' in text, "模拟调仓仍使用 postJson"


def test_workbench_put_and_post_share_request_json() -> None:
    """postJson/putJson/getJson 均委托 requestJson，且方法字面量锁死——无需新依赖即可验证。"""
    text = _read_text(WEBUI_DIR / "static" / "workbench.js")
    assert 'method: "POST", body: body' in text, "postJson 应以 POST 委托 requestJson"
    assert 'method: "PUT", body: body' in text, "putJson 应以 PUT 委托 requestJson"
    assert 'method: "GET", timeoutMs: timeoutMs' in text, "getJson 应以 GET 委托 requestJson"
    # 通用请求仅一处 fetch/AbortController，避免重复实现
    assert text.count("new AbortController()") == 1
    assert text.count("await fetch(") == 1


def test_portfolio_save_api_accepts_put_and_rejects_post(monkeypatch) -> None:
    """后端 /api/v2/portfolio 仅接受 GET/PUT，POST 应 405——锁定 PUT 契约，不写真实用户 DB。"""
    from webui.app import app

    captured = {}

    class FakeStore:
        def get_portfolio(self, profile_id):
            return {"name": "默认", "risk_profile": "balanced", "cash": 0.0, "holdings": []}

        def replace_portfolio(self, payload, profile_id):
            captured["payload"] = payload
            captured["profile_id"] = profile_id
            return payload

    monkeypatch.setattr("portfolio.store.PortfolioStore", FakeStore)
    client = app.test_client()

    ok = client.put(
        "/api/v2/portfolio",
        json={"name": "x", "risk_profile": "balanced", "cash": 1.0, "holdings": []},
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert ok.status_code == 200
    assert ok.get_json()["status"] == "ok"
    assert captured["profile_id"] == "default"
    assert captured["payload"]["name"] == "x"

    bad = client.post(
        "/api/v2/portfolio",
        json={"name": "x"},
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert bad.status_code == 405, "保存用 PUT，POST 应被拒绝以锁定契约"
