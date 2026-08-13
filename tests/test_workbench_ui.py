"""工作台共享骨架与 /report 页面基础 UI 契约。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from webui.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"

V2_TEMPLATES = [WEBUI_DIR / "templates" / "base_v2.html", WEBUI_DIR / "templates" / "report_v2.html"]
V2_STATIC = [
    WEBUI_DIR / "static" / "workbench.css",
    WEBUI_DIR / "static" / "workbench.js",
    WEBUI_DIR / "static" / "report_v2.js",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _scan_for_external_links(text: str) -> list[str]:
    """返回文本中疑似 http(s) 外链或 Google Fonts 的片段（SVG 命名空间除外）。"""
    matches = []
    for pattern in (
        r"https?://[^\s\"'`<>)]+",
        r"googleapis\.com",
        r"gstatic\.com",
        r"fonts\.googleapis",
    ):
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return [m for m in matches if "w3.org/2000/svg" not in m]


@pytest.fixture
def client():
    return app.test_client()


def test_report_page_returns_200_with_shared_assets(client) -> None:
    response = client.get("/report")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 关键中文内容
    assert "决策报告" in text
    assert "股票代码" in text
    assert "生成决策报告" in text

    # 共享骨架资源
    assert 'href="/static/workbench.css"' in text
    assert 'src="/static/workbench.js"' in text

    # report v2 专属脚本
    assert 'src="/static/report_v2.js"' in text

    # 本地 Plotly
    assert 'src="/assets/plotly.min.js"' in text

    # ARIA 与无障碍
    assert 'aria-label="查询"' in text
    assert 'aria-controls="stock-suggestions"' in text
    assert 'aria-expanded="false"' in text
    assert 'role="listbox"' in text
    assert 'aria-label="股票联想"' in text
    assert 'aria-label="决策详情"' in text


def test_report_static_assets_serve_200(client) -> None:
    for path in ("/static/workbench.css", "/static/workbench.js", "/static/report_v2.js", "/assets/plotly.min.js"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} 应返回 200"


def test_v2_templates_and_static_have_no_external_links() -> None:
    for path in V2_TEMPLATES + V2_STATIC:
        assert path.exists(), f"{path} 必须存在"
        bad = _scan_for_external_links(_read_text(path))
        assert not bad, f"{path} 发现外部链接：{bad}"


def test_legacy_report_page_still_works_and_uses_old_styles(client) -> None:
    response = client.get("/report/legacy")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 旧页面仍使用旧 style.css，不应被 workbench.css 替换
    assert 'href="/static/style.css"' in text
    assert 'href="/static/workbench.css"' not in text
    assert 'src="/static/workbench.js"' not in text
    assert 'src="/static/report_v2.js"' not in text

    # 旧功能基本存在
    assert "决策报告" in text
    assert "股票代码" in text


# ---- 视觉 Bug 修复：[hidden] 与 .icon 基准尺寸 ----
def test_workbench_css_has_hidden_attr_rule() -> None:
    """[hidden] 属性必须以 !important 覆盖类定义的 display，确保带 hidden 的元素始终隐藏。"""
    css = _read_text(WEBUI_DIR / "static" / "workbench.css")
    assert re.search(r"\[hidden\]\s*\{\s*display\s*:\s*none\s*!important\s*;?\s*\}", css), (
        "workbench.css 应包含 [hidden] { display:none !important; } 规则"
    )


def test_workbench_css_has_global_icon_sizing() -> None:
    """全局 .icon 应统一宽高，防止 SVG 默认 300x150 撑坏按钮与徽章。"""
    css = _read_text(WEBUI_DIR / "static" / "workbench.css")
    # 统一基准尺寸
    assert re.search(r"\.icon\s*\{\s*[^}]*width\s*:\s*16px", css), (
        "workbench.css 应为 .icon 设置 width:16px"
    )
    assert re.search(r"\.icon\s*\{\s*[^}]*height\s*:\s*16px", css), (
        "workbench.css 应为 .icon 设置 height:16px"
    )
    # 按钮内图标明确 16px
    assert re.search(r"\.wb-btn\s+\.icon\s*\{\s*[^}]*width\s*:\s*16px", css), (
        "workbench.css 应为 .wb-btn .icon 明确 width:16px"
    )


def test_report_elapsed_badge_and_recent_bar_keep_hidden_attr() -> None:
    """报告页 elapsed-badge 与 recent-bar 仍应带 hidden 属性，初始不显示。"""
    html = _read_text(WEBUI_DIR / "templates" / "report_v2.html")
    assert re.search(r'id="elapsed-badge"\s+hidden', html), (
        "elapsed-badge 应保留 hidden 属性"
    )
    assert re.search(r'id="recent-bar"\s+hidden', html), (
        "recent-bar 应保留 hidden 属性"
    )
