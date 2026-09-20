"""v2 工作台 UI 轻量契约测试。

验证目标：
- /report 页面 200 并继承 base_v2
- 引用本地 Plotly 和共享资源，无外部 CDN/字体
- 关键 ARIA 与中文文本存在
- 不使用动态 innerHTML 渲染 API 数据
"""
from __future__ import annotations

import re

from webui.app import app


def _report_html() -> str:
    client = app.test_client()
    response = client.get("/report")
    assert response.status_code == 200
    return response.get_data(as_text=True)


def test_report_page_returns_200_and_extends_base_v2() -> None:
    html = _report_html()
    assert "{% extends \"base_v2.html\" %}" not in html  # Jinja 已渲染
    assert "workbench-sidebar" in html
    assert "workbench-main" in html


def test_report_uses_local_plotly_and_shared_assets() -> None:
    html = _report_html()
    assert "/assets/plotly.min.js" in html
    assert "workbench.css" in html
    assert "workbench.js" in html
    assert "report_v2.js" in html


def test_report_has_no_external_http_or_https_links() -> None:
    html = _report_html()
    # SVG 命名空间 xmlns 属于标记必需属性，不计入外部资源链接
    cleaned = re.sub(r'xmlns="[^"]+"', '', html)
    assert not re.search(r"https?://", cleaned), "页面不得包含 http/https 外部资源链接"


def test_report_has_no_google_fonts() -> None:
    html = _report_html()
    assert "fonts.googleapis.com" not in html
    assert "fonts.gstatic.com" not in html


def test_report_has_required_aria_and_text() -> None:
    html = _report_html()
    assert 'role="navigation"' in html
    assert 'aria-label="主导航"' in html
    assert "决策报告" in html
    assert "五态建议" in html
    assert "证据门禁" in html
    assert "股票代码" in html


def test_report_does_not_use_innerhtml_for_api_data() -> None:
    """report_v2.js 中禁止通过 innerHTML 渲染 API 数据。"""
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "report_v2.js").read_text(encoding="utf-8")
    assert ".innerHTML" not in js, "report_v2.js 不得使用 innerHTML 处理 API 数据"


def test_report_buttons_pass_explicit_generation_modes() -> None:
    """两个按钮都必须显式传递模式，不能把 click 事件当成 mode。"""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "webui" / "static" / "report_v2.js").read_text(encoding="utf-8")
    assert re.search(
        r'btn\.addEventListener\("click", function \(\) \{\s+generateReport\("model"\);',
        js,
    )
    assert re.search(
        r'baselineBtn\.addEventListener\("click", function \(\) \{\s+generateReport\("baseline"\);',
        js,
    )
    assert 'btn.addEventListener("click", generateReport);' not in js


def test_report_stock_input_accepts_code_or_name() -> None:
    """股票输入框支持中文名称联想，但提交仍校验六位代码。"""
    html = _report_html()
    assert 'maxlength="32"' in html
    assert 'inputmode="search"' in html
    assert 'placeholder="输入六位代码或中文名称，选中后自动填入代码"' in html
    assert "股票代码或名称" in html
    # 提交端仍只接受六位数字代码
    from webui.app import app

    response = app.test_client().post(
        "/api/v2/decision-report",
        json={"stock_code": "贵州茅台"},
        headers={"Origin": "http://localhost:7070"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_REQUEST"
