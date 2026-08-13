"""研究证据页与系统诊断页的 UI 契约测试。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from webui.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"

MAIN_TEMPLATES = [
    WEBUI_DIR / "templates" / "report_v2.html",
    WEBUI_DIR / "templates" / "portfolio.html",
    WEBUI_DIR / "templates" / "research.html",
    WEBUI_DIR / "templates" / "settings.html",
]

RESEARCH_SETTINGS_TEMPLATES = [
    WEBUI_DIR / "templates" / "research.html",
    WEBUI_DIR / "templates" / "settings.html",
]

RESEARCH_SETTINGS_STATIC = [
    WEBUI_DIR / "static" / "research.js",
    WEBUI_DIR / "static" / "settings.js",
]

V2_STATIC = [
    WEBUI_DIR / "static" / "workbench.css",
    WEBUI_DIR / "static" / "workbench.js",
    WEBUI_DIR / "static" / "report_v2.js",
    WEBUI_DIR / "static" / "portfolio.js",
    WEBUI_DIR / "static" / "research.js",
    WEBUI_DIR / "static" / "settings.js",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _scan_for_external_links(text: str) -> list[str]:
    """返回文本中疑似 http(s) 外链或 Google Fonts 的片段（SVG 命名空间除外）。"""
    matches = []
    for pattern in (
        r"https?://[^\s\"'`<>)+\x0b]+",
        r"googleapis\.com",
        r"gstatic\.com",
        r"fonts\.googleapis",
    ):
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return [m for m in matches if "w3.org/2000/svg" not in m]


def _starts_with_bom(path: Path) -> bool:
    return path.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.fixture
def client():
    return app.test_client()


def test_research_page_returns_200_with_shared_assets(client) -> None:
    response = client.get("/research")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 继承 base_v2.html
    assert 'href="/static/workbench.css"' in text
    assert 'src="/static/workbench.js"' in text

    # 研究页专属脚本
    assert 'src="/static/research.js"' in text

    # 关键中文内容
    assert "研究证据" in text
    assert "运行状态" in text
    assert "证据门禁" in text
    assert "核心指标" in text
    assert "严格滚动样本外校准" in text
    assert "基线比较" in text
    assert "前瞻验证" in text
    assert "每日建议前瞻台账" in text
    assert "运行清单详情" in text
    assert "尚无评估运行记录" in text
    assert "复制" in text

    # 空态命令
    assert "python -m evaluation.run --stage smoke --as-of YYYY-MM-DD" in text
    assert "python -m evaluation.run --stage full --as-of YYYY-MM-DD" in text


def test_settings_page_returns_200_with_shared_assets(client) -> None:
    response = client.get("/settings")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 继承 base_v2.html
    assert 'href="/static/workbench.css"' in text
    assert 'src="/static/workbench.js"' in text

    # 诊断页专属脚本
    assert 'src="/static/settings.js"' in text

    # 关键中文内容
    assert "系统诊断" in text
    assert "配置路径" in text
    assert "decision/config.yaml" in text
    assert "模型与运行环境" in text
    assert "预测参数" in text
    assert "数据源与缓存" in text
    assert "WebUI" in text
    assert "最新证据" in text


def test_settings_template_has_no_input_controls(client) -> None:
    """诊断页只读展示，不得使用输入控件避免被误解为可在线修改。"""
    response = client.get("/settings")
    text = response.get_data(as_text=True)
    assert "<input" not in text.lower()
    assert "<select" not in text.lower()
    assert "<textarea" not in text.lower()


def test_settings_js_has_no_put(client) -> None:
    """诊断脚本只读，绝不写回配置。"""
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    assert "PUT" not in text
    assert "putJson" not in text
    assert "method: \"PUT\"" not in text
    assert "method: 'PUT'" not in text


def test_research_empty_commands_exist_in_template() -> None:
    """空态命令的真实契约：两条评估命令以静态空态写在 research.html 中，
    保证用户在研究证据页即可看到（参考 test_research_page_returns_200_with_shared_assets
    验证 /research 渲染结果）。research.js 不再被要求持有重复命令字符串。"""
    text = _read_text(WEBUI_DIR / "templates" / "research.html")
    assert "python -m evaluation.run --stage smoke --as-of YYYY-MM-DD" in text
    assert "python -m evaluation.run --stage full --as-of YYYY-MM-DD" in text


def test_research_js_does_not_use_innerhtml() -> None:
    """研究页渲染必须走安全 DOM 辅助，不直接 innerHTML。"""
    text = _read_text(WEBUI_DIR / "static" / "research.js")
    assert ".innerHTML" not in text


def test_research_and_settings_templates_and_static_have_no_external_links() -> None:
    for path in RESEARCH_SETTINGS_TEMPLATES + RESEARCH_SETTINGS_STATIC:
        assert path.exists(), f"{path} 必须存在"
        bad = _scan_for_external_links(_read_text(path))
        assert not bad, f"{path} 发现外部链接：{bad}"


def test_research_and_settings_files_are_utf8_without_bom() -> None:
    for path in RESEARCH_SETTINGS_TEMPLATES + RESEARCH_SETTINGS_STATIC:
        assert path.exists(), f"{path} 必须存在"
        assert not _starts_with_bom(path), f"{path} 包含 UTF-8 BOM"
        _read_text(path)


def test_all_four_main_templates_extend_base_v2() -> None:
    for path in MAIN_TEMPLATES:
        text = _read_text(path)
        assert '{% extends "base_v2.html" %}' in text, f"{path} 应继承 base_v2.html"


def test_legacy_report_contains_new_report_link(client) -> None:
    response = client.get("/report/legacy")
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "进入新版决策报告" in text
    assert 'href="/report"' in text


def test_legacy_report_still_uses_old_styles_and_scripts(client) -> None:
    response = client.get("/report/legacy")
    assert response.status_code == 200
    text = response.get_data(as_text=True)

    # 旧页面仍使用旧 style.css，不应被 workbench.css 替换
    assert 'href="/static/style.css"' in text
    assert 'href="/static/workbench.css"' not in text
    assert 'src="/static/workbench.js"' not in text
    assert 'src="/static/report_v2.js"' not in text

    # 旧交互脚本仍在
    assert "<script>" in text


def test_client_main_pages_legacy_and_static_js_return_200(client) -> None:
    pages = ["/", "/report", "/portfolio", "/research", "/settings", "/report/legacy"]
    for path in pages:
        assert client.get(path, follow_redirects=True).status_code == 200, f"{path} 应返回 200"

    static_js = [
        "/static/workbench.js",
        "/static/report_v2.js",
        "/static/portfolio.js",
        "/static/research.js",
        "/static/settings.js",
        "/assets/plotly.min.js",
    ]
    for path in static_js:
        response = client.get(path)
        assert response.status_code == 200, f"{path} 应返回 200"
        assert response.content_type.startswith("application/javascript") or "javascript" in response.content_type


def test_v2_static_js_are_utf8_without_bom() -> None:
    for path in V2_STATIC:
        assert path.exists(), f"{path} 必须存在"
        assert not _starts_with_bom(path), f"{path} 包含 UTF-8 BOM"
        _read_text(path)


def test_settings_js_uses_event_delegation_for_dynamic_copy_buttons() -> None:
    """动态创建的 hash 复制按钮通过事件委托绑定，data-target 为真正属性。"""
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    assert "document.body.addEventListener(\"click\"" in text, "应使用事件委托绑定复制按钮"
    assert 'button.getAttribute("data-target")' in text, "应从 data-target 属性读取目标"
    assert '"data-target": targetId' in text, "appendChild 中应使用 data-target 属性"


def test_settings_js_displays_all_evidence_hashes() -> None:
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    for key in ("model_hash", "rules_hash", "config_hash", "data_hash"):
        assert key in text, f"settings.js 应展示 {key}"


def test_settings_js_shows_version_match_from_gate_checks() -> None:
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    assert "checks.VERSION_MATCH" in text, "版本匹配应读取 gate_result.checks.VERSION_MATCH"
    assert '"匹配"' in text, "应显示 匹配"
    assert '"不匹配"' in text, "应显示 不匹配"
    assert '"无结果"' in text, "应显示 无结果"


def test_settings_js_shows_no_gate_result_when_checks_missing() -> None:
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    assert '"无门禁结果"' in text, "无 checks 时应显示 无门禁结果"


def test_settings_js_only_uses_get() -> None:
    """诊断页脚本只读，绝不写回配置。"""
    text = _read_text(WEBUI_DIR / "static" / "settings.js")
    assert "PUT" not in text
    assert "putJson" not in text
    assert 'method: "PUT"' not in text
    assert "method: 'PUT'" not in text


def test_research_js_percent_metrics_use_format_percent() -> None:
    """年化超额收益与回撤恶化按百分比显示，ECE/RankIC/IR 保持小数。"""
    text = _read_text(WEBUI_DIR / "static" / "research.js")
    assert "PERCENT_METRICS" in text, "应定义 PERCENT_METRICS"
    assert "annualized_excess_return: true" in text, "年化超额收益应为百分比"
    assert "drawdown_worsening: true" in text, "回撤恶化应为百分比"
    assert "coverage_80: true" in text, "coverage 应为百分比"
    assert "positive_12m_window_ratio: true" in text, "positive 窗口比例应为百分比"


def test_research_js_gate_rows_use_svg_icons_and_fail_first() -> None:
    text = _read_text(WEBUI_DIR / "static" / "research.js")
    assert "createIcon" in text, "应定义 createIcon"
    assert "#icon-check" in text, "通过状态应使用 SVG 对勾图标"
    assert "#icon-x" in text, "失败状态应使用 SVG 叉号图标"
    assert "failed.has(a.code) ? 0 : 1" in text, "失败项应优先排序"


def test_research_js_does_not_hide_gate_card_when_no_checks() -> None:
    text = _read_text(WEBUI_DIR / "static" / "research.js")
    assert 'card.hidden = false' in text, "门禁卡不应隐藏"
    assert '"无门禁结果"' in text, "无 checks 时应显示 无门禁结果"


def test_research_js_derive_phase_labels_collection_diagnostic() -> None:
    """collection_only / blocked / diagnostic 工件显示 收集/诊断，不得误标正式。"""
    text = _read_text(WEBUI_DIR / "static" / "research.js")
    assert "collection_only || manifest.blocked || manifest.diagnostic" in text
    assert '"收集/诊断"' in text
    assert "formal_protocol" in text
