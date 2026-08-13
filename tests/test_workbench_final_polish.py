"""WebUI 最终收尾回归测试。

只读契约，不连接网络、不加载模型。覆盖范围：
1) /report 输入允许中文名称；提交 JS 仍有六位校验。
2) 所有 webui 模板/静态资源无远程 http(s) 链接（SVG 命名空间除外）；
   禁止 googleapis/gstatic/jsdelivr/cdn.plot。
3) style.css 无 @import 远程字体；字体含 Segoe UI / Microsoft YaHei / Cascadia Mono。
4) index.html 引用 /assets/plotly.min.js；无 axios 远程 script；有本地 fetch axios shim；
   抽取所有内联脚本用 node --check 验证语法（写入 pytest tmp_path，不产生仓库产物）。
5) settings.js 只 GET 无 PUT；含 model_hash/rules_hash/config_hash/data_hash/VERSION_MATCH；
   动态复制使用 data-target 并依赖事件委托。
6) research.js 百分比集合含 annualized_excess_return/drawdown_worsening；
   门禁使用 createIcon；derivePhase 优先 collection_only。
7) portfolio 模板三段 radio；portfolio.js get/setSelectedRiskProfile；不依赖 select.value。
8) workbench localStorage 调用包裹 try/catch。
9) plotly_asset 路由带返回类型注解、本地 200。
10) 相关 WebUI 与本测试文件 UTF-8 无 BOM。
"""
from __future__ import annotations

import inspect
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from webui.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"
TEMPLATES_DIR = WEBUI_DIR / "templates"
STATIC_DIR = WEBUI_DIR / "static"

ALL_HTML = sorted(TEMPLATES_DIR.glob("*.html"))
ALL_CSS = sorted(STATIC_DIR.glob("*.css"))
ALL_JS = sorted(STATIC_DIR.glob("*.js"))

_W3 = "w3.org/2000/svg"
_FORBIDDEN_HOSTS = ("googleapis.com", "gstatic.com", "cdn.jsdelivr.com", "cdn.plot")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _starts_with_bom(path: Path) -> bool:
    return path.read_bytes().startswith(b"\xef\xbb\xbf")


def _scan_external(text: str) -> list:
    hits = []
    for m in re.findall(r'https?://[^\s"\'`<>)\x0b]+', text, flags=re.IGNORECASE):
        if _W3 in m:
            continue
        hits.append(m)
    for host in _FORBIDDEN_HOSTS:
        for m in re.findall(re.escape(host), text, flags=re.IGNORECASE):
            hits.append(m)
    return hits


@pytest.fixture
def client():
    return app.test_client()


# ---- 1) /report 输入允许中文名称 + 六位校验 ----
def test_report_input_allows_chinese_name(client) -> None:
    html = client.get("/report").get_data(as_text=True)
    assert 'id="stock-code"' in html
    assert 'inputmode="search"' in html
    m = re.search(r'id="stock-code"[^>]*maxlength="(\d+)"', html)
    assert m is not None, "stock-code 应显式设置 maxlength"
    assert int(m.group(1)) >= 20, "maxlength 应 >= 20 以容纳中文名称"
    assert "名称" in html

    js = _read(STATIC_DIR / "report_v2.js")
    assert "\\d{6}" in js, "提交前应保留六位数字校验"


# ---- 2) 所有 webui 资源无远程链接 ----
def test_all_webui_files_have_no_remote_links() -> None:
    files = ALL_HTML + ALL_CSS + ALL_JS
    assert files, "未发现 webui 资源文件"
    for path in files:
        bad = _scan_external(_read(path))
        assert not bad, f"{path} 发现远程链接：{bad}"


# ---- 3) style.css 无远程字体；含本地字体 ----
def test_style_css_no_remote_import_and_local_fonts() -> None:
    css = _read(STATIC_DIR / "style.css")
    assert "@import" not in css, "style.css 不应使用 @import 远程字体"
    for font in ("Segoe UI", "Microsoft YaHei", "Cascadia Mono"):
        assert font in css, f"style.css 应包含字体 {font}"


# ---- 4) index.html 本地 Plotly、无远程 axios、本地 fetch shim、内联脚本语法合法 ----
def test_index_html_local_plotly_no_remote_axios() -> None:
    html = _read(TEMPLATES_DIR / "index.html")
    assert 'src="/assets/plotly.min.js"' in html
    assert "cdn.jsdelivr.com" not in html
    assert "unpkg.com" not in html
    assert re.search(r"<script[^>]*axios", html, flags=re.IGNORECASE) is None, "禁止远程 axios script 标签"
    assert "const axios" in html and "fetch(" in html, "应提供本地 fetch axios shim"


@pytest.mark.skipif(shutil.which("node") is None, reason="node 不可用")
def test_index_html_inline_scripts_pass_node_check(tmp_path: Path) -> None:
    html = _read(TEMPLATES_DIR / "index.html")
    blocks = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
        html,
        flags=re.DOTALL,
    )
    assert blocks, "index.html 至少应有一段内联脚本"
    for i, body in enumerate(blocks):
        f = tmp_path / f"inline_{i}.js"
        f.write_text(body, encoding="utf-8")
        res = subprocess.run(["node", "--check", str(f)], capture_output=True)
        assert res.returncode == 0, (
            f"内联脚本 {i} 语法错误：{res.stderr.decode(errors='replace')}"
        )


# ---- 5) settings.js 只读 + 哈希 + data-target 事件委托 ----
def test_settings_js_readonly_hashes_and_data_target() -> None:
    js = _read(STATIC_DIR / "settings.js")
    assert "PUT" not in js, "诊断脚本不应包含 PUT"
    assert "putJson" not in js
    assert "PATCH" not in js
    assert '"POST"' not in js, "诊断脚本只读，不应有 POST 写请求"
    assert '"DELETE"' not in js

    for h in ("model_hash", "rules_hash", "config_hash", "data_hash"):
        assert h in js, f"诊断脚本应展示 {h}"
    assert "VERSION_MATCH" in js, "诊断脚本应渲染 VERSION_MATCH 检查"

    assert '"data-target"' in js, "动态复制按钮应使用标准 data-target 键"
    assert "dataTarget" not in js, "旧的 camelCase dataTarget 已不应存在"
    assert ".closest(\".wb-copy-btn\")" in js, "复制应通过事件委托委托到 .wb-copy-btn"
    assert "getAttribute(\"data-target\")" in js, "事件委托应通过 getAttribute(data-target) 读取目标"


def test_settings_page_has_no_input_controls(client) -> None:
    html = client.get("/settings").get_data(as_text=True)
    assert "<input" not in html.lower()
    assert "<select" not in html.lower()
    assert "<textarea" not in html.lower()


# ---- 6) research.js 百分比集合 + createIcon + derivePhase 优先 collection_only ----
def test_research_js_percent_metrics_create_icon_phase_order() -> None:
    js = _read(STATIC_DIR / "research.js")
    assert "annualized_excess_return" in js
    assert "drawdown_worsening" in js
    assert "function createIcon" in js, "门禁渲染应使用 createIcon 创建图标"

    co = js.find("collection_only")
    fp = js.find("formal_protocol")
    assert co != -1 and fp != -1, "derivePhase 应同时涉及 collection_only 与 formal_protocol"
    assert co < fp, "collection_only 检查应早于 formal_protocol，保证诊断工件不被误标为正式"


# ---- 7) portfolio 三段 radio + get/setSelectedRiskProfile + 不依赖 select.value ----
def test_portfolio_radio_segments_and_no_select_value() -> None:
    html = _read(TEMPLATES_DIR / "portfolio.html")
    radios = re.findall(r'<input type="radio" name="risk-profile" value="(\w+)"', html)
    assert radios == ["conservative", "balanced", "aggressive"], "应有三段顺序 radio"

    js = _read(STATIC_DIR / "portfolio.js")
    assert "function getSelectedRiskProfile" in js
    assert "function setSelectedRiskProfile" in js
    assert re.search(r'risk-profile["\']\)\.value', js) is None, (
        "不应通过 select.value 读取风险档案；应使用 radio"
    )


# ---- 8) workbench localStorage 包裹 try/catch ----
def test_workbench_js_localstorage_try_catch() -> None:
    js = _read(STATIC_DIR / "workbench.js")
    assert "localStorage" in js
    assert re.search(r"try\s*{[^}]*localStorage", js, flags=re.DOTALL) is not None, (
        "localStorage 访问应包裹 try/catch"
    )


# ---- 9) plotly_asset 返回类型注解 + 本地 200 ----
def test_plotly_asset_has_return_annotation_and_local_200(client) -> None:
    import webui.app as app_module

    fn = app_module.plotly_asset
    sig = inspect.signature(fn)
    assert sig.return_annotation is not inspect.Parameter.empty, (
        "plotly_asset 应带返回类型注解"
    )
    resp = client.get("/assets/plotly.min.js")
    assert resp.status_code == 200
    assert resp.mimetype == "application/javascript"
    assert len(resp.get_data()) > 10000


# ---- 10) 相关 WebUI 与本测试文件 UTF-8 无 BOM ----
@pytest.mark.parametrize(
    "path",
    ALL_HTML + ALL_CSS + ALL_JS + [Path(__file__)],
    ids=lambda p: p.name,
)
def test_webui_assets_and_self_no_bom(path: Path) -> None:
    assert not _starts_with_bom(path), f"{path} 含 UTF-8 BOM"
    _read(path)