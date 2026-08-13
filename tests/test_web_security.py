"""本地单用户 Web 服务的回环与文件访问边界。"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from webui.app import app, load_data_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _scan_for_external_links(text: str) -> list[str]:
    """返回文本中疑似 http(s) 外链或常见 CDN/字体域的片段（SVG 命名空间除外）。"""
    matches = []
    for pattern in (
        r"https?://[^\s\"'`<>)+\x0b]+",
        r"googleapis\.com",
        r"gstatic\.com",
        r"fonts\.googleapis",
        r"jsdelivr\.net",
        r"cdn\.plot",
    ):
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return [m for m in matches if "w3.org/2000/svg" not in m]


def _starts_with_bom(path: Path) -> bool:
    return path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_foreign_origin_cannot_write_or_receive_cors_headers() -> None:
    response = app.test_client().put(
        "/api/config",
        json={"prediction": {"timeout_seconds": 90}},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert "Access-Control-Allow-Origin" not in response.headers


def test_data_loader_rejects_file_outside_project_data(tmp_path: Path) -> None:
    source = tmp_path / "outside.csv"
    pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]}).to_csv(source, index=False)
    frame, error = load_data_file(str(source))
    assert frame is None
    assert error == "DATA_FILE_OUTSIDE_ALLOWED_DIRECTORY"


def test_data_loader_accepts_csv_inside_project_data(tmp_path: Path, monkeypatch) -> None:
    from webui import app as app_module

    allowed = tmp_path / "data"
    allowed.mkdir()
    source = allowed / "inside.csv"
    pd.DataFrame({"date": ["2024-01-02"], "open": [1], "high": [1], "low": [1], "close": [1]}).to_csv(source, index=False)
    monkeypatch.setattr(app_module, "DATA_DIRECTORY", allowed)
    frame, error = load_data_file(str(source))
    assert error is None
    assert frame is not None and len(frame) == 1


def test_latest_prediction_uses_trailing_window_and_future_timestamps(tmp_path: Path, monkeypatch) -> None:
    from webui import app as app_module

    allowed = tmp_path / "data"
    allowed.mkdir()
    source = allowed / "latest.csv"
    dates = pd.bdate_range("2024-01-02", periods=8)
    pd.DataFrame({
        "date": dates,
        "open": range(1, 9), "high": range(2, 10), "low": range(0, 8),
        "close": range(1, 9), "volume": 1000, "amount": 10000,
    }).to_csv(source, index=False)
    observed: dict[str, object] = {}

    class FakePredictor:
        def predict(self, **kwargs):
            observed["x_timestamp"] = list(pd.to_datetime(kwargs["x_timestamp"]))
            observed["y_timestamp"] = list(pd.to_datetime(kwargs["y_timestamp"]))
            return pd.DataFrame({
                "open": [8.0, 8.1], "high": [8.2, 8.3], "low": [7.8, 7.9],
                "close": [8.1, 8.2], "volume": [1000.0, 1000.0], "amount": [10000.0, 10000.0],
            })

    monkeypatch.setattr(app_module, "DATA_DIRECTORY", allowed)
    monkeypatch.setattr(app_module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(app_module, "predictor", FakePredictor())
    monkeypatch.setattr(app_module, "save_prediction_results", lambda **kwargs: None)

    response = app_module.app.test_client().post(
        "/api/predict",
        json={"file_path": str(source), "lookback": 4, "pred_len": 2},
    )

    assert response.status_code == 200
    assert observed["x_timestamp"] == list(dates[-4:])
    assert min(observed["y_timestamp"]) > dates[-1]
    assert response.get_json()["actual_data"] == []


def test_prediction_timeout_has_distinct_api_error(monkeypatch) -> None:
    from decision.errors import PredictionTimeoutError

    def fail(*args, **kwargs):
        raise PredictionTimeoutError("PREDICTION_TIMEOUT")

    monkeypatch.setattr("decision.engine.DecisionEngine.decision_report_v2", fail)
    response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"})

    assert response.status_code == 504
    assert response.get_json()["error"]["code"] == "PREDICTION_TIMEOUT"


def test_plotly_asset_route_serves_local_js_with_correct_mime() -> None:
    """本地 plotly.min.js 路由可访问、MIME 正确、带缓存头，不走 CDN。"""
    import os
    import plotly
    response = app.test_client().get("/assets/plotly.min.js")
    assert response.status_code == 200
    assert response.mimetype == "application/javascript"
    body = response.get_data()
    assert len(body) > 10000
    expected_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    assert len(body) == os.path.getsize(expected_path)
    assert "Cache-Control" in response.headers


def test_all_webui_html_css_js_have_no_external_links() -> None:
    """扫描 webui 全部 html/css/js，除 SVG 命名空间外不得有远程链接。"""
    paths = list(WEBUI_DIR.rglob("*.html")) + list(WEBUI_DIR.rglob("*.css")) + list(WEBUI_DIR.rglob("*.js"))
    assert paths, "应至少扫描一个 webui 文件"
    bad_files = []
    for path in paths:
        bad = _scan_for_external_links(_read_text(path))
        if bad:
            bad_files.append((path.name, bad[:5]))
    assert not bad_files, f"发现外部链接：{bad_files}"


def test_index_html_inline_script_passes_node_syntax_check() -> None:
    """根路由 index.html（旧版）内联脚本须通过 Node 语法检查。"""
    import re
    import subprocess
    import tempfile

    html = _read_text(WEBUI_DIR / "templates" / "index.html")
    match = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
    assert match, "index.html 应包含内联 script"
    script = match.group(1)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script)
        f.flush()
        path = f.name
    result = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    assert result.returncode == 0, f"index.html 内联脚本语法错误：{result.stderr}"


def test_all_webui_files_are_utf8_without_bom() -> None:
    """全部 webui html/css/js 应为 UTF-8 无 BOM。"""
    paths = list(WEBUI_DIR.rglob("*.html")) + list(WEBUI_DIR.rglob("*.css")) + list(WEBUI_DIR.rglob("*.js"))
    for path in paths:
        assert not _starts_with_bom(path), f"{path} 包含 UTF-8 BOM"
        _read_text(path)
