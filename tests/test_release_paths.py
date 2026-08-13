"""发布前轻量断言：确保发布文件不含个人绝对路径或非回环提示。"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 发布审查范围内的文件
RELEASE_FILES = [
    REPO_ROOT / "PROJECT_DOCS.md",
    REPO_ROOT / "docs" / "usage.md",
    REPO_ROOT / "scripts" / "daily_recommendations.py",
    REPO_ROOT / "webui" / "run.py",
    REPO_ROOT / "webui" / "start.sh",
]

# 禁止出现在发布文件中的个人/环境相关片段
FORBIDDEN_SUBSTRINGS = (
    "C:\\Users\\xzh",   # 个人用户目录
    "C:/Users/xzh",     # 正斜杠变体
    "D:\\Hermes",       # 个人工作盘
    "D:/Hermes",
    "D:\\data",         # 个人数据盘
    "D:/data",
    "localhost:7070",   # 应统一为回环地址 127.0.0.1
)


def test_release_files_exist() -> None:
    """发布文件必须存在。"""
    missing = [str(path) for path in RELEASE_FILES if not path.exists()]
    assert not missing, f"缺少发布文件：{missing}"


def test_release_files_have_no_personal_paths_or_localhost() -> None:
    """发布文件不得包含个人绝对路径或 localhost:7070 提示。"""
    offenders: list[str] = []
    for path in RELEASE_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}：'{needle}'")
    assert not offenders, "发布文件含禁止片段：\n" + "\n".join(offenders)


def test_release_files_are_utf8_without_bom() -> None:
    """发布文件必须为 UTF-8 无 BOM。"""
    for path in RELEASE_FILES:
        if not path.exists():
            continue
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} 含 UTF-8 BOM"
        raw.decode("utf-8")