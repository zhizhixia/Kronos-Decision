"""只读核验跨电脑交接文件；不导入项目、安装依赖或访问网络。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath


def checked_path(root: Path, relative: str) -> Path:
    """限制清单路径位于包内，拒绝非规范路径和重解析点。"""
    if not isinstance(relative, str) or not relative:
        raise ValueError("清单路径必须是非空字符串")
    logical = PurePosixPath(relative)
    if (logical.is_absolute() or str(logical) != relative
            or any(part in ("..", ".git") for part in logical.parts)
            or any(char in relative for char in ("\\", ":", "\x00"))):
        raise ValueError("清单中存在不允许的路径")
    target = root
    for part in logical.parts:
        target = target / part
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("清单路径含符号链接或 Windows 重解析点")
    if not target.is_file():
        raise ValueError("清单目标不是普通文件")
    return target


def file_digest(path: Path) -> tuple[int, str]:
    """流式计算实际读取字节数与 SHA-256，不依赖文件修改时间。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify(root: Path) -> tuple[int, list[str]]:
    """校验清单覆盖的每个文件，返回核验数量与不含文件内容的错误。"""
    manifest_path = checked_path(root, "verification/MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("不支持的清单版本")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("清单文件列表为空或格式错误")
    errors: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("清单条目不是对象")
        relative = entry.get("path")
        expected_size, expected_hash = entry.get("size"), entry.get("sha256")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("清单路径缺失、格式错误或重复")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError("清单文件大小格式错误")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("清单 SHA-256 格式错误")
        seen.add(relative)
        try:
            path = checked_path(root, relative)
            size, digest = file_digest(path)
            if size != expected_size or digest != expected_hash:
                errors.append(f"内容不一致：{relative}")
        except (OSError, ValueError) as exc:
            errors.append(f"无法校验：{relative}（{type(exc).__name__}）")
    return len(entries), errors


def main() -> int:
    """解析交接根目录并以明确退出码报告核验结果。"""
    parser = argparse.ArgumentParser(description="只读检查交接包，不启动 Kronos 项目")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="交接包根目录；默认采用脚本所在包的位置")
    args = parser.parse_args()
    try:
        count, errors = verify(args.root.resolve(strict=True))
    except (OSError, ValueError, TypeError) as exc:
        print(f"交接校验失败：清单或根目录不可用（{type(exc).__name__}）", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"交接校验失败：检查 {count} 个文件，异常 {len(errors)} 个。", file=sys.stderr)
        return 1
    print(f"交接文件校验通过：{count} 个文件的字节数与 SHA-256 一致。")
    print("范围说明：不证明软件可运行、策略有效或未列出的新增文件安全；清单不是数字签名。")
    print("Git 管理文件另用 git fsck 检查；本工具不运行项目代码或修改任何文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
