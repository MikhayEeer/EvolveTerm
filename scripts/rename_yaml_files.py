#!/usr/bin/env python3
"""
Batch-rename YAML files by removing specific substrings from filenames.
任务：
1. 支持多个目录指定
2. -r遍历目录，对目录下的yaml文件进行操作
3. 删除命名中的_qwen3-max_auto字符串
4. 删除命名中的_pmt_yamlv2字符串
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REMOVE_TOKENS = ("_qwen3-max_auto", "_pmt_yamlv2")


def iter_yaml_files(root: Path, recursive: bool) -> list[Path]:
    files: list[Path] = []
    patterns = ("*.yml", "*.yaml")
    if recursive:
        for pattern in patterns:
            files.extend(root.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(root.glob(pattern))
    return [p for p in files if p.is_file()]


def build_new_name(filename: str) -> str:
    new_name = filename
    for token in REMOVE_TOKENS:
        new_name = new_name.replace(token, "")
    return new_name


def rename_file(path: Path, dry_run: bool) -> bool:
    new_name = build_new_name(path.name)
    if new_name == path.name:
        return False
    new_path = path.with_name(new_name)
    if new_path.exists():
        print(f"[skip] target exists: {new_path}", file=sys.stderr)
        return False
    if dry_run:
        print(f"[dry-run] {path} -> {new_path}")
        return True
    path.rename(new_path)
    print(f"[renamed] {path} -> {new_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-rename YAML files by removing specific substrings. 支持批量改名。"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览改名结果，不实际重命名文件。",
    )
    args = parser.parse_args()

    renamed = 0
    for directory in args.directories:
        root = Path(directory)
        if not root.exists() or not root.is_dir():
            print(f"[error] not a directory: {root}", file=sys.stderr)
            continue
        for path in iter_yaml_files(root, args.recursive):
            if rename_file(path, args.dry_run):
                renamed += 1

    print(f"[done] renamed {renamed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
