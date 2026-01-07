#!/usr/bin/env python3
'''
Aim:
- 对一个目录下所有yaml进行遍历，展开\n，保证可读性
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


class LiteralDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        if isinstance(value, str) and "\n" in value and tag == "tag:yaml.org,2002:str":
            style = "|"
        return super().represent_scalar(tag, value, style)


def convert_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\\n", "\n")
    if isinstance(value, list):
        return [convert_value(item) for item in value]
    if isinstance(value, dict):
        return {key: convert_value(val) for key, val in value.items()}
    return value


def iter_yaml_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.yml")) + list(root.rglob("*.yaml"))
    return [p for p in files if p.is_file()]


def process_file(path: Path, dry_run: bool) -> bool:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] failed to parse {path}: {exc}", file=sys.stderr)
        return False

    if content is None:
        print(f"[skip] empty YAML: {path}", file=sys.stderr)
        return False

    updated = convert_value(content)
    if dry_run:
        print(f"[dry-run] update {path}")
        return True

    path.write_text(
        yaml.safe_dump(updated, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[updated] {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert literal \\n sequences in YAML to real newlines and dump with block style.",
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more root directories to scan recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    args = parser.parse_args()

    updated_count = 0
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists() or not root.is_dir():
            print(f"[error] root not found: {root}", file=sys.stderr)
            continue
        for path in iter_yaml_files(root):
            if process_file(path, args.dry_run):
                updated_count += 1

    print(f"[done] processed {updated_count} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
