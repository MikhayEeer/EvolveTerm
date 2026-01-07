#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = "results/invariant_results_PmtCot_ExPmtV2_qwen3max"
DEFAULT_COMMAND = "evolveterm invariant --input data/aeval/ -r -m yaml --extv v2 -pv yaml_cot"
DEFAULT_PMT_VER = "yaml_cot"
DEFAULT_MODEL = "qwen3max"
DEFAULT_HAS_EXTRACT = True

ORDERED_KEYS = [
    "source_file",
    "source_path",
    "task",
    "command",
    "pmt_ver",
    "model",
    "time",
    "has_extract",
    "invariants_result",
]


def order_and_fill(content: dict[str, Any]) -> dict[str, Any]:
    basic = content.get("basic")
    time_value = ""
    if isinstance(basic, dict):
        time_value = basic.get("time") or ""
    else:
        existing_time = content.get("time")
        if isinstance(existing_time, str):
            time_value = existing_time

    if not time_value:
        print("[warn] missing basic.time; using empty time value", file=sys.stderr)

    content.pop("basic", None)
    content["task"] = "invariant_inference"
    content["command"] = DEFAULT_COMMAND
    content["pmt_ver"] = DEFAULT_PMT_VER
    content["model"] = DEFAULT_MODEL
    content["time"] = time_value
    content["has_extract"] = DEFAULT_HAS_EXTRACT

    ordered: dict[str, Any] = {}
    for key in ORDERED_KEYS:
        ordered[key] = content.get(key)
    for key, value in content.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def iter_yaml_files(root: Path) -> list[Path]:
    files = list(root.rglob("*_inv.yml")) + list(root.rglob("*_inv.yaml"))
    return [p for p in files if p.is_file()]


def process_file(path: Path, dry_run: bool) -> bool:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] failed to parse {path}: {exc}", file=sys.stderr)
        return False

    if not isinstance(content, dict):
        print(f"[skip] {path} is not a YAML mapping", file=sys.stderr)
        return False

    updated = order_and_fill(content)
    if dry_run:
        print(f"[dry-run] update {path}")
        return True

    path.write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[updated] {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize *_inv.yml metadata fields by removing basic and adding standard keys."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"[error] root directory not found: {root}", file=sys.stderr)
        return 1

    files = iter_yaml_files(root)
    if not files:
        print("[done] no matching files found.")
        return 0

    updated_count = 0
    for path in files:
        if process_file(path, args.dry_run):
            updated_count += 1

    print(f"[done] processed {updated_count} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
