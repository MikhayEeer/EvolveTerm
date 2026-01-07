#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = "results/ranking_results_PmtTpl_qwen3max"
DEFAULT_COMMAND = (
    "evolveterm ranking --input results/invariant_results_PmtCot_ExtPmtV2_qwen3max/ "
    "-r -m yaml --ranking-mode template --retry-empty 2"
)
DEFAULT_PMT_VER = "template"
DEFAULT_MODEL = "qwen3max"
DEFAULT_TIME = "2026/01/06"
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
    "has_invariants",
    "ranking_results",
]

EMPTY_INVARIANTS_STRINGS = {"", "[]", "none", "empty", "error", "null"}


def is_invariants_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        if not value:
            return False
        for item in value:
            if isinstance(item, str):
                if item.strip() and item.strip().lower() not in EMPTY_INVARIANTS_STRINGS:
                    return True
            elif item is not None:
                return True
        return False
    if isinstance(value, str):
        return value.strip().lower() not in EMPTY_INVARIANTS_STRINGS
    return bool(value)


def compute_has_invariants(results: Any) -> bool:
    if not isinstance(results, list):
        return False
    for item in results:
        if isinstance(item, dict) and is_invariants_non_empty(item.get("invariants")):
            return True
    return False


def order_and_fill(
    content: dict[str, Any],
    command: str,
    pmt_ver: str,
    has_extract: bool,
) -> dict[str, Any]:
    results = content.get("ranking_results")
    if results is None:
        results = []
    has_invariants = compute_has_invariants(results)

    content["command"] = command
    content["pmt_ver"] = pmt_ver
    content["model"] = DEFAULT_MODEL
    content["time"] = DEFAULT_TIME
    content["has_extract"] = has_extract
    content["has_invariants"] = has_invariants
    content["ranking_results"] = results

    ordered: dict[str, Any] = {}
    for key in ORDERED_KEYS:
        ordered[key] = content.get(key)
    for key, value in content.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def iter_yaml_files(root: Path) -> list[Path]:
    files = list(root.rglob("*_ranking.yml")) + list(root.rglob("*_ranking.yaml"))
    return [p for p in files if p.is_file()]


def process_file(path: Path, dry_run: bool, command: str, pmt_ver: str, has_extract: bool) -> bool:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] failed to parse {path}: {exc}", file=sys.stderr)
        return False

    if not isinstance(content, dict):
        print(f"[skip] {path} is not a YAML mapping", file=sys.stderr)
        return False

    updated = order_and_fill(content, command, pmt_ver, has_extract)
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
        description="Fill ranking YAML metadata for *_ranking.yml files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="Command string to store in YAML.",
    )
    parser.add_argument(
        "--pmt-ver",
        default=DEFAULT_PMT_VER,
        help=f"Prompt version value to store in YAML. default:{DEFAULT_PMT_VER}",
    )
    parser.add_argument(
        "--has-extract",
        default=str(DEFAULT_HAS_EXTRACT).lower(),
        help="Whether the YAML is derived from extract results (true/false). default : true",
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

    has_extract_str = str(args.has_extract).strip().lower()
    has_extract = has_extract_str in {"1", "true", "yes", "y", "on"}

    files = iter_yaml_files(root)
    if not files:
        print("[done] no matching files found.")
        return 0

    updated_count = 0
    for path in files:
        if process_file(path, args.dry_run, args.command, args.pmt_ver, has_extract):
            updated_count += 1

    print(f"[done] processed {updated_count} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
