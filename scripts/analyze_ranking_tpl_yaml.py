#!/usr/bin/env python3
'''
Docstring for scripts.analyze_ranking_tpl_yaml

1. 找到只有一个循环的yaml
2. 找到有多个循环的yaml
3. 找到这轮次invariant推理失败的yaml

'''

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_ROOT = Path("results/aeval+Loopy")
DIR_PATTERN = re.compile(r"ranking_.*PmtTpl.*")


def find_target_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    targets: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and DIR_PATTERN.match(path.name):
            targets.append(path)
    return sorted(set(targets))


def extract_entries(data: Any) -> list[Any]:
    if not isinstance(data, dict):
        return []
    for key in ("ranking_results", "invariants_result", "loops"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def analyze_yaml(path: Path) -> tuple[bool, bool, bool]:
    """
    Returns:
        loopid_max_le_1, loops_gt_1, invariants_empty
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = extract_entries(data)
    if not entries:
        return False, False, False

    loop_ids: list[int] = []
    invariants_field_seen = False
    invariants_empty = False

    for idx, entry in enumerate(entries, start=1):
        loop_id = idx
        if isinstance(entry, dict):
            loop_id = entry.get("loop_id") or entry.get("id") or idx
            try:
                loop_id = int(loop_id)
            except (TypeError, ValueError):
                loop_id = idx

            if "invariants" in entry:
                invariants_field_seen = True
                invs = entry.get("invariants")
                if not invs:
                    invariants_empty = True
        loop_ids.append(loop_id)

    loopid_max_le_1 = max(loop_ids) <= 1 if loop_ids else False
    loops_gt_1 = len(entries) > 1
    invariants_empty = invariants_empty if invariants_field_seen else False
    return loopid_max_le_1, loops_gt_1, invariants_empty


def write_list(target_dir: Path, filename: str, items: Iterable[str]) -> None:
    out_path = target_dir / filename
    content = "\n".join(sorted(set(items)))
    if content:
        content += "\n"
    out_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze ranking template YAML files and generate summary lists."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing ranking_.*PmtTpl.* directories",
    )
    args = parser.parse_args()

    try:
        target_dirs = find_target_dirs(args.root)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    if not target_dirs:
        print(f"[Info] No matching directories under {args.root}")
        return 0

    for target_dir in target_dirs:
        loopid_max_files: list[str] = []
        loops_gt_files: list[str] = []
        invariants_empty_files: list[str] = []

        yaml_files = list(target_dir.rglob("*.yml")) + list(target_dir.rglob("*.yaml"))
        for yml in yaml_files:
            try:
                loopid_max_le_1, loops_gt_1, inv_empty = analyze_yaml(yml)
            except Exception as exc:
                print(f"[Warning] Failed to parse {yml}: {exc}", file=sys.stderr)
                continue

            rel_path = str(yml.relative_to(target_dir))
            if loopid_max_le_1:
                loopid_max_files.append(rel_path)
            if loops_gt_1:
                loops_gt_files.append(rel_path)
            if inv_empty:
                invariants_empty_files.append(rel_path)

        write_list(target_dir, "loopid_max_le_1.txt", loopid_max_files)
        write_list(target_dir, "loops_gt_1.txt", loops_gt_files)
        write_list(target_dir, "invariants_empty.txt", invariants_empty_files)

        print(
            f"[Info] {target_dir}: loopid_max_le_1={len(loopid_max_files)}, "
            f"loops_gt_1={len(loops_gt_files)}, invariants_empty={len(invariants_empty_files)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
