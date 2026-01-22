#!/usr/bin/env python3
import copy
import sys
from pathlib import Path
from typing import Optional
import yaml


def load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_yaml(path: Path, data) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def normalize_source_path_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).as_posix()
    except Exception:
        return text


def resolve_input_file(input_file: str, base_dir: Path) -> Optional[Path]:
    candidates = [Path(input_file)]
    if not candidates[0].is_absolute():
        candidates.append(base_dir / input_file)
        candidates.append(Path.cwd() / input_file)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_loop_map(template_content):
    entries = template_content.get("ranking_results") or []
    loop_map = {}
    if not isinstance(entries, list):
        return loop_map
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        loop_id = entry.get("loop_id") or entry.get("id")
        if loop_id is None:
            continue
        loop_map[str(loop_id)] = entry
    return loop_map


def enrich_file(path: Path) -> bool:
    content = load_yaml(path)
    if not isinstance(content, dict):
        return False
    input_file = content.get("input_file")
    if not isinstance(input_file, str) or not input_file.strip():
        return False

    template_path = resolve_input_file(input_file, path.parent)
    if not template_path:
        return False
    template_content = load_yaml(template_path)
    if not isinstance(template_content, dict):
        return False

    source_path = normalize_source_path_value(template_content.get("source_path"))
    if source_path:
        content["source_path"] = source_path

    loop_map = build_loop_map(template_content)
    entries = content.get("svmranker_result") or []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if source_path and not entry.get("input_source_path"):
                entry["input_source_path"] = source_path
            loop_id = entry.get("loop_id") or entry.get("id")
            if loop_id is None:
                continue
            loop_key = str(loop_id)
            if loop_key in loop_map:
                entry["input_loop"] = copy.deepcopy(loop_map[loop_key])

    save_yaml(path, content)
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/tmp_enrich_svmranker_yaml.py <svmranker_output_dir>")
        return 1
    root = Path(sys.argv[1])
    if not root.exists() or not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    files = list(root.rglob("*.yml")) + list(root.rglob("*.yaml"))
    total = len(files)
    updated = 0
    skipped = 0
    for path in files:
        if enrich_file(path):
            updated += 1
        else:
            skipped += 1

    print(f"Total: {total}  Updated: {updated}  Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
