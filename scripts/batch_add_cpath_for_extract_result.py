#!/usr/bin/env python3
"""Add source_path to extract_result YAML outputs."""

from __future__ import annotations

from pathlib import Path
import sys
import yaml


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def derive_c_path(yaml_path: Path) -> Path | None:
    parent_dir = yaml_path.parent.parent
    stem = yaml_path.stem
    if "_pmt_" not in stem:
        return None
    base_name = stem.split("_pmt_")[0]
    candidate = parent_dir / f"{base_name}.c"
    if candidate.exists():
        return candidate
    return None


def update_yaml(yaml_path: Path, repo_root: Path) -> bool:
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(content, dict):
        return False
    c_path = derive_c_path(yaml_path)
    if not c_path:
        return False
    try:
        rel_path = str(c_path.relative_to(repo_root))
    except ValueError:
        rel_path = str(c_path)
    if content.get("source_path") == rel_path:
        return False
    content["source_path"] = rel_path
    yaml_path.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return True


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else find_repo_root()
    repo_root = find_repo_root()
    extract_dirs = [p for p in root.rglob("extract_result") if p.is_dir()]
    updated = 0
    skipped = 0
    for extract_dir in extract_dirs:
        for yaml_path in extract_dir.glob("*.yml"):
            if update_yaml(yaml_path, repo_root):
                updated += 1
            else:
                skipped += 1
        for yaml_path in extract_dir.glob("*.yaml"):
            if update_yaml(yaml_path, repo_root):
                updated += 1
            else:
                skipped += 1
    print(f"Updated: {updated}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
