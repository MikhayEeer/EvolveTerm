#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_moved{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_invariant_files(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    moved_files = 0
    inv_dirs = list(root.rglob("extract_result/invariant_result"))
    for inv_dir in inv_dirs:
        if not inv_dir.is_dir():
            continue
        extract_dir = inv_dir.parent
        super_dir = extract_dir.parent

        for entry in inv_dir.iterdir():
            if not entry.is_file():
                continue
            dest = unique_destination(super_dir / entry.name)
            shutil.move(str(entry), str(dest))
            moved_files += 1

        try:
            inv_dir.rmdir()
        except OSError:
            pass
        try:
            extract_dir.rmdir()
        except OSError:
            pass

    return moved_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move files from <Super>/extract_result/invariant_result/ to <Super>/ and remove empty dirs."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to search for extract_result/invariant_result directories",
    )
    args = parser.parse_args()

    try:
        moved = move_invariant_files(args.root)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    print(f"[Info] Moved {moved} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
