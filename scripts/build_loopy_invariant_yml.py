#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def parse_invariants(raw: str) -> List[str]:
    if raw is None:
        return []
    text = str(raw).replace("\r", "").strip()
    if not text:
        return []
    pieces: List[str] = []
    for chunk in text.split(";"):
        if not chunk:
            continue
        for line in chunk.split("\n"):
            candidate = line.strip()
            if candidate:
                pieces.append(candidate)
    return pieces


def find_extract_yaml(root: Path, relative_path: str, stem: str) -> Optional[Path]:
    search_dir = root / relative_path
    if not search_dir.exists():
        return None
    preferred = search_dir / f"{stem}_pmt_yamlv2_extract.yml"
    if preferred.exists():
        return preferred
    fallback = search_dir / f"{stem}_pmt_yamlv1_extract.yml"
    if fallback.exists():
        return fallback
    candidates = sorted(search_dir.glob(f"{stem}*_extract.yml"))
    if candidates:
        return candidates[0]
    return None


def derive_output_path(output_dir: Path, extract_yaml: dict, fallback_stem: str) -> Path:
    source_path = extract_yaml.get("source_path") if isinstance(extract_yaml, dict) else None
    if isinstance(source_path, str) and source_path.strip():
        src_path = Path(source_path)
        parts = list(src_path.parts)
        if "loop_invariants" in parts:
            idx = parts.index("loop_invariants")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = Path(src_path.name)
        stem = rel.stem or fallback_stem
        rel = rel.with_name(f"{stem}_inv.yml")
        return output_dir / rel
    return output_dir / f"{fallback_stem}_inv.yml"


def build_output_payload(extract_yaml: dict, invariants: List[str], command: str, source_file: str) -> dict:
    loops = extract_yaml.get("loops") if isinstance(extract_yaml, dict) else None
    invariants_result = []
    if isinstance(loops, list):
        for idx, loop in enumerate(loops, start=1):
            if not isinstance(loop, dict):
                continue
            loop_id = loop.get("id") or loop.get("loop_id") or idx
            code = loop.get("code", "")
            invariants_result.append(
                {
                    "loop_id": loop_id,
                    "code": code,
                    "invariants": list(invariants),
                }
            )

    payload = {
        "source_file": source_file,
        "source_path": extract_yaml.get("source_path") or "",
        "task": "invariant_inference",
        "command": command,
        "pmt_ver": "loopy2_generated",
        "model": "loopy2",
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "has_extract": True,
        "invariants_result": invariants_result,
    }

    for key in ("loops_count", "loops_depth", "loops_ids", "loops"):
        if key in extract_yaml and key not in payload:
            payload[key] = extract_yaml[key]

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Loopy CSV invariants with extract YAML loops.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/Loopy_dataset_InvarBenchmark/loop_invariants/Loopy_Loopy2GeneratedInvariants.csv"),
        help="CSV file with generated invariants",
    )
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=Path("results/aeval+Loopy/extract_v2_qwen3max/Loopy_dataset_InvarBenchmark/loop_invariants"),
        help="Root directory containing extract YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/loopy_loop_invariants"),
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.extract_root.exists():
        raise SystemExit(f"Extract root not found: {args.extract_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    created = 0
    skipped = 0

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            relative_path = (row.get("relative_path") or "").strip()
            filename = (row.get("filename") or "").strip()
            invariants_raw = row.get("invariants")

            if not relative_path or not filename:
                skipped += 1
                continue

            stem = Path(filename).stem
            extract_path = find_extract_yaml(args.extract_root, relative_path, stem)
            if not extract_path:
                print(f"[WARN] No extract YAML for {relative_path}/{filename}")
                skipped += 1
                continue

            try:
                extract_yaml = yaml.safe_load(extract_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Failed to read {extract_path}: {exc}")
                skipped += 1
                continue

            invariants = parse_invariants(invariants_raw)
            payload = build_output_payload(extract_yaml, invariants, command, extract_path.name)

            out_path = derive_output_path(args.output_dir, extract_yaml, stem)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as out_handle:
                yaml.safe_dump(payload, out_handle, sort_keys=False, allow_unicode=True)

            created += 1

    print(f"Done. Created {created} file(s), skipped {skipped} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
