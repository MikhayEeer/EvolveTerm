#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_RESULTS = Path("results/aeval+Loopy/svmranker_results_0109")


def load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except Exception:
        return yaml.unsafe_load(text)


def normalize_status(value: Any) -> str:
    return re.sub(r"[\s\-]+", "", str(value or "")).upper()


def infer_file_label(statuses: Iterable[str]) -> str:
    if any(status == "NONTERM" for status in statuses):
        return "NONTERM"
    if any(status == "TERMINATE" for status in statuses):
        return "TERMINATE"
    return "UNKNOWN"


def detect_aeval_category(paths: Iterable[str]) -> str | None:
    for raw in paths:
        if not raw:
            continue
        norm = raw.replace("\\", "/").lower()
        if "/c_bench_nonterm/" in norm or "c_bench_nonterm" in norm:
            return "nonterm"
        if "/c_bench_term/" in norm or "c_bench_term" in norm:
            return "term"
    return None


def collect_candidate_paths(data: Any) -> list[str]:
    candidates: list[str] = []
    if not isinstance(data, dict):
        return candidates
    for key in ("input_file", "source_path", "source_file"):
        val = data.get(key)
        if isinstance(val, str):
            candidates.append(val)
    results = data.get("svmranker_result") or data.get("results") or []
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict):
                val = entry.get("input_code_path")
                if isinstance(val, str):
                    candidates.append(val)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count certain SVMRanker results for aeval c_bench_term/c_bench_nonterm."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="SVMRanker results root (contains certain/unknown/failed).",
    )
    args = parser.parse_args()

    results_root = args.results
    if results_root.is_dir() and (results_root / "certain").is_dir():
        certain_dir = results_root / "certain"
    else:
        print(f"[Error] certain directory not found under: {results_root}", file=sys.stderr)
        return 1

    yml_files = list(certain_dir.rglob("*.yml")) + list(certain_dir.rglob("*.yaml"))
    total_aeval = 0
    term_total = 0
    nonterm_total = 0
    term_success = 0
    nonterm_success = 0

    for path in yml_files:
        try:
            data = load_yaml(path)
        except Exception as exc:
            print(f"[Warning] Failed to parse {path}: {exc}", file=sys.stderr)
            continue

        candidates = collect_candidate_paths(data)
        category = detect_aeval_category(candidates)
        if category is None:
            continue

        results = data.get("svmranker_result") or data.get("results") or []
        statuses = [
            normalize_status(entry.get("status"))
            for entry in results
            if isinstance(entry, dict)
        ]
        file_label = infer_file_label(statuses)

        total_aeval += 1
        if category == "term":
            term_total += 1
            if file_label == "TERMINATE":
                term_success += 1
        else:
            nonterm_total += 1
            if file_label == "NONTERM":
                nonterm_success += 1

    print(f"Results root: {results_root}")
    print(f"AEVAL certain files: {total_aeval}")
    print(f"c_bench_term: total={term_total}, success={term_success}")
    print(f"c_bench_nonterm: total={nonterm_total}, success={nonterm_success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
