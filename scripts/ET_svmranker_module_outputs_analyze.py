#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_RESULTS = Path("results/aeval+Loopy/svmranker_results_0110_aeval_term")
STATUS_PRIORITY = ("ERROR", "NONTERM", "TERMINATE", "UNKNOWN", "MISSING")
BOOL_ORDER = ("true", "false", "missing")
CATEGORY_ORDER = ("term", "nonterm", "unknown")


def load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except Exception:
        return yaml.unsafe_load(text)


def normalize_status(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "MISSING"
    return re.sub(r"[\s\-]+", "", raw).upper()


def normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "missing"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return "true"
        if lowered in {"false", "no", "0"}:
            return "false"
        if lowered == "":
            return "missing"
        return lowered
    return str(value).lower()


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def infer_file_label(statuses: Iterable[str]) -> str:
    status_set = set(statuses)
    for status in STATUS_PRIORITY:
        if status in status_set:
            return status
    return "OTHER"


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


def extract_svm_entries(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    results = data.get("svmranker_result") or data.get("results") or []
    if not isinstance(results, list):
        return []
    return [entry for entry in results if isinstance(entry, dict)]


def extract_input_entries(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    for key in ("ranking_results", "invariants_result", "loops"):
        value = data.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


def resolve_input_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return base_dir / path


def count_invariants(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return sum(1 for item in value if item is not None and str(item).strip())
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 1


def log_has_error(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    markers = ("[Error]", "Traceback", "Exception", "ERROR")
    return any(marker in text for marker in markers)


def init_stats() -> dict[str, Any]:
    return {
        "files": 0,
        "yaml_ok": 0,
        "yaml_fail": 0,
        "loops": 0,
        "file_statuses": Counter(),
        "loop_statuses": Counter(),
        "rf_present": 0,
        "rf_missing": 0,
        "rfs_nonempty": 0,
        "rfs_empty": 0,
        "template_types": Counter(),
        "svm_modes": Counter(),
        "template_depths": Counter(),
        "categories": Counter(),
        "log_files_present": 0,
        "log_files_missing": 0,
        "log_files_error": 0,
        "input_file_refs": 0,
        "input_file_missing": 0,
        "input_file_not_found": 0,
        "input_file_parse_fail": 0,
        "input_files_parsed": 0,
        "input_has_invariants": Counter(),
        "input_has_extract": Counter(),
        "input_entries": 0,
        "input_invariants_total": 0,
        "input_invariants_nonempty_loops": 0,
        "input_invariants_empty_loops": 0,
        "input_invariants_missing_loops": 0,
        "input_invariants_any_files": 0,
        "input_invariants_empty_files": 0,
        "input_invariants_missing_files": 0,
    }


def merge_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Counter):
            target[key].update(value)
        else:
            target[key] += value


def analyze_bucket(bucket_dir: Path, base_dir: Path) -> dict[str, Any]:
    stats = init_stats()
    yml_files = list(bucket_dir.rglob("*.yml")) + list(bucket_dir.rglob("*.yaml"))
    stats["files"] = len(yml_files)

    for path in yml_files:
        try:
            data = load_yaml(path)
        except Exception:
            stats["yaml_fail"] += 1
            continue

        stats["yaml_ok"] += 1
        entries = extract_svm_entries(data)
        stats["loops"] += len(entries)

        statuses: list[str] = []
        for entry in entries:
            status = normalize_status(entry.get("status"))
            statuses.append(status)
            stats["loop_statuses"][status] += 1

            if is_present(entry.get("ranking_function")):
                stats["rf_present"] += 1
            else:
                stats["rf_missing"] += 1

            rf_list = entry.get("ranking_functions")
            if isinstance(rf_list, list) and rf_list:
                stats["rfs_nonempty"] += 1
            else:
                stats["rfs_empty"] += 1

            template_type = entry.get("template_type") or "MISSING"
            svm_mode = entry.get("svm_mode") or "MISSING"
            depth = entry.get("template_depth")
            depth_key = str(depth) if depth is not None else "MISSING"
            stats["template_types"][str(template_type)] += 1
            stats["svm_modes"][str(svm_mode)] += 1
            stats["template_depths"][depth_key] += 1

        stats["file_statuses"][infer_file_label(statuses)] += 1

        category = detect_aeval_category(collect_candidate_paths(data)) or "unknown"
        stats["categories"][category] += 1

        log_path = path.with_suffix(".svmranker.txt")
        if log_path.exists():
            stats["log_files_present"] += 1
            if log_has_error(log_path):
                stats["log_files_error"] += 1
        else:
            stats["log_files_missing"] += 1

        if isinstance(data, dict) and data.get("input_file"):
            stats["input_file_refs"] += 1
            raw_input = data.get("input_file")
            if isinstance(raw_input, str):
                input_path = resolve_input_path(raw_input, base_dir)
                if not input_path.exists():
                    stats["input_file_not_found"] += 1
                else:
                    try:
                        input_data = load_yaml(input_path)
                    except Exception:
                        stats["input_file_parse_fail"] += 1
                    else:
                        stats["input_files_parsed"] += 1
                        inv_flag = normalize_bool(
                            input_data.get("has_invariants")
                            if isinstance(input_data, dict)
                            else None
                        )
                        stats["input_has_invariants"][inv_flag] += 1
                        extract_flag = normalize_bool(
                            input_data.get("has_extract")
                            if isinstance(input_data, dict)
                            else None
                        )
                        stats["input_has_extract"][extract_flag] += 1

                        input_entries = extract_input_entries(input_data)
                        stats["input_entries"] += len(input_entries)
                        inv_seen = False
                        inv_any = False
                        inv_empty = False
                        for entry in input_entries:
                            if "invariants" in entry:
                                inv_seen = True
                                inv_count = count_invariants(entry.get("invariants"))
                                stats["input_invariants_total"] += inv_count
                                if inv_count > 0:
                                    stats["input_invariants_nonempty_loops"] += 1
                                    inv_any = True
                                else:
                                    stats["input_invariants_empty_loops"] += 1
                                    inv_empty = True
                            else:
                                stats["input_invariants_missing_loops"] += 1
                        if not input_entries:
                            stats["input_invariants_missing_files"] += 1
                        elif inv_seen:
                            if inv_any:
                                stats["input_invariants_any_files"] += 1
                            elif inv_empty:
                                stats["input_invariants_empty_files"] += 1
                            else:
                                stats["input_invariants_missing_files"] += 1
                        else:
                            stats["input_invariants_missing_files"] += 1
            else:
                stats["input_file_not_found"] += 1
        else:
            stats["input_file_missing"] += 1

    return stats


def find_default_results() -> Path | None:
    if DEFAULT_RESULTS.is_dir() and (DEFAULT_RESULTS / "certain").is_dir():
        return DEFAULT_RESULTS
    base = Path("results")
    if not base.exists():
        return None
    candidates: list[Path] = []
    for path in base.rglob("*"):
        if (
            path.is_dir()
            and (path / "certain").is_dir()
            and (path / "failed").is_dir()
            and (path / "unknown").is_dir()
        ):
            candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def format_counter(counter: Counter, order: Iterable[str] | None = None) -> str:
    items: list[str] = []
    if order:
        for key in order:
            if key in counter:
                items.append(f"{key}={counter[key]}")
    for key in sorted(counter.keys()):
        if order and key in set(order):
            continue
        items.append(f"{key}={counter[key]}")
    return ", ".join(items) if items else "none"


def print_bucket(name: str, stats: dict[str, Any]) -> None:
    loops = stats["loops"]
    files = stats["files"]
    avg_loops = f"{(loops / files):.2f}" if files else "0"

    print(f"[{name}] files={files}, yaml_ok={stats['yaml_ok']}, yaml_fail={stats['yaml_fail']}, loops={loops}, avg_loops={avg_loops}")
    print(f"[{name}] loop_statuses: {format_counter(stats['loop_statuses'], STATUS_PRIORITY)}")
    print(f"[{name}] file_statuses: {format_counter(stats['file_statuses'], STATUS_PRIORITY)}")
    print(
        f"[{name}] rf: present={stats['rf_present']}, missing={stats['rf_missing']}, "
        f"list_nonempty={stats['rfs_nonempty']}, list_empty={stats['rfs_empty']}"
    )
    print(f"[{name}] template_type: {format_counter(stats['template_types'])}")
    print(f"[{name}] template_depth: {format_counter(stats['template_depths'])}")
    print(f"[{name}] svm_mode: {format_counter(stats['svm_modes'])}")
    print(f"[{name}] categories: {format_counter(stats['categories'], CATEGORY_ORDER)}")
    print(
        f"[{name}] logs: present={stats['log_files_present']}, missing={stats['log_files_missing']}, "
        f"error_marked={stats['log_files_error']}"
    )
    print(
        f"[{name}] input_file: refs={stats['input_file_refs']}, missing_field={stats['input_file_missing']}, "
        f"not_found={stats['input_file_not_found']}, parse_fail={stats['input_file_parse_fail']}, "
        f"parsed={stats['input_files_parsed']}"
    )
    print(
        f"[{name}] has_invariants: {format_counter(stats['input_has_invariants'], BOOL_ORDER)}"
    )
    print(
        f"[{name}] has_extract: {format_counter(stats['input_has_extract'], BOOL_ORDER)}"
    )
    print(
        f"[{name}] input_invariants: total={stats['input_invariants_total']}, "
        f"loops_nonempty={stats['input_invariants_nonempty_loops']}, "
        f"loops_empty={stats['input_invariants_empty_loops']}, "
        f"loops_missing={stats['input_invariants_missing_loops']}"
    )
    print(
        f"[{name}] input_invariants_files: any={stats['input_invariants_any_files']}, "
        f"empty={stats['input_invariants_empty_files']}, missing={stats['input_invariants_missing_files']}"
    )


def to_jsonable(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SVMRanker output directories (certain/failed/unknown)."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="SVMRanker results root (contains certain/unknown/failed).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    results_root = args.results or find_default_results()
    if not results_root or not results_root.is_dir():
        print("[Error] results root not found; pass --results", file=sys.stderr)
        return 1

    certain_dir = results_root / "certain"
    failed_dir = results_root / "failed"
    unknown_dir = results_root / "unknown"
    if not (certain_dir.is_dir() and failed_dir.is_dir() and unknown_dir.is_dir()):
        print(f"[Error] missing certain/failed/unknown under {results_root}", file=sys.stderr)
        return 1

    base_dir = Path.cwd()
    buckets = {
        "certain": analyze_bucket(certain_dir, base_dir),
        "failed": analyze_bucket(failed_dir, base_dir),
        "unknown": analyze_bucket(unknown_dir, base_dir),
    }
    total = init_stats()
    for stats in buckets.values():
        merge_stats(total, stats)

    if args.format == "json":
        payload = {
            "results_root": str(results_root),
            "buckets": {name: to_jsonable(stats) for name, stats in buckets.items()},
            "total": to_jsonable(total),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Results root: {results_root}")
    print_bucket("certain", buckets["certain"])
    print_bucket("failed", buckets["failed"])
    print_bucket("unknown", buckets["unknown"])
    print_bucket("total", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
