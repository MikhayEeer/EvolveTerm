#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(content, dict):
        return content
    return None


def extract_source_path_from_svm(content: Dict[str, Any]) -> Optional[str]:
    source_path = content.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        return source_path.strip()
    entries = content.get("svmranker_result") or []
    if isinstance(entries, list) and entries:
        entry = entries[0]
        if isinstance(entry, dict):
            entry_path = entry.get("input_source_path")
            if isinstance(entry_path, str) and entry_path.strip():
                return entry_path.strip()
    return None


def extract_dataset_standard(source_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not source_path:
        return None, None
    try:
        parts = Path(source_path).parts
    except Exception:
        return None, None
    if "data" in parts:
        idx = parts.index("data")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else None
        standard = parts[idx + 2] if idx + 2 < len(parts) else None
        return dataset, standard
    return None, None


def parse_svm_time_ms(log_path: Path) -> tuple[Optional[float], Optional[float], int]:
    if not log_path.exists():
        return None, None, 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Time For .*? Is --->\\s*([0-9.]+)\\s*ms", text)
    values = []
    for m in matches:
        try:
            values.append(float(m))
        except ValueError:
            continue
    if not values:
        return None, None, 0
    return sum(values), max(values), len(values)


def build_svm_map(root: Path) -> Dict[str, Dict[str, Any]]:
    svm_map: Dict[str, Dict[str, Any]] = {}
    for yml in list(root.rglob("*.yml")) + list(root.rglob("*.yaml")):
        content = load_yaml(yml)
        if not content:
            continue
        source_path = extract_source_path_from_svm(content)
        file_name = None
        if source_path:
            file_name = Path(source_path).name
        else:
            file_name = yml.stem
        bucket = yml.parent.name

        dataset, standard = extract_dataset_standard(source_path)
        log_path = yml.with_suffix(".svmranker.txt")
        time_total, time_max, time_count = parse_svm_time_ms(log_path)

        entries = content.get("svmranker_result") or []
        statuses = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status:
                    statuses.append(str(status))

        svm_map[file_name] = {
            "file": file_name,
            "dataset": dataset,
            "standard": standard,
            "SVMRresult": bucket,
            "svmr_statuses": ",".join(sorted(set(statuses))),
            "svmr_time_ms_total": time_total,
            "svmr_time_ms_max": time_max,
            "svmr_time_ms_count": time_count,
        }
    return svm_map


def is_recur_from_feature(content: Dict[str, Any]) -> Optional[bool]:
    program_type = str(content.get("program_type") or "").strip().lower()
    recur_type = content.get("recur_type")
    if program_type == "recur":
        return True
    if recur_type is None:
        return False
    if isinstance(recur_type, str) and recur_type.strip().lower() in {"none", "null", ""}:
        return False
    return True


def build_feature_map(root: Path) -> Dict[str, Dict[str, Any]]:
    feature_map: Dict[str, Dict[str, Any]] = {}
    for yml in list(root.rglob("*.yml")) + list(root.rglob("*.yaml")):
        content = load_yaml(yml)
        if not content:
            continue
        source_path = content.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            file_name = Path(source_path).name
        else:
            file_name = yml.stem.replace("_feature", "")
        feature_map[file_name] = {
            "file": file_name,
            "is_recur": is_recur_from_feature(content),
            "program_type": content.get("program_type"),
            "recur_type": content.get("recur_type"),
            "loop_type": content.get("loop_type"),
            "loops_count": content.get("loops_count"),
            "loops_depth": content.get("loops_depth"),
            "loop_condition_variables_count": content.get("loop_condition_variables_count"),
            "has_break": content.get("has_break"),
            "loop_condition_always_true": content.get("loop_condition_always_true"),
            "initial_sat_condition": content.get("initial_sat_condition"),
            "array_operator": content.get("array_operator"),
            "pointer_operator": content.get("pointer_operator"),
            "lines": content.get("lines"),
            "language": content.get("language"),
        }
    return feature_map


def main() -> int:
    svm_root = Path("results/tpdb_known/term/svmr_rft-inv-extr_0122")
    feature_root = Path("results/tpdb_known/term/FEATUREs_glm47")
    out_path = Path("results/tpdb_known/term/svmr_features_summary.csv")

    if not svm_root.exists() or not feature_root.exists():
        print("Missing input directories.")
        return 1

    svm_map = build_svm_map(svm_root)
    feature_map = build_feature_map(feature_root)

    files = sorted(set(svm_map.keys()) | set(feature_map.keys()))

    rows = []
    for name in files:
        row = {
            "file": name,
            "dataset": None,
            "standard": None,
            "SVMRresult": None,
            "is_recur": None,
            "program_type": None,
            "recur_type": None,
            "loop_type": None,
            "loops_count": None,
            "loops_depth": None,
            "loop_condition_variables_count": None,
            "has_break": None,
            "loop_condition_always_true": None,
            "initial_sat_condition": None,
            "array_operator": None,
            "pointer_operator": None,
            "lines": None,
            "language": None,
            "svmr_statuses": None,
            "svmr_time_ms_total": None,
            "svmr_time_ms_max": None,
            "svmr_time_ms_count": None,
        }
        if name in svm_map:
            row.update(svm_map[name])
        if name in feature_map:
            row.update(feature_map[name])
        if not row["dataset"]:
            row["dataset"] = "tpdb_known"
        if not row["standard"]:
            row["standard"] = "term"
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
