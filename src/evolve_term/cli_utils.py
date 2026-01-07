"""Shared helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any
import json

from .models import KnowledgeCase

_YAML_REQUIRED_KEYS = {
    "ranking": [
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
    ],
    "inv": [
        "source_file",
        "source_path",
        "task",
        "command",
        "pmt_ver",
        "model",
        "time",
        "has_extract",
        "invariants_result",
    ],
    "ext": [
        "source_path",
        "task",
        "command",
        "pmt_ver",
        "model",
        "time",
        "loops_count",
        "loops_depth",
        "loops_ids",
        "loops"
    ],
    "feature": [
        "source_path",
        "language",
        "program_type",
        "recur_type",
        "loop_type",
        "loops_count",
        "loops_depth",
        "loop_condition_variables_count",
        "has_break",
        "loop_condition_always_true",
        "initial_sat_condition",
        "array_operator",
        "pointer_operator",
        "summary",
    ],
}


def ensure_output_dir(output: Optional[Path]) -> None:
    if output and not output.exists():
        output.mkdir(parents=True)


def collect_files(input_path: Path, recursive: bool, extensions: Optional[set[str]] = None) -> List[Path]:
    files = list(input_path.rglob("*") if recursive else input_path.glob("*"))
    files = [f for f in files if f.is_file()]
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return files


def load_references(references_file: Optional[Path]) -> List[KnowledgeCase]:
    if not references_file:
        return []
    data = json.loads(references_file.read_text(encoding="utf-8"))
    return [KnowledgeCase(**item) for item in data]


def _yaml_type_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if name.endswith(("_ranking.yml", "_ranking.yaml")):
        return "ranking"
    if name.endswith(("_inv.yml", "_inv.yaml")):
        return "inv"
    if name.endswith(("_ext.yml", "_ext.yaml")):
        return "ext"
    if name.endswith(("_feature.yml", "_feature.yaml")):
        return "feature"
    return None


def validate_yaml_required_keys(path: Path, content: Any) -> List[str]:
    yaml_type = _yaml_type_from_name(path)
    if not yaml_type:
        return []
    required = _YAML_REQUIRED_KEYS.get(yaml_type, [])
    if not isinstance(content, dict):
        return required
    return [key for key in required if key not in content]
