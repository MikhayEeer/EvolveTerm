"""Shared helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any, Dict, Set, Tuple
import json
import re

from .config import auto_load_json_config
from .llm_client import build_llm_client
from .models import KnowledgeCase
from .yaml_schema import (
    validate_yaml_file,
    validate_yaml_content,
    get_missing_required_keys,
    ValidationResult,
)
import typer
import yaml

DEFAULT_LLM_PING_PROMPT = "ping"
_LOOP_REF_PATTERN = re.compile(r"LOOP\s*(?:\{|\()?\s*(\d+)\s*(?:\}|\))?", re.IGNORECASE)

def resolve_svm_ranker_root(path: Path) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"SVMRanker 路径不存在: {path}")
    if path.is_file():
        if path.name == "CLIMain.py" and path.parent.name == "src":
            path = path.parent.parent
        else:
            raise typer.BadParameter("SVMRanker 路径应为仓库根目录或 src/CLIMain.py 文件。")

    if (path / "src" / "CLIMain.py").exists():
        return path
    if path.name == "src" and (path / "CLIMain.py").exists():
        return path.parent
    raise typer.BadParameter("SVMRanker 路径无效，未找到 src/CLIMain.py。")

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
    """Load reference cases from JSON or YAML file.
    
    Supports both .json and .yml/.yaml formats.
    """
    if not references_file:
        return []
    
    content = references_file.read_text(encoding="utf-8")
    
    # Auto-detect format by extension
    if references_file.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    
    return [KnowledgeCase(**item) for item in data]


def load_json_or_yaml(file_path: Path) -> Any:
    """Load data from JSON or YAML file, auto-detecting format by extension.
    
    Args:
        file_path: Path to .json, .yml, or .yaml file
        
    Returns:
        Parsed data structure (dict, list, etc.)
        
    Raises:
        Exception: If file cannot be parsed
    """
    content = file_path.read_text(encoding="utf-8")
    
    if file_path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(content)
    else:
        return json.loads(content)


def _yaml_type_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if name.endswith(("_ranking.yml", "_ranking.yaml")):
        return "ranking"
    if name.endswith(("_inv.yml", "_inv.yaml", "_invariant.yml", "_invariant.yaml")):
        return "inv"
    if name.endswith(("_extract.yml", "_extract.yaml", "_ext.yml", "_ext.yaml")):
        return "ext"
    if name.endswith(("_feature.yml", "_feature.yaml")):
        return "feature"
    return None


def validate_yaml_required_keys(path: Path, content: Any) -> List[str]:
    """Legacy function: get list of missing required keys.
    
    This function is maintained for backward compatibility.
    New code should use yaml_schema.validate_yaml_file() or validate_yaml_content().
    """
    return get_missing_required_keys(path, content)


def ping_llm_client(llm_config: str, prompt: str = DEFAULT_LLM_PING_PROMPT) -> dict[str, Any]:
    config = auto_load_json_config(llm_config, tag="default")
    client = build_llm_client(llm_config, config_tag="default")
    response = client.complete(prompt)
    return {"config": config, "prompt": prompt, "response": response}


def _extract_loop_entries(content: Any) -> List[Any]:
    if not isinstance(content, dict):
        return []
    if "ranking_results" in content:
        return content.get("ranking_results") or []
    if "invariants_result" in content:
        return content.get("invariants_result") or []
    if "loops" in content:
        return content.get("loops") or []
    return []


def _normalize_loop_entries(entries: List[Any]) -> List[Tuple[int, str]]:
    normalized: List[Tuple[int, str]] = []
    for idx, entry in enumerate(entries, start=1):
        loop_id = idx
        code = ""
        if isinstance(entry, dict):
            loop_id = entry.get("loop_id") or entry.get("id") or idx
            code = entry.get("code", "")
        else:
            code = str(entry)
        try:
            loop_id = int(loop_id)
        except (TypeError, ValueError):
            loop_id = idx
        normalized.append((loop_id, str(code)))
    return normalized


def _collect_loop_references(code: str) -> Set[int]:
    refs = set()
    for match in _LOOP_REF_PATTERN.findall(code or ""):
        try:
            refs.add(int(match))
        except (TypeError, ValueError):
            continue
    return refs


def check_loop_id_order_in_yaml(path: Path) -> Tuple[Dict[int, Set[int]], List[str]]:
    """
    Build loop dependency graph based on LOOP{n} placeholders and
    return warnings for forward references (referencing loops not yet seen).
    """
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = _extract_loop_entries(content)
    loop_entries = _normalize_loop_entries(entries)
    deps: Dict[int, Set[int]] = {}
    warnings: List[str] = []
    seen: Set[int] = set()

    for loop_id, code in loop_entries:
        refs = _collect_loop_references(code)
        deps[loop_id] = refs
        for ref_id in sorted(refs):
            if ref_id not in seen:
                warnings.append(
                    f"Loop {loop_id} references LOOP{ref_id} before it appears."
                )
        seen.add(loop_id)

    return deps, warnings
