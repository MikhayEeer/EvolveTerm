"""Shared helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import json

from .models import KnowledgeCase


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
