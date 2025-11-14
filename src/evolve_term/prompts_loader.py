"""Prompt loading and rendering helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from .exceptions import PromptNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"


class PromptRepository:
    """Loads prompt templates from the prompts/ directory and renders them."""

    def __init__(self, root: Path | None = None):
        self.root = root or PROMPT_DIR

    def path_for(self, name: str) -> Path:
        file_path = self.root / f"{name}.txt"
        if not file_path.exists():
            raise PromptNotFoundError(f"Prompt '{name}' was not found at {file_path}")
        return file_path

    @lru_cache(maxsize=32)
    def load(self, name: str) -> Template:
        return Template(self.path_for(name).read_text(encoding="utf-8"))

    def render(self, name: str, **kwargs) -> str:
        return self.load(name).safe_substitute(**kwargs)
