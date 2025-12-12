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

    @lru_cache(maxsize=32)
    def load(self, name: str) -> tuple[Template, Template]:
        """Load system and user templates."""
        system_path = self.root / f"{name}.system.txt"
        user_path = self.root / f"{name}.user.txt"
        
        if not system_path.exists() or not user_path.exists():
            # Fallback for legacy single file (though we deleted them, good for robustness)
            legacy_path = self.root / f"{name}.txt"
            if legacy_path.exists():
                return Template(""), Template(legacy_path.read_text(encoding="utf-8"))
            raise PromptNotFoundError(f"Prompt '{name}' files not found (checked .system.txt/.user.txt and .txt)")
            
        return (
            Template(system_path.read_text(encoding="utf-8")),
            Template(user_path.read_text(encoding="utf-8"))
        )

    def render(self, name: str, **kwargs) -> dict[str, str]:
        """Render system and user prompts."""
        system_tmpl, user_tmpl = self.load(name)
        return {
            "system": system_tmpl.safe_substitute(**kwargs),
            "user": user_tmpl.safe_substitute(**kwargs)
        }
