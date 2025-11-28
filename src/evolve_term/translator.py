"""Code translation module using LLM."""

from __future__ import annotations

from pathlib import Path

from .llm_client import LLMClient, build_llm_client
from .prompts_loader import PromptRepository

class CodeTranslator:
    """Translates code from various languages to C++ using LLM."""

    def __init__(self, llm_client: LLMClient | None = None, config_name: str = "llm_config.json"):
        # Prefer using a client with 'long-context' tag for translation tasks
        self.llm_client = llm_client or build_llm_client(config_name, config_tag="long-context")
        self.prompt_repo = PromptRepository()
        # Translation prompt will be loaded from prompts/translation.txt when needed
        # (PromptRepository handles file loading via load() and render() methods)

    def translate(self, source_code: str) -> str:
        """Translate source code to C++."""
        prompt = self.prompt_repo.render("translation", source_code=source_code)
        
        translated_code = self.llm_client.complete(prompt)
        
        # Clean up potential markdown code blocks if the LLM ignores instructions
        translated_code = self._clean_markdown(translated_code)
        
        return translated_code

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown code fences if present."""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
