"""Loop extraction orchestrated via prompts + heuristics."""

from __future__ import annotations

import json
import re
from typing import List

from .llm_client import LLMClient
from .prompts_loader import PromptRepository


class LoopExtractor:
    """Delegates loop extraction to the LLM with a heuristic fallback."""

    def __init__(self, llm_client: LLMClient, prompt_repo: PromptRepository):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo

    def extract(self, code: str, max_loops: int = 5) -> List[str]:
        prompt = self.prompt_repo.render("loop_extraction", code=code)
        response = self.llm_client.complete(prompt)
        loops = self._parse_response(response)
        if not loops:
            loops = self._heuristic_loops(code)
        return loops[:max_loops]

    def _parse_response(self, response: str) -> List[str]:
        try:
            payload = json.loads(response)
            if isinstance(payload, dict):
                candidate = payload.get("loops")
            else:
                candidate = payload
            if isinstance(candidate, list):
                return [str(item).strip() for item in candidate if str(item).strip()]
        except json.JSONDecodeError:
            pass
        matches = re.findall(r"for\s*\(.*?\)|while\s*\(.*?\)", response, flags=re.DOTALL)
        return [match.strip() for match in matches]

    def _heuristic_loops(self, code: str) -> List[str]:
        loops = re.findall(r"for\s*\(.*?\{.*?\}|while\s*\(.*?\{.*?\}", code, flags=re.DOTALL)
        if loops:
            return [re.sub(r"\s+", " ", loop.strip()) for loop in loops]
        return ["/* no loops detected */"]
