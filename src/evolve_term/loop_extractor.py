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
        # last LLM response and method used for extraction ('llm' or 'heuristic')
        self.last_response: str | None = None
        self.last_method: str | None = None

    def extract(self, code: str, max_loops: int = 5) -> List[str]:
        prompt = self.prompt_repo.render("loop_extraction", code=code)
        response = self.llm_client.complete(prompt)
        self.last_response = response
        
        # 1. Try parsing LLM response
        loops = self._parse_response(response)
        
        # 2. Verify extracted loops against original code
        verified_loops = []
        if loops:
            for loop in loops:
                if self._verify_loop_in_code(loop, code):
                    verified_loops.append(loop)
                else:
                    # Optional: Log that a hallucination was dropped
                    pass
        
        if verified_loops:
            self.last_method = "llm"
            return verified_loops[:max_loops]
            
        # 3. Fallback to heuristic if LLM failed or all extractions were hallucinations
        loops = self._heuristic_loops(code)
        self.last_method = "heuristic"
        return loops[:max_loops]

    def _verify_loop_in_code(self, loop_snippet: str, original_code: str) -> bool:
        """Check if loop_snippet exists in original_code, ignoring whitespace."""
        def normalize(s: str) -> str:
            return "".join(s.split())
        
        return normalize(loop_snippet) in normalize(original_code)

    def _parse_response(self, response: str) -> List[str]:
        """Parse the custom delimiter-separated response."""
        if "NO_LOOPS_FOUND" in response:
            return []
            
        # Split by the custom delimiter
        parts = response.split("---LOOP_SEPARATOR---")
        
        # Clean up each part
        loops = []
        for part in parts:
            cleaned = part.strip()
            # Remove potential markdown fences if LLM ignored instructions
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```c", "").replace("```", "")
            cleaned = cleaned.strip()
            if cleaned:
                loops.append(cleaned)
                
        return loops

    def _heuristic_loops(self, code: str) -> List[str]:
        loops = re.findall(r"for\s*\(.*?\{.*?\}|while\s*\(.*?\{.*?\}", code, flags=re.DOTALL)
        if loops:
            return [re.sub(r"\s+", " ", loop.strip()) for loop in loops]
        return ["/* no loops detected */"]
