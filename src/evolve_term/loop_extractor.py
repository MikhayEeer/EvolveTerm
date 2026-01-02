"""Loop extraction orchestrated via prompts + heuristics."""

from __future__ import annotations

import json
import re
import yaml
from typing import List

from .llm_client import LLMClient
from .prompts_loader import PromptRepository
from .utils import strip_markdown_fences


class LoopExtractor:
    """Delegates loop extraction to the LLM with a heuristic fallback."""

    def __init__(self, llm_client: LLMClient, prompt_repo: PromptRepository):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo
        # last LLM response and method used for extraction ('llm' or 'heuristic')
        self.last_response: str | None = None
        self.last_method: str | None = None

    def extract(self, code: str, max_loops: int = 5, prompt_name: str = "loop_extraction/yaml_v1") -> List[str]:
        # Use the new YAML-based prompt
        prompt = self.prompt_repo.render(prompt_name, code=code)
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
            print("[Debug][Extractor] Using LLM extraction method, and Successfully get verified loops.")
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
        """Parse the YAML response."""
        cleaned = strip_markdown_fences(response)
        try:
            data = yaml.safe_load(cleaned)
            if not data or "loops" not in data:
                return []
            
            loops = []
            for item in data["loops"]:
                if "code" in item:
                    loops.append(item["code"].strip())
            return loops
        except yaml.YAMLError as e:
            print(f"[Warning] YAML parsing failed: {e}")
            return []
        except Exception as e:
            print(f"[Warning] Loop extraction parsing failed: {e}")
            return []

    def _heuristic_loops(self, code: str) -> List[str]:
        loops = re.findall(r"for\s*\(.*?\{.*?\}|while\s*\(.*?\{.*?\}", code, flags=re.DOTALL)
        if loops:
            return [re.sub(r"\s+", " ", loop.strip()) for loop in loops]
        return ["/* no loops detected */"]
