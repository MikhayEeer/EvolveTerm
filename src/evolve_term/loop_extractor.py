"""Loop extraction using tree-sitter as primary method with optional LLM enhancement."""

from __future__ import annotations

import re
import yaml
from typing import List

from .c_parser import CParser, LoopInfo
from .llm_client import LLMClient
from .prompts_loader import PromptRepository
from .utils import parse_llm_yaml


class LoopExtractor:
    """Extract loops using tree-sitter parsing with optional LLM semantic enhancement.

    Strategy:
    1. tree-sitter parsing -> Get all loops with precise AST info
    2. Optional LLM enhancement -> Abstract nested loops with LOOP{n} placeholders
    """

    def __init__(self, llm_client: LLMClient, prompt_repo: PromptRepository):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo
        self.c_parser = CParser()
        # Track last extraction details for reporting
        self.last_response: str | None = None
        self.last_method: str | None = None
        self.last_loops_info: List[LoopInfo] | None = None

    def extract(
        self,
        code: str,
        max_loops: int = 5,
        prompt_name: str = "loop_extraction/yaml_v2",
        use_llm_abstraction: bool = True,
    ) -> List[str]:
        """Extract loops from C code.

        Args:
            code: C source code
            max_loops: Maximum number of loops to return
            prompt_name: Prompt name for LLM abstraction (if enabled)
            use_llm_abstraction: Whether to use LLM for semantic enhancement

        Returns:
            List of loop code strings (possibly with LOOP{n} placeholders)
        """
        self.last_method = "treesitter"
        self.last_loops_info = None

        # Step 1: tree-sitter primary extraction
        loops_info = self.c_parser.find_all_loops(code)
        self.last_loops_info = loops_info

        if not loops_info:
            self.last_method = "treesitter_empty"
            self.last_response = yaml.dump({"loops": [], "method": "treesitter_empty"})
            return ["/* no loops detected */"]

        # Build base loop data
        loops_data = [
            {
                "id": loop.loop_id,
                "type": loop.loop_type,
                "code": loop.code,
                "depth": loop.nesting_depth,
                "parent_id": loop.parent_loop_id,
                "has_braces": loop.has_braces,
            }
            for loop in loops_info
        ]

        # Step 2: Optional LLM semantic enhancement (abstraction)
        is_abstract_mode = "yaml_v2" in prompt_name

        if use_llm_abstraction and is_abstract_mode:
            loops_data = self._abstract_with_llm(code, loops_data, prompt_name)
            self.last_method = "treesitter_llm"

        # Store response for reporting
        self.last_response = yaml.dump({"loops": loops_data, "method": self.last_method})

        # Return loop codes (up to max_loops)
        result_loops = [l["abstract_code"] if "abstract_code" in l else l["code"] for l in loops_data]

        # Apply placeholder comments for abstract mode
        if is_abstract_mode:
            result_loops = [
                re.sub(r"(LOOP\d+)", r"/* \1: Placeholder for nested loop */", loop)
                for loop in result_loops
            ]

        return result_loops[:max_loops]

    def _abstract_with_llm(
        self, code: str, loops_data: List[dict], prompt_name: str
    ) -> List[dict]:
        """Use LLM to abstract nested loops with LOOP{n} placeholders.

        Args:
            code: Full source code
            loops_data: Loop data from tree-sitter extraction
            prompt_name: Prompt template name

        Returns:
            Enhanced loop data with abstract_code field
        """
        # Render prompt with tree-sitter extracted loops
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            loops=yaml.dump(loops_data, default_flow_style=False),
        )

        response = self.llm_client.complete(prompt)

        # Parse LLM response
        llm_data = parse_llm_yaml(response)
        if not isinstance(llm_data, dict) or "loops" not in llm_data:
            # LLM failed to parse, return original data
            print("[Warning] LLM abstraction parsing failed, using original loops")
            return loops_data

        # Merge LLM abstract codes with tree-sitter data
        try:
            llm_loops = llm_data["loops"]
            for llm_loop in llm_loops:
                loop_id = llm_loop.get("id")
                # Find matching tree-sitter loop
                for ts_loop in loops_data:
                    if ts_loop["id"] == loop_id:
                        if "abstract_code" in llm_loop:
                            ts_loop["abstract_code"] = llm_loop["abstract_code"]
                        break
            return loops_data
        except Exception as e:
            print(f"[Warning] LLM abstraction merge failed: {e}")
            return loops_data

    def get_loop_info(self) -> List[LoopInfo] | None:
        """Get detailed LoopInfo objects from last extraction.

        Useful for downstream components that need byte offsets.
        """
        return self.last_loops_info
