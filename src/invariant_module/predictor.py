from __future__ import annotations

import json
from typing import Any, List

from .parsing import parse_acsl_invariants, parse_llm_yaml


class InvariantPredictor:
    """LLM-based loop invariant inference."""

    def __init__(self, llm_client: Any, prompt_repo: Any):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo

    def infer_invariants(
        self,
        code: str,
        references: List[Any],
        prompt_version: str = "acsl_cot",
    ) -> List[str]:
        prompt_name = f"invariants/{prompt_version}"
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            references=json.dumps(
                [getattr(ref, "__dict__", ref) for ref in references],
                ensure_ascii=False,
                indent=2,
            ),
        )
        if prompt_version.endswith("_cot") or prompt_version.endswith("_cot_fewshot"):
            prompt["max_tokens"] = 8192

        response = self.llm_client.complete(prompt)
        invariants = self._extract_invariants(response)

        print("[Debug] Module Predict Invariant End...\n")
        if not invariants:
            print(f"[Debug] Invariant Parsing Failed or Empty. Raw Response:\n{response}\n")
            return []
        return [str(item) for item in invariants if str(item).strip()]

    def _extract_invariants(self, response: str) -> List[Any]:
        data = parse_llm_yaml(response)
        invariants: List[Any] = []

        if isinstance(data, dict):
            invariants = (
                data.get("invariants")
                or data.get("loop_invariants")
                or data.get("loop_invariant")
                or []
            )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    candidate = item.get("invariant") or item.get("expr") or item.get("formula")
                    if candidate is not None:
                        invariants.append(candidate)
                else:
                    invariants.append(item)

        if not invariants:
            invariants = parse_acsl_invariants(response)

        return invariants
