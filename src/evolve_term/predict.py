from __future__ import annotations
from typing import List
import json
from .models import KnowledgeCase
from .utils import parse_llm_json_array, parse_llm_json_object
from .exceptions import LLMUnavailableError

class Predictor:
    def __init__(self, llm_client, prompt_repo):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo
        self.last_ranking_response = None

    def infer_invariants(self, code: str, references: List[KnowledgeCase]) -> List[str]:
        prompt = self.prompt_repo.render(
            "invariant_inference",
            code=code,
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        response = self.llm_client.complete(prompt)
        invariants = parse_llm_json_array(response)
        print("[Debug] Module Predict Invariant End...\n")
        if not invariants:
            return []
        return [str(item) for item in invariants if str(item).strip()]

    def infer_ranking(self, code: str, invariants: List[str], references: List[KnowledgeCase]) -> tuple[str | None, str]:
        prompt = self.prompt_repo.render(
            "ranking_inference",
            code=code,
            invariants=json.dumps(invariants, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        # If the backend supports it, request a strict JSON object response.
        prompt["response_format"] = {"type": "json_object"}
        response = self.llm_client.complete(prompt)
        print("[Debug] Module Predict RankingFuntion Got LLM Response...\n")
        self.last_ranking_response = response
        data = parse_llm_json_object(response)
        if not isinstance(data, dict):
            return None, ""
        ranking = data.get("ranking_function")
        explanation = data.get("explanation", "")
        if ranking is not None and not isinstance(ranking, str):
            ranking = None
        if not isinstance(explanation, str):
            explanation = ""
        return ranking, explanation

    def predict(self, code: str, loops: List[str], references: List[KnowledgeCase], invariants: List[str] = None, ranking_function: str = None) -> tuple[dict, str]:
        prompt = self.prompt_repo.render(
            "prediction",
            code=code,
            loops=json.dumps(loops, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2),
            invariants=json.dumps(invariants, ensure_ascii=False, indent=2) if invariants else "[]",
            ranking_function=ranking_function or "None"
        )
        # If the backend supports it, request a strict JSON object response.
        prompt["response_format"] = {"type": "json_object"}
        raw = self.llm_client.complete(prompt)
        parsed = parse_llm_json_object(raw)
        if not isinstance(parsed, dict):
            raise LLMUnavailableError("LLM returned non-JSON response")
        return parsed, raw
