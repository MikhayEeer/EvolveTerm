from __future__ import annotations
from typing import List
import json
from .models import KnowledgeCase
from .utils import parse_llm_json_array, parse_llm_json_object, parse_llm_yaml
from .exceptions import LLMUnavailableError

class Predictor:
    def __init__(self, llm_client, prompt_repo):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo
        self.last_ranking_response = None

    @staticmethod
    def is_empty_ranking_result(ranking: str | None, metadata: dict, mode: str) -> bool:
        if mode == "template":
            template_type = metadata.get("type")
            template_depth = metadata.get("depth")
            if template_type in (None, "") or template_depth in (None, ""):
                return True
            return False
        if ranking is None:
            return True
        if isinstance(ranking, str) and not ranking.strip():
            return True
        return False

    def infer_invariants(self, code: str, references: List[KnowledgeCase], prompt_version: str = "acsl_cot") -> List[str]:
        prompt_name = f"invariants/{prompt_version}"
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        if prompt_version.endswith("_cot") or prompt_version.endswith("_cot_fewshot"):
            prompt["max_tokens"] = 8192
        response = self.llm_client.complete(prompt)
        
        # Use YAML parsing
        data = parse_llm_yaml(response)
        invariants = []
        if isinstance(data, dict) and "invariants" in data:
            invariants = data["invariants"]
            
        print("[Debug] Module Predict Invariant End...\n")
        if not invariants:
            print(f"[Debug] Invariant Parsing Failed or Empty. Raw Response:\n{response}\n")
            return []
        return [str(item) for item in invariants if str(item).strip()]

    def infer_ranking(self, code: str, invariants: List[str], references: List[KnowledgeCase], 
                      mode: str = "direct", known_terminating: bool = False, retry_empty: int = 0,
                      log_prefix: str | None = None) -> tuple[str | None, str, dict]:
        
        prompt_name = "ranking_function/rf_direct"
        if mode == "template":
            if known_terminating:
                prompt_name = "ranking_function/rf_template_known"
            else:
                prompt_name = "ranking_function/rf_template"
        elif mode == "template_fewshot":
            if known_terminating:
                prompt_name = "ranking_function/rf_template_known_fewshot"
            else:
                prompt_name = "ranking_function/rf_template_fewshot"

        max_attempts = max(1, retry_empty + 1)
        last_ranking: str | None = None
        last_explanation = ""
        last_data: dict = {}

        for attempt in range(1, max_attempts + 1):
            prompt = self.prompt_repo.render(
                prompt_name,
                code=code,
                invariants=json.dumps(invariants, ensure_ascii=False, indent=2),
                references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
            )
            # If the backend supports it, request a strict JSON object response.
            # prompt["response_format"] = {"type": "json_object"}
            response = self.llm_client.complete(prompt)
            print("[Debug] Module Predict RankingFuntion Got LLM Response...\n")
            self.last_ranking_response = response
            
            # Parse YAML
            data = parse_llm_yaml(response)
            if not isinstance(data, dict):
                print(f"[Debug] Ranking Parsing Failed (Not a dict). Raw Response:\n{response}\n")
                data = {}
            
            # Extract info from either 'ranking' or 'configuration' key
            info = data.get("ranking") or data.get("configuration") or {}
            if not isinstance(info, dict):
                info = {}
            
            # Flatten info into data for backward compatibility (so pipeline.py can access data['type'])
            data.update(info)
            
            # 'function' is used in new YAML, 'ranking_function' was legacy JSON
            ranking = info.get("function") or info.get("ranking_function")
            explanation = info.get("explanation", "")
            
            if ranking is None and mode == "direct":
                print(f"[Debug] Ranking Function is None in YAML. Raw Response:\n{response}\n")

            if ranking is not None and not isinstance(ranking, str):
                ranking = None
            if not isinstance(explanation, str):
                explanation = ""

            last_ranking = ranking
            last_explanation = explanation
            last_data = data
            is_empty = self.is_empty_ranking_result(ranking, data, mode)

            if retry_empty > 0:
                prefix = f"[{log_prefix}] " if log_prefix else ""
                status = "success" if not is_empty else "empty"
                print(f"[Info] {prefix}Ranking attempt {attempt}/{max_attempts}: {status}")

            if not is_empty:
                return ranking, explanation, data

        return last_ranking, last_explanation, last_data

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
        # prompt["response_format"] = {"type": "json_object"}
        raw = self.llm_client.complete(prompt)
        
        # Parse YAML
        data = parse_llm_yaml(raw)
        if not isinstance(data, dict):
            raise LLMUnavailableError("LLM returned non-YAML/JSON response")
            
        # Flatten 'prediction' key if present for backward compatibility
        if "prediction" in data and isinstance(data["prediction"], dict):
            data.update(data["prediction"])
            
        return data, raw
