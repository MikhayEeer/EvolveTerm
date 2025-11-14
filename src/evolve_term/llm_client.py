"""LLM client abstractions."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import requests

from .config import load_json_config
from .exceptions import LLMUnavailableError


class LLMClient(ABC):
    """Base interface for text generation."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class APILLMClient(LLMClient):
    """Generic HTTP-based LLM client."""

    def __init__(self, config_name: str = "llm_config.json"):
        config = load_json_config(config_name)
        self.endpoint = config.get("endpoint")
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.payload_template = config.get("payload_template", {})
        if not self.endpoint or not self.api_key:
            raise LLMUnavailableError("LLM endpoint or API key missing")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
        }
        payload.update(self.payload_template)

        response = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(payload),
            timeout=60,
        )
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"LLM provider error {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        # Support both OpenAI-style {choices: [{text: ...}]} and plain {output: str}
        if "choices" in data:
            return data["choices"][0].get("text") or ""
        return data.get("output") or data.get("text") or ""


class MockLLMClient(LLMClient):
    """Simple deterministic templates for offline demo."""

    def complete(self, prompt: str) -> str:
        # Extract fenced ```code``` blocks to simulate loop summaries
        code_blocks = re.findall(r"```(?:c|cpp)?\n(.*?)```", prompt, re.DOTALL)
        if code_blocks:
            body = code_blocks[-1]
        else:
            body = prompt
        loops = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith(("for", "while")):
                loops.append(line)
        if not loops:
            loops = ["/* no-loop-detected */"]
        normalized_prompt = prompt.lower()
        if "single key \"loops\"" in normalized_prompt:
            return json.dumps({"loops": loops}, ensure_ascii=False)
        if 'keys: "label"' in normalized_prompt or 'return a json payload' in normalized_prompt:
            label = "terminating" if any(token in body for token in ("--", "-=")) else "non-terminating" if "while(1" in body or "for(;;" in body else "unknown"
            result = {
                "label": label,
                "confidence": 0.73 if label != "unknown" else 0.4,
                "reasoning": "Mock reasoning: heuristic result for offline demo.",
                "report": "Predicted using mock LLM client with references.",
            }
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({"text": "Unsupported mock prompt"})


def build_llm_client(config_name: str = "llm_config.json") -> LLMClient:
    config = load_json_config(config_name)
    provider = config.get("provider", "mock").lower()
    if provider == "mock":
        return MockLLMClient()
    return APILLMClient(config_name=config_name)
