"""LLM client abstractions."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from openai import OpenAI

from .config import load_json_config
from .exceptions import LLMUnavailableError


class LLMClient(ABC):
    """Base interface for text generation."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class APILLMClient(LLMClient):
    """LLM client implemented via the OpenAI SDK chat completions API."""

    def __init__(self, config_name: str = "llm_config.json"):
        config = load_json_config(config_name)
        self.base_url = config.get("base_url") or config.get("baseurl")
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.payload_template = config.get("payload_template", {})
        if not self.base_url or not self.api_key:
            raise LLMUnavailableError("LLM base_url or API key missing")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.payload_template,
            )
        except Exception as exc:  # pragma: no cover - network path
            raise LLMUnavailableError(f"LLM provider error: {exc}") from exc

        if not response.choices:
            raise LLMUnavailableError("LLM provider returned no choices")
        message = response.choices[0].message
        if message is None:
            raise LLMUnavailableError("LLM provider returned empty message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""


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


def _demo_completion(prompt: str = "Say hello in one short sentence.") -> None:
    """Manual smoke test for the configured LLM service."""

    client = build_llm_client()
    response = client.complete(prompt)
    print("LLM response:\n", response)


if __name__ == "__main__":  # pragma: no cover - manual verification helper
    print("[LLM Demo] Using config/llm_config.json")
    try:
        _demo_completion()
    except LLMUnavailableError as exc:
        print(f"LLM test failed: {exc}")
