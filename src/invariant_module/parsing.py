from __future__ import annotations

import re

import yaml


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    start_fence_idx = text.find("```")
    if start_fence_idx == -1:
        return text

    newline_idx = text.find("\n", start_fence_idx)
    if newline_idx == -1:
        return text[start_fence_idx + 3 :].strip()

    content_start = newline_idx + 1
    end_fence_idx = text.find("\n```", content_start)
    if end_fence_idx == -1:
        if text.endswith("```") and len(text) > content_start + 3:
            return text[content_start:-3].strip()
        return text[content_start:].strip()

    return text[content_start:end_fence_idx].strip()


def parse_acsl_invariants(response_text: str) -> list[str]:
    cleaned = strip_markdown_fences(response_text)
    invariants: list[str] = []
    pattern = re.compile(r"\bloop invariant\b\s*(.*?);", re.IGNORECASE)
    for match in pattern.finditer(cleaned):
        expr = match.group(1).strip()
        expr = expr.replace("*/", "").strip()
        if expr.startswith(":"):
            expr = expr[1:].strip()
        if expr:
            invariants.append(expr)
    return invariants


def parse_llm_yaml(response_text: str) -> object | None:
    cleaned = strip_markdown_fences(response_text)
    try:
        return yaml.safe_load(cleaned)
    except Exception:
        return None
