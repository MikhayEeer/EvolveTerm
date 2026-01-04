import json
import yaml

def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    # Take first fenced block; tolerate ```json / ```python / ``` etc.
    parts = cleaned.split("\n", 1)
    if len(parts) == 1:
        return cleaned
    body = parts[1]
    if body.endswith("```"):
        body = body.rsplit("\n", 1)[0]
    else:
        # If there is a closing fence later, cut at the first one.
        fence_index = body.find("\n```")
        if fence_index != -1:
            body = body[:fence_index]
    return body.strip()


def extract_bracketed_payload(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    end = text.rfind(closer)
    if end == -1 or end <= start:
        return None
    return text[start : end + 1]


def json_loads_with_repairs(text: str) -> object:
    # Most common failure we saw: invalid JSON escape like \' inside strings.
    try:
        return json.loads(text)
    except Exception:
        repaired = text.replace("\\'", "'")
        return json.loads(repaired)


def parse_llm_json_object(response_text: str) -> dict | None:
    cleaned = strip_markdown_fences(response_text)
    candidates = [
        cleaned,
        extract_bracketed_payload(cleaned, "{", "}"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json_loads_with_repairs(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_llm_json_array(response_text: str) -> list | None:
    cleaned = strip_markdown_fences(response_text)
    candidates = [
        cleaned,
        extract_bracketed_payload(cleaned, "[", "]"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json_loads_with_repairs(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def parse_llm_yaml(response_text: str) -> dict | None:
    cleaned = strip_markdown_fences(response_text)
    try:
        return yaml.safe_load(cleaned)
    except Exception:
        return None

