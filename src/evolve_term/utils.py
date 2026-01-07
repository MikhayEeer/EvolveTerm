import json
import yaml

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    # Find the first opening fence
    start_fence_idx = text.find("```")
    if start_fence_idx == -1:
        return text
    
    # Find the end of the opening fence line (to skip language identifier)
    newline_idx = text.find("\n", start_fence_idx)
    if newline_idx == -1:
        # Only opening fence? Return everything after it
        return text[start_fence_idx+3:].strip()
        
    content_start = newline_idx + 1
    
    # Find the closing fence
    end_fence_idx = text.find("\n```", content_start)
    if end_fence_idx == -1:
        # Try finding fence at the very end without newline if it's the last thing
        if text.endswith("```") and len(text) > content_start + 3:
             return text[content_start:-3].strip()
        return text[content_start:].strip()
        
    return text[content_start:end_fence_idx].strip()


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


class LiteralDumper(yaml.SafeDumper):
    """Custom YAML Dumper that uses block style for multiline strings."""
    def represent_scalar(self, tag, value, style=None):
        if "\n" in value and tag == 'tag:yaml.org,2002:str':
            style = '|'
        return super().represent_scalar(tag, value, style)

