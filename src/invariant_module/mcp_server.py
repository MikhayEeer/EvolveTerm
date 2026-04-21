from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

try:
    from evolve_term.yaml_schema import (
        EXTRACT_SCHEMA,
        FEATURE_SCHEMA,
        INVARIANT_SCHEMA,
        RANKING_SCHEMA,
        FieldSpec,
        detect_yaml_type,
        validate_yaml_content,
    )
except ImportError:  # pragma: no cover - supports python -m src....
    from src.evolve_term.yaml_schema import (
        EXTRACT_SCHEMA,
        FEATURE_SCHEMA,
        INVARIANT_SCHEMA,
        RANKING_SCHEMA,
        FieldSpec,
        detect_yaml_type,
        validate_yaml_content,
    )

from .io import parse_invariants_content

PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "evolveterm-invariant-mcp"
SERVER_VERSION = "0.1.0"


JSONValue = Dict[str, Any]


def _read_yaml_argument(arguments: JSONValue) -> tuple[Any, str]:
    yaml_text = arguments.get("yaml_text")
    path_value = arguments.get("path")

    if yaml_text is None and path_value is None:
        raise ValueError("Either 'yaml_text' or 'path' is required.")

    if yaml_text is None:
        path = Path(str(path_value))
        yaml_text = path.read_text(encoding="utf-8")
        source_name = path.name
    else:
        source_name = str(path_value or "inline.yml")

    try:
        return yaml.safe_load(str(yaml_text)), source_name
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc


def _field_type_name(spec: FieldSpec) -> str:
    if spec.field_type is None:
        return "any"
    return spec.field_type.__name__


def _schema_to_payload(schema: Dict[str, FieldSpec]) -> List[JSONValue]:
    payload = []
    for name, spec in schema.items():
        item: JSONValue = {
            "name": name,
            "required": spec.required,
            "type": _field_type_name(spec),
        }
        if spec.allowed_values is not None:
            item["allowed_values"] = sorted(str(value) for value in spec.allowed_values)
        if spec.min_value is not None:
            item["min_value"] = spec.min_value
        if spec.max_value is not None:
            item["max_value"] = spec.max_value
        payload.append(item)
    return payload


def _validation_to_payload(result: Any) -> JSONValue:
    return {
        "valid": bool(result.valid),
        "yaml_type": result.yaml_type,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def tool_yaml_detect(arguments: JSONValue) -> JSONValue:
    content, source_name = _read_yaml_argument(arguments)
    yaml_type = detect_yaml_type(Path(source_name), content if isinstance(content, dict) else None)
    return {"yaml_type": yaml_type, "source": source_name}


def tool_yaml_validate(arguments: JSONValue) -> JSONValue:
    content, source_name = _read_yaml_argument(arguments)
    if not isinstance(content, dict):
        return {
            "valid": False,
            "yaml_type": None,
            "errors": ["YAML root must be a dictionary"],
            "warnings": [],
        }

    yaml_type = arguments.get("yaml_type") or detect_yaml_type(Path(source_name), content)
    if not yaml_type:
        return {
            "valid": False,
            "yaml_type": None,
            "errors": ["Could not detect YAML type. Pass yaml_type explicitly."],
            "warnings": [],
        }

    strict = bool(arguments.get("strict", True))
    result = validate_yaml_content(content, str(yaml_type), strict=strict)
    return _validation_to_payload(result)


def tool_invariant_normalize(arguments: JSONValue) -> JSONValue:
    content, source_name = _read_yaml_argument(arguments)
    loop_map = parse_invariants_content(content)
    return {
        "source": source_name,
        "loop_invariants": [
            {"loop_id": loop_id, "invariants": invariants}
            for loop_id, invariants in sorted(loop_map.items())
        ],
    }


TOOLS: Dict[str, Callable[[JSONValue], JSONValue]] = {
    "yaml.detect": tool_yaml_detect,
    "yaml.validate": tool_yaml_validate,
    "invariant.normalize": tool_invariant_normalize,
}


TOOL_DEFINITIONS: List[JSONValue] = [
    {
        "name": "yaml.detect",
        "description": "Detect an EvolveTerm YAML document type from YAML text or a file path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "path": {"type": "string"},
            },
        },
    },
    {
        "name": "yaml.validate",
        "description": "Validate an EvolveTerm YAML document against the normalized module schema.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "path": {"type": "string"},
                "yaml_type": {
                    "type": "string",
                    "enum": ["extract", "invariant", "ranking", "feature"],
                },
                "strict": {"type": "boolean"},
            },
        },
    },
    {
        "name": "invariant.normalize",
        "description": "Normalize supported invariant YAML/JSON shapes into loop_id -> invariants entries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_text": {"type": "string"},
                "path": {"type": "string"},
            },
        },
    },
]


RESOURCE_SCHEMAS = {
    "invariant://schema/extract": ("extract schema", EXTRACT_SCHEMA),
    "invariant://schema/invariant": ("invariant schema", INVARIANT_SCHEMA),
    "invariant://schema/ranking": ("ranking schema", RANKING_SCHEMA),
    "invariant://schema/feature": ("feature schema", FEATURE_SCHEMA),
}


RESOURCE_TEXT = {
    "invariant://module/flow": """name: invariant_module_flow
steps:
  - two_stage_candidate_generation
  - houdini_filtering
  - missing_constant_agent
  - boundary_openness_agent
  - control_flow_coverage_agent
  - optional_final_houdini_filtering
""",
    "invariant://template/invariant": """source_file: example.c
source_path: data/example.c
task: invariant_inference
command: evolveterm invariant --input data/example.c
pmt_ver: refinement
model: unknown
time: "2026-04-21T00:00"
has_extract: false
invariants_result:
  - loop_id: 1
    code: "while (i < n) { i++; }"
    invariants:
      - "i <= n"
""",
}


def _list_resources() -> List[JSONValue]:
    resources = []
    for uri in RESOURCE_SCHEMAS:
        resources.append(
            {
                "uri": uri,
                "name": uri.rsplit("/", 1)[-1],
                "description": f"EvolveTerm {uri.rsplit('/', 1)[-1]} YAML schema",
                "mimeType": "application/x-yaml",
            }
        )
    for uri in RESOURCE_TEXT:
        resources.append(
            {
                "uri": uri,
                "name": uri.rsplit("/", 1)[-1],
                "description": "Invariant module MCP YAML resource",
                "mimeType": "application/x-yaml",
            }
        )
    return resources


def _read_resource(uri: str) -> str:
    if uri in RESOURCE_SCHEMAS:
        _, schema = RESOURCE_SCHEMAS[uri]
        return yaml.safe_dump(_schema_to_payload(schema), sort_keys=False, allow_unicode=True)
    if uri in RESOURCE_TEXT:
        return RESOURCE_TEXT[uri]
    raise KeyError(uri)


def _jsonrpc_result(request_id: Any, result: JSONValue) -> JSONValue:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> JSONValue:
    error: JSONValue = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


class InvariantMCPServer:
    def handle(self, message: JSONValue) -> JSONValue | None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if request_id is None:
            # JSON-RPC notification. MCP clients commonly send notifications/initialized.
            return None

        try:
            if method == "initialize":
                return _jsonrpc_result(
                    request_id,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                        },
                        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    },
                )
            if method == "tools/list":
                return _jsonrpc_result(request_id, {"tools": TOOL_DEFINITIONS})
            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments") or {}
                if tool_name not in TOOLS:
                    return _jsonrpc_error(request_id, -32602, f"Unknown tool: {tool_name}")
                payload = TOOLS[tool_name](arguments)
                text = json.dumps(payload, ensure_ascii=False, indent=2)
                return _jsonrpc_result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": payload,
                        "isError": False,
                    },
                )
            if method == "resources/list":
                return _jsonrpc_result(request_id, {"resources": _list_resources()})
            if method == "resources/read":
                uri = str(params.get("uri", ""))
                text = _read_resource(uri)
                return _jsonrpc_result(
                    request_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/x-yaml",
                                "text": text,
                            }
                        ]
                    },
                )
            return _jsonrpc_error(request_id, -32601, f"Method not found: {method}")
        except KeyError as exc:
            return _jsonrpc_error(request_id, -32002, "Resource not found", {"uri": str(exc)})
        except ValueError as exc:
            return _jsonrpc_error(request_id, -32602, str(exc))
        except Exception as exc:  # pragma: no cover - defensive JSON-RPC boundary
            return _jsonrpc_error(request_id, -32603, "Internal error", {"detail": str(exc)})


def main() -> None:
    server = InvariantMCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except Exception as exc:
            response = _jsonrpc_error(None, -32700, "Parse error", {"detail": str(exc)})
        else:
            response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
