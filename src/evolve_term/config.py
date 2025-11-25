"""Configuration helpers for loading JSON config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_json_config(file_name: str) -> Dict[str, Any]:
    """Load a JSON config file relative to the repository root."""

    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = CONFIG_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Config file '{file_name}' does not exist at {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def auto_load_json_config(file_name: str,
                          tag: str = "default") -> Dict[str, Any]:
    """
    Using tag strategy to load multiple json config from a single file.
    Return a {} item from [{},{}] in json config
    """
    config_data = load_json_config(file_name)

    if isinstance(config_data, list):
        if not config_data:
            raise ValueError(f"Config file '{file_name}' is an empty list.")
        
        # Try to find the config with the specified tag
        for config in config_data:
            if tag in config.get("tags", []):
                return config
        
        # Fallback to the first item if tag not found
        return config_data[0]

    return config_data