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
                          tag: str) -> Dict[str, Any]:
    """
    Using tag strategy to load multiple json config from a single file.
    Return a {} item from [{},{}] in json config
    """
    print("- [ ] Need to complete auto_load_json_config function in config.py")
    return {}