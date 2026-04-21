from __future__ import annotations

from typing import Any, Dict, List


def parse_invariants_content(content: Any) -> Dict[int, List[str]]:
    """Normalize supported invariant YAML/JSON shapes into loop_id -> invariants."""
    invariants: Dict[int, List[str]] = {}

    if isinstance(content, list):
        if all(isinstance(item, str) for item in content):
            invariants[1] = [str(item) for item in content]
        else:
            for idx, entry in enumerate(content, start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        return invariants

    if isinstance(content, dict):
        if "invariants" in content and isinstance(content["invariants"], list):
            invariants[1] = [str(item) for item in content["invariants"]]
        elif "invariants_result" in content:
            for idx, entry in enumerate(content.get("invariants_result") or [], start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        elif "ranking_results" in content:
            for idx, entry in enumerate(content.get("ranking_results") or [], start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        return invariants

    return invariants
