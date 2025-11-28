"""JSON-backed knowledge base with incremental update tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .exceptions import KnowledgeBaseError
from .models import KnowledgeCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"


class KnowledgeBase:
    """Persistent JSON knowledge base with helper queries."""

    def __init__(self, path: Path | str | None = None, rebuild_threshold: int = 10, storage_path: str | None = None):
        # Support both 'path' (legacy) and 'storage_path' (new) arguments
        # If storage_path is provided as string, convert to Path
        target_path = storage_path or path
        if isinstance(target_path, str):
            target_path = Path(target_path)
            
        self.path = target_path or KB_PATH
        self.rebuild_threshold = rebuild_threshold
        self._cases: Dict[str, KnowledgeCase] = {}
        self._pending_since_rebuild = 0
        self._load()

    # Persistence helpers -------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self.path.write_text(json.dumps({"cases": []}, indent=2), encoding="utf-8")
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for entry in raw.get("cases", []):
            case = KnowledgeCase(
                case_id=entry["case_id"],
                code=entry["code"],
                label=entry["label"],
                explanation=entry.get("explanation", ""),
                loops=entry.get("loops", []),
                embedding=entry.get("embedding", []),
                metadata=entry.get("metadata", {}),
            )
            self._cases[case.case_id] = case
        self._pending_since_rebuild = raw.get("pending_since_rebuild", 0)

    def save(self) -> None:
        payload = {
            "cases": [case.__dict__ for case in self._cases.values()],
            "pending_since_rebuild": self._pending_since_rebuild,
        }
        try:
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise KnowledgeBaseError(f"Failed to persist knowledge base: {exc}") from exc

    # Queries --------------------------------------------------------------
    @property
    def cases(self) -> Sequence[KnowledgeCase]:
        return list(self._cases.values())

    def get(self, case_id: str) -> KnowledgeCase | None:
        return self._cases.get(case_id)

    def bulk_get(self, case_ids: Iterable[str]) -> List[KnowledgeCase]:
        return [case for case_id in case_ids if (case := self._cases.get(case_id))]

    # Updates --------------------------------------------------------------
    def add_case(self, case: KnowledgeCase) -> None:
        self._cases[case.case_id] = case
        self._pending_since_rebuild += 1
        self.save()

    def needs_rebuild(self) -> bool:
        return self._pending_since_rebuild >= self.rebuild_threshold

    def mark_rebuilt(self) -> None:
        self._pending_since_rebuild = 0
        self.save()
