"""Domain models used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Sequence

Label = Literal["terminating", "non-terminating", "unknown"]


@dataclass
class KnowledgeCase:
    """A labeled case stored inside the RAG knowledge base."""

    case_id: str
    code: str
    label: Label
    explanation: str
    loops: List[str]
    embedding: List[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Structured prediction returned to the user."""

    label: Label
    reasoning: str
    loops: List[str]
    references: List[KnowledgeCase]
    report_path: Path | None = None
    # Neuro-symbolic artifacts
    invariants: List[str] = field(default_factory=list)
    ranking_function: str | None = None
    verification_result: str | None = None


@dataclass
class PendingReviewCase:
    """Represents a case awaiting manual review before committing to RAG."""

    code: str
    label: Label
    explanation: str
    loops: Sequence[str]
    reviewer: str = "unknown"
