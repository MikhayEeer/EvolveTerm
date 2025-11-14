"""HNSW index management for retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import hnswlib
import numpy as np

from .exceptions import IndexNotReadyError
from .models import KnowledgeCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_PATH = DATA_DIR / "hnsw_index.bin"
INDEX_META_PATH = DATA_DIR / "hnsw_meta.json"


class HNSWIndexManager:
    """Wraps hnswlib index creation, persistence, and queries."""

    def __init__(self, dimension: int, space: str = "cosine"):
        self.dimension = dimension
        self.space = space
        self.index_path = INDEX_PATH
        self.meta_path = INDEX_META_PATH
        self.index: hnswlib.Index | None = None
        self.case_ids: List[str] = []
        self._load_if_available()

    # Persistence ----------------------------------------------------------
    def _load_if_available(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.case_ids = meta.get("case_ids", [])
        if not self.case_ids:
            return
        index = hnswlib.Index(space=self.space, dim=self.dimension)
        index.load_index(str(self.index_path))
        self.index = index

    def save(self) -> None:
        if not self.index:
            return
        self.index.save_index(str(self.index_path))
        payload = {"case_ids": self.case_ids, "dimension": self.dimension, "space": self.space}
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build ----------------------------------------------------------------
    def rebuild(self, cases: Sequence[KnowledgeCase]) -> None:
        if not cases:
            raise IndexNotReadyError("Cannot build index without cases")
        embeddings = np.array([case.embedding for case in cases], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise IndexNotReadyError(
                f"All embeddings must be of shape (*, {self.dimension}), got {embeddings.shape}"
            )
        index = hnswlib.Index(space=self.space, dim=self.dimension)
        index.init_index(max_elements=len(cases), ef_construction=200, M=64)
        index.add_items(embeddings, ids=np.arange(len(cases)))
        index.set_ef(64)
        self.index = index
        self.case_ids = [case.case_id for case in cases]
        self.save()

    # Query ----------------------------------------------------------------
    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise IndexNotReadyError("HNSW index is not ready. Build it first.")
        labels, distances = self.index.knn_query(vector, k=top_k)
        results = []
        for label, distance in zip(labels[0], distances[0]):
            case_id = self.case_ids[label]
            score = 1 - float(distance)  # cosine distance -> similarity proxy
            results.append((case_id, score))
        return results
