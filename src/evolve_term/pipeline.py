"""High-level termination analysis pipeline."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List

import numpy as np

from .embeddings import build_embedding_client
from .exceptions import EmbeddingUnavailableError, IndexNotReadyError, LLMUnavailableError
from .knowledge_base import KnowledgeBase
from .llm_client import build_llm_client
from .loop_extractor import LoopExtractor
from .models import KnowledgeCase, Label, PendingReviewCase, PredictionResult
from .prompts_loader import PromptRepository
from .rag_index import HNSWIndexManager

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class TerminationPipeline:
    """Main façade that ties together embedding, retrieval, and LLM reasoning."""

    def __init__(self, rebuild_threshold: int = 10, embed_config: str = "embed_config.json", llm_config: str = "llm_config.json"):
        self.prompt_repo = PromptRepository()
        self.llm_client = build_llm_client(llm_config)
        self.loop_extractor = LoopExtractor(self.llm_client, self.prompt_repo)
        self.embedding_client = build_embedding_client(embed_config)
        self.knowledge_base = KnowledgeBase(rebuild_threshold=rebuild_threshold)
        self.index_manager = HNSWIndexManager(dimension=self.embedding_client.dimension)

    # Core flow ------------------------------------------------------------
    def analyze(self, code: str, top_k: int = 5, auto_build_index: bool = True) -> PredictionResult:
        loops = self.loop_extractor.extract(code)
        vector = self.embedding_client.embed("\n".join(loops))
        if auto_build_index and self.index_manager.index is None:
            self._maybe_build_index()
        neighbors = self.index_manager.search(vector, top_k=top_k)
        similarity_map = {case_id: score for case_id, score in neighbors}
        references = self.knowledge_base.bulk_get(case_id for case_id, _ in neighbors)
        for ref in references:
            ref.metadata["similarity"] = round(similarity_map.get(ref.case_id, 0.0), 3)
        prediction = self._predict_with_llm(code, loops, references)
        report_path = self._persist_report(prediction)
        return PredictionResult(
            label=prediction["label"],
            confidence=float(prediction.get("confidence", 0.0)),
            reasoning=prediction.get("reasoning", ""),
            loops=loops,
            references=references,
            report_path=report_path,
        )

    def _predict_with_llm(self, code: str, loops: List[str], references: List[KnowledgeCase]) -> dict:
        prompt = self.prompt_repo.render(
            "prediction",
            code=code,
            loops=json.dumps(loops, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2),
        )
        raw = self.llm_client.complete(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError("LLM returned non-JSON response") from exc

    def _persist_report(self, prediction: dict) -> Path:
        report_id = uuid.uuid4().hex
        path = REPORTS_DIR / f"report_{report_id}.json"
        path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # RAG maintenance ------------------------------------------------------
    def ingest_reviewed_case(self, reviewed: PendingReviewCase) -> KnowledgeCase:
        embedding = self.embedding_client.embed("\n".join(reviewed.loops))
        case = KnowledgeCase(
            case_id=uuid.uuid4().hex,
            code=reviewed.code,
            label=reviewed.label,
            explanation=reviewed.explanation,
            loops=list(reviewed.loops),
            embedding=embedding.astype(float).tolist(),
            metadata={"reviewer": reviewed.reviewer},
        )
        self.knowledge_base.add_case(case)
        if self.knowledge_base.needs_rebuild():
            self.index_manager.rebuild(self.knowledge_base.cases)
            self.knowledge_base.mark_rebuilt()
        else:
            self._incremental_add(case, embedding)
        return case

    def _maybe_build_index(self) -> None:
        if not self.knowledge_base.cases:
            raise IndexNotReadyError("Knowledge base is empty. Add seed cases first.")
        self.index_manager.rebuild(self.knowledge_base.cases)
        self.knowledge_base.mark_rebuilt()

    def _incremental_add(self, case: KnowledgeCase, embedding: np.ndarray) -> None:
        if self.index_manager.index is None:
            self._maybe_build_index()
            return
        index = self.index_manager.index
        new_id = len(self.index_manager.case_ids)
        index.resize_index(new_size=new_id + 1)
        index.add_items(embedding.reshape(1, -1), ids=np.array([new_id]))
        self.index_manager.case_ids.append(case.case_id)
        self.index_manager.save()
