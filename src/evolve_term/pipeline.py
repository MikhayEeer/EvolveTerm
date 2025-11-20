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
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


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
        """Run full analysis and produce a report+log capturing each stage.

        The report JSON will include: input code, loop extraction (loops + method + llm_response),
        embedding info, RAG neighbors, references (with similarity), raw LLM prediction and
        the parsed prediction. A plain-text log will also be written for human inspection.
        """
        run_id = uuid.uuid4().hex
        # Stage 1: loop extraction
        loops = self.loop_extractor.extract(code)
        loop_details = {
            "loops": loops,
            "method": getattr(self.loop_extractor, "last_method", None),
            "llm_response": getattr(self.loop_extractor, "last_response", None),
        }

        # Stage 2: embedding
        embedding_vector = self.embedding_client.embed("\n".join(loops))
        try:
            embedding_list = embedding_vector.astype(float).tolist()
        except Exception:
            # embedding client may already return list
            embedding_list = list(embedding_vector)
        embedding_info = {
            "provider": getattr(self.embedding_client, "provider", None),
            "model": getattr(self.embedding_client, "model_name", None),
            "dimension": getattr(self.embedding_client, "dimension", None),
            "vector": embedding_list,
        }

        # Ensure index ready
        if auto_build_index and self.index_manager.index is None:
            self._maybe_build_index()

        # Stage 3: retrieval
        neighbors = self.index_manager.search(embedding_vector, top_k=top_k)
        similarity_map = {case_id: score for case_id, score in neighbors}
        references = self.knowledge_base.bulk_get(case_id for case_id, _ in neighbors)
        for ref in references:
            ref.metadata["similarity"] = round(similarity_map.get(ref.case_id, 0.0), 3)

        # Stage 4: LLM prediction
        prediction, raw_prediction = self._predict_with_llm(code, loops, references)

        # Build comprehensive report payload
        report_payload = {
            "run_id": run_id,
            "code": code,
            "loop_extraction": loop_details,
            "embedding": embedding_info,
            "neighbors": [
                {"case_id": cid, "score": score} for cid, score in neighbors
            ],
            "references": [ref.__dict__ for ref in references],
            "prediction_raw": raw_prediction,
            "prediction": prediction,
        }

        report_path = self._persist_report(report_payload)
        # Also write a human-readable log
        self._persist_log(report_payload, report_path.stem)

        return PredictionResult(
            label=prediction["label"],
            reasoning=prediction.get("reasoning", ""),
            loops=loops,
            references=references,
            report_path=report_path,
        )

    def _predict_with_llm(self, code: str, loops: List[str], references: List[KnowledgeCase]) -> tuple[dict, str]:
        prompt = self.prompt_repo.render(
            "prediction",
            code=code,
            loops=json.dumps(loops, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2),
        )
        raw = self.llm_client.complete(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError("LLM returned non-JSON response") from exc
        return parsed, raw

    def _persist_report(self, report: dict) -> Path:
        report_id = report.get("run_id", uuid.uuid4().hex)
        path = REPORTS_DIR / f"report_{report_id}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _persist_log(self, report: dict, report_stem: str) -> Path:
        """Write a human-readable log summarizing the report (one log == one run)."""
        log_path = LOGS_DIR / f"{report_stem}.log"
        lines = []
        lines.append(f"Run ID: {report.get('run_id')}")
        lines.append("--- Input Code ---")
        lines.append(report.get("code", ""))
        lines.append("--- Loop Extraction ---")
        le = report.get("loop_extraction", {})
        lines.append(f"method: {le.get('method')}")
        lines.append("loops:")
        for lp in le.get("loops", []):
            lines.append(lp)
        if le.get("llm_response"):
            lines.append("--- Raw LLM loop response ---")
            lines.append(str(le.get("llm_response")))
        lines.append("--- Embedding info ---")
        emb = report.get("embedding", {})
        lines.append(f"provider: {emb.get('provider')}, model: {emb.get('model')}, dimension: {emb.get('dimension')}")
        lines.append(f"vector_length: {len(emb.get('vector', []))}")
        lines.append("--- Neighbors ---")
        for n in report.get("neighbors", []):
            lines.append(f"{n.get('case_id')}: {n.get('score')}")
        lines.append("--- Prediction (parsed) ---")
        lines.append(json.dumps(report.get("prediction", {}), ensure_ascii=False, indent=2))
        lines.append("--- Prediction (raw) ---")
        lines.append(str(report.get("prediction_raw")))
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return log_path

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
            metadata={
                "reviewer": reviewed.reviewer,
                "embedding_provider": self.embedding_client.provider,
                "embedding_model": self.embedding_client.model_name,
                "embedding_dimension": self.embedding_client.dimension,
            },
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
