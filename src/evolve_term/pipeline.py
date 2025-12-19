"""High-level termination analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import uuid
import numpy as np

from .embeddings import build_embedding_client
from .exceptions import EmbeddingUnavailableError, IndexNotReadyError
from .knowledge_base import KnowledgeBase
from .llm_client import build_llm_client
from .loop_extractor import LoopExtractor
from .models import KnowledgeCase, Label, PendingReviewCase, PredictionResult
from .prompts_loader import PromptRepository
from .rag_index import HNSWIndexManager
from .translator import CodeTranslator
from .verifier import Z3Verifier
from .report_manager import ReportManager
from .predict import Predictor

class TerminationPipeline:
    """Main façade that ties together embedding, retrieval, and LLM reasoning."""

    def __init__(self, rebuild_threshold: int = 10, embed_config: str = "embed_config.json", llm_config: str = "llm_config.json", enable_translation: bool = False, knowledge_base_path: str | None = None):
        self.prompt_repo = PromptRepository()
        self.llm_client = build_llm_client(llm_config)
        self.loop_extractor = LoopExtractor(self.llm_client,# loop extractor also use same model with prediction 
                                            self.prompt_repo)
        self.embedding_client = build_embedding_client(embed_config)
        self.knowledge_base = KnowledgeBase(rebuild_threshold=rebuild_threshold, 
                                            storage_path=knowledge_base_path)
        self.index_manager = HNSWIndexManager(dimension=self.embedding_client.dimension)
        self.enable_translation = enable_translation
        self.translator = CodeTranslator(config_name=llm_config) if enable_translation else None
        self.verifier = Z3Verifier(self.llm_client, self.prompt_repo)
        self.report_manager = ReportManager()
        self.predictor = Predictor(self.llm_client, self.prompt_repo)

    # Core flow ------------------------------------------------------------
    def analyze(self, code: str, top_k: int = 5, auto_build_index: bool = True, use_rag_in_reasoning: bool = True) -> PredictionResult:
        """Run full analysis and produce a report+log capturing each stage.

        The report JSON will include: input code, loop extraction (loops + method + llm_response),
        embedding info, RAG neighbors, references (with similarity), raw LLM prediction and
        the parsed prediction. A plain-text log will also be written for human inspection.
        """
        start_time = datetime.now()
        start_llm_calls = getattr(self.llm_client, "call_count", 0)
        run_id = uuid.uuid4().hex

        # Stage 0: Translation (if enabled)
        original_code = code
        if self.enable_translation and self.translator:
            # We assume the input might not be C/C++ if translation is enabled.
            # However, the translator is generic.
            # For now, we just translate whatever is passed if enabled.
            # Optimization: We could check if it looks like C++ but that's hard.
            # The CLI handles the file extension check.
            code = self.translator.translate(code)

        # Stage 1: loop extraction
        #loops = self.loop_extractor.extract(code)
        loops = code
        #TODO: 关闭提取模块的消融实验
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

        # Filter references for prompt (only include high similarity cases)
        prompt_references = [
            ref for ref in references 
            if ref.metadata.get("similarity", 0.0) > 0.7
        ]
        
        # Apply user preference for RAG in reasoning
        reasoning_references = prompt_references if use_rag_in_reasoning else []

        # Stage 4: Neuro-symbolic Reasoning Pipeline
        
        # Optimization: Use extracted loops for reasoning to reduce token count and noise.
        # Fallback to full code if no loops extracted.
        reasoning_context = "\n".join(loops) if loops else code
        
        # 4.1 Invariant Inference
        #invariants = self.predictor.infer_invariants(reasoning_context, reasoning_references)
        #TODO: 消融实验，不增加不变式
        invariants = []

        # 4.2 Ranking Function Inference
        ranking_function, ranking_explanation = self.predictor.infer_ranking(reasoning_context, invariants, reasoning_references)
        ## issue: 1215fix: ranking function is none
        ##FixedTODO

        # 4.3 Z3 Verification
        print(f"[Debug] Ready to Z3 verify\nInvar:{invariants}\nRF:{ranking_function}")
        verification_result = "Skipped"
        if ranking_function:
            verification_result = self.verifier.verify(reasoning_context, invariants, ranking_function)
            ## issue: 1215fix: get failed in z3 solver function
            ##TODO

        # 4.4 Final Prediction (Synthesis)
        # We synthesize the final label based on verification result or fallback to LLM prediction
        if verification_result == "Verified":
            prediction = {
                "label": "terminating",
                "reasoning": f"Verified ranking function: {ranking_function}. Explanation: {ranking_explanation}"
            }
            raw_prediction = "Verified by Z3"
        else:
            # Fallback to standard LLM prediction if verification failed or no ranking function found
            # We include the intermediate results in the prompt context implicitly via 'loops' or we could add them
            # For now, let's stick to the original prediction prompt but maybe we should update it to include invariants?
            # To keep it simple and robust, we use the original prediction flow as fallback/synthesis
            prediction, raw_prediction = self.predictor.predict(code, loops, prompt_references, invariants, ranking_function)
            if verification_result.startswith("Failed"):
                prediction["reasoning"] += f" (Note: Proposed ranking function '{ranking_function}' failed Z3 verification: {verification_result})"
            elif verification_result == "Failed":
                prediction["reasoning"] += f" (Note: Proposed ranking function '{ranking_function}' failed Z3 verification)"

        # Build comprehensive report payload
        report_payload = {
            "run_id": run_id,
            "original_code": original_code,  # Original input code (before translation)
            "code": code,  # Code after translation (if enabled) or same as original
            "translation": {
                "enabled": self.enable_translation,
                "translated": original_code != code,
                "source_code": original_code,
                "translated_code": code if self.enable_translation and original_code != code else None,
            },
            "loop_extraction": loop_details,
            "embedding": embedding_info,
            "neighbors": [
                {"case_id": cid, "score": score} for cid, score in neighbors
            ],
            "references": [ref.__dict__ for ref in references],
            "neuro_symbolic": {
                "invariants": invariants,
                "ranking_function": ranking_function,
                "ranking_explanation": ranking_explanation,
                "verification_result": verification_result
            },
            "prediction_raw": raw_prediction,
            "prediction": prediction,
        }

        report_path = self.report_manager.persist_report(report_payload)
        # Also write a human-readable log
        self.report_manager.persist_log(report_payload, report_path.stem)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        end_llm_calls = getattr(self.llm_client, "call_count", 0)
        llm_calls = end_llm_calls - start_llm_calls

        return PredictionResult(
            label=prediction["label"],
            reasoning=prediction.get("reasoning", ""),
            loops=loops,
            references=references,
            report_path=report_path,
            invariants=invariants,
            ranking_function=ranking_function,
            verification_result=verification_result,
            run_id=run_id,
            llm_calls=llm_calls,
            duration_seconds=duration
        )




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
