"""High-level termination analysis pipeline."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
import uuid
import numpy as np

from .embeddings import build_embedding_client
from .exceptions import EmbeddingUnavailableError, IndexNotReadyError, LLMUnavailableError
from .knowledge_base import KnowledgeBase
from .llm_client import build_llm_client
from .loop_extractor import LoopExtractor
from .models import KnowledgeCase, Label, PendingReviewCase, PredictionResult
from .prompts_loader import PromptRepository
from .rag_index import HNSWIndexManager
from .translator import CodeTranslator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


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

    # Core flow ------------------------------------------------------------
    def analyze(self, code: str, top_k: int = 5, auto_build_index: bool = True) -> PredictionResult:
        """Run full analysis and produce a report+log capturing each stage.

        The report JSON will include: input code, loop extraction (loops + method + llm_response),
        embedding info, RAG neighbors, references (with similarity), raw LLM prediction and
        the parsed prediction. A plain-text log will also be written for human inspection.
        """
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

        # Filter references for prompt (only include high similarity cases)
        prompt_references = [
            ref for ref in references 
            if ref.metadata.get("similarity", 0.0) > 0.7
        ]

        # Stage 4: Neuro-symbolic Reasoning Pipeline
        
        # 4.1 Invariant Inference
        invariants = self._infer_invariants(code, prompt_references)
        
        # 4.2 Ranking Function Inference
        ranking_function, ranking_explanation = self._infer_ranking(code, invariants, prompt_references)
        
        # 4.3 Z3 Verification
        verification_result = "Skipped"
        if ranking_function:
            verification_result = self._verify_with_z3(code, invariants, ranking_function)

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
            prediction, raw_prediction = self._predict_with_llm(code, loops, prompt_references)
            if verification_result == "Failed":
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

        report_path = self._persist_report(report_payload)
        # Also write a human-readable log
        self._persist_log(report_payload, report_path.stem)

        return PredictionResult(
            label=prediction["label"],
            reasoning=prediction.get("reasoning", ""),
            loops=loops,
            references=references,
            report_path=report_path,
            invariants=invariants,
            ranking_function=ranking_function,
            verification_result=verification_result
        )

    def _infer_invariants(self, code: str, references: List[KnowledgeCase]) -> List[str]:
        prompt = self.prompt_repo.render(
            "invariant_inference",
            code=code,
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        response = self.llm_client.complete(prompt)
        try:
            # Try to parse JSON list
            # Clean up markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("\n", 1)[0]
            return json.loads(cleaned)
        except Exception:
            return []

    def _infer_ranking(self, code: str, invariants: List[str], references: List[KnowledgeCase]) -> tuple[str | None, str]:
        prompt = self.prompt_repo.render(
            "ranking_inference",
            code=code,
            invariants=json.dumps(invariants, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        response = self.llm_client.complete(prompt)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("\n", 1)[0]
            data = json.loads(cleaned)
            return data.get("ranking_function"), data.get("explanation", "")
        except Exception:
            return None, ""

    def _verify_with_z3(self, code: str, invariants: List[str], ranking_function: str) -> str:
        prompt = self.prompt_repo.render(
            "z3_verification",
            code=code,
            invariants="\n".join(invariants),
            ranking_function=ranking_function
        )
        script_response = self.llm_client.complete(prompt)
        
        # Extract python code
        script = script_response
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0].strip()
        elif "```" in script:
            script = script.split("```")[1].split("```")[0].strip()
            
        # Run in temp file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
                tmp.write(script)
                tmp_path = tmp.name
            
            result = subprocess.run(
                ["python", tmp_path], 
                capture_output=True, 
                text=True, 
                timeout=10  # 10 seconds timeout for verification
            )
            output = result.stdout.strip()
            
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
            
            if "Verified" in output:
                return "Verified"
            elif "Failed" in output:
                return "Failed"
            else:
                return f"Error: {output[:100]}..." # Return partial output for debug
                
        except Exception as e:
            return f"Execution Error: {str(e)}"

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}_{report_id}.json"
        path = REPORTS_DIR / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _persist_log(self, report: dict, report_stem: str) -> Path:
        """Write a human-readable log summarizing the report (one log == one run)."""
        log_path = LOGS_DIR / f"{report_stem}.log"
        lines = []
        
        # --- Summary Section (Matches CLI Output) ---
        lines.append(f"EvolveTerm Analysis Report")
        lines.append(f"==========================")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Run ID: {report.get('run_id')}")
        lines.append("")
        
        # Translation
        translation = report.get("translation", {})
        if translation.get("enabled"):
            lines.append("Translation Info")
            lines.append("----------------")
            lines.append(f"Translated: {translation.get('translated')}")
            if translation.get("translated"):
                lines.append("Code was translated to C++.")
            lines.append("")

        # Prediction
        pred = report.get("prediction", {})
        lines.append("Prediction")
        lines.append("----------")
        lines.append(f"Label: {pred.get('label', 'unknown')}")
        lines.append(f"Reasoning: {pred.get('reasoning', '')}")
        lines.append("")

        # References
        lines.append("Referenced Cases")
        lines.append("----------------")
        references = report.get("references", [])
        if references:
            # Header
            lines.append(f"{'Case ID':<40} | {'Label':<15} | {'Similarity'}")
            lines.append("-" * 70)
            for ref in references:
                cid = ref.get("case_id", "")
                lbl = ref.get("label", "")
                meta = ref.get("metadata", {})
                sim = meta.get("similarity", "n/a")
                lines.append(f"{cid:<40} | {lbl:<15} | {sim}")
        else:
            lines.append("No references found.")
        lines.append("")
        lines.append("")

        # --- Detailed Debug Info ---
        lines.append("Detailed Analysis Data")
        lines.append("======================")
        
        # Show original code
        lines.append("--- Original Code ---")
        lines.append(report.get("original_code", ""))
        
        if translation.get("enabled") and translation.get("translated"):
            lines.append("--- Translated Code ---")
            lines.append(translation.get("translated_code", ""))
        
        lines.append("--- Code for Analysis ---")
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
