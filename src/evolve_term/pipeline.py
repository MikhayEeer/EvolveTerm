"""High-level termination analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple
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
from .verifiers.llm_z3_verifier import Z3Verifier
from .verifiers.seahorn_verifier import SeaHornVerifier, DEFAULT_SEAHORN_IMAGE
from .report_manager import ReportManager
from .predict import Predictor
from .ranking.svm_ranker_wrapper import SVMRankerClient
from .ranking.smt_poly_gen_experimental import SMTLinearRankSynthesizer
import tempfile
import os

class TerminationPipeline:
    """Main façade that ties together embedding, retrieval, and LLM reasoning."""

    def __init__(
        self,
        rebuild_threshold: int = 10,
        embed_config: str = "embed_config.json",
        llm_config: str = "llm_config.json",
        enable_translation: bool = False,
        knowledge_base_path: str | None = None,
        svm_ranker_path: str | None = None,
        ranking_retry_empty: int = 0,
        verifier_backend: str = "z3",
        seahorn_docker_image: str = DEFAULT_SEAHORN_IMAGE,
        seahorn_timeout: int = 60,
    ):
        self.prompt_repo = PromptRepository()
        self.llm_client = build_llm_client(llm_config)
        self.loop_extractor = LoopExtractor(self.llm_client,# loop extractor also use same model with prediction 
                                            self.prompt_repo)
        self.embedding_client = build_embedding_client(embed_config)
        self.knowledge_base = KnowledgeBase(rebuild_threshold=rebuild_threshold, 
                                            storage_path=knowledge_base_path)
        
        # Derive index paths from KB path if provided
        idx_path = None
        idx_meta = None
        if knowledge_base_path:
            p = Path(knowledge_base_path)
            idx_path = p.parent / (p.stem + "_index.bin")
            idx_meta = p.parent / (p.stem + "_index_meta.json")
            
        self.index_manager = HNSWIndexManager(
            dimension=self.embedding_client.dimension,
            index_path=idx_path,
            meta_path=idx_meta
        )
        self.enable_translation = enable_translation
        self.translator = CodeTranslator(config_name=llm_config) if enable_translation else None
        self.verifier_backend = verifier_backend.lower()
        self.seahorn_docker_image = seahorn_docker_image
        self.seahorn_timeout = seahorn_timeout
        self.z3_verifier = Z3Verifier(self.llm_client, self.prompt_repo)
        self.seahorn_verifier: SeaHornVerifier | None = None
        self.report_manager = ReportManager()
        self.predictor = Predictor(self.llm_client, self.prompt_repo)
        self.svm_ranker = SVMRankerClient(svm_ranker_path) if svm_ranker_path else None
        self.smt_ranker = SMTLinearRankSynthesizer()
        self.ranking_retry_empty = max(0, ranking_retry_empty)

    def _get_seahorn_verifier(self) -> SeaHornVerifier:
        if self.seahorn_verifier is None:
            self.seahorn_verifier = SeaHornVerifier(
                docker_image=self.seahorn_docker_image,
                timeout_seconds=self.seahorn_timeout,
            )
        return self.seahorn_verifier

    # Core flow ------------------------------------------------------------
    def analyze(self, 
                code: str, 
                top_k: int = 5, 
                auto_build_index: bool = True, 
                use_rag_in_reasoning: bool = True, 
                use_svm_ranker: bool = False, 
                use_smt_synth: bool = False,
                known_terminating: bool = False,
                # Ablation parameters
                extraction_prompt_version: str = "v2", 
                use_loops_for_embedding: bool = True,
                use_loops_for_reasoning: bool = True,
                output_path: Optional[Path] = None,
                metadata: Optional[Dict[str, Any]] = None
                ) -> PredictionResult:

        """Run full analysis and produce a report+log capturing each stage.

        The report JSON will include: input code, loop extraction (loops + method + llm_response),
        embedding info, RAG neighbors, references (with similarity), raw LLM prediction and
        the parsed prediction. A plain-text log will also be written for human inspection.
        """
        start_time = datetime.now()
        start_llm_calls = getattr(self.llm_client, "call_count", 0)
        run_id = uuid.uuid4().hex
        verifier_backend = self.verifier_backend
        if verifier_backend != "seahorn":
            raise ValueError("Strict termination proof requires verifier_backend='seahorn'.")

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
        # Use configured prompt version
        prompt_name = f"loop_extraction/yaml_{extraction_prompt_version}"
        loops = self.loop_extractor.extract(code, prompt_name=prompt_name)

        if verifier_backend == "seahorn" and use_loops_for_reasoning and loops:
            print("[Warning] SeaHorn verification maps loops by source order; ensure extracted loops match source order.")
        
        loop_details = {
            "loops": loops,
            "method": getattr(self.loop_extractor, "last_method", None),
            "llm_response": getattr(self.loop_extractor, "last_response", None),
            "prompt_version": extraction_prompt_version
        }

        # Stage 2: embedding
        # Decide what to embed based on ablation settings
        text_to_embed = "\n".join(loops) if (loops and use_loops_for_embedding) else code
        embedding_vector = self.embedding_client.embed(text_to_embed)
        
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
            "source": "loops" if (loops and use_loops_for_embedding) else "code"
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

        # Stage 4: Neuro-symbolic Reasoning Pipeline (Iterative)
        
        # Decide analysis targets based on ablation settings
        if use_loops_for_reasoning and loops:
            analysis_targets = loops
        else:
            analysis_targets = [code]
        
        loop_analyses = []
        final_verification_result = "Verified" # Optimistic default
        final_ranking_functions = []
        final_invariants = []
        
        for i, loop_code in enumerate(analysis_targets):
            loop_id = i + 1
            
            # Context preparation: Replace LOOP{id} with summaries of previous loops
            current_context = loop_code
            for prev_res in loop_analyses:
                prev_id = prev_res['id']
                placeholder = f"LOOP{prev_id}"
                if placeholder in current_context:
                    # Replace placeholder with a comment summarizing the inner loop
                    # This makes the code valid for C parsers (like SVMRanker) and informative for LLMs
                    summary = f"/* Nested Loop {prev_id}: {prev_res['verification_result']} (RF: {prev_res['ranking_function']}) */"
                    current_context = current_context.replace(placeholder, summary)
            
            print(f"[Info] Analyzing Loop {loop_id}...")
            
            # 4.1 Invariant Inference
            # invariants = self.predictor.infer_invariants(current_context, reasoning_references)
            invariants = [] # Ablation: disabled

            # 4.2 Ranking Function Inference
            ranking_function = None
            ranking_explanation = ""
            verification_result = "Skipped"

            def verify_candidate(rf: str, explanation: str) -> Tuple[str, str]:
                if not rf:
                    return "Skipped", ""
                seahorn = self._get_seahorn_verifier()
                result = seahorn.verify(
                    code,
                    loop_invariants={loop_id: invariants},
                    loop_rankings={loop_id: rf},
                )
                report = seahorn.last_report or {}
                instrumented = report.get("instrumented_loop_ids") or []
                if loop_id not in instrumented:
                    return "Skipped (SeaHorn no instrumentation)", explanation
                if result.startswith("Error"):
                    return result, explanation
                return result, explanation

            # 4.2.1 SMT piecewise linear synthesis (experimental)
            if use_smt_synth:
                smt_rf = self.smt_ranker.synthesize(current_context, invariants)
                if smt_rf:
                    smt_reason = getattr(self.smt_ranker, "last_reason", "")
                    smt_expl = "SMT synthesized piecewise linear ranking function."
                    if smt_reason:
                        smt_expl = f"{smt_expl} Reason={smt_reason}."
                    verification_result, ranking_explanation = verify_candidate(smt_rf, smt_expl)
                    ranking_function = smt_rf
                    if verification_result == "Verified":
                        loop_analyses.append({
                            "id": loop_id,
                            "code": loop_code,
                            "context_used": current_context,
                            "invariants": invariants,
                            "ranking_function": ranking_function,
                            "verification_result": verification_result,
                            "verification_backend": verifier_backend,
                            "explanation": ranking_explanation
                        })
                        final_ranking_functions.append(f"Loop {loop_id}: {ranking_function}")
                        final_invariants.extend(invariants)
                        continue
                    ranking_function = None
                    ranking_explanation = ""
                    verification_result = "Skipped"
            
            # Try SVMRanker if enabled and available
            if use_svm_ranker and self.svm_ranker:
                # 1. Ask LLM for template/params
                _, explanation, metadata = self.predictor.infer_ranking(
                    current_context, invariants, reasoning_references, 
                    mode="template", known_terminating=known_terminating,
                    retry_empty=self.ranking_retry_empty, log_prefix=f"Loop {loop_id} template"
                )
                
                rf_type = metadata.get("type", "lnested")
                depth = metadata.get("depth", 1)
                rf_type_value = str(rf_type or "").lower()
                if "piecewise" in rf_type_value:
                    svm_mode = "lpiecewiseext"
                elif "multiext" in rf_type_value or "multi" in rf_type_value:
                    svm_mode = "lmultiext"
                elif "lexiext" in rf_type_value or "lexi" in rf_type_value:
                    svm_mode = "llexiext"
                else:
                    svm_mode = "llexiext"

                predicates = []
                if svm_mode == "lpiecewiseext":
                    predicates = self.predictor.infer_piecewise_predicates(
                        current_context,
                        invariants,
                        reasoning_references,
                        retry_empty=self.ranking_retry_empty,
                        log_prefix=f"Loop {loop_id} piecewise",
                    )
                
                # Run SVMRanker
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, encoding='utf-8') as tmp:
                        # For SVMRanker, we need a complete C file. 
                        # 'current_context' might be just a loop snippet.
                        # If it's a snippet, we wrap it in main() for SVMRanker?
                        # SVMRanker usually expects a full program.
                        # If 'current_context' is just "while(...) { ... }", we need to wrap it.
                        # However, 'code' is the full program.
                        # If we use 'code', SVMRanker analyzes the whole thing, not just this loop.
                        # But for abstract loops, we want to analyze the abstract version.
                        # So we write 'current_context' wrapped in main.
                        
                        content_to_write = current_context
                        if "main" not in current_context:
                            content_to_write = f"void main() {{\n{current_context}\n}}"
                            # Add variable declarations? This is tricky. 
                            # SVMRanker needs valid C. Snippets might miss declarations.
                            # Fallback: If snippet, maybe skip SVMRanker or try best effort?
                            # For now, let's try writing it. If it fails compilation, SVMRanker will fail.
                        
                        tmp.write(content_to_write)
                        tmp_path = tmp.name
                    
                    # Call SVMRanker
                    svm_result_status, svm_rf, _, _, _ = self.svm_ranker.run(
                        Path(tmp_path),
                        mode=svm_mode,
                        depth=depth,
                        predicates=predicates,
                    )
                    
                    if svm_result_status == "TERMINATE" and svm_rf:
                        ranking_function = svm_rf
                        ranking_explanation = f"Generated and Verified by SVMRanker (mode={svm_mode}, depth={depth}). LLM Explanation: {explanation}"
                        verification_result = "Verified"
                    elif svm_result_status == "NONTERM":
                         ranking_explanation = f"SVMRanker returned NONTERM (mode={svm_mode})."
                    
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"[Warning] SVMRanker execution failed for Loop {loop_id}: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # Fallback to direct LLM generation
            if not ranking_function:
                ranking_function, ranking_explanation, _ = self.predictor.infer_ranking(
                    current_context, invariants, reasoning_references, 
                    mode="direct", known_terminating=known_terminating,
                    retry_empty=self.ranking_retry_empty, log_prefix=f"Loop {loop_id} direct"
                )

            # 4.3 Verification
            if verification_result != "Verified":
                if ranking_function:
                    verification_result, _ = verify_candidate(ranking_function, ranking_explanation)
                    if verification_result == "Skipped (SeaHorn no instrumentation)":
                        print(f"[Warning] SeaHorn did not instrument Loop {loop_id}. Check braces/line start.")
                    elif verification_result.startswith("Error"):
                        print(f"[Warning] Verifier error on Loop {loop_id}: {verification_result}")
                else:
                    verification_result = "Skipped"
            
            # Store results
            loop_analyses.append({
                "id": loop_id,
                "code": loop_code,
                "context_used": current_context,
                "invariants": invariants,
                "ranking_function": ranking_function,
                "verification_result": verification_result,
                "verification_backend": verifier_backend,
                "explanation": ranking_explanation
            })
            
            final_ranking_functions.append(f"Loop {loop_id}: {ranking_function}")
            final_invariants.extend(invariants)
            
            if verification_result != "Verified":
                final_verification_result = "Failed" # One failure fails all (conservative)

        # 4.4 Final Prediction (Synthesis)
        # Synthesize based on aggregated results
        ranking_function_summary = "; ".join(final_ranking_functions)
        
        if final_verification_result == "Verified":
            prediction = {
                "label": "terminating",
                "reasoning": f"All loops verified. Ranking Functions: {ranking_function_summary}"
            }
            if verifier_backend == "seahorn":
                raw_prediction = "Verified by SeaHorn/SVMRanker"
            else:
                raw_prediction = "Verified by Z3/SVMRanker"
        else:
            # Fallback: If any loop failed verification, ask LLM for final verdict on the WHOLE code
            # but provide the loop analysis details.
            
            # Construct a detailed prompt context
            analysis_summary = "\n".join([
                f"Loop {res['id']} Analysis:\n- Code: {res['code'][:50]}...\n- RF: {res['ranking_function']}\n- Result: {res['verification_result']}"
                for res in loop_analyses
            ])
            
            # We pass the original code and the analysis summary to the predictor
            # Note: predictor.predict signature is fixed, so we might need to hack the 'loops' arg
            # to pass the summary.
            
            prediction, raw_prediction = self.predictor.predict(
                code, 
                [analysis_summary], # Pass summary as "loops" context
                prompt_references, 
                final_invariants, 
                ranking_function_summary
            )
            
            if final_verification_result != "Verified":
                 prediction["reasoning"] += f" [Verification Failed. Analysis: {analysis_summary}]"

        # Build comprehensive report payload
        report_payload = {
            "run_id": run_id,
            "original_code": original_code,
            "code": code,
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
                "invariants": final_invariants,
                "ranking_function": ranking_function_summary,
                "verification_result": final_verification_result,
                "verification_backend": verifier_backend,
                "loop_analyses": loop_analyses # Detailed per-loop results
            },
            "prediction_raw": raw_prediction,
            "prediction": prediction,
        }

        # Inject injected metadata for report
        if metadata:
            # We want meta to be the first key if possible, but python dict ensures insertion order
            # in 3.7+. We create a new dict to put meta first.
            new_payload = {"meta": metadata}
            new_payload.update(report_payload)
            report_payload = new_payload

        report_path = self.report_manager.persist_report(report_payload, custom_path=output_path)

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
