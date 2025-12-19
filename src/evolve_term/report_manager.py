import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULT_DIR = Path(__file__).resolve().parents[2] / "results"
REPORTS_DIR = RESULT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = RESULT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class ReportManager:
    def __init__(self):
        pass

    def persist_report(self, report: dict) -> Path:
        report_id = report.get("run_id", uuid.uuid4().hex)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}_{report_id}.json"
        path = REPORTS_DIR / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def persist_log(self, report: dict, report_stem: str) -> Path:
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

        # Neuro-symbolic Info
        ns = report.get("neuro_symbolic", {})
        if ns:
            lines.append("Neuro-symbolic Analysis")
            lines.append("-----------------------")
            lines.append(f"Verification Result: {ns.get('verification_result', 'N/A')}")
            if ns.get("ranking_function"):
                lines.append(f"Ranking Function: {ns.get('ranking_function')}")
                lines.append(f"Explanation: {ns.get('ranking_explanation', '')}")
            if ns.get("invariants"):
                lines.append("Invariants:")
                for inv in ns.get("invariants"):
                    lines.append(f"  - {inv}")
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
