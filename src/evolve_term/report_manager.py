import uuid
from datetime import datetime
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULT_DIR = Path(__file__).resolve().parents[2] / "results"
REPORTS_DIR = RESULT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
# LOGS_DIR is no longer needed if we don't produce .log files, 
# but keeping existing output structure clean is fine.

class ReportManager:
    def __init__(self):
        pass

    def persist_report(self, report: dict, custom_path: Path = None) -> Path:
        if custom_path:
            path = custom_path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            report_id = report.get("run_id", uuid.uuid4().hex)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}_{report_id}.yaml"
            path = REPORTS_DIR / filename
            
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return path

    # persist_log removed as YAML is human-readable

