"""Handler for the 'svmranker' command."""
from __future__ import annotations
import json
import yaml
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
from rich.console import Console
import typer

from ..svm_ranker import SVMRankerClient
from ..cli_utils import collect_files, validate_yaml_required_keys
from ..utils import LiteralDumper

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[3]

def resolve_svm_ranker_root(path: Path) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"SVMRanker 路径不存在: {path}")
    if path.is_file():
        if path.name == "CLIMain.py" and path.parent.name == "src":
            path = path.parent.parent
        else:
            raise typer.BadParameter("SVMRanker 路径应为仓库根目录或 src/CLIMain.py 文件。")

    if (path / "src" / "CLIMain.py").exists():
        return path
    if path.name == "src" and (path / "CLIMain.py").exists():
        return path.parent
    raise typer.BadParameter("SVMRanker 路径无效，未找到 src/CLIMain.py。")

class SVMRankerHandler:
    def __init__(self, svm_ranker_path: Path):
        self.root = resolve_svm_ranker_root(svm_ranker_path)
        self.client = SVMRankerClient(str(self.root))

    def run(self, input_path: Path, output: Path, recursive: bool):
        if output is None:
            raise typer.BadParameter("Output directory is required.")
        if output.exists() and not output.is_dir():
            raise typer.BadParameter("Output must be a directory.")
        output.mkdir(parents=True, exist_ok=True)
        
        def _check_yaml_required_keys(path: Path, content: Any, strict: bool) -> bool:
            missing = validate_yaml_required_keys(path, content)
            if not missing:
                return True
            console.print(f"[red]YAML missing keys in {path}: {', '.join(missing)}[/red]")
            if strict:
                return False
            return False

        def _check_template_yaml(path: Path, entries: List[Dict[str, Any]], strict: bool) -> bool:
            if not entries:
                console.print(f"[red]YAML {path} has no ranking results.[/red]")
                if strict:
                    raise typer.Exit(code=1)
                return False
            invalid_loops = []
            for entry in entries:
                if not isinstance(entry, dict):
                    invalid_loops.append("unknown")
                    continue
                template_type = entry.get("template_type") or entry.get("type")
                template_depth = entry.get("template_depth") or entry.get("depth")
                if template_type in (None, "") or template_depth in (None, ""):
                    invalid_loops.append(entry.get("loop_id") or entry.get("id") or "unknown")
            if invalid_loops:
                msg = (
                    f"YAML {path} is not a ranking template output. "
                    f"Missing template_type/template_depth for loops: {', '.join(map(str, invalid_loops))}"
                )
                console.print(f"[red]{msg}[/red]")
                if strict:
                    raise typer.Exit(code=1)
                return False
            return True

        def parse_results(content: Any) -> List[Dict[str, Any]]:
            if isinstance(content, dict) and "ranking_results" in content:
                return content["ranking_results"] or []
            if isinstance(content, list):
                return content
            return []

        def normalize_status(value: Any) -> str:
            return re.sub(r"[\s\-]+", "", str(value or "")).upper()

        def classify_output_dir(results: List[Dict[str, Any]]) -> str:
            if not results:
                return "failed"
            normalized = [normalize_status(item.get("status")) for item in results]
            if any(status in {"ERROR", "FAILED", "FAIL"} or not status for status in normalized):
                return "failed"
            if any(status == "UNKNOWN" for status in normalized):
                return "unknown"
            if all(status in {"TERMINATE", "NONTERM"} for status in normalized):
                return "certain"
            return "failed"

        def resolve_source_code(
            entry: Dict[str, Any],
            base_dir: Path,
            fallback_source_path: Optional[str],
        ) -> Tuple[str, str, Optional[str]]:
            loop_id = entry.get("loop_id") or entry.get("id")
            source_path = entry.get("source_path") or fallback_source_path
            if source_path:
                path = Path(source_path)
                candidate_paths = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path, REPO_ROOT / path]
                for candidate in candidate_paths:
                    if candidate.exists():
                        try:
                            content = candidate.read_text(encoding="utf-8")
                            print(f"[Debug] Loop {loop_id}: using source_path {candidate}")
                            return content, "source_path", str(candidate)
                        except Exception as exc:
                            print(f"[Debug] Loop {loop_id}: failed to read {candidate}: {exc}")
                            break
                tried = ", ".join(str(item) for item in candidate_paths)
                print(f"[Debug] Loop {loop_id}: source_path not found. Tried: {tried}")
                    
            code = entry.get("code", "")
            print(f"[Debug] Loop {loop_id}: falling back to entry code (len={len(code)})")
            return code, "loop_code", None

        def run_on_entry(
            entry: Dict[str, Any],
            base_dir: Path,
            fallback_source_path: Optional[str],
        ) -> Tuple[Dict[str, Any], str]:
            code, code_source, code_source_path = resolve_source_code(
                entry, base_dir, fallback_source_path
            )
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            mode = "lmulti" if "multi" in str(template_type).lower() else "lnested"
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1

            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            try:
                status, rf, rf_list, log = self.client.run(Path(tmp_path), mode=mode, depth=depth_val)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            result = {
                "loop_id": entry.get("loop_id") or entry.get("id"),
                "template_type": template_type,
                "template_depth": depth_val,
                "svm_mode": mode,
                "status": status,
                "ranking_function": rf,
                "ranking_functions": rf_list,
                "input_code_source": code_source,
            }
            if code_source_path:
                result["input_code_path"] = code_source_path

            return result, log

        def process_yaml(f: Path, strict: bool) -> Tuple[Optional[Dict[str, Any]], str]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                return {"source_file": f.name, "error": f"YAML parse error: {e}"}, f"YAML Error: {e}"
            if not _check_yaml_required_keys(f, content, strict):
                return None, "YAML validation failed"

            entries = parse_results(content)
            base_dir = f.parent
            fallback_source_path = None
            if isinstance(content, dict) and len(entries) == 1:
                top_source_path = content.get("source_path")
                if isinstance(top_source_path, str) and top_source_path.strip():
                    fallback_source_path = top_source_path

            if not _check_template_yaml(f, entries, strict):
                return None, "Non-template YAML"
            
            results = []
            full_log = []
            
            for entry in entries:
                res, log = run_on_entry(entry, base_dir, fallback_source_path)
                results.append(res)
                full_log.append(f"--- Entry Loop ID: {res.get('loop_id')} ---\n{log}\n" + "="*40 + "\n")
                
            return {
                "input_file": str(f),
                "source_file": f.name,
                "task": "svmranker",
                "svmranker_result": results,
            }, "\n".join(full_log)

        if input_path.is_file():
            result, full_log = process_yaml(input_path, True)
            if result is None:
                raise typer.Exit(code=1)
            
            results = result.get("svmranker_result") or []
            bucket = classify_output_dir(results)
            out_dir = output / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (input_path.stem + "_svm.yml")
            
            with open(out_path, "w", encoding="utf-8") as yf:
                yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
            console.print(f"Saved to {out_path}")
            
            log_path = out_path.with_suffix(".svmranker.txt")
            log_path.write_text(full_log, encoding="utf-8")
            console.print(f"Saved log to {log_path}")

        elif input_path.is_dir():
            files = collect_files(input_path, recursive, extensions={".yml", ".yaml"})

            for f in files:
                try:
                    result, full_log = process_yaml(f, False)
                    if not result: continue
                    
                    results = result.get("svmranker_result") or []
                    bucket = classify_output_dir(results)
                    try:
                        rel_path = f.relative_to(input_path)
                        out_path = output / bucket / rel_path.with_suffix(".yml")
                        out_path = out_path.parent / (out_path.stem + "_svm.yml")
                    except ValueError:
                        out_path = output / bucket / (f.stem + "_svm.yml")
                    
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as yf:
                        yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
                         
                    log_path = out_path.with_suffix(".svmranker.txt")
                    log_path.write_text(full_log, encoding="utf-8")
                        
                except Exception as e:
                    console.print(f"[red]Error processing {f.name}: {e}[/red]")
