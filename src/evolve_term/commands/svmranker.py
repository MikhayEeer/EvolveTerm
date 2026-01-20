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
BOOGIE_ROOT = Path("/home/clexma/Desktop/fox3/TermDB/TerminationDatabase/Data_boogie")

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

    @staticmethod
    def _format_entry_log_prefix(
        loop_id: Any,
        code_source: str,
        code_source_path: Optional[str],
        boogie_path: Optional[str],
        template_type: Any,
        template_depth: Any,
        mode: str,
        depth_val: int,
        depth_original: Optional[int] = None,
        depth_bump: Optional[int] = None,
    ) -> str:
        lines = [
            f"[Entry] loop_id={loop_id}",
            f"[Input] source={code_source} code_path={code_source_path or '-'} boogie_path={boogie_path or '-'}",
            f"[Template] type={template_type} depth={template_depth} svm_mode={mode} depth={depth_val}",
        ]
        if depth_original is not None and depth_bump is not None:
            lines.append(f"[Template] depth_original={depth_original} depth_bump={depth_bump}")
        return "\n".join(lines) + "\n"

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
            console.print(f"[bold red]ERROR: YAML missing keys in {path}: {', '.join(missing)}[/bold red]")
            if strict:
                return False
            return False

        def _check_template_yaml(path: Path, entries: List[Dict[str, Any]], strict: bool) -> bool:
            if not entries:
                console.print(f"[bold red]ERROR: YAML {path} has no ranking results.[/bold red]")
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
                console.print(f"[bold red]ERROR: {msg}[/bold red]")
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

        def resolve_boogie_path(source_path: Optional[str]) -> Optional[Path]:
            if not source_path:
                return None
            path = Path(source_path)
            rel = None
            if path.is_absolute():
                parts = path.parts
                if "data" in parts:
                    idx = parts.index("data")
                    rel = Path(*parts[idx + 1:])
            else:
                parts = path.parts
                if parts and parts[0] == "data":
                    rel = Path(*parts[1:])
            if rel is None:
                return None
            base = BOOGIE_ROOT / rel
            for suffix in (".tpl", ".bpl"):
                candidate = base.with_suffix(suffix)
                if candidate.exists():
                    return candidate
            return None

        def resolve_source_code(
            entry: Dict[str, Any],
            base_dir: Path,
            fallback_source_path: Optional[str],
        ) -> Tuple[str, str, Optional[str], Optional[str]]:
            loop_id = entry.get("loop_id") or entry.get("id")
            source_path = entry.get("source_path") or fallback_source_path
            boogie_path = resolve_boogie_path(source_path)
            if boogie_path:
                print(f"[Debug] Loop {loop_id}: using boogie_path {boogie_path}")
                return "", "boogie_path", str(boogie_path), str(boogie_path)
            if source_path:
                path = Path(source_path)
                candidate_paths = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path, REPO_ROOT / path]
                for candidate in candidate_paths:
                    if candidate.exists():
                        try:
                            content = candidate.read_text(encoding="utf-8")
                            print(f"[Debug] Loop {loop_id}: using source_path {candidate}")
                            return content, "source_path", str(candidate), None
                        except Exception as exc:
                            print(f"[Debug] Loop {loop_id}: failed to read {candidate}: {exc}")
                            break
                tried = ", ".join(str(item) for item in candidate_paths)
                print(f"[Debug] Loop {loop_id}: source_path not found. Tried: {tried}")
                    
            code = entry.get("code", "")
            print(f"[Debug] Loop {loop_id}: falling back to entry code (len={len(code)})")
            return code, "loop_code", None, None

        def run_on_entry(
            entry: Dict[str, Any],
            base_dir: Path,
            fallback_source_path: Optional[str],
        ) -> Tuple[Dict[str, Any], str]:
            code, code_source, code_source_path, boogie_path = resolve_source_code(
                entry, base_dir, fallback_source_path
            )
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            mode = "lmulti" if "multi" in str(template_type).lower() else "lnested"
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1

            if boogie_path:
                status, rf, rf_list, log = self.client.run(Path(boogie_path), mode=mode, depth=depth_val)
            else:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name

                try:
                    status, rf, rf_list, log = self.client.run(Path(tmp_path), mode=mode, depth=depth_val)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            log_prefix = self._format_entry_log_prefix(
                loop_id=loop_id,
                code_source=code_source,
                code_source_path=code_source_path,
                boogie_path=boogie_path,
                template_type=template_type,
                template_depth=template_depth,
                mode=mode,
                depth_val=depth_val,
            )
            def _stringify(value: Any) -> Optional[str]:
                if value is None:
                    return None
                if isinstance(value, str):
                    return value
                try:
                    return str(value)
                except Exception:
                    return repr(value)

            rf_str = _stringify(rf)
            rf_list_str = []
            for item in rf_list or []:
                rf_list_str.append(_stringify(item) or "")

            result = {
                "loop_id": entry.get("loop_id") or entry.get("id"),
                "template_type": template_type,
                "template_depth": depth_val,
                "svm_mode": mode,
                "status": status,
                "ranking_function": rf_str,
                "ranking_functions": rf_list_str,
                "input_code_source": code_source,
            }
            if code_source_path:
                result["input_code_path"] = code_source_path
            if boogie_path:
                result["input_boogie_path"] = boogie_path

            return result, log_prefix + log

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
                    console.print(f"[bold red]ERROR: Error processing {f.name}: {e}[/bold red]")

    def rerun_failed(self, input_path: Path, output: Path, recursive: bool) -> None:
        if output is None:
            raise typer.BadParameter("Output directory is required.")
        if output.exists() and not output.is_dir():
            raise typer.BadParameter("Output must be a directory.")
        output.mkdir(parents=True, exist_ok=True)

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

        def find_failed_dirs(root: Path) -> List[Path]:
            if root.name == "failed":
                return [root]
            direct = root / "failed"
            if direct.is_dir():
                return [direct]
            if recursive:
                return sorted({p for p in root.rglob("failed") if p.is_dir()})
            return []

        def load_failed_entries(content: Any) -> List[Dict[str, Any]]:
            if isinstance(content, dict):
                return content.get("svmranker_result") or []
            if isinstance(content, list):
                return content
            return []

        def stringify(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            try:
                return str(value)
            except Exception:
                return repr(value)

        def run_on_failed_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
            loop_id = entry.get("loop_id") or entry.get("id")
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            mode = "lmulti" if "multi" in str(template_type).lower() else "lnested"
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1

            input_boogie_path = entry.get("input_boogie_path")
            input_code_path = entry.get("input_code_path")
            input_code_source = entry.get("input_code_source")
            code = entry.get("code")

            temp_path: Optional[Path] = None
            target_path: Optional[Path] = None
            if input_boogie_path and Path(input_boogie_path).exists():
                target_path = Path(input_boogie_path)
                input_code_source = "boogie_path"
            elif input_code_path and Path(input_code_path).exists():
                target_path = Path(input_code_path)
                input_code_source = "source_path"
            elif isinstance(code, str) and code.strip():
                temp = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8")
                temp.write(code)
                temp.close()
                temp_path = Path(temp.name)
                target_path = temp_path
                input_code_source = "loop_code"

            if not target_path:
                msg = "[Error] Missing input code path for rerun."
                result = {
                    "loop_id": loop_id,
                    "template_type": template_type,
                    "template_depth": depth_val,
                    "svm_mode": mode,
                    "status": "ERROR",
                    "ranking_function": None,
                    "ranking_functions": [],
                    "input_code_source": input_code_source or "unknown",
                }
                if input_code_path:
                    result["input_code_path"] = input_code_path
                if input_boogie_path:
                    result["input_boogie_path"] = input_boogie_path
                return result, msg

            try:
                status, rf, rf_list, log = self.client.run(target_path, mode=mode, depth=depth_val)
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)

            log_prefix = self._format_entry_log_prefix(
                loop_id=loop_id,
                code_source=input_code_source or "unknown",
                code_source_path=input_code_path,
                boogie_path=input_boogie_path,
                template_type=template_type,
                template_depth=template_depth,
                mode=mode,
                depth_val=depth_val,
            )
            rf_str = stringify(rf)
            rf_list_str = []
            for item in rf_list or []:
                rf_list_str.append(stringify(item) or "")

            result = {
                "loop_id": loop_id,
                "template_type": template_type,
                "template_depth": depth_val,
                "svm_mode": mode,
                "status": status,
                "ranking_function": rf_str,
                "ranking_functions": rf_list_str,
                "input_code_source": input_code_source or "unknown",
            }
            if input_code_path:
                result["input_code_path"] = input_code_path
            if input_boogie_path:
                result["input_boogie_path"] = input_boogie_path

            return result, log_prefix + log

        def process_failed_yaml(f: Path) -> Tuple[Optional[Dict[str, Any]], str]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                return None, f"YAML Error: {e}"

            entries = load_failed_entries(content)
            if not entries:
                return None, "No svmranker_result entries found."

            results = []
            full_log = []
            for entry in entries:
                if not isinstance(entry, dict):
                    res = {
                        "loop_id": "unknown",
                        "template_type": "lnested",
                        "template_depth": 1,
                        "svm_mode": "lnested",
                        "status": "ERROR",
                        "ranking_function": None,
                        "ranking_functions": [],
                        "input_code_source": "unknown",
                    }
                    log = "[Error] Invalid svmranker_result entry (not a dict)."
                else:
                    res, log = run_on_failed_entry(entry)
                results.append(res)
                full_log.append(f"--- Entry Loop ID: {res.get('loop_id')} ---\n{log}\n" + "=" * 40 + "\n")

            input_file = None
            source_file = None
            if isinstance(content, dict):
                input_file = content.get("input_file")
                source_file = content.get("source_file")
            result = {
                "input_file": input_file or str(f),
                "source_file": source_file or f.name,
                "task": "svmranker",
                "svmranker_result": results,
            }
            return result, "\n".join(full_log)

        def build_output_path(root: Path, f: Path, bucket: str) -> Path:
            try:
                rel_path = f.relative_to(root)
            except ValueError:
                rel_path = Path(f.name)
            parts = list(rel_path.parts)
            if "failed" in parts:
                idx = parts.index("failed")
                parts[idx] = bucket
                rel_path = Path(*parts)
            else:
                rel_path = Path(bucket) / rel_path
            return output / rel_path

        if input_path.is_file():
            root_dir = input_path.parent
            files = [input_path]
        else:
            failed_dirs = find_failed_dirs(input_path)
            if not failed_dirs:
                raise typer.BadParameter("未找到 failed 目录，请传入 failed 目录或其父目录。")
            root_dir = input_path
            files = []
            for failed_dir in failed_dirs:
                files.extend(collect_files(failed_dir, recursive, extensions={".yml", ".yaml"}))

        for f in files:
            try:
                result, full_log = process_failed_yaml(f)
                if not result:
                    console.print(f"[bold red]ERROR: Failed to parse {f.name}[/bold red]")
                    continue
                results = result.get("svmranker_result") or []
                bucket = classify_output_dir(results)
                out_path = build_output_path(root_dir, f, bucket)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as yf:
                    yaml.dump(result, yf, sort_keys=False, allow_unicode=True)

                log_path = out_path.with_suffix(".svmranker.txt")
                log_path.write_text(full_log, encoding="utf-8")

                if bucket != "failed" and f.exists():
                    try:
                        f.unlink()
                        old_log = f.with_suffix(".svmranker.txt")
                        if old_log.exists():
                            old_log.unlink()
                    except Exception as exc:
                        console.print(f"[bold red]ERROR: Failed to delete {f.name}: {exc}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]ERROR: Error processing {f.name}: {e}[/bold red]")

    def rerun_unknown(
        self,
        input_path: Path,
        output: Path,
        recursive: bool,
        depth_bump: int = 1,
        sample_strategy: str = "CONSTRAINT",
        template_strategy: str = "FULL",
        cutting_strategy: str = "POS",
    ) -> None:
        if output is None:
            raise typer.BadParameter("Output directory is required.")
        if output.exists() and not output.is_dir():
            raise typer.BadParameter("Output must be a directory.")
        output.mkdir(parents=True, exist_ok=True)

        depth_bump = max(0, int(depth_bump))

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

        def find_unknown_dirs(root: Path) -> List[Path]:
            if root.name == "unknown":
                return [root]
            direct = root / "unknown"
            if direct.is_dir():
                return [direct]
            if recursive:
                return sorted({p for p in root.rglob("unknown") if p.is_dir()})
            return []

        def load_unknown_entries(content: Any) -> List[Dict[str, Any]]:
            if isinstance(content, dict):
                return content.get("svmranker_result") or []
            if isinstance(content, list):
                return content
            return []

        def stringify(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            try:
                return str(value)
            except Exception:
                return repr(value)

        def run_on_unknown_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
            loop_id = entry.get("loop_id") or entry.get("id")
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            mode = "lmulti" if "multi" in str(template_type).lower() else "lnested"
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1
            depth_original = depth_val
            depth_val = max(1, depth_val + depth_bump)

            input_boogie_path = entry.get("input_boogie_path")
            input_code_path = entry.get("input_code_path")
            input_code_source = entry.get("input_code_source")
            code = entry.get("code")

            temp_path: Optional[Path] = None
            target_path: Optional[Path] = None
            if input_boogie_path and Path(input_boogie_path).exists():
                target_path = Path(input_boogie_path)
                input_code_source = "boogie_path"
            elif input_code_path and Path(input_code_path).exists():
                target_path = Path(input_code_path)
                input_code_source = "source_path"
            elif isinstance(code, str) and code.strip():
                temp = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8")
                temp.write(code)
                temp.close()
                temp_path = Path(temp.name)
                target_path = temp_path
                input_code_source = "loop_code"

            if not target_path:
                msg = "[Error] Missing input code path for rerun."
                result = {
                    "loop_id": loop_id,
                    "template_type": template_type,
                    "template_depth": depth_val,
                    "svm_mode": mode,
                    "status": "ERROR",
                    "ranking_function": None,
                    "ranking_functions": [],
                    "input_code_source": input_code_source or "unknown",
                }
                if input_code_path:
                    result["input_code_path"] = input_code_path
                if input_boogie_path:
                    result["input_boogie_path"] = input_boogie_path
                return result, msg

            try:
                status, rf, rf_list, log = self.client.run(
                    target_path,
                    mode=mode,
                    depth=depth_val,
                    sample_strategy=sample_strategy,
                    cutting_strategy=cutting_strategy,
                    template_strategy=template_strategy,
                )
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)

            log_prefix = self._format_entry_log_prefix(
                loop_id=loop_id,
                code_source=input_code_source or "unknown",
                code_source_path=input_code_path,
                boogie_path=input_boogie_path,
                template_type=template_type,
                template_depth=template_depth,
                mode=mode,
                depth_val=depth_val,
                depth_original=depth_original,
                depth_bump=depth_bump,
            )
            rf_str = stringify(rf)
            rf_list_str = []
            for item in rf_list or []:
                rf_list_str.append(stringify(item) or "")

            result = {
                "loop_id": loop_id,
                "template_type": template_type,
                "template_depth": depth_val,
                "svm_mode": mode,
                "status": status,
                "ranking_function": rf_str,
                "ranking_functions": rf_list_str,
                "input_code_source": input_code_source or "unknown",
            }
            if input_code_path:
                result["input_code_path"] = input_code_path
            if input_boogie_path:
                result["input_boogie_path"] = input_boogie_path

            return result, log_prefix + log

        def process_unknown_yaml(f: Path) -> Tuple[Optional[Dict[str, Any]], str]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                return None, f"YAML Error: {e}"

            entries = load_unknown_entries(content)
            if not entries:
                return None, "No svmranker_result entries found."

            results = []
            full_log = []
            for entry in entries:
                if not isinstance(entry, dict):
                    res = {
                        "loop_id": "unknown",
                        "template_type": "lnested",
                        "template_depth": max(1, 1 + depth_bump),
                        "svm_mode": "lnested",
                        "status": "ERROR",
                        "ranking_function": None,
                        "ranking_functions": [],
                        "input_code_source": "unknown",
                    }
                    log = "[Error] Invalid svmranker_result entry (not a dict)."
                else:
                    res, log = run_on_unknown_entry(entry)
                results.append(res)
                full_log.append(f"--- Entry Loop ID: {res.get('loop_id')} ---\n{log}\n" + "=" * 40 + "\n")

            input_file = None
            source_file = None
            if isinstance(content, dict):
                input_file = content.get("input_file")
                source_file = content.get("source_file")
            result = {
                "input_file": input_file or str(f),
                "source_file": source_file or f.name,
                "task": "svmranker",
                "svmranker_result": results,
            }
            return result, "\n".join(full_log)

        def build_output_path(root: Path, f: Path, bucket: str) -> Path:
            try:
                rel_path = f.relative_to(root)
            except ValueError:
                rel_path = Path(f.name)
            parts = list(rel_path.parts)
            if "unknown" in parts:
                idx = parts.index("unknown")
                parts[idx] = bucket
                rel_path = Path(*parts)
            else:
                rel_path = Path(bucket) / rel_path
            return output / rel_path

        if input_path.is_file():
            root_dir = input_path.parent
            files = [input_path]
        else:
            unknown_dirs = find_unknown_dirs(input_path)
            if not unknown_dirs:
                raise typer.BadParameter("未找到 unknown 目录，请传入 unknown 目录或其父目录。")
            root_dir = input_path
            files = []
            for unknown_dir in unknown_dirs:
                files.extend(collect_files(unknown_dir, recursive, extensions={".yml", ".yaml"}))

        for f in files:
            try:
                result, full_log = process_unknown_yaml(f)
                if not result:
                    console.print(f"[bold red]ERROR: Failed to parse {f.name}[/bold red]")
                    continue
                results = result.get("svmranker_result") or []
                bucket = classify_output_dir(results)
                out_path = build_output_path(root_dir, f, bucket)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as yf:
                    yaml.dump(result, yf, sort_keys=False, allow_unicode=True)

                log_path = out_path.with_suffix(".svmranker.txt")
                log_path.write_text(full_log, encoding="utf-8")

                if bucket != "unknown" and f.exists():
                    try:
                        f.unlink()
                        old_log = f.with_suffix(".svmranker.txt")
                        if old_log.exists():
                            old_log.unlink()
                    except Exception as exc:
                        console.print(f"[bold red]ERROR: Failed to delete {f.name}: {exc}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]ERROR: Error processing {f.name}: {e}[/bold red]")
