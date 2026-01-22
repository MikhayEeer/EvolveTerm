"""Handler for the 'svmranker' command."""
from __future__ import annotations
import json
import yaml
import tempfile
import multiprocessing
import os
import queue
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
from rich.console import Console
import typer

from ..svm_ranker import SVMRankerClient, run_svmranker_worker
from ..cli_utils import collect_files, validate_yaml_required_keys
from ..utils import LiteralDumper

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[3]
BOOGIE_ROOT = Path("/home/clexma/Desktop/fox3/TermDB/TerminationDatabase/Data_boogie")
DEFAULT_TIMEOUT_SEC = 180

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
    def __init__(self, svm_ranker_path: Path, timeout_sec: int = DEFAULT_TIMEOUT_SEC):
        self.root = resolve_svm_ranker_root(svm_ranker_path)
        self.client = SVMRankerClient(str(self.root))
        self.timeout_sec = timeout_sec

    @staticmethod
    def _normalize_template_mode(template_type: Any) -> str:
        value = str(template_type or "").strip().lower()
        if "piecewise" in value:
            return "lpiecewiseext"
        if "multiext" in value:
            return "lmultiext"
        if "lexiext" in value or "lexi" in value:
            return "llexiext"
        if value in {"lmulti", "multi"} or "multi" in value:
            return "lmultiext"
        if value in {"lnested", "nested"} or "nested" in value:
            return "llexiext"
        return "llexiext"

    @staticmethod
    def _normalize_template_predicates(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        text = str(value).strip()
        return [text] if text else []

    def _run_isolated_svmranker(
        self,
        target_path: Path,
        mode: str,
        depth: int,
        predicates: Optional[List[str]],
        enhance_on_unknown: bool,
        timeout_sec: int,
    ) -> Tuple[str, Optional[str], List[str], Optional[str], str]:
        ctx = multiprocessing.get_context("spawn")
        result_queue: Any = ctx.Queue()
        proc = ctx.Process(
            target=run_svmranker_worker,
            args=(
                result_queue,
                str(self.root),
                str(target_path),
                mode,
                depth,
                predicates,
                enhance_on_unknown,
                timeout_sec,
            ),
        )
        proc.daemon = True
        proc.start()
        proc.join(timeout_sec)

        def _terminate_process() -> None:
            if not proc.is_alive():
                return
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                try:
                    if hasattr(proc, "kill"):
                        proc.kill()
                    else:
                        os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.join(5)

        if proc.is_alive():
            _terminate_process()
            log = f"[Error] isolated SVMRanker timeout after {timeout_sec}s (pid={proc.pid}).\n"
            return "ERROR", None, [], log

        try:
            payload = result_queue.get(timeout=1)
        except queue.Empty:
            log = f"[Error] isolated SVMRanker produced no result (exitcode={proc.exitcode}).\n"
            return "ERROR", None, [], log
        finally:
            try:
                result_queue.close()
            except Exception:
                pass

        status = payload.get("status") if isinstance(payload, dict) else "ERROR"
        rf = payload.get("ranking_function") if isinstance(payload, dict) else None
        rf_list = payload.get("ranking_functions") if isinstance(payload, dict) else []
        piecewise_rf = payload.get("piecewise_rf") if isinstance(payload, dict) else None
        log = payload.get("log") if isinstance(payload, dict) else ""
        if proc.exitcode not in (0, None):
            log = f"[Debug] isolated SVMRanker exitcode={proc.exitcode}.\n" + (log or "")
        return status, rf, rf_list, piecewise_rf, log

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
        predicates: Optional[List[str]] = None,
    ) -> str:
        lines = [
            f"[Entry] loop_id={loop_id}",
            f"[Input] source={code_source} code_path={code_source_path or '-'} boogie_path={boogie_path or '-'}",
            f"[Template] type={template_type} depth={template_depth} svm_mode={mode} depth_bound={depth_val}",
        ]
        if depth_original is not None and depth_bump is not None:
            lines.append(f"[Template] depth_original={depth_original} depth_bump={depth_bump}")
        if predicates:
            lines.append(f"[Template] predicates={len(predicates)}")
        return "\n".join(lines) + "\n"

    def run(
        self,
        input_path: Path,
        output: Path,
        recursive: bool,
        skip_certain: bool = False,
        skip_exist: bool = False,
    ):
        if output is None:
            raise typer.BadParameter("Output directory is required.")
        if output.exists() and not output.is_dir():
            raise typer.BadParameter("Output must be a directory.")
        output.mkdir(parents=True, exist_ok=True)

        input_file_cache: Dict[str, Optional[str]] = {}

        def normalize_source_path_value(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            try:
                return Path(text).as_posix()
            except Exception:
                return text

        def extract_source_path(content: Any) -> Optional[str]:
            if not isinstance(content, dict):
                return None
            source_path = content.get("source_path")
            if isinstance(source_path, str):
                return normalize_source_path_value(source_path)
            return None

        def resolve_source_path_from_input_file(input_file: Optional[str]) -> Optional[str]:
            if not input_file:
                return None
            cached = input_file_cache.get(input_file)
            if cached is not None or input_file in input_file_cache:
                return cached
            candidates = [Path(input_file)]
            if not candidates[0].is_absolute():
                candidates.append(Path.cwd() / input_file)
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    content = yaml.safe_load(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(content, dict):
                    source_path = extract_source_path(content)
                    if source_path:
                        input_file_cache[input_file] = source_path
                        return source_path
                break
            input_file_cache[input_file] = None
            return None

        def load_existing_source_paths(dirs: List[Path]) -> set[str]:
            names: set[str] = set()
            for root in dirs:
                if not root.exists():
                    continue
                for path in list(root.rglob("*.yml")) + list(root.rglob("*.yaml")):
                    try:
                        content = yaml.safe_load(path.read_text(encoding="utf-8"))
                    except Exception:
                        content = None
                    if isinstance(content, dict):
                        source_path = extract_source_path(content)
                        if source_path:
                            names.add(source_path)
                        input_file = content.get("input_file")
                        resolved = resolve_source_path_from_input_file(str(input_file)) if input_file else None
                        if resolved:
                            names.add(resolved)
                        entries = content.get("svmranker_result") or []
                        if isinstance(entries, list):
                            for entry in entries:
                                if not isinstance(entry, dict):
                                    continue
                                entry_source_path = entry.get("input_source_path")
                                entry_key = normalize_source_path_value(entry_source_path)
                                if entry_key:
                                    names.add(entry_key)
            return names
        
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
            top_source_path: Optional[str],
        ) -> Tuple[Dict[str, Any], str]:
            loop_id = entry.get("loop_id") or entry.get("id")
            code, code_source, code_source_path, boogie_path = resolve_source_code(
                entry, base_dir, fallback_source_path
            )
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            template_predicates = self._normalize_template_predicates(entry.get("template_predicates"))
            mode = self._normalize_template_mode(template_type)
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1

            if boogie_path:
                status, rf, rf_list, piecewise_rf, log = self.client.run(
                    Path(boogie_path),
                    mode=mode,
                    depth=depth_val,
                    predicates=template_predicates,
                    timeout_sec=self.timeout_sec,
                )
            else:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name

                try:
                    status, rf, rf_list, piecewise_rf, log = self.client.run(
                        Path(tmp_path),
                        mode=mode,
                        depth=depth_val,
                        predicates=template_predicates,
                        timeout_sec=self.timeout_sec,
                    )
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
                predicates=template_predicates,
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
            input_source_path = entry.get("source_path") or top_source_path
            if isinstance(input_source_path, str) and input_source_path.strip():
                result["input_source_path"] = input_source_path
                try:
                    result["input_source_stem"] = Path(input_source_path).stem
                except Exception:
                    pass
            if template_predicates:
                result["template_predicates"] = template_predicates
            if piecewise_rf:
                result["piecewise_rf"] = piecewise_rf
            if code_source_path:
                result["input_code_path"] = code_source_path
            if boogie_path:
                result["input_boogie_path"] = boogie_path

            return result, log_prefix + log

        existing_certain = (
            load_existing_source_paths([output / "certain"]) if skip_certain else set()
        )
        existing_any = (
            load_existing_source_paths([output / "failed", output / "certain", output / "unknown"])
            if skip_exist
            else set()
        )

        if skip_exist:
            existing_set = existing_any
            skip_label = "existing output"
        elif skip_certain:
            existing_set = existing_certain
            skip_label = "existing certain"
        else:
            existing_set = set()
            skip_label = "existing"

        def should_skip_existing(content: Any) -> bool:
            if not (skip_certain or skip_exist):
                return False
            source_path = extract_source_path(content)
            if not source_path:
                return False
            return source_path in existing_set

        def summarize_attempts(log_text: str) -> Tuple[List[str], bool, Optional[str]]:
            commands: List[str] = []
            result_value: Optional[str] = None
            for line in log_text.splitlines():
                if line.startswith("[Command] "):
                    commands.append(line[len("[Command] "):].strip())
                if "LEARNING RESULT:" in line:
                    match = re.search(r"LEARNING RESULT:\\s*(TERMINATE|UNKNOWN|NONTERM)", line)
                    if match:
                        result_value = match.group(1)
            enhanced = "[Attempt] enhanced" in log_text
            return commands, enhanced, result_value

        def process_yaml(f: Path, strict: bool) -> Tuple[Optional[Dict[str, Any]], str, bool, List[Dict[str, Any]]]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                return {"source_file": f.name, "error": f"YAML parse error: {e}"}, f"YAML Error: {e}", False, []
            if not _check_yaml_required_keys(f, content, strict):
                return None, "YAML validation failed", False, []

            if should_skip_existing(content):
                return {"source_file": f.name, "skipped": True}, f"Skipped ({skip_label})", True, []

            entries = parse_results(content)
            base_dir = f.parent
            fallback_source_path = None
            top_source_path = None
            if isinstance(content, dict):
                top_source_path = content.get("source_path") if isinstance(content.get("source_path"), str) else None
            if isinstance(content, dict) and len(entries) == 1:
                top_source_path = content.get("source_path")
                if isinstance(top_source_path, str) and top_source_path.strip():
                    fallback_source_path = top_source_path

            if not _check_template_yaml(f, entries, strict):
                return None, "Non-template YAML", False, []
            
            results = []
            full_log = []
            entry_summaries: List[Dict[str, Any]] = []
            
            for entry in entries:
                res, log = run_on_entry(entry, base_dir, fallback_source_path, top_source_path)
                results.append(res)
                full_log.append(f"--- Entry Loop ID: {res.get('loop_id')} ---\n{log}\n" + "="*40 + "\n")
                commands, enhanced, parsed_result = summarize_attempts(log)
                entry_summaries.append(
                    {
                        "status": res.get("status"),
                        "commands": commands,
                        "enhanced": enhanced,
                        "parsed_result": parsed_result,
                    }
                )
                
            return {
                "input_file": str(f),
                "source_file": f.name,
                "task": "svmranker",
                "svmranker_result": results,
            }, "\n".join(full_log), False, entry_summaries

        if input_path.is_file():
            console.print(f"[blue]==> Processing {input_path.name}[/blue]")
            result, full_log, skipped, summaries = process_yaml(input_path, True)
            if skipped:
                console.print(f"[yellow]Skip {skip_label}: {input_path.name}[/yellow]")
                return
            if result is None:
                raise typer.Exit(code=1)
            for idx, summary in enumerate(summaries, start=1):
                console.print(f"  [cyan]Entry {idx}[/cyan]")
                for cmd in summary.get("commands") or []:
                    console.print(f"    [dim]Command:[/dim] {cmd}")
                if summary.get("enhanced"):
                    console.print("    [yellow]Retry:[/yellow] UNKNOWN -> enhanced")
                status = summary.get("status") or summary.get("parsed_result") or "UNKNOWN"
                console.print(f"    [green]Result:[/green] {status}")
            
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
            console.print(f"[blue]<== Done {input_path.name}[/blue]")

        elif input_path.is_dir():
            files = collect_files(input_path, recursive, extensions={".yml", ".yaml"})
            total_files = len(files)
            if skip_certain or skip_exist:
                skip_count = 0
                missing_source = 0
                parse_errors = 0
                for f in files:
                    try:
                        content = yaml.safe_load(f.read_text(encoding="utf-8"))
                    except Exception:
                        parse_errors += 1
                        continue
                    source_path = extract_source_path(content)
                    if not source_path:
                        missing_source += 1
                        continue
                    if source_path in existing_set:
                        skip_count += 1
                to_run = total_files - skip_count
                console.print(
                    f"[blue]Skip-{ 'exist' if skip_exist else 'certain' } precheck: "
                    f"total={total_files} skip={skip_count} run={to_run}[/blue]"
                )
                if missing_source:
                    console.print(
                        f"[yellow]Skip-{ 'exist' if skip_exist else 'certain' } precheck: "
                        f"{missing_source} files missing source_path (will not skip)[/yellow]"
                    )
                if parse_errors:
                    console.print(
                        f"[yellow]Skip-{ 'exist' if skip_exist else 'certain' } precheck: "
                        f"{parse_errors} files failed to parse (will not skip)[/yellow]"
                    )

            for idx, f in enumerate(files, start=1):
                try:
                    console.print(f"[blue]==> Processing {f.name} ({idx}/{total_files})[/blue]")
                    result, full_log, skipped, summaries = process_yaml(f, False)
                    if skipped:
                        console.print(f"[yellow]Skip {skip_label}: {f.name}[/yellow]")
                        continue
                    if not result: continue
                    for entry_idx, summary in enumerate(summaries, start=1):
                        console.print(f"  [cyan]Entry {entry_idx}[/cyan]")
                        for cmd in summary.get("commands") or []:
                            console.print(f"    [dim]Command:[/dim] {cmd}")
                        if summary.get("enhanced"):
                            console.print("    [yellow]Retry:[/yellow] UNKNOWN -> enhanced")
                        status = summary.get("status") or summary.get("parsed_result") or "UNKNOWN"
                        console.print(f"    [green]Result:[/green] {status}")
                    
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
                    console.print(f"[blue]<== Done {f.name}[/blue]")
                        
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
            template_predicates = self._normalize_template_predicates(entry.get("template_predicates"))
            mode = self._normalize_template_mode(template_type)
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
                status, rf, rf_list, piecewise_rf, log = self.client.run(
                    target_path,
                    mode=mode,
                    depth=depth_val,
                    predicates=template_predicates,
                    timeout_sec=self.timeout_sec,
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
                predicates=template_predicates,
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
            input_source_path = entry.get("input_source_path")
            if isinstance(input_source_path, str) and input_source_path.strip():
                result["input_source_path"] = input_source_path
                try:
                    result["input_source_stem"] = Path(input_source_path).stem
                except Exception:
                    pass
            input_source_stem = entry.get("input_source_stem")
            if input_source_stem and "input_source_stem" not in result:
                result["input_source_stem"] = input_source_stem
            if template_predicates:
                result["template_predicates"] = template_predicates
            if piecewise_rf:
                result["piecewise_rf"] = piecewise_rf
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
            template_predicates = self._normalize_template_predicates(entry.get("template_predicates"))
            mode = self._normalize_template_mode(template_type)
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
                status, rf, rf_list, piecewise_rf, log = self._run_isolated_svmranker(
                    target_path=target_path,
                    mode=mode,
                    depth=depth_val,
                    predicates=template_predicates,
                    enhance_on_unknown=True,
                    timeout_sec=self.timeout_sec,
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
                predicates=template_predicates,
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
            input_source_path = entry.get("input_source_path")
            if isinstance(input_source_path, str) and input_source_path.strip():
                result["input_source_path"] = input_source_path
                try:
                    result["input_source_stem"] = Path(input_source_path).stem
                except Exception:
                    pass
            input_source_stem = entry.get("input_source_stem")
            if input_source_stem and "input_source_stem" not in result:
                result["input_source_stem"] = input_source_stem
            if template_predicates:
                result["template_predicates"] = template_predicates
            if piecewise_rf:
                result["piecewise_rf"] = piecewise_rf
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
