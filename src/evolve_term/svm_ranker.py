import sys
import os
import threading
import re
import tempfile
import subprocess
import io
from pathlib import Path
from typing import Optional, Tuple, List, Any

# Global lock to prevent race conditions in SVMRanker temp files
SVM_RANKER_LOCK = threading.Lock()

C_EXTENSIONS = {".c", ".h", ".cpp", ".cc", ".cxx"}

DEFAULT_PRINT_LEVEL = "DEBUG"

class SVMRankerClient:
    def __init__(self, tool_root: str | Path):
        self.tool_root = Path(tool_root).resolve()
        self.src_dir = self.tool_root / "src"
        self.cli_main = self.src_dir / "CLIMain.py"
        self._cli_available = self.cli_main.exists()
        if not self._cli_available:
            print(f"[Error] SVMRanker CLIMain.py not found at {self.cli_main}")

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        value = str(mode or "").strip().lower()
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
    def _normalize_predicates(predicates: Optional[List[str] | str]) -> List[str]:
        if predicates is None:
            return []
        if isinstance(predicates, str):
            predicates = [predicates]
        if not predicates:
            return []
        cleaned: List[str] = []
        for pred in predicates:
            if pred is None:
                continue
            text = str(pred).strip()
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _extract_learning_result(output: str) -> Optional[str]:
        match = re.search(r"LEARNING RESULT:\s*(TERMINATE|UNKNOWN|NONTERM)", output)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_ranking_functions(output: str) -> List[str]:
        patterns = [
            re.compile(r"Ranking Function\s*[:=]\s*(.+)", re.IGNORECASE),
            re.compile(r"Ranking function\s*[:=]\s*(.+)", re.IGNORECASE),
            re.compile(r"RF\s*[:=]\s*(.+)", re.IGNORECASE),
        ]
        results: List[str] = []
        seen = set()
        for line in output.splitlines():
            for pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                value = match.group(1).strip()
                if not value:
                    continue
                if value not in seen:
                    seen.add(value)
                    results.append(value)
        return results

    def _build_minimal_c(self, code: str) -> str:
        # Remove #include directives to avoid macro expansion issues
        lines = code.splitlines()
        filtered_lines = []
        for line in lines:
            if re.match(r"^\s*#\s*include", line):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)

    def _prepare_input(self, file_path: Path, log_buffer: io.StringIO) -> Tuple[Path, Optional[Path], Optional[str]]:
        abs_file_path = file_path.resolve()
        if abs_file_path.suffix.lower() not in C_EXTENSIONS:
            return abs_file_path, None, None

        raw_code = abs_file_path.read_text(encoding="utf-8")
        sanitized_code = self._build_minimal_c(raw_code)
        temp_c = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8")
        temp_c.write(sanitized_code)
        temp_c.close()
        temp_path = Path(temp_c.name)

        debug_path = Path.cwd() / "svmranker_last_input.c"
        try:
            debug_path.write_text(sanitized_code, encoding="utf-8")
        except OSError:
            pass
        log_buffer.write(f"[Input] kind=c temp_c={temp_path}\n")
        return temp_path, temp_path, "C"

    def _config_for_mode(self, mode: str, depth_bound: int) -> dict[str, Any]:
        if mode == "llexiext":
            return {
                "depth_bound": depth_bound,
                "template_strategy": "LINEAR",
                "sample_strategy": "ENLARGE",
            }
        if mode == "lmultiext":
            return {
                "depth_bound": depth_bound,
                "template_strategy": "LINEAR",
                "sample_strategy": "ENLARGE",
                "cutting_strategy": "MINI",
            }
        if mode == "lpiecewiseext":
            return {
                "template_strategy": "LINEAR",
                "max_iters": 60,
            }
        return {
            "depth_bound": depth_bound,
            "template_strategy": "LINEAR",
            "sample_strategy": "ENLARGE",
        }

    def _enhanced_config_for_mode(self, mode: str, depth_bound: int) -> dict[str, Any]:
        if mode == "llexiext":
            return {
                "depth_bound": max(depth_bound, 4),
                "template_strategy": "QUAD",
                "sample_strategy": "ENLARGE",
            }
        if mode == "lmultiext":
            return {
                "depth_bound": max(depth_bound, 2),
                "template_strategy": "QUAD",
                "sample_strategy": "ENLARGE",
                "cutting_strategy": "MINI",
            }
        if mode == "lpiecewiseext":
            return {
                "template_strategy": "QUAD",
                "max_iters": 100,
            }
        return {
            "depth_bound": max(depth_bound, 2),
            "template_strategy": "QUAD",
            "sample_strategy": "ENLARGE",
        }

    def _run_cli(
        self,
        mode: str,
        file_path: Path,
        config: dict[str, Any],
        predicates: List[str],
        filetype: Optional[str],
    ) -> Tuple[int, str, Optional[str]]:
        cmd = [sys.executable, str(self.cli_main), mode]

        if mode in {"llexiext", "lmultiext"}:
            cmd.extend(["--depth_bound", str(config.get("depth_bound", 1))])
            cmd.extend(["--template_strategy", str(config.get("template_strategy", "LINEAR"))])
            cmd.extend(["--sample_strategy", str(config.get("sample_strategy", "ENLARGE"))])
            if mode == "lmultiext":
                cmd.extend(["--cutting_strategy", str(config.get("cutting_strategy", "MINI"))])
        elif mode == "lpiecewiseext":
            cmd.extend(["--template_strategy", str(config.get("template_strategy", "LINEAR"))])
            cmd.extend(["--max_iters", str(config.get("max_iters", 60))])
            for pred in predicates:
                cmd.extend(["--pred", pred])

        cmd.extend(["--print_level", DEFAULT_PRINT_LEVEL])
        if filetype:
            cmd.extend(["--filetype", filetype])

        save_rf_path = None
        if mode == "lpiecewiseext":
            tmp_rf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
            tmp_rf.close()
            save_rf_path = Path(tmp_rf.name)
            cmd.extend(["--print_rf", "--save_rf", str(save_rf_path)])

        cmd.append(str(file_path))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr

        piecewise_rf = None
        if save_rf_path:
            try:
                if save_rf_path.exists():
                    content = save_rf_path.read_text(encoding="utf-8").strip()
                    if content:
                        piecewise_rf = content
            finally:
                save_rf_path.unlink(missing_ok=True)

        return result.returncode, output, piecewise_rf

    def run(
        self,
        file_path: Path,
        mode: str = "lnested",
        depth: int = 1,
        predicates: Optional[List[str]] = None,
        enhance_on_unknown: bool = True,
    ) -> Tuple[str, Optional[str], List[str], Optional[str], str]:
        """
        Run SVMRanker on the given file.

        Returns:
            (status, ranking_function, ranking_functions, piecewise_rf, log)
        """
        log_buffer = io.StringIO()
        mode = self._normalize_mode(mode)
        depth_bound = max(1, int(depth)) if isinstance(depth, int) or str(depth).isdigit() else 1
        predicates = self._normalize_predicates(predicates)

        log_buffer.write(f"[Config] mode={mode} depth_bound={depth_bound} print_level={DEFAULT_PRINT_LEVEL}\n")
        log_buffer.write(f"[Input] path={file_path.resolve()}\n")
        if predicates:
            log_buffer.write(f"[Config] predicates={len(predicates)}\n")

        if not self._cli_available:
            log_buffer.write("[Error] SVMRanker CLIMain.py not available.\n")
            return "ERROR", None, [], None, log_buffer.getvalue()

        temp_path = None
        cleanup_path = None
        filetype = None
        try:
            process_file_path, cleanup_path, filetype = self._prepare_input(file_path, log_buffer)
            temp_path = cleanup_path

            with SVM_RANKER_LOCK:
                old_cwd = os.getcwd()
                os.chdir(self.src_dir)
                try:
                    default_config = self._config_for_mode(mode, depth_bound)
                    status, rf_list, piecewise_rf, log_text = self._execute_attempt(
                        process_file_path,
                        mode,
                        default_config,
                        predicates,
                        filetype,
                        attempt_label="default",
                    )
                    default_rf_list = list(rf_list)
                    default_piecewise_rf = piecewise_rf
                    log_buffer.write(log_text)

                    if status == "UNKNOWN" and enhance_on_unknown:
                        enhanced_config = self._enhanced_config_for_mode(mode, depth_bound)
                        status, rf_list, piecewise_rf, log_text = self._execute_attempt(
                            process_file_path,
                            mode,
                            enhanced_config,
                            predicates,
                            filetype,
                            attempt_label="enhanced",
                        )
                        if not rf_list:
                            rf_list = default_rf_list
                        if not piecewise_rf:
                            piecewise_rf = default_piecewise_rf
                        log_buffer.write(log_text)

                    first_rf = rf_list[0] if rf_list else None
                    return status, first_rf, rf_list, piecewise_rf, log_buffer.getvalue()
                finally:
                    os.chdir(old_cwd)
        except Exception as exc:
            msg = f"[Error] SVMRanker execution exception: {exc}"
            log_buffer.write(msg + "\n")
            return "ERROR", None, [], None, log_buffer.getvalue()
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)

    def _execute_attempt(
        self,
        process_file_path: Path,
        mode: str,
        config: dict[str, Any],
        predicates: List[str],
        filetype: Optional[str],
        attempt_label: str,
    ) -> Tuple[str, List[str], Optional[str], str]:
        log_buffer = io.StringIO()
        log_buffer.write(f"\n[Attempt] {attempt_label}\n")
        log_buffer.write(f"[Config] {config}\n")
        log_buffer.write(f"[Command] {self.cli_main} {mode} {process_file_path}\n")

        returncode, output, piecewise_rf = self._run_cli(mode, process_file_path, config, predicates, filetype)
        log_buffer.write(f"[ReturnCode] {returncode}\n")
        if output:
            log_buffer.write(output)
            if not output.endswith("\n"):
                log_buffer.write("\n")

        status = self._extract_learning_result(output)
        if returncode != 0 and status is None:
            status = "ERROR"
        if status is None:
            status = "UNKNOWN"

        rf_list = self._extract_ranking_functions(output)
        return status, rf_list, piecewise_rf, log_buffer.getvalue()


def run_svmranker_worker(
    result_queue: "Any",
    svm_ranker_root: str,
    file_path: str,
    mode: str,
    depth: int,
    predicates: Optional[List[str]],
    enhance_on_unknown: bool,
) -> None:
    try:
        client = SVMRankerClient(svm_ranker_root)
        status, rf, rf_list, piecewise_rf, log = client.run(
            Path(file_path),
            mode=mode,
            depth=depth,
            predicates=predicates,
            enhance_on_unknown=enhance_on_unknown,
        )
        result_queue.put(
            {
                "status": status,
                "ranking_function": rf,
                "ranking_functions": rf_list,
                "piecewise_rf": piecewise_rf,
                "log": log,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "status": "ERROR",
                "ranking_function": None,
                "ranking_functions": [],
                "piecewise_rf": None,
                "log": f"[Error] isolated SVMRanker run exception: {exc}",
            }
        )
