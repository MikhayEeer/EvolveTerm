import sys
import os
import threading
import re
import tempfile
import subprocess
import io
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List

# Global lock to prevent race conditions on OneLoop.py
SVM_RANKER_LOCK = threading.Lock()

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
}

C_BUILTINS = {
    "printf", "scanf", "malloc", "free", "calloc", "realloc", "strlen",
    "memset", "memcpy", "memmove", "assert",
}

class SVMRankerClient:
    def __init__(self, tool_root: str | Path):
        self.tool_root = Path(tool_root).resolve()
        self.src_dir = self.tool_root / "src"
        
        # Add src directory to sys.path to allow imports
        if str(self.src_dir) not in sys.path:
            sys.path.append(str(self.src_dir))
            
        self._modules_loaded = False
        self._load_modules()

    def _load_modules(self):
        """Attempt to import SVMRanker modules."""
        try:
            # Delayed import to avoid errors if paths are not set up
            global parseBoogieProgramMulti, SVMLearnMulti
            # Note: These modules are expected to be in the tool_root/src directory
            from BoogieParser import parseBoogieProgramMulti
            from SVMLearn import SVMLearnMulti
            self._modules_loaded = True
        except ImportError as e:
            print(f"[Error] Failed to import SVMRanker modules: {e}")
            self._modules_loaded = False

    def _needs_main_wrapper(self, code: str) -> bool:
        print("[Info] Minimal wrapper disabled; using source code as-is.")
        return False

    def _collect_declared_identifiers(self, code: str) -> set[str]:
        print("[Info] Identifier collection disabled.")
        return set()

    def _collect_identifiers(self, code: str) -> set[str]:
        print("[Info] Identifier collection disabled.")
        return set()

    def _collect_function_calls(self, code: str) -> set[str]:
        print("[Info] Function stub generation disabled.")
        return set()

    def _build_minimal_c(self, code: str) -> str:
        # Remove #include directives to avoid macro expansion issues in C2Boogie
        lines = code.splitlines()
        filtered_lines = []
        for line in lines:
            if re.match(r'^\s*#\s*include', line):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)

    def run(self, file_path: Path, mode: str = "lnested", depth: int = 1) -> Tuple[str, Optional[str], List[str], str]:
        """
        Run SVMRanker on the given file.
        
        Args:
            file_path: Path to the file (C or Boogie).
            mode: "lnested" or "lmulti".
            depth: Depth bound.
            
        Returns:
            Tuple containing:
            - result_status: "TERMINATE", "NONTERM", "UNKNOWN", or "ERROR"
            - ranking_function: The first valid ranking function string (if any)
            - rf_list: List of all generated ranking functions
            - log: Collected stdout/stderr log
        """
        log_buffer = io.StringIO()
        if not self._modules_loaded:
            log_buffer.write("[Error] SVMRanker modules not loaded.\n")
            return "ERROR", None, [], log_buffer.getvalue()

        abs_file_path = file_path.resolve()
        temp_path: Optional[Path] = None
        temp_bpl_path: Optional[Path] = None
        if abs_file_path.suffix.lower() in {".c", ".h", ".cpp", ".cc", ".cxx"}:
            try:
                # Read original code
                raw_code = abs_file_path.read_text(encoding="utf-8")
                
                # Sanitize code (remove includes)
                sanitized_code = self._build_minimal_c(raw_code)
                
                # Write to a temporary file for processing
                temp_c = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8")
                temp_c.write(sanitized_code)
                temp_c.close()
                process_file_path = Path(temp_c.name)
                temp_path = process_file_path # Mark for cleanup

                # Debug log
                debug_path = Path.cwd() / "svmranker_last_input.c"
                debug_path.write_text(sanitized_code, encoding="utf-8")
                
            except OSError as e:
                msg = f"[Error] Failed to prepare input file: {e}"
                print(msg)
                log_buffer.write(msg + "\n")
                return "ERROR", None, [], log_buffer.getvalue()
        else:
            # For non-C files (e.g. .bpl), use as is
            process_file_path = abs_file_path
        
        # Configuration based on guide
        sample_strategy = "ENLARGE"
        cutting_strategy = "MINI"
        template_strategy = "SINGLEFULL"
        print_level = 0  # Suppress internal printing

        with SVM_RANKER_LOCK:
            try:
                # Switch cwd to SVMRanker/src because it generates temporary files there
                old_cwd = os.getcwd()
                os.chdir(self.src_dir)
                
                try:
                    if abs_file_path.suffix.lower() in {".c", ".h", ".cpp", ".cc", ".cxx"}:
                        temp_bpl = tempfile.NamedTemporaryFile(mode="w", suffix=".bpl", delete=False, encoding="utf-8")
                        temp_bpl_path = Path(temp_bpl.name)
                        temp_bpl.close()
                        cli_main = self.src_dir / "CLIMain.py"
                        # Use process_file_path (sanitized temp file) instead of abs_file_path
                        cmd = [sys.executable, str(cli_main), "parsectoboogie", str(process_file_path), str(temp_bpl_path)]
                        
                        log_buffer.write(f"Executing: {' '.join(cmd)}\n")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        log_buffer.write(f"Return Code: {result.returncode}\n")
                        if result.stdout:
                            log_buffer.write(f"STDOUT:\n{result.stdout}\n")
                        if result.stderr:
                            log_buffer.write(f"STDERR:\n{result.stderr}\n")

                        try:
                            debug_log = Path.cwd() / "svmranker_last_log.txt"
                            log_body = (
                                "command: " + " ".join(cmd) + "\n"
                                + "returncode: " + str(result.returncode) + "\n"
                                + "stdout:\n" + (result.stdout or "") + "\n"
                                + "stderr:\n" + (result.stderr or "") + "\n"
                            )
                            debug_log.write_text(log_body, encoding="utf-8")
                        except OSError:
                            pass
                        if result.returncode != 0:
                            msg = (result.stderr or result.stdout or "parseCtoBoogie failed").strip()
                            print(f"[Error] parseCtoBoogie failed: {msg}")
                            return "ERROR", None, [], log_buffer.getvalue()
                        stderr_text = result.stderr or ""
                        if "Traceback" in stderr_text or "Exception" in stderr_text:
                            msg = "parseCtoBoogie reported a crash in stderr"
                            print(f"[Error] {msg}")
                            log_buffer.write(f"[Error] {msg}\n")
                            return "ERROR", None, [], log_buffer.getvalue()
                        if not temp_bpl_path or not temp_bpl_path.exists():
                            msg = "parseCtoBoogie did not produce a .bpl file"
                            print(f"[Error] {msg}")
                            log_buffer.write(f"[Error] {msg}\n")
                            return "ERROR", None, [], log_buffer.getvalue()
                        try:
                            if temp_bpl_path.stat().st_size == 0:
                                msg = "parseCtoBoogie produced an empty .bpl file"
                                print(f"[Error] {msg}")
                                log_buffer.write(f"[Error] {msg}\n")
                                return "ERROR", None, [], log_buffer.getvalue()
                        except OSError as e:
                            msg = f"Failed to stat .bpl file: {e}"
                            print(f"[Error] {msg}")
                            log_buffer.write(f"[Error] {msg}\n")
                            return "ERROR", None, [], log_buffer.getvalue()
                        try:
                            debug_bpl = Path.cwd() / "svmranker_last_input.bpl"
                            debug_bpl.write_text(temp_bpl_path.read_text(encoding="utf-8"), encoding="utf-8")
                        except OSError:
                            pass
                        abs_file_path = temp_bpl_path

                    # Step A: Parse
                    # The guide says parseBoogieProgramMulti takes a .bpl file.
                    # If the user says C is supported, we assume this parser handles it 
                    # or we are passing a .bpl file converted elsewhere.
                    # For now, we pass the file path as is.
                    # Warning: parseBoogieProgramMulti expects the boogie file path if we just converted it?
                    # The original code passed abs_file_path, but if we did conversion, we should pass temp_bpl_path?
                    # However, SVMLearnMulti takes 'sourceFilePath'. 
                    # Based on flow: parseCtoBoogie -> output.bpl. 
                    # Then we likely parse the output.bpl.
                    target_bpl = temp_bpl_path if temp_bpl_path else process_file_path
                    
                    log_buffer.write(f"\n--- calling parseBoogieProgramMulti on {target_bpl} ---\n")
                    
                    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                        (sourceFilePath, sourceFileName, 
                        templatePath, templateFileName, 
                        Info, 
                        parse_oldtime, parse_newtime) = parseBoogieProgramMulti(str(target_bpl), "OneLoop.py")

                        # Step B: Learn
                        log_buffer.write(f"\n--- calling SVMLearnMulti ---\n")
                        result, rf_list = SVMLearnMulti(
                            sourceFilePath, 
                            sourceFileName, 
                            depth,
                            parse_oldtime, 
                            parse_newtime, 
                            sample_strategy, 
                            cutting_strategy, 
                            template_strategy, 
                            print_level
                        )
                    
                    first_rf = rf_list[0] if rf_list else None
                    return result, first_rf, rf_list, log_buffer.getvalue()

                finally:
                    # Restore cwd
                    os.chdir(old_cwd)
                    if temp_path:
                        temp_path.unlink(missing_ok=True)
                    if temp_bpl_path:
                        temp_bpl_path.unlink(missing_ok=True)

            except Exception as e:
                msg = f"[Error] SVMRanker execution exception: {e}"
                print(msg)
                log_buffer.write(f"\n{msg}\n")
                return "ERROR", None, [], log_buffer.getvalue()
