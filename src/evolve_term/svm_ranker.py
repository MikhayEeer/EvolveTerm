import sys
import os
import threading
import re
import tempfile
import subprocess
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
        return re.search(r"\bmain\s*\(", code) is None

    def _collect_declared_identifiers(self, code: str) -> set[str]:
        type_tokens = {
            "char", "short", "int", "long", "float", "double", "signed", "unsigned",
            "size_t", "bool", "_Bool", "struct", "enum", "union",
        }
        declared = set()
        for line in code.splitlines():
            if ";" not in line:
                continue
            if not any(re.search(rf"\b{re.escape(tok)}\b", line) for tok in type_tokens):
                continue
            for name in re.findall(r"\b[_a-zA-Z]\w*\b", line):
                if name in C_KEYWORDS or name in type_tokens:
                    continue
                declared.add(name)
        return declared

    def _collect_identifiers(self, code: str) -> set[str]:
        identifiers = set()
        for match in re.finditer(r"\b[_a-zA-Z]\w*\b", code):
            name = match.group(0)
            if name in C_KEYWORDS or name in C_BUILTINS or name == "main":
                continue
            i = match.start() - 1
            while i >= 0 and code[i].isspace():
                i -= 1
            if i >= 0 and code[i] in {".", ">"}:
                continue
            j = match.end()
            while j < len(code) and code[j].isspace():
                j += 1
            if j < len(code) and code[j] == "(":
                continue
            identifiers.add(name)
        for match in re.finditer(r"\b([_a-zA-Z]\w*)\s*=", code):
            name = match.group(1)
            if name in C_KEYWORDS or name in C_BUILTINS or name == "main":
                continue
            identifiers.add(name)
        return identifiers

    def _collect_function_calls(self, code: str) -> set[str]:
        calls = set()
        for match in re.finditer(r"\b([_a-zA-Z]\w*)\s*\(", code):
            name = match.group(1)
            if name in C_KEYWORDS or name in C_BUILTINS or name == "main":
                continue
            calls.add(name)
        return calls

    def _build_minimal_c(self, code: str) -> str:
        declared = self._collect_declared_identifiers(code)
        identifiers = self._collect_identifiers(code)
        array_names = set(re.findall(r"\b([_a-zA-Z]\w*)\s*\[", code))
        func_calls = self._collect_function_calls(code)
        unsigned_char_vars = set(
            re.findall(r"\b([_a-zA-Z]\w*)\s*=\s*unknown_uchar\s*\(", code)
        )
        assigned_vars = set(
            re.findall(
                r"\b([_a-zA-Z]\w*)\s*(?:=|\+=|-=|\*=|/=|%=|<<=|>>=|&=|\^=|\|=)",
                code,
            )
        )
        candidates = sorted((identifiers | assigned_vars) - declared)
        decl_lines = []
        for name in candidates:
            if name in array_names:
                decl_lines.append(f"    int {name}[10];")
            elif name in unsigned_char_vars:
                decl_lines.append(f"    unsigned char {name} = 0;")
            else:
                decl_lines.append(f"    int {name} = 0;")
        decl_block = ""
        if decl_lines:
            decl_block = "    /* auto-declared variables */\n" + "\n".join(decl_lines) + "\n"
        proto_lines = []
        for name in sorted(func_calls):
            if name.startswith("unknown_"):
                proto_lines.append(f"extern unsigned char {name}(void);")
            else:
                proto_lines.append(f"int {name}(void);")
        proto_block = ""
        if proto_lines:
            proto_block = "/* auto-declared function stubs */\n" + "\n".join(proto_lines) + "\n\n"
        body = code.rstrip()
        return proto_block + "int main(void) {\n" + decl_block + body + "\n    return 0;\n}\n"

    def run(self, file_path: Path, mode: str = "lnested", depth: int = 1) -> Tuple[str, Optional[str], List[str]]:
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
        """
        if not self._modules_loaded:
            return "ERROR", None, []

        abs_file_path = file_path.resolve()
        temp_path: Optional[Path] = None
        temp_bpl_path: Optional[Path] = None
        if abs_file_path.suffix.lower() in {".c", ".h", ".cpp", ".cc", ".cxx"}:
            try:
                code = abs_file_path.read_text(encoding="utf-8")
            except OSError:
                code = ""
            if code and self._needs_main_wrapper(code):
                wrapped = self._build_minimal_c(code)
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8")
                tmp.write(wrapped)
                tmp.flush()
                temp_path = Path(tmp.name)
                tmp.close()
                abs_file_path = temp_path
                try:
                    debug_path = Path.cwd() / "svmranker_last_input.c"
                    debug_path.write_text(wrapped, encoding="utf-8")
                except OSError:
                    pass
        
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
                        cmd = [sys.executable, str(cli_main), "parsectoboogie", str(abs_file_path), str(temp_bpl_path)]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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
                            return "ERROR", None, []
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
                    (sourceFilePath, sourceFileName, 
                     templatePath, templateFileName, 
                     Info, 
                     parse_oldtime, parse_newtime) = parseBoogieProgramMulti(str(abs_file_path), "OneLoop.py")

                    # Step B: Learn
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
                    return result, first_rf, rf_list

                finally:
                    # Restore cwd
                    os.chdir(old_cwd)
                    if temp_path:
                        temp_path.unlink(missing_ok=True)
                    if temp_bpl_path:
                        temp_bpl_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"[Error] SVMRanker execution exception: {e}")
                return "ERROR", None, []
