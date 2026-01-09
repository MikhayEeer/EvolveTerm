from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_SEAHORN_IMAGE = "seahorn/seahorn-llvm14:nightly"

_INVALID_EXPR_TOKENS = (
    "\\forall",
    "\\exists",
    "\\valid",
    "\\at",
    "\\sum",
    "\\old",
    "==>",
    "=>",
)


@dataclass(frozen=True)
class LoopHeader:
    loop_id: int
    line_start: int
    header_end: int
    indent: str


class SeaHornVerifier:
    def __init__(self, docker_image: str = DEFAULT_SEAHORN_IMAGE, timeout_seconds: int = 60):
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.last_report: Optional[Dict[str, object]] = None
        self.last_run: Optional[Dict[str, object]] = None

    def verify(
        self,
        code: str,
        loop_invariants: Optional[Dict[int, List[str]]] = None,
        loop_rankings: Optional[Dict[int, str]] = None,
    ) -> str:
        instrumented_code, report = self._instrument_code(code, loop_invariants, loop_rankings)
        self.last_report = report
        return self._run_seahorn(instrumented_code)

    def scan_loops(self, code: str) -> Tuple[List[LoopHeader], Dict[str, object]]:
        return self._scan_loops(code)

    def _run_seahorn(self, code: str) -> str:
        try:
            with tempfile.TemporaryDirectory(prefix="seahorn_") as tmp_dir:
                src_path = Path(tmp_dir) / "input.c"
                src_path.write_text(code, encoding="utf-8")

                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{tmp_dir}:/src",
                    self.docker_image,
                    "sea",
                    "pf",
                    f"/src/{src_path.name}",
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                self.last_run = {
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                }
                output = "\n".join([result.stdout or "", result.stderr or ""]).strip()
        except subprocess.TimeoutExpired:
            self.last_run = {
                "command": cmd,
                "returncode": None,
                "stdout": "",
                "stderr": "timeout",
            }
            return "Error: SeaHorn timed out"
        except Exception as exc:
            self.last_run = {
                "command": cmd if "cmd" in locals() else None,
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
            }
            return f"Error: {exc}"

        if re.search(r"\bunsat\b", output):
            return "Verified"
        if re.search(r"\bsat\b", output):
            return "Failed"

        if result.returncode != 0:
            msg = output or f"SeaHorn exited with code {result.returncode}"
            return f"Error: {msg[:200]}..."

        msg = output or "Unknown SeaHorn output"
        return f"Error: {msg[:200]}..."

    def _instrument_code(
        self,
        code: str,
        loop_invariants: Optional[Dict[int, List[str]]],
        loop_rankings: Optional[Dict[int, str]],
    ) -> Tuple[str, Dict[str, object]]:
        invariants_by_loop = loop_invariants or {}
        rankings_by_loop = loop_rankings or {}

        loop_headers, scan_report = self._scan_loops(code)

        instrumented_loop_ids: List[int] = []
        loops_without_data: List[int] = []
        dropped_invariants: List[Dict[str, str]] = []
        dropped_rankings: List[Dict[str, str]] = []
        use_llabs = False
        needs_asserts = False

        new_parts: List[str] = []
        last_idx = 0

        for header in loop_headers:
            loop_id = header.loop_id
            raw_invs = invariants_by_loop.get(loop_id, [])
            raw_rank = rankings_by_loop.get(loop_id)

            cleaned_invs, inv_drops = self._sanitize_invariants(raw_invs)
            for inv, reason in inv_drops:
                dropped_invariants.append({"loop_id": str(loop_id), "invariant": inv, "reason": reason})

            ranking_expr, rank_reason, needs_llabs = self._sanitize_ranking(raw_rank)
            if rank_reason:
                dropped_rankings.append({"loop_id": str(loop_id), "ranking_function": str(raw_rank), "reason": rank_reason})
            if needs_llabs:
                use_llabs = True

            has_payload = bool(cleaned_invs) or bool(ranking_expr)
            if not has_payload:
                loops_without_data.append(loop_id)

            new_parts.append(code[last_idx:header.line_start])

            if ranking_expr:
                new_parts.append(f"{header.indent}long long __rf_old_{loop_id} = (long long)({ranking_expr});\n")

            new_parts.append(code[header.line_start:header.header_end])

            injection_lines = []
            for inv in cleaned_invs:
                injection_lines.append(f"sassert({inv});")

            if ranking_expr:
                injection_lines.append(f"long long __rf_new_{loop_id} = (long long)({ranking_expr});")
                injection_lines.append(f"sassert(__rf_new_{loop_id} < __rf_old_{loop_id});")
                injection_lines.append(f"sassert(__rf_new_{loop_id} >= 0);")
                injection_lines.append(f"__rf_old_{loop_id} = __rf_new_{loop_id};")

            if injection_lines:
                needs_asserts = True
                instrumented_loop_ids.append(loop_id)
                injection_block = "\n" + "\n".join(
                    f"{header.indent}    {line}" for line in injection_lines
                ) + "\n"
                new_parts.append(injection_block)

                skip_newline = code[header.header_end:header.header_end + 1] == "\n"
                last_idx = header.header_end + (1 if skip_newline else 0)
            else:
                last_idx = header.header_end

        new_parts.append(code[last_idx:])
        instrumented_code = "".join(new_parts)

        if needs_asserts:
            instrumented_code = self._prepend_preamble(instrumented_code, use_llabs)

        report: Dict[str, object] = {
            "total_loops": scan_report["total_loops"],
            "skipped_loops": scan_report["skipped_loops"],
            "instrumented_loop_ids": instrumented_loop_ids,
            "loops_without_data": loops_without_data,
            "dropped_invariants": dropped_invariants,
            "dropped_rankings": dropped_rankings,
            "notes": [
                "Only brace-delimited loops starting at line beginnings are instrumented.",
                "ACSL-style invariants and non-C expressions are dropped.",
            ],
        }
        return instrumented_code, report

    def _prepend_preamble(self, code: str, use_llabs: bool) -> str:
        include_stdlib = use_llabs and not re.search(r"#\s*include\s*<stdlib.h>", code)
        stdlib_line = "#include <stdlib.h>\n" if include_stdlib else ""
        preamble = (
            "/* SeaHorn instrumentation */\n"
            f"{stdlib_line}"
            "extern void __VERIFIER_error(void);\n"
            "extern void __VERIFIER_assume(int);\n"
            "#ifndef sassert\n"
            "#define sassert(X) do { if (!(X)) __VERIFIER_error(); } while (0)\n"
            "#endif\n"
            "#ifndef assume\n"
            "#define assume(X) __VERIFIER_assume(!!(X))\n"
            "#endif\n\n"
        )
        if "SeaHorn instrumentation" in code:
            return code
        return preamble + code

    def _sanitize_invariants(self, invariants: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
        cleaned: List[str] = []
        dropped: List[Tuple[str, str]] = []
        for inv in invariants or []:
            if inv is None:
                continue
            inv_str = str(inv).strip()
            if not inv_str:
                continue
            if any(token in inv_str for token in _INVALID_EXPR_TOKENS) or "\\" in inv_str:
                dropped.append((inv_str, "non_c_expression"))
                continue
            if inv_str.endswith(";"):
                inv_str = inv_str[:-1].strip()
            cleaned.append(inv_str)
        return cleaned, dropped

    def _sanitize_ranking(self, ranking: Optional[str]) -> Tuple[Optional[str], Optional[str], bool]:
        if ranking is None:
            return None, None, False
        ranking_str = str(ranking).strip()
        if not ranking_str:
            return None, "empty", False
        if ranking_str.lower() in {"none", "null", "output", "unknown"}:
            return None, "invalid_literal", False
        if any(token in ranking_str for token in _INVALID_EXPR_TOKENS) or "\\" in ranking_str:
            return None, "non_c_expression", False
        if ranking_str.endswith(";"):
            ranking_str = ranking_str[:-1].strip()

        if ranking_str.startswith("|") and ranking_str.endswith("|") and "||" not in ranking_str:
            inner = ranking_str[1:-1].strip()
            if inner:
                return f"llabs((long long)({inner}))", None, True
            return None, "empty_abs", False

        return ranking_str, None, False

    def _scan_loops(self, code: str) -> Tuple[List[LoopHeader], Dict[str, object]]:
        headers: List[LoopHeader] = []
        skipped_loops: List[Dict[str, str]] = []
        total_loops = 0

        i = 0
        state = "code"
        length = len(code)

        while i < length:
            ch = code[i]
            next_two = code[i:i + 2]

            if state == "code":
                if next_two == "//":
                    state = "line_comment"
                    i += 2
                    continue
                if next_two == "/*":
                    state = "block_comment"
                    i += 2
                    continue
                if ch == '"':
                    state = "string"
                    i += 1
                    continue
                if ch == "'":
                    state = "char"
                    i += 1
                    continue

                keyword = None
                for kw in ("for", "while", "do"):
                    if code.startswith(kw, i) and self._is_word_boundary(code, i, kw):
                        keyword = kw
                        break

                if keyword:
                    line_start = code.rfind("\n", 0, i) + 1
                    prefix = code[line_start:i]
                    if keyword == "while" and prefix.strip().endswith("}"):
                        i += len(keyword)
                        continue

                    total_loops += 1
                    indent = prefix if prefix.strip() == "" else ""
                    if prefix.strip() != "":
                        skipped_loops.append({"loop_id": str(total_loops), "reason": "loop_header_not_at_line_start"})
                        i += len(keyword)
                        continue

                    if keyword == "do":
                        j = self._skip_whitespace(code, i + len(keyword))
                        if j is None or j >= length or code[j] != "{":
                            skipped_loops.append({"loop_id": str(total_loops), "reason": "loop_without_braces"})
                            i += len(keyword)
                            continue
                        headers.append(LoopHeader(loop_id=total_loops, line_start=line_start, header_end=j + 1, indent=indent))
                        i = j + 1
                        continue

                    j = self._skip_whitespace(code, i + len(keyword))
                    if j is None or j >= length or code[j] != "(":
                        skipped_loops.append({"loop_id": str(total_loops), "reason": "loop_header_parse_failed"})
                        i += len(keyword)
                        continue

                    match_idx = self._find_matching_paren(code, j)
                    if match_idx is None:
                        skipped_loops.append({"loop_id": str(total_loops), "reason": "unmatched_parenthesis"})
                        i += len(keyword)
                        continue

                    k = self._skip_whitespace(code, match_idx + 1)
                    if k is None or k >= length or code[k] != "{":
                        skipped_loops.append({"loop_id": str(total_loops), "reason": "loop_without_braces"})
                        i = match_idx + 1
                        continue

                    headers.append(LoopHeader(loop_id=total_loops, line_start=line_start, header_end=k + 1, indent=indent))
                    i = k + 1
                    continue

            elif state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if next_two == "*/":
                    state = "code"
                    i += 2
                    continue
            elif state == "string":
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    state = "code"
            elif state == "char":
                if ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    state = "code"

            i += 1

        report = {
            "total_loops": total_loops,
            "skipped_loops": skipped_loops,
        }
        return headers, report

    @staticmethod
    def _skip_whitespace(code: str, start: int) -> Optional[int]:
        idx = start
        length = len(code)
        while idx < length and code[idx].isspace():
            idx += 1
        return idx

    @staticmethod
    def _is_word_boundary(code: str, start: int, word: str) -> bool:
        end = start + len(word)
        if start > 0 and (code[start - 1].isalnum() or code[start - 1] == "_"):
            return False
        if end < len(code) and (code[end].isalnum() or code[end] == "_"):
            return False
        return True

    def _find_matching_paren(self, code: str, start: int) -> Optional[int]:
        depth = 0
        i = start
        length = len(code)
        state = "code"

        while i < length:
            ch = code[i]
            next_two = code[i:i + 2]

            if state == "code":
                if next_two == "//":
                    state = "line_comment"
                    i += 2
                    continue
                if next_two == "/*":
                    state = "block_comment"
                    i += 2
                    continue
                if ch == '"':
                    state = "string"
                    i += 1
                    continue
                if ch == "'":
                    state = "char"
                    i += 1
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return i
            elif state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if next_two == "*/":
                    state = "code"
                    i += 2
                    continue
            elif state == "string":
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    state = "code"
            elif state == "char":
                if ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    state = "code"

            i += 1

        return None
