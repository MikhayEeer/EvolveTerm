from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import subprocess
import tempfile

import z3

from evolve_term.llm_client import LLMClient, build_llm_client
from evolve_term.utils import parse_llm_yaml


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

_FUNC_CALL_RE = re.compile(r"\b[A-Za-z_]\w*\s*\(")
_BOOL_OP_RE = re.compile(r"(==|!=|<=|>=|<|>|\&\&|\|\|)")
_VAR_MUL_VAR_RE = re.compile(r"\b[A-Za-z_]\w*\b\s*\*\s*\b[A-Za-z_]\w*\b")
_OLD_VALUE_VAR_RE = re.compile(r"\bold_[A-Za-z_]\w*\b")

DEFAULT_SEAHORN_IMAGE = "seahorn/seahorn-llvm14:nightly"


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@dataclass(frozen=True)
class PromptPair:
    name: str
    system: str
    user: str

    def render(self, **replacements: str) -> Dict[str, str]:
        system = self.system
        user = self.user
        for key, value in replacements.items():
            placeholder = "{" + key + "}"
            system = system.replace(placeholder, value)
            user = user.replace(placeholder, value)
        return {"system": system, "user": user}


def default_prompt_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / "invariants" / "2stage_simple"


def load_prompt_pair(prompt_dir: Path, base_name: str) -> PromptPair:
    system_path = prompt_dir / f"{base_name}.system.txt"
    user_path = prompt_dir / f"{base_name}.user.txt"
    if not system_path.exists() or not user_path.exists():
        raise FileNotFoundError(f"Prompt files not found for '{base_name}' in {prompt_dir}")
    return PromptPair(
        name=base_name,
        system=system_path.read_text(encoding="utf-8"),
        user=user_path.read_text(encoding="utf-8"),
    )


@dataclass(frozen=True)
class LoopInfo:
    loop_id: int
    line_start: int
    header_end: int
    body_end: int
    loop_end: int
    indent: str
    snippet: str


class LoopScanner:
    def scan(self, code: str) -> Tuple[List[LoopInfo], Dict[str, object]]:
        loops: List[LoopInfo] = []
        skipped: List[Dict[str, str]] = []
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
                        skipped.append({"loop_id": str(total_loops), "reason": "loop_header_not_at_line_start"})
                        i += len(keyword)
                        continue

                    if keyword == "do":
                        j = self._skip_whitespace(code, i + len(keyword))
                        if j is None or j >= length or code[j] != "{":
                            skipped.append({"loop_id": str(total_loops), "reason": "loop_without_braces"})
                            i += len(keyword)
                            continue
                        header_end = j + 1
                    else:
                        j = self._skip_whitespace(code, i + len(keyword))
                        if j is None or j >= length or code[j] != "(":
                            skipped.append({"loop_id": str(total_loops), "reason": "loop_header_parse_failed"})
                            i += len(keyword)
                            continue

                        match_idx = self._find_matching_paren(code, j)
                        if match_idx is None:
                            skipped.append({"loop_id": str(total_loops), "reason": "unmatched_parenthesis"})
                            i += len(keyword)
                            continue

                        k = self._skip_whitespace(code, match_idx + 1)
                        if k is None or k >= length or code[k] != "{":
                            skipped.append({"loop_id": str(total_loops), "reason": "loop_without_braces"})
                            i = match_idx + 1
                            continue

                        header_end = k + 1

                    body_end = self._find_matching_brace(code, header_end - 1)
                    if body_end is None:
                        skipped.append({"loop_id": str(total_loops), "reason": "unmatched_brace"})
                        i = header_end
                        continue

                    loop_end = body_end + 1
                    snippet = code[line_start:loop_end]
                    loops.append(
                        LoopInfo(
                            loop_id=total_loops,
                            line_start=line_start,
                            header_end=header_end,
                            body_end=body_end,
                            loop_end=loop_end,
                            indent=indent,
                            snippet=snippet,
                        )
                    )
                    i = header_end
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

        report = {"total_loops": total_loops, "skipped_loops": skipped}
        return loops, report

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

    def _find_matching_brace(self, code: str, start: int) -> Optional[int]:
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
                if ch == "{":
                    depth += 1
                elif ch == "}":
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


def strip_c_comments_and_strings(code: str) -> str:
    result: List[str] = []
    i = 0
    state = "code"
    length = len(code)

    while i < length:
        ch = code[i]
        next_two = code[i:i + 2]

        if state == "code":
            if next_two == "//":
                state = "line_comment"
                result.append("  ")
                i += 2
                continue
            if next_two == "/*":
                state = "block_comment"
                result.append("  ")
                i += 2
                continue
            if ch == '"':
                state = "string"
                result.append(" ")
                i += 1
                continue
            if ch == "'":
                state = "char"
                result.append(" ")
                i += 1
                continue
            result.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                state = "code"
                result.append("\n")
            else:
                result.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if next_two == "*/":
                state = "code"
                result.append("  ")
                i += 2
            else:
                result.append(" ")
                i += 1
            continue

        if state == "string":
            if ch == "\\":
                result.append(" ")
                if i + 1 < length:
                    result.append(" ")
                    i += 2
                else:
                    i += 1
                continue
            if ch == '"':
                state = "code"
            result.append(" ")
            i += 1
            continue

        if state == "char":
            if ch == "\\":
                result.append(" ")
                if i + 1 < length:
                    result.append(" ")
                    i += 2
                else:
                    i += 1
                continue
            if ch == "'":
                state = "code"
            result.append(" ")
            i += 1
            continue

    return "".join(result)


def extract_modified_vars(code: str) -> List[str]:
    cleaned = strip_c_comments_and_strings(code)
    assign_op = r"(?:\+=|-=|\*=|/=|%=|(?<![=!<>])=(?![=]))"

    patterns = [
        re.compile(rf"\b([A-Za-z_]\w*)\s*{assign_op}"),
        re.compile(rf"\*\s*([A-Za-z_]\w*)\s*{assign_op}"),
        re.compile(rf"\b([A-Za-z_]\w*)\s*\[[^\]]+\]\s*{assign_op}"),
        re.compile(rf"\b([A-Za-z_]\w*)\s*(?:->|\.)\s*[A-Za-z_]\w*\s*{assign_op}"),
        re.compile(r"\b([A-Za-z_]\w*)\s*(?:\+\+|--)"),
        re.compile(r"(?:\+\+|--)\s*([A-Za-z_]\w*)"),
    ]

    seen = set()
    ordered: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                ordered.append(name)

    return ordered


def _contains_modified_var(expr: str, modified_vars: List[str]) -> bool:
    if not modified_vars:
        return True
    for name in modified_vars:
        if re.search(rf"\b{re.escape(name)}\b", expr):
            return True
    return False


def _has_unary_deref(expr: str) -> bool:
    i = 0
    length = len(expr)
    while i < length:
        if expr[i] != "*":
            i += 1
            continue
        prev = i - 1
        while prev >= 0 and expr[prev].isspace():
            prev -= 1
        if prev < 0:
            return True
        if expr[prev] in "([=,+-*/%<>&|!^~?:;":
            return True
        i += 1
    return False


def _is_tautology_pattern(expr: str) -> bool:
    compact = re.sub(r"\s+", "", expr)
    if re.search(r"\b([A-Za-z_]\w*)==\1\b", compact):
        return True
    if re.search(r"\b(\d+)==\1\b", compact):
        return True
    if re.search(r"\b([A-Za-z_]\w*)\*\1>=0\b", compact):
        return True
    if re.search(r"\b0<=([A-Za-z_]\w*)\*\1\b", compact):
        return True
    return False


def syntax_filter(
    exprs: List[str],
    modified_vars: List[str],
    require_modified_var: bool = True,
) -> Tuple[List[str], List[Dict[str, str]]]:
    accepted: List[str] = []
    dropped: List[Dict[str, str]] = []

    for expr in exprs or []:
        if expr is None:
            continue
        expr_str = str(expr).strip()
        if not expr_str:
            continue
        if expr_str.endswith(";"):
            expr_str = expr_str[:-1].strip()
        if any(token in expr_str for token in _INVALID_EXPR_TOKENS) or "\\" in expr_str:
            dropped.append({"expr": expr_str, "reason": "non_c_expression"})
            continue
        if "\\old" in expr_str or _OLD_VALUE_VAR_RE.search(expr_str):
            dropped.append({"expr": expr_str, "reason": "old_value"})
            continue
        if "#" in expr_str:
            dropped.append({"expr": expr_str, "reason": "macro"})
            continue
        if not _BOOL_OP_RE.search(expr_str):
            dropped.append({"expr": expr_str, "reason": "non_boolean"})
            continue
        if "++" in expr_str or "--" in expr_str:
            dropped.append({"expr": expr_str, "reason": "increment"})
            continue
        if _FUNC_CALL_RE.search(expr_str):
            dropped.append({"expr": expr_str, "reason": "function_call"})
            continue
        if "[" in expr_str or "]" in expr_str:
            dropped.append({"expr": expr_str, "reason": "array_access"})
            continue
        if "->" in expr_str or "." in expr_str:
            dropped.append({"expr": expr_str, "reason": "member_access"})
            continue
        if _has_unary_deref(expr_str):
            dropped.append({"expr": expr_str, "reason": "pointer_deref"})
            continue
        if _VAR_MUL_VAR_RE.search(expr_str):
            dropped.append({"expr": expr_str, "reason": "non_linear"})
            continue
        if _is_tautology_pattern(expr_str):
            dropped.append({"expr": expr_str, "reason": "tautology_pattern"})
            continue
        if require_modified_var and not _contains_modified_var(expr_str, modified_vars):
            dropped.append({"expr": expr_str, "reason": "missing_modified_var"})
            continue

        accepted.append(expr_str)

    return _dedupe_preserve_order(accepted), dropped


class ParseError(Exception):
    pass


@dataclass
class ArithNode:
    expr: z3.ArithRef
    kind: str  # const | var | expr


@dataclass(frozen=True)
class Token:
    kind: str
    value: object


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    length = len(expr)

    while i < length:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if expr.startswith("&&", i) or expr.startswith("||", i) or expr.startswith("==", i) \
            or expr.startswith("!=", i) or expr.startswith("<=", i) or expr.startswith(">=", i):
            tokens.append(Token("OP", expr[i:i + 2]))
            i += 2
            continue
        if ch in "()":
            tokens.append(Token(ch, ch))
            i += 1
            continue
        if ch in "+-*<>!":
            tokens.append(Token("OP", ch))
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < length and (expr[j].isdigit() or expr[j] in "xXabcdefABCDEF"):
                j += 1
            raw = expr[i:j]
            try:
                value = int(raw, 0)
            except ValueError as exc:
                raise ParseError(f"invalid integer literal: {raw}") from exc
            tokens.append(Token("INT", value))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < length and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            ident = expr[i:j]
            tokens.append(Token("IDENT", ident))
            i = j
            continue
        raise ParseError(f"unexpected character: {ch}")

    tokens.append(Token("EOF", None))
    return tokens


class ExprParser:
    def __init__(self, tokens: List[Token], var_map: Dict[str, z3.ArithRef]):
        self.tokens = tokens
        self.pos = 0
        self.var_map = var_map

    def parse_bool(self) -> z3.BoolRef:
        expr = self._parse_or()
        if self._peek().kind != "EOF":
            raise ParseError("unexpected token after expression")
        return expr

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _match(self, kind: str, value: Optional[str] = None) -> bool:
        token = self._peek()
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        self._advance()
        return True

    def _parse_or(self) -> z3.BoolRef:
        left = self._parse_and()
        while self._match("OP", "||"):
            right = self._parse_and()
            left = z3.Or(left, right)
        return left

    def _parse_and(self) -> z3.BoolRef:
        left = self._parse_not()
        while self._match("OP", "&&"):
            right = self._parse_not()
            left = z3.And(left, right)
        return left

    def _parse_not(self) -> z3.BoolRef:
        if self._match("OP", "!"):
            return z3.Not(self._parse_not())
        if self._peek().kind == "(" and self._looks_like_boolean_group():
            self._advance()
            expr = self._parse_or()
            if not self._match(")", ")"):
                raise ParseError("missing closing parenthesis")
            return expr
        return self._parse_relation()

    def _looks_like_boolean_group(self) -> bool:
        if self._peek().kind != "(":
            return False
        depth = 0
        idx = self.pos
        while idx < len(self.tokens):
            token = self.tokens[idx]
            if token.kind == "(":
                depth += 1
            elif token.kind == ")":
                depth -= 1
                if depth == 0:
                    return False
            elif depth == 1 and token.kind == "OP" and token.value in ("&&", "||", "==", "!=", "<=", ">=", "<", ">"):
                return True
            idx += 1
        return False

    def _parse_relation(self) -> z3.BoolRef:
        left = self._parse_arith()
        token = self._peek()
        if token.kind == "OP" and token.value in ("==", "!=", "<=", ">=", "<", ">"):
            op = token.value
            self._advance()
            right = self._parse_arith()
            if op == "==":
                return left.expr == right.expr
            if op == "!=":
                return left.expr != right.expr
            if op == "<=":
                return left.expr <= right.expr
            if op == ">=":
                return left.expr >= right.expr
            if op == "<":
                return left.expr < right.expr
            if op == ">":
                return left.expr > right.expr
        raise ParseError("missing comparison operator")

    def _parse_arith(self) -> ArithNode:
        node = self._parse_term()
        while True:
            if self._match("OP", "+"):
                rhs = self._parse_term()
                node = ArithNode(node.expr + rhs.expr, "expr")
            elif self._match("OP", "-"):
                rhs = self._parse_term()
                node = ArithNode(node.expr - rhs.expr, "expr")
            else:
                break
        return node

    def _parse_term(self) -> ArithNode:
        node = self._parse_factor()
        while self._match("OP", "*"):
            rhs = self._parse_factor()
            node = self._mul_nodes(node, rhs)
        return node

    def _mul_nodes(self, left: ArithNode, right: ArithNode) -> ArithNode:
        if left.kind == "expr" or right.kind == "expr":
            raise ParseError("nonlinear_multiplication")
        if left.kind == "var" and right.kind == "var":
            raise ParseError("nonlinear_multiplication")
        expr = left.expr * right.expr
        if left.kind == "const" and right.kind == "const":
            return ArithNode(expr, "const")
        if left.kind == "var" and right.kind == "const":
            return ArithNode(expr, "var")
        if left.kind == "const" and right.kind == "var":
            return ArithNode(expr, "var")
        return ArithNode(expr, "expr")

    def _parse_factor(self) -> ArithNode:
        if self._match("OP", "+"):
            return self._parse_factor()
        if self._match("OP", "-"):
            node = self._parse_factor()
            return ArithNode(-node.expr, node.kind)
        token = self._peek()
        if token.kind == "INT":
            self._advance()
            return ArithNode(z3.IntVal(int(token.value)), "const")
        if token.kind == "IDENT":
            self._advance()
            name = str(token.value)
            if name in {"true", "false"}:
                value = 1 if name == "true" else 0
                return ArithNode(z3.IntVal(value), "const")
            var = self.var_map.get(name)
            if var is None:
                var = z3.Int(name)
                self.var_map[name] = var
            return ArithNode(var, "var")
        if self._match("(", "("):
            node = self._parse_arith()
            if not self._match(")", ")"):
                raise ParseError("missing closing parenthesis")
            return node
        raise ParseError(f"unexpected token: {token.kind}")


def _strength_score(expr: str) -> float:
    compact = re.sub(r"\s+", "", expr)
    comparators = re.findall(r"==|!=|<=|>=|<|>", compact)
    score = 0.0
    for op in comparators:
        if op == "==":
            score += 4.0
        elif op in ("<=", ">="):
            score += 3.0
        elif op in ("<", ">"):
            score += 2.0
        else:
            score += 1.0
    score += compact.count("&&") * 2.0
    score += compact.count("||") * 0.5
    score += min(len(compact) / 20.0, 3.0)
    return score


def _z3_unsat(expr: z3.BoolRef, timeout_ms: int) -> bool:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(expr)
    return solver.check() == z3.unsat


def _z3_implied(assumptions: List[z3.BoolRef], expr: z3.BoolRef, timeout_ms: int) -> Optional[bool]:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    if assumptions:
        solver.add(*assumptions)
    solver.add(z3.Not(expr))
    result = solver.check()
    if result == z3.unsat:
        return True
    if result == z3.sat:
        return False
    return None


def z3_filter(
    exprs: List[str],
    max_keep: int,
    timeout_ms: int = 1500,
) -> Tuple[List[str], List[Dict[str, str]]]:
    status_map: Dict[str, Dict[str, str]] = {}
    parsed: List[Tuple[str, z3.BoolRef, float]] = []
    translator_vars: Dict[str, z3.ArithRef] = {}

    for expr in exprs:
        status_map[expr] = {"expr": expr, "status": "candidate", "reason": ""}
        try:
            tokens = tokenize(expr)
            parser = ExprParser(tokens, translator_vars)
            parsed_expr = parser.parse_bool()
        except ParseError as exc:
            status_map[expr] = {"expr": expr, "status": "dropped", "reason": f"parse_error:{exc}"}
            continue

        if _z3_unsat(z3.Not(parsed_expr), timeout_ms):
            status_map[expr] = {"expr": expr, "status": "dropped", "reason": "tautology"}
            continue
        if _z3_unsat(parsed_expr, timeout_ms):
            status_map[expr] = {"expr": expr, "status": "dropped", "reason": "contradiction"}
            continue

        parsed.append((expr, parsed_expr, _strength_score(expr)))

    parsed.sort(key=lambda item: item[2], reverse=True)
    kept: List[str] = []
    kept_z3: List[z3.BoolRef] = []

    for expr, parsed_expr, _score in parsed:
        implied = _z3_implied(kept_z3, parsed_expr, timeout_ms)
        if implied is True:
            status_map[expr] = {"expr": expr, "status": "dropped", "reason": "implied"}
            continue
        kept.append(expr)
        kept_z3.append(parsed_expr)
        status_map[expr] = {"expr": expr, "status": "kept", "reason": ""}

    if max_keep > 0 and len(kept) > max_keep:
        trimmed = kept[max_keep:]
        kept = kept[:max_keep]
        for expr in trimmed:
            status_map[expr] = {"expr": expr, "status": "dropped", "reason": "trimmed"}

    report = [status_map[expr] for expr in exprs if expr in status_map]
    return kept, report


def _parse_yaml_list(response: str, key: str) -> List[str]:
    data = parse_llm_yaml(response)
    if isinstance(data, dict):
        items = data.get(key, [])
        if isinstance(items, list):
            return [str(item).strip() for item in items if str(item).strip()]
        if isinstance(items, str):
            return [part.strip() for part in items.split(",") if part.strip()]

    match = re.search(rf"{re.escape(key)}\s*:\s*\[(.*?)\]", response, re.DOTALL)
    if match:
        return [part.strip() for part in match.group(1).split(",") if part.strip()]
    return []


def _format_atoms_block(atoms: List[str]) -> str:
    if not atoms:
        return "[]"
    return "\n".join(f"- {atom}" for atom in atoms)


def _ensure_seahorn_include(code: str) -> str:
    if re.search(r'#\s*include\s*[<"]seahorn/seahorn\.h[>"]', code):
        return code
    lines = code.splitlines(keepends=True)
    insert_idx = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#include"):
            insert_idx = idx + 1
    lines.insert(insert_idx, '#include "seahorn/seahorn.h"\n')
    return "".join(lines)


def instrument_code_with_invariants(
    code: str,
    loops: List[LoopInfo],
    invariants_by_loop: Dict[int, List[str]],
) -> Tuple[str, Dict[str, object]]:
    if not loops or not invariants_by_loop:
        return code, {"instrumented_loop_ids": []}

    new_parts: List[str] = []
    last_idx = 0
    instrumented_loop_ids: List[int] = []

    for loop in loops:
        new_parts.append(code[last_idx:loop.header_end])
        invs = invariants_by_loop.get(loop.loop_id, [])
        if invs:
            indent = loop.indent + "    "
            lines = [f"{indent}sassert({inv});" for inv in invs]
            block = "\n" + "\n".join(lines) + "\n"
            new_parts.append(block)
            instrumented_loop_ids.append(loop.loop_id)
        last_idx = loop.header_end

    new_parts.append(code[last_idx:])
    instrumented = "".join(new_parts)
    if instrumented_loop_ids:
        instrumented = _ensure_seahorn_include(instrumented)

    return instrumented, {"instrumented_loop_ids": instrumented_loop_ids}


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _combine_output(stdout: object, stderr: object) -> str:
    out = _coerce_text(stdout)
    err = _coerce_text(stderr)
    if out and err:
        return f"{out}\n{err}"
    return out or err


def run_seahorn_check(
    code: str,
    inv_id: int,
    log_dir: Path,
    timeout_seconds: int,
    docker_image: str = DEFAULT_SEAHORN_IMAGE,
) -> Dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    src_path = log_dir / f"inv_{inv_id}.c"
    log_path = log_dir / f"inv_{inv_id}.log"
    src_path.write_text(code, encoding="utf-8")

    volume_dir = log_dir.resolve()
    container_path = f"/src/{src_path.name}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{volume_dir}:/src",
        docker_image,
        "sea",
        "pf",
        container_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = _combine_output(result.stdout, result.stderr)
        if not output.strip():
            output = "(seahorn output is empty)"
        log_path.write_text(output, encoding="utf-8")
    except FileNotFoundError:
        log_path.write_text("docker: command not found\n", encoding="utf-8")
        return {"status": "error", "reason": "docker_not_found", "log_file": str(log_path)}
    except subprocess.TimeoutExpired as exc:
        output = _combine_output(getattr(exc, "stdout", ""), getattr(exc, "stderr", ""))
        if not output.strip():
            output = "docker: timeout"
        log_path.write_text(output, encoding="utf-8")
        return {"status": "error", "reason": "timeout", "log_file": str(log_path)}
    except Exception as exc:
        log_path.write_text(str(exc) or "docker: unknown error", encoding="utf-8")
        return {"status": "error", "reason": "exception", "log_file": str(log_path)}

    if re.search(r"\bunsat\b", output):
        return {"status": "Verified", "reason": "", "log_file": str(log_path)}
    if re.search(r"\bsat\b", output):
        return {"status": "Failed", "reason": "", "log_file": str(log_path)}
    if result.returncode != 0:
        return {"status": "error", "reason": f"exit_{result.returncode}", "log_file": str(log_path)}
    return {"status": "error", "reason": "unexpected_output", "log_file": str(log_path)}


def infer_invariants(
    code: str,
    llm_config: str = "llm_config.json",
    llm_tag: str = "default",
    prompt_dir: Optional[Path] = None,
    loop_id: int = 1,
    max_atoms: int = 12,
    max_invariants: int = 4,
    seahorn_timeout: int = 60,
    enable_seahorn: bool = True,
    seahorn_log_dir: Optional[Path] = None,
) -> Tuple[Dict[str, object], str]:
    llm_client: LLMClient = build_llm_client(llm_config, llm_tag)
    prompt_dir = prompt_dir or default_prompt_dir()
    stage1_prompts = load_prompt_pair(prompt_dir, "seahorn_stage1")
    stage2_prompts = load_prompt_pair(prompt_dir, "seahorn_stage2")

    loop_scanner = LoopScanner()
    loops, scan_report = loop_scanner.scan(code)
    if not loops:
        raise ValueError("No loops found to synthesize invariants for.")
    selected_loop = next((loop for loop in loops if loop.loop_id == loop_id), None)
    if selected_loop is None:
        raise ValueError(f"Loop id {loop_id} not found in code.")

    modified_vars = extract_modified_vars(selected_loop.snippet)

    stage1_prompt = stage1_prompts.render(CODE=selected_loop.snippet)
    stage1_response = llm_client.complete(stage1_prompt)
    stage1_atoms_raw = _parse_yaml_list(stage1_response, "atoms")

    stage1_atoms_syntax, stage1_syntax_drops = syntax_filter(
        stage1_atoms_raw,
        modified_vars,
        require_modified_var=True,
    )
    stage1_atoms_filtered, stage1_z3_report = z3_filter(stage1_atoms_syntax, max_atoms)

    atoms_block = _format_atoms_block(stage1_atoms_filtered)
    stage2_prompt = stage2_prompts.render(CODE=selected_loop.snippet, ATOMS=atoms_block)
    stage2_response = llm_client.complete(stage2_prompt)
    stage2_invs_raw = _parse_yaml_list(stage2_response, "invariants")

    stage2_invs_syntax, stage2_syntax_drops = syntax_filter(
        stage2_invs_raw,
        modified_vars,
        require_modified_var=True,
    )
    stage2_invs_filtered, stage2_z3_report = z3_filter(stage2_invs_syntax, max_invariants)

    seahorn_results: List[Dict[str, str]] = []
    final_invariants: List[str] = []
    if enable_seahorn and stage2_invs_filtered:
        if seahorn_log_dir is None:
            seahorn_log_dir = Path(tempfile.mkdtemp(prefix="seahorn_logs_"))
        for idx, inv in enumerate(stage2_invs_filtered, start=1):
            instrumented_one, _ = instrument_code_with_invariants(
                code,
                loops,
                {selected_loop.loop_id: [inv]},
            )
            check = run_seahorn_check(
                instrumented_one,
                idx,
                seahorn_log_dir,
                seahorn_timeout,
            )
            result = {"invariant": inv, **check}
            seahorn_results.append(result)
            if check["status"] == "Verified":
                final_invariants.append(inv)
    else:
        seahorn_results = [
            {"invariant": inv, "status": "skipped", "reason": "seahorn_not_enabled", "log_file": ""}
            for inv in stage2_invs_filtered
        ]

    instrument_invariants = final_invariants or stage2_invs_filtered
    instrumented_code, instrument_report = instrument_code_with_invariants(
        code, loops, {selected_loop.loop_id: instrument_invariants}
    )

    report: Dict[str, object] = {
        "modified_vars": modified_vars,
        "stage1_atoms_raw": stage1_atoms_raw,
        "stage1_atoms_syntax": stage1_atoms_syntax,
        "stage1_atoms_filtered": stage1_atoms_filtered,
        "stage1_syntax_dropped": stage1_syntax_drops,
        "stage1_z3_report": stage1_z3_report,
        "stage2_invs_raw": stage2_invs_raw,
        "stage2_invs_syntax": stage2_invs_syntax,
        "stage2_invs_filtered": stage2_invs_filtered,
        "stage2_syntax_dropped": stage2_syntax_drops,
        "stage2_z3_report": stage2_z3_report,
        "seahorn_check_results": seahorn_results,
        "final_invariants": final_invariants,
        "instrumented_invariants": instrument_invariants,
        "loop_count": len(loops),
        "selected_loop_id": selected_loop.loop_id,
        "loop_scan_report": scan_report,
        "instrument_report": instrument_report,
    }

    if seahorn_log_dir is not None:
        report["seahorn_log_dir"] = str(seahorn_log_dir)

    return report, instrumented_code
