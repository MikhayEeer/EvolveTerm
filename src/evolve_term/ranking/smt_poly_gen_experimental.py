"""SMT-based piecewise linear ranking synthesis (experimental)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import z3


_INVALID_INV_TOKENS = {"\\", "\\old", "old(", "\\forall", "\\exists"}


@dataclass
class LinearExpr:
    coeffs: Dict[str, int]
    const: int = 0

    def add(self, other: "LinearExpr") -> "LinearExpr":
        coeffs = dict(self.coeffs)
        for name, val in other.coeffs.items():
            coeffs[name] = coeffs.get(name, 0) + val
        return LinearExpr(coeffs, self.const + other.const)

    def sub(self, other: "LinearExpr") -> "LinearExpr":
        coeffs = dict(self.coeffs)
        for name, val in other.coeffs.items():
            coeffs[name] = coeffs.get(name, 0) - val
        return LinearExpr(coeffs, self.const - other.const)

    def mul_const(self, k: int) -> "LinearExpr":
        coeffs = {name: val * k for name, val in self.coeffs.items()}
        return LinearExpr(coeffs, self.const * k)

    @property
    def has_var(self) -> bool:
        return any(v != 0 for v in self.coeffs.values())


class _TokenStream:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def next(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, value: str) -> bool:
        if self.peek() == value:
            self.next()
            return True
        return False


def _tokenize(expr: str) -> List[str]:
    token_re = re.compile(r"\s*(\d+|[A-Za-z_]\w*|<=|>=|==|!=|\&\&|\|\||[()+\-*<>])")
    tokens = token_re.findall(expr)
    return [t for t in tokens if t and not t.isspace()]


def _parse_linear_expr(expr: str, env: Dict[str, LinearExpr]) -> Optional[LinearExpr]:
    tokens = _tokenize(expr)
    stream = _TokenStream(tokens)

    def parse_factor() -> Optional[LinearExpr]:
        tok = stream.peek()
        if tok is None:
            return None
        if tok == "+":
            stream.next()
            return parse_factor()
        if tok == "-":
            stream.next()
            inner = parse_factor()
            if inner is None:
                return None
            return inner.mul_const(-1)
        if tok == "(":
            stream.next()
            inner = parse_expr()
            if inner is None or not stream.expect(")"):
                return None
            return inner
        if tok.isdigit():
            stream.next()
            return LinearExpr({}, int(tok))
        if re.match(r"[A-Za-z_]\w*", tok):
            stream.next()
            return env.get(tok, LinearExpr({tok: 1}, 0))
        return None

    def parse_term() -> Optional[LinearExpr]:
        left = parse_factor()
        if left is None:
            return None
        while stream.peek() == "*":
            stream.next()
            right = parse_factor()
            if right is None:
                return None
            if left.has_var and right.has_var:
                return None
            if left.has_var:
                if right.has_var:
                    return None
                left = left.mul_const(right.const)
            elif right.has_var:
                left = right.mul_const(left.const)
            else:
                left = LinearExpr({}, left.const * right.const)
        return left

    def parse_expr() -> Optional[LinearExpr]:
        left = parse_term()
        if left is None:
            return None
        while True:
            tok = stream.peek()
            if tok == "+":
                stream.next()
                right = parse_term()
                if right is None:
                    return None
                left = left.add(right)
            elif tok == "-":
                stream.next()
                right = parse_term()
                if right is None:
                    return None
                left = left.sub(right)
            else:
                break
        return left

    expr_val = parse_expr()
    if expr_val is None or stream.peek() is not None:
        return None
    return expr_val


def _linear_to_z3(expr: LinearExpr, vars_map: Dict[str, z3.IntNumRef]) -> z3.ArithRef:
    total = z3.IntVal(expr.const)
    for name, coeff in expr.coeffs.items():
        if coeff == 0:
            continue
        total += z3.IntVal(coeff) * vars_map[name]
    return total


def _parse_condition(cond: str, env: Dict[str, LinearExpr], vars_map: Dict[str, z3.IntNumRef]) -> Optional[z3.BoolRef]:
    tokens = _tokenize(cond)
    stream = _TokenStream(tokens)

    def parse_comparison() -> Optional[z3.BoolRef]:
        left_expr_tokens = []
        while True:
            tok = stream.peek()
            if tok is None:
                break
            if tok in {"<", "<=", ">", ">=", "==", "!="}:
                break
            left_expr_tokens.append(stream.next())
        if not left_expr_tokens:
            return None
        op = stream.next()
        if op not in {"<", "<=", ">", ">=", "==", "!="}:
            return None
        right_expr_tokens = []
        while True:
            tok = stream.peek()
            if tok is None or tok in {"&&", "||", ")"}:
                break
            right_expr_tokens.append(stream.next())
        left_expr = _parse_linear_expr(" ".join(left_expr_tokens), env)
        right_expr = _parse_linear_expr(" ".join(right_expr_tokens), env)
        if left_expr is None or right_expr is None:
            return None
        left_z3 = _linear_to_z3(left_expr, vars_map)
        right_z3 = _linear_to_z3(right_expr, vars_map)
        if op == "<":
            return left_z3 < right_z3
        if op == "<=":
            return left_z3 <= right_z3
        if op == ">":
            return left_z3 > right_z3
        if op == ">=":
            return left_z3 >= right_z3
        if op == "==":
            return left_z3 == right_z3
        if op == "!=":
            return left_z3 != right_z3
        return None

    def parse_and() -> Optional[z3.BoolRef]:
        left = parse_comparison()
        if left is None:
            return None
        while stream.peek() == "&&":
            stream.next()
            right = parse_comparison()
            if right is None:
                return None
            left = z3.And(left, right)
        return left

    def parse_or() -> Optional[z3.BoolRef]:
        left = parse_and()
        if left is None:
            return None
        while stream.peek() == "||":
            stream.next()
            right = parse_and()
            if right is None:
                return None
            left = z3.Or(left, right)
        return left

    result = parse_or()
    if result is None or stream.peek() is not None:
        return None
    return result


def _strip_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code


def _find_matching(code: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for idx in range(start, len(code)):
        ch = code[idx]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_while(code: str) -> Optional[Tuple[str, str]]:
    match = re.search(r"\bwhile\s*\(", code)
    if not match:
        return None
    start = match.end() - 1
    end = _find_matching(code, start, "(", ")")
    if end == -1:
        return None
    cond = code[start + 1:end].strip()
    brace_start = code.find("{", end)
    if brace_start == -1:
        return None
    brace_end = _find_matching(code, brace_start, "{", "}")
    if brace_end == -1:
        return None
    body = code[brace_start + 1:brace_end]
    return cond, body


def _parse_statement_block(code: str, start_idx: int) -> Optional[Tuple[str, int]]:
    idx = start_idx
    while idx < len(code) and code[idx].isspace():
        idx += 1
    if idx >= len(code):
        return None
    if code[idx] == "{":
        end = _find_matching(code, idx, "{", "}")
        if end == -1:
            return None
        return code[idx + 1:end], end + 1
    end = code.find(";", idx)
    if end == -1:
        return None
    return code[idx:end + 1], end + 1


def _parse_if_else(body: str) -> Optional[Tuple[str, str, str]]:
    stripped = body.lstrip()
    if not stripped.startswith("if"):
        return None
    base_offset = body.find("if")
    idx = base_offset + 2
    while idx < len(body) and body[idx].isspace():
        idx += 1
    if idx >= len(body) or body[idx] != "(":
        return None
    cond_end = _find_matching(body, idx, "(", ")")
    if cond_end == -1:
        return None
    cond = body[idx + 1:cond_end].strip()
    block = _parse_statement_block(body, cond_end + 1)
    if block is None:
        return None
    then_block, idx = block
    while idx < len(body) and body[idx].isspace():
        idx += 1
    if not body[idx:].lstrip().startswith("else"):
        return None
    idx = body.find("else", idx) + 4
    block = _parse_statement_block(body, idx)
    if block is None:
        return None
    else_block, _ = block
    return cond, then_block, else_block


def _parse_assignments(block: str, env: Dict[str, LinearExpr]) -> Optional[Tuple[Dict[str, LinearExpr], List[str]]]:
    assigned = set()
    statements = [s.strip() for s in block.split(";") if s.strip()]
    for stmt in statements:
        if stmt.startswith("if") or stmt.startswith("while") or stmt.startswith("for"):
            return None
        m = re.match(r"^([A-Za-z_]\w*)\s*(\+\+|--)$", stmt)
        if m:
            var = m.group(1)
            delta = 1 if m.group(2) == "++" else -1
            env[var] = env.get(var, LinearExpr({var: 1}, 0)).add(LinearExpr({}, delta))
            assigned.add(var)
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*([+\-*/]=)\s*(.+)$", stmt)
        if m:
            var = m.group(1)
            op = m.group(2)
            rhs_expr = _parse_linear_expr(m.group(3), env)
            if rhs_expr is None:
                return None
            current = env.get(var, LinearExpr({var: 1}, 0))
            if op == "+=":
                env[var] = current.add(rhs_expr)
            elif op == "-=":
                env[var] = current.sub(rhs_expr)
            elif op == "*=":
                if rhs_expr.has_var:
                    return None
                env[var] = current.mul_const(rhs_expr.const)
            else:
                return None
            assigned.add(var)
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", stmt)
        if m:
            var = m.group(1)
            rhs_expr = _parse_linear_expr(m.group(2), env)
            if rhs_expr is None:
                return None
            env[var] = rhs_expr
            assigned.add(var)
            continue
        return None
    updates = {name: env[name] for name in assigned}
    return updates, list(assigned)


def _format_linear(expr: LinearExpr) -> str:
    parts: List[str] = []
    for name, coeff in sorted(expr.coeffs.items()):
        if coeff == 0:
            continue
        if coeff == 1:
            parts.append(name)
        elif coeff == -1:
            parts.append(f"-{name}")
        else:
            parts.append(f"{coeff}*{name}")
    if expr.const != 0 or not parts:
        parts.append(str(expr.const))
    result = " + ".join(parts)
    result = result.replace("+ -", "- ")
    return result


class SMTLinearRankSynthesizer:
    def __init__(self, max_coeff: int = 5, require_nonneg: bool = True):
        self.max_coeff = max_coeff
        self.require_nonneg = require_nonneg
        self.last_reason: str = ""

    def synthesize(self, loop_code: str, invariants: List[str]) -> Optional[str]:
        self.last_reason = ""
        clean_code = _strip_comments(loop_code)
        extracted = _extract_while(clean_code)
        if not extracted:
            self.last_reason = "no_while_loop_found"
            return None
        cond_raw, body = extracted

        branch = _parse_if_else(body)
        cond_piece = None
        cond_then = None
        if branch:
            cond_then, then_body, else_body = branch
            branches = [
                ("then", cond_then, then_body),
                ("else", None, else_body),
            ]
            cond_piece = cond_then
        else:
            branches = [("body", "true", body)]

        all_vars: Dict[str, LinearExpr] = {}
        base_env: Dict[str, LinearExpr] = {}
        var_candidates = set(re.findall(r"[A-Za-z_]\w*", cond_raw))
        for inv in invariants:
            var_candidates.update(re.findall(r"[A-Za-z_]\w*", inv))
        for name in var_candidates:
            if name in {"while", "if", "else", "true", "false"}:
                continue
            base_env[name] = LinearExpr({name: 1}, 0)
        all_vars.update(base_env)

        vars_map = {name: z3.Int(name) for name in all_vars}
        if not vars_map:
            self.last_reason = "no_variables_found"
            return None

        loop_guard = _parse_condition(cond_raw, base_env, vars_map)
        if loop_guard is None:
            self.last_reason = "loop_guard_parse_failed"
            return None

        inv_exprs: List[z3.BoolRef] = []
        for inv in invariants:
            inv_strip = inv.strip()
            if not inv_strip or any(token in inv_strip for token in _INVALID_INV_TOKENS):
                continue
            parsed_inv = _parse_condition(inv_strip, base_env, vars_map)
            if parsed_inv is not None:
                inv_exprs.append(parsed_inv)

        sat_branches = []
        branch_updates: List[Tuple[Optional[str], str, Dict[str, LinearExpr]]] = []
        parsed_then_guard = None
        if cond_then is not None:
            parsed_then_guard = _parse_condition(cond_then, base_env, vars_map)
            if parsed_then_guard is None:
                self.last_reason = "if_guard_parse_failed"
                return None
        for tag, guard_raw, block in branches:
            env = dict(base_env)
            parsed = _parse_assignments(block, env)
            if parsed is None:
                self.last_reason = f"unsupported_statement_in_{tag}"
                return None
            updates, assigned = parsed
            for name in updates:
                all_vars.setdefault(name, LinearExpr({name: 1}, 0))
            vars_map = {name: z3.Int(name) for name in all_vars}
            if guard_raw in {"true", "1"}:
                guard_expr = z3.BoolVal(True)
            elif guard_raw is None:
                guard_expr = z3.Not(parsed_then_guard) if parsed_then_guard is not None else None
            else:
                guard_expr = _parse_condition(guard_raw, base_env, vars_map)
            if guard_expr is None:
                self.last_reason = f"guard_parse_failed_{tag}"
                return None
            precond_parts = [loop_guard, guard_expr] + inv_exprs
            precond = z3.And(*precond_parts) if precond_parts else z3.BoolVal(True)
            check = z3.Solver()
            check.add(precond)
            if check.check() == z3.sat:
                sat_branches.append(tag)
            branch_updates.append((guard_raw, tag, updates))

        if not sat_branches:
            self.last_reason = "no_satisfiable_branch"
            return None

        solver = z3.Solver()
        coeffs: Dict[str, Dict[str, z3.Int]] = {}
        consts: Dict[str, z3.Int] = {}
        for tag in [b[1] for b in branch_updates]:
            coeffs[tag] = {}
            for name in vars_map:
                coeffs[tag][name] = z3.Int(f"a_{tag}_{name}")
                solver.add(coeffs[tag][name] >= -self.max_coeff, coeffs[tag][name] <= self.max_coeff)
            consts[tag] = z3.Int(f"b_{tag}")
            solver.add(consts[tag] >= -self.max_coeff * 5, consts[tag] <= self.max_coeff * 5)

        for guard_raw, tag, updates in branch_updates:
            if tag not in sat_branches:
                continue
            if guard_raw in {"true", "1"}:
                guard_expr = z3.BoolVal(True)
            elif guard_raw is None:
                guard_expr = z3.Not(parsed_then_guard) if parsed_then_guard is not None else None
            else:
                guard_expr = _parse_condition(guard_raw, base_env, vars_map)
            if guard_expr is None:
                continue
            precond_parts = [loop_guard, guard_expr] + inv_exprs
            precond = z3.And(*precond_parts) if precond_parts else z3.BoolVal(True)

            prime_vars = {name: z3.Int(f"{name}_next") for name in vars_map}
            update_constraints = []
            for name in vars_map:
                if name in updates:
                    update_constraints.append(prime_vars[name] == _linear_to_z3(updates[name], vars_map))
                else:
                    update_constraints.append(prime_vars[name] == vars_map[name])
            precond = z3.And(precond, *update_constraints)

            f_curr = z3.IntVal(0)
            f_next = z3.IntVal(0)
            for name in vars_map:
                f_curr += coeffs[tag][name] * vars_map[name]
                f_next += coeffs[tag][name] * prime_vars[name]
            f_curr += consts[tag]
            f_next += consts[tag]
            if self.require_nonneg:
                solver.add(z3.Implies(precond, f_curr >= 0))
            solver.add(z3.Implies(precond, f_next <= f_curr - 1))

        if solver.check() != z3.sat:
            self.last_reason = "no_model"
            return None
        model = solver.model()
        piece_exprs: Dict[str, LinearExpr] = {}
        for tag in coeffs:
            coeff_map = {}
            for name, var in coeffs[tag].items():
                val = model.evaluate(var, model_completion=True)
                coeff_map[name] = int(val.as_long())
            const_val = int(model.evaluate(consts[tag], model_completion=True).as_long())
            piece_exprs[tag] = LinearExpr(coeff_map, const_val)

        if cond_piece is None:
            return _format_linear(piece_exprs[branch_updates[0][1]])

        then_tag = branch_updates[0][1]
        else_tag = branch_updates[1][1] if len(branch_updates) > 1 else then_tag
        then_expr = _format_linear(piece_exprs[then_tag])
        else_expr = _format_linear(piece_exprs[else_tag])
        return f"(({cond_piece}) ? ({then_expr}) : ({else_expr}))"
