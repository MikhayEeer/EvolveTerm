from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import z3

# --- Regex Patterns ---
_INVALID_EXPR_TOKENS = (
    "\\forall", "\\exists", "\\valid", "\\at", "\\sum", "\\old", "==>", "=>",
)
_FUNC_CALL_RE = re.compile(r"\b[A-Za-z_]\w*\s*\(")
_BOOL_OP_RE = re.compile(r"(==|!=|<=|>=|<|>|\&\&|\|\|)")
_VAR_MUL_VAR_RE = re.compile(r"\b[A-Za-z_]\w*\b\s*\*\s*\b[A-Za-z_]\w*\b")
_OLD_VALUE_VAR_RE = re.compile(r"\bold_[A-Za-z_]\w*\b")

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result

# --- Syntax Filtering ---
def syntax_filter(
    exprs: List[str],
    modified_vars: List[str],
    require_modified_var: bool = True,
) -> Tuple[List[str], List[Dict[str, str]]]:
    accepted: List[str] = []
    dropped: List[Dict[str, str]] = []

    for expr in exprs or []:
        if expr is None: continue
        expr_str = str(expr).strip()
        if not expr_str: continue
        if expr_str.endswith(";"): expr_str = expr_str[:-1].strip()
        
        # Check invalid tokens
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

    return dedupe_preserve_order(accepted), dropped

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
        if prev < 0: return True
        if expr[prev] in "([=,+-*/%<>&|!^~?:;": return True
        i += 1
    return False

def _is_tautology_pattern(expr: str) -> bool:
    compact = re.sub(r"\s+", "", expr)
    if re.search(r"\b([A-Za-z_]\w*)==\1\b", compact): return True
    if re.search(r"\b(\d+)==\1\b", compact): return True
    return False

def extract_modified_vars(code: str) -> List[str]:
    # Simplified version for now - relying on user context or simple regex
    # Ideally should use TreeSitter since we have it wrapper in inv_assume
    # But for now, let's use the regex based one from llm2stage for compatibility
    cleaned = _strip_c_comments_and_strings(code)
    assign_op = r"(?:\+=|-=|\*=|/=|%=|(?<![=!<>])=(?![=]))"
    patterns = [
        re.compile(rf"\b([A-Za-z_]\w*)\s*{assign_op}"),
        re.compile(r"\b([A-Za-z_]\w*)\s*(?:\+\+|--)"),
        re.compile(r"(?:\+\+|--)\s*([A-Za-z_]\w*)"),
    ]
    seen = set()
    ordered = []
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered

def _strip_c_comments_and_strings(code: str) -> str:
    # A simplified stripper needed for var extraction
    code = re.sub(r'//.*', '  ', code)
    code = re.sub(r'/\*.*?\*/', '  ', code, flags=re.DOTALL)
    return code

# --- Z3 Filtering ---

class ParseError(Exception):
    pass

@dataclass
class ArithNode:
    expr: z3.ArithRef
    kind: str  # const | var | expr

@dataclass(frozen=True)
class Token:
    kind: str
    value: Any

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

    def _peek(self) -> Token: return self.tokens[self.pos]
    def _advance(self) -> Token:
        token = self.tokens[self.pos]; self.pos += 1; return token
    def _match(self, kind: str, value: Optional[str] = None) -> bool:
        token = self._peek()
        if token.kind != kind: return False
        if value is not None and token.value != value: return False
        self._advance(); return True

    def _parse_or(self) -> z3.BoolRef:
        left = self._parse_and()
        while self._match("OP", "||"):
            left = z3.Or(left, self._parse_and())
        return left

    def _parse_and(self) -> z3.BoolRef:
        left = self._parse_not()
        while self._match("OP", "&&"):
            left = z3.And(left, self._parse_not())
        return left

    def _parse_not(self) -> z3.BoolRef:
        if self._match("OP", "!"): return z3.Not(self._parse_not())
        if self._peek().kind == "(" and self._looks_like_boolean_group():
            self._advance()
            expr = self._parse_or()
            if not self._match(")", ")"): raise ParseError("missing )")
            return expr
        return self._parse_relation()

    def _looks_like_boolean_group(self) -> bool:
        if self._peek().kind != "(": return False
        # simple heuristic
        return True

    def _parse_relation(self) -> z3.BoolRef:
        left = self._parse_arith()
        token = self._peek()
        if token.kind == "OP" and token.value in ("==", "!=", "<=", ">=", "<", ">"):
            op = token.value; self._advance(); right = self._parse_arith()
            if op == "==": return left.expr == right.expr
            if op == "!=": return left.expr != right.expr
            if op == "<=": return left.expr <= right.expr
            if op == ">=": return left.expr >= right.expr
            if op == "<":  return left.expr < right.expr
            if op == ">":  return left.expr > right.expr
        raise ParseError("missing comparison operator")

    def _parse_arith(self) -> ArithNode:
        node = self._parse_term()
        while True:
            if self._match("OP", "+"): node = ArithNode(node.expr + self._parse_term().expr, "expr")
            elif self._match("OP", "-"): node = ArithNode(node.expr - self._parse_term().expr, "expr")
            else: break
        return node

    def _parse_term(self) -> ArithNode:
        node = self._parse_factor()
        while self._match("OP", "*"):
            rhs = self._parse_factor()
            # Nonlinear check could be here
            node = ArithNode(node.expr * rhs.expr, "expr") 
        return node

    def _parse_factor(self) -> ArithNode:
        if self._match("OP", "+"): return self._parse_factor()
        if self._match("OP", "-"): return ArithNode(-self._parse_factor().expr, "expr")
        token = self._peek()
        if token.kind == "INT": self._advance(); return ArithNode(z3.IntVal(int(token.value)), "const")
        if token.kind == "IDENT":
            self._advance(); name = str(token.value)
            var = self.var_map.get(name)
            if not var: var = z3.Int(name); self.var_map[name] = var
            return ArithNode(var, "var")
        if self._match("(", "("):
            node = self._parse_arith(); 
            if not self._match(")", ")"): raise ParseError("missing )")
            return node
        raise ParseError(f"unexpected token: {token.kind}")

def tokenize(expr: str) -> List[Token]:
    # Simplified tokenizer for brevity
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace(): i += 1; continue
        if expr[i:i+2] in ("&&","||","==","!=","<=",">="): tokens.append(Token("OP", expr[i:i+2])); i+=2; continue
        if expr[i] in "()+-*<>!": tokens.append(Token(expr[i] if expr[i] in "()" else "OP", expr[i])); i+=1; continue
        if expr[i].isdigit():
            j = i; 
            while j < len(expr) and expr[j].isdigit(): j += 1
            tokens.append(Token("INT", int(expr[i:j]))); i = j
            continue
        if expr[i].isalpha() or expr[i] == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j]=='_'): j+=1
            tokens.append(Token("IDENT", expr[i:j])); i=j
            continue
        raise ParseError(f"Unexpected char: {expr[i]}")
    tokens.append(Token("EOF", None))
    return tokens

def z3_filter(exprs: List[str], max_keep: int = 10, timeout_ms: int = 1500) -> Tuple[List[str], Any]:
    # Simplified z3 filter logic
    kept = []
    translator_vars = {}
    
    # 1. Parse and Check consistency
    valid_candidates = []
    for expr in exprs:
        try:
            tokens = tokenize(expr)
            parser = ExprParser(tokens, translator_vars)
            z3_expr = parser.parse_bool()
            
            s = z3.Solver(); s.set("timeout", timeout_ms)
            s.add(z3_expr)
            if s.check() == z3.unsat: continue # Contradiction
            
            s.reset(); s.add(z3.Not(z3_expr))
            if s.check() == z3.unsat: continue # Tautology
            
            valid_candidates.append((expr, z3_expr))
        except:
            continue
            
    # 2. Implication filtering
    final_kept = []
    final_z3 = []
    for expr, z3_e in valid_candidates:
        if len(final_kept) >= max_keep: break
        
        # Check if implied by existing
        s = z3.Solver(); s.set("timeout", timeout_ms)
        if final_z3: s.add(z3.And(final_z3))
        s.add(z3.Not(z3_e))
        if s.check() == z3.unsat:
            continue # Implied by existing
            
        final_kept.append(expr)
        final_z3.append(z3_e)
        
    return final_kept, None
