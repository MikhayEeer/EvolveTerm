#!/usr/bin/env python3
import re
from pathlib import Path

root = Path("data/aeval_171term_regular")

# 只匹配：if (cond) { <空白> while (while_cond) { body } }
pattern = re.compile(
    r"""
(?P<indent>[ \t]*)if\s*\((?P<if_cond>[^)]*)\)\s*\{\s*
(?P<indent2>[ \t]*)while\s*\((?P<while_cond>[^)]*)\)\s*\{
(?P<body>[^{}]*?)
\}
[ \t]*\}
""",
    re.VERBOSE | re.DOTALL,
)

def rewrite(text: str):
    def repl(m: re.Match):
        indent = m.group("indent")
        inner_indent = m.group("indent2") or indent
        if_cond = m.group("if_cond").strip()
        while_cond = m.group("while_cond").strip()
        body = m.group("body").rstrip()
        body = body + "\n" if body else ""
        return f"{indent}while ({while_cond} && ({if_cond})) {{\n{body}{indent}}}"
    return pattern.sub(repl, text)

if __name__ == "__main__":
    for path in root.rglob("*.c"):
        original = path.read_text()
        updated = rewrite(original)
        if updated != original:
            path.write_text(updated)
            print(f"modified: {path}")
