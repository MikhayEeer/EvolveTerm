# SMT Ranking Synthesis Guide (Experimental)

This module tries to synthesize a piecewise linear ranking function with Z3.
It runs before SVMRanker/LLM and falls back automatically if parsing fails.
Final termination proof is checked by SeaHorn (Docker).

## When to use
- Simple `while` loops with a single `if/else` inside.
- Linear updates only: `x = x + 1`, `x = -1`, `x += y`, `x -= 2`, `x++`, `x--`.
- Guards with linear comparisons: `x < 10`, `x >= y`, `x != 0`, combined with `&&` or `||`.

## CLI usage
```bash
evolveterm analyze --code-file path/to/file.c --smt-synth
```

Batch:
```bash
evolveterm batch-analyze --input-dir path/to/dir --smt-synth
```

## Notes and limits
- Only the first `while (...) { ... }` in the loop snippet is parsed.
- Nested `if/else` or complex expressions may be rejected.
- Non-linear updates (e.g., `x = x * y`) are not supported.
- Invariants with `\old` or quantifiers are ignored by the SMT parser.
- The synthesized function is still verified by Z3/SeaHorn if enabled.

## Example (from your case)
```c
while (x != 0) {
    if (x < 10) x++; else x = -1;
}
```
This can be handled with a piecewise linear ranking function using the `x < 10` split.
