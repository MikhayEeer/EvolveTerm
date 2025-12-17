
```bash
Analysis of Loopy_summary_sample_updated.csv
============================================
Configuration: general
  Total Files: 822
  Termination Results Distribution:
    EMPTY: 1
    FALSE: 120
    TRUE: 361
    UNKNOWN: 340
  Files with Effective Invariants: 149 (18.13%)

Configuration: lasso
  Total Files: 822
  Termination Results Distribution:
    EMPTY: 9
    FALSE: 106
    TRUE: 353
    UNKNOWN: 354
  Files with Effective Invariants: 154 (18.73%)
```


```bash
Category,Total,ET_Verified,CPA_Verified,Both,ET_Only,CPA_Only
arrays,171,150,0,0,150,0
loop_invariants,529,254,68,50,204,18
recursive_functions,31,0,0,0,0,0
termination,312,110,51,21,89,30
```
- Arrays 领域的绝对统治力 (150 vs 0)

在 arrays 类别中，EvolveTerm 成功验证了 150 个案例，而 CPA-Lasso 为 0。
结论： 这证明了您的方案在处理涉及数组操作的循环终止性方面具有压倒性优势。传统工具（如 CPA-Lasso）往往难以处理数组属性（Array Properties）或需要极其复杂的抽象域，而 LLM 能够很好地理解数组索引变化的逻辑。
- Loop Invariants 的显著提升 (254 vs 68)

在核心的 loop_invariants 类别中，您的验证数量是 CPA-Lasso 的 3.7 倍。
ET Only (204): 有 204 个案例是 CPA-Lasso 无法处理但您解决了的。这通常包括非线性循环或复杂的条件更新。
- Termination (Crafted) 的优势 (110 vs 51)

在专门构造的终止性难题 (termination) 中，您的表现依然是 CPA-Lasso 的 2 倍以上。
- Recursive Functions 的挑战

目前双方在 recursive_functions 上均为 0。这可能是因为当前的 Pipeline 主要针对循环（Loop-based）设计，对于递归调用的栈深度分析可能需要专门的 Prompt 或逻辑支持。