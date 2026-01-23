# `inv_assume` 验证指导手册

本手册旨在指导如何使用 `src/inv_assume` 模块进行基于 AST 的不变量生成与注入，并使用 `examples/miniaevalterm` 中的样例进行测试。

## 1. 模块简介

`inv_assume` 是 EvolveTerm 项目中负责精准代码 Instrumentation 的核心模块。它利用 Tree-sitter 对 C 代码进行 AST 解析，精确定位循环位置（`while`, `for`, `do-while`），并在循环体内注入 `assume(...)` 语句以辅助验证器。

**核心优势**：
- **精准定位**：比基于正则的文本替换更稳健。
- **上下文感知**：能够提取完整的循环体作为 Prompt 上下文。
- **无损注入**：仅修改循环内部，不破坏原始代码结构。

## 2. 环境准备

确保项目根目录下已配置 `llm_config.json`，并且 Python 环境安装了必要的依赖（如 `tree-sitter`, `tree-sitter-c`）。

```bash
pip install tree-sitter tree-sitter-c
```

## 3. 基本使用流程

### 3.1 单文件运行

使用 `src.inv_assume.pipeline` 对单个 C 文件进行处理。

**命令格式**：
```bash
python -m src.inv_assume.pipeline <input_c_file>
```

**示例（使用 miniaevalterm）**：
```bash
# 使用默认策略 (simple)
python -m src.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c

# 使用两阶段策略 (2stage)
python -m src.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --strategy 2stage
```

该命令会生成 `examples/miniaevalterm/nonlin_div_term_1.c.instrumented.c`。

### 3.2 批量处理

如果需要处理整个目录，可以使用 `batch_runner.py`（需确保已实现或使用脚本循环调用）。目前推荐编写简单的 shell/python 脚本遍历目录：

```powershell
Get-ChildItem examples/miniaevalterm/*.c | ForEach-Object { python -m src.inv_assume.pipeline $_.FullName --strategy 2stage }
```

## 4. 验证注入结果

生成的 `.instrumented.c` 文件包含了 `// Header for verification` 和注入的 `assume(...)`。

**策略说明**：
- `simple` (默认): 一次性 Prompt 生成，适合简单情况。
- `2stage`: 迁移自 `llm2stage`，采用 "Atoms生成 -> Z3筛选 -> Invariant组合" 的两阶段流程，适合复杂的不变量（生成质量更高，但速度较慢）。

**手动检查**：
打开生成的文件，确认：
1. 文件头部是否包含 `#include <assert.h>` 或自定义的 `assume` 定义。
2. 循环内部（尤其是循环头之后代码执行前）是否插入了 `assume(generated_invariant);`。

**SeaHorn 验证**（可选）：
如果已安装 SeaHorn，可直接验证生成的文件：
```bash
sea pf examples/miniaevalterm/nonlin_div_term_1.c.instrumented.c --vac
```

## 5. Pipeline 原理详解

1.  **解析 (CParser)**：读取源码，构建 AST，遍历寻找 Loop 节点。计算注入点的字节偏移量。
2.  **生成 (InvariantGenerator)**：提取 Loop 节点的文本内容，构造 Prompt 请求 LLM 生成不变量（Loop Invariant）。
3.  **注入 (Injector)**：
    *   在源码头部添加 `extern void __VERIFIER_assume(int);` 等辅助定义。
    *   根据偏移量，将 `__VERIFIER_assume(invariant);` 插入到循环体起始位置。

## 6. 常见问题排查

*   **注入位置错误**：`tree-sitter` 解析的注入点通常是 Compound Statement (`{ ... }`) 的第一个大括号后。如果循环没有大括号（如单行 `while`），Parser 会尝试处理，但建议规范化代码。
*   **生成为空**：检查 `llm_config.json` 配置是否正确，或 Prompt 是否需要针对该类题目优化。

## 7. 实验记录 `examples/miniaevalterm`

在 `examples/miniaevalterm` 中，针对非线性（nonlin）题目，建议关注 LLM 是否生成了非线性不变量（如乘法、模运算关系）。

---
**附录：Pipeline 调用代码示例**

```python
from src.inv_assume.pipeline import ASTInstrumentationPipeline

pipeline = ASTInstrumentationPipeline()
result = pipeline.run("examples/miniaevalterm/nonlin_div_term_1.c")
print(result["code"]) # 打印处理后的代码
```
