# RAG 模块独立运行指南

此文档介绍了如何使用独立的 `rag` 指令集来管理知识库、构建索引以及执行检索测试。这对于调试检索效果、预构建特定领域的知识库（如 `minirag`）非常有用。

## 1. 核心命令概览

所有命令均挂载在 `evolve_term rag` 子命令下：

```bash
# 查看状态
python -m src.evolve_term.cli rag status [--kb <path>]

# 添加文件到知识库
python -m src.evolve_term.cli rag add <files...> [--kb <path>] --label <terminating|non-terminating>

# 重建索引
python -m src.evolve_term.cli rag rebuild [--kb <path>]

# 执行检索测试
python -m src.evolve_term.cli rag search --file <query_file> [--kb <path>]
```

## 2. 操作示例：构建 Mini-RAG 数据集

以下步骤演示如何使用 `examples/minirag` 数据构建一个独立的知识库。

### 2.1 准备数据

即将在 `examples/minirag/minirag_kb.json` 创建知识库。

### 2.2 添加知识 (Ingestion)

将 `knowledge` 目录下的 C 文件添加到知识库：

```bash
# 假设当前在项目根目录
python -m src.evolve_term.cli rag add examples/minirag/knowledge/*.c --kb examples/minirag/minirag_kb.json --label terminating
```

*   该命令会自动调用 LLM 提取 Loop。
*   调用 Embedding 模型生成向量。
*   保存到 JSON 并自动构建 HNSW 索引（生成 `minirag_kb_index.bin`）。

### 2.3 验证状态

```bash
python -m src.evolve_term.cli rag status --kb examples/minirag/minirag_kb.json
```

应显示已添加的 Case 数量和索引状态。

### 2.4 测试检索 (Retrieval Test)

使用 `query` 目录下的文件测试命中率：

```bash
python -m src.evolve_term.cli rag search --file examples/minirag/query/nonlin_mod_term_2.c --kb examples/minirag/minirag_kb.json --top-k 3
```

这将输出最相似的 Top-3 案例，你可以据此评估：
1.  检索是否准确找到了语义相似的题目？
2.  Embedding 模型是否区分开了不同类型的循环？

## 3. 在 Pipeline 中使用自定义知识库

一旦验证通过，你可以在主分析流程中复用这个知识库：

```bash
python -m src.evolve_term.cli analyze --input examples/minirag/query/nonlin_mod_term_2.c --kb examples/minirag/minirag_kb.json
```

注意：`pipeline` 会自动根据传入的 `--kb` 路径寻找同名的 `_index.bin` 索引文件（代码已修复支持此特性）。

## 4. 常见问题

*   **路径问题**: 确保 `_index.bin` 和 JSON 文件在同一目录。
*   **LLM 依赖**: `add` 命令需要提取 Loops，因此依赖 `llm_config.json`。如果只想做纯文本检索（不提取 Loop），可后续扩展 `--no-use-loops` 参数（当前默认启用）。
