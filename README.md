# Developer's Todo Lists
- [ ] prompts与TermDatabase 进行组合优化
- [ ] 判断RAG的可用性
- [ ] 目前是text embedding，找到codeBERT的API服务
- [ ] 在linux部署环境，尝试跑通demo

> test dir git config set

# EvolveTerm

EvolveTerm 是一个面向 C 代码的终止性分析演示系统，通过 **LLM + RAG** 组合流程来判断目标程序是否会在有限步骤内结束。系统聚焦循环结构，不考虑数组、指针与并发等复杂语义，便于快速验证终止性思路与工作流。

## ✨ 核心能力一览

- **循环提炼**：LLM 根据 `prompts/loop_extraction.txt` 提取 C 代码中的 `for/while` 结构，并输出 JSON 列表；若 LLM 不可用，则退回正则启发式。  
- **相似案例检索**：使用 CodeBERT / StarCoder 等嵌入模型（通过 `config/embed_config.json` 配置）生成向量，基于 HNSW 索引 (`data/hnsw_index.bin`) 检索相似案例。  
- **LLM 预测**：结合候选案例与 `prompts/prediction.txt`，由 LLM 输出终止性标签、置信度与理由，失败时立即抛出异常。  
- **RAG 增量更新**：人工复审的典型案例通过 `review` 命令写回 `data/knowledge_base.json`，累积 **10** 个新增案例即触发一次 HNSW 全量重建。  
- **可追踪报告**：每次预测都会生成结构化报告 (`data/reports/report_*.json`)，便于审计与归档。

## 🧱 目录结构

```
config/                # LLM 与嵌入模型配置（可指向真实 API 或 mock）
data/                  # JSON 知识库、HNSW 索引、报告
prompts/               # 各模块使用的提示词（可直接编辑）
src/evolve_term/       # 核心 Python 包
tests/                 # 轻量单元测试（pytest）
pyproject.toml         # 依赖与入口脚本（Typer CLI）
```

## ⚙️ 环境与依赖

- Python ≥ 3.10（建议 3.11）  
- 依赖：`typer`, `rich`, `requests`, `pydantic`, `numpy`, `hnswlib`, `pytest`（dev）  
- Windows PowerShell 示例命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[test]
```

## 🔐 配置说明

### LLM (`config/llm_config.json`)

```json
{
	"provider": "mock",                // mock 或 real（HTTP）
	"endpoint": "https://.../complete",
	"api_key": "REPLACE_ME",
	"model": "code-termination-large",
	"payload_template": { "max_tokens": 512, "temperature": 0.0 }
}
```

### 嵌入 (`config/embed_config.json`)

```json
{
	"provider": "mock",                // mock / real
	"endpoint": "https://.../embeddings",
	"api_key": "REPLACE_ME",
	"model": "codebert-base",
	"dimension": 64,
	"payload_template": {}
}
```

- 当 provider = `mock` 时，系统会使用内置的确定性 mock，方便离线演示。  
- 当 provider ≠ `mock` 时，需保证 endpoint 可访问、API Key 可用；任一环节失败会以 `LLMUnavailableError` / `EmbeddingUnavailableError` 抛出。  
- 根据真实 API 返回结构，确保响应体中含 `embedding`（数组）或 `choices[].text` / `output` 字段。

## 🚀 运行 Demo

1. **准备种子知识库**：`data/knowledge_base.json` 已包含终止与非终止两个示例。
2. **构建 HNSW 索引**（首次运行必做）：

```powershell
evolveterm rebuild-index
```

3. **分析任意 C 文件**：

```powershell
evolveterm analyze --code-file .\examples\sample.c
```

输出将包含预测标签、置信度、引用案例表格，以及报告存储路径。

## 🔁 RAG 增量更新

人工复审通过 `review` 命令写入知识库：

```powershell
evolveterm review \
	--code-file .\examples\loop.c \
	--label terminating \
	--explanation "Loop counter strictly decreases"
```

- 每新增 1 个案例会将 `pending_since_rebuild` +1。  
- 当累积达到 10（可在 `KnowledgeBase(rebuild_threshold=10)` 调整）时，`ingest_reviewed_case` 自动触发 `hnsw_index.bin` 全量重建并将计数归零。  
- 未达阈值时，系统会调用 `hnswlib.resize_index` 并增量写入，保持在线检索。

## 🧠 约束与假设

- 仅考虑 `for`/`while` 循环；数组、指针以及并发语义的终止性暂不处理。  
- 嵌入与 LLM API 一旦不可用立即抛出自定义异常，便于外层监控。  
- 提示词统一放置在 `prompts/*.txt`，可直接编辑并热加载。  
- 知识库存储为可读 JSON，结合 `PendingReviewCase` 结构支持外部工具批量写入。

## 🧪 测试与验证

```powershell
pytest
```

- 当前提供 `tests/test_knowledge_base.py`，验证增量重建计数逻辑。  
- 可按需补充 e2e 测试（Mock LLM & Embedding）。

## 📄 报告与排错

- 预测报告位于 `data/reports/`，每个 JSON 文件包含 label、confidence、reasoning、引用案例等字段。  
- 若出现 `IndexNotReadyError`，请确认已运行 `rebuild-index` 且 `data/knowledge_base.json` 不为空。  
- 若 `KnowledgeBase` 写入失败，请检查 `data/` 的读写权限。

## 🔮 下一步可扩展点

- 接入真实 CodeBERT/StarCoder API，并引入批量嵌入流水线。  
- 针对不同循环形态调整提示词，或引入 AST 解析增强。  
- 增加 Web UI / VS Code 扩展，实现代码片段的即写即查。
