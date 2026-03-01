# Developer's Todo Lists
- removed to Obsidian notes

# Base Operators

## virtual env
- create virtual env
```bash
python -m venv <venv_name>
```
- switch to virtual environment
```bash
source .venv_evolveterm/bin/activate
```
## dependency & install

- Python ≥ 3.10（建议 3.11）  
- 依赖：`typer`, `rich`, `requests`, `pydantic`, `numpy`, `hnswlib`, `pytest`（dev）  
- Windows PowerShell 示例命令：

```bash
pip install typer rich requests pydantic numpy hnswlib pytest
pip install pycparser pcpp z3-solver
```
```bash
pip install -e .[test]
```
- install project
```bash
pip install -e .
```

## unit test
- aeval: `4NestedWith2Variables_false-no-overflow.c`, a nice test c sample
```bash
python -m evolve_term.embeddings --help

evolveterm analyze --code-file data/aeval/c_bench_term/4NestedWith2Variables_false-no-overflow.c --no-rag-reasoning
```
- Batch analyze
```bash
python -m src.evolve_term.cli batch-analyze data/SVC25_cpython -m src.evolve_term.cli batch-analyze data/SVC25_c --no-rag-reasoning
```

- loopy dataset `benchmark23_conjunctive.c`
	Bench Invar `0 <= i <= 100, j==2*i`
```bash
evolveterm analyze --code-file ../TerminationDatabase/Datasets/Loopy_dataset_InvarBenchmark/loop_invariants/sv-benchmarks/loop-zilu/benchmark23_conjunctive.c --no-rag-reasoning
```
get result
```bash
Label: terminating
Reasoning: Verified ranking function: 100 - i. Explanation: i increases by 1 each iteration and is bounded 
above by 100, so 100 - i is non-negative and strictly decreases.
Verification: Verified
Ranking Function: 100 - i
Invariants:
  - j - 2*i == \old(j) - 2*\old(i)
  - i >= \old(i)
  - j >= \old(j)
```

# EvolveTerm

EvolveTerm 是一个面向 C 代码的终止性分析演示系统，通过 **LLM + RAG** 组合流程来判断目标程序是否会在有限步骤内结束。系统聚焦循环结构，不考虑数组、指针与并发等复杂语义，便于快速验证终止性思路与工作流。

## 核心能力一览

- **循环提炼**：LLM 根据 `prompts/loop_extraction.txt` 提取 C 代码中的 `for/while` 结构，并输出 JSON 列表；若 LLM 不可用，则退回正则启发式。  
- **相似案例检索**：使用 CodeBERT / StarCoder / text-embeddings-v4 等嵌入模型（通过 `config/embed_config.json` 配置）生成向量，基于 HNSW 索引 (`data/hnsw_index.bin`) 检索相似案例。  
- **LLM 预测**：结合候选案例与 `prompts/prediction.txt`，由 LLM 输出终止性标签、置信度与理由，失败时立即抛出异常。  
- **RAG 增量更新**：人工复审的典型案例通过 `review` 命令写回 `data/knowledge_base.json`，累积 **10** 个新增案例即触发一次 HNSW 全量重建。  
- **可追踪报告**：每次预测都会生成结构化报告 (`data/reports/report_*.json`)，便于审计与归档。

## 目录结构

```
config/                # LLM 与嵌入模型配置（可指向真实 API 或 mock）
data/                  # JSON 知识库、HNSW 索引、报告
prompts/               # 各模块使用的提示词（可直接编辑）
src/evolve_term/       # 核心 Python 包
tests/                 # 轻量单元测试（pytest）
pyproject.toml         # 依赖与入口脚本（Typer CLI）
```

- 数据流向
```mermaid
flowchart TD
    A[CLI analyze/review<br>src/evolve_term/cli.py] -->|传入源码/标签| B[TerminationPipeline<br>pipeline.py]
    B -->|可选翻译| C[CodeTranslator<br>translator.py<br>LLM(long-context)]
    B --> D[LoopExtractor<br>loop_extractor.py<br>LLM + 正则兜底]
    D --> E[EmbeddingClient<br>embeddings.py]
    E --> F[HNSWIndexManager<br>rag_index.py]
    F -->|case_id列表| G[KnowledgeBase<br>knowledge_base.py]
    G -->|引用案例+相似度| H[PromptRepository<br>prompts_loader.py]
    H --> I[LLMClient.complete<br>llm_client.py]
    I -->|JSON 预测| J[报告与日志写入<br>pipeline.py → data/reports & data/logs]
    B -->|review新增| G
    G <--> F

```


## Config配置说明

- `pricing_per_millon_tokens_RMB` 是一个估算
  同时考虑了输入和输出两个价格，直接相加得到了这个成本

### LLM (`config/llm_config.json`)

```json
{
	"provider": "mock",                // mock 或 real（HTTP）
	"baseurl": "https://.../complete",
	"api_key": "REPLACE_ME",
	"model": "code-termination-large",
	"payload_template": { "max_tokens": 512, "temperature": 0.0 }
}
```

### 嵌入 (`config/embed_config.json`)

```json
{
	"provider": "mock",                // mock / real
	"baseurl": "https://.../embeddings",
	"api_key": "REPLACE_ME",
	"model": "codebert-base",
	"dimension": 64,
	"payload_template": {}
}
```
- 根据真实 API 返回结构，确保响应体中含 `embedding`（数组）或 `choices[].text` / `output` 字段。

## Tag策略 模型路由Model Routing
为不同的LLM config设计tag属性，记录不同LLM的技能，依据技能形成集合；
路由决策模块，根据接下来的待办选定模块后，如果需要LLM，再判断LLM需要的一个稀疏技能矩阵[0.3,0.2,0.4,0.1]代表不同关注项的权重，
再根据稀疏技能矩阵得到不同LLM的评分，给到LLM的选型；

OpenAI/LangChain/LangGraph 都有类似的 "Model Routing"
本系统的Model Routing的依据是 "tag"


### Tag策略 的具体tag选型

```json
default
//成本
cheap / fast
//质量
better / long-context / reasoning
// task
code / content
math / symbolic / verification / formal / translation
// special
outdated
```

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

## SMT 排名函数合成（实验性）

启用 SMT 合成会在 SVMRanker/LLM 之前，尝试用 Z3 根据循环条件与分支结构合成分段线性秩函数。
仅支持常见的 while + if/else + 线性赋值模式；解析失败会自动回退原有流程。
终止性证明的最终验证使用 SeaHorn（Docker）。

```bash
evolveterm analyze --code-file data/aeval/c_bench_term/3pieces_Caterina_TACAS16.c --smt-synth --no-rag-reasoning
```

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

### 📦 批量预向量化

在系统上线前，可先对某个目录（如 `data/SVC25_c/`）做一次离线嵌入并写入 JSON：

```powershell
python -m evolve_term.embeddings --bulk \
	--source-dir data/SVC25_c \
	--output data/prebuilt_embeddings.json \
	--label unknown
```

输出 JSON 会记录 `cases`、`embedding_info`（provider/model/dimension）及时间戳，方便后续并入 `knowledge_base.json` 并重建 HNSW。

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

### Single CPAChecker Command
```bash
./bin/cpachecker --preprocess --timelimit 300\
    --config config/lassoRankerAnalysis.properties\
    --spec config/specification/TerminatingStatements.spc\
    --heap 32G --output-path Testoutputs/\
    ../TerminationDatabase/Datasets/Loopy_dataset_InvarBenchmark/loop_invariants/code2inv/23.c
```

## 📄 报告与排错

- 预测报告位于 `data/reports/`，每个 JSON 文件包含 label、reasoning、引用案例等字段。  
- 若出现 `IndexNotReadyError`，请确认已运行 `rebuild-index` 且 `data/knowledge_base.json` 不为空。  
- 若 `KnowledgeBase` 写入失败，请检查 `data/` 的读写权限。

## 🔮 下一步可扩展点

- 接入真实 CodeBERT/StarCoder API，并引入批量嵌入流水线。  
- 针对不同循环形态调整提示词，或引入 AST 解析增强。  
- [ ] 增加 Web UI / VS Code 扩展，实现代码片段的即写即查。
