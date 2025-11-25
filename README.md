# Developer's Todo Lists
- [x] 目前是text embedding，找到codeBERT的API服务-1114
    **codebert/starcoder 只能本地部署**
    可以使用**text_embedding**先行替代
- [x] 在linux部署环境，尝试跑通demo
	- [x] 运行embeddings.py和llm_client.py的自测demo-1117
	- [x] 选取少量代码，测试embedding-1117
	- [x] 测试RAG的索引构建
	- [x] 选取sample，测试RAG的命中
- [x] 判断RAG的可用性
- [ ] prompts与TermDatabase 进行组合优化
- [x] 完成测试一轮流程，然后写软著
- [ ] 用UniTerm再去写一个软著
- [ ] 现有的RAG存储完整代码，让Prompt上下文太长
  - [ ] AST的方案？存储代码树？简化代码再向量化
  - [ ] 采纳CodeBERT？CodeBert对于同结构不同变量命名或者相似控制流的代码的向量化有没有优势
- [ ] 现有的提炼模块是基于规则的，可以看看还有什么方法
- [ ] 考虑引入路由模块了
- [x] 多个LLM的config，让路由模块可以选择
	- [ ] 完成了初版，预留了tag入口，后续的话，应该固定tag的种类数量；然后路由模块从多个已知tag选取需要的tag，再根据这个tag去选用LLM
	- [ ] tag的选用，可以让路由模块多维度进行决策(成本,时间，效果)等，组成稀疏向量，然后稀疏向量和LLM config得到这个LLM在这次路由决策的评分，从而决定本轮选用的LLM
- [ ] 一个可以开启的泛化语言模块
	- [ ] 先整理不同LLM的上下文窗口长度，并且按照目前标签策略，来完善标签
	- [ ] 然后按照这些标签，设计语言翻译模块(long-content)
- [x] 如何实现一个git检查脚本，使得提交前确认config json里面的api key不是完整的，可以不检查ignore的json

## virtual env
- create virtual env
```bash
python -m venv <venv_name>
```
- switch to virtual environment
```bash
source evolveterm/bin/activate
```
- install project
```bash
pip install -e .
```

## unit test
```bash
python -m evolve_term.embeddings --help
```

## Test Demo Output
```bash
evolveterm analyze --code-file data/SVC25_c_aug/Fibonacci04_aug3.c --top-k 3
───────────────────────────────────────────────────────── Prediction ──────────────────────────────────────────────────────────
Label: terminating 
Reasoning: The function `fib` is recursive but only called with inputs `val <= 46` due to the guard in `main`. Since `fib`     
decreases its argument on each recursive call and has base cases for `num < 1` and `num == 1`, all recursive calls eventually  
terminate.
Report saved at: /mnt/d/Users/mikhayeeer/Documents/Repos/EvolveTerm/data/reports/report_58d1cf2115704203b3fc1ab20a75d5f4.json  
        Referenced cases        
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Case ID ┃ Label ┃ Similarity ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
└─────────┴───────┴────────────┘
```

# EvolveTerm

EvolveTerm 是一个面向 C 代码的终止性分析演示系统，通过 **LLM + RAG** 组合流程来判断目标程序是否会在有限步骤内结束。系统聚焦循环结构，不考虑数组、指针与并发等复杂语义，便于快速验证终止性思路与工作流。

## 核心能力一览

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

- 当 provider = `mock` 时，系统会使用内置的确定性 mock，方便离线演示。  
- 当 provider ≠ `mock` 时，需保证 baseurl 可访问、API Key 可用；任一环节失败会以 `LLMUnavailableError` / `EmbeddingUnavailableError` 抛出。  
- 根据真实 API 返回结构，确保响应体中含 `embedding`（数组）或 `choices[].text` / `output` 字段。

## Tag策略 模型路由Model Routing
为不同的LLM config设计tag属性，记录不同LLM的技能，依据技能形成集合；
路由决策模块，根据接下来的待办选定模块后，如果需要LLM，再判断LLM需要的一个稀疏技能矩阵[0.3,0.2,0.4,0.1]代表不同关注项的权重，
再根据稀疏技能矩阵得到不同LLM的评分，给到LLM的选型；

OpenAI/LangChain/LangGraph 都有类似的 "Model Routing"
本系统的Model Routing的依据是 "tag"


## Tag策略 的具体tag选型

```json
default
//成本
cheap / fast
//质量
better / long-context / reasoning
// task
code / content
math / symbolic / verification / formal / translation
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

## 📄 报告与排错

- 预测报告位于 `data/reports/`，每个 JSON 文件包含 label、reasoning、引用案例等字段。  
- 若出现 `IndexNotReadyError`，请确认已运行 `rebuild-index` 且 `data/knowledge_base.json` 不为空。  
- 若 `KnowledgeBase` 写入失败，请检查 `data/` 的读写权限。

## 🔮 下一步可扩展点

- 接入真实 CodeBERT/StarCoder API，并引入批量嵌入流水线。  
- 针对不同循环形态调整提示词，或引入 AST 解析增强。  
- 增加 Web UI / VS Code 扩展，实现代码片段的即写即查。
