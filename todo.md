# EvolveTerm Project Roadmap & Todo List

## 1. 尾递归转循环与等价性验证 (Tail Recursion to Loop Transformation)
**难度评估**: <font color="red">High</font>
**任务描述**:
结合 LLM 的语义理解能力和传统静态分析（栈分析），将 C 语言中的尾递归函数重写为迭代循环形式。更为关键的是，需要提供一种自动化的验证方案，证明转换后的非递归程序与原递归程序在功能上等价。

**技术难点**:
1.  **代码重写**: 需要精准识别尾递归模式（Accumulator style）并安全地转换为 `while` 循环。
2.  **等价性证明**: 这是一个程序等价性检查（Translation Validation）问题。简单的输入输出测试（IO Testing）是不够的，需要使用形式化验证工具（如 SeaHorn）。
3.  **验证策略**: 构建一个 "Dual Harness"，即编写一个 `main` 函数同时调用 `original(x)` 和 `transformed(x)`，并断言 `sassert(original(x) == transformed(x))`，交给 SeaHorn 验证。

**执行计划**:
*   [ ] **Step 1**: 创建 `src/evolve_term/transform/recursion.py` 模块。
*   [ ] **Step 2**: 设计 Prompt 让 LLM 执行代码重写。
*   [ ] **Step 3**: 编写自动 Harness 生成器，将两份代码合并供 SeaHorn 验证。
*   [ ] **Step 4**: 在 `examples/minirecur/` 上进行测试。

---

## 2. 评估 `llm2stage-invariant` 的价值 (已完成)
**难度评估**: <font color="green">Low</font>

**状态**: [x] Completed (Migrated & Deleted)

**执行结果**:
*   确认 `llm2stage-invariant` 包含有价值的 "2-Stage (Atom -> Candidate)" 生成策略。
*   **迁移**: 已将相关逻辑迁移至 `src/inv_assume/strategies/two_stage.py` 和 `src/inv_assume/utils.py`。
*   **清理**: 已删除旧的 `src/llm2stage-invariant/` 目录。
*   **集成**: `src/inv_assume` 现支持 `--strategy 2stage` 参数。

---

## 3. `inv_assume` 与 Pipeline 使用指南 (已完成)
**难度评估**: <font color="orange">Medium</font>

**状态**: [x] Completed

**产出文档**:
1.  **`docs/inv_assume_guide.md`**: 详细介绍了基于 AST 的不变量注入工具 `inv_assume` 的安装、单文件运行、批量运行及策略选择 (Simple vs 2Stage)。
2.  **`docs/pipeline_analyze_guide.md`**: 详细介绍了核心命令 `evolve_term analyze` 的全链路使用，包括 Translation, RAG, Reasoning, Ranking Function 到 Verification 的完整流程。

---

## 4. RAG 增强：多维特征存储与检索架构
**难度评估**: <font color="red">High</font>

**任务描述**:
重构 RAG 存储结构，使其支持代码源码、摘要、特征、不变量、秩函数等多维信息的存储与检索。

**改造计划**:
*   [ ] **Step 1: 模型重构**: 更新 `models.py/KnowledgeCase`，增加 `summary`, `features`, `invariants`, `ranking_function` 字段。
*   [ ] **Step 2: 提取器解耦**: 从 `commands/feature.py` 中拆解出独立的 `FeatureExtractor` 类，支持输入源码返回特征字典（含 LLM Summary）。
*   [ ] **Step 3: 增强 Ingestion**: 改造 `rag add` 命令，集成 FeatureExtractor，并增加可选参数支持调用 Predictor 填充 Invariants/RF。
*   [ ] **Step 4: 多维 Embedding**: 设计新的 Embedding 文本构造策略（如 `[Summary] ... [Features] ... [Code] ...`），并在 `pipeline.py` 中更新检索逻辑。
*   [ ] **Step 5: 验证**: 使用 `minirag` 数据集验证多维检索的准确性提升。

### 推荐改造路径
推荐改造路径：

第一阶段: 仅集成 "Summary" 和 "Features"（利用现有 feature 模块）。这能极大提升“意图检索”的能力。
第二阶段: 增加不变量和秩函数的可选填充。不强制每次 add 都生成，而是作为一个额外的 --enrich 选项。

---

## 5. RAG 模块现状评估 (已并入 Task 4)
**状态**: [x] Merged into Task 4

---

## 6. 数据集准备: `examples/minirag/` (已完成)
**状态**: [x] Completed

**执行结果**:
*   已在 `examples/minirag` 下建立了 `knowledge` 和 `query` 目录。
*   包含 3 个非线性终止案例作为知识库，1 个案例作为查询。
*   已编写 `docs/rag_module_guide.md` 指导如何构建和测试。

---

## 7. RAG Pipeline 验证说明 (已完成)
**状态**: [x] Completed

**执行结果**:
*   相关指南已包含在 `docs/rag_module_guide.md` 中。
*   实现了独立的 `rag` CLI 命令组（status, add, rebuild, search）方便独立验证。

---

## 8. Subagent 架构重构 (Piecewise 等)
**难度评估**: <font color="red">High</font> (涉及核心架构调整)

**任务描述**:
目前的 `predict.py` 中，`infer_ranking`、`infer_piecewise_predicates` 等方法实质上是在执行复杂的甚至多轮的 LLM 交互。

**提案**:
*   引入 `SubAgent` 概念（或类）。
*   **特征**: 拥有独立的 Prompt 模板、独立的重试/修正逻辑、甚至独立的 Memory。
*   **实现**: 从 `APILLMClient` 之上抽象出一层 `Agent`，例如 `PiecewiseRankingAgent`。
*   **解耦**: `Pipeline` 不再直接操作 Prompt 细节，而是协调不同的 Agents。

---

## 9. 潜在的 Subagent 候选模块
**难度评估**: <font color="orange">Medium</font> (设计层面)

**分析**:
如果采用 Subagent 架构，以下模块也应独立：
1.  **InvariantRefinementAgent**: 不仅仅是生成一次，而是根据 Verifier 的反馈（Counter-example）进行多轮修正（CEGIS 循环）。
2.  **LoopExtractionAgent**: 当前的循环提取虽然有 Regex fallback，但复杂宏定义下的循环提取可能需要 LLM 的多轮确认。
3.  **TranslationAgent**: 从 Python/Java 到 C 的翻译，如果一次失败，需要根据报错信息自我修正。
4.  **RequirementAnalystAgent**: (未来扩展) 分析代码所属的领域（如驱动程序、数学计算），动态选择最合适的 Prompt 策略。
