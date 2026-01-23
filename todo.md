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

## 4. 数据集准备: `examples/minirecur/`
**难度评估**: <font color="green">Low</font>

**任务描述**:
为任务 1 准备典型的递归程序集。

**计划包含的程序 (6个)**:
1.  `fib_tail.c`: 斐波那契数列（尾递归版）。
2.  `fact_tail.c`: 阶乘（尾递归版）。
3.  `gcd_tail.c`: 最大公约数（欧几里得算法，天然尾递归）。
4.  `sum_array.c`: 数组求和（指针移动递归）。
5.  `is_even.c`: 互递归示例（IsEven/IsOdd，挑战项）。
6.  `fib_non_tail.c`: 斐波那契（普通递归，作为负样本或挑战项）。

---

## 5. RAG 模块现状评估
**难度评估**: <font color="orange">Medium</font>

**任务描述**:
深入审查 `src/evolve_term/rag_index.py` 和 `knowledge_base.py`。

**检查点**:
*   **HNSW 索引**: `hnswlib` 的保存与加载机制是否健壮？维度是否与现在的 Embedding 模型匹配？
*   **数据一致性**: `data/knowledge_base.json` 与索引文件是否脱节？
*   **检索效果**: 对比简单的余弦相似度，HNSW 是否配置正确？

---

## 6. 数据集准备: `examples/minirag/`
**难度评估**: <font color="green">Low</font>

**任务描述**:
构建一个小型的、受控的测试集，用于证明 RAG 确实检索到了正确的参考代码。

**计划内容**:
*   **Knowledge Base**: 包含 3 个典型的终止模式（例如：简单计数器、嵌套循环、位操作）。
*   **Query Code**: 3 个与上述 KB 高度相似但在变量名/结构上略有变形的代码。
*   **预期**: 运行 Pipeline 后，验证 Report 中的 `neighbors` 字段是否精确命中了对应的 KB Case。

---

## 7. RAG Pipeline 验证说明
**难度评估**: <font color="green">Low</font>

**任务描述**:
编写文档或脚本，说明如何使用 Task 6 的数据来验证 Pipeline 中的 RAG 组件工作正常。

**验证方法**:
1.  构建小型 KB 索引。
2.  使用 `evolveterm analyze --use-rag-reasoning` 运行 Query。
3.  检查生成的 YAML Report，确认 `rag_retrieval` 阶段有输出，且 `reasoning` 中提到了参考案例。
4.  进行对比实验：`--no-rag-reasoning`，观察结果差异。

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
