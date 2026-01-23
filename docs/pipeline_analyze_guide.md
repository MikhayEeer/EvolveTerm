# EvolveTerm Pipeline Analyze 指南

本手册详细介绍了如何使用 `evolve_term` 的核心命令 `analyze` 来执行端到端的代码终止性分析。该命令集成了代码特征提取、RAG 检索、LLM 推理、不变量生成、Ranking Function 合成以及形式化验证（SeaHorn）。

## 1. 快速开始

### 1.1 基础用法 (单文件)

运行以下命令对单个 C 文件进行分析：

```bash
python -m src.evolve_term.cli analyze --input examples/sample.c
```

**默认行为**：
*   自动生成报告文件，路径为 `results/reports/<filename>_<model>_<timestamp>.yml`。
*   使用默认的 LLM 配置 (`llm_config.json`)。
*   尝试检索知识库（如果有）。
*   执行 SeaHorn 验证（需 Docker 环境）。

### 1.2 指定输出路径

你可以使用 `--output` 或 `-o` 指定报告保存位置：

```bash
python -m src.evolve_term.cli analyze --input examples/sample.c --output my_results/sample_report.yml
```

### 1.3 批量分析

如果 `--input` 指定的是目录，程序会自动进入批量模式：

```bash
python -m src.evolve_term.cli analyze --input examples/miniaevalterm/ --recursive
```

*   `--recursive` (`-r`): 递归查找子目录中的 `.c` 文件。
*   批量模式下，会在控制台输出进度条和汇总表格。

## 2. 核心参数详解

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--input` | **[必选]** 输入文件或目录路径。 | `--input ./tests` |
| `--svm-ranker` | 指定 SVMRanker 工具的根目录路径。启用后会尝试使用 SVM 合成 Ranking Function。 | `--svm-ranker /path/to/SVMRanker` |
| `--use-rag-reasoning` | 是否使用 RAG 检索参考文献来辅助推理（默认开启）。使用 `--no-rag-reasoning` 关闭。 | `--no-rag-reasoning` |
| `--kb` | 指定自定义知识库 JSON 文件路径。 | `--kb data/my_kb.json` |
| `--verifier` | 指定验证后端，目前主要是 `seahorn`。 | `--verifier seahorn` |
| `--seahorn-image` | 指定 SeaHorn 的 Docker 镜像标签。 | `--seahorn-image seahorn/seahorn-llvm14:nightly` |
| `--smt-synth` | 启用实验性的 SMT 分段线性 Ranking Function 合成 (Python Z3 实现)。 | `--smt-synth` |
| `--known-terminating` | 提示工具该程序已知是终止的（用于某些 Benchmarking 场景的先验）。 | `--known-terminating` |

## 3. Pipeline 工作流程

执行 `analyze` 时，系统按以下顺序处理数据：

1.  **Translation (可选)**: 如果启用 `-t`，将非 C 代码翻译为 C/C++。
2.  **Extraction**: 解析代码，提取循环结构和特征（使用 Tree-sitter 或 Regex）。
3.  **Embedding & Retrieval**:
    *   计算代码/循环的 Embedding。
    *   在知识库中检索相似案例 (Top-K)。
4.  **Reasoning (LLM)**:
    *   结合检索到的案例（Few-Shot Context）。
    *   推理程序的终止性标签。
    *   推理循环不变量 (Invariant)。
    *   推理 Ranking Function 模板。
    *   结合 SVMRanker (如果配置) 生成具体的 Ranking Function。
5.  **Verification**:
    *   **Invariants**: 使用 SeaHorn 验证生成的不变量是否成立。
    *   **Termination**: 使用 SeaHorn 验证程序是否终止 (目前主要侧重于不变量验证)。
6.  **Reporting**:生成详细的 YAML 报告。

## 4. 报告解读

生成的 YAML 报告包含丰富的信息，是分析的核心产出。主要字段如下：

```yaml
stages: [...]         # 经历的 pipeline 阶段
input_file: ...       # 输入文件路径
prediction:
  label: terminating  # LLM 预测的标签
  reasoning: ...      # 上下文推理过程
  invariants:         # 提取/生成的不变量列表
    - "i < n"
  ranking_function: "n - i" # 生成的 Ranking Function
references:           # RAG 检索到的参考案例
  - case_id: ...
    metadata:
      similarity: 0.95
verification:
  status: Verified    # 验证结果 (Verified / Failed / Unknown)
  details: ...
result:               # 最终汇总
  success: true
```

## 5. 常见问题

*   **SeaHorn 报错**: 确保本地安装了 Docker 且运行中，可以通过 `docker pull seahorn/seahorn-llvm14:nightly` 预下载镜像。如果不希望运行验证，目前代码逻辑通常会从 `llm_config` 或默认设置中读取，暂无直接 CLI 参数完全禁用验证（可通过设置超短 timeout 或在无 docker 环境下运行跳过）。
*   **SVMRanker 找不到**: 确保传入的路径包含 `src/CLIMain.py` 等核心文件。

## 6. 使用建议

*   对于**线性循环**问题，推荐开启 `--svm-ranker`（需自行部署该模块）。
*   对于**非线性/复杂逻辑**，主要依赖 LLM 的推理能力（`--use-rag-reasoning`），可以尝试调整 `top_k` (默认 5) 来引入更多参考案例。 
