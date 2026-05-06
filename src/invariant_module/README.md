# Invariant Module Flow

This module is organized around the invariant research pipeline below:

## 快速验证可用性

参见 [快速验证指南](../../docs/invariant_module_quickstart.md) 快速验证模块是否可用。

一键验证命令：
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python3 -m src.invariant_module.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/test --config llm_config.json --verify
```

## 基本使用方式

### 单文件处理
```bash
python3 -m src.invariant_module.inv_assume.pipeline <input.c> --output <output_dir> --config llm_config.json
```

### 批量处理
```bash
python3 -m src.invariant_module.inv_assume.pipeline <input_dir> --output <output_dir> --config llm_config.json --verify
```

### 可选参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--strategy` | 生成策略 (`simple` 或 `2stage`) | `simple` |
| `--verify` | 启用 SeaHorn 验证 | 否 |
| `--seahorn-timeout` | SeaHorn 超时秒数 | 60 |

## 已验证环境版本

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | >= 3.10 | ✓ |
| tree-sitter | 0.25.2 | ✓ |
| tree-sitter-c | 0.24.2 | ✓ |
| hnswlib | 0.8.0 | ✓ |
| Docker SeaHorn | seahorn-llvm14:nightly | ✓ |

验证日期：2026-05-06

---

## Pipeline 详细流程

1. Two-stage LLM candidate generation
   - `inv_assume/strategies/two_stage.py`
   - `TwoStageStrategy.generate_candidates(code)` returns a candidate invariant set.
   - `TwoStageStrategy.generate(code)` is kept for backward compatibility and returns the first candidate.

2. Houdini filtering
   - `houdini.py`
   - `HoudiniFilter` implements the iterative remove-refuted-candidates loop.
   - The verifier/checker is injected, so the same orchestration can use SeaHorn, Z3, Boogie, or a future custom checker.

3. Multi-agent refinement network
   - `agents.py`
   - `MissingConstantAgent`: checks missing constants and bounds.
   - `BoundaryOpennessAgent`: checks open/closed boundary operators, such as `>` vs `>=`.
   - `ControlFlowCoverageAgent`: checks whether invariants cover all relevant control-flow paths.

4. Final orchestration
   - `refinement_pipeline.py`
   - `InvariantRefinementPipeline` runs:
     `two-stage candidates -> Houdini -> missing constants agent -> boundary agent -> control-flow agent -> optional final Houdini`.

The current Houdini implementation is verifier-agnostic. To make it fully semantic,
provide a `HoudiniChecker` that checks the current invariant set and returns which
candidate invariants were refuted.
