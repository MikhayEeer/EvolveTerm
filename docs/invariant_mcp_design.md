# 基于 YAML 的 invariant MCP 设计

## 结论

统一 YAML 输入输出是数据契约，不等同于 MCP。

如果这里的 MCP 指的是 Model Context Protocol，那么代码需要提供一个 MCP server，
通过 JSON-RPC 2.0 暴露 tools/resources/prompts 等能力。YAML 应该作为 MCP tool
的输入输出 payload，以及 MCP resource 暴露的 schema/template，而不是把 YAML
本身称为 MCP。

本仓库现在提供了一个轻量 MCP stdio server：

```bash
PYTHONPATH=src python -m invariant_module.mcp_server
```

打包安装后也可以使用：

```bash
evolveterm-invariant-mcp
```

## 设计分层

```text
MCP transport layer
  JSON-RPC 2.0 over stdio
  initialize / tools/list / tools/call / resources/list / resources/read

MCP capability layer
  tools:
    yaml.detect
    yaml.validate
    invariant.normalize
  resources:
    invariant://schema/extract
    invariant://schema/invariant
    invariant://schema/ranking
    invariant://schema/feature
    invariant://template/invariant
    invariant://module/flow

YAML contract layer
  extract YAML
  invariant YAML
  ranking YAML
  feature YAML

module implementation layer
  invariant_module.command
  invariant_module.predictor
  invariant_module.refinement_pipeline
  invariant_module.houdini
  invariant_module.agents
```

## MCP tools

### `yaml.detect`

输入：

```yaml
yaml_text: |
  task: invariant_inference
  invariants_result: []
path: optional_name_inv.yml
```

输出：

```yaml
yaml_type: invariant
source: optional_name_inv.yml
```

### `yaml.validate`

输入：

```yaml
yaml_type: invariant
strict: true
yaml_text: |
  source_file: example.c
  source_path: data/example.c
  task: invariant_inference
  command: evolveterm invariant --input data/example.c
  pmt_ver: refinement
  model: unknown
  time: "2026-04-21T00:00"
  has_extract: false
  invariants_result:
    - loop_id: 1
      code: "while (i < n) { i++; }"
      invariants:
        - "i <= n"
```

输出：

```yaml
valid: true
yaml_type: invariant
errors: []
warnings: []
```

### `invariant.normalize`

输入可以是 invariant YAML、ranking YAML 或直接的 invariant 列表。

输出统一为：

```yaml
loop_invariants:
  - loop_id: 1
    invariants:
      - "i <= n"
```

## MCP resources

资源用于向 MCP client 暴露上下文，而不是执行动作。

建议 URI 设计：

```text
invariant://schema/extract
invariant://schema/invariant
invariant://schema/ranking
invariant://schema/feature
invariant://template/invariant
invariant://module/flow
```

这些资源可以被 IDE、Agent、实验脚本读取，用于理解当前模块的标准 YAML
结构和 invariant pipeline。

## 客户端配置示例

典型 MCP host 可以配置一个 stdio server：

```json
{
  "mcpServers": {
    "evolveterm-invariant": {
      "command": "python",
      "args": ["-m", "invariant_module.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/EvolveTerm/src"
      }
    }
  }
}
```

如果已经安装项目包，可以改为：

```json
{
  "mcpServers": {
    "evolveterm-invariant": {
      "command": "evolveterm-invariant-mcp"
    }
  }
}
```

## 研究模块建议

后续 invariant 研究流程可以逐步扩展 MCP tools：

```text
invariant.generate_candidates
invariant.houdini_filter
invariant.refine_missing_constants
invariant.refine_boundary_openness
invariant.refine_control_flow
invariant.run_refinement_pipeline
```

但每个 tool 的输入输出仍建议保持 YAML/JSON 可序列化结构，避免把内部 Python
对象暴露到协议边界。
