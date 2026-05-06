# invariant_module 快速验证指南

本文档提供最小步骤，快速验证 `src/invariant_module/inv_assume` 模块可用。

## 一键验证

```bash
# 禁用代理后运行（socks代理不兼容）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 生成插桩代码 + SeaHorn 验证（一条命令完成）
python3 -m src.invariant_module.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/invariant_module_test --config llm_config.json --verify
```

**预期输出**:
```
Found 1 loops.
Generating invariant for loop at offset 129 using strategy 'simple'...
  -> Generated: y > 0 && x >= y
Instrumented code written to results/invariant_module_test/nonlin_div_term_1.c.instrumented.c
Verifying results/invariant_module_test/... with SeaHorn...
Verification result: safe (unsat)
```

---

## 分步验证（调试用）

### 步骤 1: 生成插桩代码

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python3 -m src.invariant_module.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/invariant_module_test --config llm_config.json
```

### 步骤 2: 查看生成的代码

```bash
cat results/invariant_module_test/nonlin_div_term_1.c.instrumented.c
```

输出示例：
```c
#ifndef _INJECTED_ASSUME_
#define _INJECTED_ASSUME_
extern void __VERIFIER_assume(int);
#define assume(X) __VERIFIER_assume(!!(X))
#endif

int main() {
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();

  while (x != y && y>0 && x/y>1) {
    assume(y > 0 && x >= y);  // ← LLM生成的循环不变量
    x--;
  }
}
```

### 步骤 3: SeaHorn 验证

```bash
docker run --rm -v "$(pwd)":/work -w /work \
  seahorn/seahorn-llvm14:nightly \
  sea pf results/invariant_module_test/nonlin_div_term_1.c.instrumented.c --vac
```

---

## 结果判读

| SeaHorn 结果 | 含义 |
|-------------|------|
| `unsat` | ✓ 验证通过，不变量成立 |
| `sat` | ✗ 发现反例，不变量不正确 |
| `unknown` | ? 无法证明（可能需要更强的不变量） |

---

## 环境依赖

### Python 版本

- **Python >= 3.10** (项目要求)

### Python 依赖

#### 核心依赖（项目级）
```bash
pip3 install typer rich numpy hnswlib openai z3-solver
```

#### invariant_module 特有依赖（C 解析 + AST）
```bash
pip3 install tree-sitter==0.25.2 tree-sitter-c==0.24.2
```

> **版本兼容性注意**: tree-sitter 和 tree-sitter-c 版本需匹配。已知组合：
> - tree-sitter 0.25.2 + tree-sitter-c 0.24.2 ✓
> - tree-sitter 0.24.0 + tree-sitter-c 0.24.2 ✗ (Language version 不兼容)

#### 一键安装全部依赖
```bash
pip3 install typer rich numpy hnswlib openai z3-solver tree-sitter==0.25.2 tree-sitter-c==0.24.2
```

### Docker 验证环境

SeaHorn 形式化验证工具（约 3.6GB）：
```bash
docker pull seahorn/seahorn-llvm14:nightly
```

验证 Docker 是否正常：
```bash
docker run --rm seahorn/seahorn-llvm14:nightly sea --version
```

### LLM 配置

需要有效的 `config/llm_config.json` 配置文件，包含：
- API Key (如 OpenAI、Qwen、GLM 等)
- Base URL
- 模型名称

检查配置：
```bash
cat config/llm_config.json
```

---

## 依赖检查脚本

运行以下命令快速检查环境：

```bash
# 检查 Python 版本
python3 --version

# 检查 Python 包
python3 -c "import tree_sitter, tree_sitter_c, hnswlib, openai, typer, rich; print('All Python dependencies OK')"

# 检查 Docker
docker --version && docker images | grep seahorn

# 检查 LLM 配置
test -f config/llm_config.json && echo "LLM config OK"
```

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Unknown scheme for proxy URL` | socks 代理不兼容 httpx | 禁用代理：`unset http_proxy https_proxy ...` |
| `Incompatible Language version 15` | tree-sitter 版本不匹配 | 升级 tree-sitter：`pip3 install tree-sitter==0.25.2` |
| `ModuleNotFoundError: hnswlib` | 缺少依赖 | 安装：`pip3 install hnswlib` |
| `Config file 'xxx' does not exist` | 配置路径错误 | 使用 `--config llm_config.json`（不带 `config/` 前缀） |
| Docker 无权限 | 用户未加入 docker 组 | `sudo usermod -aG docker $USER` 后重新登录 |
| 找不到文件 | 挂载路径错误 | 确保 `-v` 挂载路径与 `sea pf` 文件路径一致 |
| LLM API 调用失败 | 配置无效或网络问题 | 检查 `config/llm_config.json` 和网络连接 |

---

## 更多示例文件

```bash
ls examples/miniaevalterm/
```

可用测试文件：
- `nonlin_div_term_1.c` - 非线性除法（推荐首选）
- `nonlin_div_term_2.c` - 非线性除法变体
- `nonlin_mod_term_1.c` - 非线性取模
- `nonlin_mod_term_2.c` - 非线性取模变体
- `nonlin_mod_term_3.c` - 非线性取模变体
- `nonlin_jump_over_1_term.c` - 非线性跳转

---

## 批量处理

对整个目录进行分析：
```bash
python3 -m src.invariant_module.inv_assume.pipeline examples/miniaevalterm --output results/batch --config llm_config.json --verify
```

---

## 高级策略

使用 `2stage` 策略可生成更精确的不变量（更慢但质量更高）：
```bash
python3 -m src.invariant_module.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/invariant_module_test --config llm_config.json --strategy 2stage --verify
```

---

## 模块结构

```
src/invariant_module/
├── inv_assume/               # 核心插桩子模块
│   ├── pipeline.py           # 主流程入口
│   ├── c_parser.py           # C代码AST解析
│   ├── inv_generator.py      # LLM不变量生成
│   ├── injector.py           # 代码插桩
│   ├── verifier.py           # SeaHorn验证
│   └── strategies/           # 生成策略
│       └── two_stage.py      # 两阶段策略
├── agents.py                 # 多Agent细化网络
├── houdini.py                # Houdini过滤算法
├── refinement_pipeline.py    # 完整细化Pipeline
├── command.py                # CLI命令集成
├── predictor.py              # 不变量预测器
└── playground/               # 实验性比较脚本
```

---

## Pipeline 工作流程

执行 `inv_assume.pipeline` 时：

```
输入 C 代码 → AST解析循环 → LLM生成不变量 → 插入assume断言 → SeaHorn验证
```

完整 `InvariantRefinementPipeline` 流程：

```
Two-stage生成候选 → Houdini过滤 → MissingConstantAgent → BoundaryAgent → ControlFlowAgent → Final Houdini
```

---

## 已验证环境版本

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.10.x | ✓ |
| tree-sitter | 0.25.2 | ✓ |
| tree-sitter-c | 0.24.2 | ✓ |
| hnswlib | 0.8.0 | ✓ |
| openai | >= 1.42.0 | ✓ |
| Docker SeaHorn | seahorn-llvm14:nightly | ✓ |
| LLM | GLM-4.7 (default) | ✓ |

验证日期：2026-05-06