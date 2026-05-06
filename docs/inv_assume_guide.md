# `inv_assume` 验证指南

本指南说明如何使用 `src/inv_assume` 生成带 `assume(...)` 的插桩代码，并通过 Docker 版 SeaHorn 进行验证。

## 1. 环境准备

### 1.1 Python 版本

- **Python >= 3.10** (项目最低要求)

### 1.2 Python 依赖

#### 核心依赖（项目级）
```bash
pip3 install typer rich numpy hnswlib openai z3-solver
```

#### inv_assume 特有依赖（C 解析 + AST）
```bash
pip3 install tree-sitter==0.25.2 tree-sitter-c==0.24.2
```

> **版本兼容性注意**: tree-sitter 和 tree-sitter-c 版本必须匹配：
> - ✓ tree-sitter 0.25.2 + tree-sitter-c 0.24.2
> - ✗ tree-sitter 0.24.0 + tree-sitter-c 0.24.2 (Language version 15 vs 13-14 不兼容)

#### 一键安装全部依赖
```bash
pip3 install typer rich numpy hnswlib openai z3-solver tree-sitter==0.25.2 tree-sitter-c==0.24.2
```

### 1.3 Docker 验证环境

SeaHorn 形式化验证工具（约 3.6GB）：
```bash
docker pull seahorn/seahorn-llvm14:nightly
```

验证 Docker 是否正常：
```bash
docker run --rm seahorn/seahorn-llvm14:nightly sea --version
```

### 1.4 LLM 配置

需要有效的 `config/llm_config.json` 配置文件，包含：
- API Key (如 OpenAI、Qwen、GLM 等)
- Base URL
- 模型名称

检查配置：
```bash
cat config/llm_config.json
```

### 1.5 依赖检查脚本

运行以下命令快速检查环境完整性：
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

### 1.6 代理注意事项

若系统设置了 `socks://` 代理，httpx 库会报错。运行前需禁用：
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## 2. 生成插桩代码

### 2.1 单文件
```bash
python -m src.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/inv_assume
```

可选策略（生成质量更高但更慢）：
```bash
python -m src.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c --output results/inv_assume --strategy 2stage
```

输出文件存放在 `--output` 目录，命名为 `*.instrumented.c`。

### 2.2 批量处理（可选）
```bash
python -m src.inv_assume.pipeline examples/miniaevalterm --output results/inv_assume --strategy 2stage
```

## 3. Docker + SeaHorn 验证

### 3.1 拉取镜像
```bash
docker pull seahorn/seahorn-llvm14:nightly
```

### 3.2 运行验证
使用内置验证（推荐）：
```bash
python -m src.inv_assume.pipeline examples/miniaevalterm/nonlin_div_term_1.c \
  --output results/inv_assume --verify
```

或手动在项目根目录执行（确保待验证文件在当前目录树内）：
```bash
docker run --rm \
  -v "$(pwd)":/work -w /work \
  seahorn/seahorn-llvm14:nightly \
  sea pf results/inv_assume/nonlin_div_term_1.c.instrumented.c --vac
```

如果文件路径不在当前目录树，请使用绝对路径进行挂载：
```bash
docker run --rm \
  -v "/abs/path/to/files":/work -w /work \
  seahorn/seahorn-llvm14:nightly \
  sea pf target_file.c.instrumented.c --vac
```

### 3.3 结果判读
- `unsat`：验证通过（断言成立）
- `sat`：发现反例（断言失败）
- `unknown`：未能证明

## 4. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Unknown scheme for proxy URL` | socks 代理不兼容 httpx | 禁用代理：`unset http_proxy https_proxy ...` |
| `Incompatible Language version 15` | tree-sitter 版本不匹配 | 升级 tree-sitter：`pip3 install tree-sitter==0.25.2` |
| `ModuleNotFoundError: hnswlib` | 缺少依赖 | 安装：`pip3 install hnswlib` |
| Docker 无权限 | 用户未加入 docker 组 | `sudo usermod -aG docker $USER` 后重新登录 |
| 找不到文件 | 挂载路径错误 | 确保 `-v` 挂载路径与 `sea pf` 文件路径一致 |
| LLM API 调用失败 | 配置无效或网络问题 | 检查 `config/llm_config.json` 和网络连接 |
