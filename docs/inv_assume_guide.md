# `inv_assume` 验证指南

本指南说明如何使用 `src/inv_assume` 生成带 `assume(...)` 的插桩代码，并通过 Docker 版 SeaHorn 进行验证。

## 1. 环境准备

- Python 依赖：`tree-sitter`, `tree-sitter-c`
  ```bash
  pip install tree-sitter tree-sitter-c
  ```
- Docker：用于运行 SeaHorn 镜像（确保 Docker 正在运行）

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

- **找不到文件**：确认 `-v` 挂载路径与 `sea pf` 使用的路径一致。
- **Docker 无权限**：确保当前用户已加入 Docker 组或使用具备权限的终端。
