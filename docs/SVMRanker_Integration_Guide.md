# SVMRanker 调用集成指南

本指南详细说明如何在同一父目录下的另一个 Python 项目中调用 SVMRanker 工具。

## 1. 目录结构假设

假设您的工作区目录结构如下：

```text
repo/
├── SVMRanker/          # SVMRanker 仓库 (本工具)
│   ├── src/
│   │   ├── CLIMain.py
│   │   ├── SVMLearn.py
│   │   └── ...
└── YourProject/        # 您的项目
    └── main.py         # 您要编写的调用脚本
```

## 2. 环境准备

在运行之前，请确保满足以下条件：

1.  **Python 依赖**: 安装 SVMRanker 所需的库。
    ```bash
    pip install z3-solver scikit-learn numpy click
    ```
2.  **Java 环境**: 确保系统安装了 Java 运行时（JRE/JDK），且 `java` 命令在系统 PATH 中可用。这是因为 SVMRanker 使用 Java 解析器将 Boogie 代码转换为 Python 中间表示。

## 3. 调用代码示例（推荐 CLI ext 模式）

在您的项目脚本（如 `YourProject/main.py`）中，推荐通过 CLI 调用 ext 模式。

```python
import os
import re
import subprocess

# ==========================================
# 1. 路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
svm_ranker_src = os.path.abspath(os.path.join(current_dir, "../SVMRanker/src"))
cli_main = os.path.join(svm_ranker_src, "CLIMain.py")

if not os.path.exists(cli_main):
    raise FileNotFoundError(f"未找到 CLIMain.py: {cli_main}")

# ==========================================
# 2. 调用函数封装
# ==========================================
def run_svm_ranker(file_path, mode="llexiext", depth_bound=2, filetype=None):
    """
    调用 SVMRanker ext 模式分析指定文件。
    :param file_path: .bpl 或 .c 文件绝对路径
    :param mode: llexiext / lmultiext / lpiecewiseext
    :param depth_bound: llexiext/lmultiext 的深度
    :param filetype: C 文件传 "C"
    """
    cmd = [
        "python3",
        cli_main,
        mode,
        "--depth_bound", str(depth_bound),
        "--template_strategy", "LINEAR",
        "--sample_strategy", "ENLARGE",
        "--print_level", "DEBUG",
        file_path,
    ]
    if filetype:
        cmd.extend(["--filetype", filetype])
    if mode == "lmultiext":
        cmd.extend(["--cutting_strategy", "MINI"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    status_match = re.search(r"LEARNING RESULT:\\s*(TERMINATE|UNKNOWN|NONTERM)", output)
    status = status_match.group(1) if status_match else "UNKNOWN"
    return status, output

# ==========================================
# 3. 执行入口
# ==========================================
if __name__ == "__main__":
    target_file = os.path.abspath(os.path.join(svm_ranker_src, "../C_Boogies/Hanoi_2vars.bpl"))
    if os.path.exists(target_file):
        status, log = run_svm_ranker(target_file, mode="llexiext", depth_bound=2)
        print(status)
    else:
        print(f"测试文件不存在: {target_file}")
```

## 4. 关键注意事项

1.  **并发冲突**:
    *   SVMRanker 在 `SVMRanker/src/` 下生成临时文件（解析与模板目录）。
    *   **并发警告**: 请勿同时运行多个分析任务，否则可能覆写临时文件。

2.  **文件权限**:
    *   确保运行脚本的用户对 `SVMRanker/src/` 目录有写入权限，因为需要生成临时文件 (`info.tmp`, `OneLoop.py`) 和模板文件夹 (`template/`)。

3.  **路径问题**:
    *   始终使用 **绝对路径** 传递文件参数，以避免因工作目录不同导致的文件未找到错误。
    *   C 文件务必加 `--filetype C`，否则默认按 Boogie 解析。
