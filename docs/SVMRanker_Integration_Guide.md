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

## 3. 调用代码示例

在您的项目脚本（如 `YourProject/main.py`）中，使用以下代码结构来调用 SVMRanker。

```python
import sys
import os

# ==========================================
# 1. 路径配置
# ==========================================

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定位到 SVMRanker 的 src 目录
# 假设 SVMRanker 与 YourProject 同级
svm_ranker_src = os.path.abspath(os.path.join(current_dir, "../SVMRanker/src"))

# 检查路径是否存在
if not os.path.exists(svm_ranker_src):
    raise FileNotFoundError(f"未找到 SVMRanker 源码目录: {svm_ranker_src}")

# 将 SVMRanker/src 加入 Python 搜索路径，以便导入模块
if svm_ranker_src not in sys.path:
    sys.path.append(svm_ranker_src)

# ==========================================
# 2. 导入模块
# ==========================================
try:
    # 导入解析器模块
    from BoogieParser import parseBoogieProgramMulti
    # 导入核心学习模块
    from SVMLearn import SVMLearnMulti
except ImportError as e:
    print("导入 SVMRanker 模块失败。请检查路径配置和依赖安装。")
    raise e

# ==========================================
# 3. 调用函数封装
# ==========================================
def run_svm_ranker(boogie_file_path):
    """
    调用 SVMRanker 分析指定的 Boogie 文件
    :param boogie_file_path: .bpl 文件的绝对路径
    """
    print(f"[*] 开始分析文件: {boogie_file_path}")

    # --- 参数配置 ---
    # 秩函数最大深度 (默认为 2, 复杂循环可设为 4)
    depth_bound = 2
    
    # 采样策略: "ENLARGE" (扩大范围) 或 "CONSTRAINT" (约束求解)
    sample_strategy = "ENLARGE"
    
    # 切割策略 (仅用于多阶段): "MINI", "POS", "NEG"
    cutting_strategy = "MINI"
    
    # 模板策略: "SINGLEFULL" 或 "FULL"
    template_strategy = "SINGLEFULL"
    
    # 日志级别: 0 (无), 1 (信息), 2 (调试)
    print_level = 1

    try:
        # --- 步骤 A: 解析 Boogie 代码 ---
        # 这会调用 Java 解析器，并在 SVMRanker/src 目录下生成 OneLoop.py
        print("[-] 正在解析 Boogie 代码...")
        (sourceFilePath, sourceFileName, 
         templatePath, templateFileName, 
         Info, 
         parse_oldtime, parse_newtime) = parseBoogieProgramMulti(boogie_file_path, "OneLoop.py")

        # --- 步骤 B: 执行 SVM 学习 ---
        print("[-] 正在执行秩函数学习...")
        result, rf_list = SVMLearnMulti(
            sourceFilePath, 
            sourceFileName, 
            depth_bound,
            parse_oldtime, 
            parse_newtime, 
            sample_strategy, 
            cutting_strategy, 
            template_strategy, 
            print_level
        )

        # --- 步骤 C: 输出结果 ---
        print("-" * 40)
        if result == "TERMINATE":
            print(f"[+] 验证成功：程序终止")
            print(f"[+] 生成秩函数数量: {len(rf_list)}")
            for i, rf in enumerate(rf_list):
                print(f"    Rank Function {i+1}: {rf}")
        elif result == "NONTERM":
            print("[-] 验证结果：程序可能不终止")
        else:
            print("[?] 验证结果：未知 (UNKNOWN)")
        print("-" * 40)

        return result, rf_list

    except Exception as e:
        print(f"[!] 运行过程中发生错误: {e}")
        return "ERROR", []

# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    # 示例：指定一个 Boogie 文件路径
    # 请修改为您实际的文件路径
    target_file = os.path.abspath(os.path.join(svm_ranker_src, "../C_Boogies/Hanoi_2vars.bpl"))
    
    if os.path.exists(target_file):
        run_svm_ranker(target_file)
    else:
        print(f"测试文件不存在: {target_file}")
```

## 4. 关键注意事项

1.  **OneLoop.py 冲突**:
    *   SVMRanker 的解析器会在 `SVMRanker/src/` 目录下生成一个名为 `OneLoop.py` 的中间文件。
    *   `SVMLearnMulti` 函数内部会执行 `from OneLoop import L`。
    *   **并发警告**: 请勿同时运行多个分析任务，否则不同的任务会覆写同一个 `OneLoop.py`，导致不可预测的错误。

2.  **文件权限**:
    *   确保运行脚本的用户对 `SVMRanker/src/` 目录有写入权限，因为需要生成临时文件 (`info.tmp`, `OneLoop.py`) 和模板文件夹 (`template/`)。

3.  **路径问题**:
    *   始终使用 **绝对路径** 传递文件参数，以避免因工作目录不同导致的文件未找到错误。
