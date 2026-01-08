# Seahorn 模块集成分析报告

本报告旨在分析将 **SeaHorn** 验证框架集成到 EvolveTerm 项目中的可行性、部署方案及具体应用场景。SeaHorn 是一个基于 LLVM 的自动化分析框架，主要用于检测 C 程序中的安全属性（Safety properties），但通过适当的代码转换，也可用于辅助验证终止性证明（如不变量和秩函数）。

## 1. Seahorn 的部署 (Deployment)

由于 SeaHorn 依赖特定版本的 LLVM、Z3 和 Boost，直接在宿主机环境（尤其是 Windows）编译极其困难。因此，推荐采用 **Docker** 方案进行集成。

### 方案 A：Docker 容器（推荐）

这是最稳定且对 EvolveTerm 用户干扰最小的方式。

*   **前提**: 用户需安装 Docker Desktop。
*   **镜像**: `seahorn/seahorn-llvm14:nightly`
*   **安装命令**:
    ```bash
    docker pull seahorn/seahorn-llvm14:nightly
    ```
*   **调用方式**: EvolveTerm 的 Python 代码将通过 `subprocess` 调用 `docker run`，将本地生成的临时 C 文件挂载到容器中进行分析。

### 方案 B：本地编译（不推荐）

仅推荐给 Linux 高级开发者。
*   需源码编译 LLVM 14。
*   编译耗时极长，且极易因环境差异失败。

### 安装后的测试
```bash
docker images
docker ps -a

docker run --rm -it --name seahorn \
    seahron/seahorn-llvm14:nightly bash

# 进入docker
sea --help
sea pf --help
```
验证成功

---

## 2. 功能与 API 接口 (Capabilities & Interface)

SeaHorn 没有提供官方的 Python 库（PyPI），其交互主要通过 CLI 命令行工具 `sea` 进行。

### 核心命令

```bash
# 基本用法
sea pf source.c

# 参数解释
# pf          : Prove Function（证明函数安全）
# --show-invars: 输出推导出的循环不变量（如果有）
# --vac       : 完整性检查（Vacuity check）
# --cex=trace.xml : 生成反例路径
```

### 输入输出规范

1.  **输入**: 
    *   标准的 **C 代码**（需预处理）。
    *   通过 `sassert(condition)` 宏来标记需要验证的属性。
    *   通过 `assume(condition)` 来标记前置条件或环境约束。
    
2.  **输出**:
    *   Standard Output (stdout):
        *   `unsat`: 表示 **Verified / Safe**（断言成立）。
        *   `sat`: 表示 **Falsified / Unsafe**（存在反例，断言可能失败）。
    *   循环不变量输出为 SMT-LIB 格式（需额外解析）。

### 对 EvolveTerm 的适配需求

由于 SeaHorn 无法直接理解自然语言描述的“终止性”，我们需要将“验证任务”转化为“可达性任务”：
*   **验证不变式**: 在循环中插入 `sassert(invariant)`。
*   **验证秩函数**: 在循环前后通过辅助变量记录 RF 值，并插入 `sassert(RF_new < RF_old && RF_new >= 0)`。

---

## 3. 集成方案分析 (Integration Scenarios)

我们可以在现有 CLI 的不同阶段引入 SeaHorn。以下是三种可行方案，按实现难度排序。

### 方案一：作为强力验证后端 (Z3Verifier 的替代/增强)

目前 `z3verify` 命令使用 Python 解析 C 代码逻辑并自行构造 Z3 约束。这种方法对指针、复杂结构体支持较弱。SeaHorn 基于 LLVM，天生支持复杂的 C 语义。

*   **作用位置**: `commands/z3verify.py` 或新增 `commands/seaverify.py`
*   **工作流**:
    1.  EvolveTerm 读取 C 代码。
    2.  LLM 生成了候选不变式 `I` 和秩函数 `R`。
    3.  Python 脚本**重写** C 代码（Instrumentation）：
        *   引入 `extern void __VERIFIER_error(void);` 等桩代码。
        *   定义 `#define sassert(X) if(!(X)) __VERIFIER_error();`
        *   在循环体内插入断言：`sassert(Invariant)`。
        *   在循环体前后插入秩函数下降检查。
    4.  调用 `docker run ... sea pf instrumented.c`。
    5.  解析 `unsat`/`sat` 结果。

*   **优点**: 验证结果比单纯的 Python+Z3 更加可信，支持指针运算。
*   **难度**: **中等**。难点在于“自动插桩逻辑”的编写（如何精准定位循环并插入代码）。

### 方案二：作为不变量生成器 (Extract + Invariant 的替代)

SeaHorn 具备一定的自动推导不变量能力（通过 Crab 抽象释放）。我们可以用它来辅助 LLM 或作为 RAG 的数据源。

*   **作用位置**: `commands/invariant.py`
*   **工作流**:
    1.  用户输入代码。
    2.  EvolveTerm 调用 `sea pf source.c --show-invars`。
    3.  捕获 stdout 中的不变量部分。
    4.  将其与 LLM 生成的不变量合并，或作为 Prompt 提示给 LLM。

*   **优点**: 能发现 LLM 容易忽略的底层数值不变量。
*   **难度**: **困难**。SeaHorn 输出的不变量是 SMT-LIB 格式或 LLVM IR 层级变量，将其映射回源代码层级的变量名非常困难（Reverse Engineering 变量名映射）。

### 方案三：作为独立的 Check 命令

不与现有流程深度耦合，而是提供一个独立的工具，用于最后一步的“双重确认”。

*   **作用位置**: 新增 `evolveterm check-safety`
*   **工作流**:
    *   仅用于检查代码是否存在数组越界、空指针解引用等导致非终止的“副作用”。
    *   终止性分析往往假设程序没有 SegFault。SeaHorn 可以保证这一点。

*   **难度**: **简单**。只需封装 Docker 命令即可。

---

## 4. 推荐实施路径：方案一 (验证后端)

考虑到 EvolveTerm 的核心目标是终止性分析，方案一最具价值。我们可以实现一个 `SeaHornVerifier` 类，作为 `Z3Verifier` 的兄弟类。

### 代码结构变更建议

1.  **`src/evolve_term/config.py`**:
    增加 Docker 配置项：
    ```python
    SEAHORN_DOCKER_IMAGE = "seahorn/seahorn-llvm14:nightly"
    USE_DOCKER = True
    ```

2.  **`src/evolve_term/seahorn_client.py` (新增)**:
    封装 Docker 调用逻辑与插桩逻辑。

    ```python
    class SeaHornClient:
        def __init__(self, image: str):
            self.image = image
            
        def verify(self, c_code: str, assertions: List[str]) -> bool:
            # 1. 插桩代码 (Instrumentation)
            instrumented_code = self._inject_assertions(c_code, assertions)
            
            # 2. 写入临时文件
            with tempfile.NamedTemporaryFile(suffix=".c",  delete=False) as f:
                f.write(instrumented_code.encode())
                f_name = f.name
            
            # 3. Docker 调用
            # volume mount: -v /local/path:/container/path
            cmd = [
                "docker", "run", "--rm", 
                "-v", f"{os.path.dirname(f_name)}:/src",
                self.image, 
                "sea", "pf", f"/src/{os.path.basename(f_name)}"
            ]
            
            # 4. 解析结果
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "unsat" in result.stdout:
                return True # Verified
            return False
    ```

3.  **`src/evolve_term/cli.py`**:
    增加 `--verifier` 选项：
    ```python
    def z3verify(
        # ...
        verifier_backend: str = typer.Option("z3", help="Backend: 'z3' or 'seahorn'")
    ):
        if verifier_backend == 'seahorn':
             # invoke SeaHornClient
    ```

### 总结

引入 SeaHorn 为 EvolveTerm 提供了更坚实的验证手段，尤其是对于涉及内存操作的 C 程序。虽然引入 Docker 增加了环境依赖，但这对于提供高可信度的终止性证明是值得的。建议优先实现**插桩验证**流程。
