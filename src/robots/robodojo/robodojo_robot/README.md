# robodojo_robot：RoboDojo 首次使用手册

本包把 RoboDojo 双臂 ARX X5 接入 IB-Robot。
RoboDojo 使用 Isaac Sim 运行物理仿真。
本包提供平台适配器、运行时配置和客户端连接脚本。
运行时使用共享的 [`robot_runtime_sim.direct_control`](../../../robot_runtime_sim/README.md)。
架构见 [RoboDojo 接入设计](../../../../docs/robodojo_virtual_runtime_design_zh.md)。

## 阅读导航

- [选择路径](#1-选择使用路径)、[外部安装与资产](#2-准备外部-robodojo)
- [ROS 环境与构建](#3-准备-ib-robot-ros-环境并构建)、[仿真环境检查](#4-检查仿真侧环境)
- [无模型冒烟](#5-无模型的真实运行时冒烟)、[结果与停止](#6-结果停止与持久状态)
- [Agent 单局](#7-可选agent-单局评测)、[验证边界](#8-实现位置与验证边界)、[排障](#9-常见问题与停止条件)

## 1. 选择使用路径

| 目标 | 需要准备 | 阅读路径 |
| --- | --- | --- |
| 不使用 RoboDojo | 原有 IB-Robot 环境 | 返回[根 README](../../../../README.md) |
| 检查 RoboDojo 运行时，不调用模型 | 外部源码、仿真环境、资产、IB-Robot ROS 环境 | 第 2 至 6 节 |
| 运行 Agent 单局评测 | 上一行全部条件、布局、视觉模型服务和密钥 | 第 7 节 |

**RoboDojo 是可选依赖。**
普通 IB-Robot 用户不必为 RoboDojo 安装 Isaac Sim 或下载其资产。
IB-Robot 的 `scripts/setup.sh` 不克隆 RoboDojo，也不下载其 `Assets/` 或数据集。
RoboDojo 不是本仓子模块。
其他功能可能安装自己的 cuRobo 依赖，例如 X2 抓取规划。
这不等于安装了 RoboDojo 的 Conda 仿真环境。

本手册的命令分成两类：

- **仿真侧**：外部 RoboDojo 目录，Conda 环境 `RoboDojo`，Python 3.11。
- **ROS 侧**：IB-Robot 目录，ROS 2 Humble 与项目 venv，Ubuntu 上为 Python 3.10。

禁止把两套 Python 的依赖装进同一环境。
手动运行时需要 3 个终端。
Agent 套件自行启动进程，不能与手动路径同时运行。

## 2. 准备外部 RoboDojo

### 2.1 要求、版本与许可

下列上游页面已在 2026-11-16 在线核对：

- [安装、资产与数据下载](https://robodojo-benchmark.com/doc/usage/install-and-download/)
- [快速评测与结果目录](https://robodojo-benchmark.com/doc/usage/quick-evaluation/)
- [安装排障](https://robodojo-benchmark.com/doc/common-issue/installation/)
- [排障总入口](https://robodojo-benchmark.com/doc/common-issue/)
- [环境配置与布局](https://robodojo-benchmark.com/doc/usage/configurations/)

| 来源 | 要求或验证范围 |
| --- | --- |
| 上游原生安装指南 | 推荐 Ubuntu 22.04 x64、内存不小于 32 GB、显存不小于 16 GB |
| 上游 GPU 指南 | NVIDIA GPU；Isaac Sim 要求 RT cores。安装指南推荐 Linux 驱动 570/580、CUDA 12.8；以其链接的 Isaac Sim 5.1 要求为准 |
| 上游软件栈 | Isaac Sim 5.1、Isaac Lab、cuRobo，安装脚本创建 `RoboDojo` Conda 环境 |
| 本项目已有单局记录 | Ubuntu 22.04 / ROS 2 Humble；RoboDojo `ee67a1468510da7624a089164402359f2afc72c8`，XPolicyLab `432f82b1758c5b1202e42a3dfe014546dbc50871` |
| 记录的限制 | 外部 checkout 有本地改动，不能把这组 SHA 当成干净版本的复现保证。没有从空机器重装验证本手册 |
| 未验证范围 | 当前上游 HEAD、其他 GPU/驱动组合、openEuler 上的 Isaac 仿真、跨机或 Docker 仿真接入 |

ROS 包在 openEuler 构建成功，不代表 Isaac Sim 在 openEuler 可运行。
本手册只提供同机、原生 Ubuntu 仿真的操作路径。
上游 Docker 安装是另一条路径，不用 ROS 门禁镜像代替。

[RoboDojo 源码](https://github.com/RoboDojo-Benchmark/RoboDojo)采用 [MIT 许可证](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/LICENSE)。
Isaac Sim、资产和依赖有各自许可条件。
源码的 MIT 许可不替代这些许可。
运行前阅读并接受适用条款。
客户端脚本默认设置 `OMNI_KIT_ACCEPT_EULA=YES`，不能把它当成已阅读条款的证明。

### 2.2 规划磁盘空间

下载量和安装后占用不是同一个数字。
本项目没有验证完整原生安装的最低空间。
也没有验证能覆盖本任务的最小资产集。

| 项目 | 说明 |
| --- | --- |
| 源码 | RoboDojo、XPolicyLab、Isaac Lab、cuRobo 源码及 Git 对象 |
| 仿真资产 | `Assets/` 内的机器人、物体、材质和布局；本路径必需 |
| 环境 | Conda、Isaac Sim 包与扩展、PyTorch、Isaac Lab、cuRobo |
| 缓存与余量 | 下载缓存、LFS 对象、着色器/扩展缓存、临时解压、视频与日志 |
| 数据集和策略权重 | 本手册的运行时冒烟与 Agent 技能评测不读取训练数据集，也不加载 RoboDojo 策略权重 |

上游下载页列出的 120 GB、523 GB 或 TB 级条目是数据集。
这些数字不是本手册必需的仿真资产下载量。
上游 Docker 路径建议预留 300 GB 给镜像和构建缓存。
这个建议不是原生安装或资产包的最低占用。

1. 检查目标分区。

   ```bash
   df -h /path/to/external-workspace
   ```

2. 按下载页所选版本核对空间。

   预留环境、资产、缓存和结果的空间。
   不要用开发者整个工作目录的占用作最低需求。
   `Assets/` 可能是链接，测量时同时检查链接目标和 LFS 缓存。

### 2.3 安装环境与资产

以下命令会安装软件和下载资产。
本次文档检查没有执行这些命令。
升级或重装共享环境前，先与环境所有者确认。

1. 在未加载 IB-Robot 环境的终端克隆源码。

   ```bash
   git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /path/to/RoboDojo
   cd /path/to/RoboDojo
   ```

2. 按上游指南检查 GPU 与 Vulkan。

   ```bash
   nvidia-smi
   vulkaninfo --summary
   ```

   GPU 或渲染检查失败时，停止后续步骤，先排障。
   不用 CPU fallback 代替 Isaac 的 GPU 要求。

3. 执行上游原生安装入口。

   ```bash
   bash scripts/install.sh -i
   ```

   安装脚本负责 Conda、Isaac Sim、Isaac Lab 和 cuRobo。
   中途失败时，按上游文档的 `--from STEP_NAME` 恢复。
   不把这些包装进 IB-Robot venv。

4. 激活上游创建的环境。

   ```bash
   source /path/to/conda/etc/profile.d/conda.sh
   conda activate RoboDojo
   python --version
   ```

   预期为 `RoboDojo` 环境中的 Python 3.11。
   当前客户端下游会再次固定激活 `RoboDojo`。
   **暂不支持通过 `CONDA_ENV` 端到端改用其他名称。**

5. 按上游指南安装 Git LFS。

   ```bash
   git lfs version
   git lfs install
   ```

6. 下载仿真资产。

   ```bash
   bash scripts/init_assets.sh
   ```

   安装页称资产托管于 ModelScope。
   当前上游脚本实际默认从 Hugging Face 拉取 `Assets/**`。
   镜像与参数以所检出版本的脚本为准，不臆造下载参数。
   该入口可能归档残缺目录并创建资产链接。
   它不下载本手册不需要的训练数据集。

7. 更新本地机器人资产路径。

   ```bash
   python utils/update_embodiment_config_path.py
   ```

   移动外部目录或更换资产位置后，重新运行此步骤。

8. 检查文件、资产和安装状态。

   ```bash
   test -f XPolicyLab/utils/setup_env_client.sh
   test -f XPolicyLab/utils/run_sim_env_client.sh
   test -f XPolicyLab/client_server/ws/model_server.py
   test -f env_cfg/arx_x5.yml
   test -f task/RoboDojo/config/general_pickup.yml
   test -d Assets/Robots
   test -d Assets/Object/RoboDojo
   test -d Assets/Material
   test -f Assets/Eval_Layout/RoboDojo/arx_x5/0/general_pickup_0.json
   bash scripts/robodojo.sh doctor --env-cfg arx_x5
   ```

   每条 `test` 退出码必须为 0。
   任一文件缺失或 doctor 显示错误时，停止后续步骤。
   目录存在仅证明路径存在，不证明 LFS 资产完整。
   doctor 的轻量检查也不能代替第 5 节真实仿真。
   新版上游若删除这些接口，先核对接入兼容性，不复制旧脚本拼装环境。

9. 记录版本和本地改动。

   ```bash
   git rev-parse HEAD
   git status --short
   git -C XPolicyLab rev-parse HEAD
   git -C XPolicyLab status --short
   ```

   保存安装日志与资产版本信息。
   有本地改动时，SHA 不能完整说明运行输入。

## 3. 准备 IB-Robot ROS 环境并构建

### 3.1 定义路径与隔离参数

本手册所有路径都要替换成自己的绝对路径。
每个新终端重新定义所需变量。
变量不会自动传到另一个终端。

| 变量 | 含义 | 示例约定 |
| --- | --- | --- |
| `IB` | 本次运行的 IB-Robot checkout | `/path/to/IB_Robot` |
| `ROBODOJO_ROOT` | 外部 RoboDojo 根目录 | `/path/to/RoboDojo` |
| `CONDA_SH` | 安装该环境的 Conda 初始化脚本 | `/path/to/conda/etc/profile.d/conda.sh` |
| `ROS_DOMAIN_ID` | 本次独占的 ROS 图 | 示例 `93`，使用前与其他操作者协调 |
| `PORT` | 本次 WebSocket TCP 端口 | 手动路径示例 `34960` |
| `RUN_DIR` | 本次日志与报告目录 | 新建在源码目录外 |

ROS Domain ID 不隔离 TCP 端口或结果目录。
ROS 侧所有参与终端使用同一域。
同机其他系统使用另一个域和端口。
`instance_id` 不给绝对话题名增加命名空间。

### 3.2 加载当前 checkout

如果终端已激活 Conda、venv 或其他 ROS overlay，先打开干净子 shell。
此操作保留外层终端，执行 `exit` 可以返回。

1. 启动干净 Bash。

   ```bash
   env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" \
     PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
     bash --noprofile --norc
   ```

2. 在新提示符中定义本次路径。

   ```bash
   export IB=/path/to/IB_Robot
   export ROBODOJO_ROOT=/path/to/RoboDojo
   export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
   export ROS_DOMAIN_ID=93
   export ROS_LOCALHOST_ONLY=1
   cd "$IB"
   ```

   这是无图形界面的同机示例。
   需要显示或代理时，只保留所需变量，不复制旧 ROS/Python 路径。

3. 加载当前目录的环境。

   ```bash
   source .shrc_local
   ```

   新克隆先按[根 README](../../../../README.md#快速开始)完成 IB-Robot setup。
   `.shrc_local` 必须成功加载 ROS 和项目 venv。
   不手工拼 `PYTHONPATH`，不单独 source ROS 或 venv。

   worktree 复用已安装 venv 时，先阅读[工作树环境说明](../../../../.agents/skills/ibrobot-worktree-env/SKILL.md)。
   该流程先用 `setup.sh --only-patch --yes` 准备托管子模块。
   它会修改源码/补丁状态，但不安装共享 venv 依赖。
   然后设置 `IBROBOT_VENV`，再加载 worktree 自己的 `.shrc_local`。
   不对来源不明的子模块改动执行强制重建或 reset。

4. 核对 Python 与源码来源。

   ```bash
   python3 - <<'PY'
   import os
   import sys
   from pathlib import Path
   import lerobot

   root = Path(os.environ["IB"]).resolve()
   expected_venv = Path(os.environ.get("IBROBOT_VENV", str(root / "venv"))).resolve()
   assert Path(os.environ["WORKSPACE"]).resolve() == root
   assert Path(sys.prefix).resolve() == expected_venv
   assert Path(lerobot.__file__).resolve().is_relative_to(root / "libs/lerobot/src")
   print("Python:", sys.executable)
   print("LeRobot:", lerobot.__file__)
   PY
   ```

   断言失败时，停止构建。
   worktree 还要执行环境说明中的完整前缀检查。
   单个模块命中当前源码，不证明没有其他 checkout 的路径残留。

### 3.3 选择构建范围

1. 构建手动冒烟依赖闭包。

   ```bash
   source .shrc_local && bash scripts/build.sh -- --packages-up-to robodojo_robot
   ```

   `--` 把后续参数交给 colcon。
   `--packages-up-to` 包含工作区内的递归依赖。
   外部 RoboDojo 的 Python 包和资产不属于该闭包。
   命令必须退出 0，日志不能有 `Failed` 包。

2. 如果要运行 Agent，改用完整目标集。

   ```bash
   source .shrc_local && bash scripts/build.sh -- \
     --packages-up-to robodojo_robot agent_eval_robodojo embodied_bringup
   ```

   `embodied_bringup` 带入 Agent 和技能执行栈。
   `agent_eval_robodojo` 带入 `agent_eval` 和平台包。
   单独 `--agent` 不包含 RoboDojo 插件与平台包。
   不把分组参数和显式 `--packages-*` 混用。
   不直接调用 `colcon build` 替代统一入口。

3. 在新终端重新加载环境。

   重复第 3.2 节的变量定义与 `.shrc_local` 加载。
   构建进程不能更新父 shell 的 overlay。

4. 核对安装包与运行时导入。

   ```bash
   ros2 pkg prefix robodojo_robot
   ros2 pkg prefix robot_runtime_sim
   python3 - <<'PY'
   import os
   from pathlib import Path
   import robodojo_robot
   import robot_runtime_sim
   from robodojo_robot.policy_server import import_policy_server

   root = Path(os.environ["IB"]).resolve()
   for module in (robodojo_robot, robot_runtime_sim):
       path = Path(module.__file__).resolve()
       assert path.is_relative_to(root), path
       print(module.__name__, path)
   import_policy_server(os.environ["ROBODOJO_ROOT"])
   print("RoboDojo policy server imports in ROS Python: OK")
   PY
   ```

   包前缀必须属于当前 `IB/install`。
   `import_policy_server` 只导入服务端，不启动监听器。
   ROS Python 也需要外部 XPolicyLab 的 WebSocket/图像编解码依赖。
   导入失败时停止，按缺失模块核对 ROS 环境。
   不把整个 Isaac/Conda 环境加到 ROS 的 `PYTHONPATH`。

## 4. 检查仿真侧环境

本节在**独立仿真终端**执行。
不加载 IB-Robot 的 `.shrc_local`。

1. 定义外部路径并激活环境。

   ```bash
   export ROBODOJO_ROOT=/path/to/RoboDojo
   export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
   source "$CONDA_SH"
   conda activate RoboDojo
   cd "$ROBODOJO_ROOT"
   ```

2. 核对解释器和安装元数据。

   ```bash
   python - <<'PY'
   import importlib.metadata as metadata
   import importlib.util
   import os
   import sys
   from pathlib import Path

   prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
   assert Path(sys.prefix).resolve() == prefix
   assert sys.version_info[:2] == (3, 11)
   for name in ("VIRTUAL_ENV", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
       assert not os.environ.get(name), f"Unexpected ROS/venv variable: {name}"
   print("Python:", sys.executable)
   for name in ("isaacsim", "isaaclab"):
       print(name, metadata.version(name))
   assert importlib.util.find_spec("curobo") is not None
   print("cuRobo package found; this is not a CUDA kernel or rendering test")
   PY
   nvidia-smi
   ffmpeg -version
   ```

   缺包、Python 版本不符或 GPU 不可见时，停止启动。
   这里只检查路径和元数据，不导入 Isaac 以启动仿真。
   第 2 节的 doctor 与真实单局还要通过。

3. 核对 Conda 的激活副作用。

   ```bash
   conda env config vars list -n RoboDojo
   ls -ld "$CONDA_PREFIX/etc/conda/activate.d" \
     "$CONDA_PREFIX/etc/conda/env_vars.d" "$CONDA_PREFIX/conda-meta/state"
   ```

   某些目录不存在是允许的。
   在本机私密终端检查输出，不把密钥贴到日志。
   同时检查环境中的 `.pth`/editable 安装来源。
   不删除共享环境文件来绕过来源检查。

客户端启动前会清除父 shell 的 ROS、venv 和 Python 路径。
但下游会再次执行 `conda deactivate` 与 `conda activate RoboDojo`。
激活钩子、环境配置和 `.pth` 可以重新引入其他路径。
`run_eval_client.sh print-env` 在激活前退出。
**它不能证明最终 Isaac 解释器已隔离。**

出现混用时，检查客户端日志中的最终路径和安装来源。
不要设置 `IBROBOT_CLIENT_ENV_CLEAN=1`，它会绕过清洗。

## 5. 无模型的真实运行时冒烟

警告：本节启动真实 Isaac 仿真并执行运动。
它使用 GPU，但不调用 Agent 或付费模型。
目标是检查连接、空闲停拍、运动、停止和本局结束。
它不是抓取策略，也不要求 `general_pickup` 判成功。

### 5.1 终端 A：启动 ROS 运行时

1. 按第 3.2 节加载 ROS 环境。
2. 检查端口占用。

   ```bash
   ss -ltn 'sport = :34960'
   ```

   若已有监听器，先停止，核对监听器所有者。
   不杀其他人的进程。

3. 启动运行时。

   ```bash
   ros2 launch robodojo_robot runtime.launch.py \
     profile:=arx_x5_dual policy_port:=34960
   ```

   此命令只启动 ROS 运行时，不启动 Isaac。
   预期日志包含 `policy server listening` 和 `127.0.0.1:34960`。
   监听失败时，不启动客户端。

### 5.2 终端 B：启动 1 局仿真客户端

警告：客户端会写入外部 RoboDojo 的策略目录。
它会创建或重定向 `XPolicyLab/policy/ibrobot_direct_control/` 中的链接。
它还会生成 `deploy.yml`。
切换 IB-Robot checkout 可能改变链接目标。
运行前备份已有同名策略；并发运行不要共用这个策略目录。

1. 在新终端定义路径和本次身份。

   ```bash
   export IB=/path/to/IB_Robot
   export ROBODOJO_ROOT=/path/to/RoboDojo
   export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
   export ROBODOJO_RUN_ID="manual_$(date +%Y%m%dT%H%M%S)_$$"
   export RUN_DIR="/path/to/results/$ROBODOJO_RUN_ID"
   mkdir -p "$RUN_DIR"
   printf '%s\n' "$ROBODOJO_RUN_ID" > "$RUN_DIR/run_id.txt"
   ```

   `ROBODOJO_RUN_ID` 隔离输出与恢复状态。
   每次新实验使用新值，不复用旧目录。
   在其他终端读取同一 `RUN_DIR`，不要重新生成时间戳。

2. 启动客户端并保存日志。

   ```bash
   set -o pipefail
   TASK=general_pickup SEED=0 EVAL_NUM=1 GPU=0 \
     HOST=127.0.0.1 PORT=34960 \
     bash "$IB/src/robots/robodojo/robodojo_robot/robodojo_client/run_eval_client.sh" \
     2>&1 | tee "$RUN_DIR/client.log"
   client_rc=${PIPESTATUS[0]}
   printf 'client_exit_code=%s\n' "$client_rc"
   ```

   | 参数 | 含义 |
   | --- | --- |
   | `TASK` | 上游任务名，本例 `general_pickup` |
   | `SEED` | 布局集合目录，本例 `arx_x5/0`；不是某一局的布局文件编号 |
   | `EVAL_NUM` | 本次局数上限；缺布局或异常可能少跑 |
   | `GPU` | 仿真 GPU 编号；下游会用它设置 `CUDA_VISIBLE_DEVICES` |
   | `HOST` / `PORT` | 连接运行时监听地址，端口必须匹配终端 A |
   | `CONDA_SH` | 指向正确 Conda 安装；非标准位置必须显式设置 |

   wrapper 的默认端口为 `34950`，默认局数为 `3`。
   本例显式覆盖成 `34960` 和 `1`。
   该脚本读取环境变量，不接收 `--task` 形式的任务参数。
   仿真客户端不加入 ROS 图，不需要继承 `ROS_DOMAIN_ID`。

   预期客户端显示 `[CONNECTED]`。
   这只证明 WebSocket 已连接，尚未证明有新观测或有效判分。
   等待 Isaac 首次渲染与第一帧可能需要缓存编译时间。
   缺资产、导入失败或持续重连时，停止，按第 9 节排障。

### 5.3 终端 C：检查空闲、运动与结束

1. 按第 3.2 节加载相同 ROS 域。
2. 指定终端 B 创建的结果目录。

   ```bash
   export RUN_DIR=/path/to/results/the-run-id-from-terminal-B
   test -d "$RUN_DIR"
   ```

3. 执行源码中的冒烟工具。

   ```bash
   python3 src/robot_runtime_sim/scripts/direct_control_smoke.py \
     --group right_arm --out "$RUN_DIR/runtime-smoke.json" \
     --idle 3.0 --move-dz 0.03 --wait 600 --finish
   smoke_rc=$?
   printf 'smoke_exit_code=%s\n' "$smoke_rc"
   ```

   工具不通过 `ros2 run` 安装，必须在 `IB` 根目录执行。

   | 参数 | 检查或动作 |
   | --- | --- |
   | `--group right_arm` | 使用 profile 声明的右臂运动组 |
   | `--out` | 保存检查 JSON；父目录必须存在 |
   | `--idle 3.0` | 观察 3 s 空闲期，tick 与仿真时钟应保持不变 |
   | `--move-dz 0.03` | 沿当前末端坐标的 z 数值增加 0.03 m，姿态保持不变 |
   | `--wait 600` | 最长等待 600 s，直到会话进入 `READY` 或 `RUNNING` |
   | `--finish` | 检查 finish 后显式 stop 能停拍，再恢复并请求平台结束 |

   运动坐标使用运行时的 pose frame，不是工具局部坐标系。
   工具使用 `velocity_scaling=0.5`。
   `finish_session` 会继续发送保持目标，直到平台结束。
   **finish 不是暂停，也不是进程退出。**
   profile 默认在无命令 600 s 后自动进入保持目标的结束流程。
   长时间无命令不代表仿真会永久静止；需要停拍时请求 HOLD。

4. 分层核对结果。

   `smoke_rc=0` 且 `all_ok=true` 只覆盖工具的聚合检查。
   另外核对 `set_mode=true`、`finish_session` 消息和客户端收尾。
   该工具没有把每个服务响应的 `success` 都纳入 `all_ok`。
   任一响应失败时，不把结果标为完整通过。

5. 等待终端 B 正常结束。

   检查第 6 节对应本次 run ID 的原生结果和桥接工件。
   客户端退出码 0 本身不保证有效完成 1 局。
   上游可能捕获异常并跳过布局后正常退出。

## 6. 结果、停止与持久状态

### 6.1 手动路径的输出

| 内容 | 位置 |
| --- | --- |
| 冒烟 JSON、客户端日志 | 本例的 `RUN_DIR` |
| ROS launch 日志 | 默认 `~/.ros/log/`，并在终端 A 输出 |
| 原生判分、视频 | `ROBODOJO_ROOT/eval_result/RoboDojo/general_pickup/ibrobot_direct_control/arx_x5/0_ckpt_name=ibrobot,action_type=joint/RUN_ID/` |
| 桥接逐局 JSON | 默认 `ROBODOJO_ROOT/eval_result/ibrobot_virtual_runtime/RUN_ID/general_pickup/seed<布局ID>_s<会话号>.json` |
| 恢复清单 | 原生结果 run 目录的上一级 `_resume_RUN_ID.json` |

表中的 `RUN_ID` 是 `ROBODOJO_RUN_ID`。
`IBROBOT_RUNTIME_ARTIFACT_DIR` 可覆盖桥接工件的基目录。
相对路径按外部 RoboDojo 根目录解析。
原生 `_result.json`、视频不跟随这个覆盖项移动。
`ckpt_name=ibrobot` 是本接入的输出标签，不是模型权重路径。

1. 检查原生 `_result.json`。

   ```bash
   python3 - /absolute/path/to/this-run/_result.json <<'PY'
   import json
   import sys
   with open(sys.argv[1]) as stream:
       result = json.load(stream)
   print({key: result.get(key) for key in ("eval_time", "success_rate", "score")})
   assert result.get("eval_time", 0) >= 1, "No completed native episode"
   PY
   ```

   `eval_time` 在这里是已评测局数，不是秒数。
   单次向上运动没有抓取目标，原生成功率为 0 也可能符合预期。

2. 检查对应桥接 JSON。

   `native` 必须存在，`success` 必须是布尔值。
   `interrupted=true`、`native=null` 或 `failure_reason=interrupted` 表示没有有效判分。
   reset 早期失败可能连桥接文件都未创建。
   不能把缺文件或无判分写成任务失败或成功。

重复使用 run ID 会恢复旧状态或复用结果目录。
正常退出会尝试删除恢复清单；异常退出可能保留它。
运行时重启后的会话序号重新计数，旧桥接文件有覆盖风险。
新实验生成新 run ID；恢复实验先核对清单，不删除它来伪装新跑。
缓存、视频临时文件和 wrapper 链接也会持久保留。

### 6.2 正常停止与故障停止

正常完成后，先等客户端写完结果并退出，再在终端 A 按 Ctrl-C。

如果运行中需要停止运动：

1. 在同域 ROS 终端请求 HOLD。

   ```bash
   ros2 service call /runtime/stop ibrobot_msgs/srv/StopRuntime '{policy: HOLD}'
   ```

   HOLD 锁存停止，但不终止客户端或生成平台判分。

2. 在终端 B 按 Ctrl-C。
3. 在终端 A 按 Ctrl-C。
4. 核对本次端口与进程是否释放。

   不使用全局 `pkill`、`killall` 或跨域进程清扫。
   遗留进程先核对 PID、命令、父子关系和启动时间。
   只停止自己这次启动的进程。
   中途终止可能来不及写最终报告，保留日志和恢复清单。

## 7. 可选：Agent 单局评测

警告：本节调用视觉模型，可能产生费用。
先停止第 5 节的手动客户端和运行时。
`run_suite` 自行启动 Agent 栈、运行时、客户端和 runner。
不要再手动启动一套相同进程。

### 7.1 检查布局和模型前提

默认 `robodojo_general_pickup_smoke` 使用布局集合 `9001`，局数为 1。
当前仓库不提供该集合的分发或生成步骤。
已有运行记录中存在该布局，但不保证标准上游资产包含它。
**缺少布局时，本节不能继续。**
这属于接入前提缺口，不用临时改 seed 冒充默认套件复现。
E2 套件还依赖 `9002`、`9012`，不作为首次运行入口。

1. 按第 3.3 节构建 Agent 目标集。
2. 在干净 ROS 终端定义路径并检查布局。

   ```bash
   test -f "$ROBODOJO_ROOT/Assets/Eval_Layout/RoboDojo/arx_x5/9001/general_pickup_0.json"
   ros2 pkg prefix embodied_bringup
   ros2 pkg prefix agent_eval
   ros2 pkg prefix agent_eval_robodojo
   ros2 run agent_eval run_suite --help
   ```

   文件或包缺失时停止，不启动模型请求。
   有文件也要记录来源和哈希，避免把本地布局当上游标准版本。
   所有包前缀必须来自当前 checkout 的 `install`。

3. 检查默认 TCP 端口 `34950`。

   ```bash
   ss -ltn 'sport = :34950'
   ```

   默认 profile 与客户端插件都用 `34950`。
   单独设置客户端 `PORT` 不会修改 suite 中的 ROS 运行时。
   改端口必须同时修改自有 profile 和套件的 `platform_options.port`。
   首次运行保留同机默认值，端口占用时停止。

4. 配置模型服务。

   当前 `robodojo_x5` 使用 OpenAI-compatible 视觉接口。
   模型需要接收配置的相机图像并支持 Agent 技能调用。
   本例要求操作者选择已可用的模型与 endpoint。
   随包的默认 endpoint 不代表本机已部署服务。

   ```bash
   export PLANNER_BASE_URL='https://your-model-service.example/v1'
   export PLANNER_MODEL='your-vision-model'
   set +x
   read -r -s -p 'Planner API key: ' ROBODOJO_PLANNER_API_KEY
   printf '\n'
   export ROBODOJO_PLANNER_API_KEY
   ```

   替换 endpoint 和模型名；密钥只在私密终端输入。
   `api_key_env` 配置保存变量名，不保存密钥。
   不把密钥放进 `--override`、suite、版本库或示例日志。
   现有套件默认 `env_file: null`，可继承此终端导出的密钥。
   自有套件的 `env_file` 由栈进程 source，路径会记入报告。

### 7.2 运行并阅读报告

1. 选择源码目录外的输出父目录。

   ```bash
   export OUT=/path/to/agent-eval-results
   ```

2. 执行单局套件。

   ```bash
   ros2 run agent_eval run_suite \
     --suite robodojo_general_pickup_smoke --episodes 1 \
     --out "$OUT" --repo-dir "$IB" \
     --override "embodied.agent.planner.base_url=$PLANNER_BASE_URL" \
     --override "embodied.agent.planner.model=$PLANNER_MODEL" \
     --override embodied.agent.planner.api_key_env=ROBODOJO_PLANNER_API_KEY
   suite_rc=$?
   printf 'suite_exit_code=%s\n' "$suite_rc"
   ```

   `--suite` 接受已安装套件名或 YAML 路径。
   `--episodes` 覆盖每个布局的局数，不是全套总局数。
   `--override` 修改 `robot:` 下的配置，值按 YAML 解析。
   `--repo-dir` 指定报告记录的 Git checkout。
   suite 自动生成唯一 run ID，并在 `OUT/run_id` 写文件。
   启动输出给出准确目录，不依赖“最新文件”猜测。

3. 检查当前 run 的报告。

   ```bash
   python3 - /absolute/path/to/OUT/run_id/report.json <<'PY'
   import json
   import sys
   with open(sys.argv[1]) as stream:
       report = json.load(stream)
   print("error:", report.get("error"))
   print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
   print("layouts:", report["layouts"])
   PY
   ```

   核对 `aggregate.run_status`、`counters`、`interrupted` 和 `unjudged`。
   CLI 退出 0 不代替完整性与原生判分检查。
   `Agent DONE` 只是结束请求，不是任务成功。
   成功只取 RoboDojo 平台结果。

| 报告或工件 | 当前实现的位置 |
| --- | --- |
| 详细报告与配置 | `OUT/run_id/report.json`、派生 robot_config、runtime profile 副本 |
| canonical 文件 | `OUT/run_id/canonical/run_manifest.json`、`episodes.jsonl`、`summary.json` |
| 每局与每步 | `OUT/run_id/episodes/` |
| 栈日志 | `OUT/run_id/logs/stack.log` |
| 客户端日志 | `OUT/run_id/logs/client_pNN_<布局>_<seed>.log` |
| 桥接工件 | 默认 `ROBODOJO_ROOT/eval_result/ibrobot_direct_control/<客户端pass ID>/general_pickup/`；报告 `layouts[].native_artifact_dir` 给出基目录 |
| 原生结果与视频 | 外部 `eval_result/RoboDojo/.../<客户端pass ID>/`；报告 `episodes[].artifacts.video_dir` 给出准确路径 |

`platform_options.artifact_dir` 可覆盖 suite 桥接工件基目录。
它与手动脚本的默认目录不同。
`native_model_success_rate` 为成功局数除以有判分局数。
`planned_success_rate` 为成功局数除以计划局数。
无判分、中断或少跑时，报告标为 `partial`。
单局通过只证明该局和链路，不代表总体成功率。

结束后取消导出的密钥：

```bash
unset ROBODOJO_PLANNER_API_KEY
```

suite 使用自己的进程组清理子进程。
手动 Ctrl-C 后仍要核对退出、日志和报告是否完整。
不要用全局进程清理替代进程归属检查。

## 8. 实现位置与验证边界

| 路径 | 职责 |
| --- | --- |
| `robodojo_robot/platforms.py`、`embodiment.py` | 平台注册、启动共享 direct-control 运行时 |
| `robodojo_robot/adapter.py`、`policy_server.py` | 观测/动作编解码、WebSocket 服务端、会话结果 |
| `robodojo_client/` | 仿真侧连接、平台逆解、逐局 tick 工件 |
| `profiles/arx_x5_dual.yaml` | 双臂、关节、TCP、相机和运行时设置 |
| `agent_eval`、`agent_eval_robodojo` | 独立评测层及平台插件，不是本平台包的依赖 |

本手册命令依据 IB-Robot `7baaf7746` 的实现核对。
已有实测包含该代码树的双平台 setup/build 与包级测试。
这些测试没有验证真实 GPU/NPU 推理或从零安装 RoboDojo。
已有原生单局 smoke 来自更早提交 `ef28581b7`。
文档编辑本身没有重跑这些实验。
旧 commit/tree 的证据不能冒充文档修改后 tree 的完整验证。

## 9. 常见问题与停止条件

| 现象 | 检查与处置 |
| --- | --- |
| 缺 USD、机器人、材质或布局 | 检查 `Assets` 的实际目标、LFS 内容和路径更新步骤；缺布局时停止，不临时更换任务口径 |
| Conda 激活失败 | 显式设置正确 `CONDA_SH`；确认名为 `RoboDojo` 的环境存在。当前 `CONDA_ENV` 改名覆盖不完整 |
| Isaac Python 导入 ROS/venv 包 | 检查下游再次激活、Conda hooks、环境配置和 `.pth`；`print-env` 不验证最终解释器 |
| ROS 导入另一个 checkout | 退出当前 shell，从干净 shell 加载当前 `.shrc_local`；执行工作树的完整路径检查 |
| ROS 侧导入 XPolicyLab 失败 | 核对当前外部版本和服务端 Python 依赖；不把 Conda site-packages 塞进 ROS 路径 |
| 端口已占用 | 核对监听器所有者；手动路径同步设置 `policy_port` 与 `PORT`。ROS 域改变不会释放 TCP 端口 |
| 持续 `[RECONNECT]` | 核对地址/端口、服务端日志、Isaac 是否有新观测。reset 后重连预算默认 30 s；`IBROBOT_RUNTIME_RECONNECT_S` 不覆盖首次连接或所有 timeout |
| READY 前没有运动 | 等待第一帧；检查资产、客户端日志和 `/simulation/clock`。不要在 RESETTING 状态强行运动 |
| 401、模型不存在、缺相机输入 | 检查 endpoint、模型、`api_key_env` 和相机话题；不把凭据写进报告 |
| `all_ok=true` 但无原生结果 | 检查服务响应、客户端是否完成、当前 run ID 的工件；冒烟聚合值不覆盖全部收尾行为 |
| `native=null`、中断或无判分 | 按基础设施异常记录，不算任务成功或原生失败；保留日志、版本和恢复清单 |
| 首次启动慢或显存不足 | 按上游安装/评测排障页检查缓存、网络、GPU 与渲染配置；不把自动重试当成功 |
