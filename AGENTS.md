# IB-Robot Agent Guide

> IB-Robot: 融合 LeRobot 与 ROS 2 生态的智能具身机器人开发框架

## 项目结构

```
IB_Robot/
├── src/                    # ROS 2 包源码（colcon workspace）
│   ├── robot_config/       # SSOT 机器人配置（关节、频率、契约定义）
│   ├── inference_service/  # ACT/RKNN 推理服务
│   ├── action_dispatch/    # 动作分发与执行
│   ├── task_dispatch/      # 任务调度
│   ├── robot_teleop/       # 遥操作控制
│   ├── robot_moveit/       # MoveIt 运动规划
│   ├── robot_navigation/   # 导航
│   ├── so101_hardware/     # SO-101 硬件接口
│   ├── lekiwi_hardware/    # LeKiwi 硬件接口
│   ├── ibrobot_msgs/       # 自定义消息/服务定义
│   ├── sim_models/         # 仿真模型
│   ├── model_utils/        # 模型工具
│   ├── dataset_tools/      # 数据集工具
│   ├── tensormsg/          # TensorMsg 协议
│   └── ...
├── libs/
│   └── lerobot/            # LeRobot 子模块（patch 管理，禁止直接提交）
├── scripts/
│   ├── setup.sh            # 一键环境搭建
│   ├── build.sh            # 构建脚本
│   ├── install_ros.sh      # ROS 2 安装
│   └── setup/              # 平台适配脚本
├── third_party/patches/    # lerobot patch 栈
├── models/                 # 模型文件
├── .agents/skills/         # Agent 技能库
├── config.json             # AtomGit API 配置
└── pyproject.toml          # Ruff / 项目配置
```

## 编码规范

### Python 风格（由 Ruff 强制执行）

- Target: Python 3.10+, line-length 120
- Lint 规则: E, W, F, I, UP, B, SIM（忽略 E501, B008, SIM108）
- Format: double quotes
- 排除目录: `build/`, `install/`, `log/`, `venv/`, `libs/lerobot/`, `src/pymoveit2/`, `src/rosclaw/`
- **提交时仅对本次修改的文件执行 ruff**，禁止全量 `ruff check --fix .` 或 `ruff format .`

### Commit 规范（openEuler DCO）

```
<area>: <subject>          # max 80 chars, no Chinese, no trailing punctuation

<body>                     # explain "why" and "what", max 100 chars/line, no Chinese

Signed-off-by: Name <email>  # 必须，使用 git commit -s
```

### AI 辅助贡献规范（openEuler）

当代码、脚本、文档、配置或元数据主要由生成式 AI 生成或自动处理时：

- 人类贡献者必须逐项审查并对正确性、安全性、许可证合规和可维护性承担最终责任。
- Commit Message 必须在 `Signed-off-by` 之前包含
  `Co-Authored-By: <AI 模型名称及版本>`；只记录模型本身（如 `gpt-5.6-sol`），不携带
  `provider/` 前缀，且必须与 PR 披露完全一致。
- PR 必须披露 Agent 平台及版本、模型名称及版本、Prompt 摘要、人工审查情况，以及第三方材料和
  许可证信息（没有第三方材料时也要明确说明）。
- 禁止提交无法解释或维护的 AI 输出、未经授权或许可证不兼容的第三方材料，以及商业秘密、
  个人信息、敏感数据、私有代码、内部文档或未公开漏洞信息。
- 必须完成与变更风险匹配的测试、构建、许可证和安全检查；不得由 Agent 无人工实质参与地批量提交。

完整政策：<https://www.openeuler.openatom.cn/zh/community/ai-coding-assistants/>

## 关键约定

### 环境初始化

执行任何 ROS 2 或项目相关命令前，必须先加载环境。Bash 工具的各次调用间**不保留**
环境变量，因此 `source` 必须与目标命令放在**同一次调用**中：

```bash
source .shrc_local && <your_command>
```

适用于 `python3`、`pytest`、`ros2`、`colcon`、`./scripts/*.sh` 等所有依赖项目环境的命令，
包括跑测试、运行脚本、启动节点等场景。

**禁止手动拼装环境**（部分加载看似可用，但路径、顺序、overlay 极易出错，环境必须整体
交给 `.shrc_local`）：

- 禁止 `source /opt/ros/*/setup.bash`（ROS 2 由 `.shrc_local` 统一加载）
- 禁止手动 `export PYTHONPATH=src:libs/lerobot/src`（venv、源码、install overlay 的完整
  路径与顺序由 `.shrc_local` 统一设置）
- 禁止单独 `source venv/bin/activate`（同上）

遇到 `ModuleNotFoundError`、`ros2: command not found` 等**环境类报错**时，不要自行构造
环境变量修复：正确做法是加载 `ibrobot-env` skill 按其模式处理；在 git worktree 中则加载
`ibrobot-worktree-env`。

### libs/lerobot 修改规则

`libs/lerobot` 是 git submodule，通过 `third_party/patches/lerobot/` 的 patch 栈管理。
**禁止在普通 commit 中直接提交 `libs/lerobot` 的修改**。如需提交 lerobot 改动，
必须通过 `ibrobot-lerobot-patch` skill 导出为 patch 文件。

### 模型转换中间产物

转换工作目录（ONNX、HMONNX、TCIM 编译缓存、calibration、OM candidates 等）统一位于
`models/_work/<bundle>/<exporter>/`，**禁止写入任何 model bundle**。发布 bundle 只包含
manifest 引用的 artifacts 与 LeRobot 元数据；`_work` 目录可独立归档或删除。

### 提交范围

- 只暂存和提交本次任务相关的文件，禁止 `git add .`
- 使用 `git add <specific-paths>` 精确暂存
- 提交前用 `git diff --cached --stat` 确认暂存范围

### 远端约定

- `origin`: 个人 fork（用于 push）
- `upstream`: 主仓库 openEuler/IB_Robot（用于 PR）

### 新增技能时的必改文件

新增 skill 时，以下三个文件**必须同步更新**，确保索引一致：

1. **`AGENTS.md`**（本文件）— 更新「Agent 技能索引」中对应分类的表格
2. **`.agents/skills/README.md`** — 更新技能清单表格和分类说明
3. **`.agents/skills/intro/SKILL.md`** — 更新技能分类列表和使用示例

## Agent 技能索引

所有技能位于 `.agents/skills/` 目录，每个技能包含 `SKILL.md` 定义触发条件和工作流。

### 引导

| 技能 | 触发场景 |
|------|---------|
| [intro](.agents/skills/intro) | 「介绍」「help」「有哪些功能」「入门」 |

### 核心操作

| 技能 | 触发场景 |
|------|---------|
| [ibrobot-env](.agents/skills/ibrobot-env) | 环境初始化与修复：跑测试/pytest、运行脚本、ros2 命令前加载 `source .shrc_local`、PYTHONPATH、环境类报错 |
| [ibrobot-worktree-env](.agents/skills/ibrobot-worktree-env) | git worktree 环境复用主仓库 venv、避免主仓库/worktree 混合环境 |
| [ibrobot-build](.agents/skills/ibrobot-build) | 编译、colcon build、构建错误 |
| [ibrobot-launch](.agents/skills/ibrobot-launch) | 分平台启动 Ubuntu/openEuler 工作区或 OpenHarmony 板端机器人、仿真、mock/契约测试、推理、teleop |
| [ibrobot-architecture](.agents/skills/ibrobot-architecture) | 架构、SSOT、契约、robot_config、数据流 |
| [ibrobot-robot-skill-design](.agents/skills/ibrobot-robot-skill-design) | 交互式设计/新增机器人 skill、Hermes/Agent 动作、真机验证方案 |
| [ibrobot-control](.agents/skills/ibrobot-control) | Hermes/Agent 发现、校验、执行或取消现有机器人高层技能 |

### 板端（OpenHarmony）

| 技能 | 触发场景 |
|------|---------|
| [oh-constraints](.agents/skills/oh-constraints) | OpenHarmony 板端运行时约束（toybox/musl/只读 rootfs/无 systemd），板端操作前必读 |
| [oh-access](.agents/skills/oh-access) | HDC/SSH 连接板端、推送/拉取文件、SSH 配置 |
| [oh-build-roboframe](.agents/skills/oh-build-roboframe) | 主机侧交叉编译 RoboFrame 发布包（`build_roboframe_oh.sh`） |
| [oh-cross-build-ros-pkg](.agents/skills/oh-cross-build-ros-pkg) | 交叉编译第三方 ROS 2 包到板端 |
| [ohloha-build-pkg](.agents/skills/ohloha-build-pkg) | 用 tools_ohloha_pkgs 交叉编译第三方包（bash/zsh/vim 等）到板端 |
| [oh-rebuild-kernel](.agents/skills/oh-rebuild-kernel) | 重新编译并刷入 OpenHarmony 内核 |

### 模型

| 技能 | 触发场景 |
|------|---------|
| [om-convert](.agents/skills/om-convert) | Ascend OM 唯一入口、ACT/PI05 转换、新 policy 实验性支持开发及可选性能优化 |
| [rknn-convert](.agents/skills/rknn-convert) | ONNX 转 RKNN、NPU 部署、模型转换 |
| [hmm-convert](.agents/skills/hmm-convert) | PI0.5/SmolVLA 后摩 HMM 打包、xh2 编译、tcim、xhquant、LQ50/M50 |

### 工作流与验证

| 技能 | 触发场景 |
|------|---------|
| [ibrobot-git-flow](.agents/skills/ibrobot-git-flow) | git commit、git push、DCO sign-off |
| [ibrobot-lerobot-patch](.agents/skills/ibrobot-lerobot-patch) | 导出 lerobot patch、patch 栈管理 |
| [ibrobot-docker-verify](.agents/skills/ibrobot-docker-verify) | Ubuntu Docker 验证 setup+build |
| [ibrobot-docker-verify-oee](.agents/skills/ibrobot-docker-verify-oee) | openEuler aarch64 Docker 验证 |
| [sync-github](.agents/skills/sync-github) | 同步 AtomGit master 到 GitHub |
| [skill-creator](.agents/skills/skill-creator) | 新建/重构 Agent skill、编写 SKILL.md、按 agentskills.io 规范校验 |

### 文档工具

| 技能 | 触发场景 |
|------|---------|
| [deepwiki-config](.agents/skills/deepwiki-config) | 生成 DeepWiki `doc_config.json` 配置 |
| [deepwiki-translator](.agents/skills/deepwiki-translator) | DeepWiki Markdown 中文翻译、配置标题本地化 |
| [mermaid-syntax-validation](.agents/skills/mermaid-syntax-validation) | Mermaid 图语法检查、修复与渲染验证 |

### AtomGit 协作

| 技能 | 触发场景 |
|------|---------|
| [atomgit-collaboration](.agents/skills/atomgit-collaboration) | 泛化协作请求路由（先识别再分流） |
| [atomgit-pr](.agents/skills/atomgit-pr) | 创建 PR、更新 PR 描述、生成 PR 摘要 |
| [atomgit-issue](.agents/skills/atomgit-issue) | 创建/查看/更新/关闭 Issue |
| [atomgit-pr-review](.agents/skills/atomgit-pr-review) | 代码审查、PR review、检查 Bug |
| [atomgit-pr-architecture-review](.agents/skills/atomgit-pr-architecture-review) | 架构审查、SSOT 合规、契约检查 |
| [atomgit-review-resolution](.agents/skills/atomgit-review-resolution) | 修复评审意见、回复评论、闭环 review |

> 详见 `.agents/skills/README.md` 获取完整的分类说明和使用指南。
