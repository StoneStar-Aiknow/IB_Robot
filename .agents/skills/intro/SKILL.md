---
name: intro
description: 'IB-Robot 技能引导入口。当用户输入"介绍"、"有哪些功能"、"有哪些skill"、"我应该用哪个skill"、"help"、"帮助"、"入门"、"intro"、"getting started"、"what skills"、"list skills"、"available commands"、"能做什么"、"怎么用"或首次接触项目时使用。作为所有其他 skill 的统一导航起点，展示分类列表、使用示例并根据仓库当前状态推荐最合适的 skill。'
---

# 🤖 IB-Robot Copilot Skill 引导中心

Agent 在触发本 skill 时，**必须首先**向用户展示以下欢迎文案（原样输出，不做修改）：

> 🤖 **欢迎使用 IB-Robot AI Agent！**

---

## 📋 技能分类列表

### 🤖 机器人操作

| Skill | 一句话描述 |
| :--- | :--- |
| **ibrobot-launch** | 分平台启动 Ubuntu/openEuler 或 OpenHarmony 板端机器人、仿真、推理与遥操作 |
| **ibrobot-build** | 编译整个工作空间或指定 package（`colcon build`） |
| **ibrobot-env** | 初始化运行环境，加载 `.shrc_local`、设置 `ROS_DOMAIN_ID` |
| **ibrobot-architecture** | 理解 SSOT 架构设计、配置规范与数据流 |
| **ibrobot-robot-skill-design** | 交互式设计机器人 skill，明确 anchor、动作空间、安全链路、MCP 暴露和验证计划 |

### 🔌 板端（OpenHarmony）

| Skill | 一句话描述 |
| :--- | :--- |
| **oh-constraints** | OpenHarmony 板端运行时约束（toybox/musl/只读 rootfs/无 systemd），板端操作前必读 |
| **oh-access** | 连接 OpenHarmony 开发板，执行 hdc shell / file send / file recv、SSH 配置 |
| **oh-build-roboframe** | 用 `build_roboframe_oh.sh` 主机侧交叉编译 IB_Robot 自有 OpenHarmony 包，并确认 lerobot OH patch 真正进入产物 |
| **oh-cross-build-ros-pkg** | 将第三方 ROS 2 包（如 usb_cam）交叉编译移植到 OpenHarmony 板 |
| **ohloha-build-pkg** | 用 `tools_ohloha_pkgs` / `builder.sh` 交叉编译第三方包（bash、zsh、vim 等）到 OpenHarmony 板端 |
| **oh-rebuild-kernel** | 重新编译并刷入 OpenHarmony 内核 (boot_linux.img)，启用 USB ACM 等驱动 |

### 🧠 模型

| Skill | 一句话描述 |
| :--- | :--- |
| **rknn-convert** | 将 ONNX 模型转换为 RKNN，并明确主 venv 导出 ONNX、`.venv-rknn` 转 RKNN 的分层流程 |
| **hmm-convert** | 将 PI0.5 / SmolVLA 编译产物打包为后摩 HMM deployment（xh2 NPU）；ACT HMM 不支持 |

### 🚀 工作流与验证

| Skill | 一句话描述 |
| :--- | :--- |
| **ibrobot-git-flow** | 规范提交代码，确保符合 openEuler DCO/Commit 规范，push 后自动同步 PR 描述 |
| **ibrobot-lerobot-patch** | 将 `libs/lerobot` 的本地改动导出为受管 patch，并同步 `series.txt` / `manifest.yaml` / 测试夹具 |
| **ibrobot-docker-verify** | 在干净 Ubuntu 22.04 Docker 容器中端到端验证 setup.sh + build.sh |
| **ibrobot-docker-verify-oee** | 在 openEuler Embedded (aarch64) Docker 容器中端到端验证 setup.sh + build.sh |
| **sync-github** | 将 AtomGit master 分支同步推送到 GitHub |

### 📚 文档工具

| Skill | 一句话描述 |
| :--- | :--- |
| **deepwiki-config** | 根据 DeepWiki 目录结构生成 `deepwiki_processor.py` 所需的 `doc_config.json` |
| **deepwiki-translator** | 按 config-first 流程将 DeepWiki 英文 Markdown 翻译为中文文档 |
| **mermaid-syntax-validation** | 检查、修复并验证 Markdown/Sphinx 文档中的 Mermaid 图语法 |

### 🔍 代码协作

| Skill | 一句话描述 |
| :--- | :--- |
| **atomgit-collaboration** | 拦截泛化的 AtomGit 协作请求，并分流到 PR / Issue / review / comment 对应流程 |
| **atomgit-pr-review** | 对 PR 进行代码质量审查，并读取已有评论一起判断风险 |
| **atomgit-pr-architecture-review** | 检查 PR 是否符合 SSOT、契约驱动等架构规范 |
| **atomgit-review-resolution** | 根据 AtomGit 上的检视意见修复代码、回复评论并闭环 |

### 📝 项目管理

| Skill | 一句话描述 |
| :--- | :--- |
| **atomgit-pr** | 创建、读取或更新合并请求（PR），自动生成/同步描述 |
| **atomgit-issue** | 创建、读取或管理 Issue，报告 Bug、提出建议 |

---

## 💡 使用示例

只需用自然语言告诉 Agent 你想做什么：

```
帮我审查 #25 号 PR              → atomgit-pr-review
帮我看看 PR #25 有没有问题      → atomgit-pr-review
帮我看看 PR #25                 → atomgit-collaboration
帮我看看这个 AtomGit PR 链接      → atomgit-collaboration
帮我更新 PR 描述                → atomgit-pr
帮我提交一个 Issue              → atomgit-issue
修复 PR 里的评审意见            → atomgit-review-resolution
编译一下项目                    → ibrobot-build
板端写脚本/部署前看约束         → oh-constraints
看看板子上的 OH 环境    → oh-constraints
连接板端并推送文件              → oh-access
编译 OpenHarmony 板端 IB_Robot     → oh-build-roboframe
用 build_roboframe_oh.sh 构建 → oh-build-roboframe
把 ONNX 转成 RKNN               → rknn-convert
把 PI0.5/SmolVLA 打包成后摩 HMM  → hmm-convert
把 usb_cam 移植到板端            → oh-cross-build-ros-pkg
编译 bash/zsh/vim 到板端         → ohloha-build-pkg
重新编译板端内核                 → oh-rebuild-kernel
启动机器人仿真或 OpenHarmony 板端运行 → ibrobot-launch
设计一个新的机器人动作           → ibrobot-robot-skill-design
让 Hermes 调用一个庆祝动作       → ibrobot-robot-skill-design
初始化环境                      → ibrobot-env
提交代码                        → ibrobot-git-flow
把 libs/lerobot 的改动做成 patch  → ibrobot-lerobot-patch
导出 lerobot patch 并更新 series  → ibrobot-lerobot-patch
Docker 验证一下 setup 和 build   → ibrobot-docker-verify
验证 openEuler 构建              → ibrobot-docker-verify-oee
同步到 GitHub                   → sync-github
生成 DeepWiki 配置               → deepwiki-config
翻译 DeepWiki 文档               → deepwiki-translator
检查 Mermaid 图语法              → mermaid-syntax-validation
检查架构合规性                  → atomgit-pr-architecture-review
解释系统架构                    → ibrobot-architecture
有哪些功能 / help / 入门       → intro (本技能)
```

---

## 🎯 当前推荐（上下文感知）

Agent 在触发本 skill 时，**必须**执行以下脚本来获取基于仓库当前状态的智能推荐：

```bash
source .shrc_local && python3 .agents/skills/intro/scripts/intro.py
```

脚本会检测以下仓库状态并输出推荐信息：

| 检测条件 | 推荐 Skill |
| :--- | :--- |
| 有未提交的代码改动 (`git status`) | `ibrobot-git-flow` 或 `atomgit-pr` |
| 编译产物缺失（`install/` 目录为空或不存在） | `ibrobot-build` |
| 存在 open 状态的 PR 且有未回复评论 | `atomgit-review-resolution` |
| 无特殊状态 | 展示「今日推荐」skill |

Agent 应将脚本输出的推荐内容展示给用户，帮助用户快速进入正确的工作流。

## 🌐 AtomGit 路由优先级

当用户在 IB_Robot 仓库里提到 PR / merge request / Issue / review / comments，且没有明确说 GitHub / github.com 时，Agent 应默认优先选择 AtomGit 相关 skill，而不是 GitHub 默认能力。

如果用户的表达仍然比较泛，例如“帮我看看这个 PR / Issue / 评论”，优先先进入 `atomgit-collaboration`，再分流到具体 skill。

对通用的 `atomgit-pr`、`atomgit-issue`、`atomgit-pr-review`、`atomgit-review-resolution`，Agent 可以使用 `--owner` / `--repo` / `--url` 指向其他 AtomGit 仓库；但 `atomgit-pr-architecture-review` 仍然只服务于 IB_Robot。

---

## 🔧 技术细节

- **环境依赖**: 执行推荐检测脚本前需先 `source .shrc_local` 加载环境
- **配置文件**: AtomGit 相关功能依赖 `config.json` 中的 Token 配置
- **脚本位置**: `.agents/skills/intro/scripts/intro.py`
