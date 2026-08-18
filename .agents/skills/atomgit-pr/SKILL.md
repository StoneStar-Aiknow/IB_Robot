---
name: atomgit-pr
description: "AtomGit PR 工作流工具。当用户需要在本仓库或 AtomGit 上“创建PR”、“更新PR描述”、“同步PR标题/正文”、“生成PR摘要”、“create pull request”、“open merge request”、“update PR description”、“generate PR summary”或围绕 PR 管理动作工作时调用。它负责 PR 资源的创建和维护，不负责通用代码 review；只要目标是本仓库的 PR / merge request 管理，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit PR Workflow Tool

创建新 PR、提取 PR 管理上下文或更新现有 PR 描述。

如果用户的目标是“**review 一个 PR / 帮我看看这个 PR 有没有问题 / 分析已有评论**”，不要使用本 skill，改用 `atomgit-pr-review`；如果目标是 SSOT / 契约 / 架构职责边界检查，改用 `atomgit-pr-architecture-review`。

在 IB_Robot 仓库中，只要用户提到 PR / merge request 且未明确指定 GitHub，默认视为 AtomGit 工作流并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库** 指定目标：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的仓库或 PR 链接自动解析 `owner/repo`

## ⚠️ 依赖准备

本 skill 依赖 PyPI 包 `atomgit-sdk`，其 Python 导入模块名为 `atomgit_sdk`。
仓库默认通过 `requirements/*.txt` 安装；如果当前环境未安装，请先运行
`./scripts/setup.sh`，或在当前 Python 环境中安装 `atomgit-sdk`。

## ⚠️ 获取 Fork Owner（必需）

在创建 PR 前，**必须**先通过 `git remote -v` 获取 fork owner：

```bash
git remote -v
```

输出示例：
```
origin    git@atomgit.com:YourName/IB_Robot.git (fetch)
origin    git@atomgit.com:YourName/IB_Robot.git (push)
upstream  git@atomgit.com:openEuler/IB_Robot.git (fetch)
upstream  git@atomgit.com:openEuler/IB_Robot.git (push)
```

从中提取 fork owner（即个人仓库的用户名，如 `YourName`），然后通过 `--fork-owner` 参数传递给脚本。

## 快速使用

### 创建 PR (推荐 Agent 方式)

Agent 在创建 PR 时，**必须**遵循 [PR #32](https://atomgit.com/openeuler/IB_Robot/pull/32) 的极高专业水准。描述文件应围绕本次提交真正的审阅重点组织内容；复杂流程或架构变化优先使用 **Mermaid 图表**，简单或纯文档类变更不要机械套用重型模板。

当变更命中双平台 Docker 门禁时，在调用任何 Docker skill 前必须用交互问题让用户选择：

- **WIP 初步提交**：传 `--pr-stage wip`，标题自动变为 `[WIP] <title>`，暂缓双平台 Docker。
- **正式检视**：传 `--pr-stage review`，移除 `[WIP]`，执行并校验当前 tree 的双平台 Docker。

不得替用户默认选择正式检视并直接启动耗时验证，也不得用 WIP 豁免其他合规检查。

**PR 描述强制要求：**

1.  **默认使用中文撰写**：PR 标题和正文描述均默认使用中文。仅在用户明确要求时切换为英文。
2.  **超链接使用**: 对相关的 Issue、PR、技术规范或设计文档，**必须**使用 Markdown 超链接进行关联（如 `[PR #32](https://atomgit.com/openeuler/IB_Robot/pull/32)`），以方便审阅者查阅背景。
3.  **深度结构化内容**:
    *   **按提交内容动态组织**: 不要机械要求每个 PR 都包含同一组标题。围绕 commit 真正的变化点组织内容，确保审阅者快速看到最重要的信息。
    *   **背景与动机**: 说明问题根源、业务痛点或需求背景；简单修复可简写，但不要省略必要上下文。
    *   **方案概述**: 描述解决思路、关键设计决策。只有在流程、状态转换或架构关系较复杂时，才使用 Mermaid 流程图或时序图；不要为了凑模板硬加图。
    *   **技术细节**: 按模块拆解代码、配置、脚本或文档层面的关键变更，解释为什么这样改。
    *   **README / 文档联动**: 如果提交改变了用户可见的安装、构建、运行、依赖、接口、配置或使用方式，应同步更新对应 README / 使用文档；如果判断不需要改文档，也应基于变更内容做出明确判断，而不是机械忽略。
    *   **影响范围**: 仅在确有影响时说明对系统行为、接口、依赖、性能、部署或使用方式的影响；无明显影响时可简洁说明。
    *   **验证结果（条件性章节）**:
        *   只有当本次 PR 做过**真实验证**且该验证对审阅结论有价值时才写。
        *   必须写清楚 **Scenario（什么场景下验证）**、**Method（如何验证，可含命令）**、**Result（验证结果是什么）**。
        *   禁止把 `git diff`、`git status`、文件列表这类仅用于查看变更的命令当作 Verification。
        *   对纯文档、注释、gitignore、纯元数据等**不涉及运行时行为**的提交，可以省略 Verification，而不是生硬补一个无意义小节。
        *   如果正式检视 PR 修改了 ROS 包的 `package.xml` 依赖声明，或修改了全局 setup/build 流程相关文件（如 `scripts/setup.sh`、`scripts/build.sh`、`scripts/setup/platforms/*.sh`、`scripts/setup/verify_env.sh`、`scripts/setup/python_venv.sh`、`scripts/install_ros.sh`、顶层 `CMakeLists.txt`、顶层 `pyproject.toml`、`requirements/*.txt` 等直接影响 pip/rosdep 依赖安装的文件），则 Verification **必须提供**，且必须包含双平台纯净 Docker `setup.sh + build.sh` 完整验证结果。ROS 包内的 `setup.py` 普通改动不单独触发该门禁。
         *   正式检视阶段触发上述门禁时，Agent 记录目标 commit 的完整 `git rev-parse HEAD^{tree}`，再调用两个 Docker skill 验证该 tree 的隔离快照。当前工作区无需 clean，但直接复制 dirty 工作区的结果只能用于本地调试，不能作为 PR 证据。Verification 必须且只能包含一个标准字段 `**Verified tree:** \`<40位 SHA>\``，并写入两平台真实结果。
         *   创建或更新 PR 时，脚本会将该 tree SHA 与已推送分支或 AtomGit PR 最新 head commit 的 tree 比对。源码 tree 改变会阻止创建/更新并要求重跑；只修改 commit message、作者或 trailer 而 tree 不变时，已有结果仍然有效。提交 PR 属于作者侧发布流程，不同于 review；不要把 review 的“只检查开发者声明”规则套用到本 skill。
         *   在真正启动双平台 Docker 前，Agent 必须询问用户当前 PR 是临时 WIP 还是准备交给 reviewer 正式检视。命中门禁时，`pr_creation.py` / `pr_management.py` 要求显式传入 `--pr-stage wip|review`：`wip` 会把标题规范化为 `[WIP] <title>` 并暂缓 Docker；`review` 会移除 `[WIP]` 并恢复 tree-bound 门禁。WIP 只豁免双平台 Docker 证据，不豁免 DCO、AI 披露、其他测试或 CI。
4.  **openEuler AI 贡献披露**：Agent 创建或更新 PR 时必须提供真实的 Agent 平台及版本、AI 模型名称及版本、Prompt 摘要、人工审查确认，以及第三方材料来源和许可证信息。提交/更新前，coding agent 必须自行执行实际工具的 `<tool> --version`（或等价版本命令），并将工具名和版本传给 `--agent-tool`；仓库不维护工具白名单，也不替未知工具执行命令，脚本只校验结构、占位符和注入字符。模型字段只记录模型本身（如 `gpt-5.6-sol`），不携带 `xunxing/` 等 provider 前缀；同一 PR 使用多个模型时以逗号分隔并完整列出。脚本要求至少一个 AI-assisted commit 包含 `Co-Authored-By`，并检查 PR 披露覆盖所有 commit 实际记录的 AI 模型；不同 commit 可以使用不同模型，纯人工 commit 也不要求添加 AI trailer。人类共同作者应使用 `Co-Authored-By: Name <email>`，不会被当作 AI 模型。缺失、未披露或无法验证的工具/模型信息会阻止提交。禁止使用 `ai`、`agent`、`unknown` 等占位值。完整政策见 [openEuler 社区生成式AI工具使用与开源贡献策略](https://www.openeuler.openatom.cn/zh/community/ai-coding-assistants/)。

```bash
# 1. 获取变更信息（仅用于分析变更，不可直接当作 Verification）
git diff upstream/master..HEAD

# 2. Agent 深度分析并生成专业描述文件 pr_description.md
# 根据 commit 内容选择合适章节；仅在做过真实验证时包含 Verification。
# Mermaid 仅用于能显著提升理解的复杂流程或架构变更。
# 如果变更影响用户使用方式，要判断并同步 README / 使用文档。
# 如果变更触发双平台门禁，先询问用户并选择 --pr-stage wip 或 review。
# WIP 自动添加 [WIP] 前缀并跳过 Docker；review 才验证隔离 tree 快照并写入
# **Verified tree:** `<40位 SHA>`。
# 创建/更新 PR 前先运行实际 Agent 工具的版本命令，例如 `opencode --version`，再把完整输出中的
# 工具名和版本（如 `OpenCode 1.17.20`）传给 `--agent-tool`。

# 3. 创建 PR
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu \
  --title "feat(scope): technical summary" --description-file pr_description.md \
  --pr-stage review \
  --agent-tool "OpenCode 1.17.20" --ai-model "gpt-5.6-sol" \
  --prompt-summary "Implement the requested feature and verify the affected workflows" \
  --third-party-materials "无" --human-reviewed
```

### 基础用法

```bash
# 步骤1: 获取 fork owner
git remote -v

# 步骤2: 创建 PR（章节按实际变更组织；Verification 仅在存在真实验证时提供）
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --title "fix: specific issue" \
  --description-file pr_description.md \
  --agent-tool "OpenCode 1.17.20" --ai-model "gpt-5.6-sol" \
  --prompt-summary "Fix the reported issue and add focused verification" \
  --third-party-materials "无" --human-reviewed

# 如果本次变更做过真实验证，再补充 Verification 小节，写清场景 / 方法 / 结果

# 跨仓库：直接指定目标仓库；描述始终从 Markdown 文件读取
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --owner some-org --repo some-repo \
  --description-file pr_description.md

# 跨仓库：从链接自动解析 owner/repo
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu \
  --url https://atomgit.com/some-org/some-repo --description-file pr_description.md
```

### 生成/更新 PR 描述 (Agent 驱动)

当需要为已有 PR 生成高质量描述时，遵循以下 Agent 工作流：

**步骤 1: 提取 PR 上下文**
```bash
python3 pr_management.py --pr 123 --fetch-info

# 默认会包含 PR 评论；如只看提交和 Diff，可显式关闭
python3 pr_management.py --pr 123 --fetch-info --no-comments
```
Agent 会读取生成的 `tmp/{repo}_pr_123_context.json`，其中默认包含提交记录、修改文件、代码 Diff (patch) 以及 PR 评论。
`metadata.head_sha` 用于定位 PR 最新 commit，`metadata.head_tree` 是正式检视门禁比较的 tree SHA；`metadata.wip` 和 `metadata.dual_docker_gate_required` 分别标识 WIP 状态和文件门禁。仅 commit 元数据变化且 tree 不变时可以沿用 Docker 结果。

**步骤 2: Agent 分析与同步**
Agent 分析完 Diff 后，会生成一份 `description.json`:
```json
{
  "title": "feat: 新功能标题",
  "description": "详细的变更逻辑说明..."
}
```
然后运行同步命令：
```bash
python3 pr_management.py --pr 123 --update-pr description.json \
  --pr-stage review \
  --agent-tool "OpenCode 1.17.20" --ai-model "gpt-5.6-sol" \
  --prompt-summary "Synchronize the PR description with the complete branch diff" \
  --third-party-materials "无" --human-reviewed
```

## API 说明

### pr_creation.py

创建新的 Pull Request。

**参数**:
- `--branch`: 分支名（可选，默认当前分支）
- `--fork-owner`: Fork 仓库的 owner（**必需**，通过 `git remote -v` 获取）
- `--title`: PR 标题（可选，自动生成）
- `--description-file`: PR 描述 Markdown 文件（必需）。PR 创建不接受 `--body`，避免 shell 转义破坏多行 Markdown
- `--base`: 目标分支（默认：master）
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: AtomGit / GitCode 仓库或 PR 链接（可选，自动解析 `owner/repo`）
- `--draft`: 创建草稿 PR（可选）
- `--pr-stage`: `wip` 或 `review`；命中双平台门禁时必须在询问用户后指定。`wip` 自动添加 `[WIP]` 标题前缀并跳过 Docker，`review` 移除前缀并执行门禁
- `--dry-run`: 仅显示计划，不创建
- `--agent-tool`: coding agent 通过实际 `<tool> --version`（或等价命令）确认后的工具名和版本（必需，如 `OpenCode 1.17.20`）。脚本不维护工具白名单，只校验具体版本格式和安全字符；不得传裸工具 ID、`latest`、`unknown` 或臆填版本
- `--ai-model`: AI 模型名称及版本（必需，不含 provider 前缀；多个模型用逗号分隔，必须覆盖 commit trailer 中的所有 AI 模型）
- `--prompt-summary`: 核心提示词或核心意图摘要（必需）
- `--third-party-materials`: 第三方材料、来源及许可证；没有时明确写“无”（必需）
- `--human-reviewed`: 确认开发者已人工审查；非 dry-run 创建时必需

**示例**:
```bash
# 完整示例
python3 pr_creation.py --branch feat/new-feature --fork-owner BreezeWu \
  --description-file pr_description.md

# 指定标题
python3 pr_creation.py --branch feat/new-feature --fork-owner BreezeWu \
  --title "feat: add new feature" --description-file pr_description.md
```

### pr_management.py

管理和维护已有 PR 的数据。

**模式**:
1. `--pr <NUM> --fetch-info`: 提取 PR 的完整上下文 (提交、文件、Diff)，Agent 学习用。
2. `--pr <NUM> --update-pr <JSON>`: 将 Agent 生成的描述同步到服务器。

**参数**:
- `--pr`: PR 编号（可由 `--url` 自动解析）
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: PR 链接（可选，自动解析 `owner/repo/pr_number`）
- `--output-dir`: JSON 输出目录 (默认: ./tmp)
- `--no-comments`: 在 `--fetch-info` 模式下跳过 PR 评论抓取
- `--pr-stage`: `wip` 或 `review`；更新命中门禁的 PR 时必须显式指定
- `--agent-tool` / `--ai-model` / `--prompt-summary` / `--third-party-materials`: 更新 PR 时必需的 openEuler AI 披露元数据；工具版本须由 coding agent 在调用前自行执行版本命令确认
- `--human-reviewed`: 确认开发者已人工审查；非 dry-run 更新时必需
- `--dry-run`: 预览生成的描述但不执行更新

## PR 描述格式

PR 描述**默认使用中文**撰写（标题仍用英文以符合 commit 规范，正文描述用中文）。如果用户明确要求英文，则切换为英文。

PR 描述通常应包含与本次提交最相关的内容，而不是固定模板。常见章节包括：

- **背景与动机**：为什么要改
- **方案概述 / 技术细节**：改了什么、为什么这样改
- **README / 文档联动**：用户可见使用方式变更时，说明同步更新了哪些文档；若无需更新，也应基于变更内容判断
- **影响范围**：影响范围与风险
- **验证结果（可选）**：仅在存在真实验证时写清场景、方法与结果

对于纯文档、注释、`.gitignore`、说明文字等不涉及运行时行为的 PR，可以不写 Verification。
但若变更涉及 ROS 包 `package.xml` 依赖声明或全局 setup/build 流程（含 `requirements/*.txt`），Agent 必须先询问 PR 阶段。`[WIP]` PR 可暂缓双平台 Docker；正式检视 PR 的 Verification 为**必填**，且必须覆盖 Ubuntu 与 openEuler 纯净 Docker 的 `setup.sh + build.sh` 完整验证。ROS 包内 `setup.py` 普通改动不单独触发该门禁。正式检视前，Agent 必须记录目标 commit 的完整 tree SHA，让两个 Docker skill 验证该 tree 的隔离快照，并写入唯一标准字段 `**Verified tree:** \`<40位 SHA>\``。脚本将此字段与最新 head tree 强制比对；移除 `[WIP]` 时门禁立即恢复。

## 注意事项

1. **分支命名**: 建议使用 `feat/`, `fix/`, `docs/`, `refactor/` 等前缀
2. **提交历史规范的唯一入口**: 创建或更新 PR 前，必须调用 `ibrobot-git-flow` 执行 commit hygiene 检查。提交格式、commit 数量、review 修复应折回已有 commit 还是作为独立新 commit，以及历史重写/推送方式，均以该 skill 的当前规则和例外为唯一事实来源；本 skill 不复制或覆盖这些判定。
3. **AI 元数据完整性**: PR 的模型信息必须覆盖所有 AI-assisted commit 的 `Co-Authored-By`；不同 commit 可记录不同模型，未披露模型才是阻塞错误。
4. **代码审查**: 创建 PR 后等待代码审查
5. **CI 检查**: 确保 CI 通过后再合并
6. **跨仓库前提**: 创建 PR 时当前本地 worktree 仍需与目标仓库代码相匹配；`--owner/--repo/--url` 只负责切换 AtomGit API 目标，不会替你切换本地 Git 工作区
