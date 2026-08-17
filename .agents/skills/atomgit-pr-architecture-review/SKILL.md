---
name: atomgit-pr-architecture-review
description: "AtomGit PR 架构评审工具。当用户需要“架构审查PR”、“review architecture compliance”、“检查架构规范”、“审阅PR架构”、“SSOT 合规性检查”、“契约驱动检查”或对指定 PR 做架构维度扫描并提交检视意见时调用。只要目标是本仓库的 PR 架构评审，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
license: MIT
---

# AtomGit PR Architecture Review

提取 PR 信息并提交 IB_Robot 架构合规审查评论到 AtomGit。

在 IB_Robot 仓库中，只要用户提到 PR 的架构评审、SSOT 合规性、契约驱动检查且未明确指定 GitHub，默认视为 AtomGit 架构评审流程并优先使用本 skill。

如果只是通用代码 review、逻辑检查或“帮我看看这个 PR”，应优先使用 `atomgit-pr-review`，不要直接进入本 skill。

## ⚠️ 依赖准备

本 skill 依赖 PyPI 包 `atomgit-sdk`，其 Python 导入模块名为 `atomgit_sdk`。
仓库默认通过 `requirements/*.txt` 安装；如果当前环境未安装，请先运行
`./scripts/setup.sh`，或在当前 Python 环境中安装 `atomgit-sdk`。

## ⚠️ 文件读取说明

**输出文件位于项目 `./tmp` 目录**，AI Agent 应使用 shell 命令读取：

```bash
# 读取 PR 信息
cat ./tmp/ib_robot_pr_123_arch_info.json

# 读取架构审查结果（提交前确认）
cat ./tmp/ib_robot_pr_123_arch_issues.json
```

**PR 正文**（AI 声明、Verification 等）在 `.pr.body` 字段。

## 快速使用

```bash
# 步骤1: 提取 PR 信息
python3 architecture_review.py --pr 123

# 步骤2: 你分析代码架构并生成 arch_issues.json

# 步骤3: 人类确认审查结果

# 步骤4: 提交审查结果（⚠️ 必须指定 --ai-model）
python3 architecture_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_arch_issues.json --ai-model glm-5.2
```

**重要**: 
- 在步骤3，你必须将审查结果展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `glm-5.2`、`glm-5.1`、`gpt-5.6-sol`、`gpt-5.6-terra`、`claude-fable-5`、`claude-opus-5`）
- 文件名格式：`./tmp/{repo}_pr_{number}_arch_issues.json`

## ⚠️ 禁止本地重复执行 pre-commit 已覆盖的检查

- IB_Robot 的 `.pre-commit-config.yaml` 已把 `ruff --fix` 与 `ruff-format` 作为强制 pre-commit hook，且 `.git/hooks/pre-commit` 随仓库安装；开发者提交时必然已通过 ruff 校验，**PR 上线代码不会再有 `ruff check` / `ruff format` 报错**。
- 架构审查关注的是 SSOT / 契约 / 控制流 / 包职责 / README 一致性，这些**全部可以通过静态阅读 diff 与本仓库源码（`Read` / `Grep` / `Glob`）判断**，不需要执行任何命令。
- 因此 review 时**禁止**在本地做以下动作：
  - `git apply` / `git checkout` / `git diff` 拼接 PR diff 后跑 `ruff check` / `ruff format --check` / `pyright` / `mypy` / `py_compile`
  - 切到 PR 分支跑 `colcon build` 来“验证”代码能否编译——验证属于开发者侧 Verification，由代码 review skill 的门禁条款管理
- 如怀疑某行存在风格 / 类型 / 命名问题，**直接在 inline 评论中指出并附修复建议**，由开发者在下一次提交时让 pre-commit 自动修复，不要在本地复跑。
- **例外**：用户在当前请求中明确要求“你帮我跑一下 ruff / typecheck / build 看看”时才执行相应命令；“架构审查这个 PR”“review 架构合规”本身**不构成**授权。

## 架构审查支柱

此工具会检查以下 IB_Robot 架构支柱：

1. **SSOT (Single Source of Truth)**
   - 配置来源唯一性
   - 数据流一致性
   - API 契约统一性

2. **Contract-Driven Design**
   - 接口定义完整性
   - 依赖注入模式
   - 契约验证

3. **Control Mode Architecture**
   - 控制模式分离
   - 状态管理一致性
   - 控制流清晰度

4. **Package-Specific Architecture Compliance**
   - **核心原则**: 每个 ROS 包必须遵循其在 `ibrobot-architecture` 中定义的职责边界。
   - **关键检查**: 
     - 检查改动是否符合该包的设计初衷（如 `robot_teleop` 应保持轻量，不应引入运动学 IK 或重型规划逻辑）。
     - 验证包之间的依赖是否符合分层设计，严禁职责越界。
     - 识别“职责蔓延”：如果一个简单的驱动包开始处理复杂的业务逻辑，必须提出警告。

5. **README Documentation Consistency**
   - **核心原则**: 每个包的 `README.md` 是该包对外暴露的本地架构契约，必须与代码行为、职责边界、启动方式、配置入口、数据流和限制保持同步。
   - **关键检查**:
     - 如果 PR 修改了某个包的核心职责、公开接口、launch 参数、配置项、数据流、依赖边界或使用方式，必须检查该包 README 是否同步更新。
     - 如果 README 在 PR 中被修改，必须反向检查 README 描述是否真实反映代码实现，避免文档夸大、过时或描述未实现能力。
     - 如果代码变更使 README 关键说明失效、缺失或误导使用者，应作为架构问题提出，而不是当作普通文档风格问题忽略。

## 架构审查协议 (Mandatory Protocol)

在分析代码前，你 **必须** 遵循以下协议以确保对各包职责有准确认知：

1. **同步上下文 (Context Sync)**: 
   - 针对 PR 涉及的每一个包（如 `robot_teleop`），首先调用 `ibrobot-architecture` skill 或读取该包根目录下的 `README.md`。
   - 确认该包的设计初衷、职责边界及禁止的行为（如禁止引入运动学、禁止直接操作硬件等）。
2. **定位变更层级**: 判断变更文件位于哪一层（硬件层、驱动层、业务层、模型层）。
3. **依赖审计**: 检查是否引入了不符合分层原则的跨包依赖。
4. **SSOT 验证**: 检查配置是否统一来自 `robot_config`，严禁硬编码。
5. **职责合规性判断**: 根据步骤 1 获取的契约，判断当前改动是否导致了“职责蔓延”或“架构违越”。
6. **README 一致性审计 (Mandatory)**:
   - 对 PR 涉及的每个包，必须读取最近的包级 `README.md`（通常为 `src/<package>/README.md`）。
   - 对比 README 与本次代码改动，判断 README 是否仍准确描述包职责、关键 API/CLI/launch/config、数据流、依赖边界和已知限制。
   - 若代码变更影响用户使用或架构契约但 README 未同步，必须输出 `pillar: "docs"` 的架构问题。
   - 若 README 被修改但与代码实现不一致，也必须输出 `pillar: "docs"` 的架构问题。
7. **输出审查结果**: 生成符合 JSON 格式的 `arch_issues.json`。

## API 说明

### 提取 PR 信息

```bash
python3 architecture_review.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_arch_info.json`

### 提交架构审查

```bash
python3 architecture_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_arch_issues.json --ai-model glm-5.2
```

## ⚠️ JSON 格式规范

`arch_issues.json` 必须是数组格式（非对象），每项含 `file` / `line` / `title` / `description` / `severity` / `pillar` 必填字段。完整字段说明、JSON 示例和验证命令见 `references/json-format.md`。

## 架构问题严重性

- 🔴 **critical**: 违反核心架构原则
- 🟠 **error**: 重要架构问题
- 🟡 **warning**: 架构建议改进
- 💡 **suggestion**: 最佳实践建议

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| arch_issues.json 完整字段说明、JSON 示例、验证命令 | `references/json-format.md` |

Do not expose these references as separate skills.
