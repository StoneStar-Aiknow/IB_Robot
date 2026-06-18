---
name: atomgit-issue
description: "AtomGit Issue 工作流工具。当用户需要在本仓库或 AtomGit 上“创建Issue”、“查看Issue详情”、“更新Issue”、“关闭/重开Issue”、“create issue”、“fetch issue info”、“update issue”、“report bug”、“feature request”或围绕 Issue 做任何创建/读取/更新动作时调用。只要目标是本仓库的 Issue，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit Issue Workflow Tool

创建、读取、更新或关闭 Issue。

在 IB_Robot 仓库中，只要用户提到 Issue / bug / feature request 且未明确指定 GitHub，默认视为 AtomGit 工作流并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库** 指定目标：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的 Issue 或仓库链接自动解析 `owner/repo/issue_number`

## ⚠️ 环境准备

**重要**: 在使用此 skill 前，必须先加载环境配置：

```bash
source .shrc_local
```

这将把 `libs/atomgit_sdk/src` 添加到 PYTHONPATH
使 skill 能够导入 AtomGit SDK。

## ⚠️ 获取仓库配置（必需）

在使用前，建议通过 `git remote -v` 确认仓库的 owner 和 repo：

```bash
git remote -v
```

脚本会自动从环境变量或 `git remote` 中推断，也可以通过参数指定。

## 🏷️ 标签规范（openEuler 社区预定义，必读）

IB_Robot 沿用 [openEuler 社区标签约定](https://atomgit.com/openeuler/community/blob/master/zh/sig-infrastructure/label.md)。创建 / 分类 Issue 时**只能**从下列预定义标签中选取，**严禁**自造 `bug`、`feature`、`enhancement` 等裸名标签——仓库里不存在这些标签，SDK 的 `_validate_labels_exist` 会直接报 `Unknown labels`。

#### ⚠️ 直接使用下方词表，禁止自己去 API 获取 label

下列词表就是 IB_Robot issue 标签的**唯一来源（SSOT）**，选取标签时**直接查表**即可，**严禁**再通过 API（如 `client.get_labels()`、`GET /repos/:owner/:repo/labels`、或任何「列出仓库全部标签」的调用）去「发现」可用标签。原因：

- 仓库实际标签列表混合了大量个人 `lgtm-*`、`sig/*`、CI 类（`cam_*` / `ai_*` / `docs-ci-*`）标签，与 issue 分类无关，直接拉取会严重干扰判断；
- SDK 的 `get_labels()` **不分页**，只能取回第一页，对第二页及之后的标签会误判为不存在（已实证：`kind/feature` 在第二页，被误报 `Unknown labels`）；
- 词表已经过裁剪和校验，是 issue 分类应使用的正确子集。

正确流程永远是：**看下方词表 → 选定 `kind/<x>` / `priority/<x>` → 用 slash 命令让机器人打标**，不要绕道 API。

允许的标签词表：

- **Kind**（类型，`kind/<x>`）：`api-change`、`bug`、`cleanup`、`design`、`documentation`、`failing-test`、`feature`、`enhancement`
- **Priority**（优先级，`priority/<x>`）：`high`、`medium`、`low`
- **平台 / 社区默认**：`help-wanted`（寻求帮助）、`invalid`（无效的 PR/Issue）、`question`（此 Issue 为一个问题）、`newcomer`（第一次在社区提 PR/Issue）

> 命名注意：类型标签是 `kind/bug`、`kind/feature`，**不是**裸的 `bug`/`feature`。

### ⚠️ 应用方式

这些标签在 IB_Robot 仓库**默认不存在**，且当前 `ATOMGIT_TOKEN` 没有直接写标签的权限（`create-repo-labels` 返回 `403`）。因此 `issue_management.py` **不提供任何打标签的参数**（AtomGit 对 `kind/*` / `priority/*` 直传必然校验失败），标签一律通过 slash 命令添加：

- **正确做法**：先创建 Issue，再用 **slash command 评论**让 openEuler 社区机器人打标签：

```bash
# 创建 Issue（不打标签）
python3 issue_management.py --title "[Bug] xxx" --body "..."

# 用斜杠命令让机器人添加 kind/*/priority/* 标签（机器人会自动建标并打上）
python3 issue_management.py --issue <N> --comment "/kind bug"
python3 issue_management.py --issue <N> --comment "/priority high"
# 移除：/remove-kind bug、/remove-priority high
```

- 平台默认标签（`help-wanted`/`invalid`/`question`/`newcomer`）若仓库未预置，需仓库 admin 在 Settings → Labels 中先创建，再由 slash 命令或网页端添加。
- **指派（assignees）同样不提供 CLI 参数**：AtomGit 的 PATCH issue 接口不接受 `assignees` 字段，直传必然 400。指派走 slash 命令：

```bash
python3 issue_management.py --issue <N> --comment "/assign @someone"
# 指派给自己用 /assign；取消指派用 /unassign
```

## 快速使用

### 创建 Issue

```bash
# 提交一个简单的 Issue（标签请通过下方 slash 命令添加，不要在这里直接传 kind/*）
python3 issue_management.py --title "发现一个 Bug" --body "在执行 build.sh 时报错..."

# 创建后用社区 slash 命令打标签（机器人会添加 kind/bug）
python3 issue_management.py --issue <N> --comment "/kind bug"

# 跨仓库：直接指定 owner/repo
python3 issue_management.py --owner some-org --repo some-repo --title "[Bug] xxx"
```

### 获取 Issue 信息 (Agent 驱动)

当需要分析已有 Issue 时，Agent 可以调用：

```bash
python3 issue_management.py --issue 123 --fetch-info

# 直接从链接解析
python3 issue_management.py --url https://atomgit.com/some-org/some-repo/issues/123 --fetch-info

# 如只需要 Issue 主体，显式关闭评论抓取
python3 issue_management.py --issue 123 --fetch-info --no-comments
```
默认会一并抓取 Issue 评论并写入 `comments_detail` 字段。Agent 会读取生成的 `tmp/{repo}_issue_123_context.json`。

## API 说明

### issue_management.py

创建或更新 Issue。

**参数**:
- `--title`: Issue 标题（创建时**必需**）
- `--body`: Issue 描述
- `--issue`: Issue 编号（用于更新或获取信息，可由 `--url` 自动解析）
- `--state`: Issue 状态（open 或 closed，用于更新）
- `--fetch-info`: 提取 Issue 详情到 JSON 文件
- `--comment`: 在已有 Issue 下发表评论或社区斜杠命令（如 `/kind bug`）
- `--no-comments`: 在 `--fetch-info` 模式下跳过评论抓取
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: Issue 或仓库链接（可选，自动解析 `owner/repo/issue_number`）
- `--dry-run`: 仅显示计划，不执行实际操作

**示例**:
```bash
# 更新 Issue 状态
python3 issue_management.py --issue 123 --state closed

# 修改 Issue 标题和内容
python3 issue_management.py --issue 123 --title "已修正: 编译错误" --body "通过更新依赖已解决。"
```

## 注意事项

1. **环境配置**: 确保 `ATOMGIT_TOKEN` 已正确配置在环境变量中。
2. **Issue 规范**: 建议在标题中使用清晰的前缀，如 `[Bug]`, `[Feature]`, `[Task]` 等。
3. **标签管理**: 只使用 openEuler 社区预定义标签（见上方「标签规范」）。`kind/*`、`priority/*` 通过 `/kind`、`/priority` 斜杠命令（评论）由社区机器人添加，**不要**自造裸名标签（`bug`/`feature` 等）。CLI 不提供打标签参数，一律走斜杠命令。
