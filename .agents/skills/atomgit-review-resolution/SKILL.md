---
name: atomgit-review-resolution
description: "AtomGit 评审意见处理工具。当用户需要“修复PR评审意见”、“处理review comments”、“apply fixes”、“reply to review”、“回复评论”、“resolve review discussions”、“闭环评审流程”或基于未解决评论继续推进 PR 时使用。只要目标是本仓库在 AtomGit 上的 review comments / discussions，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit Review Resolution

响应别人对你代码的审查意见，自动修复并回复评论。

在 IB_Robot 仓库中，只要用户提到 review comments / unresolved comments / 回复评论 / 修复评审意见 且未明确指定 GitHub，默认视为 AtomGit review follow-up 流程并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库的 PR 评论** 做通用跟进：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的 PR 链接自动解析 `owner/repo/pr_number`

## ⚠️ 依赖准备

本 skill 依赖 PyPI 包 `atomgit-sdk`，其 Python 导入模块名为 `atomgit_sdk`。
仓库默认通过 `requirements/*.txt` 安装；如果当前环境未安装，请先运行
`./scripts/setup.sh`，或在当前 Python 环境中安装 `atomgit-sdk`。

## ⚠️ 文件读取说明

**输出文件位于项目 `./tmp` 目录**，AI Agent 应使用 shell 命令读取：

```bash
# 读取未解决的评论
cat ./tmp/ib_robot_pr_123_unresolved_comments.json

# 读取修复结果（提交前确认）
cat ./tmp/ib_robot_pr_123_fix_results.json
```

## 快速使用

```bash
# 步骤1: 获取未解决的评论
python3 review_resolution.py --pr 123

# 直接从链接解析目标 PR
python3 review_resolution.py --url https://atomgit.com/some-org/some-repo/pull/123

# 步骤2: 你分析评论并生成修复方案

# 步骤3: 人类确认修复方案

# 步骤4: 应用、推送并在远端验证后回复（⚠️ 必须指定 --ai-model）
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --push --ai-model glm-5.2

# pending_push / pending_replies 后从原子状态文件恢复
python3 review_resolution.py --resume ./tmp/ib_robot_pr_123_review_resolution_<transaction-id>.json
```

**重要**: 
- 在步骤3，你必须将修复方案展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `glm-5.2`、`glm-5.1`、`gpt-5.6-sol`、`gpt-5.6-terra`、`claude-fable-5`、`claude-opus-5`）
- 代码修复项必须逐项提供 `fixup_target`；只有确认整批都属于同一 commit 时，才可改用全局 `--fixup-target`
- 不带 `--push` 时会完成本地 autosquash、写入恢复状态文件并返回 `pending_push`（退出码 2），不回复评论
- 回复部分失败时返回 `pending_replies`（退出码 2）；使用输出的 `--resume <state.json>` 只重试 pending 项
- 文件名格式：`./tmp/{repo}_pr_{number}_fix_results.json`
- 恢复状态格式：`./tmp/{repo}_pr_{number}_review_resolution_{transaction_id}.json`

### ⚠️ Commit 提交规范（遵循 ibrobot-git-flow §3 PR Commit Hygiene）

本脚本在 `--apply-fixes` 模式下会自动提交代码修复。提交方式遵循 ibrobot-git-flow skill 的 PR commit hygiene 规则。当前 SDK 不提供受支持的修复生成 API，因此 CLI 不暴露 `--auto`；修复方案必须由 Agent 分析并经人类确认后通过 JSON 输入。

**禁止的做法**（旧版本曾使用，现已移除）:
- ❌ `git commit --amend -m "fix: resolve review comments"`
- ❌ 创建任何形如 `fix: address review comments` / `fix reviewer's意见` / `apply review suggestions` 的噪声 commit
- ❌ `git add .`（会把无关文件一并暂存）
- ❌ 无 pathspec 的 `git add -u`（会暂存整个 working tree）
- ❌ `git add -u .` 全目录暂存回退（`files_fixed` 为空时直接拒绝，不猜测暂存范围）
- ❌ 在 index 已有用户预先暂存内容时自动提交
- ❌ 在 tracked worktree 已有 unstaged WIP 时重写历史
- ❌ 使用可变 branch ref 或裸 `--force-with-lease` 作为安全边界

**当前实现的元数据绑定工作流**:

1. 在写文件前检查 PR 的 openEuler AI 披露和目标 commit：模型只记录名称及版本，不携带 provider 前缀；PR 模型列表必须包含本次真实 `--ai-model`。不同 commit 可以保留各自实际使用的模型；若当前 AI 修复将 autosquash 到尚未记录本模型的目标 commit，应在历史整理时为该 commit 补充本模型 trailer 并同步 PR 模型列表，而不是因原有模型不同而阻止处理。
2. 在写文件前读取 PR `base.sha`、`head.sha`、`head.ref`、`head.repo`；要求本地 `HEAD == head.sha`，且 `base.sha` 是 HEAD 的祖先。
3. 通过 SDK 的 PR 作用域全量评论接口验证每个 `comment_id` 确实属于目标 PR，并保留 `discussion_id` 等元数据；校验发生在任何本地历史重写前。
4. 先校验全部 fixes；只有存在 code fix 时才要求 index 和 tracked worktree 均干净。纯 `reply_only` 批次不依赖本地 Git 状态。
5. `code_fix.original_code` 必须非空且在文件中恰好匹配一次。同一文件多项修复默认拒绝；唯一例外是同一 target 的多个 `delete_lines`，它们会合并行号并基于同一快照一次应用。
6. 按每项解析后的 `fixup_target` SHA 分组，逐组以 literal pathspec 执行 `git add -- <paths>` 和 `git commit --fixup=<sha>`，再仅对不可变 `base.sha` 执行一次 autosquash。同一文件跨多个目标时拒绝并要求拆批。
7. `base.sha..HEAD` 中存在 merge commit 时拒绝 autosquash，避免普通 rebase 静默线性化 PR 拓扑。
8. 异常、KeyboardInterrupt、SIGINT/SIGTERM/SIGHUP 会触发事务清理：只 abort 本次启动的 rebase，恢复原 HEAD/index，以及文件内容、存在状态和 mode。
9. 只接受一个 remote 上唯一、精确匹配 PR source repository 的 push URL；remote 有多个 pushurl，或多个 remote/URL 都匹配时拒绝。push 和 `ls-remote` 均使用这一 URL，并绑定 `--force-with-lease=<ref>:<old-head-sha>`。
10. autosquash 后立即原子写入恢复状态，记录 old/new HEAD、精确 URL/ref、回复正文/哈希/marker/discussion/status；远端 ref 验证为新 OID 后才发送修复完成回复。

**回复评论时序**：代码修复批次只有在 source branch 推送并验证成功后才发送“已修复”回复；本地成功但未推送时返回 `pending_push`，不伪装闭环。每条远端回复后都会原子 checkpoint；恢复时先按隐藏事务 marker/正文哈希对账，只发送 `pending` 项。纯 `reply_only` 批次也使用同一持久化回复账本。

**本地事务语义**：任一修复项处理失败（字段缺失、路径非法、行号越界、原文不唯一等）时，脚本恢复全部文件到本次运行前状态，不提交代码、不发送任何回复。push/reply 属于不可回滚远端副作用，因此通过状态文件逐阶段 checkpoint 和幂等恢复，而不是伪装成可回滚的“全有或全无”。每个修复项必须携带属于目标 PR 的 `comment_id`。

**关键 CLI 参数**:

- fixes 项中的 `fixup_target`: 每项代码修复要折回的目标 SHA/ref；必须位于 PR `base.sha..HEAD`。
- `--fixup-target <SHA|ref>`: 仅作为缺少逐项字段时的显式整批 fallback，无默认值。
- `--push`: 明确授权使用 PR source metadata、唯一精确 push URL 和显式 OID lease 推送；省略时返回 `pending_push`。
- `--resume <state.json>`: 恢复待推送/待回复事务；验证本地和远端 OID 后，必要时推送并只发送尚未完成的回复。
- `--base-branch`: 仅保留旧命令兼容，参数值被忽略；安全基线始终来自 PR metadata `base.sha`。

完整可复制示例、autosquash 失败行为、PR commit 数量约束见 `references/commit-hygiene-examples.md`。

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| 完整 CLI 参数表、回复单条评论的详细命令与 reply-mode 语义、审查者复查闭环流程、修改 discussion 解决状态、resolve 仅对 inline diff comment 有效的限制说明 | `references/cli-reference.md` |
| 修复类型、fix_results.json 输入格式 JSON 示例、获取未解决评论 / 应用修复方案 API 说明、SDK / API 设计决策 | `references/json-format-and-sdk.md` |
| Commit 提交规范的完整示例块（逐项 / 整批 fallback / pending_push / dry-run）、autosquash 失败行为、PR commit 数量约束 | `references/commit-hygiene-examples.md` |

Do not expose these references as separate skills.
