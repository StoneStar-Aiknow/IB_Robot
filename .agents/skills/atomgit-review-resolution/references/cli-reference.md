# CLI Reference & Reply / Resolve Workflows

## When to Read

- 需要查询某个 CLI 参数的完整语义与默认值
- 用户要求「只回复某条评论」「针对 comment_id 回复」「这条意见单独回一下」
- 审查者一方需要复查已修复意见、闭环自己提交的 review
- 需要修改某条 discussion 的 resolved 状态，或遇到 `--resolve-comment` 返回 HTTP 400

## 完整 CLI 参数补充表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pr` | PR 编号（可由 `--url` 自动解析） | 必填或由 `--url` 解析 |
| `--owner` | 目标仓库 owner（覆盖 `config.json`） | `config.json` |
| `--repo` | 目标仓库 repo（覆盖 `config.json`） | `config.json` |
| `--url` | PR 链接，自动解析 `owner/repo/pr_number` | — |
| `--reply-comment` | 回复指定 PR review 评论 ID | — |
| `--reply-body` | 直接传入回复正文 | — |
| `--reply-file` | 从文件读取回复正文 | — |
| `--reply-mode` | 回复模式，`threaded` 或 `visible` | `threaded` |
| `--resolve-comment` | 按评论 ID 修改其 discussion 解决状态 | — |
| `--resolve-discussion` | 按 discussion ID 修改解决状态 | — |
| `--resolved` | 解决状态，`true` 或 `false` | `true` |
| `--fixup-target` | 缺少逐项 `fixup_target` 时的显式整批 fallback | 无 |
| `--push` | 授权按 PR source metadata 和显式 OID lease 推送；验证后才回复 | `false` |
| `--resume` | 从原子 JSON 状态恢复待推送或待回复事务 | — |
| `--base-branch` | 旧命令兼容参数，已忽略；实际基线固定为 PR `base.sha` | 无 |

CLI 不提供 `--auto`：当前受支持的 `atomgit-sdk` 没有修复生成 API。Agent 应先抓取评论、分析并向用户展示 JSON 修复方案，再使用 `--apply-fixes`。

代码修复命令不带 `--push` 时，本地 autosquash 成功后写入 `./tmp/{repo}_pr_{number}_review_resolution_{transaction_id}.json`，返回 `pending_push` / 退出码 2，并且不会发送“已修复”回复。push 或回复部分失败时保留同一状态文件；使用：

```bash
python3 review_resolution.py --resume ./tmp/ib_robot_pr_123_review_resolution_<transaction-id>.json
```

恢复会验证状态文件中的 repository/PR、精确 push URL/ref、old/new OID 和当前远端状态；本地 HEAD 允许处于 old 或 new OID，但 new commit 对象必须仍可用。每条回复带隐藏事务 marker 并逐条原子 checkpoint；恢复先对账 PR 评论，只发送状态仍为 `pending` 且远端尚无相同 marker/正文哈希的条目。CLI 会输出单行 `RESUME_STATE={...}` JSON 摘要，便于机器读取。

## 回复单条 review 意见（reply-comment）

当用户明确要求「只回复某条评论 / 针对 comment_id 回复 / 这条意见单独回一下」时，不需要生成 fixes.json，直接使用单评论回复模式：

```bash
python3 review_resolution.py --pr 123 --reply-comment 456 --reply-body "已确认，这里按建议补充边界检查。" --ai-model gpt-5.5

# 回复内容较长时，从文件读取，避免 shell 转义导致 Markdown 损坏
python3 review_resolution.py --pr 123 --reply-comment 456 --reply-file ./tmp/reply_456.md --ai-model gpt-5.5

# 如需显式指定 discussion 下的嵌套回复，可写 threaded（默认就是 threaded）
python3 review_resolution.py --pr 123 --reply-comment 456 --reply-file ./tmp/reply_456.md --reply-mode threaded --ai-model gpt-5.5

# 只有在明确需要额外发一条页面可见评论时，才显式指定 visible
python3 review_resolution.py --pr 123 --reply-comment 456 --reply-file ./tmp/reply_456.md --reply-mode visible --ai-model gpt-5.5
```

### reply-mode 语义

脚本默认使用 `--reply-mode threaded`：

- 对已有 review discussion，直接在 **原 review 线程下追加详细回复**
- 这样可以避免「线程下已有简短回复，同时又额外生成一条顶层/行内评论」的冗余展示

只有在显式指定 `--reply-mode visible` 时，才会：

- 对 PR 总评 / 普通评论，发送 **页面可见的 PR 顶层评论**
- 对 DiffNote / 行内评论，发送 **页面可见的 inline comment**

`visible` 适合需要额外补一条页面可见评论的场景；常规 review 跟进默认应使用 threaded。

## 审查者复查闭环（开发者已修复 review 意见后）

本 skill 默认面向**被审查者**（修复别人提的 review 意见）。但审查者一方也有闭环需求：开发者按意见更新代码后，审查者需要复查最新提交、确认修复，并把可标记的意见标记为已解决。推荐流程：

1. 用 `atomgit-pr-review` 的 `pr_review.py --pr N` 重新提取最新 PR 上下文，逐条核对开发者是否真的修复了每条意见（看最新 diff 与测试是否同步更新）。
2. 用 `review_resolution.py --pr N --include-self-comments` 抓取**自己之前提交的**未解决评论——审查者要闭环的正是自己提的意见，而脚本默认会过滤当前用户的评论，**必须加 `--include-self-comments`**。
3. 对每条已确认修复的意见：
   - 先用 `--reply-comment <id> --reply-file <回复文件> --reply-mode threaded` 在原线程下回复确认（正文明确写出「已修复并复查确认通过，标记为已解决」）。
   - 若该条是 `diff_comment`（行内评论），再追加 `--resolve-comment <id> --resolved true` 标记已解决。
   - 若该条是 `pr_comment`（PR 级评论），**跳过 resolve**（会 400 失败，见下一节限制说明），回复确认即视为闭环。
4. 若存在顶层摘要评论，可单独回复一条「本次 review 的 N 条意见均已修复确认、全部闭环」作为整体收尾。

## 修改某条 review discussion 的解决状态

```bash
# 按评论 ID 自动查找 discussion_id 并标记已解决
python3 review_resolution.py --pr 123 --resolve-comment 456 --resolved true --ai-model gpt-5.5

# 已知 discussion_id 时直接操作
python3 review_resolution.py --pr 123 --resolve-discussion abcdef --resolved false --ai-model gpt-5.5
```

脚本使用官方文档中的 `PUT /api/v5/repos/:owner/:repo/pulls/:number/comments/:discussion_id` 接口修改解决状态。

### ⚠️ resolve 仅对 inline diff comment 有效

该接口**只能切换绑定到具体代码行的行内评论（`diff_comment` / DiffNote）的 resolved 状态**。对以下类型调用 `--resolve-comment` / `--resolve-discussion` 会返回 **HTTP 400**，无法标记：

- PR 级 review 评论（`comment_type == "pr_comment"`），例如 `atomgit-pr-architecture-review` 提交的架构审查评论，或 `atomgit-pr-review` 提交的顶层摘要评论；
- 普通 PR 顶层评论。

闭环时的判断规则（基于 `unresolved_comments.json` 的 `comment_type` 字段）：

- `diff_comment` → 回复确认后，用 `--resolve-comment --resolved true` 标记已解决。
- `pr_comment` → **只能**通过 `--reply-comment` 回复确认修复来实现实质闭环，**不要**对其调用 `--resolve-comment`。

只要回复正文明确写出「已修复并复查确认通过」，即使平台未切换 resolved 标记，对作者而言该意见也已闭环。
