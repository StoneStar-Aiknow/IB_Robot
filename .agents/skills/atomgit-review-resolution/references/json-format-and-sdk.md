# JSON Format, API Behavior & CLI Design

## When to Read

- 需要编写 `fix_results.json` 修复方案文件
- 想了解四种修复类型（代码修复 / 回复说明 / 回退文件 / 删除行）的语义
- 需要查询「获取未解决评论」「应用修复方案」两个 API 的输出与默认行为
- 想理解平台访问层（官方 AtomGit CLI + `_shared/atomgit_cli.py` 共享边界）的设计决策

## 修复类型

1. **代码修复**: 提供具体的代码修改建议
2. **回复说明**: 仅需要回复解释
3. **回退文件**: 建议回退整个文件
4. **删除行**: 建议删除特定行

## 输入格式（fix_results.json）

文件名格式：`./tmp/{repo}_pr_{number}_fix_results.json`

```json
[
  {
    "type": "code_fix",
    "comment_id": 12345,
    "file_path": "src/main.py",
    "fixup_target": "abc1234",
    "line_number": 10,
    "fix_description": "修复说明",
    "original_code": "old code",
    "fixed_code": "new code",
    "reason": "修复原因"
  }
]
```

字段约束：

- 所有项必须有 `comment_id`。
- 脚本在任何文件写入或历史重写前，通过 `pr comments --json --limit 0`（0 = 全部分页）的 PR 作用域结果验证每个 `comment_id`；其他 PR、已删除或拼错的 ID 会整批拒绝。状态文件会保留 `discussion_id`、评论类型、路径等可用元数据。
- `code_fix`、`delete_lines`、`revert_file` 必须有 `file_path` 和 `fixup_target`。`fixup_target` 必须解析到 PR metadata `base.sha..HEAD` 范围内的 commit。
- `code_fix.original_code` 必须非空，并且在目标文件中恰好出现一次；0 次或多次匹配都会拒绝。
- 只有确认整个批次都应折回同一 commit 时，才可省略逐项字段并在 CLI 显式传 `--fixup-target <SHA|ref>`；该参数没有隐式 `HEAD` 默认值。
- 同一文件不能在一个批次中分配给多个 target；请拆成独立确认的批次。
- 同一文件在同一 target 下也只能有一个修复。唯一例外是多个 `delete_lines`：脚本会合并、去重行号并对同一初始快照一次倒序删除，同时仍为每个原始 `comment_id` 生成独立回复。
- `reply_only` 只要求非空 `reply`，不检查 index/worktree，也不触发 commit、rebase 或 push。

## API 说明

### 获取未解决评论

```bash
python3 review_resolution.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_unresolved_comments.json`

补充说明：

- 脚本会自动抓取 **所有分页评论**，不再受默认 20 条限制
- 输出按 `discussion_id` 聚合线程，`comments[].thread_comments` 中可看到嵌套回复
- 默认会过滤当前登录用户自己的评论和 bot 评论；可通过 `--include-self-comments` / `--include-bot-comments` 打开
- 当前登录用户读取失败时**不会静默声称已完成 self 过滤**：`metadata.self_filter` 会标记 `unavailable` 并附原因，stdout 同步告警
- AtomGit / GitCode 当前读取接口**不会稳定返回 resolved 状态**，因此输出文件会在 `metadata.resolved_state_note` 中说明这一点

### 应用修复方案

```bash
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --push --ai-model glm-5.2
```

脚本从 `get_pull_request()` 的 `base.sha`、`head.sha`、`head.ref`、`head.repo` 绑定本次历史重写和推送。若 `base.sha..HEAD` 包含 merge commit，会在 autosquash 前拒绝，避免普通 rebase 线性化拓扑。

source repository 必须对应一个 remote 上的唯一精确 push URL：匹配 remote 含多个 pushurl，或多个 remote/URL 都匹配时拒绝。带 `--push` 时 push 和 `ls-remote` 均使用该精确 URL，并使用 `--force-with-lease=refs/heads/<branch>:<old-head-sha>`。远端 ref 验证为新 HEAD 后才回复评论。

不带 `--push` 时返回 `pending_push`（退出码 2），不回复评论，并保存可恢复状态。状态包含 PR/repository、old/new HEAD、精确 URL/ref、回复正文/哈希/marker、discussion 元数据、attempts/reply_id/status。回复部分失败返回 `pending_replies`（退出码 2）；`--resume <state.json>` 只重试未完成项。

## 平台访问层设计决策（官方 CLI + 共享边界）

所有 AtomGit 平台读写都通过官方 AtomGit CLI（已验证 v0.14.1，`gitcode` 入口）完成；`_shared/atomgit_cli.py` 提供薄共享边界（工具身份/版本门禁、目标仓库解析、参数数组执行、JSON/分页/错误分类），本脚本内的 `RepairCli` 网关承载 review 语义（评论归属证明、线程回复、resolve、marker 对账）。原因：

1. 认证、平台接口与官方工具统一：开发者与 Agent 使用同一 CLI 登录（`gitcode auth login --hostname atomgit.com --with-token`），项目不再安装或导入 `atomgit-sdk`。
2. 薄边界不重建 SDK：单条 review 回复、解决状态等能力由 CLI 命令直接承载（`pr reply` / `pr comment [--path --position]` / `pr comment resolve|unresolve`），长尾 API 经登记的 `gitcode api` 通用入口分页读取。
3. 项目脚本专注工作流与门禁：评论归属校验、行内定位 new 侧语义、状态账本（marker/正文哈希、uncertain outcome）与 Git 事务留在本地，写入前零副作用可测。

CLI 提供 review 评论读写与 resolve，但不提供 `generate_fix`。因此本 skill 不暴露自动 LLM 修复入口；LLM 分析由 Agent 完成，脚本只消费经确认的 JSON 计划。
