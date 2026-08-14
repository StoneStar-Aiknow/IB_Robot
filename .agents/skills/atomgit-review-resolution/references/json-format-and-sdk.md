# JSON Format, API Behavior & SDK Design

## When to Read

- 需要编写 `fix_results.json` 修复方案文件
- 想了解四种修复类型（代码修复 / 回复说明 / 回退文件 / 删除行）的语义
- 需要查询「获取未解决评论」「应用修复方案」两个 API 的输出与默认行为
- 想理解 SDK / API 抽象层（`atomgit-sdk` / `atomgit_sdk`）的设计决策

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
- 脚本在任何文件写入或历史重写前，通过 `RepairService.get_pr_comments(pr_number)` 的 PR 作用域全量分页结果验证每个 `comment_id`；其他 PR、已删除或拼错的 ID 会整批拒绝。状态文件会保留 `discussion_id`、评论类型、路径等可用元数据。
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
- AtomGit / GitCode 当前读取接口**不会稳定返回 resolved 状态**，因此输出文件会在 `metadata.resolved_state_note` 中说明这一点

### 应用修复方案

```bash
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --push --ai-model glm-5.2
```

脚本从 `get_pull_request()` 的 `base.sha`、`head.sha`、`head.ref`、`head.repo` 绑定本次历史重写和推送。若 `base.sha..HEAD` 包含 merge commit，会在 autosquash 前拒绝，避免普通 rebase 线性化拓扑。

source repository 必须对应一个 remote 上的唯一精确 push URL：匹配 remote 含多个 pushurl，或多个 remote/URL 都匹配时拒绝。带 `--push` 时 push 和 `ls-remote` 均使用该精确 URL，并使用 `--force-with-lease=refs/heads/<branch>:<old-head-sha>`。远端 ref 验证为新 HEAD 后才回复评论。

不带 `--push` 时返回 `pending_push`（退出码 2），不回复评论，并保存可恢复状态。状态包含 PR/repository、old/new HEAD、精确 URL/ref、回复正文/哈希/marker、discussion 元数据、attempts/reply_id/status。回复部分失败返回 `pending_replies`（退出码 2）；`--resume <state.json>` 只重试未完成项。

## SDK / API 设计决策

保留 PyPI 包 `atomgit-sdk` 提供的 `atomgit_sdk` 导入模块作为唯一 AtomGit API 抽象层；skill 脚本只做工作流编排，不直接散落 HTTP 请求。原因：

1. 单条 review 回复、解决状态、评论编辑/删除等能力会被多个 skill 复用，放在 SDK 中更稳定。
2. 官方 API 文档目前主要提供 endpoint 标题、HTTP 方法和路径；SDK 已沉淀为 `APICatalog`，常用协作 API 提供 typed wrapper，长尾 API 可通过 `client.call_api(...)` 或 `APICatalog.from_docs()` 使用。
3. 这样比每个 skill 脚本各自拼 curl 更简洁，也能统一认证、错误处理、重试和 URL 解析。

当前 SDK 提供 `get_pr_comments`、`get_review_threads`、`get_pr_comment`、`reply_to_comment` 等 review API，但不提供 `generate_fix`。因此本 skill 不暴露自动 LLM 修复入口；LLM 分析由 Agent 完成，脚本只消费经确认的 JSON 计划。
