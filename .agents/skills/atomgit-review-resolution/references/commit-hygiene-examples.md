# Commit Hygiene: Full Examples & Failure Behavior

## When to Read

- 需要查看 fixup+autosquash 工作流的完整可复制命令示例
- 想确认默认 / 指定目标 / 不同 base branch / dry-run 四种场景下的参数组合
- autosquash 执行失败，需要了解脚本会保留什么、打印什么、以及常见失败原因
- 需要核对 PR commit 数量约束（≤ 5）与本脚本的关系

## Commit 提交规范（遵循 ibrobot-git-flow §3 PR Commit Hygiene）

本脚本仅在 `--apply-fixes` 模式下自动提交代码修复。当前 SDK 没有受支持的 fix generation API，因此 CLI 不提供 `--auto`。提交方式遵循 ibrobot-git-flow skill 的 PR commit hygiene 规则。

**禁止的做法**（旧版本曾使用，现已移除）:

- ❌ `git commit --amend -m "fix: resolve review comments"`
- ❌ 创建任何形如 `fix: address review comments` / `fix reviewer's意见` / `apply review suggestions` 的噪声 commit
- ❌ `git add .`（会把无关文件一并暂存）
- ❌ 无 pathspec 的 `git add -u`（在 Git 2.0+ 会暂存整个 working tree，而非当前目录）

**当前实现的工作流**:

1. 先确认 PR 的 Model 披露只含模型名称及版本、不含 provider 前缀，并包含本次 `--ai-model`。不同 `fixup_target` 可记录不同模型；若当前修复会折入尚未记录本模型的目标 commit，应补充该模型的 `Co-Authored-By` 并同步 PR 模型列表。
2. 从 PR API 读取不可变 `base.sha`、开始处理时的 `head.sha`、source branch/repository；本地 `HEAD` 必须等于 `head.sha`，且 base 必须是 HEAD 的祖先。
3. 任何本地改写前验证所有 `comment_id` 属于目标 PR，并拒绝 `base.sha..HEAD` 中的 merge commit。
4. 按 fixes 中逐项 `fixup_target` 解析后的 SHA 分组；每组使用 `GIT_LITERAL_PATHSPECS=1 git add -- <paths>`、`git commit --fixup=<sha>`，不会把 pathspec magic 展开到其他路径。
5. `GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base-sha>`：所有临时 fixup 创建后只执行一次 autosquash。
6. 无 `--push` 时保存原子状态并返回 `pending_push`（退出码 2），不回复评论；用 `--resume <state.json>` 继续。
7. 有 `--push` 时执行 `git push --force-with-lease=refs/heads/<branch>:<old-head-sha> <exact-push-url> HEAD:refs/heads/<branch>`；`ls-remote` 使用同一 URL 验证远端 ref 后才回复。
8. 每条回复后原子更新状态；部分失败返回 `pending_replies`，恢复只发送 pending 条目。

## 完整示例

```bash
# 推荐: fixes JSON 每个代码修复项都携带 fixup_target；确认后推送并闭环回复
python3 review_resolution.py --pr 123 \
  --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json \
  --push \
  --ai-model glm-5.2

# 整批确实属于同一 commit 时，可以显式提供 fallback target
python3 review_resolution.py --pr 123 \
  --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json \
  --fixup-target abc1234 \
  --push \
  --ai-model glm-5.2

# 只完成本地重写，不授权 push：返回 pending_push / exit 2，且不回复
python3 review_resolution.py --pr 123 \
  --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json \
  --ai-model glm-5.2

# 使用上一步输出的状态文件恢复精确 lease push、远端验证和剩余回复
python3 review_resolution.py \
  --resume ./tmp/ib_robot_pr_123_review_resolution_<transaction-id>.json

# 只想看会怎么改, 不实际提交
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --dry-run --ai-model glm-5.2
```

`--base-branch` 仅为旧命令兼容保留，值会被忽略。不要再用它选择 rebase 基线；脚本只信任 PR metadata 的 `base.sha`。

## autosquash 失败时的行为

- 只在本次事务启动的 rebase 仍存在时执行 `git rebase --abort`，随后恢复原 HEAD/index 和文件快照（含 executable mode）。
- KeyboardInterrupt 与 SIGINT/SIGTERM/SIGHUP 走同一清理路径；已有 rebase 会在写入前直接拒绝，不会被脚本 abort。
- 常见失败原因: 本地 HEAD 与 PR head 不一致、PR base 不是 HEAD 祖先、PR 含 merge commit、目标 SHA 不在 `<base-sha>..HEAD`、tracked WIP、rebase 冲突。
- push 失败不会回退已经完成的本地重写，而是保存 old/new OID、精确 URL/ref 和回复账本后返回 `pending_push`；显式 lease 保证远端已前进时不会被覆盖。
- 回复失败不会重复已经 checkpoint 为 `sent` 的回复；恢复前还会按隐藏 marker/正文哈希对账远端，处理“服务端成功但客户端超时”的不确定结果。

## PR commit 数量约束

- 单个 PR 的 commit 数应 ≤ 5（见 ibrobot-git-flow §3）。本脚本的 fixup+autosquash 流程不会增加 commit 数，因此天然满足此约束。
- 如果 autosquash 后 commit 数仍 > 5，需要人工做一次整理（合并相关 commit），不要用 "fix review" 命名的新 commit 来"凑数"。
