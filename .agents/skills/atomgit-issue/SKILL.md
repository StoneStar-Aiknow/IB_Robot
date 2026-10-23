---
name: atomgit-issue
description: "AtomGit Issue 工作流工具。当用户需要在本仓库或 AtomGit 上“创建Issue”、“查看Issue详情”、“更新Issue”、“关闭/重开Issue”、“create issue”、“fetch issue info”、“update issue”、“report bug”、“feature request”或围绕 Issue 做任何创建/读取/更新动作时调用。只要目标是本仓库的 Issue，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit Issue

Agent 直接使用官方 `gitcode`，不调用项目 CRUD 包装脚本。参数按需查看
`gitcode issue <command> --help`。本仓库的 Issue 默认指 AtomGit，除非用户明确指定 GitHub。

## 目标与认证

- 从用户链接/明确指定的仓库确认 owner/repo/Issue 编号；信息冲突时先澄清。
  未指定仓库时可参考唯一 `upstream`，再显式传入 `-R`。
  `origin` 是个人 fork，不能自动当作目标；仓库级 `config.json` 已删除。
- 使用 PATH 上的 `gitcode`（不是 Graphviz `gc`），每次指定 `GC_HOST=atomgit.com`。
- 未认证时指引用户在私密终端运行 `gitcode auth login --hostname atomgit.com`；
  不读取旧 token，不把凭证写进命令正文或日志。

## 常用操作

以下以 `openEuler/IB_Robot` 的 Issue 123 为例，执行前替换为已确认的目标。
正文写入本地文件，保持换行和特殊字符原样。

```bash
GC_HOST=atomgit.com gitcode issue view 123 -R openEuler/IB_Robot --json
GC_HOST=atomgit.com gitcode issue comments 123 -R openEuler/IB_Robot --limit 0 --order asc --json
GC_HOST=atomgit.com gitcode issue create -R openEuler/IB_Robot --title '具体问题' --body-file tmp/issue.md --json --no-interactive
GC_HOST=atomgit.com gitcode issue comment 123 -R openEuler/IB_Robot --body-file tmp/comment.md --json --no-interactive
GC_HOST=atomgit.com gitcode issue edit 123 -R openEuler/IB_Robot --body-file tmp/issue.md --json --no-interactive
GC_HOST=atomgit.com gitcode issue edit 123 -R openEuler/IB_Robot --state closed --yes --json --no-interactive
# 重开：将上行 --state closed 改为 --state reopen
```

写入后以回读为准核对目标、正文和状态；gitcode 写命令（create/comment/edit）退出码双向
不可靠：超时、非零退出或未返回编号都可能已经写入，退出 0 也不免除回读。
结果不确定时先等约 20 s 再只读对账，不能盲目重复创建或评论：

```bash
GC_HOST=atomgit.com gitcode issue list -R openEuler/IB_Robot --state all --limit 50 --json
```

`issue list` 实测无 `--paginate`（unknown flag），翻页用 `--page`/`--per-page`，
或按需加大 `--limit`（只有 `pr list` 才有 `--paginate`）。按仓库 + 标题 + 正文核对候选：
唯一匹配即已创建；无匹配或身份对不上先人工核对，不自动重建。一次 create 出现多条
仓库/标题/正文完全相同的 Issue 属 CLI 超时重试复制的重复：停止，列清单请用户确认，
保留最小编号，其余先评论指向保留者再 `issue edit --state closed` 关闭；
标题/正文不同的 Issue 一律不关。重发评论前必须先回读评论列表确认未发出。
读取失败不能当作“没有 Issue/评论”，回读不了就如实报告“写入结果未知”；
完整对账流程见 [atomgit-pr](../atomgit-pr/SKILL.md)。

## IB_Robot 分类规则

分类只用下列词表，不通过列出仓库全部标签来猜分类，也不自造裸名 `bug`、`feature`：

- `kind/`：`api-change`、`bug`、`cleanup`、`design`、`documentation`、`failing-test`、`feature`、`enhancement`。
- `priority/`：`high`、`medium`、`low`。
- 其他社区标签：`help-wanted`、`invalid`、`question`、`newcomer`；仓库未预置时由管理员处理。

IB_Robot 使用社区机器人 slash 评论进行分类和指派；CLI 有标签/assignee 参数不代表
当前账号具备权限，也不改变这里的社区流程。先创建，再按用户意图发表评论：

```bash
GC_HOST=atomgit.com gitcode issue comment 123 -R openEuler/IB_Robot --body '/kind bug' --no-interactive
GC_HOST=atomgit.com gitcode issue comment 123 -R openEuler/IB_Robot --body '/priority high' --no-interactive
# 指派：/assign @username；自己：/assign；取消：/unassign
# 移除分类：/remove-kind bug、/remove-priority high
```

跨仓库操作遵循目标仓库自己的约定，不把 IB_Robot 的机器人命令强加给其他仓库。
