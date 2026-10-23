---
name: atomgit-pr
description: "AtomGit PR 工作流工具。当用户需要在本仓库或 AtomGit 上“创建PR”、“更新PR描述”、“同步PR标题/正文”、“生成PR摘要”、“create pull request”、“open merge request”、“update PR description”、“generate PR summary”或围绕 PR 管理动作工作时调用。它负责 PR 资源的创建和维护，不负责通用代码 review；只要目标是本仓库的 PR / merge request 管理，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit PR

日常读写由 Agent 直接调用官方 `gitcode`；参数按需查询 `gitcode pr <command> --help`。
项目只保留 [pr_body_gate.py](scripts/pr_body_gate.py) 做发布前的本地正文检查，
它不访问平台、不创建 PR。代码评审使用 `atomgit-pr-review`。

## 确认目标与读取

从用户链接/明确指定的仓库确认目标；缺省时参考唯一 `upstream`，
再显式传入 `-R owner/repo`。`origin` 是个人 fork，只用于 push 和 source；
仓库级 `config.json` 已删除。使用 `GC_HOST=atomgit.com`、PATH 上的 `gitcode`（不是 `gc`）。
未认证时由用户在私密终端 `gitcode auth login --hostname atomgit.com`，不搬运旧 token。

```bash
GC_HOST=atomgit.com gitcode pr view 123 -R openEuler/IB_Robot --json
GC_HOST=atomgit.com gitcode pr comments 123 -R openEuler/IB_Robot --limit 0 --json
GC_HOST=atomgit.com gitcode api 'repos/openEuler/IB_Robot/pulls/123/files?page=1&per_page=100'
GC_HOST=atomgit.com gitcode api 'repos/openEuler/IB_Robot/pulls/123/commits?page=1&per_page=100'
```

files 的完整 patch 在 `patch.diff`；`pr diff --json` 是统计摘要。按页获取并核对
`pr view` 的 `changed_files`/`commits` 总数；重复页不能当完整性证据。
源文件内容从 PR 的 head 仓库和完整 SHA 获取，不回退到目标仓库 HEAD。

## 生成并检查描述

1. 按 `ibrobot-git-flow` 提交/push：DCO sign-off、AI commit trailer、至多 5 个提交。
   获取最新目标/源分支，确认本地检查的 source/base 与实际 PR 完全一致，不能用其他
   分支的检查结果。查看 `git remote -v`，不要将 fork 的 master 当作上游基线。
2. 依据整个 PR 的改动写 Markdown，说明问题、结果、关键取舍和真实验证。
   按变更同步用户文档，不能把 `git diff` 当测试结果，也不能虚报测试通过。
3. 当前 Agent 执行实际工具的 `--version` 确认版本；`--agent-tool` 必须写**产品全称 + 版本**，
   产品名取工具官方名称，不是 CLI 命令名（如 `DeepSeek Harness 0.1.7-rc.2`，不是 `dsh 0.1.7-rc.2`；
   `--version` 只输出版本号时尤其注意）。只写 CLI 名或缩写会被社区门禁判为
   `ai-compliance-failed`（先例：#441 / #458）。收集 **PR 全部提交** 的 AI 模型集合，
   去掉 provider 前缀。记录 Prompt 摘要及第三方材料/许可证。披露块由工具生成，勿手写重复块。
4. 用户已明确要求提交/更新时沿用授权，不重复确认。人工实质审查责任仍由贡献者承担。披露默认标记待审查；只有开发者明确确认已完成
   逐项审查，才传 `--human-review-status completed`，不能以发布授权代替完成确认。PR 以正式检视发布（首次以非 `[WIP]`、非 draft
   创建，或 WIP 转正式）时，Agent 必须先通过 ask-user 工具询问开发者是否已完成逐项人工审查，确认才传 completed，否则保持
   pending；WIP/draft 阶段不询问。命中双平台 Docker 门禁的正式发布把该问题合并进已有的阶段询问，一次交互完成。更新已有 PR
   时先抓取当前正文传 `--previous-body-file`：上次披露为 completed 且没有新增 AI-assisted 提交则继续传 completed，有新 AI
   提交或状态不明则重新询问。completed 披露不允许静默降级：草稿中粘贴的旧披露块或 `--previous-body-file` 里的 completed
   遇到更低的 `--human-review-status` 会被工具拒绝；确需降级必须先重新询问开发者并显式传 `--review-status-reset`。
5. 命中双平台 Docker 条件时，尚未选择阶段才询问 WIP 或正式检视；已有选择继续生效。
   WIP 暂缓 Docker，保持 `[WIP]` 标题和 draft；正式检视补齐当前代码的两平台证据。

在仓库根目录、同次加载 `.shrc_local` 后运行。下列尖括号值必须替换为实际信息：

```bash
source .shrc_local && python3 .agents/skills/atomgit-pr/scripts/pr_body_gate.py \
  --branch <source-branch> --base upstream/master \
  --title '具体标题' --description-file tmp/pr-draft.md --output tmp/pr-body.md \
  --pr-stage wip --agent-tool '<产品全称及版本>' --ai-model '<完整模型集合>' \
  --prompt-summary '<本次意图>' --third-party-materials '<来源和许可证，或无>' --publication-authorized
```

工具从本地 Git 读取 commit/diff，而非未提交工作区。正式阶段检查本地源 head 与
`origin/<branch>` tracking ref 一致，并从真实 Git 对象推导 tree；运行前先 fetch。
更新已有 PR 时先抓取当前正文传 `--previous-body-file tmp/current-pr-body.md`，按相同输入规则复用
既有证据，不手改指纹；该文件同时驱动人工审查状态防降级校验（见第 4 条）。缺失/截断 patch 不支持证据复用。

项目描述规则：

- 变更量 additions+deletions >2000 时，必须有唯一 `## Reuse Self-Check`，包含具体答案：
  `**Reinvented workflows:**`、`**Reused components:**`、`**Reinvention justification:**`、
  `**Architecture conformance:**`。WIP 不豁免；说明与已有模块/LeRobot 的复用关系及偏离理由。
- 正式 Docker 证据位于唯一 `## Docker Verification`，四个字段为
  `**Docker verification mode:**`、`**Verified inputs:**`、`**Tested source tree:**`、
  `**Docker environment:**`，并保留 Ubuntu/openEuler 两平台真实结果。
  输入指纹不是 commit/tree SHA；缺证据时工具报错提供期望值，不能只补字段伪造通过。
- AI 模型披露必须覆盖所有 AI-assisted commit；人类 `Name <email>` coauthor 不算模型。
  每次 push 后重新检查并同步现有 PR 正文。

## 直接发布与回读

发布前再次确认远端 source head 与检查使用的本地 head 一致，目标仓库/base 正确。
使用工具输出的标题和正文文件；本地 `--base upstream/master` 对应远端分支 `master`。
Docker 指纹输入来自本地确定性 diff（`pr_gates.collect_diff_file_infos`：固定 flag 集、
禁用 rename 配对，与评审侧同源）；目标 base 必须与平台 PR 的 base 一致——作者默认
`upstream/master`，评审按 `pr.base.sha`，两侧 base 漂移（如 rebase 后未同步）会使指纹
失去可比性，运行前核对 `pr view` 的 base SHA。
以下以跨 fork WIP 为例，替换源仓库、分支、标题和 PR 编号：

```bash
GC_HOST=atomgit.com gitcode pr create -R openEuler/IB_Robot --fork BreezeWu/IB_Robot \
  --head <source-branch> --base master --title '[WIP] 具体标题' \
  --body-file tmp/pr-body.md --draft --json --no-interactive
# 更新现有 PR；同仓创建省略 --fork
GC_HOST=atomgit.com gitcode pr edit 123 -R openEuler/IB_Robot \
  --title '[WIP] 具体标题' --body-file tmp/pr-body.md --draft true --json --no-interactive
GC_HOST=atomgit.com gitcode pr view 123 -R openEuler/IB_Robot --json
```

回读核对目标、source repo/branch/SHA、base、标题、正文与 draft。
gitcode 写命令（create/edit/close/comment）的退出码双向不可靠：超时/非零退出可能已写入
（CLI 会在客户端超时后内部重试），退出 0 也不免除回读。实测创建可能已成功却返回非零退出，
且 WIP 标题/draft 可能需要显式 edit 后才能正确读回。
正式检视仅在证据补齐后用规范化标题和 `--draft false`。

写入结果不确定（超时、非零退出、缺编号）时，先等约 20 s 让服务端写入与索引落定，
再只读对账，绝不自动重发任何写命令。一次 create 可能因 CLI 内部重试产生多条完全相同的
PR（实测先例 !519–!521 同秒创建三条），必须拉全量列表核对，不能只看最新一条：

```bash
GC_HOST=atomgit.com gitcode pr list -R openEuler/IB_Robot --state all --paginate --json
# 候选过多可加 --head <source-branch> --base master 过滤；跨 fork 时过滤结果不作完整证据
```

`--paginate`、`--head`、`--base` 为 `pr list --help` 实测支持；`issue list` 没有
`--paginate`（见 atomgit-issue）。按完整身份核对候选：源仓库、head branch + 完整 SHA、
base、标题、正文哈希。

- 唯一匹配：回读确认后继续。
- 无匹配或身份对不上：人工核对，不自动重建。
- 多条完全一致：CLI 重试复制的重复 PR。停止发布流程，把清单交给用户确认；确认后保留
  最小编号，其余逐个带指向评论关闭；head/base/正文任一不同的 PR 一律不关：

```bash
GC_HOST=atomgit.com gitcode pr close 520 -R openEuler/IB_Robot \
  --comment 'CLI 超时重试产生的重复 PR，请改看 <保留编号>。' --yes --json --no-interactive
```

close/edit/comment 同样以回读为准（close 回读 state、comment 回读评论、edit 回读标题/正文），
退出码只作参考——实测 close 退出码 5 却已关闭并发出评论。回读失败如实报告
“写入结果未知”或“已写入但核验失败”，不能自动重发。
