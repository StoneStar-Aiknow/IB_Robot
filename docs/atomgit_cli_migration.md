# AtomGit 协作工具迁移与验证

## 运行入口

项目的 AtomGit 平台 API 访问统一使用官方 CLI v0.14.1；`gitcode` 是项目入口名，
可链接至官方二进制，也可通过 `IBROBOT_ATOMGIT_CLI` 设置绝对路径。不能用系统 Graphviz 的 `gc`。
CLI 认证由用户在私密终端完成；不得在聊天、PR、日志中粘贴 token。

```bash
export IBROBOT_ATOMGIT_CLI=/absolute/path/to/gitcode
gitcode auth login --hostname atomgit.com
```

`config.json` 保留非敏感仓库默认值及原有 LLM 配置，但不再承担 AtomGit 认证。
显式目标优先于配置，再回退至唯一 upstream；origin 只供源仓库识别。

## 保留的项目行为

- 六个协作 skill 的路由与职责，AI 披露、DCO 提交流程、人工审查确认、WIP、复用检查。
- 评审上下文绑定目标 owner/repo/PR、源仓库、完整 head SHA 与可读文件内容。
- 修复中的本地事务回滚、精确源 URL/OID lease、推送验证成功后回复、逐条持久化账本。
- 新恢复状态使用 schema v2 和来源标记；SDK 时代及中间 CLI 版本的 v1 状态不自动恢复。
- 写入结果不确定时不自动重发；marker/正文哈希对账不能证明结果时需人工检查。

## 实测版本与边界（2026-09-29）

官方 CLI v0.14.1，构建 commit `21e19618984ec0628192f6283bca066c76e12511`。
只读平台样本包含主仓库 PR #456–459 和 Issue #141。

| 场景 | 已确认的行为 |
| --- | --- |
| PR/Issue number | PR 为整数，Issue 为数字字符串；入口显式检查资源身份 |
| 文件 patch | files API 返回 `patch.diff`；`pr diff --json` 是摘要，不替代完整 patch |
| 大 PR 文件读取 | #456 返回 127 文件；该端点重复返回集合。重复页本身不证明完整性，需与独立 PR 总数核对 |
| 跨 fork 内容 | 用 head.repo 的源仓库和完整 head SHA 读取，不回退到目标仓库 HEAD |
| 评论 | 可读 `discussion_id`、`user.login`、布尔 `resolved`；不能假设 `resolved_at` 存在 |
| 错误 | 不存在的 PR/Issue 实测退出 3；未认证实测退出 4；非零退出携 JSON 仍属失败 |
| 正文 | 优先文件/stdin；仅支持 --body 的命令使用参数数组，绝不经过 shell 拼接 |

## 验证层次

1. 本地行为测试由可执行 fake CLI 驱动，并用真实 import blocker 禁止 SDK 导入，
   覆盖入口子进程；这不等同于真实平台验收。
2. 已认证只读验证核对响应形状、分页、内容与 verification 指纹。
3. 远端写入验证只在个人 fork 的专用测试 PR/Issue 上执行，资源编号与结果在 PR 描述中列出。
4. 独立 Agent 根据 skill/help/reference 完成命令发现演练，不阅读实现源码。
5. Docker 验证按用户要求暂缓；依赖/setup 变化仍需在正式检视前补齐双平台证据。

完整仓库入口有已知环境/基线限制，不能将局部测试通过称为全仓验收通过：
`test_canonical_skill_contract_passes` 为已有失败；三个测试文件此前因
huggingface_hub、jsonschema.protocols、onnx 依赖缺口被排除，最终结果在 PR 中如实记录。

## 证据与规格

OpenSpec 使用主仓库集中保存方式，当前 worktree 的 `openspec/` 被忽略。
迁移 change 为 `migrate-atomgit-skills-to-cli`，包含三轮设计评审、版本绑定的
能力矩阵、实际 help 和独立演练记录。提交的本文提供可公开审阅的摘要；
不把认证文件、原始敏感平台数据、临时 CLI 二进制或本机路径当作公开交付物。

SDK 适配代码的来源和许可见 `third_party/licenses/atomgit-sdk/README.md`。

## 真实写入 smoke 结果

2026-09-29 在个人 fork `BreezeWu/IB_Robot` 本次新建的 Issue #1、PR #1 上验证，
测试完成后两项资源均关闭，临时分支 `test/atomgit-cli-20260929-rs1` 已删除：

- Issue 创建/读取/评论/关闭/重开/再次关闭；修复了入口缺少 `reopen` 枚举的问题。
- PR 创建、更新、回读、提取上下文、摘要及第 8 行行内评论发表；支持同仓测试 PR，跨 fork 发布仍独立验证。
- 行内 discussion 的 resolve/unresolve 写入与布尔 resolved 读回闭环。
- PR 级 visible 回复的事务 marker 原文读回成功。

真实平台限制（不是已通过能力）：

- threaded 回复写入成功且网页可见，但测试过的 CLI/API 读取通道未返回嵌套回复。
  该模式的 uncertain 写入不能自动对账，必须人工核对；绝不因列表没找到 marker 而重发。
- `diff_comment` 的 `diff_position` 是含 new-side 行号的对象，但本次读回没有文件路径。
  对这类评论进行 visible 行内回复时明确 unsupported，不能猜文件或退成顶层评论。

提交前主 Agent 独立结果：SDK 阻断下协作测试 **402 passed**；三个 setup shell harness
分别 **18/16/3 passed**；OpenSpec strict 与 `git diff --check` 通过。
不排除文件的全仓测试在收集时因 huggingface_hub 缺失退出 3；排除前述三个环境缺口文件后
**408 passed、1 failed、退出 1**。既有失败为
`scripts/test_check_agent_skill.py::test_canonical_skill_contract_passes`，不称全仓通过。

## 测试精简后的边界（2026-09-29）

官方 CLI 的命令实现由上游维护；本仓库保留项目适配和工作流回归，不以 fake CLI
证明官方实现本身正确。迁移初版中重复的基础成功响应、登录提示、命令参数快照和
格式化辅助函数测试已移除；对应实际命令已有前述平台 smoke 证据。

| 精简项 | 保留的覆盖 |
| --- | --- |
| 两份自动评审测试合并 | `test_auto_review_gates.py` 参数化覆盖两种评审，保留缺 head、部分/全部读取失败、dry-run、完整结果发布 |
| 多组 PR 超时测试共用准备逻辑 | 唯一匹配、错误正文、同 owner 不同 repo、身份缺失/歧义均保留，且只允许一次创建尝试 |
| 重复 fake CLI 类移除 | 边界和入口共用 `entrypoint_fixture.py`；主机绑定、正文参数数组保真、Graphviz 冲突仍有测试 |
| 基础 JSON/成功/登录提示重复用例移除 | 非零退出携 JSON 不可当成功、读取不完整不可发布、仓库/来源绑定和无写入门禁继续覆盖 |
| SDK blocker 自检缩减 | 保留父进程与子进程的真实 SDK 导入阻断；全部入口回归仍在阻断环境下执行 |

文件事务、精确 Git lease、schema v2 准入、未知回复不重发、重复页完整性证明、
真实平台 `diff_position` 适配的测试没有删除。测试用例数降低不表示这些保护被移除。


## 与上游文档工具重组对齐

基于 `23913118` 的目录重组，链接校验器及对应测试位于
`.agents/skills/doc-link-validator/`。DeepWiki 翻译流程仍通过该独立 skill 进行后置校验；
AtomGit 访问继续走共享 CLI，第三方 HTTP 链接保留上游新增的非 ASCII URL 百分号编码行为。
旧 `deepwiki-translator/scripts/link_validator.py` 路径不再提供入口。
