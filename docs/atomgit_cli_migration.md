# AtomGit CLI 使用与迁移说明

## 日常操作直接使用 CLI

Agent 按对应 skill 直接调用官方 `gitcode` CLI 完成 PR、Issue 和评审操作；命令参数按需查
`gitcode <command> --help`。不再经由项目 Python 脚本转发普通平台读写。

```bash
GC_HOST=atomgit.com gitcode pr view 460 -R openEuler/IB_Robot --json
GC_HOST=atomgit.com gitcode pr comments 460 -R openEuler/IB_Robot --limit 0 --json
```

显式确认目标仓库，再传入 `-R owner/repo`；个人 fork 的 `origin` 是推送来源，
不能自动当作 PR 的目标仓库。仓库级 `config.json` 已删除。

认证由用户在私密终端完成，凭证由 CLI 管理，不在聊天、PR 或日志中传递：

```bash
gitcode auth login --hostname atomgit.com
GC_HOST=atomgit.com gitcode auth status
```

普通调用使用 PATH 上的 `gitcode`，不要使用 Graphviz 的 `gc`。
保留脚本还支持通过 `IBROBOT_ATOMGIT_CLI` 指定 CLI 的绝对路径；该变量是项目脚本的约定，
不会改变 Agent 直接执行 `gitcode` 时的 PATH 查找。

## 正式安装（Linux）

日常使用不依赖工作树的 `tmp/`，直接安装到 PATH 目录，不维护版本目录和符号链接：

```text
/usr/local/bin/gitcode   官方二进制
/usr/bin/gc              保留 Graphviz
```

从官方 Release 获取与你的系统架构匹配的独立二进制，核验来源和 checksum 后安装。
以下命令以 Linux x86_64、当前目录已有 `gc_linux_amd64` 为例；ARM 主机必须选择对应架构发行物。
若目标已经存在，先核对它的安装来源，不要覆盖其他安装渠道管理的文件。
升级沿用同一安装命令，并对照所下载版本的官方校验值核验文件。

```bash
sha256sum gc_linux_amd64
# 与所下载版本的官方校验值比较后安装
sudo install -m 0755 gc_linux_amd64 /usr/local/bin/gitcode
command -v gitcode
gitcode version
GC_HOST=atomgit.com gitcode auth status
```

这里的安装仅部署二进制和入口，不搬运、覆盖或复制认证配置；同一用户仍通过 CLI 管理认证。
确保 `/usr/local/bin` 在 PATH 上；不同 worktree 可以复用同一安装，不需要逐仓设置可执行文件路径。
项目不限定 CLI 发布版本；脚本只检查可执行文件身份，命令或参数不兼容时报告实际错误。
下文的版本号记录历史验证范围，不是安装或升级约束。
原先 `tmp/tools/atomgit-cli/` 仅为迁移验证阶段的临时安装位置，历史证据保留该路径。

## 项目脚本的边界

| 工作 | 执行方式 |
| --- | --- |
| PR/Issue 创建、读取、修改，评审读取和发布 | Agent 直接使用官方 CLI；具体流程见对应 skill |
| PR 描述生成与发布前检查 | `atomgit-pr/scripts/pr_body_gate.py` 读取本地 Git，生成通过项目检查的正文；Agent 再调用 CLI 发布 |
| PR 专项检查 | `atomgit-pr-review/scripts/review_gates.py` 检查通过 CLI 获取的数据；Docker 指纹改从本地确定性 diff 推导（base/head 均 fetch 到本地，与作者侧同源）；分析和发布仍由 Agent 执行 |
| 批量链接校验 | 保留专用脚本；脚本内自含最小只读 CLI 调用（argv/超时/`GC_HOST`/身份探测），无跨技能共享包装，可随技能独立携带 |
| intro 状态引导 | 纯本地脚本（git 状态、编译产物、分支）；不再远端抽查 PR，需要平台状态时 Agent 直接用 CLI |
| 评审意见修复与线程跟进 | Agent 编辑与验证，复用 Git 提交推送流程，直接通过 CLI 回复/resolve |

PR 作者侧和评审侧共用 `_shared/pr_gates.py`，统一 AI 披露、模型集合、复用自查和
WIP/Docker 验证规则。该模块不负责平台读写，唯一子进程用途是两侧共用的确定性本地
diff 助手（`collect_diff_file_infos`：refs 先解析为完整 OID，再从显式 merge-base 以
固定 flag 集 `--no-renames --no-ext-diff --no-textconv --no-indent-heuristic
--diff-algorithm=myers` diff，`-z` 解析路径、`--literal-pathspecs` 传路径）——作者与
评审由此对同一 base/head 得到完全相同的指纹输入，平台 files 的 rename 配对差异不再
影响指纹。普通评审不再通过项目内置 LLM 自动评审器，而由当前 Agent 按三步流程执行。

链接校验脚本内自含最小只读 CLI 调用；跨技能共享的 `_shared/atomgit_cli.py` 包装已删除。
intro 状态引导改为纯本地检查，不再远端抽查 PR 评论。根目录 `config.json` 和相关读取逻辑已删除。
CLI 自己的认证配置、模型配置和 DeepWiki `doc_config.json` 保留。
项目 requirements 不再安装 AtomGit SDK，也不再保留其移植代码或配套许可证目录。

### 已删除入口与替代方式

- `issue_management.py`、`pr_creation.py`、`pr_management.py`：改为 CLI 直接操作；
  PR 发布前检查收敛到 `pr_body_gate.py`。
- `pr_review.py`、`architecture_review.py` 及自动 LLM/评论格式化库：改为 Agent 读取、
  分析和发布；专项项目检查复用 `review_gates.py`。
- 旧包装入口和自动评审器的测试随对应实现删除。项目规则测试迁移到共享规则模块和新的
  本地检查入口；链接校验测试继续保留，intro 平台抽查测试与通用 CLI 包装测试随功能删除。
- `review_resolution.py`、`review_helpers.py` 及其测试、旧 JSON 协议和 SDK import blocker
  已删除。自动执行/恢复行为不再提供，改为 Agent 按 skill 直接执行。

直接评审必须记录目标 owner/repo/PR、源仓库及完整 head SHA，读取相关文件的完整上下文；
发布前再次核对源仓库和 head SHA。无法证明内容完整或版本已变时，停止发布并重新评审。
这些步骤由 skill 指导 Agent 执行，不再由旧上下文导出/发布包装器代办。

## 已验证的平台行为与限制

以下结果来自 CLI v0.14.1 的 help、已认证只读检查及个人 fork 的写入 smoke，
不是对未来 CLI 版本的保证。主仓库只读样本为 PR #456–460 和 Issue #141。

| 场景 | 行为与处理 |
| --- | --- |
| URL 形式 | v0.14.1 实测 `repo view` 接受 atomgit.com/gitcode.com 仓库 URL；`pr view` 只接受编号。链接校验识别两个主机和数字 pull/pulls/merge_requests 路径，资源查询交给 CLI，不自建 API 域名映射 |
| 完整 diff | files API 的 patch 位于 `patch.diff`；`pr diff --json` 是统计摘要，不可当完整 patch |
| 分页 | PR #456 的 files 端点重复返回 127 个文件；重复页本身不证明完整，需核对独立 PR 总数 |
| 跨 fork 源码 | 使用 head.repo 的源仓库与完整 head SHA，不回退到目标仓库 HEAD |
| 评论 | 可读 `discussion_id`、`user.login`、布尔 `resolved`；不要假定存在 `resolved_at` |
| 正文传递 | 优先 `--body-file`；仅支持 `--body` 时安全传入参数，不拼接正文成 shell 代码 |
| 写入不确定 | 请求失败或缺返回编号仍可能已写入；先只读核对，不能盲目再创建或回复 |
| threaded 回复 | smoke 中网页可见，但已测试 CLI/API 读回未返回嵌套回复；uncertain 结果须人工核对 |
| visible 行内回复 | 读回 `diff_position` 有 new-side 行号但缺文件路径时，拒绝猜测定位或自动降级为顶层评论 |
| files rename 配对 | PR460 实测：`pr view` changed_files=61，本地 `git diff --name-only` 默认 62、`--no-renames` 65；平台把「删旧 + 增新」合成 `status=renamed` 且 `previous_filename` 为空，配对阈值与本地不同。指纹统一走本地确定性 diff，平台清单仅自洽校验；不据 `previous_filename` 虚构删除文件 |
| 行内评论回读定位 | PR460 实测回读缺 `path`/`position`（发布时带）；缺定位=定位未知，不能断言 PR 级；带定位发表的两条经网页核验确为 inline 挂载（细节见当轮 tmp 证据）。发表后保存响应与稳定 `discussion_id` 并 UI 回读 |
| ID 形态 | PR460 观察到发布响应 id 与列表 id 类型不同（十六进制/数字）；`discussion_id` 稳定。关联线程只用 `discussion_id`，不互转不假设一致 |
| 计数对账 | 平台 changed_files/commits 计数仅平台自洽；与本地清单不一致时逐路径比对并以 base/head 树核对确认实际 rename 配对，核对后仍无法解释的差异按不完整处理 |

2026-09-29 在个人 fork `BreezeWu/IB_Robot` 的测试 Issue #1、PR #1 完成创建、读取、
更新、评论、关闭/重开、行内发表、resolve/unresolve 和 visible marker 读回。
两项资源已关闭，测试分支已删除。上述历史 smoke 覆盖 CLI/platform 能力，
不替代精简后项目工具的回归测试。

旧修复状态文件不再自动恢复或迁移。中断后重新核对远端 head 与评论，
无法确定回复是否已发表时人工核对，不能因列表没找到回复就重发。

## 验证范围

保留脚本使用可执行 fake CLI 验证项目自己的逻辑。
测试不重复证明官方 CLI 的基础命令实现。当前变更的实际测试命令与结果
记录在 PR 描述中。

仓库级测试此前存在 `test_canonical_skill_contract_passes` 既有失败，以及
`huggingface_hub`、`jsonschema.protocols`、`onnx` 缺失引起的收集限制。
局部回归通过不表示全仓测试通过。

PR 按用户选择保持 WIP；Ubuntu/openEuler Docker 验证暂缓，正式检视前补齐。

## 迁移证据与许可

历史 OpenSpec change `migrate-atomgit-skills-to-cli` 和平台证据在主仓库集中保存，
`openspec/` 被 Git 忽略。本文描述当前架构；历史的完整包装层设计不再作为日常调用方式。
不提交认证文件、原始敏感平台数据或 CLI 二进制。

原 SDK 派生实现及其配套许可证目录已经一并移除。官方 CLI 独立安装，
仓库不分发其源码或二进制。

链接校验器及测试位于 `.agents/skills/doc-link-validator/`；保留上游 Unicode URL
百分号编码，DeepWiki 翻译通过该独立 skill 做交付检查。
