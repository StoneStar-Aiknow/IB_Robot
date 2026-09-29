---
name: atomgit-pr-architecture-review
description: "AtomGit PR 架构评审工具。当用户需要“架构审查PR”、“review architecture compliance”、“检查架构规范”、“审阅PR架构”、“SSOT 合规性检查”、“契约驱动检查”或对指定 PR 做架构维度扫描并提交检视意见时调用。只要目标是本仓库的 PR 架构评审，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
license: MIT
---

# AtomGit PR 架构评审

用于 IB_Robot 的 SSOT、契约、包职责及架构一致性审查。普通 Bug/逻辑审查使用
`atomgit-pr-review`。平台操作直接调用官方 `gitcode`，不再使用项目自动 LLM 评审脚本。

## 执行方式

复用 [通用评审流程](../atomgit-pr-review/SKILL.md)：先展示问题与目标、独立方案，
再读取实现并做比较；同样执行项目专项检查、完整性核对和发布前源仓库/head SHA 核对。
认证、CLI 读取、分页与评论命令不在此重复，参数按 `--help` 查询。

分析各包前，读取 `ibrobot-architecture` 及目标分支基线的包 README，确定既有职责。
进入实现比较后，再读取 PR head 的代码与 README，避免把 PR 的架构主张当既有约定。
重点检查：

| 维度 | 要求 |
| --- | --- |
| SSOT | 配置来自 `robot_config` 等权威来源，不增加平行配置和硬编码 |
| 契约 | 接口、状态和数据流明确，依赖注入与验证符合既有契约 |
| 控制模式 | 控制模式分离、状态管理一致，安全/授权边界不被绕过 |
| 包职责 | 按硬件/驱动/业务/模型层检查依赖；例如轻量 teleop 不吸纳重型运动规划 |
| README | 公开接口、启动参数、配置、职责、数据流和限制与实现一致；必要变更未同步视为问题 |
| 复用 | 先查仓库与 LeRobot 已有能力；平行 setup/推理/数据集流程必须有具体必要性论证 |

变更量 >2000 行时审计 `## Reuse Self-Check` 四项声明，不仅看字段齐全：复用模块是否真的
调用、重复实现是否必要、架构是否确实一致。虚假/空泛声明作为阻塞发现，优先建议接入
已有 `inference_service`、`robot_config`、`tensormsg` 或 LeRobot 能力。
LeRobot 源码修改走 patch 栈，不以普通根仓库 gitlink 提交替代。

默认静态审查，不自行跑 lint、类型检查、build 或 Docker；用户明确要求验证时才按相应
skill 执行。不要仅因存在 pre-commit 就宣称代码必然通过，应核对作者证据与实际 diff。

## 结果与发布

按具体文件/职责给出发现，说明所属维度、影响及最小修复路径。使用
[架构发现格式](references/review-findings.md)，直接以 Markdown 提交发现，无须生成 JSON。
先向用户展示结果并获得发布确认，再按通用评审 skill 直接用 CLI 发表评论。

发布前再次确认目标仓库/PR、源仓库与完整 head SHA 未变化，相关上下文完整。
可定位的问题使用当前 diff new-side 行内评论；跨文件架构问题明确用 PR 级评论。
注意 `pr comments` 回读可能缺 `path`/`position`：缺定位只是「定位未知」，不能据此
断言某条行内评论实为 PR 级；需定位核验时在网页确认。
读失败、版本漂移或行内位置不可证明时停止，不能发“架构通过”摘要。
