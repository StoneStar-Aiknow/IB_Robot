---
name: "deepwiki-config"
description: "将 DeepWiki 仓库的 Wiki 页面结构转换为 doc_config.json，供 deepwiki-translator 翻译流水线使用；支持新仓库生成与已有配置更新。Use when users mention 'generate doc_config', 'DeepWiki config', 'DeepWiki 配置', '生成 doc_config', '更新 doc_config', or '重新生成配置'."
---

# DeepWiki 配置生成器

调用 `mcp_deepwiki_read_wiki_structure` 获取仓库 Wiki 结构，交给技能脚本 `generate_config.py` 生成 `doc_config.json` 配置文件。该配置的消费者是 **deepwiki-translator 技能**的 `deepwiki_processor.py` 脚本，hierarchy 与 label 格式以其输入约定为权威。

## 何时调用

- 用户需要为一个新的 DeepWiki 仓库生成 `doc_config.json`
- 用户提供了 `owner/repo` 格式的 GitHub 仓库名，并要求生成配置
- 用户要求更新或重新生成现有 `doc_config.json`

## 输入参数

用户必须提供（或从 DeepWiki URL 中提取）：

| 参数 | 说明 | 示例 |
|---|---|---|
| `repo` | `owner/repo` 格式的 GitHub 仓库 | `wuxiaoqiang12/IB_Robot` |
| `output_path` | 生成的 doc_config.json 保存路径（可选，默认为当前工作目录下的 `doc_config.json`） | `migration/doc_config.json` |

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| `generate_config.py` 的规则设计说明（解析、label、hierarchy、三向验证的意图、边界处理与示例），维护/调试/评审脚本时，或第 2 步退出码为 `1` 且用户要求定位修复时阅读 | `references/script-design.md` |

Do not expose these references as separate skills.

## 生成产物结构

`doc_config.json` 由三部分组成：

| 字段 | 用途 |
|---|---|
| `id_to_label` | 章节 ID（如 `"2.1"`）→ label（如 `"environment_setup"`） |
| `title_to_label` | 原始标题 → label |
| `hierarchy` | 输出文件/目录树（叶子节点为 `<label>.md`，目录节点含 `title` + `subs`） |

hierarchy 混合模式：无子页面的一级章节为叶子节点 `<label>.md`；有子页面的一级章节为目录节点 `<label>`，其 `subs` 为子页面 `<sub_label>.md` → 子页面标题。目录节点自身的 label 在 `deepwiki_processor.py` 输出时映射为 `<label>/overview.md`（配置中不含该 key）。

## 工作流程

### 第 1 步 — 获取 Wiki 结构

调用 `mcp_deepwiki_read_wiki_structure`，参数 `repoName = <repo>`。将返回的页面列表保存到临时文件（如 `tmp/wiki_structure.txt`），格式形如：

```
- 1 IB-Robot Overview
- 2 Getting Started
  - 2.1 Environment Setup
  - 2.2 Building the Project
- 3 Core Concepts
  - 3.1 Single Source of Truth Pattern
...
```

### 第 2 步 — 运行技能脚本

```bash
python -X utf8 <skill_dir>/scripts/generate_config.py tmp/wiki_structure.txt --output <output_path> [--repo <repo>]
```

`<skill_dir>` 指本技能目录（本 SKILL.md 所在目录，即 `scripts/` 的父目录），执行时替换为实际路径。

解析、label 生成、hierarchy 组装、一致性验证、写入的全部规则均内聚在脚本中，不要绕过脚本手工编写 doc_config.json。规则的设计意图与示例见 `references/script-design.md`，仅在维护或调试脚本时阅读，执行时无需。

退出码约定：

- `0` — 成功，stdout 输出摘要（页面总数、叶子/目录节点、label 对照表、验证结果、冲突/跳过报告、已写入路径）
- `1` — 一致性验证失败，stderr 输出结构化错误清单，未写入文件
- `2` — 输入错误（文件不存在、未解析到页面、重复标题、悬空二级子页面、已存在的输出文件损坏）

### 第 3 步 — 处理脚本输出

- 退出码 `0`：向用户转述脚本摘要；提示 label 为自动生成（更新场景下已有标签已保留），如需调整可手动编辑 `doc_config.json`
- 退出码 `1`（一致性验证失败）：属脚本自身缺陷而非输入问题（正常路径验证必然通过）——向用户报告 stderr 错误清单后停止；不重试输入，不手工编写 `doc_config.json` 绕过。用户要求修复时进入脚本维护模式：先读 `references/script-design.md` 对应规则定位偏差（规格基准），再修改 `scripts/generate_config.py` 并重跑
- 退出码 `2`（输入错误）：转述 stderr 中的错误，按错误类别处理（补充父章节、修复或删除损坏的输出文件等）后重跑；禁止手工改写脚本已拒绝的产物

## 错误处理

- 如果 `read_wiki_structure` 调用失败，报告错误并停止执行
- label 冲突（不同页面生成相同 label）由脚本自动追加 `_2`、`_3` 等后缀，并在摘要中报告
- 无法识别的行由脚本跳过并在摘要中报告
- 重复标题、悬空二级子页面、已有输出文件损坏，均由脚本以退出码 `2` 拒绝执行
