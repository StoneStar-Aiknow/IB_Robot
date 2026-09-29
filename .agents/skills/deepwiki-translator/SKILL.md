---
name: "deepwiki-translator"
description: "将 DeepWiki 英文 Markdown 端到端转换为可交付的中文文档（全量/增量翻译、构建目录与索引、校验链接与锚点）。Use when users mention 'translate DeepWiki markdown', 'DeepWiki 翻译', '汉化 DeepWiki', 'localize doc_config', '全量翻译', '增量翻译', or 'incremental translation'."
---

# DeepWiki 翻译器

使用现有 `doc_config.json` schema 将 DeepWiki 生成的英文 Markdown 翻译为中文。流程采用"配置优先"：先翻译 `doc_config.json` 中的标题，再翻译各页面，并让每个页面的 H1 直接使用本地化配置中的中文标题。

## 何时调用

- 用户需要从零开始将所有英文 `raw_md/*.md` 翻译为中文 `raw_md_zh/*.md`（全量翻译）。
- 用户已翻译部分页面，只需翻译新增或修改的页面（增量翻译）。
- 用户希望中文页面标题生效，但不重构 `doc_config.json` schema。
- 用户希望翻译后的 Markdown 能兼容当前 `deepwiki_processor.py` 的标题匹配逻辑。
- 用户需要检查翻译后的链接、H1、配置一致性、Mermaid 图、代码块或 Sphinx 输出。

## 输入参数

| 参数 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `mode` | 否 | 翻译模式：`full`（默认，全量）或 `incremental`（增量） | `incremental` |
| `source_dir` | 否 | 英文 Markdown 源目录，默认 `raw_md/` | `migration/raw_md` |
| `target_dir` | 否 | 中文 Markdown 输出目录，默认 `raw_md_zh/` | `migration/raw_md_zh` |
| `source_config` | 否 | 英文 `doc_config.json` 路径 | `migration/doc_config.json` |
| `target_config` | 否 | 本地化配置路径；只有用户明确要替换当前配置时才直接写 `doc_config.json` | `migration/doc_config_zh.json` |
| `output_dir` | 否 | 生成后的中文文档目录 | `migration/ib_robot_zh` |
| `branch` | 否 | 传给 `deepwiki_processor.py` 的 AtomGit 分支，默认 `master` | `master` |

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| 内容保护规则、术语表、链接安全分析，执行第 4 步翻译前必读 | `references/content-protection.md` |
| 配置本地化细则、校验清单、映射构建规则、目录 overview「概述」后缀规则的完整 JSON 示例，执行第 2 步前必读 | `references/config-localization.md` |
| 生成命令、输出验证清单、失败处置，执行第 5 步前必读 | `references/output-pipeline.md` |
| 三层防护、链接修复循环、源链接形态规范、已知坑，第 6 步退出码非 0 或出现 `(dangling anchor)` 条目时必读 | `references/link-repair.md` |

Do not expose these references as separate skills.

## 核心原则

不改变配置 schema。

本地化后的配置仍然只保留原有三个顶层字段：

```json
{
    "id_to_label": {},
    "title_to_label": {},
    "hierarchy": {}
}
```

只翻译标题字符串：

- 将 `title_to_label` 的 key 从英文翻译为中文。
- 将每个 `hierarchy.*.title` 的值从英文翻译为中文。
- 将每个 `hierarchy.*.subs` 的值从英文翻译为中文。
- `id_to_label` 所有条目必须原样保留。
- 所有 label 值必须原样保留。
- 所有 hierarchy key、目录名和 Markdown 文件名必须原样保留。

目录 overview 标题后缀规则：`hierarchy` 条目含 `subs` 字段（目录 overview 页面）时，翻译后的 `title` 必须以"概述"结尾（如 "Getting Started" → "入门指南概述"），避免与章节同名；叶子页面不加后缀。完整规则与 JSON 示例见 `references/config-localization.md`。

这样可以兼容 `deepwiki_processor.py`，因为它本来就要求 `title_to_label` 与 `hierarchy` 中的标题和输入 Markdown 的 H1 完全一致。

## 执行流程

本技能唯一权威流程。`scripts/` 下 `deepwiki_config` / `deepwiki_generator` / `deepwiki_links` / `deepwiki_pages` 为内部模块（无 CLI，仅被命令行脚本导入，不可独立执行）。命令中的 `<skill_dir>` 指本技能目录（本 SKILL.md 所在目录，即 `scripts/`、`references/` 的父目录），执行时替换为实际路径；脚本默认参数（如 `raw_md/`、`doc_config.json`）以当前工作目录为基准。

### 第 1 步 — 拆分原始 Markdown

运行 `python -X utf8 <skill_dir>/scripts/split_md.py <raw_md_file> <source_dir>`，将 DeepWiki 导出的 `# Page: <Title>` 原始 Markdown 拆分为扁平页面文件（文件名由标题 slug 生成并自动补全 H1）。

### 第 2 步 — 本地化配置与构建映射

**执行本步前必须完整阅读 `references/config-localization.md`**。按其要求生成同 schema 的 `target_config` 并完成校验（增量模式下已存在则直接复用，跳过生成与校验）；随后从 `target_config.hierarchy` 构建"文件名 -> 中文 H1"映射。

### 第 3 步 — 确定翻译范围

按 `mode` 选择翻译范围，文件判定逻辑仅在此定义。

#### 全量翻译（`mode=full`）

从零翻译所有页面。适用场景：

- 新启动翻译项目。
- 源内容变化较大，需要完全重新翻译。
- 用户明确要求全量重翻。

翻译范围：`source_dir` 中每一个 `.md` 文件。

#### 增量翻译（`mode=incremental`）

仅翻译新增或修改的页面，保留已有翻译。适用场景：

- 源目录只新增了少量页面。
- 特定页面有更新需要重新翻译。
- 用户希望避免重复翻译已完成的页面。

先用 Glob 分别列出 `source_dir` 和 `target_dir` 中的 `.md` 文件，再按下表逐文件判定：

| 文件状态 | 判定 |
|---|---|
| 在 `source_dir` 中但不在 `target_dir` 中 | 新页面，必须翻译 |
| 两个目录都存在 | 默认跳过；仅在用户明确要求或确认源文件已变更时重新翻译 |
| 在 `target_dir` 中但不在 `source_dir` 中 | 报告为可能过时的文件，未经用户确认不删除 |

增量只减少待翻译文件，不缩小后续步骤范围：第 5 步始终读取整个 `target_dir`（旧译文 + 新译文）重新生成并整体重建 `output_dir`，第 5-6 步的输出检查与锚点校验始终覆盖整个 `output_dir`。

### 第 4 步 — 逐文件翻译

**禁止编写脚本执行翻译，LLM 本身就是翻译引擎**，Read → 翻译 → Write 循环在对话中逐文件执行。本步同时适用于全量和增量模式。

**执行本步前必须完整阅读 `references/content-protection.md`**。对每个待翻译文件：

1. 根据文件名从映射中找到对应中文 H1（缺失时停止并报告，不得编造）。
2. 使用 **Read** 工具从 `source_dir` 完整读取源 Markdown。
3. 按 `references/content-protection.md` 的规则翻译为简洁技术中文，保持 Markdown 结构。
4. 将第一个 H1 替换为 `# <target_config 中的中文标题>`。
5. 使用 **Write** 工具以相同文件名和 UTF-8 编码写入 `target_dir`。

### 第 5 步 — 生成中文文档并检查输出

**执行本步前必须完整阅读 `references/output-pipeline.md`**。按其命令生成中文文档（**输出目录会被重建，先清空**）并按其「验证清单」检查输出；汇总全部失败项，统一修复源（`raw_md_zh` 或 `target_config`）后重新生成并完整重检，本步全绿后才进入第 6 步。

### 第 6 步 — 锚点与链接校验

运行 `python -X utf8 <skill_dir>/scripts/verify_anchors.py --output-dir <output_dir>` 对生成结果做锚点与相对链接全量校验，并检查 `link_conversions.json` 中的 `(dangling anchor)` 条目；退出码 0 且无该类条目为干净。

**出现任一问题时必须完整阅读 `references/link-repair.md`**，按其「链接修复循环」处理。

生成文档的外部 URL 与 AtomGit 源链接校验由独立的 `doc-link-validator` 技能承担，用户要求交付前链接体检或检查 AtomGit 链接时调用。

不要让页面翻译过程自行发挥生成 H1。配置是页面标题的唯一事实来源。

## 输出摘要

完成后输出：

- 使用的翻译模式（全量或增量）与本地化配置路径（新建或复用）。
- 翻译的配置标题数量、翻译的 Markdown 文件数量（增量模式含跳过的文件数）。
- 缺少配置映射的源文件、空标题或重复标题情况。
- `references/output-pipeline.md`「验证清单」检查结果（如有失败项）。
- `verify_anchors.py` 校验结果（扫描文件数、问题数）。
- 生成目录和链接转换报告路径。

## 约束

- 本流程不向 `doc_config.json` 添加新字段。
- 不修改 label、文件名、hierarchy key 或 `id_to_label`。
- 不翻译代码、命令、包名、API 名、文件路径、URL、anchor 或 Mermaid 图的任何内容（包括展示标签和节点文本）。
- 不以生成后的 `ib_robot/` 作为主要翻译源。
- 不手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。
- 禁止编写脚本执行翻译（见第 4 步）。
