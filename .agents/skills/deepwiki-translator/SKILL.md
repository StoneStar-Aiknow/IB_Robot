---
name: "deepwiki-translator"
description: "使用配置优先流程将 DeepWiki Markdown 翻译为中文：先本地化 doc_config.json 标题，再让页面 H1 严格使用本地化配置中的标题，同时保持 label、文件名、链接和处理脚本兼容。支持全量和增量翻译模式。"
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
| 内容保护规则、术语表、链接安全分析 | `references/content-protection.md` |
| 配置本地化详细步骤（第 1-3 步） | `references/config-localization.md` |
| 输出生成与验证（第 5-6 步） | `references/verification.md` |

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

这样可以兼容 `deepwiki_processor.py`，因为它本来就要求 `title_to_label` 与 `hierarchy` 中的标题和输入 Markdown 的 H1 完全一致。

## 推荐流程

1. 使用 `split_md.py` 将 DeepWiki 原始内容拆分成英文 `raw_md/*.md`。
2. 翻译 `source_config` 中的标题，生成同 schema 的 `target_config`（详见 `references/config-localization.md`）。
3. 从 `target_config.hierarchy` 构建"文件名 -> 中文 H1"映射。
4. 翻译每个英文 Markdown 到 `target_dir`，文件名保持不变（根据 `mode` 选择全量或增量）。
5. 每个译文文件的第一个 H1 必须设置为 `target_config` 中对应的中文标题。
6. 使用 `--input-dir=target_dir --config-file=target_config` 运行 `deepwiki_processor.py`（详见 `references/verification.md`）。
7. 检查 warnings 和 `link_conversions.xlsx`。

不要让页面翻译过程自行发挥生成 H1。配置是页面标题的唯一事实来源。

## 翻译模式

### 全量翻译（`mode=full`）

从零翻译所有页面。适用场景：

- 新启动翻译项目。
- 源内容变化较大，需要完全重新翻译。
- 用户明确要求全量重翻。

流程：

1. 本地化 `doc_config.json` → `target_config`（如已存在则覆盖）。
2. 校验本地化配置。
3. 构建文件名到 H1 的映射。
4. 翻译 `source_dir` 中**每一个** `.md` 文件，写入 `target_dir`。
5. 运行 `deepwiki_processor.py` 并验证。

### 增量翻译（`mode=incremental`）

仅翻译新增或修改的页面，保留已有翻译。适用场景：

- 源目录只新增了少量页面。
- 特定页面有更新需要重新翻译。
- 用户希望避免重复翻译已完成的页面。

流程：

1. 如果 `target_config` 不存在，先本地化 `doc_config.json`；否则复用已有的 `target_config`。
2. 校验本地化配置。
3. 使用 Glob 分别列出 `source_dir` 和 `target_dir` 中的文件。
4. 判断哪些文件需要翻译：
   - 在 `source_dir` 中但**不在** `target_dir` 中 → 新页面，必须翻译。
   - 两个目录都存在的文件 → 默认跳过；仅在用户明确要求或确认源文件已变更时重新翻译。
   - 在 `target_dir` 中但**不在** `source_dir` 中 → 报告为可能过时的文件，未经用户确认不删除。
5. 对每个需要翻译的文件，执行读取、翻译、写入。
6. 运行 `deepwiki_processor.py` 并验证。

## 翻译方式约束

**禁止编写脚本来执行翻译工作。** LLM 本身就是翻译引擎。对每个需要翻译的文件：

1. 使用 **Read** 工具从 `source_dir` 读取源 Markdown 文件。
2. LLM 就地翻译内容，遵循所有内容保护规则和术语表（详见 `references/content-protection.md`）。
3. 使用 **Write** 工具将翻译后的内容写入 `target_dir`，文件名不变，UTF-8 编码。

此约束同时适用于全量和增量模式。禁止创建 Python、Bash 或任何其他脚本来自动化翻译循环。Read → LLM 翻译 → Write 的循环在对话中逐文件执行。

## 第 4 步：翻译 Markdown 页面

### 全量模式

对 `source_dir` 中每个英文文件：

1. 根据文件名从映射中找到对应中文 H1。
2. 使用 **Read** 工具完整读取源 Markdown。
3. 保持 Markdown 结构。
4. 将自然语言内容翻译为简洁技术中文。
5. 将第一个 H1 替换为 `# <target_config 中的中文标题>`。
6. 使用 **Write** 工具以相同文件名和 UTF-8 编码写入 `target_dir`。

### 增量模式

1. 使用 Glob 分别列出 `source_dir` 和 `target_dir` 中所有 `.md` 文件。
2. 识别新文件（在 `source_dir` 中但不在 `target_dir` 中）。
3. 对每个新文件，执行与全量模式相同的 Read → 翻译 → Write 流程。
4. 两个目录都存在的文件，默认跳过；仅在用户明确要求时重新翻译。
5. 报告 `target_dir` 中已不在 `source_dir` 中的文件为可能过时的文件。

如果某个源文件无法在本地化配置中找到标题，停止并报告。不要自行编造标题。

## 约束

- 本流程不向 `doc_config.json` 添加新字段。
- 不修改 label、文件名、hierarchy key 或 `id_to_label`。
- 不翻译代码、命令、包名、API 名、文件路径、URL、anchor 或 Mermaid 图的任何内容（包括展示标签和节点文本）。
- 不以生成后的 `ib_robot/` 作为主要翻译源。
- 不手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。
- 禁止编写脚本执行翻译。使用 Read 工具读取源文件，使用 LLM 能力翻译，使用 Write 工具写入结果。
