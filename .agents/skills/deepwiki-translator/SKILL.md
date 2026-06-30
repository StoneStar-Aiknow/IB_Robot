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
2. LLM 就地翻译内容，遵循所有内容保护规则和术语表。
3. 使用 **Write** 工具将翻译后的内容写入 `target_dir`，文件名不变，UTF-8 编码。

此约束同时适用于全量和增量模式。禁止创建 Python、Bash 或任何其他脚本来自动化翻译循环。Read → LLM 翻译 → Write 的循环在对话中逐文件执行。

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
2. 翻译 `source_config` 中的标题，生成同 schema 的 `target_config`。
3. 从 `target_config.hierarchy` 构建"文件名 -> 中文 H1"映射。
4. 翻译每个英文 Markdown 到 `target_dir`，文件名保持不变（根据 `mode` 选择全量或增量）。
5. 每个译文文件的第一个 H1 必须设置为 `target_config` 中对应的中文标题。
6. 使用 `--input-dir=target_dir --config-file=target_config` 运行 `deepwiki_processor.py`。
7. 检查 warnings 和 `link_conversions.xlsx`。

不要让页面翻译过程自行发挥生成 H1。配置是页面标题的唯一事实来源。

## 第 1 步：本地化 doc_config.json

读取 `source_config`，生成同 schema 的 `target_config`。

增量模式下，如果 `target_config` 已存在，则复用，除非用户要求重新本地化。

### 必须原样保留

- 顶层字段名：`id_to_label`、`title_to_label`、`hierarchy`。
- `id_to_label` 的所有 key 和 value。
- `title_to_label` 的所有 value。
- `hierarchy` 的所有对象 key。
- `subs` 的所有 key，例如 `environment_setup.md`。
- label 字符串，例如 `environment_setup`、`motion_planning`、`5dof_kinematic_constraints`。

### 需要翻译

- `title_to_label` 的 key。
- `hierarchy.<page>.title` 的值。
- `hierarchy.<section>.subs.<file>` 的值。

### 目录 Overview 标题后缀规则

当 `hierarchy` 条目包含 `subs` 字段（即该条目是目录 overview 页面）时，其翻译后的 `title` 必须以"概述"结尾。这是为了避免 overview 页面与其所属章节同名。例如，英文标题为 "Getting Started"，则中文标题应为"入门指南概述"，而不是"入门指南"。"概述"后缀明确表示这是该章节的概述页，与章节名称本身区分开。

叶子页面（不含 `subs` 的条目）**不**加"概述"后缀。

### 示例

英文源配置：

```json
{
    "title_to_label": {
        "Environment Setup": "environment_setup"
    },
    "hierarchy": {
        "getting_started": {
            "title": "Getting Started",
            "subs": {
                "environment_setup.md": "Environment Setup"
            }
        }
    }
}
```

中文目标配置（注意目录 overview 标题加了"概述"后缀）：

```json
{
    "title_to_label": {
        "环境配置": "environment_setup"
    },
    "hierarchy": {
        "getting_started": {
            "title": "入门指南概述",
            "subs": {
                "environment_setup.md": "环境配置"
            }
        }
    }
}
```

## 第 2 步：校验本地化配置

翻译页面前必须校验：

1. `id_to_label` 中的 label 集合未变化。
2. `title_to_label` 中的 label 集合与 `id_to_label` 中的 label 集合一致。
3. `hierarchy` key 与 `subs` key 推导出的 label 集合与 `id_to_label` 中的 label 集合一致。
4. 每个 `hierarchy.title` 和 `hierarchy.subs` 标题都存在于 `title_to_label`，并映射到相同 label。
5. 中文标题不能为空。
6. 中文标题不能重复。

任一校验失败时停止，并报告具体不一致项。不要基于错误配置翻译页面。

## 第 3 步：构建文件名到 H1 的映射

使用 `target_config.hierarchy` 构建每个译文 Markdown 文件的标准 H1：

- 叶子页面：`"overview.md": {"title": "IB-Robot 概述"}` 映射 `overview.md` -> `IB-Robot 概述`。
- 目录 overview：`"getting_started": {"title": "入门指南概述"}` 映射源文件 `getting_started.md` -> `入门指南概述`。目录 overview 标题必须带"概述"后缀。
- 子页面：`"environment_setup.md": "环境配置"` 映射 `environment_setup.md` -> `环境配置`。

注意：目录 overview 的源文件仍然是 flat raw markdown 中按 label 命名的文件，例如 `getting_started.md`；后续生成时才会变成 `getting_started/overview.md`。

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

## 内容保护规则

以下内容不得翻译或改写：

- Markdown 链接目标：`[text](target)` 中的 `target`。
- 图片目标：`![alt](target)` 中的 `target`。
- 文件路径、目录名、文件名、anchor、URL、源码行号引用。
- 空链接源码引用文本，例如 `[scripts/setup.sh:1-62]()`。
- 反引号包裹的行内代码。
- 围栏代码块内容和代码块语言标记。
- Mermaid 图全部内容：图语法、节点 ID、连线操作符、subgraph ID、展示标签（引号包裹或无引号）、节点文本及所有其他 Mermaid 标记。
- HTML 标签和属性。
- 如果出现 RST 指令和选项，也必须保持结构。
- 产品名、包名、命令、API、类名、函数名、topic、action、参数名和环境变量名。

以下内容可以安全翻译：

- 段落正文。
- 非 H1 标题。
- 链接目标非空且保持不变时的链接显示文本。
- 表格中的自然语言单元格，代码/路径单元格保持原样。

## 术语表

全站保持术语一致：

| 英文 | 中文 |
|---|---|
| inference | 推理 |
| deployment | 部署 |
| teleoperation | 遥操作 |
| dataset | 数据集 |
| pipeline | 流水线 |
| action dispatch | 动作分发 |
| motion planning | 运动规划 |
| configuration | 配置 |
| validation | 验证 |
| submodule | 子模块 |
| workspace | 工作空间 |

以下名称默认不翻译，除非用户明确要求：`IB-Robot`、`LeRobot`、`ROS 2`、`MoveIt`、`openEuler`、`OpenHarmony`、`AtomGit`、`Hugging Face`、`Conda`、`venv`、`colcon`、`rosdep`。

## 第 5 步：生成中文文档

运行 `deepwiki_processor.py` 时使用：

```bash
python deepwiki_processor.py --input-dir <target_dir> --output-dir <output_dir> --config-file <target_config> --branch <branch>
```

不要手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。

## 第 6 步：验证输出

验证是必需步骤：

1. 每个配置标题都作为且只作为一个译文 Markdown 文件的第一个 H1 出现。
2. 每个译文 Markdown 文件都被 `hierarchy` 使用。
3. `deepwiki_processor.py` 不输出 `Configured title missing from input` warning。
4. `deepwiki_processor.py` 不输出 `Input page not used by hierarchy` warning。
5. 围栏代码块数量与英文源文件一致。
6. 在处理器转换前，Markdown 链接目标和图片目标与英文源文件一致。
7. `link_conversions.xlsx` 中的转换符合预期，没有被翻译过的路径或 URL。
8. 生成的 `index.rst` toctree 条目指向实际存在的生成文件。

## 链接安全分析

只要 label 和文件名保持不变，配置优先流程可以保持链接稳定。

高风险错误：

- 翻译 `environment_setup` 这类 label 值。
- 翻译 hierarchy key 或 `subs` 文件名。
- 翻译链接目标、anchor、源码引用或仓库路径。
- 页面 H1 与本地化 `doc_config.json` 标题不一致。
- `title_to_label` 中出现重复中文标题。
- 翻译 Mermaid 图的任何部分，包括展示标签、节点文本或代码块。

安全做法：

- 只翻译配置中的标题字符串，不翻译 label 或 key。
- 页面 H1 直接复制本地化配置中的标题。
- 可以翻译普通链接显示文本，但必须保留 target。
- 文件名和输出路径继续使用英文 slug。

## 输出摘要

完成后输出：

- 使用的翻译模式（全量或增量）。
- 本地化配置路径。
- 翻译的配置标题数量。
- 翻译的 Markdown 文件数量（增量模式下同时报告跳过的文件数）。
- 缺少配置映射的源文件。
- 空标题或重复标题情况。
- 链接验证结果。
- `deepwiki_processor.py` warnings，如有。
- 生成目录和链接转换报告路径。

## 约束

- 本流程不向 `doc_config.json` 添加新字段。
- 不修改 label、文件名、hierarchy key 或 `id_to_label`。
- 不翻译代码、命令、包名、API 名、文件路径、URL、anchor 或 Mermaid 图的任何内容（包括展示标签和节点文本）。
- 不以生成后的 `ib_robot/` 作为主要翻译源。
- 不手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。
- 禁止编写脚本执行翻译。使用 Read 工具读取源文件，使用 LLM 能力翻译，使用 Write 工具写入结果。
