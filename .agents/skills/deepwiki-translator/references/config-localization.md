# 配置本地化详细步骤

## When to Read

- 执行第 2 步「本地化配置与构建映射」时（开始本地化前必须完整阅读）
- 校验本地化配置时
- 构建英文 H1 到中文 H1 的映射时
- 需要目录 overview「概述」后缀规则的完整 JSON 示例时

## 本地化 doc_config.json

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
    "id_to_label": {
        "1": "getting_started",
        "1.1": "environment_setup"
    },
    "title_to_label": {
        "Getting Started": "getting_started",
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
    "id_to_label": {
        "1": "getting_started",
        "1.1": "environment_setup"
    },
    "title_to_label": {
        "入门指南概述": "getting_started",
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

## 校验本地化配置

新建或重新本地化 `target_config` 后必须校验：

1. `id_to_label` 中的 label 集合未变化。
2. `title_to_label` 中的 label 集合与 `id_to_label` 中的 label 集合一致。
3. `hierarchy` key 与 `subs` key 推导出的 label 集合与 `id_to_label` 中的 label 集合一致。
4. 每个 `hierarchy.title` 和 `hierarchy.subs` 标题都存在于 `title_to_label`，并映射到相同 label。
5. 中文标题不能为空。
6. 中文标题不能重复。

任一校验失败时停止，并报告具体不一致项。不要基于错误配置翻译页面。

## 构建英文 H1 到中文 H1 的映射

映射以页面英文 H1 为键，经 label 连接两个配置，与源文件名无关：

1. 读取源文件的第一个 H1（即英文页面标题）。
2. 用 `source_config.title_to_label` 将英文标题映射到 label。
3. 用该 label 在 `target_config.title_to_label` 中反查中文标题。

示例（label = `getting_started`）：

- 目录 overview：英文 H1 "Getting Started" -> 中文 "入门指南概述"。目录 overview 标题必须带"概述"后缀。
- 子页面：英文 H1 "Environment Setup" -> "环境配置"。
- 叶子页面：英文 H1 "IB-Robot Overview" -> "IB-Robot 概述"。

译文文件名保持与源文件一致；第 5 步生成时按 H1 标题匹配页面，文件名不参与匹配。

### 映射校验（全量与增量均执行）

映射构建后、开始第 4 步前必须校验：

1. `source_dir` 中全部文件的 H1 不得重复；重复即停止并报告涉及的文件与标题。
2. 本轮待翻译文件（增量模式下为第 3 步判定需翻译的文件）的 H1 必须命中映射；未命中即停止并报告，不得编造标题。
3. 增量模式复用旧 `target_config` 时，新增页面 H1 无法命中说明配置已过时：重新执行本文件的本地化生成与校验，不得跳过。
