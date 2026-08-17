# 配置本地化详细步骤

## When to Read

- 执行第 1 步（本地化 doc_config.json）时
- 执行第 2 步（校验本地化配置）时
- 执行第 3 步（构建文件名到 H1 的映射）时
- 需要了解目录 overview 标题后缀规则

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
