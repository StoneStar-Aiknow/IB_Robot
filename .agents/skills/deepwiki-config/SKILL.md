---
name: "deepwiki-config"
description: "根据 DeepWiki MCP 的 read_wiki_structure 返回内容，自动生成 deepwiki_processor 所需的 doc_config.json。当用户需要为新的 DeepWiki 仓库生成配置文件时调用。"
---

# DeepWiki 配置生成器

根据 DeepWiki MCP 的 `read_wiki_structure` 返回内容，自动生成 `deepwiki_processor.py` 所需的 `doc_config.json` 配置文件。

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
| 第 2-7 步详细执行步骤（解析 Wiki 结构、生成 label、生成 hierarchy、组装 JSON、一致性验证、写入文件） | `references/detailed-steps.md` |

Do not expose these references as separate skills.

## 生成产物结构

`doc_config.json` 由三部分组成：

| 字段 | 用途 |
|---|---|
| `id_to_label` | 章节 ID（如 `"2.1"`）→ label（如 `"environment_setup"`） |
| `title_to_label` | 原始标题 → label |
| `hierarchy` | 输出文件/目录树（叶子节点为 `<label>.md`，目录节点含 `title` + `subs`） |

## 工作流程

### 第 1 步 — 获取 Wiki 结构

调用 `mcp_deepwiki_read_wiki_structure`，参数 `repoName = <repo>`。

MCP 返回类似如下格式的页面列表：

```
- 1 IB-Robot Overview
- 2 Getting Started
  - 2.1 Environment Setup
  - 2.2 Building the Project
- 3 Core Concepts
  - 3.1 Single Source of Truth Pattern
...
```

### 第 2 步 — 解析 Wiki 结构

逐行解析返回内容，构建页面对照表（每条记录含章节 ID、标题、层级、父章节 ID）。

详细的解析规则见 `references/detailed-steps.md` 第 2 步。

### 第 3 步 — 生成 label

对每个页面生成简洁 label。**更新场景下保留已有标签**，仅对新页面生成 label。

label 生成规则：去除 `(xxx)` 括号内容 + slug 生成 + 去除常见虚词（and/or/of/the/for/with/a/an/in/on/to）。

详细规则和 10 行示例见 `references/detailed-steps.md` 第 3 步。

### 第 4 步 — 生成 hierarchy

采用**混合模式**：

- 无子页面的一级章节 → 叶子节点（直接输出 `<label>.md`）
- 有子页面的一级章节 → 目录节点（含 `overview.md` + 子页面 `<sub_label>.md`）

节点类型判断、目录名生成、hierarchy 组装结构详见 `references/detailed-steps.md` 第 4 步。

### 第 5 步 — 组装完整 doc_config.json

将 `id_to_label`、`title_to_label`、`hierarchy` 三部分组装为完整 JSON。

完整 JSON 示例见 `references/detailed-steps.md` 第 5 步。

### 第 6 步 — 一致性验证

在写入文件之前，验证三者完全一致：

- **6a**: `id_to_label` ↔ `title_to_label` 双向校验
- **6b**: `id_to_label` ↔ `hierarchy` 标签集合校验（孤立/缺失 label 检测）
- **6c**: `title_to_label` ↔ `hierarchy` 标题映射校验
- **6d**: 全部通过 → 继续第 7 步；任一失败 → 报告所有不一致项并停止

验证逻辑和错误报告示例见 `references/detailed-steps.md` 第 6 步。

### 第 7 步 — 写入文件

将生成的 JSON 写入 `output_path`（默认 `doc_config.json`），使用 4 空格缩进，`ensure_ascii=False`。

### 第 8 步 — 输出摘要

生成完成后，输出以下信息：

- 页面总数
- 叶子节点数量和列表
- 目录节点数量和列表（含子页面数）
- 生成的 label 对照表（供用户检查是否需要手动调整）
- 验证结果：三者一致，或列出不一致项
- 提示用户：label 自动生成；更新场景下已有标签会被保留，如需调整可手动编辑 `doc_config.json`

## 错误处理

- 如果 `read_wiki_structure` 调用失败，报告错误并停止执行
- 如果解析过程中遇到无法识别的行格式，跳过该行并在摘要中报告
- 如果生成的 label 存在冲突（不同页面生成相同 label），在冲突的 label 后追加 `_2`、`_3` 等后缀，并在摘要中报告

## 约束

- label 通过去除括号内容 + slug 去虚词生成
- 更新场景（已存在 `doc_config.json`）下，保留已有章节 ID 的旧 label，仅对新页面生成 label
- hierarchy 采用混合模式：无子页面的一级章节为叶子节点，有子页面的为目录节点
- 生成的 JSON 必须与 `deepwiki_processor.py` 的输入格式完全兼容
- 目录名和文件名中的连字符统一转为下划线
