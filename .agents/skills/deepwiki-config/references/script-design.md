# generate_config.py 设计说明

## When to Read

- 修改或调试 `scripts/generate_config.py` 时
- 评审脚本规则变更（虚词表、hierarchy 模式、验证项、退出码）时
- 需要理解脚本某个行为的设计意图时
- 第 2 步退出码 `1`（一致性验证失败）、用户要求定位修复脚本缺陷时

退出码 `0`（成功）或 `2`（输入错误，修正后重跑）时**无需**阅读本文档：全部规则已内聚在脚本中，本文档面向脚本的维护者与排障者。执行时文档与脚本行为不一致，以脚本为准；唯退出码 `1` 的修复场景方向相反——以本文档为比对规格，脚本为被检对象。

## 输入格式

输入为 DeepWiki MCP `read_wiki_structure` 返回的页面列表文本，每行一个页面，项目符号与缩进可选：

```
- 1 IB-Robot Overview
- 2 Getting Started
  - 2.1 Environment Setup
```

## 解析规则

脚本逐行匹配 `\d+(?:\.\d+)?` 提取页面记录：

| 字段 | 提取方式 |
|---|---|
| 章节 ID | 行首数字前缀（如 `2.1`） |
| 标题 | 章节 ID 后的其余内容（去首尾空白） |
| 层级 | 无小数点 → 一级章节；有小数点 → 二级子页面 |
| 父章节 ID | 章节 ID 的整数部分（`2.1` → `2`） |

边界处理：

- 无法识别的行：跳过，摘要中报告行号与原文
- 未解析到任何页面：拒绝（exit 2）
- 重复标题：拒绝（exit 2）——重复标题会折叠 `title_to_label` 条目，必须人工修正输入
- 二级子页面的父章节不存在：拒绝（exit 2）
- 已存在的输出文件无法解析（损坏）：拒绝（exit 2），要求修复或删除后重试

排序：所有输出（`id_to_label`、label 对照表、hierarchy）按章节 ID **数值**排序（`10` 排在 `9` 之后），非字典序。

## label 生成规则

### 更新场景（输出文件已存在且可解析）

- 复用已有 `id_to_label` 中同章节 ID 的旧 label，**不重新生成**——保护用户手动简化的标签不被覆盖
- 仅对新章节 ID 生成新 label

### 新 label 生成

1. 去除标题中的 `(xxx)` 括号内容：`Configuration System (robot_config)` → `Configuration System`
2. slug：转小写 → 去虚词（`and` / `or` / `of` / `the` / `for` / `with` / `a` / `an` / `in` / `on` / `to`）→ 非字母数字字符替换为单个下划线 → 去首尾下划线 → 合并连续下划线

示例：

| 标题 | label |
|---|---|
| `IB-Robot Overview` | `ib_robot_overview` |
| `Getting Started` | `getting_started` |
| `Configuration System (robot_config)` | `configuration_system` |
| `Protocol Conversion (tensormsg)` | `protocol_conversion` |
| `Dataset Conversion (bag_to_lerobot)` | `dataset_conversion` |
| `Motion Planning (MoveIt)` | `motion_planning` |
| `5DOF Kinematic Constraints` | `5dof_kinematic_constraints` |
| `Single Source of Truth Pattern` | `single_source_truth_pattern` |
| `Social Control and AI Agent Integration` | `social_control_ai_agent_integration` |
| `Camera Tools (Alignment and ISP Calibration)` | `camera_tools` |

### 冲突处理

不同页面生成相同 label 时，后到者追加 `_2`、`_3` 等数字后缀，摘要中报告原始 label 与实际分配结果。

## hierarchy 构建规则（混合模式）

- 无子页面的一级章节 → 叶子节点：`"<label>.md": {"title": "<原始标题>"}`
- 有子页面的一级章节 → 目录节点：`"<label>": {"title": "<原始标题>", "subs": {"<sub_label>.md": "<子页面标题>"}}`

目录节点自身的 label 不出现在配置的任何 key 中；`deepwiki_processor.py`（deepwiki-translator 技能）在输出阶段把它映射为 `<label>/overview.md`。

## 一致性验证规则

写入前对三部分做三向校验，任一失败 → exit 1，输出结构化错误清单，**不写入文件**：

- **6a** `id_to_label` ↔ `title_to_label`：借助页面表逐页验证两条映射给出相同 label；`title_to_label` 不得有多余条目
- **6b** `id_to_label` ↔ `hierarchy`：两侧 label 集合必须完全相同；只在前者 → 孤立 label，只在后者 → 缺失 label
- **6c** `title_to_label` ↔ `hierarchy`：两侧标题集合必须完全相同

错误清单示例：

```
验证失败：
- 孤立 label（id_to_label 中存在但 hierarchy 中缺失）："7.5" → "model_export_validation"
- 缺失 label（hierarchy 中存在但 id_to_label 中缺失）："model_export_and_validation.md"
- id_to_label["7.6"] label='attention_visualization' 但 title_to_label['Attention Visualization (attention_viz)'] label='attention_viz'
```

三项均由同一份页面记录确定性派生，正常路径下验证必然通过；验证的价值在于拦截脚本自身缺陷与未来规则变更引入的回归。验证失败应视为脚本 bug 处理（对照本文档检查对应规则），而不是手工修 JSON 绕过。

## 写入格式与摘要

- JSON：4 空格缩进、`ensure_ascii=False`、文件末尾换行
- 摘要（stdout）：页面总数（一级/二级）、叶子/目录节点清单、验证结果、更新场景复用数、冲突与跳过报告、label 对照表（更新场景标注「已有」）、已写入路径
