# 详细执行步骤（第 2-7 步）

## When to Read

- 执行第 2 步（解析 Wiki 结构）时
- 执行第 3 步（生成 label）时
- 执行第 4 步（生成 hierarchy）时
- 执行第 5 步（组装 doc_config.json）时
- 执行第 6 步（一致性验证）时
- 执行第 7 步（写入文件）时
- 需要参考 label 生成规则、目录 overview 后缀规则、JSON 示例时

## 第 2 步 — 解析 Wiki 结构

逐行解析返回内容，构建页面对照表。每条记录包含：

| 字段 | 说明 | 示例 |
|---|---|---|
| 章节 ID | 行首的数字前缀 | `2.1` |
| 标题 | 章节 ID 后的其余内容 | `Environment Setup` |
| 层级 | 由章节 ID 格式决定（无小数点 = 一级章节，有小数点 = 二级子页面） | `1` |
| 父章节 ID | 所属一级章节的编号 | `2` |

解析规则：
1. 提取每行中的章节 ID（匹配 `\d+(?:\.\d+)?` 模式的数字部分）
2. 根据章节 ID 格式判断层级：
   - **无小数点**（如 `1`、`2`、`10`）→ 一级章节（层级 0）
   - **有小数点**（如 `2.1`、`10.3`）→ 二级子页面（层级 1）
3. 标题是章节 ID 后的其余内容（去除首尾空白）
4. 父章节 ID = 章节 ID 的整数部分（如 `2.1` → `2`，`1` → `1`）

## 第 3 步 — 生成 label

对每个页面，根据标题生成简洁的 label。如果已存在 `doc_config.json`（更新场景），**保留已有标签**，仅对新页面生成 label。

### 3a — 检查已有标签（更新场景）

如果 `output_path` 已存在有效的 `doc_config.json`：
1. 读取现有文件，提取 `id_to_label` 映射
2. 对于已存在的章节 ID，**复用旧 label**（不重新生成）
3. 仅对旧映射中不存在的章节 ID 生成新 label

这确保用户手动简化的标签在更新时不会被覆盖。

### 3b — label 生成规则

对新页面（或不存在已有配置时），按以下规则生成：

**第 1 步 — 去除括号内容**：
将标题中的 `(xxx)` 整体去除，使用剩余文本生成 label。
- `Configuration System (robot_config)` → `Configuration System`
- `Motion Planning (MoveIt)` → `Motion Planning`

**第 2 步 — slug 生成 + 去除虚词**：
1. 将剩余标题转为小写
2. 去除常见虚词：`and`、`or`、`of`、`the`、`for`、`with`、`a`、`an`、`in`、`on`、`to`
3. 将所有非字母数字字符替换为单个下划线
4. 去除首尾下划线
5. 合并连续下划线为单个

示例：

| 标题 | 去除 `(xxx)` 后 | label |
|---|---|---|
| `IB-Robot Overview` | （无变化） | `ib_robot_overview` |
| `Getting Started` | （无变化） | `getting_started` |
| `Configuration System (robot_config)` | `Configuration System` | `configuration_system` |
| `Protocol Conversion (tensormsg)` | `Protocol Conversion` | `protocol_conversion` |
| `Dataset Conversion (bag_to_lerobot)` | `Dataset Conversion` | `dataset_conversion` |
| `Motion Planning (MoveIt)` | `Motion Planning` | `motion_planning` |
| `5DOF Kinematic Constraints` | （无变化） | `5dof_kinematic_constraints` |
| `Single Source of Truth Pattern` | （无变化） | `single_source_truth_pattern` |
| `Social Control and AI Agent Integration` | （无变化） | `social_control_ai_agent_integration` |
| `Camera Tools (Alignment and ISP Calibration)` | `Camera Tools` | `camera_tools` |

## 第 4 步 — 生成 hierarchy

根据页面的层级关系，生成 `hierarchy` 配置：

**混合模式**：
- **无子页面的一级章节** → 叶子节点（直接输出 `.md` 文件）
- **有子页面的一级章节** → 目录节点（创建子目录，含 `overview.md` + 子页面）

### 4a — 判断节点类型

遍历所有一级章节：
- 如果该章节 ID 下存在任何二级子页面 → **目录节点**
- 否则 → **叶子节点**

### 4b — 生成目录名和文件名

从 label 派生目录名和文件名：

| 节点类型 | 目录名/文件名 key | 说明 |
|---|---|---|
| 叶子节点 | `<label>.md` | 直接作为输出文件名 |
| 目录节点 | `<label>` | 作为子目录名 |
| 目录节点的子页面 | `<sub_label>.md` | 放在子目录下 |

示例：

| 一级章节 | label | 子页面 | hierarchy key |
|---|---|---|---|
| `IB-Robot Overview`（无子页面） | `ib_robot_overview` | 无 | `"ib_robot_overview.md"` |
| `Getting Started`（有子页面） | `getting_started` | 2.1, 2.2 | `"getting_started"` |
| `System Architecture`（无子页面） | `system_architecture` | 无 | `"system_architecture.md"` |

### 4c — 组装 hierarchy 结构

**叶子节点**：
```json
"<label>.md": {"title": "<原始标题>"}
```

**目录节点**：
```json
"<label>": {
    "title": "<原始标题>",
    "subs": {
        "<sub_label>.md": "<子页面标题>",
        ...
    }
}
```

## 第 5 步 — 组装完整 doc_config.json

将三部分组装为完整的 JSON：

```json
{
    "id_to_label": {
        "1": "ib_robot_overview",
        "2": "getting_started",
        "2.1": "environment_setup",
        ...
    },
    "title_to_label": {
        "IB-Robot Overview": "ib_robot_overview",
        "Getting Started": "getting_started",
        "Environment Setup": "environment_setup",
        ...
    },
    "hierarchy": {
        "ib_robot_overview.md": {"title": "IB-Robot Overview"},
        "getting_started": {
            "title": "Getting Started",
            "subs": {
                "environment_setup.md": "Environment Setup",
                "building_the_project.md": "Building the Project"
            }
        },
        "configuration_system": {
            "title": "Configuration System (robot_config)",
            "subs": {
                "robot_configuration_files.md": "Robot Configuration Files",
                ...
            }
        },
        ...
    }
}
```

## 第 6 步 — 一致性验证

在写入文件之前，验证 `id_to_label`、`title_to_label` 和 `hierarchy` 三者完全一致，无遗漏或孤立条目。

### 6a — 交叉校验 `id_to_label` ↔ `title_to_label`

1. 对 `id_to_label` 中的每条记录，验证对应标题在 `title_to_label` 中存在，且映射到**相同 label**
2. 对 `title_to_label` 中的每条记录，验证对应章节 ID 在 `id_to_label` 中存在，且映射到**相同 label**
3. 如发现不一致，报告具体条目并停止

### 6b — 交叉校验 `id_to_label` ↔ `hierarchy`

1. 收集 `id_to_label` 中所有 label 组成集合 `L`
2. 收集 `hierarchy` 中使用的所有 label：
   - 叶子节点 key：去除 `.md` 后缀（如 `ib_robot_overview.md` → `ib_robot_overview`）
   - 目录节点 key：直接使用（如 `getting_started`）
   - `subs` 下的子页面 key：去除 `.md` 后缀
3. 验证 `L == hierarchy_labels`（两个集合必须完全相同）
4. 如果 label 在 `id_to_label` 中存在但不在 `hierarchy` 中 → 报告为**孤立 label**
5. 如果 label 在 `hierarchy` 中存在但不在 `id_to_label` 中 → 报告为**缺失 label**

### 6c — 交叉校验 `title_to_label` ↔ `hierarchy`

1. 收集 `hierarchy` 中所有标题：
   - 叶子节点：`title` 值
   - 目录节点：`title` 值
   - 子页面：`subs` 下的标题值
2. 验证 `hierarchy` 中的每个标题在 `title_to_label` 中存在，且映射到**相同 label**
3. 验证 `title_to_label` 中的每个标题在 `hierarchy` 中出现

### 6d — 验证结果

- **全部通过**：继续第 7 步
- **任一失败**：以结构化列表报告所有不一致项，然后停止，不写入文件

示例错误报告：
```
验证失败：
- 孤立 label（id_to_label 中存在但 hierarchy 中缺失）："7.5" → "model_export_validation_model_utils"
- 缺失 label（hierarchy 中存在但 id_to_label 中缺失）："model_export_and_validation_model_utils.md"
- 标题不一致：id_to_label["7.6"] label="attention_visualization_attention_viz"，但 title_to_label["Attention Visualization (attention_viz)"] label="attention_viz"
```

## 第 7 步 — 写入文件

将生成的 JSON 写入 `output_path` 指定的路径（默认为 `doc_config.json`），使用 4 空格缩进，确保中文字符不被转义（`ensure_ascii=False`）。
