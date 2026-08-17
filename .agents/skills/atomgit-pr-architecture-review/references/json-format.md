## When to Read

阅读本文件的典型场景：

- 需要了解 `arch_issues.json` 的完整字段说明、必填/可选字段类型与含义
- 需要查看完整的 JSON 示例（数组结构、单条问题对象）
- 提交审查前需要执行 JSON 格式验证命令
- 提交时出现 `AttributeError: 'str' object has no attribute 'get'` 等结构错误，需要确认数组 vs 对象写法
- 需要确认 `severity` / `pillar` 字段的合法取值

## JSON 格式规范

**重要**: `arch_issues.json` 必须遵循以下格式，否则提交会失败。

### 文件结构

JSON 文件必须是一个**数组**，直接包含问题对象：

```json
[
  {
    "file": "path/to/file.py",
    "line": 42,
    "title": "问题标题",
    ...
  }
]
```

**错误示例** ❌ (会导致 `AttributeError: 'str' object has no attribute 'get'`):
```json
{
  "issues": [...],
  "summary": {...}
}
```

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | string | 文件路径（相对于项目根目录） |
| `line` | int | 问题起始行号 |
| `title` | string | 问题标题（简洁明了） |
| `description` | string | 问题描述（可包含 `\n` 换行） |
| `severity` | string | 严重性：`critical` / `error` / `warning` / `suggestion` |
| `pillar` | string | 架构支柱：`ssot` / `contract` / `control_mode` / `tensormsg` / `ros2` / `python` / `docs` 等 |

### 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `fix` | string | 修复建议（可包含代码示例） |
| `context_code` | string | 相关代码片段 |

### 完整示例

```json
[
  {
    "file": "src/robot_config/robot_config/tools/camera_alignment.py",
    "line": 1,
    "title": "robot_config包职责蔓延：引入了389行重型交互式UI工具",
    "description": "camera_alignment.py包含大量交互式UI逻辑...\n\n违反原则: Package-Specific Architecture Compliance\n\n影响:\n- 包职责边界模糊\n- 违背单一职责原则",
    "severity": "critical",
    "pillar": "python",
    "fix": "将camera_alignment工具迁移到dataset_tools包:\n\n```\nsrc/dataset_tools/\n├── dataset_tools/\n│   ├── camera_alignment.py\n```\n\n理由:\n1. dataset_tools职责: Episode recording\n2. 数据流程: 校准 → 录制"
  },
  {
    "file": "src/robot_config/robot_config/tools/camera_alignment.py",
    "line": 18,
    "title": "OpenCV模块加载逻辑应提取为独立工具模块",
    "description": "包含约90行系统级路径搜索...",
    "severity": "warning",
    "pillar": "python",
    "fix": "提取为opencv_utils.py模块"
  }
]
```

### 验证 JSON 格式

提交前可以验证 JSON 格式：

```bash
python3 -m json.tool ./tmp/ib_robot_pr_123_arch_issues.json > /dev/null && echo "✅ JSON格式正确"
```
