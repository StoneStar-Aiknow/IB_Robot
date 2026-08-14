# 静态风险扫描与修复规则

## When to Read

- 执行"静态风险扫描"环节时
- 需要判断某段 Mermaid 是否属于高风险模式时
- 需要参考 flowchart 边/节点 label、state diagram transition label 的修复规则时

## 静态风险扫描

编辑前先做源文件扫描。该扫描不能替代浏览器验证。

```bash
python3 - <<'PY'
from pathlib import Path
import re

patterns = [
    r'^\s*\["',
    r'--?>\s*\["',
    r'\.->\s*\["',
    r'^\s*subgraph\s+"[^"]+"\s+\[',
    r'\s--\s*"',
    r'\s--"',
]
compiled = [re.compile(pattern) for pattern in patterns]
for path in sorted(Path("docs").rglob("*.md")):
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if any(pattern.search(line) for pattern in compiled):
            print(f"{path}:{line_number}:{line}")
PY
```

<details><summary>Windows PowerShell 可选扩展</summary>

```powershell
Get-ChildItem -LiteralPath "docs" -Recurse -File -Filter "*.md" |
  Select-String -Pattern '^\s*\["','--?>\s*\["','\.->\s*\["','^\s*subgraph\s+"[^"]+"\s+\[','\s--\s*"','\s--"'
```

</details>

## 高风险模式

| 模式 | 失败原因 | 仅语法修复 |
|---|---|---|
| `A --> ["label"]` | 匿名节点作为边端点 | 创建稳定节点 id：`A --> node_id["label"]` |
| `["label"]` 独立声明 | 匿名节点声明 | 创建稳定节点 id：`node_id["label"]` |
| `subgraph "ID" ["Title"]` | subgraph 写法不兼容 | `subgraph ID["Title"]` |
| `A -- "label" --> B` | 旧式边 label 可能失败 | `A -->|"label"| B` |
| `A --> B : "label"` | flowchart 旧式尾随 label | `A -->|"label"| B` |
| `A -->|/topic (type)| B` | edge label 含未加引号的括号 | `A -->|"/topic (type)"| B` |
| `B{func(arg)}` | diamond 节点 label 含未加引号的括号 | `B{"func(arg)"}` |
| `A --> B: file.py:1-2` in `stateDiagram-v2` | transition label 内含额外冒号 | `A --> B: file.py#58;1-2` |

## 修复规则

### Flowchart 边 Label

使用 pipe label 语法。

```mermaid
A -->|"label with (parentheses), /slashes, or :colon"| B
```

当 label 含有 `(`、`)`、`/`、`:`、`<`、`>`、`,` 或其他标点时，优先加引号。

### Flowchart 节点 Label

节点形状内的 label 如果包含标点，应加引号。

```mermaid
B{"get_scene_file(scene_name, platform)"}
C["/base_velocity_controller/commands (std_msgs/Float64MultiArray)"]
```

### State Diagram Transition Label

`stateDiagram-v2` 使用 `A --> B: label`。如果 label 自身还需要额外冒号，应把内部冒号转成 Mermaid entity 文本：

```mermaid
Idle --> GoalEvaluation: handle_goal()<br/>episode_recorder.py#58;277-288
```

### 保留原内容

除 Mermaid 语法转义外，保持可见 label 不变。例如 `file.py#58;277-288` 应在 Mermaid 中显示为预期的 `file.py:277-288`。
