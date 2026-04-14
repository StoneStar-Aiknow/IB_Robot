"""
LLM Architecture Reviewer
使用 LLM 进行架构合规性审查
"""

import os
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ArchitectureIssue:
    """架构问题"""

    file: str
    line: int
    title: str
    description: str
    severity: str
    pillar: str
    fix: Optional[str] = None
    context_code: Optional[str] = None


class LLMArchitectureReviewer:
    """LLM 架构审查器"""

    ARCHITECTURE_PILLARS = """
IB_Robot 架构四大支柱：

1. **Spec-Driven Configuration (SSOT)**
   - 所有配置项应从 YAML 文件加载
   - 禁止硬编码阈值、超时、频率等参数
   - 使用 robot_config.yaml 和 skill_config.yaml

2. **Layered Decoupling**
   - 硬件层、驱动层、业务层严格分离
   - 使用接口抽象，避免直接依赖
   - 遵循依赖注入原则

3. **TensorMsg Protocol Conversion**
   - ROS 消息与 Tensor 之间必须通过 tensormsg 模块转换
   - 禁止直接操作 tensor 或绕过协议层
   - 确保跨域数据转换的统一性

4. **ROS 2 Native Integration**
   - 使用 ROS 2 原生的 topic/service/action 通信
   - 禁止使用 socket、subprocess、HTTP 等非 ROS 方式
   - 硬件访问通过 ros2_control

5. **Package-Specific Architecture Compliance**
   - **包职责隔离**: 每个 ROS 包必须遵循其在架构中定义的职责边界。
   - **严禁职责越界**: 
     - 例如 `robot_teleop` 应仅负责传感器数据映射，严禁引入 IK (Inverse Kinematics) 或运动规划逻辑。
     - 如果驱动包开始持有 ROS Node 句柄并调用其他重型服务，必须提出警告。
     - 检查改动是否使包演变成了“重型节点”，违背了关注点分离原则。
"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def review_file(
        self, file_path: str, file_content: str, diff: str
    ) -> List[ArchitectureIssue]:
        """审查单个文件的架构合规性"""
        prompt = self._build_review_prompt(file_path, file_content, diff)
        response = self._call_llm(prompt)
        return self._parse_issues(response, file_path)

    def _build_review_prompt(self, file_path: str, file_content: str, diff: str) -> str:
        """构建架构审查提示词"""
        return f"""你是一个机器人软件架构审查专家。请审查以下代码是否遵循 IB_Robot 架构规范。

{self.ARCHITECTURE_PILLARS}

**文件**: {file_path}

**代码变更 (Diff)**:
```
{diff or "(无法获取diff)"}
```

**完整文件内容**:
```
{file_content}
```

请检查代码是否违反以上四大架构支柱，并按照以下JSON格式输出发现的问题：

```json
[
  {{
    "line": 10,
    "title": "简短的问题标题",
    "description": "详细的问题描述",
    "severity": "error|warning|suggestion|info",
    "pillar": "ssot|tensormsg|ros2|python",
    "fix": "修复建议",
    "context_code": "相关代码片段"
  }}
]
```

**重要**：
- 只输出JSON数组，不要包含其他文字
- 如果没有问题，输出 `[]`
- pillar 应该是以下之一：ssot, tensormsg, ros2, python
- severity: error（必须修复）, warning（应该修复）, suggestion（建议）, info（信息）"""

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API"""
        if self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _call_anthropic(self, prompt: str) -> str:
        """调用 Anthropic Claude API"""
        import anthropic

        client_options = {"api_key": self.api_key}
        if self.base_url:
            client_options["base_url"] = self.base_url

        client = anthropic.Anthropic(**client_options)

        response = client.messages.create(
            model=self.llm_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _parse_issues(self, response: str, file_path: str) -> List[ArchitectureIssue]:
        """解析 LLM 响应中的问题"""
        issues = []

        # 尝试提取 JSON
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return issues

        try:
            data = json.loads(json_match.group(0))
            for item in data:
                issue = ArchitectureIssue(
                    file=file_path,
                    line=item.get("line", 1),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    severity=item.get("severity", "warning"),
                    pillar=item.get("pillar", "python"),
                    fix=item.get("fix"),
                    context_code=item.get("context_code"),
                )
                issues.append(issue)
        except json.JSONDecodeError:
            pass

        return issues
