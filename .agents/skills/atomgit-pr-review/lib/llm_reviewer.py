"""
LLM Code Reviewer
使用 LLM 进行代码审查
"""

import os
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CodeIssue:
    """代码问题"""

    file: str
    line: int
    type: str
    severity: str
    confidence: int
    title: str
    description: str
    context_code: Optional[str] = None
    fix_code: Optional[str] = None
    fix_explanation: Optional[str] = None


class LLMCodeReviewer:
    """LLM 代码审查器"""

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
    ) -> List[CodeIssue]:
        """审查单个文件"""
        prompt = self._build_review_prompt(file_path, file_content, diff)
        response = self._call_llm(prompt)
        return self._parse_issues(response, file_path)

    def _build_review_prompt(self, file_path: str, file_content: str, diff: str) -> str:
        """构建审查提示词"""
        return f"""你是一个专业的代码审查专家。请审查以下代码变更。

**文件**: {file_path}

**代码变更 (Diff)**:
```
{diff or "(无法获取diff)"}
```

**完整文件内容**:
```
{file_content}
```

请检查以下方面：
1. **代码质量**: 是否有潜在的bug、逻辑错误、边界条件
2. **最佳实践**: 是否符合Python最佳实践和项目规范
3. **性能**: 是否有性能问题
4. **安全性**: 是否有安全漏洞
5. **可维护性**: 代码是否清晰易懂

请按照以下JSON格式输出发现的问题（如果没有问题，输出空数组）：

```json
[
  {{
    "line": 10,
    "type": "bug|security|performance|maintainability|best_practice",
    "severity": "error|warning|suggestion|info",
    "confidence": 85,
    "title": "简短的问题标题",
    "description": "详细的问题描述",
    "context_code": "相关代码片段",
    "fix_code": "修复建议代码（可选）",
    "fix_explanation": "修复说明（可选）"
  }}
]
```

**重要**：
- 只输出JSON数组，不要包含其他文字
- 如果没有问题，输出 `[]`
- line 应该是问题所在的实际行号
- confidence 范围是 0-100"""

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

    def _parse_issues(self, response: str, file_path: str) -> List[CodeIssue]:
        """解析 LLM 响应中的问题"""
        issues = []

        # 尝试提取 JSON
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return issues

        try:
            data = json.loads(json_match.group(0))
            for item in data:
                issue = CodeIssue(
                    file=file_path,
                    line=item.get("line", 1),
                    type=item.get("type", "bug"),
                    severity=item.get("severity", "warning"),
                    confidence=item.get("confidence", 80),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    context_code=item.get("context_code"),
                    fix_code=item.get("fix_code"),
                    fix_explanation=item.get("fix_explanation"),
                )
                issues.append(issue)
        except json.JSONDecodeError:
            pass

        return issues
