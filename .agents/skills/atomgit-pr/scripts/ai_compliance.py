"""openEuler AI contribution metadata helpers for AtomGit PR workflows."""

from __future__ import annotations

import re

DISCLOSURE_START = "<!-- openEuler-ai-disclosure:start -->"
DISCLOSURE_END = "<!-- openEuler-ai-disclosure:end -->"
PLACEHOLDERS = {"ai", "agent", "model", "unknown", "n/a", "none"}


def _require_value(name: str, value: str) -> str:
    value = value.strip()
    if not value or value.lower() in PLACEHOLDERS:
        raise ValueError(f"{name} must contain the actual value, not a placeholder")
    return value


def _require_model(value: str) -> str:
    value = _require_value("AI model", value)
    if "/" in value:
        raise ValueError(
            "AI model must not include a provider prefix; use only the model name and version "
            f"(for example, {value.rsplit('/', 1)[-1]!r})"
        )
    return value


def add_ai_disclosure(
    description: str,
    *,
    agent_tool: str,
    ai_model: str,
    prompt_summary: str,
    third_party_materials: str,
) -> str:
    """Add or replace the machine-identifiable openEuler AI disclosure block."""
    agent_tool = _require_value("Agent platform", agent_tool)
    ai_model = _require_model(ai_model)
    prompt_summary = _require_value("Prompt summary", prompt_summary)
    third_party_materials = _require_value("Third-party materials", third_party_materials)

    disclosure = f"""{DISCLOSURE_START}
### 当前PR是否有AI参与:
- [ ] 否
- [x] 是

1. Agent平台信息（Tool）: {agent_tool}
2. 模型信息 (Model): {ai_model}
3. Prompt摘要 (Prompt Summary): {prompt_summary}
4. 人工审查情况: 开发者已逐项审查 AI 辅助内容，并确认其符合预期
5. 第三方材料及许可证: {third_party_materials}

### 希望检视人员了解:
代码或文档由 AI 辅助开发者完成，最终正确性、安全性、合规性和维护责任由人类贡献者承担。
{DISCLOSURE_END}"""

    if DISCLOSURE_START in description or DISCLOSURE_END in description:
        pattern = re.compile(
            rf"{re.escape(DISCLOSURE_START)}.*?{re.escape(DISCLOSURE_END)}",
            re.DOTALL,
        )
        if pattern.search(description) is None:
            raise ValueError("PR description contains an incomplete openEuler AI disclosure block")
        return pattern.sub(disclosure, description, count=1)

    return f"{disclosure}\n\n{description.strip()}"


def _commit_message(commit: dict) -> str:
    api_message = commit.get("commit", {}).get("message")
    if isinstance(api_message, str):
        return api_message
    message = commit.get("message")
    if isinstance(message, str):
        return message
    return "\n".join(part for part in (commit.get("subject", ""), commit.get("body", "")) if part)


def validate_commit_ai_model(commits: list[dict], ai_model: str) -> None:
    """Require disclosed AI-assisted commits to use the same model as the PR."""
    ai_model = _require_model(ai_model)
    expected = f"Co-Authored-By: {ai_model}"
    disclosed = 0
    mismatched = []

    for commit in commits:
        message = _commit_message(commit)
        trailers = [line.strip() for line in message.splitlines() if line.strip().startswith("Co-Authored-By:")]
        identifier = commit.get("hash") or commit.get("sha") or "unknown"
        if not trailers:
            continue
        disclosed += 1
        if expected not in trailers:
            mismatched.append(f"{identifier[:12]} ({', '.join(trailers)})")

    problems = []
    if not disclosed:
        problems.append(f"no AI-assisted commit contains {expected!r}")
    if mismatched:
        problems.append(f"model does not match PR disclosure: {'; '.join(mismatched)}")
    if problems:
        raise ValueError("AI commit metadata check failed; " + "; ".join(problems))
