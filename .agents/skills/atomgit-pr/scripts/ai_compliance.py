"""openEuler AI contribution metadata helpers for AtomGit PR workflows."""

from __future__ import annotations

import re

DISCLOSURE_START = "<!-- openEuler-ai-disclosure:start -->"
DISCLOSURE_END = "<!-- openEuler-ai-disclosure:end -->"
PLACEHOLDERS = {"ai", "agent", "model", "unknown", "n/a", "none"}
_HUMAN_COAUTHOR_RE = re.compile(r"<[^<>]+>\s*$")
_AGENT_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/@:+~\- ]*\s+v?\d+(?:\.\d+){1,5}(?:[-+][0-9A-Za-z.-]+)?$")


def _require_value(name: str, value: str) -> str:
    value = value.strip()
    if not value or value.lower() in PLACEHOLDERS:
        raise ValueError(f"{name} must contain the actual value, not a placeholder")
    return value


def validate_agent_tool(value: str) -> str:
    """Validate the tool/version reported by the coding agent.

    The caller must run the tool's real version command before invoking the PR
    workflow. The repository intentionally does not maintain a tool allowlist.
    """
    value = _require_value("Agent platform", value)
    if _AGENT_TOOL_RE.fullmatch(value) is None:
        raise ValueError(
            "Agent platform must contain the actual tool name and version reported by the coding agent "
            "(for example, 'OpenCode 1.17.20'); run '<tool> --version' instead of inventing a value"
        )
    return value


def _require_model(value: str) -> str:
    value = _require_value("AI model", value)
    if "/" in value:
        raise ValueError(
            "AI model must not include a provider prefix; use only the model name and version "
            f"(for example, {value.rsplit('/', 1)[-1]!r})"
        )
    return value


def _require_models(value: str) -> list[str]:
    models = [_require_model(item) for item in re.split(r"[,，;；\n]", value) if item.strip()]
    if not models:
        raise ValueError("AI model must contain at least one model name and version")
    return list(dict.fromkeys(models))


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
    agent_tool = validate_agent_tool(agent_tool)
    ai_models = _require_models(ai_model)
    prompt_summary = _require_value("Prompt summary", prompt_summary)
    third_party_materials = _require_value("Third-party materials", third_party_materials)

    disclosure = f"""{DISCLOSURE_START}
### 当前PR是否有AI参与:
- [ ] 否
- [x] 是

1. Agent平台信息（Tool）: {agent_tool}
2. 模型信息 (Model): {", ".join(ai_models)}
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
    """Require the PR disclosure to cover every AI model used by its commits."""
    disclosed_models = set(_require_models(ai_model))
    commit_models = set()

    for commit in commits:
        message = _commit_message(commit)
        trailers = [line.strip() for line in message.splitlines() if line.strip().startswith("Co-Authored-By:")]
        identifier = commit.get("hash") or commit.get("sha") or "unknown"
        for trailer in trailers:
            model = trailer.partition(":")[2].strip()
            if not _HUMAN_COAUTHOR_RE.search(model):
                try:
                    commit_models.add(_require_model(model))
                except ValueError as exc:
                    raise ValueError(f"invalid AI model metadata in {identifier[:12]}: {exc}") from exc

    problems = []
    if not commit_models:
        problems.append("no AI-assisted commit contains Co-Authored-By model metadata")
    missing = commit_models - disclosed_models
    if missing:
        problems.append(f"PR disclosure does not include commit models: {', '.join(sorted(missing))}")
    if problems:
        raise ValueError("AI commit metadata check failed; " + "; ".join(problems))
