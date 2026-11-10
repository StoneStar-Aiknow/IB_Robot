"""Presentation only: preserve CLI JSON contracts and never infer motion safety."""

import os
import sys
import textwrap
import unicodedata


def use_human_output(command, requested):
    if command not in {"status", "list-skills", "discover", "connect"}:
        return False
    if requested is not None:
        return requested == "human"
    if os.environ.get("IBROBOT_HERMES_MANAGED_REQUIRED") == "1":
        return False
    return sys.stdout.isatty()


def _text(value):
    if value is None or value == "":
        return "—"
    # Do not let vendor text inject terminal escapes or extra table rows.
    return "".join(" " if unicodedata.category(c).startswith("C") else c for c in str(value))


def _width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _table(headers, rows):
    rows = [[_text(cell) for cell in row] for row in rows]
    if not rows:
        return "（无条目）"
    # Keep identifiers intact for copying; wrap reasons/descriptions, never drop them.
    widths = [max(_width(row[i]) for row in [headers, *rows]) for i in range(len(headers) - 1)]
    lines = [
        "  ".join(
            text + " " * (widths[i] - _width(text)) if i < len(widths) else text for i, text in enumerate(headers)
        )
    ]
    for row in rows:
        tail = textwrap.wrap(row[-1], width=72, break_long_words=False, break_on_hyphens=False) or ["—"]
        prefix = "  ".join(row[i] + " " * (widths[i] - _width(row[i])) for i in range(len(widths))) + "  "
        lines.append(prefix + tail[0])
        lines.extend(" " * _width(prefix) + part for part in tail[1:])
    return "\n".join(lines)


def _bool(value):
    return "是" if value is True else "否" if value is False else "未知"


def render(envelope):
    command = envelope.get("command", "unknown")
    if not envelope.get("ok"):
        error = envelope.get("error") or {}
        return f"{command} 失败 [{_text(error.get('code'))}]\n{_text(error.get('message'))}"
    data = envelope.get("data") or {}
    lines = [f"机器人：{_text(data.get('robot_name'))}"]
    if command == "status":
        capabilities = data.get("capabilities") or []
        lines.extend(
            [
                f"控制面：{_text(data.get('control_plane_state'))}  就绪：{_bool(data.get('control_plane_ready'))}",
                f"动作授权：{_bool(data.get('motion_authorized'))}  忙碌：{_bool(data.get('busy'))}",
                f"控制模式：{_text(data.get('active_control_mode'))}  当前任务：{_text(data.get('active_task_id'))}",
                f"技能：{len(capabilities)} 项，{sum(c.get('ready') is True for c in capabilities)} 项报告就绪",
            ]
        )
        for key in ("control_plane_error_code", "request_error_code", "request_state"):
            if data.get(key):
                lines.append(f"{key}：{_text(data[key])}")
        lines.extend(
            [
                "",
                _table(
                    ["技能", "就绪", "原因"],
                    [[c.get("name"), _bool(c.get("ready")), c.get("reason")] for c in capabilities],
                ),
                "",
                "就绪是 Gateway 准入状态，不代表真机验收或全部厂商条件已确认。",
            ]
        )
    else:
        skills = data.get("skills") or []
        lines.extend(
            [
                f"技能：{len(skills)} 项",
                "",
                _table(
                    ["技能", "控制模式", "说明"],
                    [[s.get("name"), s.get("required_control_mode"), s.get("summary")] for s in skills],
                ),
                "",
                "目录存在不代表动作可执行；执行前查看 status / describe / validate。",
            ]
        )
    lines.append("完整字段与标识：在子命令后加 --json。")
    return "\n".join(lines)
