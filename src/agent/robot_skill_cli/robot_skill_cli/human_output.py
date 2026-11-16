"""Presentation only: preserve CLI JSON contracts and never infer motion safety."""

import os
import re
import sys
import textwrap
import unicodedata

# A vendor action number arrives as an optional catalog display tag, for
# example ``unitree_action_id_99``. This adapter only formats it; the tag
# never selects a firmware action, and the machine-readable JSON keeps the
# raw tag so a caller can trace the number to the catalog/profile.
_VENDOR_ACTION_TAG = re.compile(r"^(?P<vendor>[a-z][a-z0-9]*)_action_id_(?P<action_id>[0-9]{1,6})$")
_TAG_LABELS = {"preset": "预置", "control_ownership": "控制权操作"}
_VENDOR_LABELS = {"unitree": "宇树"}


def _vendor_action(skill):
    """Return ``(vendor label, action id)`` from the display tags, or ``None``."""
    for tag in skill.get("tags") or []:
        match = _VENDOR_ACTION_TAG.match(str(tag))
        if match is not None:
            vendor = _VENDOR_LABELS.get(match.group("vendor"), match.group("vendor"))
            return vendor, int(match.group("action_id"))
    return None


def _badges(skill):
    labels = []
    for tag in skill.get("tags") or []:
        label = _TAG_LABELS.get(str(tag))
        if label is not None:
            labels.append(label)
    return labels


def _title(skill):
    display_name = skill.get("display_name")
    name = skill.get("name")
    if display_name and display_name != name:
        return f"{display_name}（{name}）"
    return name


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
    prefix_width = sum(widths) + 2 * len(widths)
    # Keep one row readable on an 80-120 column terminal: a wide identifier
    # column must not push the wrapped reason off the right edge.
    tail_width = max(36, 116 - prefix_width)
    for row in rows:
        tail = textwrap.wrap(row[-1], width=tail_width, break_long_words=False, break_on_hyphens=False) or ["—"]
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
        rows = []
        for skill in skills:
            action = _vendor_action(skill)
            rows.append(
                [
                    _title(skill),
                    f"ID{action[1]}" if action else None,
                    "、".join(_badges(skill)) or None,
                    skill.get("required_control_mode"),
                    skill.get("summary"),
                ]
            )
        lines.extend(
            [
                f"技能：{len(skills)} 项",
                "",
                _table(["技能", "ID", "标记", "控制模式", "说明"], rows),
                "",
                "目录存在不代表动作可执行；执行前查看 status / describe / validate。",
            ]
        )
        if any(_vendor_action(skill) for skill in skills):
            vendors = sorted({_vendor_action(skill)[0] for skill in skills if _vendor_action(skill)})
            lines.extend(
                [
                    f"ID 列是{('、'.join(vendors))}厂商动作编号，来源是技能目录的展示标签"
                    "（可追溯到运行时 profile 的 vendor.arm_actions）。",
                    "它不是列表序号、请求 ID 或工作流步骤，也不提供执行入口。",
                ]
            )
        if any("control_ownership" in (skill.get("tags") or []) for skill in skills):
            lines.append("控制权操作是展示分类：不授予执行权限，不会自动执行、恢复或收尾。")
    lines.append("完整字段与标识：在子命令后加 --json。")
    return "\n".join(lines)
