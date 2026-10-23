"""Install a checked, reversible native-hook adapter into the Hermes runtime."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ADAPTER_VERSION = 3
SUPPORTED_REVISION = "cf64ca20c5ab99ebf7e8ca272c69edc7ea0636ed"


def _base_replacements() -> dict[str, list[tuple[str, str]]]:
    return {
        "hermes_cli/plugins.py": [
            (
                "VALID_HOOKS: Set[str] = {\n",
                "VALID_HOOKS: Set[str] = {\n"
                '    "on_control_input", "on_control_turn_start", "on_control_turn_end", "execute_managed_tool",\n',
            )
        ],
        "cli.py": [
            (
                "            if text or has_images:\n                # Handle /model directly on the UI thread",
                "            if text or has_images:\n"
                "                from hermes_cli.robot_control import control_input\n"
                "                control = None if has_images else control_input(\n"
                '                    text, getattr(self.agent, "session_id", None) or self.session_id,\n'
                "                    busy=self._agent_running, agent=self.agent,\n"
                "                )\n"
                "                if control is not None:\n"
                '                    _cprint(control.get("message", ""))\n'
                "                    event.app.current_buffer.reset(append_to_history=True)\n"
                "                    event.app.invalidate()\n"
                "                    return\n"
                "                # Handle /model directly on the UI thread",
            )
        ],
        "run_agent.py": [
            (
                "        try:\n            # Serialize the full load -> run -> flush region across Hermes",
                "        from hermes_cli.robot_control import begin_turn as _control_begin_turn\n"
                "        _control_begin_turn(self, session_id, relay_turn_id)\n"
                "        try:\n            # Serialize the full load -> run -> flush region across Hermes",
            ),
            (
                "        finally:\n            try:\n                if relay_turn is not None:\n",
                "        finally:\n"
                "            from hermes_cli.robot_control import end_turn as _control_end_turn\n"
                "            _control_end_turn(self)\n"
                "            try:\n                if relay_turn is not None:\n",
            ),
        ],
        "agent/tool_executor.py": [
            (
                "        try:\n            return execute(final_args)\n        finally:\n            _hb_stop.set()",
                "        try:\n"
                "            from hermes_cli.robot_control import execute_managed\n"
                "            managed = execute_managed(agent, function_name, final_args)\n"
                "            if managed is not None:\n"
                '                return managed["result"]\n'
                "            return execute(final_args)\n        finally:\n            _hb_stop.set()",
            )
        ],
        "gateway/run.py": [
            (
                "        effective_mode = self._effective_busy_input_mode(event.source)\n\n"
                "        # --- Draining case (gateway restarting/stopping) ---",
                "        from hermes_cli.robot_control import gateway_input\n"
                "        control = await gateway_input(self, event, session_key, busy=True)\n"
                "        if control is not None:\n"
                "            adapter = self._adapter_for_source(event.source)\n"
                "            if adapter:\n"
                "                anchor = self._reply_anchor_for_event(event)\n"
                "                await adapter._send_with_retry(\n"
                '                    chat_id=event.source.chat_id, content=control.get("message", ""),\n'
                "                    reply_to=anchor, metadata=self._thread_metadata_for_source(event.source, anchor),\n"
                "                )\n"
                "            return True\n"
                "        effective_mode = self._effective_busy_input_mode(event.source)\n\n"
                "        # --- Draining case (gateway restarting/stopping) ---",
            ),
            (
                "        # Global emergency stop (`hermes pause`): give new turns a brief",
                "        from hermes_cli.robot_control import gateway_input\n"
                "        control = await gateway_input(\n"
                "            self, event, self._session_key_for_source(source), busy=False,\n"
                "        )\n"
                "        if control is not None:\n"
                '            return control.get("message", "")\n\n'
                "        # Global emergency stop (`hermes pause`): give new turns a brief",
            ),
        ],
    }


def _upgrade_replacements() -> dict[str, list[tuple[str, str]]]:
    """Tool-batch hooks included in every fresh adapter installation."""
    return {
        "hermes_cli/plugins.py": [
            (
                '    "on_control_input", "on_control_turn_start", "on_control_turn_end", "execute_managed_tool",\n',
                '    "on_control_input", "on_control_turn_start", "on_control_turn_end", "execute_managed_tool",\n'
                '    "finalize_tool_batch",\n',
            )
        ],
        "agent/tool_executor.py": [
            (
                "managed = execute_managed(agent, function_name, final_args)",
                "managed = execute_managed(agent, function_name, final_args, tool_call_id=tool_call_id)",
            )
        ],
        "agent/conversation_loop.py": [
            (
                "                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)",
                "                from hermes_cli.robot_control import begin_tool_batch, finish_tool_batch\n"
                "                begin_tool_batch(agent, assistant_message)\n"
                "                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)",
            ),
            (
                "                # Reset per-turn retry counters after successful tool\n",
                "                _direct_response = finish_tool_batch(agent, messages)\n"
                "                if _direct_response is not None:\n"
                '                    _turn_exit_reason = "managed_tool_response"\n'
                "                    final_response = _direct_response\n"
                "                    agent._mute_post_response = False\n"
                '                    append_message(messages, {"role": "assistant", "content": final_response})\n'
                "                    break\n\n"
                "                # Reset per-turn retry counters after successful tool\n",
            ),
        ],
    }


def _feishu_replacements():
    return {
        "hermes_cli/plugins.py": [
            (
                '    "on_control_input", "on_control_turn_start", "on_control_turn_end", "execute_managed_tool",\n'
                '    "finalize_tool_batch",\n',
                '    "on_control_input", "on_control_turn_start", "on_control_turn_end", "execute_managed_tool",\n'
                '    "finalize_tool_batch",\n'
                '    "is_priority_control_text",\n',
            )
        ],
        "plugins/platforms/feishu/adapter.py": [
            (
                '        """Apply Feishu-specific burst protection before entering the base adapter."""\n',
                '        """Apply Feishu-specific burst protection before entering the base adapter."""\n'
                "        from hermes_cli.robot_control import priority_control_text\n"
                "        if priority_control_text(event):\n"
                "            await self._handle_message_with_guards(event)\n"
                "            return\n",
            )
        ],
    }


def sync_runtime(root: Path, helper: Path, *, dry_run: bool, write, backup) -> bool:
    """Prevalidate every file before any write; refuse unsupported or edited adapters."""
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if revision.returncode or revision.stdout.strip() != SUPPORTED_REVISION:
        raise ValueError(f"unsupported Hermes runtime revision; robot control adapter requires {SUPPORTED_REVISION}")
    marker = root / ".ibrobot-control-adapter.json"
    manifest = json.loads(marker.read_text()) if marker.is_file() else {}
    version = manifest.get("version", 0)
    if manifest:
        if version != ADAPTER_VERSION or manifest.get("revision") != SUPPORTED_REVISION:
            raise ValueError("unsupported installed Hermes adapter version")
        expected_files = {
            *_base_replacements(),
            *_upgrade_replacements(),
            *_feishu_replacements(),
            "hermes_cli/robot_control.py",
        }
        if manifest.get("extensions"):
            raise ValueError("unsupported Hermes adapter extensions")
        if set(manifest.get("files", {})) != expected_files:
            raise ValueError("incomplete installed Hermes adapter manifest")
        for relative, expected in manifest["files"].items():
            path = root / relative
            if not path.is_file():
                raise ValueError(f"Hermes adapter file changed since installation: {relative}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Hermes adapter file changed since installation: {relative}")
    replacements = _base_replacements() if not manifest else {name: [] for name in _base_replacements()}
    for group in (_upgrade_replacements(), _feishu_replacements()):
        for relative, operations in group.items():
            replacements.setdefault(relative, []).extend([] if manifest else operations)
    desired = {}
    for relative, operations in replacements.items():
        path = root / relative
        current = path.read_text()
        text = current
        for old, new in operations:
            if text.count(old) != 1:
                raise ValueError(f"Hermes adapter anchor mismatch: {relative}")
            text = text.replace(old, new, 1)
        ast.parse(text, filename=str(path))
        desired[path] = text
    helper_target = root / "hermes_cli/robot_control.py"
    if helper_target.exists() and not manifest:
        raise ValueError("unmanaged Hermes robot_control.py already exists")
    desired[helper_target] = helper.read_text()
    ast.parse(desired[helper_target], filename=str(helper_target))
    desired_manifest = {
        "version": ADAPTER_VERSION,
        "revision": SUPPORTED_REVISION,
        "files": {
            str(path.relative_to(root)): hashlib.sha256(text.encode()).hexdigest() for path, text in desired.items()
        },
    }
    desired[marker] = json.dumps(desired_manifest, indent=2) + "\n"
    changed = any(not path.is_file() or path.read_text() != text for path, text in desired.items())
    if changed and not dry_run:
        for path, text in desired.items():
            if not path.is_file() or path.read_text() != text:
                backup(path)
                write(path, text)
    return changed
