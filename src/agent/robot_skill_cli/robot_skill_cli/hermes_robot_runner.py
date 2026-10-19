"""Hermes-side owner for one foreground robot workflow (standard library only)."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

try:
    from .agent_terminal_contract import classify_agent_terminal
except ImportError:
    from embodied_common.agent_terminal_contract import classify_agent_terminal
from .hermes_stop_state import StopStateError, StopStore

MOTION_COMMANDS = {"run-workflow", "execute-plan", "confirm-plan"}


def workflow_arguments(command: str, executable: str, environment: dict | None = None) -> list[str] | None:
    """Accept one argv invocation; never evaluate shell syntax for motion."""
    words = shlex.split(command)
    candidates = [i for i, word in enumerate(words) if Path(word).name == "robot-skill"]
    if not candidates:
        return None
    index = candidates[0]
    args = words[index + 1 :]
    if not any(arg in MOTION_COMMANDS for arg in args):
        return None
    # Older sessions sometimes prepend these two already-bound transport
    # assignments. Strip only exact inherited values; no shell expressions.
    for assignment in words[:index]:
        name, separator, value = assignment.partition("=")
        if separator != "=" or name not in {"RMW_IMPLEMENTATION", "CYCLONEDDS_URI"}:
            raise StopStateError("managed robot motion requires a single robot-skill command")
        if value != (environment if environment is not None else os.environ).get(name):
            raise StopStateError("robot transport is bound by hermes-robot-configure")
    if words[index] not in {"robot-skill", executable} or len(candidates) != 1:
        raise StopStateError("robot executable is bound by hermes-robot-configure")
    if not args or args[0] != "run-workflow":
        raise StopStateError("Hermes motion must use run-workflow")
    if args == ["run-workflow", "--help"]:
        return None
    allowed = {"--text", "--workflow-json", "--request-id"}
    seen = set()
    for offset in range(1, len(args), 2):
        flag = args[offset]
        if flag not in allowed or flag in seen or offset + 1 >= len(args):
            raise StopStateError(
                "invalid managed workflow arguments; use run-workflow --text TEXT --workflow-json JSON [--request-id ID]"
            )
        seen.add(flag)
    if not {"--text", "--workflow-json"} <= seen:
        raise StopStateError("workflow text and typed steps are required")
    return args


def terminal_state(presentation: dict | None, terminal: dict | None) -> str:
    if not isinstance(presentation, dict) or not isinstance(terminal, dict):
        return "unknown"
    if not presentation.get("task_id") or terminal.get("task_id") != presentation["task_id"]:
        return "unknown"
    result = terminal.get("result", {})
    if not isinstance(result, dict) or type(result.get("success")) is not bool:
        return "unknown"
    if terminal.get("stopped_before_execution") is True and result.get("success") is False:
        return "not_submitted" if result.get("error_code") == "SKILL_CANCELLED" else "unknown"
    identity = presentation.get("registry_identity", {})
    if not isinstance(identity, dict) or not isinstance(presentation.get("steps"), list):
        return "unknown"
    if not all(presentation.get(key) for key in ("plan_id", "plan_digest")) or not all(
        identity.get(key) for key in ("registry_epoch", "registry_generation", "registry_digest")
    ):
        return "unknown"
    expectation = {
        "plan_id": presentation.get("plan_id"),
        "plan_digest": presentation.get("plan_digest"),
        "registry_epoch": identity.get("registry_epoch"),
        "registry_generation": identity.get("registry_generation"),
        "registry_digest": identity.get("registry_digest"),
        "step_count": len(presentation.get("steps", [])),
    }
    return classify_agent_terminal(terminal.get("goal_status"), result, expectation) or "unknown"


class RobotRunner:
    def __init__(
        self, store: StopStore, executable: str, *, cancel_timeout: float = 75.0, environment: dict | None = None
    ):
        self.store = store
        self.executable = executable
        self.cancel_timeout = cancel_timeout
        self.environment = dict(environment or {})

    def run(
        self,
        args: list[str],
        session: str,
        turn: str,
        *,
        interrupted=lambda: False,
        timeout=600.0,
        presentation_callback=None,
        terminal_callback=None,
    ) -> str:
        token = uuid.uuid4().hex
        with self.store.owner_lock(token) as owner_fd:
            return self._run(
                args,
                session,
                turn,
                token,
                owner_fd,
                interrupted=interrupted,
                timeout=timeout,
                presentation_callback=presentation_callback,
                terminal_callback=terminal_callback,
            )

    def _run(
        self, args, session, turn, token, owner_fd, *, interrupted, timeout, presentation_callback, terminal_callback
    ):
        self.store.claim(session, turn, token)
        environment = os.environ.copy()
        environment.update(self.environment)
        environment.update(
            IBROBOT_STOP_ROOT=str(self.store.root),
            IBROBOT_STOP_SESSION=session,
            IBROBOT_STOP_TOKEN=token,
            IBROBOT_STOP_PRESENTATION_ACK="1" if presentation_callback is not None else "0",
        )
        output = []
        lines = queue.Queue()
        presentation = None
        terminal = None
        terminals = 0
        process = None
        state = "unknown"
        started = time.monotonic()
        stop_deadline = None
        try:
            if self.store.stopped(session, token):
                state = "not_submitted"
                return json.dumps({"error": "STOP_LATCHED", "message": "stopped before process submission"})
            try:
                process = subprocess.Popen(
                    [self.executable, *args],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                    pass_fds=(owner_fd,),
                )
            except OSError:
                state = "not_submitted"
                raise
            self.store.update(session, token, state="running")

            def drain():
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            eof = False
            while not eof or process.poll() is None:
                if interrupted() or time.monotonic() - started > float(timeout):
                    self.store.user_input(session, stop=True)
                if self.store.stopped(session, token) and stop_deadline is None:
                    stop_deadline = time.monotonic() + self.cancel_timeout
                if stop_deadline is not None and time.monotonic() >= stop_deadline:
                    # Client termination proves nothing about the action server.
                    # Keep the ledger unknown and prohibit another motion.
                    process.kill()
                    process.wait(timeout=2.0)
                    break
                try:
                    line = lines.get(timeout=0.02)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                    continue
                output.append(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("command") != "run-workflow":
                    continue
                if event.get("event") == "workflow_presentation":
                    if presentation is not None:
                        self.store.user_input(session, stop=True)
                        raise StopStateError("duplicate workflow presentation")
                    presentation = event["data"]
                    self.store.update(session, token, presentation=presentation)
                    if presentation_callback is not None:
                        try:
                            presentation_callback(json.dumps(presentation, ensure_ascii=False))
                        except Exception:
                            self.store.update(session, token, presentation_failed=True)
                            raise
                        self.store.update(session, token, presentation_flushed=True)
                elif event.get("event") == "workflow_terminal":
                    terminals += 1
                    terminal = event["data"]
            reader.join(timeout=1.0)
            admission_error = None
            if terminals == 0:
                saved = self.store.snapshot(session)["owner"]
                if saved and saved["token"] == token:
                    presentation = saved.get("presentation")
                    terminal = saved.get("terminal")
                    if terminal is not None:
                        terminals = 1
                    elif (
                        saved["state"] == "not_submitted"
                        and saved.get("admission_error")
                        and (not presentation or saved.get("fresh_submission_pending") is True)
                    ):
                        state = "not_submitted"
                        admission_error = saved["admission_error"]["code"]
            if terminals == 1:
                state = terminal_state(presentation, terminal)
            error = admission_error if state != "unknown" else "ROBOT_STOP_UNKNOWN: no verified workflow terminal"
            if error is None and terminal_callback is not None:
                terminal_callback(presentation, terminal)
            return json.dumps(
                {"output": "".join(output), "exit_code": process.returncode, "error": error}, ensure_ascii=False
            )
        finally:
            if process is not None and process.poll() is None:
                # Unexpected owner failures still request cooperative stop.
                self.store.user_input(session, stop=True)
                try:
                    process.wait(timeout=self.cancel_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            self.store.update(session, token, state=state, terminal=terminal, finished_at=time.time())

    def recover(self, session: str) -> None:
        """Only a dead owner permits external cancellation of its exact plan."""
        owner = self.store.snapshot(session)["owner"]
        if not owner or self.store.owner_alive(owner["token"]):
            return
        token = owner["token"]
        try:
            with self.store.owner_lock(token):
                presentation = owner.get("presentation")
                if not presentation:
                    self.store.update(session, token, state="unknown")
                    return
                identity = presentation["registry_identity"]
                arguments = [
                    self.executable,
                    "cancel-plan",
                    "--task-id",
                    presentation["task_id"],
                    "--plan-id",
                    presentation["plan_id"],
                    "--plan-digest",
                    presentation["plan_digest"],
                    "--registry-epoch",
                    identity["registry_epoch"],
                    "--registry-generation",
                    str(identity["registry_generation"]),
                    "--registry-digest",
                    identity["registry_digest"],
                    "--expected-step-count",
                    str(len(presentation["steps"])),
                ]
                completed = subprocess.run(
                    arguments,
                    capture_output=True,
                    text=True,
                    timeout=self.cancel_timeout,
                    check=False,
                    env={**os.environ, **self.environment},
                )
                response = json.loads(completed.stdout)["data"]
                terminal = {
                    "task_id": presentation["task_id"],
                    "goal_status": response.get("goal_status"),
                    "result": response.get("result", {}),
                }
                self.store.update(session, token, state=terminal_state(presentation, terminal), terminal=terminal)
        except BlockingIOError:
            return
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            self.store.update(session, token, state="unknown")
