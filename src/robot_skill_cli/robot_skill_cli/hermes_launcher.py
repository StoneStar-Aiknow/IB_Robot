"""Prerequisite-checked launcher for Hermes with the IB-Robot control skill."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from robot_config.config_path import resolve_robot_config_path
from robot_skill_cli.catalog import load_runtime_context
from robot_skill_cli.cli import _create_bridge
from robot_skill_cli.ros_bridge import BridgeError

_MIN_HERMES_VERSION = (0, 16, 0)
_VERSION_PATTERN = re.compile(r"Hermes Agent v(\d+)\.(\d+)\.(\d+)")
_PREFLIGHT_TIMEOUT_FLOOR_SEC = 15.0
_PREFLIGHT_ATTEMPTS = 2
_PREFLIGHT_RETRY_DELAY_SEC = 0.25


class LauncherError(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int = 4) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-robot", description="Launch Hermes with the IB-Robot control skill")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-name")
    config_group.add_argument("--config-path")
    parser.add_argument("hermes_args", nargs=argparse.REMAINDER, help="arguments passed to Hermes after --")
    return parser


def _build_hermes_arguments(hermes_path: str, passthrough: Sequence[str]) -> list[str]:
    arguments = list(passthrough)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if "--skills" in arguments:
        index = arguments.index("--skills")
        if index + 1 >= len(arguments) or "ibrobot-control" not in arguments[index + 1].split(","):
            raise LauncherError("INVALID_HERMES_ARGUMENTS", "--skills must include ibrobot-control", exit_code=2)
    else:
        arguments.extend(["--skills", "ibrobot-control"])
    return [hermes_path, *arguments]


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise LauncherError("PREREQUISITE_MISSING", f"required executable is not installed: {name}")
    return path


def _check_hermes_version(hermes_path: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    try:
        completed = subprocess.run(
            [hermes_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherError("HERMES_VERSION_UNAVAILABLE", "failed to query Hermes version") from exc
    match = _VERSION_PATTERN.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        raise LauncherError("HERMES_VERSION_UNAVAILABLE", "Hermes returned an unsupported version response")
    version = tuple(int(value) for value in match.groups())
    if version < _MIN_HERMES_VERSION:
        minimum = ".".join(str(value) for value in _MIN_HERMES_VERSION)
        raise LauncherError("HERMES_VERSION_UNSUPPORTED", f"Hermes {minimum} or newer is required")


def _installed_skill_path() -> Path:
    skill_path = Path(get_package_share_directory("robot_skill_cli")) / "skills" / "ibrobot-control" / "SKILL.md"
    if not skill_path.is_file():
        raise LauncherError("AGENT_SKILL_NOT_FOUND", "installed ibrobot-control skill is missing")
    return skill_path


def _prepare_hermes_workspace(skill_path: Path) -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    workspace = cache_root / "ibrobot" / "hermes-workspace"
    target_dir = workspace / ".agents" / "skills" / "ibrobot-control"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"
    source_bytes = skill_path.read_bytes()
    if not target.is_file() or target.read_bytes() != source_bytes:
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(source_bytes)
        temporary.replace(target)
    return workspace


def _shell_environment_assignment(name: str, value: str | None) -> str:
    if value is None:
        return f"unset {name}"
    return f"export {name}={shlex.quote(value)}"


def _prepare_robot_skill_wrapper(
    workspace: Path,
    robot_skill_path: str,
    python_path: str,
    config_path: Path,
    runtime_environment: dict[str, str],
) -> Path:
    wrapper_dir = workspace / ".ibrobot" / "bin"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / "robot-skill"
    bound_environment = "\n".join(
        _shell_environment_assignment(name, runtime_environment.get(name))
        for name in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION")
    )
    content = (
        "#!/bin/sh\n"
        'for argument in "$@"; do\n'
        '  case "$argument" in\n'
        "    --config-name|--config-name=*|--config-path|--config-path=*)\n"
        "      printf '%s\\n' 'robot-skill: configuration is bound by hermes-robot' >&2\n"
        "      exit 2\n"
        "      ;;\n"
        "  esac\n"
        "done\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        f"export ROBOT_CONFIG={shlex.quote(str(config_path))}\n"
        f"{bound_environment}\n"
        f'exec {shlex.quote(robot_skill_path)} --config-path "$ROBOT_CONFIG" "$@"\n'
    )
    if not wrapper.is_file() or wrapper.read_text(encoding="utf-8") != content:
        temporary = wrapper.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o755)
        temporary.replace(wrapper)
    return wrapper_dir


def _check_robot_runtime(config_name: str | None, config_path: str | None) -> Path:
    resolved = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    context, transport = load_runtime_context(config_name=config_name, config_path=config_path)
    timeout_sec = max(float(context.view["timeout_policy"]["rpc_timeout_sec"]), _PREFLIGHT_TIMEOUT_FLOOR_SEC)
    last_bridge_error = None
    for attempt in range(_PREFLIGHT_ATTEMPTS):
        bridge = _create_bridge(transport)
        if not bridge.start():
            raise LauncherError("ROS_UNAVAILABLE", "failed to initialize the robot Gateway bridge")
        try:
            status = bridge.get_status(timeout_sec=timeout_sec)
            if not status["control_plane_ready"]:
                raise LauncherError(
                    status["control_plane_error_code"] or "SKILL_REGISTRY_NOT_READY",
                    "robot Gateway control plane is not ready",
                )
            if not bridge.wait_for_agent_plan_interfaces(timeout_sec=timeout_sec):
                raise LauncherError("AGENT_PLAN_UNAVAILABLE", "Agent plan services/action are not discoverable")
            return resolved
        except BridgeError as exc:
            last_bridge_error = exc
        finally:
            bridge.close()
        if attempt + 1 < _PREFLIGHT_ATTEMPTS:
            time.sleep(_PREFLIGHT_RETRY_DELAY_SEC)
    raise last_bridge_error


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        hermes_path = _require_binary("hermes")
        robot_skill_path = _require_binary("robot-skill")
        _check_hermes_version(hermes_path)
        if os.environ.get("ROS_SECURITY_ENABLE", "").lower() == "true":
            os.environ["ROS_SECURITY_ENCLAVE_OVERRIDE"] = "/hermes_cli"
        config_path = _check_robot_runtime(args.config_name, args.config_path)
        workspace = _prepare_hermes_workspace(_installed_skill_path())
        hermes_arguments = _build_hermes_arguments(hermes_path, args.hermes_args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hermes-robot: INVALID_ARGUMENT: {exc}", file=sys.stderr)
        return 2
    except LauncherError as exc:
        print(f"hermes-robot: {exc.code}: {exc}", file=sys.stderr)
        return exc.exit_code
    except BridgeError as exc:
        print(f"hermes-robot: {exc.code}: {exc}", file=sys.stderr)
        return exc.exit_code

    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH", "")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["ROBOT_CONFIG"] = str(config_path)
    wrapper_dir = _prepare_robot_skill_wrapper(workspace, robot_skill_path, python_path, config_path, environment)
    environment["PATH"] = f"{wrapper_dir}{os.pathsep}{environment.get('PATH', '')}"
    os.chdir(workspace)
    os.execvpe(hermes_path, hermes_arguments, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
