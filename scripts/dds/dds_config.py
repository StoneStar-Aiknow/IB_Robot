"""Public set/unset/show CLI; shell-changing commands use the Bash function."""

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

if not __package__:
    spec = importlib.util.spec_from_file_location("ibrobot_dds_cli", Path(__file__).with_name("__init__.py"))
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    __package__ = spec.name

configuration = importlib.import_module(".configuration", __package__)
core = importlib.import_module(".core", __package__)
show_output = importlib.import_module(".show_output", __package__)


def command_parser(command):
    """One argument definition shared by help and the shell bridge."""
    parser = argparse.ArgumentParser(prog=f"dds-config {command}")
    if command == "set":
        parser.add_argument("--config", type=Path, help="读取 YAML；省略时进入问答向导")
    if command in ("set", "unset"):
        parser.add_argument("--tmp", action="store_true", help="仅影响当前终端，不修改用户默认配置")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def show(home):
    report = {
        "disk_selection": None,
        "inherited": {
            key: os.environ.get(key)
            for key in (
                "RMW_IMPLEMENTATION",
                "IBROBOT_DDS_STATUS",
                "ROS_DOMAIN_ID",
                "ROS_LOCALHOST_ONLY",
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                "CYCLONEDDS_URI",
            )
        },
    }
    if "IBROBOT_DDS_STATE" in os.environ:
        current = importlib.import_module(".current_state", __package__)
        report["current"] = current.inspect_current(home, os.environ)
    fixed = Path(home) / ".config/ibrobot/dds/config.yaml"
    if fixed.exists() or fixed.is_symlink():
        try:
            _, config = configuration.read_configuration(fixed)
            report.update(disk_selection=config["backend"], default_configuration=config)
            status = configuration.generation_status(home, configuration.plan(fixed))
            report["default_binding"] = status["binding"]
            if not status["prepared"]:
                raise configuration.ConfigurationError("默认配置缺失或过期；请重新 set 默认配置。", 5)
        except configuration.ConfigurationError as exc:
            report["default_diagnostic"] = {"code": exc.code, "message": str(exc)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="DDS 配置：set 设置，unset 恢复原环境，show 查看。")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("set", "设置并立即生效"),
        ("unset", "恢复原环境，默认移除用户配置"),
        ("show", "查看当前终端和用户默认配置"),
    ):
        commands.add_parser(name, parents=[command_parser(name)], add_help=False, help=help_text)
    args = parser.parse_args(argv)
    try:
        home = os.environ.get("HOME", "")
        core.state_root(home)
        if args.command != "show":
            raise configuration.ConfigurationError(
                f"dds-config {args.command} 目前仅支持 Bash；请在 Bash 中先 source .shrc_local，"
                "再通过当前终端的函数执行。",
                3,
            )
        report = show(home)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            show_output.print_show(report)
        return 0
    except (configuration.ConfigurationError, core.ConfigError, OSError, ValueError) as exc:
        code = getattr(exc, "code", 6)
        if args.json:
            print(json.dumps({"error": str(exc), "code": code}), file=sys.stderr)
        else:
            print(f"DDS：{exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
