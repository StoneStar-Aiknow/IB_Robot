"""Private data-only bridge for Bash activation. Never emits shell code."""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from . import configuration, core, dds_config, environment_plan, wizard

KEYS = sorted(environment_plan.MANAGED) + ["IBROBOT_DDS_STATE", "IBROBOT_DDS_STATUS"]


def check_owned(values, home, *, reapply=False):
    """Validate inherited last-applied values before considering any new target."""
    raw = values.get("IBROBOT_DDS_STATE")
    if raw is None:
        if "IBROBOT_DDS_STATUS" in values:
            raise configuration.ConfigurationError("继承 DDS 状态但无接管记录；请使用干净终端。", 4)
        return False
    try:
        ledger = json.loads(raw)
        if not isinstance(ledger, dict):
            raise ValueError("invalid ledger object")
        expected = ledger["values"]
        if (
            type(ledger["version"]) is not int
            or ledger["version"] != 1
            or ledger["home"] != home
            or not isinstance(expected, dict)
            or set(expected) != environment_plan.MANAGED
        ):
            raise ValueError("unknown ledger or HOME mismatch")
        if any(value is not None and not isinstance(value, str) for value in expected.values()):
            raise ValueError("invalid recorded value")
        if not reapply:
            for name, value in expected.items():
                if (name in values) != (value is not None) or (value is not None and values[name] != value):
                    raise ValueError(f"modified {name}")
    except (ValueError, KeyError, TypeError) as exc:
        raise configuration.ConfigurationError(f"DDS 接管环境冲突，未修改变量：{exc}", 4) from exc
    return True


def encode_snapshot(snapshot):
    return {
        **snapshot,
        "content": None if snapshot["content"] is None else base64.b64encode(snapshot["content"]).decode(),
    }


def decode_snapshot(snapshot):
    return {**snapshot, "content": None if snapshot["content"] is None else base64.b64decode(snapshot["content"])}


def prepare(directory, mode, arguments, values):
    home = os.environ.get("HOME", "")
    core.state_root(home)
    if mode == "unset":
        args = dds_config.command_parser("unset").parse_args(arguments)
        fixed = Path(home) / ".config/ibrobot/dds/config.yaml"
        target = None if args.tmp else core.snapshot(fixed)
        if target is not None and fixed.is_symlink():
            raise configuration.ConfigurationError(
                "默认配置是符号链接；请先处理该链接，或用 unset --tmp 仅退出当前终端。", 6
            )
        return {
            "action": "unset",
            "mode": mode,
            "home": home,
            "temporary": args.tmp,
            "json": args.json,
            "target_snapshot": encode_snapshot(target) if target else None,
            "restored_current_shell": (directory / "restore-owned").exists(),
        }
    # Explicit set/source replace the allowlisted DDS variables, including
    # accidental manual overrides. Read-only show/checks remain strict. Target
    # validation and generation still complete before changing the parent shell.
    owned = check_owned(values, home, reapply=True)
    fixed = Path(home) / ".config/ibrobot/dds/config.yaml"
    args = dds_config.command_parser("set").parse_args(arguments)
    target = core.snapshot(fixed) if mode == "source" or not args.tmp else None
    if mode == "source" and target is not None and target["content"] is None:
        return {"action": "restore", "json": False}
    source = fixed if mode == "source" else args.config
    if mode == "set" and source is None:
        previous = None
        if fixed.exists() or fixed.is_symlink():
            _, previous = configuration.read_configuration(fixed)

        def interfaces():
            return json.loads(
                subprocess.run(
                    ["ip", "-j", "-4", "addr", "show"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout
            )

        try:
            # stdin holds shell snapshot records; prompts belong to the
            # controlling terminal, never to captured stdout or plan data.
            with open("/dev/tty", encoding="utf-8") as reader, open("/dev/tty", "w", encoding="utf-8") as writer:
                selected = wizard.configure(previous, interfaces, reader, writer, args.tmp)
        except OSError as exc:
            raise configuration.ConfigurationError("无交互终端；请使用 set --config 文件 [--tmp]。", 2) from exc
        source = directory / "input.yaml"
        source.write_bytes(core.json_bytes(selected))
    plan = configuration.plan(source)
    if mode == "set":
        configuration.prepare(home, plan)
    status = configuration.generation_status(home, plan)
    if mode == "source" and not status["prepared"]:
        # Saved defaults may outlive a generator, network address or package
        # update. Rebuild only after plan() has rechecked the current facts;
        # source never installs system packages or overwrites old generations.
        status = configuration.prepare(home, plan)
    if not status["prepared"]:
        raise configuration.ConfigurationError("默认配置产物缺失或过期；请 dds-config set。", 5)
    inherited = {key: value for key, value in values.items() if key not in KEYS} if owned else values
    preview = configuration.preview_environment(home, plan, inherited)
    configuration.check_environment(preview)
    applied = {key: preview["set"].get(key) for key in environment_plan.MANAGED}
    record = {
        "version": 1,
        "home": home,
        "backend": plan["config"]["backend"],
        "generation": status["generation"],
        "values": dict(applied),
        "temporary": mode == "set" and args.tmp,
        "configuration": plan["config"],
    }
    if plan["discovery_domain"] is not None:
        record["discovery_domain"] = plan["discovery_domain"]
    applied["IBROBOT_DDS_STATE"] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    applied["IBROBOT_DDS_STATUS"] = "ready"
    # NUL-delimited records: no eval, source, command substitution or shell grammar.
    with (directory / "values").open("wb") as stream:
        for key in KEYS:
            value = applied[key]
            for field in (key, "0" if value is None else "1", value or ""):
                stream.write(field.encode() + b"\0")
    return {
        "action": "apply",
        "home": home,
        "source": str(source),
        "source_snapshot": encode_snapshot(plan["source"]),
        "target_snapshot": encode_snapshot(target) if target else None,
        "identity": plan["identity"],
        "generation": status["generation"],
        "save": mode == "set" and not args.tmp,
        "temporary": mode == "set" and args.tmp,
        "json": args.json,
        "backend": plan["config"]["backend"],
        "mode": mode,
        "discovery": plan["config"]["network"]["discovery"],
        "interactive": mode == "set" and args.config is None,
    }


def commit(transaction):
    if transaction["action"] == "restore":
        return
    if os.environ.get("HOME") != transaction["home"]:
        raise configuration.ConfigurationError("HOME 已变化", 4)
    if transaction["action"] == "unset":
        removed = False
        if transaction["target_snapshot"] is not None:
            target = decode_snapshot(transaction["target_snapshot"])
            removed = core.publish_configuration(target, target, None, [], remove=True)
        return removed
    plan = configuration.plan(transaction["source"])
    if plan["identity"] != transaction["identity"] or plan["source"] != decode_snapshot(transaction["source_snapshot"]):
        raise configuration.ConfigurationError("配置或接口/运行库已变化，已恢复本次环境修改；请重试。", 6)
    if not configuration.generation_status(transaction["home"], plan)["prepared"]:
        raise configuration.ConfigurationError("准备的产物已失效，请重试。", 5)
    if transaction["save"]:
        core.publish_configuration(
            plan["source"],
            decode_snapshot(transaction["target_snapshot"]),
            plan["source"]["content"],
            [Path(transaction["generation"])],
        )


def main():
    operation, directory_name, *arguments = sys.argv[1:]
    directory = Path(directory_name)
    json_output = "--json" in arguments

    def error(message, code):
        advice = {
            2: "检查配置文件路径和字段；默认配置损坏时可用 set --config 指定有效文件。",
            3: "检查提示中的运行库或版本要求；缺依赖请使用项目 setup 流程。",
            4: "用 dds-config show 查看冲突；配置状态损坏或 HOME 改变时请在新终端初始化。",
            5: "重新执行 dds-config set 准备配置，临时配置加 --tmp。",
            6: "检查文件权限或并发配置操作，处理后重试。",
            130: "需要时重新执行命令。",
        }.get(code, "检查上述错误后重试。")
        if json_output:
            print(json.dumps({"error": message, "code": code, "advice": advice}), file=sys.stderr)
        else:
            print(f"DDS：{message}\n建议：{advice}", file=sys.stderr)
        return code

    try:
        if operation in ("set", "source", "unset"):
            # Shell captures exported AND unexported values in the stdin record stream.
            records = sys.stdin.buffer.read().split(b"\0")
            values = {}
            if records[-1] == b"":
                records.pop()
            for index in range(0, len(records), 3):
                name, present, value = (item.decode() for item in records[index : index + 3])
                if name not in KEYS + ["ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY"]:
                    raise ValueError("Unknown shell field")
                if present == "1":
                    values[name] = value
            transaction = prepare(directory, operation, arguments, values)
            (directory / "transaction.json").write_text(json.dumps(transaction))
            print(transaction["action"])
        else:
            transaction = json.loads((directory / "transaction.json").read_text())
            json_output = transaction.get("json", False)
            removed = commit(transaction)
            if transaction["action"] == "unset":
                if transaction["json"]:
                    print(
                        json.dumps(
                            {
                                "restored_current_shell": transaction["restored_current_shell"],
                                "temporary": transaction["temporary"],
                                "removed_default": removed,
                            }
                        )
                    )
                else:
                    print(wizard.emphasize("── 取消设置成功 ──", sys.stdout))
                    print(
                        "已退出 DDS 配置管理，当前终端已恢复原环境。"
                        if transaction["restored_current_shell"]
                        else "当前终端未由 dds-config 接管，环境保持不变。"
                    )
                    print(
                        "全局默认配置未修改。"
                        if transaction["temporary"]
                        else "全局默认配置已移除。"
                        if removed
                        else "全局默认配置未配置。"
                    )
                    if transaction["temporary"]:
                        print("重新应用默认配置：source .shrc_local")
                    print("已运行的节点未更改配置。")
            if transaction["action"] == "apply" and transaction["mode"] == "set":
                if transaction["json"]:
                    print(
                        json.dumps(
                            {
                                "backend": transaction["backend"],
                                "temporary": transaction["temporary"],
                                "applied_to_current_shell": True,
                                "saved_as_default": transaction["save"],
                                "communication_verified": False,
                                "limits": "bounded participant search"
                                if transaction["discovery"] == "static_peers"
                                else None,
                            }
                        )
                    )
                else:
                    print()
                    print(wizard.emphasize("── 设置成功 ──", sys.stdout))
                    backend_label = {"fastdds": "Fast DDS", "cyclonedds": "CycloneDDS"}[transaction["backend"]]
                    print(f"当前终端 DDS：{backend_label}")
                    print(
                        "当前终端已生效，已保存为全局默认（当前用户）。"
                        if transaction["save"]
                        else "当前终端已生效，全局默认配置未修改。"
                    )
                    if not transaction["save"]:
                        print("返回默认配置：source .shrc_local")
                    print("已运行的节点未更改配置。")
                    if transaction["discovery"] == "static_peers" and not transaction["interactive"]:
                        print("注意：发现索引 0..32；端口占满或超出范围可能无法发现。")
        return 0
    except (wizard.Cancelled, KeyboardInterrupt):
        return error("已取消；配置和当前环境未修改。", 130)
    except configuration.ConfigurationError as exc:
        return error(str(exc), exc.code)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return error(str(exc), 6)


if __name__ == "__main__":
    raise SystemExit(main())
