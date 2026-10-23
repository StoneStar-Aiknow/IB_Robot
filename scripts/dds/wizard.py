"""Small terminal questionnaire; returns schema data, never applies or saves it."""

import copy
import ipaddress
import os
import re

from . import platform_facts


class Cancelled(Exception):
    pass


class WizardError(ValueError):
    pass


def emphasize(text, writer):
    """Only decorate an actual terminal; never insert escapes into plain output."""
    if writer.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ:
        return f"\033[1;36m{text}\033[0m"
    return text


def configure(previous, interface_reader, reader, writer, temporary=False):
    step = 0

    def say(text):
        print(text, file=writer, flush=True)

    def section(title):
        nonlocal step
        step += 1
        say("")
        say(emphasize(f"── {step}. {title} ──", writer))

    def answer(prompt, default=""):
        text = f"> {prompt}" + (f" [{default}]" if default else "") + ": "
        print(emphasize(text, writer), end="", file=writer, flush=True)
        line = reader.readline()
        if not line:
            raise Cancelled()
        value = line.strip()
        return value or default

    def choose(title, options, default=None):
        section(title)
        for index, (_, label) in enumerate(options, 1):
            say(f"  {index}. {label}")
        default_index = next((str(i) for i, (key, _) in enumerate(options, 1) if key == default), "1")
        while True:
            selected = answer("请选择编号", default_index)
            if selected.isdigit() and 1 <= int(selected) <= len(options):
                return options[int(selected) - 1][0]
            say("请输入列表中的编号。")

    config = copy.deepcopy(previous) if previous else {"schema_version": 1, "backends": {}}
    old_network = config.get("network", {})
    say("DDS 配置 · Ctrl-C 取消")
    backend = choose("使用哪个 DDS？", [("fastdds", "Fast DDS"), ("cyclonedds", "CycloneDDS")], config.get("backend"))
    mode = choose(
        "通信范围？",
        [
            ("local_only", "仅本机"),
            ("lan_multicast", "局域网自动发现（组播）"),
            ("static_peers", "固定对端（单播发现）"),
        ],
        old_network.get("discovery"),
    )
    network = {"discovery": mode}
    if mode != "local_only":
        interfaces = interface_reader()
        choices = []
        addresses = {}
        for item in interfaces:
            name = item.get("ifname", "")
            if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,15}", name):
                continue
            eligible = []
            for entry in item.get("addr_info", []):
                if entry.get("family") != "inet":
                    continue
                try:
                    bound = platform_facts.resolve_binding(
                        {"discovery": mode, "interface": name, "address": entry["local"]},
                        interfaces,
                    )
                except (platform_facts.FactsError, KeyError):
                    continue
                if bound["address"] not in eligible:
                    eligible.append(bound["address"])
            if eligible:
                addresses[name] = eligible
                choices.append((name, f"{name} · {', '.join(eligible)}"))
        if not choices:
            raise WizardError("没有可用的 IPv4 网卡；请检查接口连接和地址。")
        name = choose("使用哪张网卡？", choices, old_network.get("interface"))
        candidates = addresses[name]
        if len(candidates) == 1:
            address = candidates[0]
        else:
            address = choose("使用哪个本机 IPv4？", [(ip, ip) for ip in candidates], old_network.get("address"))
        network.update(interface=name, address=address)
        if mode == "static_peers":
            section("设置对端 IPv4")
            while True:
                text = answer("对端 IPv4（逗号分隔）", ",".join(old_network.get("peers", [])))
                try:
                    peers = sorted({str(ipaddress.IPv4Address(value.strip())) for value in text.split(",")})
                    check = {**network, "peers": peers}
                    # Real routes are checked by the existing platform adapter
                    # after confirmation; here reject local/broadcast entries.
                    platform_facts.validate_peer_routes(
                        check,
                        {"interface": name, "address": address},
                        interfaces,
                        {peer: [{"dev": name, "prefsrc": address}] for peer in peers},
                    )
                except (ValueError, platform_facts.FactsError):
                    say("请输入远端单播 IPv4；不能填本机、组播或广播地址。")
                    continue
                network["peers"] = peers
                break
    config.update(backend=backend, network=network)
    config["backends"][backend] = {
        "profile": {
            "source": "generated",
            "preset": f"{backend}/conservative-v1",
            "parameters": {},
        }
    }
    backend_label = {"fastdds": "Fast DDS", "cyclonedds": "CycloneDDS"}[backend]
    mode_label = {"local_only": "仅本机发现", "lan_multicast": "局域网自动发现", "static_peers": "固定对端发现"}[mode]
    section("确认配置")
    say(f"配置摘要：{backend_label} · {mode_label}")
    if backend == "fastdds":
        say("Fast DDS：所选 IP 用于发现，不限制内置传输的网卡；数据传输仍由运行库选择（含可用的共享内存）。")
    if mode != "local_only":
        say(f"网卡：{network['interface']} · {network['address']}")
    if mode == "static_peers":
        say("对端：" + ", ".join(network["peers"]))
        say(
            "注意：Fast DDS 初始对端搜索范围由所装运行库决定；端口占满或超出范围可能无法发现。"
            if backend == "fastdds"
            else "注意：CycloneDDS participant 搜索范围由所装运行库决定；端口占满或超出范围可能无法发现。"
        )
    say(
        "仅当前终端生效，全局默认配置不变。"
        if temporary
        else "保存为全局默认（当前用户；已运行节点不会更新），并立即应用到当前终端。"
    )
    while True:
        confirmation = answer("确认应用？y/N", "n").lower()
        if confirmation in ("y", "yes"):
            return config
        if confirmation in ("n", "no"):
            raise Cancelled()
        say("请输入 y 或 n。")
