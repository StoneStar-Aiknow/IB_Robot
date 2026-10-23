"""Human-readable saved/current configuration; independent of activation."""

import sys

from .wizard import emphasize

NAMES = {"fastdds": "Fast DDS", "cyclonedds": "CycloneDDS"}
MODES = {"local_only": "仅本机", "lan_multicast": "局域网自动发现（组播）", "static_peers": "固定对端（单播发现）"}


def selected_view(config, binding=None):
    if not config:
        return None
    backend = config["backend"]
    network = dict(config.get("network", {}))
    if "peers" in network:
        network["peers"] = sorted(set(network["peers"]))
    if binding and network.get("discovery") != "local_only":
        network["address"] = binding["address"]
    return {"backend": backend, "network": network}


def print_config(label, config, binding=None):
    view = selected_view(config, binding)
    assert view is not None
    network = view["network"]
    print(f"{label} DDS：{NAMES.get(view['backend'], view['backend'])}")
    print(f"  通信范围：{MODES.get(network.get('discovery'), '未确认')}")
    if network.get("interface"):
        print(f"  本机 IPv4：{network.get('address', '自动选择（尚未确认）')}（{network['interface']}）")
    if network.get("peers"):
        print("  对端 IPv4：" + ", ".join(network["peers"]))


def print_show(report):
    def section(title):
        print(emphasize(f"── {title} ──", sys.stdout))

    default = report.get("default_configuration")
    default_binding = report.get("default_binding")
    current = report.get("current") or {}
    active = current.get("configuration")
    identical = (
        default is not None
        and active is not None
        and current.get("status") == "ready"
        and "default_diagnostic" not in report
        and selected_view(default, default_binding) == selected_view(active, current.get("binding"))
    )
    section("全局默认")
    if default:
        print_config("默认", default, default_binding)
    else:
        print("默认 DDS：" + ("无法读取" if "default_diagnostic" in report else "未配置"))
    print()
    section("当前终端")
    if identical:
        print("当前终端：与全局默认一致")
    elif active:
        print_config("当前终端", active, current.get("binding"))
    elif current:
        print(f"当前终端 DDS：{NAMES.get(current.get('backend'), '无法确认')}")
        print("当前终端配置：无法完整确认")
    else:
        rmw = report["inherited"]["RMW_IMPLEMENTATION"]
        backend = {"rmw_fastrtps_cpp": "Fast DDS", "rmw_cyclonedds_cpp": "CycloneDDS"}.get(rmw)
        if backend:
            print(f"当前终端 DDS：{backend}（外部环境设置，未经本工具确认）")
        elif rmw is not None:
            print("当前终端 DDS：无法识别的 RMW 设置（用 --json 查看详情）")
        else:
            print("当前终端 DDS：未显式选择（由 ROS 默认选择）")
    print()
    section("状态")
    if not current and report["inherited"].get("IBROBOT_DDS_STATUS") is None:
        status = "未启用 dds-config"
    else:
        status = "ready" if current.get("status") == "ready" else "not ready"
    print(f"状态：{status}")
    if "diagnostic" in current:
        print(f"诊断：{current['diagnostic']['message']}")
    if "default_diagnostic" in report:
        print(f"默认配置诊断：{report['default_diagnostic']['message']}")
