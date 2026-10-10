"""Local launch admission for the DDS shell record v1.

Installed launch files import this module from robot_config. scripts/dds is a
source-workspace shell tool, not an installed Python package; do not import it
from launch. This module only reads inherited data: no YAML, disk/network
probing, ROS imports or environment mutation. Successful admission does not
certify communication.
"""

import json
import os
from pathlib import PurePosixPath
from typing import NoReturn

_MANAGED = frozenset(
    {
        "RMW_IMPLEMENTATION",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
        "FASTDDS_BUILTIN_TRANSPORTS",
        "RMW_FASTRTPS_USE_QOS_FROM_XML",
        "RMW_FASTRTPS_PUBLICATION_MODE",
        "SKIP_DEFAULT_XML_FILE",
        "ROS_DISCOVERY_SERVER",
        "ROS_SUPER_CLIENT",
        "FASTDDS_ENVIRONMENT_FILE",
    }
)
_RMW = {"fastdds": "rmw_fastrtps_cpp", "cyclonedds": "rmw_cyclonedds_cpp"}
_NAMES = {"rmw_fastrtps_cpp": "Fast DDS", "rmw_cyclonedds_cpp": "CycloneDDS"}


class DDSEnvironmentError(RuntimeError):
    """An actionable, local error that must stop this launch."""


class _DiscoveryDomainMismatch(ValueError):
    """A valid domain changed after generating managed Fast DDS discovery."""


def _fail(reason, advice) -> NoReturn:
    raise DDSEnvironmentError(f"DDS 启动检查失败，已停止启动。\n原因：{reason}\n建议：{advice}")


def _record(env):
    raw = env.get("IBROBOT_DDS_STATE")
    if not isinstance(raw, str) or len(raw) > 65536:
        raise ValueError("dds-config 配置状态缺失或大小异常")
    record = json.loads(raw)
    if not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != 1:
        raise ValueError("dds-config 配置状态版本与当前应用不兼容")
    backend = record.get("backend")
    if not isinstance(backend, str) or backend not in _RMW:
        raise ValueError("dds-config 配置中的后端无法识别")
    if record.get("home") != env.get("HOME") or not env.get("HOME"):
        raise ValueError("HOME 与执行 dds-config 时不一致")
    values = record.get("values")
    if not isinstance(values, dict) or set(values) != _MANAGED:
        raise ValueError("dds-config 配置状态不完整")
    if any(value is not None and not isinstance(value, str) for value in values.values()):
        raise ValueError("dds-config 配置状态含无效变量值")
    if type(record.get("temporary")) is not bool:
        raise ValueError("无法确认 dds-config 配置是默认还是临时设置")
    generation = record.get("generation")
    if not isinstance(generation, str) or not PurePosixPath(generation).is_absolute():
        raise ValueError("dds-config 生成的配置路径无效")
    config = record.get("configuration")
    if not isinstance(config, dict) or config.get("backend") != backend:
        raise ValueError("无法确认 dds-config 已生效配置")
    profile = config.get("backends", {}).get(backend, {}).get("profile", {})
    # Only the shipped generated-v1 contract is understood here. In particular,
    # do not silently apply the embodied legacy transport override to user XML.
    if (
        profile.get("source") != "generated"
        or profile.get("preset") != f"{backend}/conservative-v1"
        or profile.get("parameters", {}) != {}
    ):
        raise ValueError("此 DDS 配置来源/预设尚不在本机启动检查的支持范围")
    mode = config.get("network", {}).get("discovery")
    if mode not in ("local_only", "lan_multicast", "static_peers"):
        raise ValueError("dds-config 配置中的通信范围无法识别")
    if backend == "fastdds" and mode != "lan_multicast":
        domain = record.get("discovery_domain")
        active_domain = env.get("ROS_DOMAIN_ID", "0")
        if not isinstance(domain, str) or not domain.isascii() or not domain.isdecimal():
            raise ValueError("Fast DDS 的发现 domain 记录无效；请重新 source .shrc_local")
        if not isinstance(active_domain, str) or not active_domain.isascii() or not active_domain.isdecimal():
            raise ValueError("ROS_DOMAIN_ID 无效")
        if int(active_domain) != int(domain):
            raise _DiscoveryDomainMismatch(
                f"ROS_DOMAIN_ID 与生成 Fast DDS 发现端口时的 domain 不一致：当前 {active_domain}，生成时 {domain}"
            )
    profile_path = PurePosixPath(generation) / "profile.xml"
    expected = dict.fromkeys(_MANAGED)
    expected["RMW_IMPLEMENTATION"] = _RMW[backend]
    if backend == "fastdds":
        expected.update(FASTRTPS_DEFAULT_PROFILES_FILE=str(profile_path), SKIP_DEFAULT_XML_FILE="1")
    else:
        expected["CYCLONEDDS_URI"] = profile_path.as_uri()
    if values != expected:
        raise ValueError("dds-config 配置中的后端与 XML 设置不一致")
    return record


def assess_dds_environment(env):
    """Return the known generated backend, or None for unmanaged environments.

    Raise before admission on a known failure or unsupported record.
    Unmanaged means unchecked, not ready. Peers are never inspected.
    """
    if "IBROBOT_DDS_STATE" not in env and "IBROBOT_DDS_STATUS" not in env:
        return None

    def fail(reason, advice) -> NoReturn:
        _fail(reason, advice)

    if env.get("IBROBOT_DDS_STATUS") == "degraded":
        fail(
            "上次 DDS 配置未生效",
            "查看 set/source 的报错，修复后重新设置；用 dds-config show 查看当前配置。",
        )
    if env.get("IBROBOT_DDS_STATUS") != "ready":
        fail("dds-config 配置状态缺失或无法识别", "在新终端 source .shrc_local，再用 dds-config show 检查。")
    try:
        record = _record(env)
    except _DiscoveryDomainMismatch as exc:
        fail(
            str(exc),
            "默认配置：在当前终端重新 source .shrc_local；临时配置：重新 dds-config set --tmp"
            "（使用原 YAML 时加 --config 文件），按当前 ROS_DOMAIN_ID 重新生成；或恢复生成时的 domain。",
        )
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as exc:
        fail(
            f"进程的 dds-config 配置状态无法确认：{exc}",
            "在新终端用同一工作区的 dds-config 重新配置；应用版本较旧时用 ./scripts/build.sh 重新构建。",
        )
    for key, expected in record["values"].items():
        if env.get(key) != expected or (key in env) != (expected is not None):
            configured = _NAMES.get(expected, repr(expected)) if key == "RMW_IMPLEMENTATION" else repr(expected)
            actual = env.get(key)
            current = _NAMES.get(actual, repr(actual)) if key == "RMW_IMPLEMENTATION" else repr(actual)
            fail(
                f"进程的 {key} 与 dds-config 已生效配置不一致。\n  dds-config 配置：{configured}\n  当前值：{current}",
                f"检查 {key} 的额外设置；在新终端用 dds-config set 重新配置（临时配置加 --tmp）。",
            )
    if env.get("ROS_LOCALHOST_ONLY", "0") not in ("0", "1"):
        fail("ROS_LOCALHOST_ONLY 应为 0 或 1", "按应用需要显式设置有效值后重试。")
    if record["configuration"]["network"]["discovery"] != "local_only" and env.get("ROS_LOCALHOST_ONLY") == "1":
        fail(
            "ROS_LOCALHOST_ONLY=1 与当前跨机发现配置冲突",
            "跨机使用请显式设置 ROS_LOCALHOST_ONLY=0；只需本机则用 dds-config set 选择仅本机。",
        )
    return record["backend"]


def check_launch_dds(context):
    """Read the current launch environment (os.environ on ROS 2 Humble)."""
    env = getattr(context, "environment", os.environ)
    configurations = getattr(context, "launch_configurations", {})
    if configurations.get("_ibrobot_embodied_fast_udp") == "true" and env.get("FASTDDS_BUILTIN_TRANSPORTS") == "UDPv4":
        # Embodied's explicit workaround is applied after its initial strict
        # preflight. Validate every other field against the unmodified record;
        # a standalone launch without this scoped marker stays strict.
        without_override = dict(env)
        without_override.pop("FASTDDS_BUILTIN_TRANSPORTS")
        backend = assess_dds_environment(without_override)
        if backend == "fastdds":
            return backend
    return assess_dds_environment(env)


def guard_launch_dds(context):
    """OpaqueFunction adapter: no participant, subprocess or extra launch action."""
    check_launch_dds(context)
    return []
