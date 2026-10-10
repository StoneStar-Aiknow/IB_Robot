"""Pure DDS environment mapping; the shell owns baseline and rollback."""

from pathlib import Path

MANAGED = frozenset(
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


def preview(backend, profile, discovery, inherited):
    profile = Path(profile)
    if not profile.is_absolute():
        raise ValueError("DDS profile path must be absolute")
    if backend == "fastdds":
        values = {
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "FASTRTPS_DEFAULT_PROFILES_FILE": str(profile),
            "SKIP_DEFAULT_XML_FILE": "1",
        }
    elif backend == "cyclonedds":
        values = {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "CYCLONEDDS_URI": profile.as_uri()}
    else:
        raise ValueError("Unknown DDS backend")
    conflicts = [
        {"variable": name, "reason": "外部设置与目标配置不同；请在新终端配置或先恢复外部设置"}
        for name in sorted(MANAGED)
        if name in inherited and (name not in values or inherited[name] != values[name])
    ]
    if discovery != "local_only" and inherited.get("ROS_LOCALHOST_ONLY") == "1":
        conflicts.append({"variable": "ROS_LOCALHOST_ONLY", "reason": "仅本机设置与跨机发现冲突"})
    return {"set": values, "unset": sorted(MANAGED - values.keys()), "conflicts": conflicts}
