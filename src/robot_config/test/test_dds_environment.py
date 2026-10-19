"""Core managed DDS admission contracts; no participants or hardware required."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "robot_config" / "dds_environment.py"
_SPEC = importlib.util.spec_from_file_location("dds_environment_under_test", _PATH)
dds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dds)


def managed_environment(backend="fastdds", temporary=False):
    values = dict.fromkeys(dds._MANAGED)
    if backend == "fastdds":
        values.update(
            RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
            FASTRTPS_DEFAULT_PROFILES_FILE="/dds/generated/profile.xml",
            SKIP_DEFAULT_XML_FILE="1",
        )
    else:
        values.update(RMW_IMPLEMENTATION="rmw_cyclonedds_cpp", CYCLONEDDS_URI="file:///dds/generated/profile.xml")
    record = {
        "version": 1,
        "home": "/dds/home",
        "backend": backend,
        "generation": "/dds/generated",
        "temporary": temporary,
        "values": values,
        "discovery_domain": "19",
        "configuration": {
            "backend": backend,
            "network": {"discovery": "local_only"},
            "backends": {
                backend: {"profile": {"source": "generated", "preset": f"{backend}/conservative-v1", "parameters": {}}}
            },
        },
    }
    return {
        **{name: value for name, value in values.items() if value is not None},
        "HOME": record["home"],
        "ROS_DOMAIN_ID": "19",
        "IBROBOT_DDS_STATUS": "ready",
        "IBROBOT_DDS_STATE": json.dumps(record),
    }


@pytest.mark.parametrize("backend", ["fastdds", "cyclonedds"])
def test_unmanaged_and_valid_managed_environments_are_admitted(backend):
    assert dds.assess_dds_environment({}) is None
    assert dds.assess_dds_environment(managed_environment(backend)) == backend


@pytest.mark.parametrize("failure", ["degraded", "corrupt_record", "changed_rmw"])
def test_known_managed_failures_stop_launch(failure):
    env = managed_environment()
    if failure == "degraded":
        env["IBROBOT_DDS_STATUS"] = "degraded"
    elif failure == "corrupt_record":
        env["IBROBOT_DDS_STATE"] = "not JSON"
    else:
        env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    with pytest.raises(dds.DDSEnvironmentError, match="已停止启动"):
        dds.assess_dds_environment(env)


@pytest.mark.parametrize("temporary", [False, True])
def test_changed_domain_is_rejected_with_configuration_not_build_advice(temporary):
    env = managed_environment(temporary=temporary)
    env["ROS_DOMAIN_ID"] = "20"
    with pytest.raises(dds.DDSEnvironmentError, match="ROS_DOMAIN_ID") as raised:
        dds.assess_dds_environment(env)
    message = str(raised.value)
    assert "19" in message and "20" in message
    assert "dds-config set --tmp" in message if temporary else "source .shrc_local" in message
    assert "恢复" in message
    assert "build.sh" not in message


@pytest.mark.parametrize("backend,mode", [("fastdds", "lan_multicast"), ("cyclonedds", "local_only")])
def test_changed_domain_does_not_add_a_unicast_fast_guard_to_other_modes(backend, mode):
    env = managed_environment(backend)
    record = json.loads(env["IBROBOT_DDS_STATE"])
    record["configuration"]["network"]["discovery"] = mode
    env.update(IBROBOT_DDS_STATE=json.dumps(record), ROS_DOMAIN_ID="20")
    assert dds.assess_dds_environment(env) == backend


def test_embodied_fast_override_is_local_and_does_not_hide_other_conflicts():
    env = managed_environment()
    env["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
    context = SimpleNamespace(environment=env, launch_configurations={})
    with pytest.raises(dds.DDSEnvironmentError, match="FASTDDS_BUILTIN_TRANSPORTS"):
        dds.check_launch_dds(context)
    context.launch_configurations["_ibrobot_embodied_fast_udp"] = "true"
    before = dict(env)
    assert dds.check_launch_dds(context) == "fastdds"
    assert env == before
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    with pytest.raises(dds.DDSEnvironmentError, match="RMW_IMPLEMENTATION"):
        dds.check_launch_dds(context)
    context.environment = {**managed_environment("cyclonedds"), "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"}
    with pytest.raises(dds.DDSEnvironmentError, match="FASTDDS_BUILTIN_TRANSPORTS"):
        dds.check_launch_dds(context)
