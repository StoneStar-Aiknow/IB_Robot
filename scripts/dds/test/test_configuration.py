"""Discovery, immutable generation and shell lifecycle without system changes."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("dds_configuration_under_test", ROOT / "scripts/dds/__init__.py")
_PACKAGE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PACKAGE
_SPEC.loader.exec_module(_PACKAGE)
from dds_configuration_under_test import (  # noqa: E402 - load this checkout's scripts package first
    configuration,
    core,
    cyclonedds,
    environment_plan,
    fastdds,
    shell_bridge,
)


def config(backend="fastdds", mode="local_only"):
    network = {"discovery": mode}
    if mode != "local_only":
        network.update(interface="eth0", address="192.0.2.10")
    if mode == "static_peers":
        network["peers"] = ["192.0.2.20"]
    return {
        "schema_version": 1,
        "backend": backend,
        "network": network,
        "backends": {
            backend: {"profile": {"source": "generated", "preset": f"{backend}/conservative-v1", "parameters": {}}}
        },
    }


def local_facts(_backend, _network):
    return {"packages": {}}, {"interface": "lo", "address": "127.0.0.1"}


@pytest.mark.parametrize("mode", ["local_only", "static_peers"])
def test_unicast_discovery_covers_indices_through_32_without_replacing_transports(mode):
    network = config(mode=mode)["network"]
    address = "127.0.0.1" if mode == "local_only" else "192.0.2.10"
    binding = {"interface": "lo" if mode == "local_only" else "eth0", "address": address}
    cyclone = ET.fromstring(cyclonedds.render(network, binding))
    assert cyclone.findtext(".//{*}MaxAutoParticipantIndex") == "32"
    assert cyclone.findtext(".//{*}ParticipantIndex") == "auto"
    assert cyclone.findtext(".//{*}AllowMulticast") == "false"
    fast = ET.fromstring(fastdds.render(network, binding, domain_id=19))
    peers = fast.findall(".//{*}initialPeersList/{*}locator/{*}udpv4")
    for ip in [address, *network.get("peers", [])]:
        ports = {item.findtext("{*}port") for item in peers if item.findtext("{*}address") == ip}
        assert ports == {None, *(str(7400 + 250 * 19 + 10 + 2 * index) for index in range(4, 33))}
    for tag in ("userTransports", "useBuiltinTransports", "defaultUnicastLocatorList"):
        assert fast.find(f".//{{*}}{tag}") is None


def test_multicast_keeps_vendor_discovery_and_transport_defaults():
    network = config(mode="lan_multicast")["network"]
    binding = {"interface": "eth0", "address": "192.0.2.10"}
    for renderer in (fastdds, cyclonedds):
        root = ET.fromstring(renderer.render(network, binding))
        for tag in ("initialPeersList", "MaxAutoParticipantIndex", "Transport", "userTransports"):
            assert root.find(f".//{{*}}{tag}") is None


@pytest.mark.parametrize("invalid", ["domain", "parameters", "empty_peers"])
def test_invalid_configuration_is_rejected_before_generation(monkeypatch, invalid):
    value = config(mode="static_peers")
    if invalid == "domain":
        monkeypatch.setenv("ROS_DOMAIN_ID", "233")
        with pytest.raises(configuration.ConfigurationError, match="ROS_DOMAIN_ID"):
            configuration.plan_configuration(value, fact_reader=local_facts)
    else:
        if invalid == "parameters":
            value["backends"]["fastdds"]["profile"]["parameters"] = {"custom": True}
        else:
            value["network"]["peers"] = []
        with pytest.raises(core.ConfigError):
            core.parse_config(json.dumps(value))


def test_source_regenerates_for_current_domain_without_installing_or_overwriting(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ROS_DOMAIN_ID", "19")
    monkeypatch.setattr(configuration, "collect_facts", local_facts)
    if hasattr(shell_bridge, "dependencies"):
        monkeypatch.setattr(
            shell_bridge.dependencies, "install_if_missing", lambda *_: pytest.fail("source must not install packages")
        )
    default = tmp_path / ".config/ibrobot/dds/config.yaml"
    default.parent.mkdir(parents=True)
    content = json.dumps(config())
    default.write_text(content)
    first = configuration.prepare(str(tmp_path), configuration.plan(default))
    old_xml = (Path(first["generation"]) / "profile.xml").read_bytes()
    monkeypatch.setenv("ROS_DOMAIN_ID", "20")
    transaction = shell_bridge.prepare(tmp_path, "source", [], {})
    assert transaction["action"] == "apply"
    assert transaction["generation"] != first["generation"]
    assert default.read_text() == content
    assert (Path(first["generation"]) / "profile.xml").read_bytes() == old_xml
    new_xml = ET.fromstring((Path(transaction["generation"]) / "profile.xml").read_bytes())
    assert str(7400 + 250 * 20 + 10 + 2 * 32) in {node.text for node in new_xml.findall(".//{*}port")}
    (Path(transaction["generation"]) / "profile.xml").write_text("damaged")
    with pytest.raises(configuration.ConfigurationError):
        shell_bridge.prepare(tmp_path, "source", [], {})


def test_bash_set_source_unset_preserve_default_and_original_environment(tmp_path):
    """Exercise the real Bash transaction path, stubbing only host facts/install."""
    command_bin = tmp_path / "bin"
    command_bin.mkdir()
    driver = command_bin / "shell_entry.py"
    driver.write_text(
        f"import importlib.util, sys\nfrom pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('dds_shell_test', {str(ROOT / 'scripts/dds/__init__.py')!r})\n"
        "package = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = package\n"
        "spec.loader.exec_module(package)\n"
        "from dds_shell_test import configuration, shell_bridge\n"
        "configuration.collect_facts = lambda *_: ({}, {'interface': 'lo', 'address': '127.0.0.1'})\n"
        "if hasattr(shell_bridge, 'dependencies'):\n"
        "    shell_bridge.dependencies.install_if_missing = lambda *_: False\n"
        "raise SystemExit(shell_bridge.main())\n"
    )
    source = tmp_path / "selection.yaml"
    source.write_text(json.dumps(config()))
    script = r"""
set -e
source "$1/scripts/dds/shell_activation.bash"
_IBROBOT_DDS_COMMAND_BIN="$HOME/bin"
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
dds-config set --config "$HOME/selection.yaml"
saved=$(cat "$HOME/.config/ibrobot/dds/config.yaml")
test "$IBROBOT_DDS_STATUS" = ready
export ROS_DOMAIN_ID=20
dds-config set --config "$HOME/selection.yaml" --tmp
test "$saved" = "$(cat "$HOME/.config/ibrobot/dds/config.yaml")"
dds-config unset --tmp
test "$RMW_IMPLEMENTATION" = rmw_fastrtps_cpp
test "$(declare -p RMW_IMPLEMENTATION)" = 'declare -- RMW_IMPLEMENTATION="rmw_fastrtps_cpp"'
test -z "${FASTRTPS_DEFAULT_PROFILES_FILE+x}"
test -z "${IBROBOT_DDS_STATUS+x}"
_ib_dds_activate source
test "$IBROBOT_DDS_STATUS" = ready
dds-config set --config "$HOME/missing.yaml" && exit 1
test "$IBROBOT_DDS_STATUS" = ready
test "$saved" = "$(cat "$HOME/.config/ibrobot/dds/config.yaml")"
dds-config unset
test ! -e "$HOME/.config/ibrobot/dds/config.yaml"
test -z "${IBROBOT_DDS_STATUS+x}"
test "$RMW_IMPLEMENTATION" = rmw_fastrtps_cpp
"""
    env = {name: value for name, value in os.environ.items() if name not in environment_plan.MANAGED}
    env.update(HOME=str(tmp_path), ROS_DOMAIN_ID="19")
    env.pop("IBROBOT_DDS_STATE", None)
    env.pop("IBROBOT_DDS_STATUS", None)
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script, "dds-test", str(ROOT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
