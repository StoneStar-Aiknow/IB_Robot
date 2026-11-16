"""Unitree G1 runtime launch setup generation and peripheral fragments.

runtime-deployment-peripherals task 4.2: the launch setup must generate —
``build_description`` is called with ``use_sim=`` (the ``simulated=`` keyword
mismatched the signature since the use_sim rename and made every launch-setup
render raise ``TypeError``).

Task 4.3 (D7): a deployment fragment disabling the declared ``head_front``
RealSense camera removes it from the description the runtime node serves and
from the synthetic perception actions; an empty fragment is unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import launch
import pytest
import yaml
from launch_ros.actions import Node

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"
_PRESET_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_arm_presets.yaml"
_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"

#: The simulated-transport binding the shipped profile declares. Tests compare
#: against it (and against values derived from it) instead of naming a domain,
#: so the profile stays the single source and no test hardcodes one.
_PROFILE_SIM = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))["sim"]
_SIM_DOMAIN = int(_PROFILE_SIM["vendor_domain_id"])


def _text(value):
    if isinstance(value, list | tuple):
        return "".join(_text(item) for item in value)
    return str(getattr(value, "text", value))


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("unitree_g1_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(peripherals_file: str, simulated: str = "true", vendor_domain_id: str = ""):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "use_sim": "",
            "sim_platform": "",
            "simulated": simulated,
            "instance_id": "",
            "initial_mode": "",
            "network_interface": "",
            "vendor_domain_id": vendor_domain_id,
            "peripherals": peripherals_file,
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _render_transport(simulated: str, network_interface: str):
    """One render of the shipped profile on the requested transport."""
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "use_sim": "",
            "sim_platform": "",
            "simulated": simulated,
            "instance_id": "",
            "initial_mode": "",
            "network_interface": network_interface,
            "vendor_domain_id": "",
            "peripherals": "",
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _render_preset(**overrides):
    """One render of the arm-preset profile on the requested transport."""
    context = launch.LaunchContext()
    configurations = {
        "profile": str(_PRESET_PROFILE),
        "use_sim": "false",
        "sim_platform": "",
        "simulated": "",
        "instance_id": "",
        "initial_mode": "",
        "network_interface": "",
        "vendor_domain_id": "",
        "peripherals": "",
        "runtime_network_interface": "",
        "preset_operator_confirmed": "false",
        "preset_expected_fsm_id": "",
    }
    configurations.update(overrides)
    context.launch_configurations.update(configurations)
    return _runtime_launch_module()._launch_setup(context)


def _profile_with_validation(tmp_path, **validation):
    profile = yaml.safe_load(_PRESET_PROFILE.read_text(encoding="utf-8"))
    profile["vendor"]["operator_validation"] = validation
    path = tmp_path / "g1_29dof_arm_presets_with_validation.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _node_params(node):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters

    params = {}
    for entry in evaluate_parameters(LaunchContext(), node._Node__parameters):
        params.update(entry)
    return params


def test_launch_setup_generates_on_the_simulated_path():
    """The pre-existing build_description(simulated=) keyword mismatch made
    every launch-setup render fail with TypeError; with use_sim= the setup
    generates (task 4.2)."""
    actions = _render("")
    executables = {node._Node__node_executable for node in actions if isinstance(node, Node)}
    assert "runtime_node" in executables
    assert "vendor_mock_node" in executables


def test_launch_setup_applies_the_fragment_to_the_description(tmp_path):
    """D7: a fragment disabling the declared head_front camera removes it
    from the description the runtime node serves and from the synthetic
    perception publishers; an empty fragment changes nothing."""
    empty = _render("")
    runtime = next(node for node in empty if isinstance(node, Node) and node._Node__node_executable == "runtime_node")
    empty_description = json.loads(_node_params(runtime)["interface_description_json"])
    assert any(name.startswith("camera.head_front.") for name in empty_description["interfaces"])
    empty_synthetic = next(
        node for node in empty if isinstance(node, Node) and node._Node__node_executable == "synthetic_perception"
    )
    assert any(
        name.startswith("camera.head_front.")
        for name in json.loads(_node_params(empty_synthetic)["description_json"])["interfaces"]
    )

    fragment = tmp_path / "deployment_peripherals.yaml"
    fragment.write_text(
        yaml.safe_dump({"peripherals": [{"type": "camera", "name": "head_front", "disabled": True}]}),
        encoding="utf-8",
    )
    disabled = _render(str(fragment))
    runtime = next(
        node for node in disabled if isinstance(node, Node) and node._Node__node_executable == "runtime_node"
    )
    description = json.loads(_node_params(runtime)["interface_description_json"])
    assert not any(name.startswith("camera.head_front.") for name in description["interfaces"])
    synthetic = next(
        node for node in disabled if isinstance(node, Node) and node._Node__node_executable == "synthetic_perception"
    )
    assert not any(
        name.startswith("camera.head_front.")
        for name in json.loads(_node_params(synthetic)["description_json"])["interfaces"]
    )
    # the lidar surface stays
    assert "lidar.chest_front.points" in description["interfaces"]


def test_launch_declares_the_peripherals_argument():
    """D7 (m3): the provider's real launch description declares the
    peripherals argument every provider contract requires."""
    from launch.actions import DeclareLaunchArgument

    def text(value):
        return "".join(part.text if hasattr(part, "text") else str(part) for part in value)

    description = _runtime_launch_module().generate_launch_description()
    names = {
        text(action._DeclareLaunchArgument__name)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "peripherals" in names


def test_arm_preset_launch_refuses_physical_transport(monkeypatch):
    import pytest

    monkeypatch.setattr(__import__(__name__), "_PROFILE", _PROFILE.with_name("g1_29dof_arm_presets.yaml"))
    with pytest.raises(RuntimeError, match="PRESET_HARDWARE_DISABLED"):
        _render("", simulated="false")


def test_arm_preset_simulation_is_loopback_and_has_no_dex3_description(monkeypatch):
    monkeypatch.setattr(__import__(__name__), "_PROFILE", _PROFILE.with_name("g1_29dof_arm_presets.yaml"))
    nodes = [item for item in _render("") if isinstance(item, Node)]
    assert len(nodes) == 2
    runtime = next(n for n in nodes if n._Node__node_executable == "runtime_node")
    params = _node_params(runtime)
    assert params["vendor_network_interface"] == "lo"
    assert params["vendor_domain_id"] != 0
    assert params["preset_test_transport"] is True
    description = json.loads(params["interface_description_json"])
    assert description["execution"] == "simulated"
    assert "motion.named" in description["interfaces"]


def test_arm_preset_launch_declares_the_per_run_commissioning_arguments():
    from launch.actions import DeclareLaunchArgument

    description = _runtime_launch_module().generate_launch_description()
    arguments = {
        _text(action._DeclareLaunchArgument__name): action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert arguments["preset_operator_confirmed"].default_value[0].text == "false"
    assert arguments["runtime_network_interface"].default_value[0].text == ""
    assert arguments["preset_expected_fsm_id"].default_value[0].text == ""


def test_arm_preset_physical_defaults_fail_closed():
    with pytest.raises(RuntimeError, match="PRESET_HARDWARE_DISABLED"):
        _render_preset()


@pytest.mark.parametrize("runtime_network_interface", ["", "lo", "  "])
def test_arm_preset_physical_requires_a_robot_facing_nic(runtime_network_interface):
    with pytest.raises(RuntimeError, match="PRESET_HARDWARE_DISABLED.*runtime_network_interface"):
        _render_preset(
            preset_operator_confirmed="true",
            runtime_network_interface=runtime_network_interface,
            preset_expected_fsm_id="802",
        )


@pytest.mark.parametrize("expected_fsm", ["", "0", "-1", "802.0", "not-an-id"])
def test_arm_preset_physical_requires_a_valid_expected_fsm(expected_fsm):
    with pytest.raises(RuntimeError, match="PRESET_HARDWARE_DISABLED|positive integer FSM id"):
        _render_preset(
            preset_operator_confirmed="true",
            runtime_network_interface="eth9",
            preset_expected_fsm_id=expected_fsm,
        )


def test_arm_preset_physical_rejects_an_unparseable_confirmation():
    with pytest.raises(RuntimeError, match="preset_operator_confirmed must be a boolean"):
        _render_preset(
            preset_operator_confirmed="on-site",
            runtime_network_interface="eth9",
            preset_expected_fsm_id="802",
        )


def test_arm_preset_physical_passes_per_run_parameters_without_a_profile_file(monkeypatch):
    def _no_temp_profile(*_args, **_kwargs):
        raise AssertionError("physical preset commissioning must not write a profile file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _no_temp_profile)
    actions = _render_preset(
        preset_operator_confirmed="true",
        runtime_network_interface="eth9",
        preset_expected_fsm_id="802",
    )
    runtime = next(node for node in actions if isinstance(node, Node))
    params = _node_params(runtime)
    assert params["preset_operator_confirmed"] is True
    assert params["runtime_network_interface"] == "eth9"
    assert params["preset_expected_fsm_id"] == 802
    assert params["vendor_network_interface"] == "eth9"
    assert params["vendor_domain_id"] == 0
    assert params["preset_test_transport"] is False
    # The node reads the audited source profile; the per-run values ride as
    # ROS parameters, so no run-scoped or confirmed profile exists anywhere.
    assert params["profile"] == str(_PRESET_PROFILE)
    source_profile = yaml.safe_load(_PRESET_PROFILE.read_text(encoding="utf-8"))
    assert "operator_validation" not in source_profile["vendor"]


def test_arm_preset_physical_accepts_an_audited_profile_fsm(tmp_path):
    profile = _profile_with_validation(tmp_path, expected_fsm_id=500)
    actions = _render_preset(
        profile=str(profile),
        preset_operator_confirmed="true",
        runtime_network_interface="eth9",
    )
    params = _node_params(next(node for node in actions if isinstance(node, Node)))
    assert params["profile"] == str(profile)
    assert params["preset_expected_fsm_id"] == 500


def test_arm_preset_physical_rejects_conflicting_overrides(tmp_path):
    profile = _profile_with_validation(tmp_path, expected_fsm_id=500)
    with pytest.raises(RuntimeError, match="conflicting expected FSM ids"):
        _render_preset(
            profile=str(profile),
            preset_operator_confirmed="true",
            runtime_network_interface="eth9",
            preset_expected_fsm_id="802",
        )
    with pytest.raises(RuntimeError, match="conflicting network interface overrides"):
        _render_preset(
            preset_operator_confirmed="true",
            runtime_network_interface="eth9",
            preset_expected_fsm_id="802",
            network_interface="eth0",
        )
    profile_nic = _profile_with_validation(tmp_path, network_interface="eth0", expected_fsm_id=802)
    with pytest.raises(RuntimeError, match="conflicting network interface overrides"):
        _render_preset(
            profile=str(profile_nic),
            preset_operator_confirmed="true",
            runtime_network_interface="eth9",
        )


def test_arm_preset_simulation_does_not_apply_the_physical_gate():
    nodes = [
        item for item in _render_preset(use_sim="true", preset_expected_fsm_id="ignored") if isinstance(item, Node)
    ]
    runtime = next(node for node in nodes if node._Node__node_executable == "runtime_node")
    params = _node_params(runtime)
    assert params["preset_test_transport"] is True
    assert params["vendor_network_interface"] == "lo"


# --- E1: the vendor DDS binding is a parameter on every path -------------------


def _db(binding):
    """One render of a non-preset profile onto the simulated/physical path."""
    from unitree_g1_robot.runtime_node import vendor_dds_binding

    return vendor_dds_binding(*binding)


def test_simulated_transport_binds_the_private_domain_on_loopback():
    params_by_executable = {}
    for node in _render(""):
        if not isinstance(node, Node):
            continue
        params_by_executable.setdefault(node._Node__node_executable, []).append(_node_params(node))
    for executable in ("runtime_node", "vendor_mock_node"):
        params = params_by_executable[executable][0]
        assert params["vendor_domain_id"] != 0, "simulation must never join the robot's production domain"
        assert params["vendor_network_interface"] == "lo"
        # nothing overrides it, so it is exactly the profile's declaration
        assert params["vendor_domain_id"] == _SIM_DOMAIN
        # and the pair is one the node accepts
        assert _db((params["vendor_domain_id"], params["vendor_network_interface"])) == (
            _SIM_DOMAIN,
            _PROFILE_SIM["vendor_network_interface"],
        )


def test_real_transport_passes_the_named_nic_to_the_vendor_tier():
    actions = _render_transport("false", "eth9")
    runtime = next(n for n in actions if isinstance(n, Node) and n._Node__node_executable == "runtime_node")
    params = _node_params(runtime)
    assert params["vendor_domain_id"] == 0
    assert params["vendor_network_interface"] == "eth9"
    assert _db((0, "eth9")) == (0, "eth9")


def test_real_transport_without_a_nic_fails_closed():
    with pytest.raises(RuntimeError, match="VENDOR_DDS_INTERFACE_REQUIRED"):
        _render_transport("false", "")
    with pytest.raises(RuntimeError, match="VENDOR_DDS_ISOLATION"):
        _render_transport("false", "lo")


@pytest.mark.parametrize(
    ("binding", "refusal"),
    [
        ((0, ""), "VENDOR_DDS_INTERFACE_REQUIRED"),
        ((0, "lo"), "VENDOR_DDS_ISOLATION_REQUIRED"),
        ((_SIM_DOMAIN, ""), "VENDOR_DDS_ISOLATION_REQUIRED"),
        ((_SIM_DOMAIN, "eth0"), "VENDOR_DDS_ISOLATION_REQUIRED"),
    ],
)
def test_the_node_refuses_an_inconsistent_vendor_binding(binding, refusal):
    from unitree_g1_robot.runtime_node import vendor_dds_binding

    with pytest.raises(RuntimeError, match=refusal):
        vendor_dds_binding(*binding)


@pytest.mark.parametrize("binding", [(_SIM_DOMAIN, "lo"), (_SIM_DOMAIN + 1, " lo "), (0, "eth0")])
def test_the_node_accepts_a_consistent_vendor_binding(binding):
    from unitree_g1_robot.runtime_node import vendor_dds_binding

    domain_id, interface = vendor_dds_binding(*binding)
    assert (domain_id, interface) == (int(binding[0]), binding[1].strip())


def test_the_vendor_domain_can_be_overridden_for_parallel_runs():
    """N3: the profile's domain is the default, not the only option.

    Parallel simulated deployments need distinct vendor domains; the argument
    reaches both nodes, and a value that is not a domain id fails at the launch.
    """
    override = _SIM_DOMAIN + 1
    params_by_executable = {}
    for node in _render("", vendor_domain_id=str(override)):
        if not isinstance(node, Node):
            continue
        params_by_executable.setdefault(node._Node__node_executable, []).append(_node_params(node))
    for executable in ("runtime_node", "vendor_mock_node"):
        params = params_by_executable[executable][0]
        assert params["vendor_domain_id"] == override
        assert params["vendor_network_interface"] == _PROFILE_SIM["vendor_network_interface"]
        # the overridden pair is still one the nodes accept
        assert _db((params["vendor_domain_id"], params["vendor_network_interface"])) == (
            override,
            _PROFILE_SIM["vendor_network_interface"],
        )

    with pytest.raises(RuntimeError, match="must be an integer"):
        _render("", vendor_domain_id="not-a-domain")
