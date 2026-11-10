"""Unitree G1 runtime launch entry (robot-runtime-packaging spec).

Brings the G1 up from a runtime profile alone: one bridge process that serves
the public runtime contract over the vendor DDS tier, plus — in simulated
transport — the in-repo vendor mock that stands in for the robot's own stack.

    ros2 launch unitree_g1_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [instance_id:=<id>] [initial_mode:=<mode>] \
        [network_interface:=<nic>] [peripherals:=<fragment.yaml>] \
        [runtime_network_interface:=<nic>] [preset_operator_confirmed:=true] \
        [preset_expected_fsm_id:=<fsm-id>]

The arm-preset physical path additionally requires ``preset_operator_confirmed:=true``,
a non-loopback ``runtime_network_interface`` and an expected FSM id (explicitly,
or an audited ``vendor.operator_validation.expected_fsm_id`` in the profile).
Those values are assembled into a run-scoped profile for the node (and removed
when the launch process exits); the source profile is never rewritten and no
``confirmed: true`` deployment file is required or produced.

DDS (design D2, revised by Issue #171 PR 2): the vendor tier speaks
CycloneDDS **inside the official SDK**, which binds its own cyclonedds-python
participants independently of the ROS RMW. The in-process coexistence was
verified end-to-end on this repository's CI host (vendor channels + rclpy
traffic + clean teardown, on both rmw_cyclonedds_cpp and rmw_fastrtps_cpp
deployments; see the PR 2 report), so the runtime no longer requires the
whole deployment to run a specific RMW: the launch does not set or verify
``RMW_IMPLEMENTATION`` — the deployment keeps whatever choice it already
made, and the vendor tier works under either.

The vendor tier's domain and NIC are passed to the nodes as parameters
(``vendor_domain_id``, ``vendor_network_interface``) and reach the SDK through
``UnitreeSdk.initialize`` → ``ChannelFactoryInitialize``, which builds its own
CycloneDDS domain configuration and hands it to ``Domain(id, config)``: that
explicit configuration wins over ``CYCLONEDDS_URI``, so the environment
variable is not how the vendor tier is isolated. Both transports therefore
name their binding explicitly, and both nodes refuse to start when the pair is
inconsistent (see ``vendor_dds_binding`` in the runtime node):

- simulated transport: the profile's ``sim.vendor_domain_id`` (a nonzero
  domain) on ``sim.vendor_network_interface`` (``lo``) — the mock and the
  runtime share that private domain and never join the robot's production
  domain;
- real transport: domain ``0`` on the explicitly named ``network_interface``
  (the NIC the robot is wired to, a deployment fact, never hardcoded).

``network_interface`` keeps its other job — a deployment fact for the
composed peripheral drivers and, when non-empty, the interface the deployment's
own ROS participants are bound to via ``CYCLONEDDS_URI`` (CycloneDDS
configuration only; harmless under any other RMW).

``robot_state_publisher`` renders the vendored URDF for TF only (design D6):
the D435/MID-360 mounts and RViz need the frames, but the URDF is deliberately
NOT passed to ``build_description``, so no public model is published and
consumers needing normalization metadata fail closed on the G1.

No controller manager, no ros2_control hardware component and no controller
spawner is launched: the vendor owns balance, whole-body control and the
low-level tier (design D1/D11). In real transport the composed peripheral
drivers (design D8) — the D435 camera and MID-360 lidar the vendor SDK does
not publish — come up from the profile defaults merged with the deployment
fragment, together with the fast-lio lidar odometry chain when the merged
configuration enables it. In simulated transport the contract layer's
synthetic publishers stand in for them on exactly the declared endpoints.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import json
import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import (
    resolve_profile_argument,
    resolve_sim_selection,
    sim_embodiment_actions,
)
from robot_runtime.peripherals import (
    fast_lio_nodes,
    load_peripherals_file,
    merge_peripherals,
    peripheral_nodes,
    synthetic_perception_nodes,
)
from robot_runtime.profile import load_profile
from robot_runtime.sim_platforms import is_physics_selection

#: The vendored description ships exactly one URDF (29 DoF + Dex3 hands).
URDF_NAME = "g1_29dof_with_hand_rev_1_0.urdf"


def _sim_binding(profile: dict, *, override: str = "", what: str) -> tuple[int, str]:
    """The simulated-transport vendor (domain, NIC) binding.

    Read from ``sim.vendor_domain_id`` / ``sim.vendor_network_interface`` so the
    deployment facts live with the rest of the profile instead of in this
    entry: the mock and the runtime share that private domain on loopback and
    never join the robot's production domain (domain 0).

    ``override`` (the ``vendor_domain_id`` launch argument) replaces the
    profile's default. It exists so several simulated deployments can run side
    by side without meeting on one vendor graph — the domain is what separates
    them — and the isolation rule (nonzero domain, loopback only) still applies:
    the nodes refuse any other pair before opening a channel.
    """
    sim = profile.get("sim") or {}
    domain_id = sim.get("vendor_domain_id")
    network_interface = sim.get("vendor_network_interface")
    if domain_id is None or network_interface is None:
        raise RuntimeError(
            f"profile declares no sim.vendor_domain_id/sim.vendor_network_interface ({what}); "
            "simulated transport needs the private domain and the loopback NIC"
        )
    if str(override).strip():
        try:
            domain_id = int(override)
        except ValueError as exc:
            raise RuntimeError(f"vendor_domain_id must be an integer domain id, got {override!r}") from exc
    return int(domain_id), str(network_interface)


def _cyclonedds_uri(network_interface: str) -> str:
    """Write the interface-binding CycloneDDS XML; return its ``file://`` URI.

    The minimal form the vendor documents: DDS on the named interface only.
    CycloneDDS itself fails loudly when the interface does not exist, so a
    typo names itself at startup instead of silently reaching for another NIC.
    """
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<CycloneDDS xmlns="https://cdds.io/config">\n'
        "  <Domain>\n"
        "    <General>\n"
        "      <Interfaces>\n"
        f'        <NetworkInterface name="{network_interface}"/>\n'
        "      </Interfaces>\n"
        "    </General>\n"
        "  </Domain>\n"
        "</CycloneDDS>\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="unitree_g1_cyclonedds_", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(xml)
        return f"file://{handle.name}"


_PRESET_DISABLED = "PRESET_HARDWARE_DISABLED"


def _parse_preset_confirmation(raw: str) -> bool:
    """Strict boolean for the per-run confirmation flag (default false)."""
    text = str(raw or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("", "0", "false", "no", "off"):
        return False
    raise RuntimeError(f"preset_operator_confirmed must be a boolean, got {raw!r}")


def _parse_expected_fsm_id(raw: object, *, source: str) -> int:
    """Parse a positive vendor FSM id, naming the argument on failure."""
    text = str(raw or "").strip()
    try:
        fsm_id = int(text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source} must be a positive integer FSM id, got {raw!r}") from exc
    if fsm_id <= 0:
        raise RuntimeError(f"{source} must be a positive integer FSM id, got {raw!r}")
    return fsm_id


def _preset_run_binding(profile: dict, context) -> tuple[str, int]:
    """Resolve the per-run (NIC, expected FSM) binding for the preset path.

    The confirmation is per launch invocation and never read from a persisted
    ``confirmed: true`` profile. The FSM may come from the explicit argument or
    an audited profile field; two disagreeing values are a conflict.
    """
    validation = (profile.get("vendor") or {}).get("operator_validation") or {}
    if not isinstance(validation, dict):
        raise RuntimeError("vendor.operator_validation must be a mapping")
    if not _parse_preset_confirmation(context.launch_configurations.get("preset_operator_confirmed", "false")):
        raise RuntimeError(
            f"{_PRESET_DISABLED}: on-site operator confirmation required (preset_operator_confirmed:=true)"
        )
    nic = str(context.launch_configurations.get("runtime_network_interface", "") or "").strip()
    if not nic or nic == "lo":
        raise RuntimeError(
            f"{_PRESET_DISABLED}: runtime_network_interface must name the robot-facing NIC (non-empty, not loopback)"
        )
    profile_nic = str(validation.get("network_interface", "") or "").strip()
    if profile_nic and profile_nic != nic:
        raise RuntimeError(
            "preset commissioning has conflicting network interface overrides: "
            f"runtime_network_interface={nic!r}, vendor.operator_validation.network_interface={profile_nic!r}"
        )
    configured_nic = str(context.launch_configurations.get("network_interface", "") or "").strip()
    if configured_nic and configured_nic != nic:
        raise RuntimeError(
            "preset commissioning has conflicting network interface overrides: "
            f"network_interface={configured_nic!r}, runtime_network_interface={nic!r}"
        )
    explicit_fsm = str(context.launch_configurations.get("preset_expected_fsm_id", "") or "").strip()
    profile_fsm = validation.get("expected_fsm_id")
    if explicit_fsm:
        expected_fsm = _parse_expected_fsm_id(explicit_fsm, source="preset_expected_fsm_id")
        if profile_fsm not in (None, ""):
            audited_fsm = _parse_expected_fsm_id(profile_fsm, source="vendor.operator_validation.expected_fsm_id")
            if audited_fsm != expected_fsm:
                raise RuntimeError(
                    "preset commissioning has conflicting expected FSM ids: "
                    f"preset_expected_fsm_id={expected_fsm}, "
                    f"vendor.operator_validation.expected_fsm_id={audited_fsm}"
                )
        return nic, expected_fsm
    if profile_fsm not in (None, ""):
        return nic, _parse_expected_fsm_id(profile_fsm, source="vendor.operator_validation.expected_fsm_id")
    raise RuntimeError(
        f"{_PRESET_DISABLED}: preset_expected_fsm_id is required (or an audited "
        "vendor.operator_validation.expected_fsm_id in the profile)"
    )


def _remove_run_profile(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _write_preset_run_profile(profile: dict, *, network_interface: str, expected_fsm_id: int) -> str:
    """Write the run-scoped profile the node reads; never touch the source file.

    The node's preflight reads ``vendor.operator_validation.expected_fsm_id``
    from its profile, so the per-run value is merged into a temporary copy. The
    file is removed when the launch process exits and is never a deployment
    artifact: the shipped profile stays without ``confirmed: true``.
    """
    run_profile = copy.deepcopy(profile)
    vendor = run_profile.setdefault("vendor", {})
    validation = dict(vendor.get("operator_validation") or {})
    validation.update(
        {
            "confirmed": True,
            "network_interface": network_interface,
            "expected_fsm_id": expected_fsm_id,
        }
    )
    vendor["operator_validation"] = validation
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="unitree_g1_preset_run_", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(run_profile, handle, default_flow_style=False, sort_keys=False)
        path = handle.name
    atexit.register(_remove_run_profile, path)
    return path


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("unitree_g1_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    use_sim, _sim_platform = resolve_sim_selection(
        profile,
        LaunchConfiguration("use_sim").perform(context),
        LaunchConfiguration("sim_platform").perform(context),
        LaunchConfiguration("simulated").perform(context),
        source=str(profile_path),
    )
    simulated = use_sim
    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    initial_mode = LaunchConfiguration("initial_mode").perform(context).strip()
    network_interface = LaunchConfiguration("network_interface").perform(context).strip()
    vendor_domain_override = LaunchConfiguration("vendor_domain_id").perform(context).strip()
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()

    if (profile.get("vendor") or {}).get("preset_only") is True:
        if not simulated:
            nic, expected_fsm_id = _preset_run_binding(profile, context)
            if peripherals_file:
                raise RuntimeError("preset commissioning has conflicting network/peripheral overrides")
            description = build_description(profile, use_sim=False)
            run_profile_path = _write_preset_run_profile(
                profile, network_interface=nic, expected_fsm_id=expected_fsm_id
            )
            return [
                Node(
                    package="unitree_g1_robot",
                    executable="runtime_node",
                    parameters=[
                        {
                            "profile": run_profile_path,
                            "initial_mode": initial_mode,
                            "preset_operator_confirmed": True,
                            "preset_test_transport": False,
                            "vendor_domain_id": 0,
                            "vendor_network_interface": nic,
                            "interface_description_json": json.dumps(description),
                        }
                    ],
                    output="screen",
                )
            ]
        if network_interface not in ("", "lo") or peripherals_file:
            raise RuntimeError("preset offline entry requires loopback and no peripherals")
        description = build_description(profile, use_sim=True)
        # Mock uses the full sensor fixture; the runtime exposes only this profile.
        fixture = resolve_profile_argument("unitree_g1_robot", "g1_29dof_dex3")
        vendor_domain_id, vendor_network_interface = _sim_binding(
            profile, override=vendor_domain_override, what="preset simulated transport"
        )
        sim = profile.get("sim") or {}
        initial_fsm_id = sim.get("mock_initial_fsm_id")
        if initial_fsm_id is None:
            raise RuntimeError(
                "profile declares no sim.mock_initial_fsm_id; the preset mock fixture needs the FSM it reports"
            )
        # The arm service's advertised action table is the profile's own action
        # table, not a second literal: `vendor.arm_actions` already names it.
        action_list = [str(entry["action_name"]) for entry in (profile["vendor"].get("arm_actions") or {}).values()]
        return [
            SetEnvironmentVariable("CYCLONEDDS_URI", _cyclonedds_uri(str(vendor_network_interface))),
            Node(
                package="unitree_g1_robot",
                executable="vendor_mock_node",
                parameters=[
                    {
                        "profile": str(fixture),
                        "vendor_domain_id": vendor_domain_id,
                        "vendor_network_interface": vendor_network_interface,
                        "initial_fsm_id": int(initial_fsm_id),
                        "hand_missing": "both",
                        "arm_service_status": 0,
                        "action_list": action_list,
                    }
                ],
            ),
            Node(
                package="unitree_g1_robot",
                executable="runtime_node",
                parameters=[
                    {
                        "profile": str(profile_path),
                        "initial_mode": initial_mode,
                        "preset_test_transport": True,
                        "vendor_domain_id": vendor_domain_id,
                        "vendor_network_interface": vendor_network_interface,
                        "interface_description_json": json.dumps(description),
                    }
                ],
            ),
        ]

    # D2 (revised): the vendor tier runs inside the official SDK's own
    # cyclonedds participants — independent of the ROS RMW (see the module
    # docstring for the coexistence evidence). No RMW check here.
    # D8: profile defaults merged with the deployment fragment by (type, name).
    # The same merged set feeds the interface description the runtime serves
    # and the drivers (or synthetic publishers) that serve those endpoints,
    # so the advertised surface and the composed one cannot drift apart.
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    fast_lio_config = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}

    if instance_id:
        profile["runtime"]["instance_id"] = instance_id
    # D6: the description is projected from the profile alone — the URDF is
    # never passed, so no public model is published. ``sim_platform`` selects
    # the execution identity (and the effective sensor traits) of the
    # description, so the physics branch's embodiment receives one carrying
    # the platform the launch resolved.
    description = build_description(
        profile, peripherals=peripherals, use_sim=simulated, sim_platform=_sim_platform, profile_path=profile_path
    )

    # E1: the vendor tier's domain/NIC binding is a node parameter, never a
    # CYCLONEDDS_URI side effect — the SDK builds its own CycloneDDS domain
    # configuration and hands it to Domain(id, config), which overrides the
    # environment variable. Both transports therefore name their pair
    # explicitly, and both nodes refuse to start on an inconsistent one.
    if simulated:
        vendor_domain_id, vendor_network_interface = _sim_binding(
            profile, override=vendor_domain_override, what="simulated transport"
        )
    else:
        vendor_domain_id = 0
        vendor_network_interface = network_interface
        if not vendor_network_interface:
            raise RuntimeError(
                "VENDOR_DDS_INTERFACE_REQUIRED: real transport needs network_interface:=<robot-facing NIC>; "
                "the vendor tier is bound to that NIC as a parameter, not through CYCLONEDDS_URI"
            )
        if vendor_network_interface == "lo":
            raise RuntimeError("VENDOR_DDS_ISOLATION: real transport must name the robot-facing NIC, not loopback")

    # The DDS action precedes every node so the deployment's own ROS
    # participants inherit it. It is CycloneDDS configuration only (harmless
    # under any other RMW) and says nothing about the vendor tier's binding,
    # which travels as the parameters above.
    dds_actions: list[SetEnvironmentVariable] = []
    if network_interface:
        dds_actions.append(SetEnvironmentVariable("CYCLONEDDS_URI", _cyclonedds_uri(network_interface)))

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": (
                    Path(get_package_share_directory("unitree_g1_description")) / "urdf" / URDF_NAME
                ).read_text(encoding="utf-8")
            }
        ],
    )
    runtime = Node(
        package="unitree_g1_robot",
        executable="runtime_node",
        name="unitree_g1_runtime",
        output="screen",
        parameters=[
            {
                "profile": str(profile_path),
                "initial_mode": initial_mode,
                "interface_description_json": json.dumps(description),
                "vendor_domain_id": vendor_domain_id,
                "vendor_network_interface": vendor_network_interface,
            }
        ],
    )

    if not simulated:
        return [
            *dds_actions,
            robot_state_publisher,
            runtime,
            *peripheral_nodes(peripherals, use_sim=False),
            *fast_lio_nodes(fast_lio_config, bridge_package="unitree_g1_robot", use_sim=False),
        ]

    if is_physics_selection(use_sim, _sim_platform):
        # sim-platform-registry D3: a physics platform dispatches to the
        # effective embodiment (the profile's simulation.<platform>.embodiment
        # override when declared, else the platform's registered default) —
        # the vendor mock and the synthetic data plane are the sdk stand-ins
        # and have no place on a physics platform. No G1 physics route is
        # shipped; a test platform proves the branch.
        sim_world = LaunchConfiguration("sim_world").perform(context).strip()
        spawn_pose = tuple(
            float(LaunchConfiguration(f"spawn_{axis}").perform(context)) for axis in ("x", "y", "z", "yaw")
        )
        return sim_embodiment_actions(
            profile,
            profile_path,
            str(_sim_platform),
            peripherals_file,
            instance_id,
            interface_description=description,
            world=sim_world,
            spawn_pose=spawn_pose,
        )

    # Simulated transport: the in-repo vendor mock stands in for the robot's
    # own stack on the vendor DDS tier — the runtime's D10 startup wait
    # tolerates the discovery race, so the nodes need no stagger — and the
    # contract layer's synthetic publishers stand in for the composed
    # peripheral drivers (D8) on exactly the declared camera and lidar
    # endpoints. The bridged IMU needs no stand-in: the runtime republishes
    # /imu/data from the (mock) LowState feed, so the synthetic set covers the
    # peripherals only.
    mock = Node(
        package="unitree_g1_robot",
        executable="vendor_mock_node",
        name="unitree_vendor_mock",
        output="screen",
        parameters=[
            {
                "profile": str(profile_path),
                # Same private vendor domain and loopback NIC as the runtime:
                # the two share exactly one isolated vendor graph.
                "vendor_domain_id": vendor_domain_id,
                "vendor_network_interface": vendor_network_interface,
            }
        ],
    )
    return [*dds_actions, robot_state_publisher, mock, runtime, *synthetic_perception_nodes(peripherals)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "use_sim",
                default_value="",
                description="Run simulated (true: the in-repo vendor mock) or physical (false); "
                "default: the profile's selection",
            ),
            DeclareLaunchArgument(
                "sim_platform",
                default_value="",
                description="Simulation platform (sdk); implies use_sim:=true when given alone",
            ),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Deprecated alias of use_sim (a warning names use_sim)",
            ),
            DeclareLaunchArgument(
                "sim_world",
                default_value="",
                description="Platform-neutral world hand-off: the Gazebo world name or the MuJoCo scene name "
                "(empty: the runtime starts its own minimal world/scene)",
            ),
            DeclareLaunchArgument("spawn_x", default_value="0.0", description="Robot spawn pose x (Gazebo platform)"),
            DeclareLaunchArgument("spawn_y", default_value="0.0", description="Robot spawn pose y (Gazebo platform)"),
            DeclareLaunchArgument("spawn_z", default_value="0.0", description="Robot spawn pose z (Gazebo platform)"),
            DeclareLaunchArgument(
                "spawn_yaw", default_value="0.0", description="Robot spawn yaw in radians (Gazebo platform)"
            ),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument(
                "initial_mode",
                default_value="",
                description="Initial runtime mode override (the G1 boots in idle only)",
            ),
            DeclareLaunchArgument(
                "network_interface",
                default_value="",
                description="Network interface the robot is wired to. Real transport passes it to the vendor tier "
                "as vendor_network_interface (required there) and binds the deployment's own ROS participants "
                "to it via CYCLONEDDS_URI; simulated transport always uses loopback",
            ),
            DeclareLaunchArgument(
                "vendor_domain_id",
                default_value="",
                description="Vendor DDS domain for simulated transport (empty: the profile's "
                "sim.vendor_domain_id). Must be nonzero: simulated transport only binds loopback, and "
                "both nodes refuse any other domain/NIC pair",
            ),
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...], fast_lio: {...}} "
                "(overrides profile defaults by (type, name))",
            ),
            DeclareLaunchArgument(
                "runtime_network_interface",
                default_value="",
                description="Robot-facing NIC for the physical arm-preset commissioning path "
                "(non-empty and not loopback; never persisted)",
            ),
            DeclareLaunchArgument(
                "preset_operator_confirmed",
                default_value="false",
                description="Per-run operator confirmation for physical arm-preset dispatch; false fails closed",
            ),
            DeclareLaunchArgument(
                "preset_expected_fsm_id",
                default_value="",
                description="Expected vendor FSM id for the physical arm-preset session; "
                "empty uses an audited vendor.operator_validation.expected_fsm_id",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
