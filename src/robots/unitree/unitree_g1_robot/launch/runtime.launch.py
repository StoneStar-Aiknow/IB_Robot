"""Unitree G1 runtime launch entry (robot-runtime-packaging spec).

Brings the G1 up from a runtime profile alone: one bridge process that serves
the public runtime contract over the vendor DDS tier, plus — in simulated
transport — the in-repo vendor mock that stands in for the robot's own stack.

    ros2 launch unitree_g1_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [instance_id:=<id>] [initial_mode:=<mode>] \
        [network_interface:=<nic>] [peripherals:=<fragment.yaml>]

DDS (design D2): the vendor tier speaks CycloneDDS. The launch does NOT set
``RMW_IMPLEMENTATION``: a runtime pinned alone would strand every application
node still on the ambient implementation — service calls do not interoperate
across RMW implementations (topic data flows, request/response does not) —
and a ``SetEnvironmentVariable`` inside an included launch leaks into whatever
the parent launch does after the include, so its scope is uncontrollable. The
deployment owns the choice: in real transport the environment must already
carry ``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` — for the whole deployment,
runtime and application nodes alike — and this entry verifies that and fails
closed with a named error when it does not. Simulated transport performs no
check: the mock serves whatever RMW the ambient environment provides.

``network_interface`` names the NIC the robot is wired to — a deployment fact,
never hardcoded — and, when non-empty, is expanded into a CycloneDDS XML that
binds DDS to that one interface (``CYCLONEDDS_URI``) in either transport; when
empty, no ``CYCLONEDDS_URI`` is set and CycloneDDS keeps its default (or an
externally provided) configuration.

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

import json
import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import resolve_profile_argument, resolve_simulated
from robot_runtime.peripherals import (
    fast_lio_nodes,
    load_peripherals_file,
    merge_peripherals,
    peripheral_nodes,
    synthetic_perception_nodes,
)
from robot_runtime.profile import load_profile

#: The vendor DDS tier is CycloneDDS, so the whole deployment — this runtime
#: and every application node calling its services — must run this
#: implementation (design D2). The launch never sets it itself; in real
#: transport it verifies the environment already carries it and fails closed.
REQUIRED_RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"

#: The vendored description ships exactly one URDF (29 DoF + Dex3 hands).
URDF_NAME = "g1_29dof_with_hand_rev_1_0.urdf"


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


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("unitree_g1_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    simulated = resolve_simulated(profile, LaunchConfiguration("simulated").perform(context))
    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    initial_mode = LaunchConfiguration("initial_mode").perform(context).strip()
    network_interface = LaunchConfiguration("network_interface").perform(context).strip()
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()

    # D2: the vendor DDS tier speaks CycloneDDS. The launch never sets the
    # RMW itself (see the module docstring); in real transport it verifies the
    # deployment made the choice and fails closed when it did not.
    if not simulated:
        current = os.environ.get("RMW_IMPLEMENTATION") or "unset"
        if current != REQUIRED_RMW_IMPLEMENTATION:
            raise RuntimeError(
                f"RMW_IMPLEMENTATION is {current!r}, but the G1 vendor DDS tier speaks CycloneDDS: "
                "the whole deployment — this runtime and every application node that calls its "
                "services — must run rmw_cyclonedds_cpp. Export "
                "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp before launching; service calls do not "
                "interoperate across RMW implementations."
            )

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
    # never passed, so no public model is published.
    description = build_description(profile, peripherals=peripherals, simulated=simulated)

    # The DDS action precedes every node so the spawned processes inherit it.
    # It is CycloneDDS configuration only (harmless under any other RMW) and
    # applies in either transport.
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
        parameters=[{"profile": str(profile_path)}],
    )
    return [*dds_actions, robot_state_publisher, mock, runtime, *synthetic_perception_nodes(peripherals)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Override profile.simulated: run against the in-repo vendor mock",
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
                description="Network interface the robot is wired to; binds CycloneDDS to it via CYCLONEDDS_URI "
                "(empty: CycloneDDS default configuration)",
            ),
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...], fast_lio: {...}} "
                "(overrides profile defaults by (type, name))",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
