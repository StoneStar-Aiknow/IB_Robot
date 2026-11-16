"""A cross-process fake sim-platform launch test (openspec sim-platform-registry D10).

Proves dispatch, facade and description end-to-end ACROSS PROCESSES without a
real simulator: a throw-away ``testsim`` platform distribution (dist-info on
``PYTHONPATH``, so the launch subprocess and the facade discover it through
real ``importlib.metadata``) whose embodiment returns the SO-101 SDK-transport
stack (the production controller stack on the vendor SDK virtual bus,
``simulated=true``) plus a tiny ``/clock`` publisher (a physics selection runs
on sim time).

The runtime reaches ACTIVE with ``sim_platform: testsim`` in the facade's
public description, proving the whole chain — profile declaration, registry
selection, embodiment dispatch, facade capability/description projection —
for a third-party platform nobody hard-coded.

Uses the workspace-root conftest domain isolation like the other tests in
this package (no pytest.ini here; the root conftest applies). Runs under
``colcon test`` (never skipped by default), bounded by watchdog timeouts on
both the readiness wait and the whole case.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1]
_PROFILE = _PACKAGE / "profiles" / "so101_single_arm.yaml"

#: Wall-clock budget for the whole case: launch + readiness + query + teardown.
_WHOLE_CASE_TIMEOUT_S = 240.0

_PLATFORM_MODULE = textwrap.dedent(
    '''
    """A fake physics sim platform (so101_robot launch test, D10).

    Import-light by contract (D1): stdlib plus ``robot_runtime.sim_platforms``
    types only — building an interface description imports this module, so it
    must never import ``launch`` or a simulator package at module scope. The
    embodiment reference is resolved lazily by the dispatch entries.
    """

    from robot_runtime.sim_platforms import SimPlatform, SimSensorTraits

    SPEC = SimPlatform(
        name="testsim",
        package="testsim_platform",
        embodiment="testsim_embodiment:EMBODIMENT",
        world=None,
        scene_task=None,
        sensor_traits=SimSensorTraits(),
    )
    '''
)

#: The fake embodiment: the SO-101 SDK-transport stack (the hardware/sdk body
#: of ``runtime_stack_actions``: RSP + ros2_control + spawners + facade) on
#: ``simulated=true``, plus a tiny ``/clock`` publisher (a physics selection
#: runs on sim time, and the facade's readiness clock needs one).
_EMBODIMENT_MODULE = textwrap.dedent(
    '''
    """The testsim platform's embodiment: the SO-101 SDK-transport stack."""

    from launch_ros.actions import Node

    from robot_runtime.launch_support import (
        profile_controllers,
        render_robot_description,
        runtime_facade_node,
        spawner_nodes,
        write_controller_manager_params,
    )
    from robot_runtime.path_subst import resolve_path
    from so101_robot_launch_test_clock import CLOCK_PUBLISHER_NODE


    class TestsimEmbodiment:
        """The SDK-transport stack reused as a fake physics embodiment (D3)."""

        sensor_traits = None  # the platform's default traits apply

        def render_description(
            self, profile, profile_path, *, peripherals=None,
            spawn_pose=(0.0, 0.0, 0.0, 0.0), world="",
        ):
            return render_robot_description(profile, profile_path, True)

        def actions(
            self, profile, profile_path, peripherals_file="", instance_id="",
            *, robot_description=None, interface_description=None,
            world="", spawn_pose=(0.0, 0.0, 0.0, 0.0),
        ):
            robot_description = robot_description or self.render_description(
                profile, profile_path
            )
            ros2_control = profile.get("ros2_control") or {}
            controllers_config = resolve_path(ros2_control["controllers_config"])
            cm_params = write_controller_manager_params(robot_description)
            broadcasters, active, inactive = profile_controllers(profile, "testsim")
            return [
                Node(
                    package="robot_state_publisher",
                    executable="robot_state_publisher",
                    output="screen",
                    parameters=[{"robot_description": robot_description}],
                ),
                Node(
                    package="controller_manager",
                    executable="ros2_control_node",
                    parameters=[cm_params, controllers_config],
                    remappings=[("~/robot_description", "/robot_description")],
                    output="screen",
                ),
                *spawner_nodes(
                    broadcasters, active, inactive,
                    controller_manager=str(profile["controller_manager"]),
                ),
                runtime_facade_node(
                    profile,
                    profile_path,
                    peripherals_file,
                    use_sim=True,
                    sim_platform="testsim",
                    instance_id=instance_id,
                    interface_description=interface_description,
                    use_sim_time=True,
                ),
                CLOCK_PUBLISHER_NODE,
            ]


    EMBODIMENT = TestsimEmbodiment()
    '''
)

#: A tiny sim-time clock publisher: a physics selection's nodes (facade,
#: controllers) run on ``/clock``; without a publisher the readiness wait
#: would hang on an unstamped clock.
_CLOCK_PUBLISHER_SCRIPT = textwrap.dedent(
    """
    #!/usr/bin/env python3
    import rclpy
    from rosgraph_msgs.msg import Clock

    rclpy.init()
    node = rclpy.create_node("testsim_clock_publisher")
    pub = node.create_publisher(Clock, "/clock", 10)

    def tick():
        msg = Clock()
        msg.clock = node.get_clock().now().to_msg()
        pub.publish(msg)

    node.create_timer(0.02, tick)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    """
)

_CLOCK_PUBLISHER_MODULE = textwrap.dedent(
    '''
    """The testsim /clock publisher as a launch action (module-scope ExecuteProcess).

    A script in the test's distribution dir, not a package console script:
    launch's Node executable lookup would demand an installed so101_robot
    executable, and the point of this platform is that it installs nothing.
    """

    import sys
    from pathlib import Path

    from launch.actions import ExecuteProcess

    _SCRIPT = Path(__file__).resolve().parent / "bin" / "so101_robot_test_clock"

    CLOCK_PUBLISHER_NODE = ExecuteProcess(
        cmd=[sys.executable, str(_SCRIPT)],
        output="log",
        additional_env={"PYTHONUNBUFFERED": "1"},
    )
    '''
)


def _write_platform_distribution(root: Path) -> None:
    """A throw-away distribution registering the ``testsim`` platform.

    ``<root>/testsim_platform.py`` is the import-light spec module;
    ``<root>/testsim_embodiment.py`` the embodiment (imported lazily through
    the spec's ``module:attr`` reference); ``<root>/bin`` the console script
    the /clock publisher Node names. ``<root>/testsim_platform-0.0.0.dist-info``
    carries the entry-point metadata ``importlib.metadata`` discovers. Putting
    ``root`` (and ``root/bin``) on ``PYTHONPATH`` of the launch subprocess is
    the whole plugin install.
    """
    (root / "testsim_platform.py").write_text(_PLATFORM_MODULE, encoding="utf-8")
    (root / "testsim_embodiment.py").write_text(_EMBODIMENT_MODULE, encoding="utf-8")
    (root / "so101_robot_launch_test_clock.py").write_text(_CLOCK_PUBLISHER_MODULE, encoding="utf-8")

    dist_info = root / "testsim_platform-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: testsim_platform\n"
        "Version: 0.0.0\n"
        "Summary: A fake ibrobot sim platform for the so101 launch test\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[ibrobot.sim_platforms]\ntestsim = testsim_platform:SPEC\n",
        encoding="utf-8",
    )

    script = root / "bin" / "so101_robot_test_clock"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_CLOCK_PUBLISHER_SCRIPT, encoding="utf-8")
    script.chmod(0o755)


def _testsim_profile(tmp_path: Path) -> Path:
    """A copy of the shipped profile declaring ``testsim``."""
    import yaml

    profile = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))
    profile["sim_platforms"] = ["sdk", "gazebo", "mujoco", "testsim"]
    path = tmp_path / "so101_testsim.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _launch_environment(root: Path) -> dict[str, str]:
    """The launch subprocess environment: this test's domain, the dist on PYTHONPATH."""
    env = dict(os.environ)
    # the root conftest reserved a fresh ROS domain for THIS pytest process;
    # the launch subprocess must join it (never the default domain 0)
    env["ROS_DOMAIN_ID"] = env.get("ROS_DOMAIN_ID", "0")
    env["ROS_LOCALHOST_ONLY"] = "1"
    # make the fake platform distribution discoverable (import + entry points
    # + the console script the /clock publisher Node names)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), str(root / "bin"), env.get("PYTHONPATH", "")) if p)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("IBROBOT_REQUIRE_SIM_PLATFORMS", None)
    return env


def _kill_process_group(process: subprocess.Popen) -> None:
    """Always kill the whole launch process group (children included)."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def _watchdog(process: subprocess.Popen, budget_s: float) -> threading.Timer:
    """A hard wall-clock bound: kill the process group when the budget is gone."""

    def fire():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    timer = threading.Timer(budget_s, fire)
    timer.daemon = True
    return timer


def test_a_fake_platform_launch_reaches_active_across_processes(tmp_path):
    """The real fake-platform run (D10): ``ros2 launch so101_robot
    runtime.launch.py profile:=<testsim copy> sim_platform:=testsim`` in a
    subprocess reaches ACTIVE with ``sim_platform: testsim`` in the facade's
    public description, the platform discovered only through real entry
    points (dist-info on PYTHONPATH)."""
    root = tmp_path
    _write_platform_distribution(root)
    profile = _testsim_profile(root)
    env = _launch_environment(root)

    command = [
        "ros2",
        "launch",
        "so101_robot",
        "runtime.launch.py",
        f"profile:={profile}",
        "sim_platform:=testsim",
        "display:=false",
    ]
    log = root / "launch.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(_PACKAGE),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group: teardown kills them all
        )
    watchdog = _watchdog(process, _WHOLE_CASE_TIMEOUT_S)
    watchdog.start()
    try:
        # the readiness wait runs in a SECOND process on the same domain: the
        # facade must publish RuntimeStatus with lifecycle ACTIVE and the
        # testsim platform identity
        try:
            wait = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robot_runtime.wait_for_runtime",
                    "--runtime-name",
                    "so101_robot",
                    "--timeout",
                    "150",
                ],
                cwd=str(_PACKAGE),
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"the so101 testsim runtime never became ready within 180 s\n"
                f"launch log tail:\n{log.read_text(encoding='utf-8')[-2000:]}"
            )
        output = wait.stdout + wait.stderr
        assert wait.returncode == 0, (
            f"the so101 testsim runtime did not reach readiness:\n{output[-2000:]}\n"
            f"launch log tail:\n{log.read_text(encoding='utf-8')[-2000:]}"
        )
        assert "Runtime ready: so101_robot" in output

        # the facade's public description reports the fake platform: the
        # exported snapshot carries sim_platform testsim with a valid digest
        description_path = root / "description.yaml"
        wait2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_runtime.wait_for_runtime",
                "--runtime-name",
                "so101_robot",
                "--timeout",
                "30",
                "--description-output",
                str(description_path),
            ],
            cwd=str(_PACKAGE),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert wait2.returncode == 0, f"status query failed:\n{wait2.stdout}\n{wait2.stderr}"
        import yaml

        # the snapshot is the description document itself (write_description
        # dumps it top-level); its identity block is nested under "robot"
        description = yaml.safe_load(description_path.read_text(encoding="utf-8"))
        assert description["robot"]["runtime_name"] == "so101_robot"
        assert description["sim_platform"] == "testsim"
        assert description["execution"] == "simulated"
    finally:
        # always kill the whole process group (launch children included)
        _kill_process_group(process)
        watchdog.cancel()


def test_the_fake_platform_distribution_registers_through_real_entry_points(tmp_path):
    """The distribution alone (no injection): ``importlib.metadata`` finds it.

    A cheap companion that fails fast (no launch, no processes) when the
    dist-info layout breaks — the cross-process test above then fails for a
    real reason instead of "platform not installed".
    """
    root = tmp_path
    _write_platform_distribution(root)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from robot_runtime.sim_platforms import clear_sim_platform_cache, discover_sim_platforms; "
        "clear_sim_platform_cache(); platforms = discover_sim_platforms(); "
        "assert 'testsim' in platforms, platforms; "
        "print(platforms['testsim'].name)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"discovery failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip() == "testsim"
