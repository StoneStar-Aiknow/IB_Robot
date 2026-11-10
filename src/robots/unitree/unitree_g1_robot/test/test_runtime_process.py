"""Process-level lifecycle tests: the runtime entry's own ``main()``.

The bridge suite exercises the node in-process; these tests run the real
``runtime_node`` entry as a subprocess against the in-repo vendor mock (the
launch composition) and observe it on the public ROS contract surface and
through the mock's recorded vendor effects:

* repeated startup — a second runtime process must come up cleanly after the
  first exited, so leftovers from a previous run cannot poison the next;
* SDK initialization failure — a network interface the vendor rejects must
  exit nonzero, bounded, with the adapter's guidance, not a traceback with the
  rclpy context left initialized;
* sustained send/receive — continuous vendor streams plus repeated preset
  executions, then a SIGTERM: the process must exit 0 within a bound
  (teardown order: executor, SDK channels, node, context).

The ROS context and the spawned runtime processes use the domain the workspace
conftest reserved for THIS pytest process (``IBROBOT_TEST_ROS_DOMAIN_ID``), so
no suite carries a domain literal of its own. Every wait is bounded.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import pytest as _pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from unitree_g1_robot import vendor_gateway as _vg

if not _vg.VENDOR_AVAILABLE:
    # The runtime and the mock speak the vendor tier, which only exists with
    # the official SDK installed (setup full profile). Skip the module at
    # collection instead of failing it — mirrors the bridge suite's skip.
    _pytest.skip("unitree_sdk2py is not installed", allow_module_level=True)

from mock_process import (  # noqa: E402 - the module-level vendor-defect skip must run first
    MockProcess,
    vendor_test_domain,
)

from ibrobot_msgs.action import ExecuteNamedMotion  # noqa: E402
from ibrobot_msgs.msg import RuntimeStatus  # noqa: E402
from robot_runtime.contract import STATUS_TOPIC  # noqa: E402

#: The vendor DDS domain this pytest process owns (see the workspace conftest):
#: the observer node, the mock subprocess and every spawned runtime share it, so
#: a parallel suite cannot meet their publishers or vendor participants.
TEST_DOMAIN_ID = vendor_test_domain()

#: Bounded waits: cold DDS matching plus the startup checks, and the teardown
#: bound a SIGTERM exit must fit inside.
STARTUP_BOUND_S = 30.0
EXIT_BOUND_S = 20.0
GOAL_BOUND_S = 15.0

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PRESET_PROFILE = PACKAGE_ROOT / "profiles" / "g1_29dof_arm_presets.yaml"


class RuntimeProcess:
    """The runtime entry as a subprocess, mirroring ``MockProcess``."""

    _run_counter = 0

    def __init__(self, *, domain_id: int, extra_overrides: dict[str, Any] | None = None) -> None:
        RuntimeProcess._run_counter += 1
        self.log_file = (
            Path(tempfile.gettempdir()) / f"g1_runtime_process_{os.getpid()}_{RuntimeProcess._run_counter}.log"
        )
        overrides = {
            "profile": str(PRESET_PROFILE),
            "preset_test_transport": True,
            "startup_timeout_s": 15.0,
            "vendor_domain_id": int(domain_id),
            "vendor_network_interface": "lo",
            **(extra_overrides or {}),
        }
        argv = [sys.executable, "-m", "unitree_g1_robot.runtime_node", "--ros-args"]
        for name, value in overrides.items():
            rendered = ("true" if value else "false") if isinstance(value, bool) else str(value)
            argv += ["-p", f"{name}:={rendered}"]
        env = dict(
            os.environ,
            ROS_DOMAIN_ID=str(int(domain_id)),
            ROS_LOCALHOST_ONLY="1",
            PYTHONUNBUFFERED="1",
        )
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdout=open(self.log_file, "a"),  # noqa: SIM115 - belongs to the subprocess
            stderr=subprocess.STDOUT,
            env=env,
        )

    def alive(self) -> bool:
        return self._proc.poll() is None

    def exit_code(self, *, timeout_s: float) -> int:
        try:
            return self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)
            raise AssertionError(
                f"the runtime process did not exit within {timeout_s}s; log tail: {self._log_tail()}"
            ) from None

    def sigterm(self) -> int:
        self._proc.send_signal(signal.SIGTERM)
        return self.exit_code(timeout_s=EXIT_BOUND_S)

    def _log_tail(self) -> str:
        try:
            return self.log_file.read_text(errors="replace")[-2000:]
        except OSError:
            return "<no log>"


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Own the ROS context on this suite's private domain, if nothing else does."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init(domain_id=TEST_DOMAIN_ID)
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture(scope="module")
def mock_process():
    """One vendor mock subprocess for the module, as the bridge suite runs it."""
    spawner = rclpy.create_node("g1_process_mock_spawner")
    spin_executor = rclpy.executors.SingleThreadedExecutor()
    spin_executor.add_node(spawner)
    threading.Thread(target=spin_executor.spin, daemon=True).start()
    handle = MockProcess(
        node=spawner,
        profile=PACKAGE_ROOT / "profiles" / "g1_29dof_dex3.yaml",
        domain_id=TEST_DOMAIN_ID,
        records_file=Path(tempfile.gettempdir()) / f"g1_process_mock_{os.getpid()}.jsonl",
    )
    handle.wait_ready()
    yield handle
    handle.terminate()
    spin_executor.shutdown()
    spawner.destroy_node()


class _Observer:
    """Contract-side observation of one runtime subprocess."""

    def __init__(self) -> None:
        self.node = rclpy.create_node("g1_runtime_process_observer")
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()
        self._lock = threading.Lock()
        self._latest: RuntimeStatus | None = None
        self.node.create_subscription(RuntimeStatus, STATUS_TOPIC, self._on_status, 10)

    def _on_status(self, message: RuntimeStatus) -> None:
        with self._lock:
            self._latest = message

    def reset(self) -> None:
        """Drop the cached status so the next wait observes a fresh publisher.

        A previous runtime process's last ACTIVE message would otherwise
        satisfy the next wait instantly and let a test SIGTERM a runtime that
        is still in early startup, before its signal handlers exist.
        """
        with self._lock:
            self._latest = None

    def wait_until(self, predicate, what: str, timeout_s: float = STARTUP_BOUND_S):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            with self._lock:
                last = self._latest
            if last is not None and predicate(last):
                return last
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what} (last: {last and last.lifecycle})")

    def wait_active(self):
        return self.wait_until(lambda status: status.lifecycle == "ACTIVE", "the runtime to reach ACTIVE")

    def action_client(self):
        client = ActionClient(self.node, ExecuteNamedMotion, "/motion/execute_named")
        if not client.wait_for_server(timeout_sec=STARTUP_BOUND_S):
            raise AssertionError("the named motion server never appeared")
        return client

    def send_preset(self, client, name: str):
        goal = ExecuteNamedMotion.Goal()
        goal.name = name
        handle_future = client.send_goal_async(goal)
        deadline = time.monotonic() + GOAL_BOUND_S
        while not handle_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert handle_future.done(), f"the {name!r} goal decision timed out"
        handle = handle_future.result()
        assert handle.accepted, f"the {name!r} goal was rejected"
        result_future = handle.get_result_async()
        deadline = time.monotonic() + GOAL_BOUND_S
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert result_future.done(), f"the {name!r} result timed out"
        return result_future.result().result

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()


@pytest.fixture
def observer():
    the_observer = _Observer()
    yield the_observer
    the_observer.close()


def _preset_mock_baseline(mock_process):
    """Point the shared mock at the preset profile's vendor action table.

    The mock subprocess runs the dex3 profile, so its baseline GetActionList
    answers the dex3 table; the preset runtime only arms the action its own
    profile names, and startup drops anything the vendor does not report
    (design D7), so the mock must report the preset profile's name.
    """
    mock_process.apply({"action_list": ["both_hands_up"]})


def test_repeated_process_startup_and_clean_sigterm_exit(mock_process, observer):
    """Two sequential runtime processes: ACTIVE, SIGTERM, exit 0 — twice.

    A runtime that left DDS participants or vendor-tier readers behind would
    poison the second process's startup (Issue #207's reliable-write stall
    against a stopped reader); a clean bounded exit is what makes repeated
    startup safe.
    """
    _preset_mock_baseline(mock_process)
    for attempt in (1, 2):
        runtime = RuntimeProcess(domain_id=TEST_DOMAIN_ID)
        observer.reset()  # only THIS process's status may satisfy the wait
        assert runtime.alive(), f"attempt {attempt}: the runtime process died at spawn"
        status = observer.wait_active()
        assert status.active_mode == "idle"
        assert runtime.alive(), f"attempt {attempt}: the runtime died while serving"
        assert runtime.sigterm() == 0, f"attempt {attempt}: unclean exit; log tail: {runtime._log_tail()}"


def test_startup_failure_exits_nonzero_with_guidance():
    """A refused vendor binding must fail as one bounded nonzero exit.

    The constructor closes its facade and destroys its half-built node; main()
    releases the rclpy context and returns 1 with the binding's guidance instead
    of a traceback that leaves the context initialized. On this process's
    isolated domain the refusal comes from ``vendor_dds_binding`` (an isolated
    domain may only bind loopback), which is the same startup-failure shape the
    adapter's ``ip link`` guidance takes when the vendor itself rejects a NIC —
    that adapter message stays covered by ``unitree_g1_adapter``.
    """
    runtime = RuntimeProcess(
        domain_id=TEST_DOMAIN_ID,
        extra_overrides={
            "profile": str(PACKAGE_ROOT / "profiles" / "g1_29dof_dex3.yaml"),
            "preset_test_transport": False,
            "vendor_network_interface": "no-such-nic",
        },
    )
    exit_code = runtime.exit_code(timeout_s=STARTUP_BOUND_S)
    tail = runtime._log_tail()
    assert exit_code == 1, f"expected a clean nonzero exit, got {exit_code}; log tail: {tail}"
    assert "VENDOR_DDS_ISOLATION_REQUIRED" in tail, f"the binding's guidance is missing: {tail}"
    assert "no-such-nic" in tail, f"the refusal must name the offending interface: {tail}"
    assert "startup failed" in tail, f"the failure must be one bounded stderr line: {tail}"
    assert "Traceback" not in tail, "an initialization failure must not surface as a traceback"


def test_sustained_vendor_traffic_then_clean_exit(mock_process, observer):
    """Continuous vendor streams plus repeated preset executions, then SIGTERM.

    The mock publishes the 50 Hz vendor state stream the whole time; six
    sequential preset actions ride the vendor RPC tier and the contract
    surface answers status in between. The process must survive the sustained
    traffic (Issue #207's crash form was observed under load) and still exit 0
    within the bound afterwards.
    """
    watermark = mock_process.seq_snapshot()
    _preset_mock_baseline(mock_process)
    runtime = RuntimeProcess(domain_id=TEST_DOMAIN_ID)
    try:
        observer.wait_active()
        client = observer.action_client()
        try:
            for index in range(6):
                result = observer.send_preset(client, "arms_horizontal")
                assert result.success, f"preset {index} failed: {result.message}"
                latest = observer.wait_until(
                    lambda status: status.lifecycle == "ACTIVE", f"status after preset {index}"
                )
                assert latest.active_mode == "idle"
                assert runtime.alive(), "the runtime died under sustained traffic"
            assert mock_process.record("last_action_id", after_seq=watermark) == 15
        finally:
            client.destroy()
    finally:
        assert runtime.sigterm() == 0, f"unclean exit after sustained traffic; log tail: {runtime._log_tail()}"
