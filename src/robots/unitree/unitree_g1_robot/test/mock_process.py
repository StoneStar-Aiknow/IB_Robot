"""The vendor mock as a SUBPROCESS (the launch composition, in tests).

Running the mock and the runtime in ONE process shares the vendor binding's
cyclonedds library between the mock's writers and the runtime's readers; the
vendor's ``ddspy_write`` has been observed blocking with the GIL held during
same-process delivery, freezing executors, startup checks and teardown.
Cross-process delivery cannot share that state, so the e2e suites run the
mock exactly the way the launch entry does: as its own process speaking the
vendor wire.

The handle mirrors the in-process mock's test interface:

- constructor fault-injection parameters go to the subprocess as CLI
  parameter overrides (``--ros-args -p name:=value``);
- ``set_parameters`` rides the ROS parameter services;
- ``record(name)`` reads the mock's JSON-Lines records file (written by the
  mock on every recorded vendor request);
- ``terminate()`` stops it (SIGTERM -> the node's clean shutdown path).

The helper only spawns when the caller could run the in-process mock anyway:
it requires ``sys.executable`` to be able to import the vendor SDK, which the
suites' module-level skip guards already demand.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

MOCK_NODE_NAME = "unitree_vendor_mock"

#: The workspace conftest reserves one ROS domain per pytest process and records
#: it here. A simulated deployment's vendor tier uses that same number, so two
#: suites running in parallel never meet on one vendor graph. Tests read this
#: instead of naming a domain of their own.
VENDOR_DOMAIN_ENV = "IBROBOT_TEST_ROS_DOMAIN_ID"


def vendor_test_domain() -> int:
    """The vendor DDS domain owned by this pytest process.

    Raises instead of defaulting: a missing or zero domain means the run has no
    isolation, and a shared or production domain is exactly what these suites
    must not use. Run through ``colcon test`` (or ``pytest`` from the workspace
    root) so the root conftest allocates one per process.
    """
    raw = os.environ.get(VENDOR_DOMAIN_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"{VENDOR_DOMAIN_ENV} is not set: these tests need the per-process domain the workspace "
            "conftest allocates (run them through colcon test, or pytest from the workspace root)"
        )
    domain = int(raw)
    if domain == 0:
        raise RuntimeError(f"{VENDOR_DOMAIN_ENV}={domain} is the robot's production domain, not a test domain")
    return domain


def _profile_vendor(profile: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - test helper

    with open(profile, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["vendor"]


class MockProcess:
    """One vendor mock subprocess plus its cross-process test surface."""

    def __init__(
        self,
        *,
        node: Any,
        profile: Path,
        domain_id: int,
        records_file: Path,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.records_file = Path(records_file)
        self._records_file = self.records_file
        overrides: list[str] = [
            f"profile:={Path(profile)}",
            f"vendor_domain_id:={int(domain_id)}",
            "vendor_network_interface:=lo",
            f"records_file:={self._records_file}",
        ]
        for name, value in (params or {}).items():
            overrides.append(f"{name}:={self._cli_value(value)}")
        argv = [sys.executable, "-m", "unitree_g1_robot.vendor_mock_node", "--ros-args"]
        for override in overrides:
            argv += ["-p", override]
        env = dict(os.environ, ROS_DOMAIN_ID=str(int(domain_id)), PYTHONUNBUFFERED="1")
        self._log = self._records_file.parent / "mock_stdout.log"
        self._last_argv = argv
        self._last_env = env
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdout=open(self._log, "w"),  # noqa: SIM115 - the handle belongs to the subprocess
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._set_parameters_client = node.create_client(SetParameters, f"/{MOCK_NODE_NAME}/set_parameters")
        self.spawner_node = node
        self.spawner_node = node
        from unitree_g1_robot import projection  # noqa: PLC0415 - test helper

        self._baseline = {
            "hand_missing": "",
            "publish_lowstate": True,
            "silent_api_ids": [],
            "reject_fsm_ids": [],
            "mode_machine": int(_profile_vendor(profile)["mode_machine"]),
            "action_list": [str(entry["action_name"]) for entry in _profile_vendor(profile)["arm_actions"].values()],
            "initial_fsm_id": int(projection.FSM_IDS["zero_torque"]),
        }

    @staticmethod
    def _cli_value(value: Any) -> str:
        """One parameter override in rclpy's YAML-ish CLI syntax."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list | tuple):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return str(value)

    def alive(self) -> bool:
        return self._proc.poll() is None

    def ensure_alive(self, *, node: Any, profile: Path, domain_id: int, params: dict[str, Any] | None = None) -> bool:
        """Respawn the subprocess if it died; returns True when it was restarted.

        The vendor binding has been observed crashing its process under
        sustained traffic plus reader churn; a respawn contains the blast
        radius to the tests running while it was down. Callers re-apply their
        fault parameters afterwards.
        """
        if self.alive():
            return False
        self._proc.wait()
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            self._last_argv,
            stdout=open(self._log, "a"),  # noqa: SIM115 - the handle belongs to the subprocess
            stderr=subprocess.STDOUT,
            env=self._last_env,
        )
        self._set_parameters_client = node.create_client(SetParameters, f"/{MOCK_NODE_NAME}/set_parameters")
        self.spawner_node = node
        self.spawner_node = node
        self.wait_ready()
        self.apply(params or {}, baseline=True)
        return True

    def wait_ready(self, *, timeout_s: float = 20.0) -> None:
        """Wait for the parameter service; the records file is touched at construction."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._set_parameters_client.service_is_ready() and self._records_file.exists():
                return
            if not self.alive():
                raise RuntimeError(
                    "the vendor mock subprocess exited during startup; see "
                    f"{self._log}: {self._log.read_text(errors='replace')[-2000:]}"
                )
            time.sleep(0.05)
        raise RuntimeError(f"the vendor mock subprocess never became ready within {timeout_s}s")

    def set_parameters(self, parameters: list[Parameter]) -> None:
        """Apply fault-injection parameters through the ROS parameter services."""
        request = SetParameters.Request()
        request.parameters = [parameter.to_parameter_msg() for parameter in parameters]
        future = self._set_parameters_client.call_async(request)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert future.done(), "the mock's set_parameters call never completed"
        results = future.result().results
        rejected = [r.reason for r in results if not r.successful]
        assert not rejected, f"the mock rejected parameters: {rejected}"

    def seq_snapshot(self) -> int:
        """The current highest record sequence number (a watermark for readers)."""
        return self._read_entries((-1, ""))[0]

    def record(self, name: str, *, after_seq: int = 0) -> Any:
        """The latest recorded payload for ``name`` past ``after_seq``, or None.

        The watermark lets a harness that shares a module-scoped mock see only
        the records produced by ITS test.
        """
        _, value = self._read_entries((name, after_seq))
        return value

    def _read_entries(self, query: tuple) -> tuple:
        """Scan the JSON-Lines file; returns (max_seq, last matching value).

        ``query`` is either ``(-1, "")`` (max seq only) or ``(name, after_seq)``.
        """
        name, after_seq = query
        max_seq = 0
        found: Any = None
        if not self._records_file.exists():
            return max_seq, found
        try:
            with open(self._records_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn partial line: skip, the next stands
                    seq = int(entry.get("seq", 0))
                    max_seq = max(max_seq, seq)
                    if name and entry.get("name") == name and seq > after_seq:
                        found = entry.get("value")
        except OSError:
            return max_seq, found
        return max_seq, found

    def apply(self, overrides: dict[str, Any], *, baseline: bool = True) -> None:
        """Set mock parameters; the profile-aware baseline runs first unless
        suppressed, so a module-scoped mock starts every test from a known
        state (tests that override construction-time values rely on this)."""
        merged = dict(self._baseline) if baseline else {}
        merged.update(overrides)
        if not merged:
            return
        self.set_parameters([Parameter(name, value=value) for name, value in merged.items()])

    def terminate(self) -> None:
        """SIGTERM the subprocess (its own signal handler closes the vendor tier)."""
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5.0)
