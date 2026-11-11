"""CPU test doubles for the tests of this package and of task_dispatch (not part of the runtime).

Test support only (like the rest of ``robot_runtime_sim.testing``): no runtime launch path
imports it. Other packages' tests import it as ``robot_runtime_sim.testing.direct_control_fakes``.

Two simulators that share nothing with any real platform or with each other:

* ``DualArmPull``: two 6-DOF arms with grippers, dt 0.04, the platform owns the loop and
  asks for each tick (platform pull). Native keys ``arm/<g>/q``, ``arm/<g>/grip``.
* ``SingleArmStep``: one 7-DOF arm, no gripper, dt 0.02, the host calls ``step`` (host step).
  Native observation is a flat ``qpos`` vector plus ``tcp`` list.

Both use a toy forward kinematics (the first three joints are the flange x, y, z) so IK is
exact and pose commands can be checked end to end.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_runtime_sim.direct_control.engine import Command, MotionConfig, SimulationCore
from robot_runtime_sim.direct_control.ports import (
    CLOCK_HOST_STEP,
    CLOCK_PLATFORM_PULL,
    AdapterDescriptor,
    GroupSpec,
    IkReply,
    IkRequest,
    ObservationFrame,
    PlatformResult,
)

FAST = MotionConfig(min_motion_ticks=2, max_motion_ticks=20, settle_ticks=1, gripper_ticks=2, idle_wait_s=0.01)


def dual_arm_descriptor(**overrides) -> AdapterDescriptor:
    groups = tuple(
        GroupSpec(name=g, joints=tuple(f"{g}_j{i}" for i in range(1, 7)), gripper=f"{g}_grip") for g in ("a", "b")
    )
    kwargs = dict(
        platform="fake_pull",
        clock_mode=CLOCK_PLATFORM_PULL,
        dt_s=0.04,
        groups=groups,
        supported_commands=frozenset({"joint", "gripper", "pose", "ik"}),
        ik="platform",
    )
    kwargs.update(overrides)
    return AdapterDescriptor(**kwargs)


def single_arm_descriptor(**overrides) -> AdapterDescriptor:
    kwargs = dict(
        platform="fake_step",
        clock_mode=CLOCK_HOST_STEP,
        dt_s=0.02,
        groups=(GroupSpec(name="arm", joints=tuple(f"j{i}" for i in range(7))),),
        supported_commands=frozenset({"joint", "pose"}),
        ik="platform",
    )
    kwargs.update(overrides)
    return AdapterDescriptor(**kwargs)


class ToyEnv:
    """Kinematic simulator: targets are reached in one step; flange xyz = first three joints."""

    def __init__(self, descriptor: AdapterDescriptor, step_lim: int = 500):
        self.descriptor = descriptor
        self.step_lim = step_lim
        self.q = {g.name: [0.1] * g.dof for g in descriptor.groups}
        self.grip = {g.name: 1.0 for g in descriptor.groups if g.gripper}
        self.tick = 0
        self.steps: list[Any] = []
        self.ended = False
        self.session_mark = 0

    def flange(self, group: str) -> list[float]:
        q = self.q[group]
        return [q[0], q[1], q[2], 1.0, 0.0, 0.0, 0.0]

    def solve_ik(self, request: IkRequest) -> IkReply:
        dof = self.descriptor.group(request.group).dof
        x, y, z = request.pose[:3]
        if abs(x) > 5.0:
            return IkReply(request.request_id, False, message="out of reach")
        return IkReply(request.request_id, True, joints=(x, y, z, *([0.0] * (dof - 3))))

    def reset(self, seed=None):
        self.tick = 0
        return self.native()

    def apply(self, targets: Mapping[str, Mapping[str, Any]]) -> None:
        for group, entry in targets.items():
            self.q[group] = list(entry["joints"])
            if "gripper" in entry:
                self.grip[group] = float(entry["gripper"])
        self.tick += 1
        self.steps.append({g: dict(e) for g, e in targets.items()})

    def done(self) -> bool:
        return self.ended or self.tick >= self.step_lim

    def ticks(self) -> int:
        return self.tick

    def native_result(self) -> PlatformResult:
        return PlatformResult(native_success=False, native_reason="timeout", take_action_cnt=self.tick)


class DualArmCodec:
    """Native format of the pull fake: one key per group and quantity."""

    def __init__(self, descriptor: AdapterDescriptor):
        self.descriptor = descriptor

    def decode_observation(self, native, *, session_id: str, generation: int) -> ObservationFrame:
        groups = [g.name for g in self.descriptor.groups]
        return ObservationFrame(
            session_id=session_id,
            generation=generation,
            step_id=int(native["t"]),
            joints={g: native[f"arm/{g}/q"] for g in groups},
            grippers={g: native[f"arm/{g}/grip"] for g in groups},
            ee_poses={g: native[f"arm/{g}/flange"] for g in groups},
            images={cam: native[f"img/{cam}"] for cam in self.descriptor.cameras if f"img/{cam}" in native},
        )

    def encode_action(self, targets):
        return {f"arm/{g}/cmd": [*e["joints"], e.get("gripper", 0.0)] for g, e in targets.items()}


class DualArmEnv(ToyEnv):
    def native(self):
        out: dict[str, Any] = {"t": self.tick}
        for cam in self.descriptor.cameras:  # a tiny frame whose pixels encode the session and tick
            import numpy as np

            out[f"img/{cam}"] = np.full((2, 2, 3), (self.session_mark, self.tick % 256, 0), dtype=np.uint8)
        for g in self.q:
            out[f"arm/{g}/q"] = list(self.q[g])
            out[f"arm/{g}/grip"] = self.grip[g]
            out[f"arm/{g}/flange"] = self.flange(g)
        return out

    def step(self, native_action):
        targets = {}
        for key, values in native_action.items():
            group = key.split("/")[1]
            targets[group] = {"joints": values[:-1], "gripper": values[-1]}
        self.apply(targets)
        return self.native()


class SingleArmCodec:
    """Native format of the step fake: flat qpos and a separate tcp entry."""

    def __init__(self, descriptor: AdapterDescriptor):
        self.descriptor = descriptor

    def decode_observation(self, native, *, session_id: str, generation: int) -> ObservationFrame:
        return ObservationFrame(
            session_id=session_id,
            generation=generation,
            step_id=int(native["frame"]),
            joints={"arm": native["qpos"]},
            ee_poses={"arm": native["tcp"]},
        )

    def encode_action(self, targets):
        return list(targets["arm"]["joints"])


class SingleArmEnv(ToyEnv):
    def native(self):
        return {"frame": self.tick, "qpos": list(self.q["arm"]), "tcp": self.flange("arm")}

    def step(self, native_action):
        self.apply({"arm": {"joints": list(native_action)}})
        return self.native()


class PullPlatform:
    """Plays a platform main loop that calls the runtime once per tick (platform pull)."""

    WRITER = "fake_pull"

    def __init__(self, core: SimulationCore, codec, env: ToyEnv, step_delay_s: float = 0.0):
        self.core, self.codec, self.env = core, codec, env
        self.step_delay_s = step_delay_s
        self.idle_polls = 0
        self.stop = threading.Event()
        core.claim_writer(self.WRITER)
        ident = core.reset()
        self.session_id, self.generation = ident["session_id"], ident["generation"]
        core.begin_session({"task": "toy", "instruction": "do it", "step_lim": env.step_lim})
        self.push()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def push(self):
        self.core.on_observation(
            self.codec.decode_observation(self.env.native(), session_id=self.session_id, generation=self.generation)
        )

    def loop(self):
        replies = []
        while not self.stop.is_set() and not self.env.done():
            decision = self.core.next_tick(self.WRITER, self.session_id, self.generation, replies)
            replies = [self.env.solve_ik(r) for r in decision.ik_requests]
            if decision.targets is None:
                self.idle_polls += 1
                continue
            time.sleep(self.step_delay_s)
            self.env.step(self.codec.encode_action(decision.targets))
            self.push()

    def start(self):
        self.thread.start()
        return self

    def join(self):
        self.stop.set()
        self.thread.join(timeout=5.0)


def submit_async(core: SimulationCore, cmd: Command, timeout_s: float = 5.0):
    box: dict[str, Any] = {}

    def run():
        box["result"] = core.submit(cmd, timeout_s)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


TOY_PROFILE = Path(__file__).resolve().with_name("direct_control_toy_dual_arm.yaml")  # installed as package data


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    return wait_for(predicate, timeout_s)


class ToyAdapter:
    """Platform-pull adapter whose platform is an in-process kinematic toy (profile ``toy_dual_arm``)."""

    overlay_cameras = ()

    def __init__(self, profile_path=None, step_delay_s: float = 0.02):
        import yaml

        from robot_runtime_sim.direct_control.ports import descriptor_from_profile

        profile = yaml.safe_load(Path(profile_path or TOY_PROFILE).read_text(encoding="utf-8"))
        self.descriptor = descriptor_from_profile(profile, platform="toy", clock_mode=CLOCK_PLATFORM_PULL)
        self.step_delay_s = step_delay_s
        self.env = None
        self.platform = None

    def start(self, core, _logger):
        self.env = DualArmEnv(self.descriptor, step_lim=100000)
        self.platform = PullPlatform(core, DualArmCodec(self.descriptor), self.env, step_delay_s=self.step_delay_s)
        self.platform.start()

    def stop(self):
        if self.platform is not None:
            self.platform.join()


def attach_toy_gate(node) -> None:
    """An ``extensions`` entry for the node tests: holds motion until ``node.toy_gate_open`` has the generation.

    It is the shape an evaluation layer's admission takes, without any evaluation concept.
    """
    node.toy_gate_open = set()
    node.core.set_motion_gate(lambda generation: "" if generation in node.toy_gate_open else "held by the toy gate")


def __getattr__(name):
    if name == "SimulationRuntime":  # lazy: keeps the pure doubles importable without rclpy
        from robot_runtime_sim.direct_control.node import SimulationRuntime

        return SimulationRuntime
    raise AttributeError(name)
