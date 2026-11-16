"""MuJoCo LINK ground truth through the simulator's own kinematics (#205 D3).

The link-pose stream for MuJoCo: forward kinematics of the exact MJCF the
provider loaded — not the URDF/MoveIt model — at the simulator's live
state. ``/joint_states`` supplies the joint positions (MuJoCo ``qpos``);
``/simulator/floating_base_state`` supplies the full floating-base pose
(quaternion kept; needed only when the target link's ancestor chain
carries a free joint — resolved per link, not from joint 0, review round 1
item 17). A joint-states message and a floating-base sample
further than :data:`MAX_PAIRING_GAP_SEC` apart in sim time are never paired
(spike 1.3 measured 0-10 ms). The model is loaded with
``load_mujoco_engine_plugins`` first when it declares engine extensions,
exactly as the provider loads it. Nothing here knows a robot name, link or
offset: the caller passes the model path, the entity and the link.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from robot_runtime_sim.testing.entity_ground_truth import (
    MAX_PAIRING_GAP_SEC,
    RESET_REWIND_MARGIN_SEC,
    GroundTruthPose,
    _quat_xyzw_to_matrix,
)
from robot_runtime_sim.testing.sim_ground_truth import _SAMPLE_LIMIT

try:  # the parser is importable without the simulator; only kinematics needs it
    import mujoco
except ImportError:  # pragma: no cover - the mujoco import is guarded in tests too
    mujoco = None


def _rotation_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """The xyzw quaternion of a rotation matrix (Shepperd's branch order)."""
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        )
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        return (
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
            (rotation[2, 1] - rotation[1, 2]) / s,
        )
    if rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        return (
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
        )
    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
    return (
        (rotation[0, 2] + rotation[2, 0]) / s,
        (rotation[1, 2] + rotation[2, 1]) / s,
        0.25 * s,
        (rotation[1, 0] - rotation[0, 1]) / s,
    )


class _MjcfKinematics:
    """The loaded MJCF and the joint-name -> qpos address mapping."""

    def __init__(self, mjcf_path: str):
        if mujoco is None:
            raise RuntimeError("the mujoco python package is required for MuJoCo link ground truth")
        from sim_models.scene_compiler import load_mujoco_engine_plugins, mjcf_engine_extensions

        path = Path(mjcf_path)
        if not path.is_file():
            raise FileNotFoundError(f"the provider's MJCF not found: {path}")
        extensions = mjcf_engine_extensions(path)
        if extensions:
            load_mujoco_engine_plugins(tuple(extensions))
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.joint_qposadr = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): int(self.model.jnt_qposadr[joint_id])
            for joint_id in range(self.model.njnt)
        }
        self._chain_joints_cache: dict[str, frozenset[str]] = {}
        self._free_joint_cache: dict[str, int | None] = {}

    def free_joint_for(self, link: str) -> int | None:
        """The qpos address of the floating (free) joint on ``link``'s
        ancestor chain, or None when no ancestor carries one.

        A body with an ``mjJNT_FREE`` joint defines the floating joint for
        every body below it: the walk climbs ``body_parentid`` from the
        link and returns the FIRST free joint it meets (the nearest
        floating ancestor). This resolves a free joint that is NOT joint 0
        (a free-jointed sub-body — joint 0 may belong to an unrelated fixed
        subtree) and yields None for a link with no floating ancestor (the
        model's own root placement is then exact). Cached.
        """
        if link in self._free_joint_cache:
            return self._free_joint_cache[link]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0:
            raise KeyError(f"the MJCF has no body named {link!r}")
        address: int | None = None
        body = int(body_id)
        while body > 0:  # body 0 is the world
            first = int(self.model.body_jntadr[body])
            for joint_id in range(first, first + int(self.model.body_jntnum[body])):
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    address = int(self.model.jnt_qposadr[joint_id])
                    break
            if address is not None:
                break
            body = int(self.model.body_parentid[body])
        self._free_joint_cache[link] = address
        return address

    def chain_joints(self, link: str) -> frozenset[str]:
        """The non-free joints on the world→``link`` body path — the only
        joints whose qpos move ``link``.

        Derived from the model, not from robot knowledge: a joint off the
        path (e.g. the wheels for an arm link) cannot change the link's
        pose, so a state that does not carry it is still complete.
        """
        cached = self._chain_joints_cache.get(link)
        if cached is not None:
            return cached
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0:
            raise KeyError(f"the MJCF has no body named {link!r}")
        names: set[str] = set()
        body = int(body_id)
        while body > 0:  # body 0 is the world
            first = int(self.model.body_jntadr[body])
            for joint_id in range(first, first + int(self.model.body_jntnum[body])):
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if name:
                    names.add(name)
            body = int(self.model.body_parentid[body])
        chain = frozenset(names)
        self._chain_joints_cache[link] = chain
        return chain

    def link_pose(
        self, link: str, base_pose, joint_positions: dict[str, float], reference_body: str | None = None
    ) -> GroundTruthPose | None:
        """Run kinematics and return the link's pose, or None when a joint
        the link's chain needs is missing from ``joint_positions``.

        ``base_pose`` is ``(position[3], quat_xyzw[4])`` or None (no free
        joint: the model's own root placement is used). With
        ``reference_body`` the pose is returned RELATIVE to that body (the
        arm root, matching spike 1.3's proven composition): the free-joint
        qpos IS the root body's world pose (measured: ``mj_kinematics``
        places a free-joint body exactly at its qpos, ignoring the body's
        own ``pos`` attribute — that attribute only sets ``qpos0``), so a
        world-frame return composed through a separately-read base pose is
        also exact, but the relative return cancels the base pose entirely
        and needs no second reader. Without ``reference_body`` the absolute
        world pose is returned. The floating joint is resolved on THIS
        link's ancestor chain (``free_joint_for``): a free-jointed sub-body
        is not joint 0, and a link with no floating ancestor uses the
        model's own root placement (``base_pose`` is then unused). The
        entity (the robot model name) is NOT a body of the composed MJCF —
        only the link is validated.
        """
        qpos = self.data.qpos
        free_joint_qposadr = self.free_joint_for(link)
        if free_joint_qposadr is not None:
            if base_pose is None:
                return None
            adr = free_joint_qposadr
            position, quat_xyzw = base_pose
            qpos[adr : adr + 3] = position
            # MuJoCo free-joint qpos quaternion is w-x-y-z
            qpos[adr + 3] = quat_xyzw[3]
            qpos[adr + 4 : adr + 7] = quat_xyzw[:3]
        # the joints the link's pose depends on (the world->link chain): a
        # state missing one of them is incomplete. Joints OFF the chain that
        # /joint_states does not carry (the arm-only broadcaster publishes
        # no wheel joints) keep the model's default — they cannot move the
        # link. Every joint the state DOES carry is written into qpos.
        chain = self.chain_joints(link)
        for name, address in self.joint_qposadr.items():
            if free_joint_qposadr is not None and address == free_joint_qposadr:
                continue
            if name in joint_positions:
                qpos[address] = joint_positions[name]
            elif name in chain:
                return None  # a joint the link's pose depends on is missing
        mujoco.mj_kinematics(self.model, self.data)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0:
            raise KeyError(f"the MJCF has no body named {link!r}")
        w, x, y, z = self.data.xquat[body_id]
        if reference_body is None:
            return GroundTruthPose(0.0, self.data.xpos[body_id], (x, y, z, w))
        ref_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, reference_body)
        if ref_id < 0:
            raise KeyError(f"the MJCF has no body named {reference_body!r}")
        ref_w, ref_x, ref_y, ref_z = self.data.xquat[ref_id]
        ref_r = np.array(_quat_xyzw_to_matrix(ref_x, ref_y, ref_z, ref_w))
        rel_position = ref_r.T @ (self.data.xpos[body_id] - self.data.xpos[ref_id])
        link_r = np.array(_quat_xyzw_to_matrix(x, y, z, w))
        return GroundTruthPose(0.0, rel_position, _rotation_to_quat_xyzw(ref_r.T @ link_r))


class MujocoLinkGroundTruthStream:
    """The link-pose stream of one model on MuJoCo (FK of the loaded MJCF).

    Subscribes on a private ROS context to ``/joint_states`` and
    ``/simulator/floating_base_state``; every joint-states message is
    paired with the newest floating-base sample within
    :data:`MAX_PAIRING_GAP_SEC`, kinematics runs, and the requested body's
    world pose is recorded under the pair's sim-time stamp.
    """

    def __init__(
        self,
        mjcf_path: str,
        entity: str,
        link: str,
        env: dict[str, str] | None = None,
        reference_body: str | None = None,
    ):
        self._kinematics = _MjcfKinematics(mjcf_path)
        self._entity = str(entity)
        self._link = str(link)
        # the floating joint of THIS stream's link, resolved once at
        # construction (review round 1, item 17): a free-jointed sub-body is
        # not joint 0, and None means the model's own root placement
        # governs — no base pairing is needed for this link at all
        self._free_joint_qposadr = self._kinematics.free_joint_for(self._link)
        self._reference_body = reference_body
        self._env = env
        self._samples: list[GroundTruthPose] = []
        self._base: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]] = []
        self._lock = threading.Lock()
        # world resets (a rewound clock) recorded inline below: a reset
        # mid-run voids the run's ground truth and the consumer fails the
        # run explicitly — the history restart is never silent
        self._reset_count = 0
        self._context = None
        self._node = None
        self._executor = None
        self._thread = None

    def start(self) -> MujocoLinkGroundTruthStream:
        if self._node is not None:
            return self
        import rclpy  # noqa: PLC0415  (test-support module: no ROS import at module load)
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState

        from robot_runtime_sim.mujoco import FLOATING_BASE_STATE_TOPIC

        domain = str((self._env or {}).get("ROS_DOMAIN_ID") or os.environ.get("ROS_DOMAIN_ID", "0"))
        previous = os.environ.get("ROS_DOMAIN_ID")
        os.environ["ROS_DOMAIN_ID"] = domain
        try:
            self._context = rclpy.Context()
            rclpy.init(args=[], context=self._context)
        finally:
            if previous is None:
                os.environ.pop("ROS_DOMAIN_ID", None)
            else:
                os.environ["ROS_DOMAIN_ID"] = previous
        self._node = rclpy.create_node(
            "mujoco_link_ground_truth",
            context=self._context,
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self._node.create_subscription(Odometry, FLOATING_BASE_STATE_TOPIC, self._on_base, 100)
        self._node.create_subscription(JointState, "/joint_states", self._on_joints, 100)
        self._executor = rclpy.executors.SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        return self

    def _on_base(self, message) -> None:
        pose = message.pose.pose
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        with self._lock:
            self._base.append(
                (
                    stamp,
                    (pose.position.x, pose.position.y, pose.position.z),
                    (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                )
            )
            del self._base[:-200]

    def _on_joints(self, message) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        positions = {str(name): float(value) for name, value in zip(message.name, message.position, strict=False)}
        with self._lock:
            base_pose = None
            if self._free_joint_qposadr is not None:
                if not self._base:
                    return
                # the newest base sample within the pairing window
                for base_stamp, position, quat in reversed(self._base):
                    if abs(base_stamp - stamp) <= MAX_PAIRING_GAP_SEC:
                        base_pose = (position, quat)
                        break
                    if base_stamp < stamp - MAX_PAIRING_GAP_SEC:
                        break
                if base_pose is None:
                    return
            sample = self._kinematics.link_pose(self._link, base_pose, positions, reference_body=self._reference_body)
            if sample is None:
                return  # a missing joint or an unpaired base: never half-assembled
            sample.stamp_sec = stamp
            if self._samples and stamp <= self._samples[-1].stamp_sec:
                # a rewind beyond the margin is a world reset (the history
                # restarts and the reset is RECORDED — reset_detected: the
                # consumer fails the run explicitly); within the margin it is
                # a publisher hiccup (measured on the GUI screenshot runs)
                # and only the non-advancing sample is dropped
                if self._samples[-1].stamp_sec - stamp > RESET_REWIND_MARGIN_SEC:
                    self._samples.clear()
                    self._reset_count += 1
                else:
                    return
            self._samples.append(sample)
            if len(self._samples) > 2 * _SAMPLE_LIMIT:
                del self._samples[:-_SAMPLE_LIMIT]

    # the same reader API as the other entity streams
    def wait_for(self, n: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._samples) >= n:
                    return True
            time.sleep(0.05)
        with self._lock:
            return len(self._samples) >= n

    def samples(self) -> list[GroundTruthPose]:
        with self._lock:
            return list(self._samples)

    def latest(self) -> GroundTruthPose | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def reset_detected(self) -> bool:
        """Whether the stream's clock rewound since ``start()`` (a world
        reset): the consumer must FAIL the run explicitly — a silent
        history restart would present the post-reset span as the whole
        run (the same rule as the other entity streams)."""
        with self._lock:
            return self._reset_count > 0

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._context is not None:
            try:
                self._context.shutdown()
                self._context.try_destroy()
            except Exception:  # noqa: BLE001  (a torn-down context is success here)
                pass
            self._context = None
