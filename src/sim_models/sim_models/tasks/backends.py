"""Platform backends for the pick_banana scene task (so101-mujoco-runtime-platform D9).

A backend owns everything platform-specific: placing the objects (the shared
randomisation index produces identical placements on both platforms) and
reporting live object positions for evaluation. Platform message packages
(``mujoco_ros2_control_msgs``, ``ros_gz_interfaces``, ``tf2_msgs``) are
imported lazily inside the backend classes, so importing this module — and
running the node on the other platform — never requires them.

Readiness is revalidated on every call (:meth:`PickBananaBackend.check_ready`)
rather than latched at startup: the task refuses to move anything unless the
backend can place right now. Gazebo uses only the bridged ``SetEntityPose``
service (the T0 spike chose the bridge; there is no CLI fallback).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from sim_models.scene_compiler import (
    _layout_dynamic_objects,
    _quat_from_extrinsic_xyz,
    randomization_grid,
)

# Evaluation treats an object pose older than this as unavailable (D9).
POSE_STALE_S = 1.0


def _wait_done(done: threading.Event, timeout_s: float, stop_event=None) -> bool:
    """Wait for an RPC completion event; False on timeout or stop (sliced)."""
    if stop_event is None:
        return done.wait(timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if done.wait(timeout=min(remaining, 0.1)):
            return True
        if stop_event.is_set():
            return False


class PickBananaBackend:
    """Shared placement/observation contract over the scene layout (D9).

    Placements come from one generator — ``scene_compiler.randomization_grid``
    — so index *k* is the same banana pose on every platform: MuJoCo resets to
    keyframe ``random_{k:03d}`` (injected from the same grid) and Gazebo sets
    the entity pose from ``grid[k]`` directly. ``index=None`` means the layout
    default placement (keyframe ``default`` on MuJoCo).
    """

    #: Objects the backend must observe freshly to be ready.
    REQUIRED_OBJECTS: tuple[str, ...] = ()

    def __init__(self, node, layout: dict, *, callback_group) -> None:
        self._node = node
        self._cbg = callback_group
        layout = layout or {}
        self._grid = randomization_grid(layout)
        self._dynamic = {name: (position, euler) for name, position, euler in _layout_dynamic_objects(layout)}
        self._startup_ready = False

    @property
    def grid_size(self) -> int:
        return len(self._grid)

    def check_ready(self) -> tuple[bool, str]:
        """Live readiness revalidation; called before every task operation."""
        raise NotImplementedError

    def wait_ready(self, timeout_s: float) -> tuple[bool, str]:
        """Bounded startup initialisation: poll the live check until ready."""
        deadline = time.monotonic() + timeout_s
        result = (False, "initialisation timed out")
        while time.monotonic() < deadline:
            result = self.check_ready()
            if result[0]:
                self._startup_ready = True
                return result
            time.sleep(0.1)
        return result

    def start_readiness(self, timeout_s: float = 10.0) -> None:
        """Run the bounded initialisation on a daemon thread and log it (D9)."""

        def _wait() -> None:
            ok, message = self.wait_ready(timeout_s)
            state = "ready" if ok else f"NOT ready ({message})"
            self._node.get_logger().info(f"{type(self).__name__}: {state}")

        threading.Thread(target=_wait, daemon=True, name=f"{type(self).__name__}-readiness").start()

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        raise NotImplementedError

    def object_position(self, name: str) -> np.ndarray | None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def _banana_pose(self, index: int | None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Banana placement (position, w-x-y-z quaternion) for a grid index or the default."""
        if index is None:
            position, euler = self._dynamic["banana"]
            return tuple(position[:3]), _quat_from_extrinsic_xyz(*euler)  # type: ignore[return-value]
        x, y, z, yaw = self._grid[index]
        return (x, y, z), _quat_from_extrinsic_xyz(0.0, 0.0, yaw)

    def _plate_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        position, euler = self._dynamic["plate"]
        return tuple(position[:3]), _quat_from_extrinsic_xyz(*euler)  # type: ignore[return-value]


# ─── MuJoCo ───────────────────────────────────────────────────────────────────


class ShadowSim:
    """A local MuJoCo instance mirroring the live simulation (mujoco-only).

    The plugin does not publish free-body poses, so the task observes the
    banana through this shadow: arm joints are fed from ``/joint_states`` and
    the shadow steps the same composed model, so pushed objects roughly track
    the real simulation. Loading the exact composed model (robot + scene with
    the injected keyframes) is essential — a robot-less compile could never
    see the gripper pushing the banana.
    """

    def __init__(self, model_path: str) -> None:
        import mujoco as mj  # lazy: mujoco-only dependency

        self._mj = mj
        self._model = mj.MjModel.from_binary_path(model_path)
        self._data = mj.MjData(self._model)
        self._lock = threading.Lock()
        self._running = True
        self._dt = float(self._model.opt.timestep)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ShadowSim")
        self._thread.start()

    def reset_to_keyframe(self, name: str) -> bool:
        kid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            return False
        with self._lock:
            self._mj.mj_resetDataKeyframe(self._model, self._data, kid)
        return True

    def set_joint_pos(self, joint_name: str, value: float) -> None:
        jid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return
        with self._lock:
            self._data.qpos[self._model.jnt_qposadr[jid]] = value

    def get_body_pos(self, body_name: str) -> np.ndarray | None:
        bid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None
        with self._lock:
            return self._data.xpos[bid].copy()

    def shutdown(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                self._mj.mj_step(self._model, self._data)
            time.sleep(max(self._dt, 0.02))


class MujocoBackend(PickBananaBackend):
    """Places objects through the plugin's ``ResetWorld`` service (D9).

    Every reset names a keyframe (``random_KKK`` or ``default``) — never the
    empty keyframe: ``ResetWorld("")`` restores the plugin's captured startup
    state, which holds zero arm joints instead of the rest pose the runtime
    just reached. The same keyframe is applied to the shadow model so
    evaluation observes the commanded placement.
    """

    REQUIRED_OBJECTS = ("banana", "plate")

    def __init__(
        self,
        node,
        layout: dict,
        *,
        scene_model_path: str,
        reset_world_service: str = "",
        callback_group,
    ) -> None:
        from mujoco_ros2_control_msgs.srv import ResetWorld  # lazy: mujoco-only dependency
        from sensor_msgs.msg import JointState

        super().__init__(node, layout, callback_group=callback_group)
        self._reset_world_name = reset_world_service or "/mujoco_ros2_control_node/reset_world"
        self._reset_world = node.create_client(ResetWorld, self._reset_world_name, callback_group=callback_group)
        self._arm_joint_names = [str(j) for j in (layout or {}).get("randomization", {}).get("arm_joint_names", [])]
        self._last_joint_update = 0.0
        self._shadow: ShadowSim | None = None
        if scene_model_path:
            try:
                self._shadow = ShadowSim(scene_model_path)
                node.get_logger().info(f"ShadowSim running from {scene_model_path}")
            except Exception as exc:  # noqa: BLE001
                node.get_logger().warning(f"ShadowSim unavailable: {exc}")
        node.create_subscription(JointState, "/joint_states", self._on_joint_states, 10, callback_group=callback_group)

    def check_ready(self) -> tuple[bool, str]:
        if self._shadow is None:
            return False, "shadow model unavailable (composed model missing or mujoco not importable)"
        if not self._reset_world.service_is_ready():
            return False, f"reset_world service not available ({self._reset_world_name})"
        return True, "reset_world service and composed model present"

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        keyframe = "default" if index is None else f"random_{index:03d}"
        ok, message = self._call_reset_world(keyframe, stop_event=stop_event)
        if ok and self._shadow is not None and not self._shadow.reset_to_keyframe(keyframe):
            self._node.get_logger().warning(f"shadow model has no keyframe '{keyframe}'")
        return ok, message

    def object_position(self, name: str) -> np.ndarray | None:
        if self._shadow is None:
            return None
        if time.monotonic() - self._last_joint_update > POSE_STALE_S:
            return None
        return self._shadow.get_body_pos(name)

    def shutdown(self) -> None:
        if self._shadow is not None:
            self._shadow.shutdown()

    def _on_joint_states(self, msg) -> None:
        self._last_joint_update = time.monotonic()
        if self._shadow is None:
            return
        for name, pos in zip(msg.name, msg.position, strict=False):
            if name in self._arm_joint_names:
                self._shadow.set_joint_pos(str(name), float(pos))

    def _call_reset_world(self, keyframe: str, timeout_s: float = 15.0, stop_event=None) -> tuple[bool, str]:
        """Call reset_world from the calling (executor or background) thread."""
        if not self._reset_world.wait_for_service(timeout_sec=5.0):
            return False, f"reset_world service not available ({self._reset_world_name})"
        request = self._reset_world.srv_type.Request()
        request.keyframe = keyframe
        future = self._reset_world.call_async(request)
        done = threading.Event()
        result_box: list[tuple[bool, str]] = []

        def _cb(f):
            try:
                response = f.result()
                result_box.append((bool(response.success), str(response.message)))
            except Exception as exc:  # noqa: BLE001
                result_box.append((False, str(exc)))
            done.set()

        future.add_done_callback(_cb)
        if not _wait_done(done, timeout_s, stop_event):
            if stop_event is not None and stop_event.is_set():
                return False, "reset_world interrupted (stop)"
            return False, "reset_world timeout"
        return result_box[0]


# ─── Gazebo ───────────────────────────────────────────────────────────────────


class GazeboBackend(PickBananaBackend):
    """Places objects through the bridged ``SetEntityPose`` service (D9).

    Only the bridged transport is used (the T0 spike verified service
    bridging works on Fortress); readiness requires the bridge and fresh
    samples of every scene object the task places.
    """

    REQUIRED_OBJECTS = ("banana", "plate")

    def __init__(
        self,
        node,
        layout: dict,
        *,
        world_name: str,
        scene: str,
        callback_group,
    ) -> None:
        from ros_gz_interfaces.msg import Entity  # lazy: gazebo-only dependency
        from ros_gz_interfaces.srv import SetEntityPose  # lazy: gazebo-only dependency
        from tf2_msgs.msg import TFMessage  # lazy: gazebo-only dependency

        super().__init__(node, layout, callback_group=callback_group)
        self._world_name = world_name
        self._entity_type = Entity.MODEL
        self._set_pose_service = f"/world/{world_name}/set_pose"
        self._set_pose = node.create_client(SetEntityPose, self._set_pose_service, callback_group=callback_group)
        self._poses: dict[str, tuple[np.ndarray, float]] = {}
        node.create_subscription(
            TFMessage, f"/sim/{scene}/object_poses", self._on_object_poses, 10, callback_group=callback_group
        )

    def check_ready(self) -> tuple[bool, str]:
        if not self._set_pose.service_is_ready():
            return False, f"set_pose bridge not available ({self._set_pose_service})"
        stale = [name for name in self.REQUIRED_OBJECTS if not self._fresh_sample(name)]
        if stale:
            return False, f"no fresh object_poses sample for {', '.join(stale)} on /sim"
        return True, "set_pose bridge and object_poses stream present"

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        banana_position, banana_quat = self._banana_pose(index)
        plate_position, plate_quat = self._plate_pose()
        for name, position, quat in (("banana", banana_position, banana_quat), ("plate", plate_position, plate_quat)):
            ok, message = self._set_entity_pose(name, position, quat, stop_event=stop_event)
            if not ok:
                return False, f"set pose of {name} failed: {message}"
        return True, "placed"

    def object_position(self, name: str) -> np.ndarray | None:
        sample = self._poses.get(name)
        if sample is None:
            return None
        position, received_at = sample
        if time.monotonic() - received_at > POSE_STALE_S:
            return None
        return position.copy()

    def _fresh_sample(self, name: str) -> bool:
        sample = self._poses.get(name)
        return sample is not None and time.monotonic() - sample[1] <= POSE_STALE_S

    def _on_object_poses(self, msg) -> None:
        received_at = time.monotonic()
        for transform in msg.transforms:
            translation = transform.transform.translation
            self._poses[transform.child_frame_id] = (
                np.array([translation.x, translation.y, translation.z]),
                received_at,
            )

    def _set_entity_pose(
        self,
        name: str,
        position: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
        stop_event=None,
    ) -> tuple[bool, str]:
        if not self._set_pose.service_is_ready():
            return False, "set_pose bridge not available"
        request = self._set_pose.srv_type.Request()
        request.entity.name = name
        request.entity.type = self._entity_type
        request.pose.position.x, request.pose.position.y, request.pose.position.z = position
        w, x, y, z = quat_wxyz
        request.pose.orientation.w, request.pose.orientation.x = w, x
        request.pose.orientation.y, request.pose.orientation.z = y, z
        future = self._set_pose.call_async(request)
        done = threading.Event()

        def _cb(_future):
            done.set()

        future.add_done_callback(_cb)
        if not _wait_done(done, 5.0, stop_event):
            if stop_event is not None and stop_event.is_set():
                return False, "set_pose interrupted (stop)"
            return False, "set_pose timeout"
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return bool(response.success), "ok"
