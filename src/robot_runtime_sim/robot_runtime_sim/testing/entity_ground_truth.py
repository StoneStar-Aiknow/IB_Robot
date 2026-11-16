"""Entity ground truth as a shared test-support module (#205 grasp stage D3).

Tests that need the pose of a SCENE OBJECT or of one LINK of a model read it
through this one platform-neutral helper — the generic sibling of
:mod:`robot_runtime_sim.testing.sim_ground_truth` (whose yaw-only model-pose API
stays unchanged for the navigation gates).

``open_entity_ground_truth(platform, *, world, scene, entity, link=None,
env, mjcf_path=None)`` returns a stream of :class:`GroundTruthPose` samples
(``stamp_sec``, ``position[3]``, ``quat_xyzw[4]``) with the reader API the
#205 consumers use (``start/wait_for/latest/samples/close``). ``entity``
names a scene object or the robot model; ``link`` selects one link of that
model. Nothing is advertised or consumed by the runtime: every source lives
on simulator-private topics.

Sources (spike 1.3, measured):

* Gazebo, an entity without ``link``: ``/world/<world>/dynamic_pose/info``
  — the dynamic models' world poses, published continuously (a resting
  object keeps publishing).
* Gazebo, an entity with ``link``: ``/world/<world>/pose/info`` — the
  message carries the model's WORLD pose and, as separate flat entries, the
  model-relative LINK poses. The link entries carry bare names, so a link
  is scoped to its model through the entity ids (gz-sim assigns them
  depth-first: every link of a model falls between the model's id and the
  next model's id — pinned by the unit tests). The stream itself resolves
  the world's models by NAME (the scene layout's models + the world
  scaffolding + the entity, review round 1 item 17). Link world pose =
  model world pose ∘ model-relative link pose (one composition; the spike
  proved the frame semantics with a 3-link fixture).
* MuJoCo, an entity without ``link``: ``/sim/<scene>/free_joint_states``
  (``mujoco_ros2_control_msgs/FreeJointStateArray``, entries by body name,
  ``geometry_msgs`` xyzw quaternions, one sim-time stamp per message).
* MuJoCo, an entity with ``link``: forward kinematics of the simulator's
  own MJCF — the caller passes the model file the provider actually loaded
  (``mjcf_path``); the stream pairs ``/joint_states`` with
  ``/simulator/floating_base_state`` (pairs further than 20 ms apart in sim
  time are rejected), assembles the full ``qpos`` through the model's
  ``jnt_qposadr`` mapping (the floating base's quaternion is MuJoCo
  w-x-y-z, converted to xyzw) and runs ``mj_kinematics``; the body pose by
  name is the sample. Engine extensions declared by the model are loaded
  first (``load_mujoco_engine_plugins``), so extension-declaring models
  load exactly as the provider loads them.

A rewound clock (a world reset) starts a new epoch: the pre-reset history is
dropped, because every reader (``latest``/``samples``) must see a strictly
time-ordered span — and the reset is RECORDED (``reset_detected``): a world
reset mid-run voids the run's ground truth, and the consumer fails the run
explicitly instead of the history silently restarting. A rewind within
:data:`RESET_REWIND_MARGIN_SEC` is a publisher hiccup (a duplicate or briefly
reordered stamp, measured on the GUI screenshot runs), not a reset: only the
non-advancing sample is dropped.

No robot names, links or offsets are hard-coded here: the caller passes the
entity and link (the LeKiwi tests pass ``lekiwi_robot``/``gripper``). This
is test-support API, not runtime API: importing this module imports no
simulator and no ``rclpy``; only ``start()`` spawns a child or a context.
"""

from __future__ import annotations

import codecs
import math
import re
import subprocess
import threading
import time
from pathlib import Path

from robot_runtime_sim.testing.sim_ground_truth import (
    _BUFFER_KEEP,
    _BUFFER_LIMIT,
    _READ_CHUNK,
    _SAMPLE_LIMIT,
    mirror_partition_env,
)

_STAMP_RE = re.compile(r"stamp \{\s*sec: (\d+)\s*nsec: (\d+)")
_POSITION_RE = re.compile(r"position \{\s*(?:x: ([-\d.e+]+))?\s*(?:y: ([-\d.e+]+))?\s*(?:z: ([-\d.e+]+))?\s*\}")
_ORIENTATION_RE = re.compile(
    r"orientation \{\s*(?:x: ([-\d.e+]+))?\s*(?:y: ([-\d.e+]+))?\s*(?:z: ([-\d.e+]+))?\s*(?:w: ([-\d.e+]+))?\s*\}"
)

#: The maximum sim-time distance between the /joint_states and
#: /simulator/floating_base_state samples the MuJoCo link kinematics pairs
#: (spike 1.3 measured 0-10 ms; pairs further apart are rejected, D3).
MAX_PAIRING_GAP_SEC = 0.02

#: The margin that separates a REAL world reset from a publisher hiccup
#: (review r1 item 8, measured on the GUI screenshot runs): a world reset
#: rewinds the simulator clock by seconds (the timeline restarts), while a
#: capture-stalled GUI publisher re-emits a duplicate or briefly reordered
#: stamp within one publish period. Resets VOID the run's evidence and fail
#: it explicitly; hiccups drop only the non-advancing sample so the span
#: keeps its strictly time-ordered order without throwing away a run.
RESET_REWIND_MARGIN_SEC = 0.1


class GroundTruthPose:
    """One entity pose, stamped in simulation time by the pose message."""

    __slots__ = ("stamp_sec", "position", "quat_xyzw")

    def __init__(self, stamp_sec: float, position, quat_xyzw):
        self.stamp_sec = float(stamp_sec)
        self.position = (float(position[0]), float(position[1]), float(position[2]))
        self.quat_xyzw = (
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GroundTruthPose(stamp_sec={self.stamp_sec:.3f}, position={self.position}, quat_xyzw={self.quat_xyzw})"


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    """An xyzw quaternion -> a 3x3 rotation matrix (plain lists)."""
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _quat_multiply(a, b) -> tuple[float, float, float, float]:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _entry_pose(block: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """The position/xyzw-orientation of one pose entry's text block."""
    position = _POSITION_RE.search(block)
    orientation = _ORIENTATION_RE.search(block)
    if not (position and orientation):
        return None

    def _component(value: str | None, default: float) -> float:
        return float(value) if value is not None else default

    return (
        (
            _component(position.group(1), 0.0),
            _component(position.group(2), 0.0),
            _component(position.group(3), 0.0),
        ),
        (
            _component(orientation.group(1), 0.0),
            _component(orientation.group(2), 0.0),
            _component(orientation.group(3), 0.0),
            _component(orientation.group(4), 1.0),
        ),
    )


def parse_entity_pose_messages(
    buffer: str, entity: str, link: str | None, model_names: set[str], *, final: bool = False
) -> tuple[list[GroundTruthPose], str]:
    """Parse complete gz ``Pose_V`` text messages for one entity/link pair.

    Each message carries one sim-time ``header`` and flat ``pose {`` entries
    (``name``, optional ``id``, ``position``/``orientation``). Without
    ``link`` the entity's own entry is its world pose. With ``link`` the
    entity's entry (the model's world pose) is composed with the LINK entry
    — the link entry whose bare ``name`` equals ``link`` AND whose ``id``
    falls between the model's id and the next model's id (the model's
    subtree scope). A message is complete once the next ``header {``
    arrives, so the trailing chunk stays in the returned tail unless
    ``final``.

    ``model_names`` (review round 1, item 17) is the set of MODEL names the
    world carries — the scene layout's models, the world scaffolding and
    the spawned robot (``_gazebo_model_names``). The models are exactly the
    entries whose name is in the set (deterministic, no id-ordering
    assumption — a SINGLE-model world has no id descent, so an
    id-prefix heuristic would eat its links as models and never find the
    link); the requested entity must be among the named models (else no
    samples), and the link scope ``(model_id, next_model_id)`` is computed
    from the named-model ids.
    """
    samples: list[GroundTruthPose] = []
    chunks = buffer.split("header {")
    complete = chunks[:-1] if not final else chunks
    tail = "" if final else chunks[-1]
    for chunk in complete:
        stamp = _STAMP_RE.search(chunk)
        if stamp is None:
            continue
        stamp_sec = int(stamp.group(1)) + int(stamp.group(2)) / 1e9
        spans = [match.span() for match in re.finditer(r"pose \{", chunk)]
        entries: list[tuple[str, int | None]] = []
        for match in re.finditer(r'pose \{\s*name: "([^"]+)"(?:\s*\n\s*id: (\d+))?', chunk):
            entries.append((match.group(1), int(match.group(2)) if match.group(2) else None))

        def _block(index: int, _chunk: str = chunk, _spans=spans) -> str:
            return _chunk[_spans[index][0] : _spans[index + 1][1] if index + 1 < len(_spans) else len(_chunk)]

        model_ids = [(entity_id, name) for name, entity_id in entries if name in model_names and entity_id is not None]
        entity_entry = next((e for e in entries if e[0] == entity), None)
        if entity_entry is None or entity not in model_names:
            continue  # a missing entity, or one outside the named models
        index = entries.index(entity_entry)
        model_pose = _entry_pose(_block(index))
        if model_pose is None:
            continue
        if link is None:
            samples.append(GroundTruthPose(stamp_sec, model_pose[0], model_pose[1]))
            continue
        model_id = entity_entry[1]
        if model_id is None or (model_id, entity) not in model_ids:
            continue  # a link stream needs the entity to be a model entry
        above = [mid for mid, _name in model_ids if mid > model_id]
        next_model_id = min(above) if above else math.inf
        link_index = next(
            (
                i
                for i, (name, entity_id) in enumerate(entries)
                if name == link and entity_id is not None and model_id < entity_id < next_model_id
            ),
            None,
        )
        if link_index is None:
            continue
        link_pose = _entry_pose(_block(link_index))
        if link_pose is None:
            continue
        # link world pose = model world pose ∘ model-relative link pose:
        # the model quaternion's rotation matrix applied to the link offset
        r00, r01, r02 = (
            1 - 2 * (model_pose[1][1] ** 2 + model_pose[1][2] ** 2),
            2 * (model_pose[1][0] * model_pose[1][1] - model_pose[1][2] * model_pose[1][3]),
            2 * (model_pose[1][0] * model_pose[1][2] + model_pose[1][1] * model_pose[1][3]),
        )
        r10, r11, r12 = (
            2 * (model_pose[1][0] * model_pose[1][1] + model_pose[1][2] * model_pose[1][3]),
            1 - 2 * (model_pose[1][0] ** 2 + model_pose[1][2] ** 2),
            2 * (model_pose[1][1] * model_pose[1][2] - model_pose[1][0] * model_pose[1][3]),
        )
        r20, r21, r22 = (
            2 * (model_pose[1][0] * model_pose[1][2] - model_pose[1][1] * model_pose[1][3]),
            2 * (model_pose[1][1] * model_pose[1][2] + model_pose[1][0] * model_pose[1][3]),
            1 - 2 * (model_pose[1][0] ** 2 + model_pose[1][1] ** 2),
        )
        world_position = (
            model_pose[0][0] + r00 * link_pose[0][0] + r01 * link_pose[0][1] + r02 * link_pose[0][2],
            model_pose[0][1] + r10 * link_pose[0][0] + r11 * link_pose[0][1] + r12 * link_pose[0][2],
            model_pose[0][2] + r20 * link_pose[0][0] + r21 * link_pose[0][1] + r22 * link_pose[0][2],
        )
        world_orientation = _quat_multiply(model_pose[1], link_pose[1])
        samples.append(GroundTruthPose(stamp_sec, world_position, world_orientation))
    return samples, tail


def _record_samples(samples: list[GroundTruthPose], history: list[GroundTruthPose]) -> int:
    """Append parsed samples, restarting the history on a world reset.

    A sample whose stamp rewinds the newest kept stamp by MORE than
    :data:`RESET_REWIND_MARGIN_SEC` is the first sample of a new epoch (a
    world reset rewinds the simulator clock): the pre-reset history is
    dropped so ``latest``/``samples`` keep a strictly time-ordered span —
    and the reset is COUNTED (the return value): a world reset mid-run
    voids the run's evidence, and the consumer must FAIL the run explicitly
    (``reset_detected``) instead of the history silently restarting as if
    nothing happened.

    A rewind WITHIN the margin is a publisher hiccup, not a reset (measured
    on the GUI screenshot runs: a capture-stalled publisher re-emits a
    duplicate or briefly reordered stamp): that sample is DROPPED — it
    carries no new state — and the run is not voided.
    """
    resets = 0
    for sample in samples:
        if history and sample.stamp_sec <= history[-1].stamp_sec:
            if history[-1].stamp_sec - sample.stamp_sec > RESET_REWIND_MARGIN_SEC:
                history.clear()  # a world reset rewinds the clock: a new epoch
                resets += 1
            else:
                continue  # a publisher hiccup: the non-advancing sample is dropped
        history.append(sample)
    return resets


class EntityGroundTruthStream:
    """The live entity-pose stream (the platform-agnostic sample store).

    ``start()`` spawns the source (an ``ign topic -e`` child on Gazebo) and
    a reader thread; ``close()`` terminates it and joins the thread. Samples
    carry the messages' own simulation-time stamps.
    """

    def __init__(
        self, topic: str, entity: str, link: str | None, env: dict[str, str] | None, model_names: set[str] | None = None
    ):
        self._topic = str(topic)
        self._entity = str(entity)
        self._link = None if link is None else str(link)
        self._env = env
        # the model-name set (review round 1, item 17): the parser resolves
        # the models by NAME — a single-model world has no id descent, so an
        # id-prefix heuristic would eat its links as models. The Gazebo
        # streams always receive the scene-derived set.
        self._model_names = set(model_names or ())
        self._samples: list[GroundTruthPose] = []
        self._last_sample_wall = 0.0
        self._buffer = ""
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        # world resets (a rewound clock) recorded by _record_samples: a reset
        # mid-run voids the run's ground truth and the consumer fails the
        # run explicitly — the history restart is never silent
        self._reset_count = 0

    def start(self) -> EntityGroundTruthStream:
        """Spawn the pose echo child and its reader (no-op on restart)."""
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            ["ign", "topic", "-e", "-t", self._topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=mirror_partition_env(self._env),
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = self._proc.stdout.read1(_READ_CHUNK)
            if not chunk:
                with self._lock:
                    samples, tail = parse_entity_pose_messages(
                        self._buffer, self._entity, self._link, final=True, model_names=self._model_names
                    )
                    self._buffer = tail
                    if samples:
                        self._reset_count += _record_samples(samples, self._samples)
                        self._last_sample_wall = time.time()
                return
            with self._lock:
                self._buffer += decoder.decode(chunk)
                samples, tail = parse_entity_pose_messages(
                    self._buffer, self._entity, self._link, model_names=self._model_names
                )
                self._buffer = tail
                if samples:
                    self._reset_count += _record_samples(samples, self._samples)
                    self._last_sample_wall = time.time()
                if len(self._samples) > 2 * _SAMPLE_LIMIT:
                    del self._samples[:-_SAMPLE_LIMIT]
                if len(self._buffer) > _BUFFER_LIMIT:
                    self._buffer = self._buffer[-_BUFFER_KEEP:]

    def wait_for(self, n: int, timeout: float) -> bool:
        """Wait until ``n`` samples are recorded; False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._samples) >= n:
                    return True
            time.sleep(0.05)
        with self._lock:
            return len(self._samples) >= n

    def samples(self) -> list[GroundTruthPose]:
        """Snapshot of the recorded samples, oldest first."""
        with self._lock:
            return list(self._samples)

    def latest(self) -> GroundTruthPose | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def reset_detected(self) -> bool:
        """Whether the stream's clock rewound since ``start()`` (a world
        reset): the recorded history restarted an epoch, so the consumer
        must FAIL the run explicitly — a silent history restart would
        present the post-reset span as if it were the whole run."""
        with self._lock:
            return self._reset_count > 0

    def close(self) -> None:
        """Terminate the echo child and join the reader thread."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        if self._thread is not None:
            self._thread.join(timeout=5)


class MujocoEntityGroundTruthStream(EntityGroundTruthStream):
    """The free-joint object stream of one scene entity (MuJoCo).

    Subscribes, on a private ROS context (the domain taken from ``env``),
    to ``/sim/<scene>/free_joint_states`` — the ``FreeJointStateArray`` the
    simulator's own publisher puts out for the scene's dynamic bodies
    (``geometry_msgs`` xyzw quaternions, one sim-time stamp per message).
    """

    def __init__(self, scene: str, entity: str, env: dict[str, str] | None):
        super().__init__(f"/sim/{scene}/free_joint_states", entity, None, env)
        self._context = None
        self._node = None
        self._executor = None
        self._spin_thread = None

    def start(self) -> MujocoEntityGroundTruthStream:
        if self._node is not None:
            return self
        import os

        import rclpy  # noqa: PLC0415  (test-support module: no ROS import at module load)
        from mujoco_ros2_control_msgs.msg import FreeJointStateArray  # noqa: PLC0415

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
            "mujoco_entity_ground_truth",
            context=self._context,
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self._node.create_subscription(FreeJointStateArray, self._topic, self._on_free_joint_states, 100)
        self._executor = rclpy.executors.SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        return self

    def _on_free_joint_states(self, message) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        samples = [
            GroundTruthPose(
                stamp,
                (entry.pose.pose.position.x, entry.pose.pose.position.y, entry.pose.pose.position.z),
                (
                    entry.pose.pose.orientation.x,
                    entry.pose.pose.orientation.y,
                    entry.pose.pose.orientation.z,
                    entry.pose.pose.orientation.w,
                ),
            )
            for entry in message.free_joints
            if str(entry.name) == self._entity
        ]
        with self._lock:
            self._reset_count += _record_samples(samples, self._samples)
            if samples:
                self._last_sample_wall = time.time()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=5)
            self._spin_thread = None
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


class MujocoFloatingBaseGroundTruthStream(EntityGroundTruthStream):
    """The floating-BASE pose stream of the robot (MuJoCo).

    The robot is not a scene dynamic object — ``/sim/<scene>/free_joint_states``
    carries only the layout's dynamic bodies (``scene_dynamic_objects``) — so
    its base pose is read from ``/simulator/floating_base_state`` (the
    ``nav_msgs/Odometry`` the simulator's own floating-base plugin publishes,
    never advertised or consumed by the runtime; D3/D9). ``body`` is the
    free-joint BODY name of the robot's root (the composed MJCF names it
    after the robot's root link, not the model name — the caller passes it;
    messages of other bodies are ignored via ``child_frame_id``).
    """

    def __init__(self, body: str, env: dict[str, str] | None):
        super().__init__(None, body, None, env)  # type: ignore[arg-type]  (no ign topic: ROS subscription)
        self._context = None
        self._node = None
        self._executor = None
        self._spin_thread = None

    def start(self) -> MujocoFloatingBaseGroundTruthStream:
        if self._node is not None:
            return self
        import os

        import rclpy  # noqa: PLC0415  (test-support module: no ROS import at module load)
        from nav_msgs.msg import Odometry  # noqa: PLC0415

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
            "mujoco_floating_base_ground_truth",
            context=self._context,
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self._node.create_subscription(Odometry, FLOATING_BASE_STATE_TOPIC, self._on_odometry, 100)
        self._executor = rclpy.executors.SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        return self

    def _on_odometry(self, message) -> None:
        if str(message.child_frame_id or "") != self._entity:
            return
        pose = message.pose.pose
        sample = GroundTruthPose(
            message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
        )
        with self._lock:
            self._reset_count += _record_samples([sample], self._samples)
            self._last_sample_wall = time.time()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=5)
            self._spin_thread = None
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


#: The world-template scaffolding every shipped Gazebo scene carries
#: (``sim_models/scenes/*/*.world.template``): the ground plane model and
#: the sun light — both appear in a world's entity graph alongside the
#: layout's own models, so a name-derived model set must include them.
_GAZEBO_WORLD_SCAFFOLDING = {"ground_plane", "sun"}


def _gazebo_model_names(scene: str, entity: str) -> set[str]:
    """The model names a scene's gz ``pose/info`` stream may carry (item 17).

    One definition per source: the scene layout's ``static_geometry`` and
    ``objects`` names (``sim_models.scene_compiler.get_scene_layout`` — the
    same ``layout.yaml`` the world templates and maps are generated from),
    the world-template scaffolding (``ground_plane``/``sun``) and the
    requested entity itself (the spawned robot model is named by the
    caller, never by the layout). Handing this set to the parser removes
    the ascending-id-prefix assumption, which breaks on a single-model
    world. An unknown/empty scene contributes only the scaffolding names.
    """
    from sim_models.scene_compiler import get_scene_layout

    layout = get_scene_layout(scene) if scene else {}
    names = {
        str(entry["name"])
        for entry in layout.get("static_geometry") or []
        if isinstance(entry, dict) and entry.get("name")
    }
    names |= {
        str(entry["name"]) for entry in layout.get("objects") or [] if isinstance(entry, dict) and entry.get("name")
    }
    names |= _GAZEBO_WORLD_SCAFFOLDING
    names.add(str(entity))
    return names


def open_entity_ground_truth(
    sim_platform: str,
    *,
    world: str,
    scene: str,
    entity: str,
    link: str | None = None,
    env: dict[str, str] | None = None,
    mjcf_path: str | Path | None = None,
    reference_body: str | None = None,
):
    """The entity-pose ground-truth stream on ``sim_platform`` (D3).

    Gazebo reads the world's dynamic-pose info topic for an entity, or the
    pose-info topic (model ∘ model-relative link) when ``link`` is set —
    the stream resolves the world's models by NAME from the scene layout
    (``static_geometry`` + ``objects`` + the world-template scaffolding +
    the entity), so the parser never leans on the ascending-id-prefix
    assumption (item 17). MuJoCo reads the scene's free-joint states for an
    entity, or — when ``link`` is set — forward kinematics of the MJCF at
    ``mjcf_path`` (the model file the PROVIDER loaded; required for links,
    rejected for entities without one). ``reference_body`` (MuJoCo links)
    returns the pose RELATIVE to that body instead of the world. The stream
    is returned unstarted: only ``start()`` spawns a process or a context.
    """
    if sim_platform == "gazebo":
        topic = f"/world/{world}/dynamic_pose/info" if link is None else f"/world/{world}/pose/info"
        return EntityGroundTruthStream(topic, entity, link, env, model_names=_gazebo_model_names(scene, entity))
    if sim_platform == "mujoco":
        if link is None:
            return MujocoEntityGroundTruthStream(scene, entity, env)
        if mjcf_path is None:
            raise ValueError(
                "open_entity_ground_truth: a MuJoCo LINK stream needs mjcf_path — the model "
                "file the provider actually loaded (from the rendered robot_description)"
            )
        from robot_runtime_sim.testing.mujoco_link_ground_truth import MujocoLinkGroundTruthStream

        return MujocoLinkGroundTruthStream(str(mjcf_path), entity, link, env, reference_body=reference_body)
    raise ValueError(
        f"no entity ground-truth implementation for platform {sim_platform!r} (supported: 'gazebo', 'mujoco')"
    )
