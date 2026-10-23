"""Platform backends for the pick_banana scene task (D9).

Unit tests with fake clients: the shared randomisation (index k → identical
banana placement on both platforms), default placement, stale-pose rejection,
bounded readiness (a delayed Gazebo bridge, missing pose stream, MuJoCo
readiness) and the ``ign``/``gz`` ``service`` CLI fallback for set_pose.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np

from sim_models.scene_compiler import _quat_from_extrinsic_xyz, get_scene_layout, randomization_grid
from sim_models.tasks.backends import GazeboBackend, MujocoBackend


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _Request:
    pass


def _set_pose_request():
    return SimpleNamespace(
        entity=SimpleNamespace(name="", type=0, id=0),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


class FakeServiceClient:
    def __init__(self, *, available=True, available_after=None, response=None, request_factory=_Request):
        self.available = available
        self.available_after = available_after
        self.response = response or SimpleNamespace(success=True, message="ok")
        self.request_factory = request_factory
        self.requests = []

    @property
    def srv_type(self):
        return SimpleNamespace(Request=self.request_factory)

    def wait_for_service(self, timeout_sec=None):
        if self.available_after is not None and time.monotonic() >= self.available_after:
            self.available = True
        return self.available

    def service_is_ready(self):
        return self.wait_for_service(0)

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(self.response)


class FakeNode:
    """Records clients/subscriptions; scripted per service name."""

    def __init__(self, clients=None):
        self.logger = FakeLogger()
        self.clients = clients or {}
        self.subscriptions = []

    def get_logger(self):
        return self.logger

    def create_client(self, srv_type, name, callback_group=None):
        return self.clients.get(name, FakeServiceClient())

    def create_subscription(self, msg_type, topic, callback, qos, callback_group=None):
        self.subscriptions.append((topic, callback))


class FakeShadow:
    def __init__(self):
        self.keyframes = []

    def reset_to_keyframe(self, name):
        self.keyframes.append(name)
        return True


def _layout():
    return get_scene_layout("pick_banana")


def make_mujoco_backend(*, reset_client=None):
    node = FakeNode()
    if reset_client is not None:
        node.clients["/mujoco_ros2_control_node/reset_world"] = reset_client
    backend = MujocoBackend(node, _layout(), scene_model_path="", callback_group=None)
    return backend, node


def make_gazebo_backend(*, set_pose_client=None):
    node = FakeNode()
    if set_pose_client is None:
        set_pose_client = FakeServiceClient(request_factory=_set_pose_request)
    else:
        set_pose_client.request_factory = _set_pose_request
    node.clients["/world/demo/set_pose"] = set_pose_client
    backend = GazeboBackend(node, _layout(), world_name="demo", scene="pick_banana", callback_group=None)
    return backend, node


def _pose_message(name, x, y, z):
    return SimpleNamespace(
        transforms=[
            SimpleNamespace(
                child_frame_id=name,
                transform=SimpleNamespace(translation=SimpleNamespace(x=x, y=y, z=z)),
            )
        ]
    )


# ── shared randomisation ──────────────────────────────────────────────────────


def test_index_k_places_the_same_banana_pose_on_both_backends():
    grid = randomization_grid(_layout())
    k = 137
    x, y, z, yaw = grid[k]

    mujoco, _ = make_mujoco_backend(reset_client=FakeServiceClient())
    ok, _ = mujoco.place(k)
    assert ok
    reset_request = mujoco._reset_world.requests[0]
    assert reset_request.keyframe == f"random_{k:03d}"

    gazebo, _ = make_gazebo_backend()
    ok, _ = gazebo.place(k)
    assert ok
    requests = gazebo._set_pose.requests
    banana = next(r for r in requests if r.entity.name == "banana")
    assert (banana.pose.position.x, banana.pose.position.y, banana.pose.position.z) == (x, y, z)
    quat_wxyz = _quat_from_extrinsic_xyz(0.0, 0.0, yaw)
    assert (
        banana.pose.orientation.w,
        banana.pose.orientation.x,
        banana.pose.orientation.y,
        banana.pose.orientation.z,
    ) == (quat_wxyz)


def test_default_placement_uses_the_layout_poses_and_the_default_keyframe():
    layout = _layout()
    banana_default = next(o for o in layout["objects"] if o["name"] == "banana")["position"]
    plate_default = next(o for o in layout["objects"] if o["name"] == "plate")["position"]

    mujoco, _ = make_mujoco_backend(reset_client=FakeServiceClient())
    ok, _ = mujoco.place(None)
    assert ok
    assert mujoco._reset_world.requests[0].keyframe == "default", "never the empty startup-state keyframe"

    gazebo, _ = make_gazebo_backend()
    ok, _ = gazebo.place(None)
    assert ok
    requests = gazebo._set_pose.requests
    banana = next(r for r in requests if r.entity.name == "banana")
    plate = next(r for r in requests if r.entity.name == "plate")
    assert (banana.pose.position.x, banana.pose.position.y, banana.pose.position.z) == tuple(banana_default)
    assert (plate.pose.position.x, plate.pose.position.y, plate.pose.position.z) == tuple(plate_default)


def test_mujoco_place_applies_the_keyframe_to_the_shadow_model():
    backend, _ = make_mujoco_backend(reset_client=FakeServiceClient())
    backend._shadow = FakeShadow()

    ok, _ = backend.place(4)

    assert ok
    assert backend._shadow.keyframes == ["random_004"]


# ── object position freshness ─────────────────────────────────────────────────


def test_gazebo_object_position_rejects_stale_samples():
    gazebo, _ = make_gazebo_backend()
    gazebo._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic() - 2.0)
    assert gazebo.object_position("banana") is None

    gazebo._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic())
    assert gazebo.object_position("banana") is not None


def test_mujoco_object_position_requires_fresh_joint_states():
    backend, _ = make_mujoco_backend()
    backend._shadow = FakeShadow()
    backend._last_joint_update = time.monotonic() - 2.0
    assert backend.object_position("banana") is None
    backend._last_joint_update = time.monotonic()
    backend._shadow.get_body_pos = lambda name: np.array([0.1, 0.0, 0.05])
    assert backend.object_position("banana") is not None


def test_gazebo_object_position_selects_by_child_frame_id():
    gazebo, node = make_gazebo_backend()
    topic, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.05, 0.04, 0.05))

    assert np.allclose(gazebo.object_position("banana"), [0.05, 0.04, 0.05])
    assert gazebo.object_position("plate") is None


# ── bounded readiness ─────────────────────────────────────────────────────────


def test_gazebo_waits_for_a_delayed_bridge_and_fresh_scene_objects():
    delayed = FakeServiceClient(available=False, available_after=time.monotonic() + 0.3)
    backend, node = make_gazebo_backend(set_pose_client=delayed)
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    ok, message = backend.wait_ready(3.0)

    assert ok, message
    assert "set_pose bridge" in message
    assert delayed.wait_for_service(), "the wait covered the bridge coming up"


def test_gazebo_not_ready_without_the_pose_stream():
    backend, _ = make_gazebo_backend()

    ok, message = backend.wait_ready(0.4)

    assert not ok and "object_poses" in message


def test_gazebo_check_ready_requires_fresh_samples_of_every_placed_object():
    backend, node = make_gazebo_backend()
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))

    ok, message = backend.check_ready()

    assert not ok and "plate" in message, "a fresh banana alone is not readiness"


def test_gazebo_check_ready_requires_the_bridge():
    backend, node = make_gazebo_backend(set_pose_client=FakeServiceClient(available=False))
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    ok, message = backend.check_ready()

    assert not ok and "set_pose bridge not available" in message


def test_gazebo_place_without_the_bridge_fails_without_a_fallback():
    backend, _ = make_gazebo_backend(set_pose_client=FakeServiceClient(available=False))

    ok, message = backend.place(0)

    assert not ok and "set_pose bridge not available" in message
    assert not backend._set_pose.requests, "nothing is sent when the bridge is down"


def test_mujoco_readiness_needs_the_service_and_the_composed_model():
    backend, _ = make_mujoco_backend(reset_client=FakeServiceClient())
    ok, message = backend.check_ready()
    assert not ok and "shadow model unavailable" in message

    backend._shadow = FakeShadow()
    ok, message = backend.check_ready()
    assert ok, message

    missing_service, _ = make_mujoco_backend(reset_client=FakeServiceClient(available=False))
    missing_service._shadow = FakeShadow()
    ok, message = missing_service.check_ready()
    assert not ok and "reset_world" in message


def test_mujoco_readiness_is_revalidated_not_latched():
    backend, _ = make_mujoco_backend(reset_client=FakeServiceClient())
    backend._shadow = FakeShadow()
    assert backend.check_ready()[0]

    backend._reset_world.available = False  # the service dropped between calls
    ok, message = backend.check_ready()
    assert not ok and "reset_world" in message


# ── startup readiness thread ──────────────────────────────────────────────────


def test_backend_start_readiness_logs_the_outcome():
    backend, node = make_gazebo_backend()
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    backend.start_readiness(timeout_s=2.0)
    deadline = time.monotonic() + 3.0
    while not any("ready" in text for _level, text in node.logger.messages) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert any("GazeboBackend: ready" in text for _level, text in node.logger.messages)


def test_concurrent_place_requests_stay_thread_safe():
    gazebo, _ = make_gazebo_backend()
    errors = []

    def worker(index):
        try:
            gazebo.place(index % gazebo.grid_size)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(gazebo._set_pose.requests) == 16, "8 workers × (banana + plate)"
