"""agent_eval's evaluation layer on the direct-control runtime (an ``extensions`` entry).

``run_suite`` starts the runtime with a copy of its profile whose
``simulation.<platform>.extensions`` lists :func:`attach`; the runtime calls it once with its
node, before the platform connects. A runtime started without it has no evaluation concept:
motion is never held and the platform's result is only logged.

What it adds:

* **Admission**: ``/evaluation/admit_episode`` (``AdmitSimulationEpisode``). The runner opens
  the current session's generation for motion once that session's observations arrived and
  no Agent request of an earlier episode is still open. Until then the engine's motion gate
  refuses every motion of the generation, so a request planned in an earlier episode (or one
  that arrives after a cancel, racing the reset) can never move the robot in a later one.
* **Verdict channel**: ``/evaluation/native_result`` (latched ``std_msgs/String`` JSON), the
  platform's result of each session as the runtime hands it to result listeners. Only the
  evaluator subscribes; the Agent never sees it.

The gate and the listener run inside the engine's lock: they only read or replace one value
and publish, never block and never call back into the engine. The admission service runs on
its own callback group and may (it checks the session against the engine).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

ADMIT_SERVICE = "/evaluation/admit_episode"
NATIVE_RESULT_TOPIC = "/evaluation/native_result"
NOT_ADMITTED = "not admitted: the evaluation runner has not opened this episode for motion"


class EvaluationLayer:
    """Admission gate + verdict publisher for one runtime node (see the module doc)."""

    def __init__(self, core, publish_result=None):
        self._core = core
        self._publish_result = publish_result or (lambda _payload: None)
        self._lock = threading.Lock()
        self._admitted: int | None = None  # the one generation open for motion

    def gate(self, generation: int) -> str:
        """The engine's motion gate (engine lock held): only the admitted generation may move."""
        with self._lock:
            return "" if generation == self._admitted else NOT_ADMITTED

    def admit(self, session_id: str, generation: int) -> tuple[bool, str]:
        """Open ``generation`` for motion if it is the runtime's active session."""
        try:
            self._core.check_session(session_id, generation)
        except Exception as exc:  # the engine's SessionError: not imported, the layer needs only the node
            return False, f"not admitted: {exc}"
        with self._lock:
            self._admitted = int(generation)
        return True, f"session {session_id} generation {generation} admitted for motion"

    def on_result(self, result: dict[str, Any]) -> None:
        """Platform result listener (engine lock held): publish it on the verdict channel."""
        self._publish_result({**result, "wall_time": time.time()})


def attach(node) -> EvaluationLayer:
    """``extensions`` entry: install the gate and the listener, serve admission, publish verdicts."""
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    from ibrobot_msgs.srv import AdmitSimulationEpisode

    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(String, NATIVE_RESULT_TOPIC, latched)
    layer = EvaluationLayer(node.core, lambda payload: publisher.publish(String(data=json.dumps(payload, default=str))))

    def on_admit(request, response):
        response.success, response.message = layer.admit(str(request.session_id), int(request.generation))
        node.get_logger().info(f"admission of {request.session_id} g{request.generation}: {response.message}")
        return response

    node.create_service(
        AdmitSimulationEpisode, ADMIT_SERVICE, on_admit, callback_group=MutuallyExclusiveCallbackGroup()
    )
    node.core.set_motion_gate(layer.gate)
    node.add_platform_result_listener(layer.on_result)
    node.evaluation_layer = layer
    return layer
