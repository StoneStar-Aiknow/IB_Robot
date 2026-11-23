"""ROS wiring of the episode runner: subscriptions feed the EventBook, publishers are its IO."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from .body import BodyDescription, load_body
from .episode_context import EpisodeContext
from .runner import EpisodeRunner, EventBook, RunnerSettings
from .runtime_extension import ADMIT_SERVICE, NATIVE_RESULT_TOPIC
from .task_spec import load_recipe, load_task_spec


@dataclass(frozen=True)
class Topics:
    """Endpoint names: the direct-control runtime, agent_eval's runtime extension, the Agent's public entry."""

    session: str = "/simulation/session"
    clock: str = "/simulation/clock"
    finish_session: str = "/simulation/finish_session"
    abort_session: str = "/simulation/abort_session"
    admit_episode: str = ADMIT_SERVICE
    native_result: str = NATIVE_RESULT_TOPIC
    joint_states: str = "/joint_states"
    agent_request: str = "/agent/request"
    agent_event: str = "/agent/event"
    agent_response: str = "/agent/response"
    agent_control: str = "/agent/control"
    agent_ready: str = "/ibrobot_agent_node/ready"


def _json(msg) -> dict[str, Any] | None:
    try:
        data = json.loads(msg.data)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class RosRunner:
    """Owns one rclpy node + executor thread; exposes the RunnerIO surface."""

    def __init__(
        self,
        body: BodyDescription,
        topics: Topics = Topics(),
        node_name: str = "agent_eval_runner",
        camera_topics: tuple[str, ...] = (),
    ):
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self._String = String
        self._Trigger = Trigger
        self.node = node = Node(node_name)
        self.book = book = EventBook(body, camera_topics=camera_topics)

        def stamp(msg) -> float:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        latched = QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )

        def feed(handler):
            def callback(msg):
                data = _json(msg)
                if data is not None:
                    handler(data)

            return callback

        node.create_subscription(String, topics.session, feed(book.on_session), latched)
        node.create_subscription(String, topics.native_result, feed(book.on_native_result), latched)
        node.create_subscription(String, topics.clock, feed(book.on_clock), 10)
        node.create_subscription(String, topics.agent_event, feed(book.on_agent_event), 100)
        node.create_subscription(String, topics.agent_response, feed(book.on_agent_response), 100)
        for arm in body.arms:

            def on_pose(msg: PoseStamped, side=arm.side):
                p, q = msg.pose.position, msg.pose.orientation
                book.on_ee(side, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), stamp(msg))

            node.create_subscription(PoseStamped, arm.ee_pose_topic, on_pose, 10)
        node.create_subscription(
            JointState, topics.joint_states, lambda m: book.on_joints(m.name, m.position, stamp(m)), 10
        )
        for topic in camera_topics:
            node.create_subscription(Image, topic, lambda m, t=topic: book.on_camera(t, stamp(m)), 2)
        self._request_pub = node.create_publisher(String, topics.agent_request, 10)
        self._control_pub = node.create_publisher(String, topics.agent_control, 10)
        self._finish = node.create_client(Trigger, topics.finish_session)
        self._abort = node.create_client(Trigger, topics.abort_session)
        self._ready = node.create_client(Trigger, topics.agent_ready)
        from ibrobot_msgs.srv import AdmitSimulationEpisode

        self._Admit = AdmitSimulationEpisode
        self._admit = node.create_client(AdmitSimulationEpisode, topics.admit_episode)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

    def publish_request(self, payload: dict[str, Any]) -> None:
        self._request_pub.publish(self._String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_control(self, payload: dict[str, Any]) -> None:
        self._control_pub.publish(self._String(data=json.dumps(payload)))

    def _call(self, client, timeout_s: float):
        future = client.call_async(self._Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def call_finish(self) -> str:
        if not self._finish.wait_for_service(timeout_sec=5.0):
            return "finish service unavailable"
        result = self._call(self._finish, 10.0)
        return result.message if result is not None else "finish timed out"

    def abort_session(self) -> str:
        if not self._abort.wait_for_service(timeout_sec=5.0):
            return "abort service unavailable"
        result = self._call(self._abort, 10.0)
        return result.message if result is not None else "abort timed out"

    def admit_episode(self, session_id: str, generation: int | None) -> tuple[bool, str]:
        if not self._admit.wait_for_service(timeout_sec=5.0):
            return False, "admit service unavailable (is agent_eval's runtime extension attached?)"
        request = self._Admit.Request(session_id=str(session_id), generation=int(generation or 0))
        future = self._admit.call_async(request)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        result = future.result() if future.done() else None
        if result is None:
            return False, "admit timed out"
        return bool(result.success), str(result.message)

    def agent_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready.wait_for_service(timeout_sec=2.0):
                result = self._call(self._ready, max(0.0, deadline - time.monotonic()))
                if result is not None and result.success:
                    return True
            time.sleep(1.0)
        return False

    def log(self, text: str) -> None:
        self.node.get_logger().info(text)

    def close(self) -> None:
        self._executor.shutdown()
        self._spin.join(timeout=5.0)
        self.node.destroy_node()


class RunnerSession:
    """rclpy + node + runner for a whole suite; the Agent must be ready before episodes run."""

    def __init__(
        self,
        body: BodyDescription,
        recipe_facts,
        settings: RunnerSettings,
        *,
        topics: Topics = Topics(),
    ):
        self._args = (body, recipe_facts, settings, topics)
        self._ros: RosRunner | None = None
        self.runner: EpisodeRunner | None = None

    def __enter__(self) -> RunnerSession:
        import rclpy

        body, recipe_facts, settings, topics = self._args
        rclpy.init()
        try:
            self._ros = RosRunner(body, topics, camera_topics=tuple(settings.camera_topics))
            context = EpisodeContext(recipe_facts, body.facts())
            self.runner = EpisodeRunner(self._ros.book, self._ros, context, settings, log=self._ros.log)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def wait_agent_ready(self, timeout_s: float) -> bool:
        return self._ros.agent_ready(timeout_s)

    def run_episodes(self, episodes: int, episode_wait_s: float, abort=None) -> list[dict[str, Any]]:
        """Run ``episodes`` episodes; ``abort()`` names a reason to stop waiting (a process exited)."""
        self._ros.book.set_abort_check(abort)
        records = []
        for _ in range(episodes):
            start = self._ros.book.next_episode(episode_wait_s)
            if start is None:
                raise TimeoutError("no episode started in time")
            records.append(self.runner.run_episode(start))
        return records

    def __exit__(self, *exc) -> None:
        import rclpy

        if self._ros is not None:
            self._ros.close()
            self._ros = None
        rclpy.try_shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Agent evaluation episodes as the platform starts them (the runtime must have "
        "agent_eval.runtime_extension:attach in its profile's simulation.<platform>.extensions)."
    )
    parser.add_argument("--out", required=True, help="directory for per-episode records")
    parser.add_argument("--task", required=True, help="task spec name or path")
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--runtime-profile", default=None, help="runtime profile path (default: provider share)")
    parser.add_argument("--recipe", default=None, help="recipe name or path (default: the task spec's)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--decision-timeout", type=float, default=None)
    parser.add_argument("--episode-wait", type=float, default=1800.0, help="max wait for the next episode start")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-episode-tag", action="store_true", help="omit the per-episode random tag")
    args = parser.parse_args(argv)
    spec = load_task_spec(args.task)
    recipe = load_recipe(args.recipe or spec.recipe)
    body = load_body(args.robot_config, profile_path=args.runtime_profile)
    settings = RunnerSettings(
        out_dir=args.out,
        max_decisions=args.max_decisions or spec.max_decisions,
        decision_timeout_s=args.decision_timeout or spec.decision_timeout_s,
        observation_fields=spec.observation_fields,
        episode_tag=not args.no_episode_tag,
        run_id=args.run_id,
    )
    try:
        with RunnerSession(body, recipe.facts(), settings) as session:
            if not session.wait_agent_ready(args.ready_timeout):
                raise RuntimeError("the Agent node is not ready")
            session.run_episodes(args.episodes, args.episode_wait)
    except (RuntimeError, TimeoutError) as exc:
        print(f"agent_eval runner: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
