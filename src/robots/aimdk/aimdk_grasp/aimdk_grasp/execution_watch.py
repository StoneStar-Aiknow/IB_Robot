"""The X2 target watch: is the target still where it was planned, while the arm approaches it.

ExecutionWatcher captures, masks the robot out and votes on its own thread
during the approach and grasp segments; the close ends the watch (the hand is
meant to touch the target then). ``log`` only records the votes. ``stop`` halts
the plan once the votes decide "moved", and then a frame whose camera shift
neither the background patches nor the chest IMU measure votes "unknown":
the shift search alone is too small for the grasp segment (live 2026-10-13:
the image drifted 16-21 px as the hand reached the bottle; 3 of 6 still grasp
runs read moved there).

The chest IMU (``imu.chest``) runs on a node and executor thread of its own
and is paused while a request plans: its ~500 Hz Python callbacks contended
with perception and planning for the GIL (live 2026-10-14, plan-only, same
bottle: planning_call 1.82 s with them, 1.20 s paused, 1.15 s with the watch
off). The watch needs orientations only around the planned image and during
the execution.

Migrated from the X2 task node's target watch and IMU handling.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from copy import copy
from types import SimpleNamespace
from typing import Any

from manipulation_execution.grasp_pipeline.contracts import GraspPlan, Observation, Proposals

from .observation import AdmittedState, LiveState
from .target_watch import ExecutionWatcher, OrientationBuffer, WatchConfig

WATCH_MODES = ("log", "stop")
WATCH_PERIOD_S = 0.15
CROP_MARGIN_PX = 40
IMU_INTERFACE = "imu.chest"
# The planned image's orientation needs IMU samples on both sides of its
# stamp (OrientationBuffer.at); this much after it is kept before pausing.
IMU_PAUSE_MARGIN_NS = 20_000_000
# IMU samples kept while paused: 1 s of the 500 Hz chest IMU, more than a
# capture's image age plus registration (0.1-0.3 s live).
IMU_QUEUE_DEPTH = 500


def moved_reason(record: Mapping) -> str:
    votes = [round(record.get(key, 0.0), 3) for key in ("depth_changed_fraction", "rgb_changed_fraction")]
    return f"target moved (votes {votes})"


class ChestImu:
    """Body orientations from the runtime's chest IMU, by header stamp; paused between images and while idle."""

    def __init__(self, node, description: Mapping[str, Any]):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu

        from robot_config.contract_utils import qos_profile_from_dict

        interface = (description.get("interfaces") or {}).get(IMU_INTERFACE) or {}
        shape = (interface.get("kind"), interface.get("direction"), interface.get("message_type"))
        if shape != ("topic", "publish", "sensor_msgs/msg/Imu"):
            raise ValueError(f"interface description must publish {IMU_INTERFACE} as a sensor_msgs/msg/Imu topic")
        qos = qos_profile_from_dict(interface.get("qos")) or copy(qos_profile_sensor_data)
        qos.depth = IMU_QUEUE_DEPTH
        self.orientation = OrientationBuffer()
        self._pause_after: int | None = None
        self._spinning, self._stopped = threading.Event(), threading.Event()
        self._spinning.set()
        # Not the process's global arguments: a launch's -r __node:= would name it like the owner.
        self._node = rclpy.create_node(node.get_name() + "_imu", context=node.context, use_global_arguments=False)
        self._subscription = self._node.create_subscription(Imu, str(interface["endpoint"]), self._on_imu, qos)
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True, name="chest_imu")
        self._thread.start()

    def _on_imu(self, message) -> None:
        q = message.orientation
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        self.orientation.add(stamp, (q.x, q.y, q.z, q.w))
        pause_after = self._pause_after
        if pause_after is not None and stamp >= pause_after:
            self._spinning.clear()

    def _spin(self) -> None:
        while not self._stopped.is_set():
            self._spinning.wait()
            if not self._stopped.is_set():
                # Queued samples first, without the executor's wait per sample:
                # 500 after a pause took 0.19 s through spin_once on the idle Orin,
                # 0.04 s so (live 2026-10-15 p_stop st3: the watch's first 0.6 s,
                # as the arm set off, had no orientation and never voted).
                while self._spinning.is_set() and self._take():
                    pass
                self._executor.spin_once(timeout_sec=0.1)

    def _take(self) -> bool:
        """Pass one queued IMU sample to its callback; False when none is queued."""
        subscription = self._subscription
        with subscription.handle:
            taken = subscription.handle.take_message(subscription.msg_type, subscription.raw)
        if taken is None:
            return False
        self._on_imu(taken[0])
        return True

    def pause_after(self, image_stamp_ns: int) -> None:
        """Keep taking samples until one is past the image, then pause (no Python callbacks)."""
        self._pause_after = int(image_stamp_ns) + IMU_PAUSE_MARGIN_NS
        self._spinning.set()

    def pause(self) -> None:
        """Stop taking samples now; they wait in the queue until the next image or resume.

        Taking every ~440 Hz sample in Python while nothing executes cost the
        pipeline a third of its idle GIL on the Orin (live 2026-10-21).
        """
        self._pause_after = None
        self._spinning.clear()

    @property
    def paused(self) -> bool:
        return not self._spinning.is_set()

    def resume(self) -> None:
        self._pause_after = None
        self._spinning.set()

    def close(self) -> None:
        self._stopped.set()
        self._spinning.set()
        self._thread.join(2.0)
        self._executor.shutdown(timeout_sec=2.0)
        self._node.destroy_node()


class X2TargetWatch:
    """``TargetWatch`` over ExecutionWatcher; a watch that cannot start only observes nothing."""

    def __init__(
        self,
        capture_state: Callable[[], LiveState],
        spheres: Callable[[Any], Any],
        *,
        mode: str = "stop",
        imu: ChestImu | None = None,
        period_s: float = WATCH_PERIOD_S,
        crop_margin_px: int = CROP_MARGIN_PX,
        config: WatchConfig = WatchConfig(),
        source_clock: Callable[[], float] | None = None,
    ):
        if mode not in WATCH_MODES:
            raise ValueError(f"watch mode must be one of {WATCH_MODES}")
        self.capture_state, self.spheres, self.mode, self.imu = capture_state, spheres, mode, imu
        self.period_s, self.crop_margin_px, self.config = period_s, crop_margin_px, config
        self.source_clock = source_clock
        self._watcher: ExecutionWatcher | None = None
        self._phase, self._moved, self._error = "", None, None

    def arm(self, observation: Observation, proposals: Proposals, plan: GraspPlan) -> None:
        previous, self._watcher = self._watcher, None
        if previous is not None:
            # Never two watchers: an old one would wake on the new phases and
            # compare against an old image and mask.
            previous.stop()
        self._phase, self._moved, self._error = "", None, None
        admitted = observation.robot_state
        try:
            if not isinstance(admitted, AdmittedState):
                raise TypeError("the X2 watch compares X2 captures")
            planned = admitted.live.observation
            request = SimpleNamespace(
                mask=proposals.target_mask,
                rgb=planned.frame.rgb,
                depth=planned.frame.depth,
                intrinsics=planned.frame.intrinsics,
                transform=planned.transform,
                image_feedback=planned.image_feedback,
                feedback=admitted.live.feedback,
            )
            stop = self.mode == "stop"
            self._watcher = ExecutionWatcher(
                request,
                self.capture_state,
                self.spheres,
                lambda: self._phase,
                config=self.config,
                period_s=self.period_s,
                crop_margin_px=self.crop_margin_px,
                orientation=None if self.imu is None else self.imu.orientation.at,
                require_orientation=stop,
                on_moved=self._on_moved if stop else None,
                source_clock=self.source_clock,
            ).start()
        except Exception as exc:  # noqa: BLE001 - the watch only observes; execution goes on without it
            self._watcher, self._error = None, f"not started: {type(exc).__name__}: {exc}"

    def phase(self, name: str, event: str) -> None:
        if event == "start":
            self._phase = name

    def _on_moved(self, record: Mapping) -> None:
        self._moved = moved_reason(record)

    def moved(self) -> str | None:
        return self._moved

    def disarm(self) -> Mapping:
        watcher, self._watcher = self._watcher, None
        if self.imu is not None:
            self.imu.pause()  # nothing watches until the next request has planned
        if watcher is None:
            return dict(mode=self.mode, error=self._error)
        watcher.stop()
        report = watcher.report()
        report.pop("records")  # per-frame votes; too large for the action result
        return dict(report, mode=self.mode, halted=self._moved)
