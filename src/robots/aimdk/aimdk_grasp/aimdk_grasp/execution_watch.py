"""The X2 target watch: is the target still where it was planned, while the arm approaches it.

ExecutionWatcher captures, masks the robot out and votes on its own thread
during the approach and grasp segments; the close ends the watch (the hand is
meant to touch the target then, and lifting and carrying move it with the
hand). ``log`` only records the votes. ``stop`` halts the plan once the votes
decide "moved", and then a frame whose camera shift neither the background
patches nor the chest IMU measure votes "unknown": the shift search alone is
too small for the grasp segment (live 2026-10-13: the image drifted 16-21 px
as the hand reached the bottle; 3 of 6 still grasp runs read moved there).

The watch's state (WatchState) tells a still target from one the watch can
no longer vouch for. In ``stop`` a watch that cannot start refuses the
execution, and one that stops getting fresh frames, or cannot decide for
longer than WatchTimeouts allow, fails it (``failed``): the sequence then
stops the arm in place, unlike a move, which is retraced and planned again.
``log`` records the same failures and lets the execution go on.

The chest IMU (``imu.chest``) runs on a node and executor thread of its own
and is paused while a request plans: its ~500 Hz Python callbacks contended
with perception and planning for the GIL (live 2026-10-14, plan-only, same
bottle: planning_call 1.82 s with them, 1.20 s paused, 1.15 s with the watch
off). The watch needs orientations only around the planned image and during
the execution.

Migrated from the X2 task node's target watch and IMU handling.
"""

from __future__ import annotations

import enum
import math
import threading
import time
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, GraspPlan, Observation, Proposals

from .observation import AdmittedState, LiveState
from .target_watch import WATCHED_PHASES, ExecutionWatcher, OrientationBuffer, WatchConfig

WATCH_MODES = ("log", "stop")
#: The error code of a watch that cannot vouch for the target (TargetWatch.failed).
WATCH_FAILED = "WATCH_FAILED"
WATCH_PERIOD_S = 0.15
CROP_MARGIN_PX = 40
IMU_INTERFACE = "imu.chest"
# The planned image's orientation needs IMU samples on both sides of its
# stamp (OrientationBuffer.at); this much after it is kept before pausing.
IMU_PAUSE_MARGIN_NS = 20_000_000
# IMU samples kept while paused: 1 s of the 500 Hz chest IMU, more than a
# capture's image age plus registration (0.1-0.3 s live).
IMU_QUEUE_DEPTH = 500


class WatchState(enum.IntEnum):
    """What the watch can say about the target now (aimdk_grasp_msgs/WatchStatus.state)."""

    #: Armed; no watched phase has started yet.
    WAITING = 0
    #: The newest fresh frame read the target where it was planned.
    STABLE = 1
    #: The votes decided the target moved.
    MOVED = 2
    #: Frames arrive but do not decide (the hand covers the target, no camera shift measured).
    UNKNOWN = 3
    #: The watch cannot look: it did not start, it stopped, or no fresh frame came in time.
    UNAVAILABLE = 4
    #: The watched phases are over (the close started); nothing is compared any more.
    ENDED = 5


#: No watch timeout may exceed this: past it the watch would, in effect, never time out.
MAX_TIMEOUT_S = 60.0


def finite_seconds(value: Any, minimum: float, maximum: float) -> bool:
    """A real number (not a bool, not NaN or infinite) above ``minimum`` and at most ``maximum``."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and minimum < value <= maximum
    )


@dataclass(frozen=True)
class WatchTimeouts:
    """How long a watch in a watched phase may go without what it needs, on its own host's clock.

    From 123 armed executions of the X2 watch on the robot (2026-10-12..20,
    ~/aimdk/log/validation/live_exec_*/**/report.json, per-frame records):

    - ``frame_s``: no fresh frame for this long and the watch is unavailable.
      Frames came every 0.21 s (p50), 0.47 s (p99), 0.55 s at most.
    - ``unknown_s``: fresh frames that all fail to decide for this long and the
      target is not observable. Since 2026-10-13 the longest such run was
      2.62 s, always the hand covering the target at the end of the grasp
      segment; a watch blind for its whole execution (2026-10-15 p_stop st1,
      no IMU orientation) stayed undecided for 9.4 s.
    - ``image_age_s``: an image older than this when the watch takes it is not
      fresh. Ages were 0.17 s (p50), 0.24 s (p99), 0.34 s at most.
    """

    frame_s: float = 1.0
    unknown_s: float = 4.0
    image_age_s: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not finite_seconds(value, 0.0, MAX_TIMEOUT_S):
                raise ValueError(f"watch timeout {name} must be a finite number of seconds in (0, {MAX_TIMEOUT_S:g}]")


@dataclass(frozen=True)
class WatchStatus:
    state: WatchState
    phase: str = ""
    #: Newer images taken while a watched phase ran, compared or not.
    fresh_frames: int = 0
    last_image_stamp_ns: int = 0
    #: UNKNOWN: how long no frame has decided.
    unknown_s: float = 0.0
    reason: str = ""


def watch_failure(status: WatchStatus, timeouts: WatchTimeouts) -> str | None:
    """Why a watch in ``status`` no longer vouches for the target, or None while it does (or need not)."""
    if status.state is WatchState.UNAVAILABLE:
        return f"watch unavailable: {status.reason}"
    if status.state is WatchState.UNKNOWN and status.unknown_s > timeouts.unknown_s:
        return f"target not observable for {status.unknown_s:.1f} s (over {timeouts.unknown_s:g} s): {status.reason}"
    return None


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
    """``TargetWatch`` over ExecutionWatcher, with its state (``status``) and ``failed``."""

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
        timeouts: WatchTimeouts = WatchTimeouts(),
        on_moved: Callable[[str], None] | None = None,
        source_clock: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        side: str = "left",
    ):
        """``on_moved(reason)`` is told once the target has moved (the grasp head publishes it at once).

        ``side`` is the moving arm: its spheres (from ``spheres``) are left out of the compared pixels.
        """
        if mode not in WATCH_MODES:
            raise ValueError(f"watch mode must be one of {WATCH_MODES}")
        self.capture_state, self.spheres, self.mode, self.imu = capture_state, spheres, mode, imu
        self.period_s, self.crop_margin_px, self.config = period_s, crop_margin_px, config
        self.timeouts, self.on_moved, self.source_clock, self.clock = timeouts, on_moved, source_clock, clock
        self.side = side
        self._watcher: ExecutionWatcher | None = None
        self._armed = False
        self._phase, self._moved, self._error, self._fault = "", None, None, None
        self._watching_since: float | None = None
        self._ended = False
        #: The failing state the watch was in as its watched steps ended, kept as its last word.
        self._ended_failing: WatchStatus | None = None

    def arm(self, observation: Observation, proposals: Proposals, plan: GraspPlan) -> None:
        """Start watching; in ``stop`` raise ``GraspFailure`` (WATCH_FAILED) when the watch cannot start."""
        previous, self._watcher = self._watcher, None
        if previous is not None:
            # Never two watchers: an old one would wake on the new phases and
            # compare against an old image and mask.
            previous.stop()
        self._armed = True
        self._phase, self._moved, self._error, self._fault = "", None, None, None
        self._watching_since, self._ended, self._ended_failing = None, False, None
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
                clock=self.clock,
                orientation=None if self.imu is None else self.imu.orientation.at,
                require_orientation=stop,
                on_moved=self._on_moved if stop else None,
                source_clock=self.source_clock,
                max_image_age_s=self.timeouts.image_age_s,
                side=self.side,
            ).start()
        except Exception as exc:  # noqa: BLE001 - reported; stop refuses to execute without the watch
            self._watcher, self._error = None, f"not started: {type(exc).__name__}: {exc}"
            self._fault = f"watch unavailable: {self._error}"
            if self.mode == "stop":
                raise GraspFailure(WATCH_FAILED, f"target watch {self._error}") from exc

    @property
    def started(self) -> bool:
        """The last arm started a watcher (``log`` arms without one when it cannot start)."""
        return self._watcher is not None

    def phase(self, name: str, event: str) -> None:
        if event != "start":
            return
        if name in WATCHED_PHASES:
            if self._watching_since is None:
                self._watching_since = self.clock()
        elif self._watching_since is not None and not self._ended:
            # No new judgements from here on, but what the watch already found stands:
            # a failure reached during the watched steps is its last word.
            final = self.status()
            if watch_failure(final, self.timeouts) is not None:
                self._ended_failing = final
            self._ended = True
        self._phase = name

    def _on_moved(self, record: Mapping) -> None:
        self._moved = moved_reason(record)
        if self.on_moved is not None:
            self.on_moved(self._moved)

    def status(self) -> WatchStatus:
        """What the watch can say now; the first failure it shows is kept for the report."""
        status = self._status()
        if self._fault is None:
            self._fault = watch_failure(status, self.timeouts)
        return status

    def _status(self) -> WatchStatus:
        watcher, phase = self._watcher, self._phase
        if watcher is None:
            return WatchStatus(WatchState.UNAVAILABLE, phase, reason=self._error or "not armed")
        progress = dict(phase=phase, fresh_frames=watcher.fresh_frames, last_image_stamp_ns=watcher.last_image_stamp_ns)
        # Only a verdict accepted by ExecutionWatcher's dispatch gate crosses
        # this state interface. Raw votes may finish after close/stop and must
        # not make the remote pipeline retract a sound grasp.
        if watcher.notified is not None:
            return WatchStatus(WatchState.MOVED, reason=self._moved or moved_reason(watcher.notified), **progress)
        if self._ended:
            return self._ended_failing or WatchStatus(WatchState.ENDED, **progress)
        if self._watching_since is None:
            return WatchStatus(WatchState.WAITING, **progress)
        if watcher.error is not None:
            return WatchStatus(WatchState.UNAVAILABLE, reason=f"watch stopped: {watcher.error}", **progress)
        if not watcher.running:
            return WatchStatus(WatchState.UNAVAILABLE, reason="watch stopped", **progress)
        now = self.clock()
        fresh = max(self._watching_since, watcher.last_fresh_at or self._watching_since)
        if now - fresh > self.timeouts.frame_s:
            why = watcher.last_skip or "no frame taken"
            return WatchStatus(
                WatchState.UNAVAILABLE, reason=f"no fresh image for {now - fresh:.1f} s ({why})", **progress
            )
        if watcher.last_vote == "still":
            return WatchStatus(WatchState.STABLE, **progress)
        decided = max(self._watching_since, watcher.last_still_at or self._watching_since)
        return WatchStatus(WatchState.UNKNOWN, unknown_s=now - decided, reason=self._undecided(watcher), **progress)

    @staticmethod
    def _undecided(watcher: ExecutionWatcher) -> str:
        # The newest frame that says why; polls between camera frames ("not newer") do not.
        record = next((row for row in reversed(watcher.records) if row.get("skipped") != "not newer"), {})
        if "skipped" in record:
            return record["skipped"]
        if "unknown" in record:
            return record["unknown"]
        if record.get("vote") == "moved":
            return "a frame voted moved"
        if "visible_fraction" in record:
            return f"target {record['visible_fraction']:.0%} visible"
        return "no frame compared yet"

    def moved(self) -> str | None:
        return self._moved

    def failed(self) -> str | None:
        """In ``stop``, why the watch no longer vouches for the target; ``log`` only records it."""
        if not self._armed:
            return None
        reason = watch_failure(self.status(), self.timeouts)
        return reason if self.mode == "stop" else None

    @property
    def armed(self) -> bool:
        """Between arm and disarm: an execution this watch belongs to is under way."""
        return self._armed

    def disarm(self) -> Mapping:
        state = self.status().state.name.lower() if self._armed else None
        watcher, self._watcher, self._armed = self._watcher, None, False
        if self.imu is not None:
            self.imu.pause()  # nothing watches until the next request has planned
        summary = dict(mode=self.mode, state=state, fault=self._fault)
        if watcher is None:
            return dict(summary, error=self._error)
        watcher.stop()
        report = watcher.report()
        report.pop("records")  # per-frame votes; too large for the action result
        return dict(report, **summary, halted=self._moved)
