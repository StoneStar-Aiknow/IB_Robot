"""ROS adapter for the X2 mirror photo (``mirror_photo`` package).

During every MIRRORING round the photo node keeps candidate frames from the
camera the session already reads, picks one photo when the round ends and
sends it, with its person box, to the upper PC. All the deciding is in
``mirror_photo``; this node only subscribes, runs two threads and publishes
``~/status``. It never publishes anything the mirror reads, and the mirror
never waits on it: a node that cannot start logs an ERROR and exits.

Parameters (filled by launch, see ``PHOTO_PARAMETERS``):

``photo_json``         the checked ``mirror_session.photo`` section
                       (``mirror_photo.config.to_json``); not readable or not
                       enabled: ERROR and a clean exit
``camera_topic``       ``sensor_msgs/CompressedImage``, the stream PEAR reads
``pear_frame_topic``   ``ibrobot_msgs/PearFrame``; empty: no held-pose trigger
                       and no person box
``hand_frame_topic``   ``ibrobot_msgs/HandFrame``; empty: no gesture trigger
``status_period_sec``  ``~/status`` period (default 1 s)

Interfaces:

* subscribes ``/mirror_session/state`` (reliable + transient_local, depth 1,
  as the operator); its arrival time drives the round (``STATE_STALE_SEC``);
* subscribes the camera, the PEAR frames and the hand frames with the sensor
  data QoS (best effort, as the retarget node): a frame missed is a frame
  not offered, nothing more. Frames are kept as the JPEG bytes the camera
  sent, never decoded in a callback;
* subscribes ``mirror_photo_contract.TRIGGER_TOPIC`` (``std_msgs/String``,
  depth ``TRIGGER_DEPTH``): an external request for a candidate;
* ``~/status`` (``std_msgs/String``, JSON snapshot, every
  ``status_period_sec``).

Threads: the ROS callbacks (one executor thread, ``PhotoCore``), the round
worker (``mirror_photo.worker``: choosing, the outbox on disk) and the
uploader (``mirror_photo.uploader``: HTTP to the upper PC, retried until it
confirms). The node runs at a lower priority (``nice`` +10) than the mirror.
On exit a running round is ended and its photo queued on disk before the
threads stop; a photo not yet delivered is sent after the next start.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from embodied_common.mirror_photo_contract import NODE_NAME, TRIGGER_DEPTH, TRIGGER_TOPIC, parse_trigger
from embodied_common.mirror_session_contract import STATE_TOPIC
from ibrobot_msgs.msg import HandFrame, PearFrame
from manipulation_execution.mirror_photo.config import PhotoSection, from_json
from manipulation_execution.mirror_photo.outbox import Outbox
from manipulation_execution.mirror_photo.selector import PhotoCore
from manipulation_execution.mirror_photo.uploader import Uploader
from manipulation_execution.mirror_photo.worker import RoundWorker

TICK_PERIOD_S = 0.2
NICE = 10
# The source of a PEAR frame that found a person; any other (center_fallback)
# has no actor: its pose is no one's and its box no one's.
PEAR_PERSON_SOURCE = "yolox"
# How long the threads get to finish on exit, past the scoring deadline / the send timeout.
JOIN_MARGIN_SEC = 3.0

# Every parameter the node declares, with its default. Launch may pass no other
# name and no other type: rclpy drops an override it has no declaration for
# without a warning, and refuses one of the wrong type at startup.
PHOTO_PARAMETERS = {
    "photo_json": "",
    "camera_topic": "",
    "pear_frame_topic": "",
    "hand_frame_topic": "",
    "status_period_sec": 1.0,
}


@dataclass(frozen=True)
class PhotoSettings:
    """The parameters, checked. No ROS."""

    section: PhotoSection
    camera_topic: str
    pear_frame_topic: str
    hand_frame_topic: str
    status_period_sec: float

    @classmethod
    def from_parameters(cls, values: Mapping[str, Any]) -> PhotoSettings:
        section = from_json(values["photo_json"])
        if not section.enabled:
            raise ValueError("photo_json: the photo section is not enabled")
        for name in ("camera_topic", "pear_frame_topic", "hand_frame_topic"):
            value = values[name]
            if not isinstance(value, str) or value.strip() != value:
                raise ValueError(f"{name} must be a ROS name")
            if value and not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute ROS name")
        if not values["camera_topic"]:
            raise ValueError("camera_topic is required")
        period = values["status_period_sec"]
        if isinstance(period, bool) or not isinstance(period, int | float) or not period > 0.0:
            raise ValueError("status_period_sec must be positive")
        return cls(
            section=section,
            camera_topic=values["camera_topic"],
            pear_frame_topic=values["pear_frame_topic"],
            hand_frame_topic=values["hand_frame_topic"],
            status_period_sec=float(period),
        )


def _stamp(header) -> float:
    return int(header.stamp.sec) + int(header.stamp.nanosec) * 1e-9


class MirrorPhotoNode(Node):
    def __init__(self, **kwargs):
        super().__init__(NODE_NAME, **kwargs)
        for name, default in PHOTO_PARAMETERS.items():
            self.declare_parameter(name, default)
        self.core: PhotoCore | None = None
        self._threads: list[threading.Thread] = []
        try:
            settings = PhotoSettings.from_parameters(
                {name: self.get_parameter(name).value for name in PHOTO_PARAMETERS}
            )
        except ValueError as exc:
            self.get_logger().error(f"mirror photo off, the mirror goes on without photos: {exc}")
            return
        section = settings.section
        upload = section.upload
        self.outbox = Outbox(
            section.photos_dir,
            max_files=upload.outbox_max_files,
            max_mb=upload.outbox_max_mb,
            min_free_gb=upload.min_free_gb,
            log=self._log,
        )
        try:
            self.outbox.prepare()
        except OSError as exc:
            self.get_logger().error(
                f"mirror photo off, the mirror goes on without photos: cannot use {self.outbox.path} ({exc})"
            )
            return
        self._settings = settings
        self.uploader = Uploader(
            self.outbox,
            target_file=section.target_file,
            default_target=upload.default_target,
            timeout_sec=upload.timeout_sec,
            log=self._log,
        )
        self.worker = RoundWorker(section, self.outbox, wake_uploader=self.uploader.wake, log=self._log)
        self.core = PhotoCore(section, on_round_end=self.worker.finished, on_provisional=self.worker.provisional)
        self._inputs = dict.fromkeys(("images", "unstamped", "pear", "hands", "triggers", "bad_triggers"), 0)

        self.create_subscription(
            String,
            STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(CompressedImage, settings.camera_topic, self._on_image, qos_profile_sensor_data)
        if settings.pear_frame_topic:
            self.create_subscription(PearFrame, settings.pear_frame_topic, self._on_pear, qos_profile_sensor_data)
        if settings.hand_frame_topic:
            self.create_subscription(HandFrame, settings.hand_frame_topic, self._on_hand, qos_profile_sensor_data)
        self.create_subscription(String, TRIGGER_TOPIC, self._on_trigger, TRIGGER_DEPTH)
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self.create_timer(TICK_PERIOD_S, self._tick)
        self.create_timer(settings.status_period_sec, self._publish_status)

        for name, target in (("photo-worker", self.worker.run), ("photo-upload", self.uploader.run)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.get_logger().info(
            f"mirror photo: {settings.camera_topic} during MIRRORING -> one photo per round -> "
            f"upper PC from {section.target_file} (default {upload.default_target or 'none'}); "
            f"outbox {self.outbox.path} ({len(self.outbox.names())} waiting, at most {upload.outbox_max_files}); "
            f"PEAR {settings.pear_frame_topic or 'off'}, hands {settings.hand_frame_topic or 'off'}, "
            f"events {TRIGGER_TOPIC}"
        )

    def _log(self, level: str, message: str) -> None:
        # rclpy pins the severity of each call site; keep one call per level.
        # Called from the worker and upload threads too: rclpy's logger is thread-safe.
        if level == "error":
            self.get_logger().error(message)
        elif level == "warning":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    # -- inputs ------------------------------------------------------------------------

    def _on_state(self, message: String) -> None:
        self.core.on_state(message.data, time.monotonic())

    def _tick(self) -> None:
        self.core.tick(time.monotonic())

    def _on_image(self, message: CompressedImage) -> None:
        if not self.core.mirroring:
            return
        self._inputs["images"] += 1
        stamp = _stamp(message.header)
        if stamp == 0.0:
            self._inputs["unstamped"] += 1
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self.core.on_image(stamp, bytes(message.data))

    def _on_pear(self, message: PearFrame) -> None:
        if not self.core.mirroring:
            return
        self._inputs["pear"] += 1
        person = message.source == PEAR_PERSON_SOURCE
        raw = np.asarray(message.smplx_pose_raw, dtype=np.float64) if person else np.empty(0)
        self.core.on_pear(_stamp(message.header), raw, tuple(message.bbox) if person else None)

    def _on_hand(self, message: HandFrame) -> None:
        if not self.core.mirroring:
            return
        self._inputs["hands"] += 1
        seen = [
            (hand.gesture, float(hand.gesture_score))
            for hand in (message.left, message.right)
            if hand.detected and hand.gesture
        ]
        self.core.on_hand(_stamp(message.header), seen)

    def _on_trigger(self, message: String) -> None:
        try:
            request = parse_trigger(message.data)
        except ValueError as exc:
            self._inputs["bad_triggers"] += 1
            self._log("warning", f"mirror photo: trigger on {TRIGGER_TOPIC} ignored: {exc}")
            return
        self._inputs["triggers"] += 1
        self.core.on_event(request)

    # -- status ------------------------------------------------------------------------

    def _publish_status(self) -> None:
        status = self.core.status()
        status["inputs"] = dict(self._inputs)
        status["worker"] = self.worker.status()
        status["upload"] = self.uploader.status()
        status["outbox"] = self.outbox.stats()
        self._status_publisher.publish(String(data=json.dumps(status, ensure_ascii=False, sort_keys=True)))

    # -- exit --------------------------------------------------------------------------

    def finish(self) -> None:
        """End a running round, let its photo reach the outbox, stop the threads. No ROS call."""
        if self.core is None:
            return
        self.core.shutdown()
        self.worker.stop()
        worker, upload = self._threads
        worker.join(self._settings.section.scoring_deadline_sec + JOIN_MARGIN_SEC)
        if worker.is_alive():
            self._log("error", "mirror photo: the last round was not queued in time; its crash copy is sent next start")
        self.uploader.stop()
        upload.join(self._settings.section.upload.timeout_sec + 1.0)


def main(args=None) -> None:
    # Behind the mirror for the CPU; a failure here only leaves the priority as it was.
    with contextlib.suppress(OSError):
        os.nice(NICE)
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    node = None
    try:
        node = MirrorPhotoNode()
        if node.core is not None:
            executor.add_node(node)
            executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.finish()
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
