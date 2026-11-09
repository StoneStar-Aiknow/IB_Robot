"""Observe the current mirror actor and publish confirmed pose-match events.

This node has no actuator publisher or runtime action client. An accepted
match is a perception event, not a claim that a robot motion was executed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from ibrobot_msgs.msg import ActorState, PearFrame
from manipulation_execution.preset_motions import HeartBounds, HeartRuleConfig, MatcherConfig, PoseMatcher
from manipulation_execution.preset_motions.config import (
    MATCH_TOPIC,
    STATE_TOPIC,
    STATUS_TOPIC,
    matcher_settings,
    topic,
)
from manipulation_execution.preset_motions.events import match_event
from manipulation_execution.preset_motions.session_gate import SessionGate


def make_config(value):
    settings = matcher_settings(value)
    rule = settings.pop("rule", {})
    defaults = HeartRuleConfig()
    enter = HeartBounds(**{**asdict(defaults.enter), **rule.get("enter", {})})
    keep = HeartBounds(**{**asdict(defaults.keep), **rule.get("keep", {})})
    return MatcherConfig(rule=HeartRuleConfig(enter=enter, keep=keep), **settings)


class PresetMotionMatcherNode(Node):
    def __init__(self):
        super().__init__("preset_motion_matcher")
        defaults = {
            "pear_frame_topic": "/motion_imitation_stream/pear_frame",
            "actor_state_topic": "/motion_imitation_stream/actor_state",
            "session_state_topic": STATE_TOPIC,
            "match_topic": MATCH_TOPIC,
            "status_topic": STATUS_TOPIC,
            "matcher_json": "{}",
            "motion_enabled": False,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        if self.get_parameter("motion_enabled").value is not False:
            raise ValueError("observer motion output must remain disabled until handoff validation")
        self._matcher = PoseMatcher(make_config(json.loads(self.get_parameter("matcher_json").value)))
        self._gate = SessionGate()
        self._result = None
        self._events = 0
        group = MutuallyExclusiveCallbackGroup()
        self._events_pub = self.create_publisher(String, self._topic("match_topic"), 10)
        self._status_pub = self.create_publisher(String, self._topic("status_topic"), 10)
        self.create_subscription(
            PearFrame, self._topic("pear_frame_topic"), self._pear, qos_profile_sensor_data, callback_group=group
        )
        self.create_subscription(
            ActorState, self._topic("actor_state_topic"), self._actor, qos_profile_sensor_data, callback_group=group
        )
        self.create_subscription(String, self._topic("session_state_topic"), self._state, 10, callback_group=group)
        self.create_timer(0.05, self._tick, callback_group=group)
        self.create_timer(1.0, self._status, callback_group=group)
        self.get_logger().info("overhead-heart observer enabled; robot motion output remains disabled")

    def _topic(self, name):
        return topic(self.get_parameter(name).value, name)

    def _times(self):
        return self.get_clock().now().nanoseconds / 1e9, time.monotonic()

    def _permit(self, ros, mono):
        epoch, active = self._gate.permit(now_ros_s=ros, now_mono_s=mono)
        self._accept(self._matcher.set_session(epoch=epoch, active=active))
        self._accept(self._matcher.tick(now_ros_s=ros, now_mono_s=mono))

    def _accept(self, result):
        self._result = result
        event = match_event(result, self.get_clock().now().nanoseconds)
        if event is not None:
            self._events += 1
            self._events_pub.publish(String(data=json.dumps(event, sort_keys=True, allow_nan=False)))
            self.get_logger().info(f"matched {result.template_name}: score={result.score:.3f} actor={result.actor_id}")

    def _state(self, message):
        ros, mono = self._times()
        self._gate.observe(message.data, now_ros_s=ros, now_mono_s=mono)
        self._permit(ros, mono)

    def _actor(self, message):
        ros, mono = self._times()
        self._permit(ros, mono)
        self._matcher.observe_actor(message, now_mono_s=mono)

    def _pear(self, message):
        ros, mono = self._times()
        self._permit(ros, mono)
        self._accept(self._matcher.observe_pear(message, now_ros_s=ros, now_mono_s=mono))

    def _tick(self):
        ros, mono = self._times()
        self._permit(ros, mono)

    def _status(self):
        result = asdict(self._result) if self._result is not None else {}
        result.update(
            schema_version=1,
            events=self._events,
            motion_enabled=False,
            motion_state="disabled_pending_validated_template_and_handoff",
        )
        self._status_pub.publish(String(data=json.dumps(result, sort_keys=True, allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PresetMotionMatcherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
