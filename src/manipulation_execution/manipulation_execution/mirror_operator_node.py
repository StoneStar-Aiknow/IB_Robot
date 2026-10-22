"""ROS adapter for the X2 mirror operator (``mirror_operator`` package).

The operator watches the camera for cover gestures and the head for a touch,
and turns them into calls to the Gateway and the mirror session; the light
strip follows the session. All the deciding is in ``mirror_operator.core``;
this node only subscribes, makes the calls it asks for and publishes
``~/status``.

Parameters (filled by launch):

``runtime_description_json``  the bound ``runtime.interface_description``;
                              ``touch.head`` is taken from it by ID
``camera_topic``              the camera, ``sensor_data`` QoS
``camera_kind``               ``compressed`` (``sensor_msgs/CompressedImage``)
                              or ``image`` (``sensor_msgs/Image``)
``business_config``           absolute path of the business config file; its
                              ``operator`` and ``cover_judge`` sections are
                              read; empty: every section empty
``cover_judge``               ``""`` (never covered), ``demo`` or
                              ``pkg.mod:Class`` (``mirror_operator.judge``)
``send_enabled``              false (default): every call is recorded in
                              ``~/status`` as not sent, nothing is sent
``gateway_status_service``    the Gateway's ``GetSkillGatewayStatus``,
``gateway_validate_service``  ``ValidateSkill`` and ``SkillCommand`` action
``gateway_skill_action``      (``robot_skill_cli``'s defaults)
``status_period_sec``         ``~/status`` period (default 1 s)

Interfaces:

* subscribes the camera; frames over ``operator.frame_rate_max_hz`` are
  dropped before decoding;
* subscribes ``touch.head`` with the type named in the runtime description
  and the QoS given there; with ``touch.head`` missing or its type not
  loadable, the touch is off and ``~/status`` says why;
* subscribes ``/mirror_session/state`` (reliable + transient_local, depth 1);
  silence is measured by this node's receive time;
* calls the Gateway (double cover in WAIT: ``mirror_start``), the session's
  ``~/set_session`` with false (long cover in STANDBY/MIRRORING) and its
  ``~/operator_touch`` (head touch); a touch the session answers with
  ``success=False`` ("nothing to do") is a normal answer;
* sets the light (``operator.led``, type loaded by name); with the type not
  loadable, the light is off and ``~/status`` says why;
* ``~/status`` (``std_msgs/String``, JSON snapshot, every
  ``status_period_sec``).

Every call is asynchronous; ``core`` frees a channel when its call times out
and ignores a late answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from unique_identifier_msgs.msg import UUID

from embodied_common.mirror_session_contract import (
    OPERATOR_NODE_NAME,
    OPERATOR_TOUCH_SERVICE,
    SESSION_SERVICE,
    STATE_TOPIC,
)
from ibrobot_msgs.action import SkillCommand
from ibrobot_msgs.srv import GetSkillGatewayStatus, ValidateSkill
from manipulation_execution.mirror_operator import vendor
from manipulation_execution.mirror_operator.config import LedColor, OperatorConfig
from manipulation_execution.mirror_operator.core import Done, LedDone, Operator, TouchEndpoint, resolve_touch_endpoint
from manipulation_execution.mirror_operator.frames import FrameThrottle, gray_from_compressed, gray_from_image
from manipulation_execution.mirror_operator.gateway import GatewayError, MirrorStartRequest
from manipulation_execution.mirror_operator.judge import CoverJudge, build_cover_judge
from manipulation_execution.mirror_session.business import load_business_config

TICK_PERIOD_S = 0.1
CAMERA_KINDS = {"compressed": (CompressedImage, gray_from_compressed), "image": (Image, gray_from_image)}

# Every parameter the node declares, with its default. Launch may pass no other
# name and no other type: rclpy drops an override it has no declaration for
# without a warning, and refuses one of the wrong type at startup.
OPERATOR_PARAMETERS = {
    "runtime_description_json": "",
    "camera_topic": "",
    "camera_kind": "compressed",
    "business_config": "",
    "cover_judge": "",
    "send_enabled": False,
    "gateway_status_service": "/embodied/get_skill_gateway_status",
    "gateway_validate_service": "/embodied/validate_skill",
    "gateway_skill_action": "/embodied/execute_skill",
    "status_period_sec": 1.0,
}
_ROS_NAMES = ("camera_topic", "gateway_status_service", "gateway_validate_service", "gateway_skill_action")


@dataclass(frozen=True)
class OperatorSettings:
    """The parameters, checked. No ROS."""

    description: dict
    camera_topic: str
    camera_kind: str
    business_config: str
    cover_judge: str
    send_enabled: bool
    gateway_status_service: str
    gateway_validate_service: str
    gateway_skill_action: str
    status_period_sec: float

    @classmethod
    def from_parameters(cls, values: Mapping[str, Any]) -> OperatorSettings:
        try:
            description = json.loads(values["runtime_description_json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime_description_json must be the bound runtime interface description") from exc
        if not isinstance(description, dict):
            raise ValueError("runtime_description_json must be the bound runtime interface description")
        for name in _ROS_NAMES:
            value = values[name]
            if not isinstance(value, str) or value.strip() != value or not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute ROS name")
        if values["camera_kind"] not in CAMERA_KINDS:
            raise ValueError(f"camera_kind must be one of {sorted(CAMERA_KINDS)}")
        for name in ("business_config", "cover_judge"):
            if not isinstance(values[name], str):
                raise ValueError(f"{name} must be a string")
        if type(values["send_enabled"]) is not bool:
            raise ValueError("send_enabled must be a boolean")
        period = values["status_period_sec"]
        if isinstance(period, bool) or not isinstance(period, int | float) or not period > 0.0:
            raise ValueError("status_period_sec must be positive")
        return cls(
            description=description,
            status_period_sec=float(period),
            **{
                name: values[name]
                for name in OPERATOR_PARAMETERS
                if name not in ("runtime_description_json", "status_period_sec")
            },
        )


@dataclass(frozen=True)
class OperatorParts:
    """What the settings and the business config make, before any ROS object. No ROS."""

    config: OperatorConfig
    judge: CoverJudge
    touch: TouchEndpoint | None
    touch_type: Any
    touch_unavailable: str
    led_type: Any
    led_unavailable: str

    @classmethod
    def build(cls, settings: OperatorSettings) -> OperatorParts:
        business = load_business_config(settings.business_config)
        config = OperatorConfig.from_mapping(business.operator)
        judge = build_cover_judge(settings.cover_judge, business)
        touch, touch_unavailable = resolve_touch_endpoint(settings.description)
        touch_type = None
        if touch is not None:
            touch_type, touch_unavailable = vendor.load_message(touch.message_type)
        led_type, led_unavailable = vendor.load_service(config.led.service_type)
        return cls(config, judge, touch, touch_type, touch_unavailable, led_type, led_unavailable)


class RosOperatorCalls:
    """``OperatorCalls`` over ROS clients.

    Nothing here waits: each call is sent and its answer comes back through
    ``done``. A service or action server that is not there answers at once
    with ``None`` (the call failed); a Gateway refusal answers ``False`` with
    the Gateway's error code.
    """

    def __init__(self, *, status, validate, skill, session, touch, led=None, led_type=None):
        self._status = status
        self._validate = validate
        self._skill = skill
        self._session = session
        self._touch = touch
        self._led = led
        self._led_type = led_type

    def mirror_start(self, done: Done) -> None:
        request = MirrorStartRequest()

        def on_status(response) -> None:
            request.accept_status(response)
            validate = request.fill_validate_request(ValidateSkill.Request())
            self._call(self._validate, validate, on_validation, done, "validate_skill")

        def on_validation(response) -> None:
            request.accept_validation(response)
            if not self._skill.server_is_ready():
                done(None, "Gateway skill action is not available")
                return
            goal = request.fill_goal(SkillCommand.Goal())
            future = self._skill.send_goal_async(goal, goal_uuid=UUID(uuid=request.goal_uuid))
            self._then(future, on_goal, done, "mirror_start goal")

        def on_goal(handle) -> None:
            if not handle.accepted:
                raise GatewayError("SKILL_GOAL_REJECTED", "the Gateway did not accept the mirror_start goal")
            self._then(handle.get_result_async(), on_result, done, "mirror_start result")

        def on_result(answer) -> None:
            if answer.result is None:
                done(None, "mirror_start result: no answer")
                return
            done(True, request.accept_result(answer.result))

        status = request.fill_status_request(GetSkillGatewayStatus.Request())
        self._call(self._status, status, on_status, done, "Gateway status")

    def stop_session(self, done: Done) -> None:
        self._call(
            self._session,
            SetBool.Request(data=False),
            lambda response: done(bool(response.success), str(response.message)),
            done,
            "set_session(false)",
        )

    def operator_touch(self, done: Done) -> None:
        self._call(
            self._touch,
            Trigger.Request(),
            lambda response: done(bool(response.success), str(response.message)),
            done,
            "operator_touch",
        )

    def set_led(self, color: LedColor, priority: int, done: LedDone) -> None:
        request = vendor.led_request(self._led_type, color.mode, color.rgb, priority)
        self._call(self._led, request, lambda response: done(vendor.led_status_code(response), ""), done, "light")

    def _call(self, client, request, step: Callable[[Any], None], done: Callable, what: str) -> None:
        if not client.service_is_ready():
            done(None, f"{what} service is not available")
            return
        self._then(client.call_async(request), step, done, what)

    @staticmethod
    def _then(future, step: Callable[[Any], None], done: Callable, what: str) -> None:
        def answered(future) -> None:
            try:
                answer = future.result()
            except Exception as exc:  # noqa: BLE001 - the call failed: no answer
                done(None, f"{what} failed: {exc}")
                return
            if answer is None:
                done(None, f"{what}: no answer")
                return
            try:
                step(answer)
            except GatewayError as exc:
                done(False, str(exc))
            except Exception as exc:  # noqa: BLE001 - an answer that cannot be read is no answer
                done(None, f"{what}: {exc}")

        future.add_done_callback(answered)


class MirrorOperatorNode(Node):
    def __init__(self):
        super().__init__(OPERATOR_NODE_NAME)
        for name, default in OPERATOR_PARAMETERS.items():
            self.declare_parameter(name, default)
        settings = OperatorSettings.from_parameters(
            {name: self.get_parameter(name).value for name in OPERATOR_PARAMETERS}
        )
        parts = OperatorParts.build(settings)
        self._settings = settings

        from robot_config.contract_utils import qos_profile_from_dict

        calls_group = ReentrantCallbackGroup()
        led = None
        if parts.led_type is not None:
            led = self.create_client(parts.led_type, parts.config.led.service, callback_group=calls_group)
        calls = RosOperatorCalls(
            status=self.create_client(
                GetSkillGatewayStatus, settings.gateway_status_service, callback_group=calls_group
            ),
            validate=self.create_client(ValidateSkill, settings.gateway_validate_service, callback_group=calls_group),
            skill=ActionClient(self, SkillCommand, settings.gateway_skill_action, callback_group=calls_group),
            session=self.create_client(SetBool, SESSION_SERVICE, callback_group=calls_group),
            touch=self.create_client(Trigger, OPERATOR_TOUCH_SERVICE, callback_group=calls_group),
            led=led,
            led_type=parts.led_type,
        )
        self.operator = Operator(
            parts.config,
            parts.judge,
            calls,
            send_enabled=settings.send_enabled,
            touch_unavailable=parts.touch_unavailable,
            led_unavailable=parts.led_unavailable,
            log=self._log,
        )
        self._throttle = FrameThrottle(parts.config.frame_rate_max_hz)
        camera_type, self._decode = CAMERA_KINDS[settings.camera_kind]
        self.create_subscription(
            camera_type,
            settings.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        inputs_group = MutuallyExclusiveCallbackGroup()
        if parts.touch_type is not None:
            self.create_subscription(
                parts.touch_type,
                parts.touch.endpoint,
                self._on_touch,
                qos_profile_from_dict(parts.touch.qos) or 10,
                callback_group=inputs_group,
            )
        self.create_subscription(
            String,
            STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=inputs_group,
        )
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self.create_timer(TICK_PERIOD_S, self.operator.tick, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(
            settings.status_period_sec, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup()
        )
        touch = (
            f"subscribed to {parts.touch.endpoint}"
            if parts.touch_type is not None
            else f"off ({parts.touch_unavailable})"
        )
        light = parts.config.led.service if parts.led_type is not None else f"off ({parts.led_unavailable})"
        self.get_logger().info(
            f"mirror operator: camera {settings.camera_topic} ({settings.camera_kind}); "
            f"cover_judge {settings.cover_judge!r} ({type(parts.judge).__name__}), "
            f"config {settings.business_config or '(none)'}; touch.head {touch}; light {light}"
        )
        if settings.send_enabled:
            self.get_logger().warning("send_enabled=true: gestures and touches WILL call the Gateway and the session")
        else:
            self.get_logger().info("send_enabled=false: calls are recorded in ~/status, nothing is sent")

    def _log(self, level: str, message: str) -> None:
        # rclpy pins the severity of each call site; keep one call per level.
        if level == "warning":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _on_camera(self, message) -> None:
        if not self._throttle.admit(time.monotonic()):
            return
        try:
            frame = self._decode(message)
        except ValueError as exc:
            self.operator.frame_failed(str(exc))
            return
        self.operator.on_frame(frame)

    def _on_touch(self, message) -> None:
        self.operator.on_touch(vendor.touch_event_type(message))

    def _on_state(self, message: String) -> None:
        self.operator.on_state(message.data)

    def _publish_status(self) -> None:
        status = self.operator.status()
        status["cover_judge"] = self._settings.cover_judge
        status["camera_topic"] = self._settings.camera_topic
        self._status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = None
    try:
        node = MirrorOperatorNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
