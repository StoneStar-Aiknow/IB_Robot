"""The X2 grasp head: capture, scene, cuRobo and the target watch, served to the grasp pipeline.

Runs on the Orin beside the head camera and its GPU, started by this
package's x2_grasp_head.launch.py (aimdk_grasp.split) and bound to the runtime
the robot-side host runs. The grasp pipeline there (``aimdk_grasp.remote_head``)
drives it request by request:

1. ``capture``: the head takes and registers one RGB-D set, keeps it under a
   capture id, returns it compressed and starts the mask-independent scene
   stage at once, while the pipeline detects and segments.
2. ``complete_scene``: the target mask finishes the scene; NO_SUPPORT when
   nothing supports the target. The planner preloads it while the pipeline
   generates grasp candidates.
3. ``plan`` (action): cuRobo plans one of the candidates against that scene;
   it may capture again here for its target recheck.
4. ``refine`` during execution, and ``watch`` (arm / phase / disarm) around it:
   the watch compares images here and, while a session is armed, publishes its
   state (``WatchStatus``) every STATUS_PERIOD_S and at once when the target moved.
   The pipeline host judges from those whether the watch still looks.

The chest IMU idles as in the single-host backends: it pauses after each
capture a request plans from, resumes with a plan and pauses again when the
watch disarms or planning fails.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

import numpy as np
from rclpy.node import Node

from manipulation_execution.pick.pipeline.contracts import GraspFailure, GraspPlan, Observation, Proposals

from . import head_wire as wire
from .backend import WARMUP_BUDGET_S, head_parts, x2_robot
from .grasp_request import check_initial_hand
from .head_camera import gpu_section
from .head_interface import (
    NO_SUPPORT,
    NOT_READY,
    STATUS_DEPTH,
    STATUS_PERIOD_S,
    UNKNOWN,
    head_endpoints,
    parse_session,
)
from .proposer import measured_spheres
from .rgbd_to_cloud import mask_depth_to_points
from .scene_estimation import NoSupportError, complete_scene, prepare_scene
from .split import head_options

#: Captures and plans kept for the requests that follow: one request uses a
#: few captures (its own, a recheck, the watch's), and a plan only while it runs.
KEEP = 8
#: Pipeline processes whose newest watch session is remembered (a restart is a new one).
WATCH_CLIENTS = 16
PREPARE_RETRY_S = 5.0


@dataclass
class _Capture:
    observation: Observation
    prepared: object
    mask: np.ndarray | None = None
    points: np.ndarray | None = None
    estimate: object = None
    report: dict = field(default_factory=dict)


def _remember(store: OrderedDict, key: str, value) -> None:
    store[key] = value
    while len(store) > KEEP:
        store.popitem(last=False)


class X2GraspHeadNode(Node):
    def __init__(self, **node_options) -> None:
        from aimdk_grasp_msgs.action import PlanGraspCandidates
        from aimdk_grasp_msgs.msg import WatchStatus
        from aimdk_grasp_msgs.srv import CaptureGraspObservation, CompleteGraspScene, RefineGraspPlan, WatchGraspTarget
        from rclpy.action import ActionServer, CancelResponse
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        super().__init__("x2_grasp_head", **node_options)
        self.declare_parameter("grasp_execution_json", "{}")
        self.declare_parameter("interface_description_json", "{}")
        config = json.loads(self.get_parameter("grasp_execution_json").value)
        description = json.loads(self.get_parameter("interface_description_json").value)
        self._robot = x2_robot(config)
        names = head_endpoints(head_options(self._robot.options).namespace)
        self._status_type = WatchStatus
        # Volatile: a status is about now; a late joiner must not read an old session's.
        self._statuses = self.create_publisher(
            WatchStatus,
            names.status,
            QoSProfile(
                depth=STATUS_DEPTH, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
            ),
        )
        self._session, self._sequence = "", 0
        # Watch commands use their own mutually-exclusive callback group. The lock
        # also protects direct calls in tests and keeps ARM, PHASE and DISARM from
        # interleaving. Publishing takes only the status lock: the watcher's move
        # callback must not wait for a DISARM that is joining its thread.
        self._command_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._newest_sessions: OrderedDict[str, int] = OrderedDict()
        self._closers: list = []
        self._head = head_parts(
            self, self._robot, description, self._closers, on_moved=lambda _reason: self._publish_status()
        )
        self._lock = threading.Lock()
        self._captures: OrderedDict[str, _Capture] = OrderedDict()
        self._plans: OrderedDict[str, tuple[str, GraspPlan]] = OrderedDict()
        self._ids = itertools.count(1)
        self._scene_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grasp_head_scene")
        self._ready, self._closing = threading.Event(), threading.Event()
        group = ReentrantCallbackGroup()
        for kind, name, handler in (
            (CaptureGraspObservation, names.capture, self._on_capture),
            (CompleteGraspScene, names.scene, self._on_scene),
            (RefineGraspPlan, names.refine, self._on_refine),
        ):
            self.create_service(kind, name, self._answering(handler), callback_group=group)
        self.create_service(
            WatchGraspTarget,
            names.watch,
            self._answering(self._on_watch),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._plan_server = ActionServer(
            self,
            PlanGraspCandidates,
            names.plan,
            execute_callback=self._on_plan,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        # Its own group: a long plan or capture must not delay the status the other host times.
        self.create_timer(STATUS_PERIOD_S, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup())
        threading.Thread(target=self._prepare, daemon=True, name="grasp_head_prepare").start()
        self.get_logger().info(f"grasp head serving under {names.capture.rsplit('/', 1)[0]}")

    # -- warmup -------------------------------------------------------------------

    def _prepare(self) -> None:
        """Warm cuRobo on a fresh image with the hand open; captures are refused until it succeeds."""
        head = self._head
        while not self._closing.is_set():
            try:
                observation = head.camera.capture(
                    deadline=time.monotonic() + WARMUP_BUDGET_S, cancelled=self._closing.is_set
                )
                admitted = observation.robot_state
                check_initial_hand(self._robot.hand, admitted)
                prepared = self._prepare_scene(observation)
                with head.gpu:
                    estimate = complete_scene(prepared, None, lazy_validation=False)
                head.service.prepare(admitted.live.feedback, estimate)
            except Exception as exc:  # noqa: BLE001 - retried until the robot and the camera are up
                self.get_logger().warning(f"grasp head not ready: {exc}", throttle_duration_sec=5.0)
                self._closing.wait(PREPARE_RETRY_S)
                continue
            finally:
                if head.imu is not None:
                    head.imu.pause()  # idle until a request has a plan to execute
            self._ready.set()
            self.get_logger().info("grasp head prepared; accepting captures")
            return

    def _prepare_scene(self, observation: Observation):
        """The mask-independent scene stage of one capture."""
        registered = observation.robot_state.live.observation
        frame, transform = registered.frame, registered.transform
        with self._head.gpu:
            return prepare_scene(
                frame.depth,
                frame.intrinsics,
                transform,
                stamp_ns=transform.stamp_ns,
                camera_frame=frame.frame_id,
                received_at=registered.received_at,
                robot_spheres_torso=measured_spheres(self._head.spheres, registered.image_feedback),
                depth_scale=frame.depth_scale,
                config=self._robot.perception.scene,
            )

    # -- requests -----------------------------------------------------------------

    @staticmethod
    def _fail(response, code: str, message: str):
        response.success, response.message = False, message
        if hasattr(response, "error_code"):
            response.error_code = code
        return response

    def _answering(self, handler):
        """``handler`` that answers INTERNAL_ERROR on any error instead of raising.

        An exception escaping a callback is raised out of the executor's spin,
        which ends this process: the pipeline would then wait out its deadline
        for a reply that never comes, and every later goal fail.
        """

        def answered(request, response):
            try:
                return handler(request, response)
            except Exception as exc:  # noqa: BLE001 - see above
                self.get_logger().error(f"{handler.__name__} failed: {type(exc).__name__}: {exc}")
                return self._fail(response, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")

        return answered

    def _capture(self, capture_id: str) -> _Capture | None:
        with self._lock:
            return self._captures.get(capture_id)

    def _on_capture(self, request, response):
        if not self._ready.is_set():
            return self._fail(response, NOT_READY, "the grasp head is warming up")
        head = self._head
        try:
            observation = head.camera.capture(
                deadline=time.monotonic() + max(0.0, request.timeout_s), cancelled=self._closing.is_set
            )
        except GraspFailure as failure:
            return self._fail(response, failure.code, str(failure))
        if head.imu is not None:
            head.imu.pause_after(observation.stamp_ns)
        capture_id = f"{observation.stamp_ns}-{next(self._ids)}"
        prepared = self._scene_pool.submit(self._prepare_scene, observation)
        with self._lock:
            _remember(self._captures, capture_id, _Capture(observation, prepared))
        rgb, depth = wire.encode_rgb(observation.rgb), wire.encode_depth(observation.depth_m)
        rgb.header.frame_id = depth.header.frame_id = observation.frame_id
        response.success, response.capture_id, response.rgb, response.depth = True, capture_id, rgb, depth
        response.intrinsics = wire.matrix_to_list(observation.intrinsics, (3, 3))
        response.stamp_ns, response.base_frame = int(observation.stamp_ns), observation.base_frame
        response.base_from_camera = wire.matrix_to_list(observation.base_from_camera, (4, 4))
        response.report_json = wire.to_json(dict(observation.report))
        return response

    def _on_scene(self, request, response):
        capture = self._capture(request.capture_id)
        if capture is None:
            return self._fail(response, UNKNOWN, f"no capture {request.capture_id!r} is kept")
        try:
            mask = wire.decode_mask(request.target_mask)
            registered = capture.observation.robot_state.live.observation
            frame = registered.frame
            if mask.shape != frame.depth.shape:
                raise wire.WireError(f"mask {mask.shape} is not on the capture's {frame.depth.shape} grid")
            deadline = time.monotonic() + max(0.0, request.timeout_s)
            waited = time.perf_counter()
            prepared = capture.prepared.result(timeout=max(0.0, request.timeout_s))
            started = time.perf_counter()
            with gpu_section(self._head.gpu, deadline=deadline, cancelled=self._closing.is_set):
                estimate = complete_scene(prepared, mask, lazy_validation=True)
            settings = self._robot.perception
            points = mask_depth_to_points(
                mask,
                frame.depth,
                frame.intrinsics,
                depth_scale=frame.depth_scale,
                min_depth_m=settings.min_depth_m,
                max_depth_m=settings.max_depth_m,
            )
        except NoSupportError as exc:
            return self._fail(response, NO_SUPPORT, str(exc))
        except GraspFailure as failure:  # the GPU stayed busy past the request's deadline
            return self._fail(response, failure.code, str(failure))
        except FutureTimeout:
            return self._fail(response, "SKILL_TIMEOUT", "the scene's first stage did not finish in time")
        except wire.WireError as exc:
            return self._fail(response, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
        except (ValueError, RuntimeError) as exc:
            # As the single-host proposer reports a scene it cannot build (e.g. a
            # mask with too little valid depth): this candidate set has no plan.
            return self._fail(response, "NO_EXECUTABLE_CANDIDATE", f"{type(exc).__name__}: {exc}")
        report = {"prepare_wait_s": started - waited, "scene_s": time.perf_counter() - started}
        try:
            began = time.perf_counter()
            self._head.service.preload(capture.observation.robot_state.live.feedback, estimate)
            report["scene_load_s"] = time.perf_counter() - began
        except Exception as exc:  # noqa: BLE001 - plan repeats the load and reports its own error
            report["scene_preload_error"] = f"{type(exc).__name__}: {exc}"
        with self._lock:
            capture.mask, capture.points, capture.estimate, capture.report = mask, points, estimate, report
        response.success, response.report_json = True, wire.to_json(report)
        return response

    def _on_plan(self, goal_handle):
        from aimdk_grasp_msgs.action import PlanGraspCandidates

        request, result = goal_handle.request, PlanGraspCandidates.Result()
        capture = self._capture(request.capture_id)
        head = self._head
        try:
            if capture is None or capture.estimate is None:
                raise GraspFailure(UNKNOWN, f"no completed scene for capture {request.capture_id!r}")
            proposals = Proposals(
                capture.mask,
                capture.points,
                wire.poses_from_list(request.grasps),
                np.asarray(request.confidence, dtype=np.float64),
                capture.report,
                scene=capture.estimate,
            )
            plan = head.planner.plan(
                capture.observation,
                proposals,
                deadline=time.monotonic() + max(0.0, request.timeout_s),
                cancelled=lambda: goal_handle.is_cancel_requested or self._closing.is_set(),
            )
        except GraspFailure as failure:
            if head.imu is not None:
                head.imu.pause()
            result.success, result.error_code, result.message = False, failure.code, str(failure)
            result.retryable = bool(failure.retryable)
        except Exception as exc:  # noqa: BLE001 - answer the goal; an escaping error ends the head process
            if head.imu is not None:
                head.imu.pause()
            result.success, result.error_code = False, "INTERNAL_ERROR"
            result.message = f"{type(exc).__name__}: {exc}"
        else:
            if head.imu is not None:
                head.imu.resume()
            plan_id = f"{request.capture_id}/{next(self._ids)}"
            with self._lock:
                _remember(self._plans, plan_id, (request.capture_id, plan))
            result.success, result.plan_id, result.plan = True, plan_id, wire.plan_to_msg(plan)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return result

    def _on_refine(self, request, response):
        with self._lock:
            kept = self._plans.get(request.plan_id)
        if kept is None:
            return self._fail(response, UNKNOWN, f"no plan {request.plan_id!r} is kept")
        capture_id, plan = kept
        try:
            refined = self._head.planner.refine(
                plan,
                int(request.completed),
                deadline=time.monotonic() + max(0.0, request.timeout_s),
                cancelled=self._closing.is_set,
            )
        except GraspFailure as failure:
            return self._fail(response, failure.code, str(failure))
        with self._lock:
            self._plans[request.plan_id] = (capture_id, refined)
        response.success, response.plan = True, wire.plan_to_msg(refined)
        return response

    def _on_watch(self, request, response):
        watch = self._head.watch
        response.success = True
        if watch is None:
            response.message = "the watch is off"
            return response
        try:
            client, number = parse_session(request.session_id)
        except ValueError as exc:
            response.success, response.message = False, str(exc)
            return response
        with self._command_lock:
            newest = self._newest_sessions.get(client, 0)
            # Seen now: an ARM of this session or an older one that arrives later is stale.
            self._newest_sessions[client] = max(newest, number)
            self._newest_sessions.move_to_end(client)
            while len(self._newest_sessions) > WATCH_CLIENTS:
                self._newest_sessions.popitem(last=False)
            if request.command == request.ARM:
                if number <= newest:
                    response.success = False
                    response.message = f"session {request.session_id!r} is not newer than {client}:{newest}"
                    return response
                return self._arm_watch(watch, request, response)
            if request.command not in (request.PHASE, request.DISARM):
                response.success, response.message = False, f"unknown watch command {request.command}"
                return response
            if request.session_id != self._session:
                response.success = False
                response.message = f"session {request.session_id!r} is not the armed one ({self._session or 'none'})"
                return response
            if request.command == request.PHASE:
                watch.phase(request.phase_name, request.phase_event)
                self._publish_status()
            else:
                with self._status_lock:
                    self._session = ""
                response.report_json = wire.to_json(watch.disarm())
            return response

    def _arm_watch(self, watch, request, response):
        with self._status_lock:
            self._session = ""  # nothing is published for the session before
        capture = self._capture(request.capture_id)
        if capture is None or capture.mask is None:
            response.success, response.message = False, f"no completed scene for capture {request.capture_id!r}"
            return response
        proposals = Proposals(capture.mask, capture.points, np.zeros((0, 4, 4)), np.zeros(0), scene=capture.estimate)
        try:
            watch.arm(capture.observation, proposals, None)
        except GraspFailure as failure:  # stop: the pipeline must not execute
            response.success, response.message = False, str(failure)
            return response
        status = watch.status()
        # log keeps a watch that did not start: its status says so, the execution goes on.
        response.success, response.message = watch.started, status.reason
        with self._status_lock:
            self._session, self._sequence = request.session_id, 0
        self._publish_status()
        return response

    def _publish_status(self) -> None:
        """The armed session's watch state; nothing while no session is armed."""
        watch = self._head.watch
        if watch is None:
            return
        with self._status_lock:
            if not self._session:
                return
            status = watch.status()
            self._sequence += 1
            self._statuses.publish(
                self._status_type(
                    session_id=self._session,
                    sequence=self._sequence,
                    state=int(status.state),
                    phase=status.phase,
                    fresh_frames=int(status.fresh_frames),
                    last_image_stamp_ns=int(status.last_image_stamp_ns),
                    unknown_s=float(status.unknown_s),
                    reason=status.reason,
                )
            )

    # -- shutdown -------------------------------------------------------------------

    def close(self) -> None:
        self._closing.set()
        self._scene_pool.shutdown(wait=False, cancel_futures=True)
        for close in reversed(self._closers):
            close()


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = X2GraspHeadNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
