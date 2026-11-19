"""Typed, bounded hand-gesture plugin for the existing ModelServiceNode.

The generic host owns ROS and diagnostics. ModelRuntimeHandle owns native
lifecycle/admission. This plugin owns only the gesture protocol and semantics.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass

from embodied_common.hand_gesture_contract import SERVICE_TYPE, RequestMeta, ResultMeta
from inference_service.backends import RuntimeContext
from inference_service.model_service_plugin import ModelServiceError, ModelServicePlugin, PluginRuntimeStatus
from inference_service.runtime_composition import require_runtime_dependencies
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelRuntimeHandle,
    RuntimeAssembly,
)

from .hand_gesture_adapter import HandGestureAdapter, validate_control_image
from .hand_gesture_session import TensorRtHandGestureSession


@dataclass
class _Source:
    generation: int
    retired_through: int = 0
    last_infer: int = 0


class HandGestureClassifyPlugin(ModelServicePlugin):
    service_type = SERVICE_TYPE
    request_queue_depth = 1
    _session_factory = staticmethod(TensorRtHandGestureSession)
    _clock = staticmethod(time.monotonic)

    def __init__(self, host, validated, options, *, registry_set=None, providers=None):
        del host
        require_runtime_dependencies(registry_set, providers, owner=type(self).__name__)
        unknown = sorted(set(options) - {"confidence", "max_infer_rate_hz"})
        if unknown:
            raise ValueError(f"unknown gesture options: {unknown}; deployment selects the backend/device")
        self._confidence = self._option(options, "confidence", 0.25, 0.0, 1.0)
        rate = self._option(options, "max_infer_rate_hz", 10.0, 0.0, 10.0)
        self._period = 1.0 / rate
        self._model_lock, self._state_lock = threading.Lock(), threading.Lock()
        self._sources = {}
        self._last_started = None
        self._closed = False
        self._execution_uncertain = False
        self.server_boot_id = str(uuid.uuid4())
        self.adapter = HandGestureAdapter()
        self.adapter.validate_deployment(validated.deployment_name)
        self.adapter.validate_identity(validated.manifest.model.semantic_identity)
        model = validated.manifest.model
        if (model.interface, model.model_type, model.operation) != ("tensor_model", "hand_gesture", "classify"):
            raise ValueError("hand gesture plugin requires tensor_model/hand_gesture/classify")
        self.validated = validated
        self.session = self._session_factory()
        context = RuntimeContext(validated, runtime_profile=validated.runtime_profile)
        # Same explicit native assembly pattern as EchoServicePlugin. This is
        # one narrow engine resource, not a new general TensorRT registry/host.
        self._runtime_handle = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=self.session,
                session=self.session,
                execution_contract=ExecutionContract(),
                runtime_id="x2-hand-gesture",
                load_context=context,
                providers=providers,
            )
        )
        self._runtime_handle.load(context)
        self.engine_sha256 = self.session.engine_sha256

    @staticmethod
    def _option(options, name, default, lower, upper):
        value = options.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
            raise ValueError(f"{name} must be finite and in ({lower}, {upper}]")
        return float(value)

    def _idle(self):
        return not self._model_lock.locked() and self.session.quiescent

    def _result(self, meta, status, *, gesture="none", score=0.0, idle=None):
        with self._state_lock:
            state = self._sources.get(meta.source_key[:2])
            retired = 0
            if state is not None:
                retired = state.retired_through if state.generation == meta.stream_generation else meta.retire_through
        return ResultMeta(
            request=meta,
            server_boot_id=self.server_boot_id,
            engine_sha256=self.engine_sha256,
            status=status,
            gesture=gesture,
            score=score,
            idle=self._idle() if idle is None else idle,
            retired_through=retired,
        ).to_json()

    def _fail(self, meta, status, message, *, idle=None):
        raise ModelServiceError(message, result_json=self._result(meta, status, idle=idle))

    def _source(self, meta):
        with self._state_lock:
            key = meta.source_key[:2]
            if key not in self._sources:
                if len(self._sources) >= 64:
                    raise ValueError("gesture source ledger is full; refusing to evict retirement evidence")
                self._sources[key] = _Source(meta.stream_generation)
            state = self._sources[key]
            if meta.stream_generation > state.generation:
                # The fixed client monotonically increments generation within
                # one producer lifetime. Fence all older generations compactly,
                # rather than filling the ledger on ordinary mirror sessions.
                state.generation = meta.stream_generation
                state.retired_through = state.last_infer = 0
            return state

    def accepts_unready_request(self, request):
        try:
            return RequestMeta.from_json(request.metadata_json).op in ("probe", "retire")
        except (ValueError, TypeError, AttributeError):
            return False

    def handle(self, request, response):
        started = self._clock()  # local processing budget; never a client monotonic clock
        try:
            meta = RequestMeta.from_json(request.metadata_json)
        except (ValueError, TypeError) as exc:
            raise ModelServiceError(f"invalid gesture metadata: {exc}") from exc
        try:
            state = self._source(meta)
            if meta.op != "infer":
                validate_control_image(request.image)
        except (ValueError, TypeError) as exc:
            self._fail(meta, "rejected", str(exc))
        if meta.op == "retire":
            # Publish the tombstone before checking execution, so a queued old
            # infer can never become a new execution after the acknowledgement.
            with self._state_lock:
                if meta.stream_generation == state.generation:
                    state.retired_through = max(state.retired_through, meta.retire_through)
            if not self._model_lock.acquire(blocking=False):
                self._fail(meta, "busy", "gesture execution still owns the model", idle=False)
            try:
                if not self.session.quiescent:
                    self._fail(meta, "busy", "CUDA completion is unconfirmed", idle=False)
                response.result_json = self._result(meta, "retired", idle=True)
                return "gesture requests retired and model idle"
            finally:
                self._model_lock.release()
        if meta.op == "probe":
            status = self.runtime_status()
            if not status.ready:
                self._fail(meta, "not_ready", status.failure_reason or status.state)
            response.result_json = self._result(meta, "ok")
            return "gesture model ready"
        if not self._model_lock.acquire(blocking=False):
            self._fail(meta, "busy", "only one gesture inference may execute", idle=False)
        try:
            if not self.runtime_status().ready:
                self._fail(meta, "not_ready", "gesture model is not ready")
            with self._state_lock:
                rejected = meta.stream_generation != state.generation or meta.request_id <= max(
                    state.retired_through, state.last_infer
                )
                if not rejected:
                    state.last_infer = meta.request_id
            if rejected:
                self._fail(meta, "rejected", "retired, duplicate or out-of-order gesture inference")
            now = self._clock()
            if self._last_started is not None and now - self._last_started < self._period:
                self._fail(meta, "busy", "gesture inference rate limit", idle=True)
            try:
                image = request.image
                stamp = image.header.stamp
                if (
                    not 0 <= stamp.nanosec < 1_000_000_000
                    or stamp.sec < 0
                    or stamp.sec * 1_000_000_000 + stamp.nanosec != meta.stamp_ns
                ):
                    raise ValueError("image stamp and immutable request stamp differ")
                inputs = self.adapter.preprocess(image)
            except (ValueError, TypeError) as exc:
                self._fail(meta, "rejected", str(exc), idle=True)
            if (self._clock() - started) * 1000 >= meta.budget_ms:
                self._fail(meta, "rejected", "gesture processing budget expired before inference", idle=True)
            self._last_started = self._clock()
            try:
                result = self._runtime_handle.execute(
                    ModelRequest(inputs, metadata={"service_type": self.service_type}),
                    ExecutionContext(f"{meta.requester_boot_id}:{meta.stream_generation}:{meta.request_id}"),
                )
                gesture, score = self.adapter.postprocess(result, confidence=self._confidence)
            except Exception as exc:
                self._execution_uncertain = not self.session.quiescent
                self._fail(
                    meta, "error", f"gesture inference failed: {type(exc).__name__}: {exc}", idle=self.session.quiescent
                )
            if (self._clock() - started) * 1000 >= meta.budget_ms:
                self._fail(meta, "rejected", "gesture processing budget expired after inference", idle=True)
            with self._state_lock:
                retired = meta.stream_generation != state.generation or meta.request_id <= state.retired_through
            if retired:
                self._fail(meta, "rejected", "gesture source retired during inference", idle=True)
            response.result_json = self._result(meta, "ok", gesture=gesture, score=score, idle=True)
            return "gesture crop classified"
        finally:
            self._model_lock.release()

    def runtime_status(self):
        diagnostics = self._runtime_handle.diagnostics()
        version = self.session.runtime_version
        pressure = self.session.resource_headroom_reason()
        uncertain = self._execution_uncertain and not self.session.quiescent
        ready = diagnostics.health.ready and bool(version) and not self._closed and not pressure and not uncertain
        return PluginRuntimeStatus(
            state="failed"
            if uncertain
            else ("degraded" if pressure and diagnostics.health.ready else diagnostics.state.value),
            ready=ready,
            failure_reason=""
            if ready
            else (
                "CUDA completion is unconfirmed"
                if uncertain
                else (pressure or diagnostics.health.message or "gesture runtime is not ready")
            ),
            metadata={"runtime_version": version} if version else {},
        )

    def close(self):
        with self._model_lock:
            if not self._closed:
                self._closed = True
                self._runtime_handle.close()
