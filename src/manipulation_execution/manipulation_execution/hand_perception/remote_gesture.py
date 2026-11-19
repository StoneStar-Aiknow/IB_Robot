"""Optional asynchronous model-service client. The hand worker only posts references.

One image call remains occupied through timeout and retirement. Control calls
carry no image, are idempotent, and never produce a gesture vote. Neither this
module nor its transport owns a robot, a local model or an image recording.
"""

from __future__ import annotations

import collections
import math
import threading
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np

from embodied_common.hand_gesture_contract import (
    GESTURE_SCHEMA,
    RequestMeta,
    ResultMeta,
    SourceState,
    digest,
    dump_json,
    number,
    session_token,
)
from manipulation_execution.preset_motions.session_gate import SessionGate

from . import gesture_crop as gc


@dataclass(frozen=True)
class RemoteConfig:
    timeout_s: float = 0.4
    health_s: float = 2.5
    probe_s: float = 1.0
    max_rate_hz: float = 10.0
    source_age_s: float = 0.8
    ready_samples: int = 2
    fail_limit: int = 3
    conf: float = 0.25
    expected_engine_sha256: str = ""

    def __post_init__(self):
        for name in ("timeout_s", "health_s", "probe_s", "max_rate_hz", "source_age_s"):
            number(getattr(self, name), name, minimum=1e-6)
        for name in ("ready_samples", "fail_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        number(self.conf, "conf", maximum=1.0)
        digest(self.expected_engine_sha256, "expected_engine_sha256", empty=True)
        if self.timeout_s > 0.8 or self.source_age_s > 0.8:
            raise ValueError("remote mode cannot extend the existing source-age budget")


@dataclass(frozen=True)
class Snapshot:
    stamp_ns: int
    actor_id: int
    image: np.ndarray
    box: np.ndarray
    landmarks: tuple
    generation: int
    local_revision: int
    token: str
    pear_stamp_ns: int
    frame_id: str


@dataclass
class Call:
    meta: RequestMeta
    started: float
    future: object = None
    expired: bool = False
    crop_ms: float = 0.0
    local_revision: int = 0
    response_checked: bool = False


class RemoteGestureRunner:
    off = False  # A transient outage must never delete the runner from the hand node.
    is_remote = True

    def __init__(
        self,
        crop,
        config,
        *,
        transport,
        publish,
        publish_source,
        log=lambda _text: None,
        warn=None,
        clock=time.monotonic,
        ros_clock=time.time,
        boot_id=None,
    ):
        self.cfg, self.crop = config, crop
        self.transport, self.publish, self.publish_source = transport, publish, publish_source
        self.log, self.clock, self.ros_clock = log, clock, ros_clock
        self.warn, self._last_warning = warn or log, ""
        self.boot_id = boot_id or str(uuid.uuid4())
        self.started_ns = int(clock() * 1e9)
        self.generation, self._request_seq, self._result_seq, self._state_seq = 1, 0, 0, 0
        self._lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None
        self._job = self._call = self._control = None
        self._session = SessionGate()
        self._session_seq = None
        self._session_signature = None
        self._context_revision = self._applied_revision = 0
        self._token, self._context = "", None
        self._last_frame_actor = -1  # Diagnostic metadata from the already selected PEAR frame, not another selector.
        self._healthy, self._ready_count, self._failures = False, 0, 0
        self._last_health, self._next_probe = -math.inf, 0.0
        self._server_boot, self._engine = "", config.expected_engine_sha256
        self._reason = "awaiting_remote"
        self._last_source, self._last_source_at = None, -math.inf
        self._last_log = clock()
        self._last_post_stamp, self._clock_jump_pending = None, False
        self.stats = collections.Counter()
        self.post_ms_max = 0.0
        self.rate = gc.RateGate(replace(crop, max_rate_hz=min(crop.max_rate_hz, config.max_rate_hz)))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._main, name="hand-gesture-remote", daemon=True)
            self._thread.start()

    def stop(self, timeout=0.0):
        self._stop.set()
        self._wake.set()  # No join, network wait, future.cancel or remote shutdown on the caller.

    def observe_session(self, text, *, now_ros_s, now_mono_s):
        try:
            import json

            if not isinstance(text, str) or len(text) > 4096 or len(text.encode("utf-8")) > 4096:
                raise ValueError("session state too long")
            state = json.loads(text)
            seq = state.get("seq")
            with self._lock:
                epoch, active = self._session.observe(state, now_ros_s=now_ros_s, now_mono_s=now_mono_s)
                if type(seq) is int and (self._session_seq is None or seq != self._session_seq):
                    self._session_seq = seq
                    self._token = session_token(state) if active else ""
                if not active:
                    self._token = ""
                self._session_version(epoch, active)
        except (ValueError, TypeError, AttributeError, RecursionError):
            with self._lock:
                self._token = ""
                self._context_revision += 1
        self._wake.set()

    def _session_version(self, epoch, active):
        signature = epoch, active, self._token if active else ""
        if signature != self._session_signature:
            self._session_signature = signature
            self._context_revision += 1

    def _context_now(self, ros, mono):
        epoch, active = self._session.permit(now_ros_s=ros, now_mono_s=mono)
        self._session_version(epoch, active)
        return self._token if active and self._token else None

    def post(self, stamp_ns, actor_id, image, box, landmarks, busy=False, *, pear_stamp_ns=0, frame_id=""):
        started = time.perf_counter()
        if self._stop.is_set() or not self._lock.acquire(blocking=False):
            return False
        try:
            if not self._healthy or self._context is None or type(actor_id) is not int or actor_id < 0:
                return False
            if self._context_revision != self._applied_revision:
                return False
            if self._last_post_stamp is not None and stamp_ns < self._last_post_stamp:
                self._clock_jump_pending = True
            self._last_post_stamp = stamp_ns
            self._last_frame_actor = actor_id
            # Only small fixed-size metadata is copied here. No crop/resize/encode/RPC.
            arrays = tuple(np.asarray(lm) for lm in landmarks[:2])
            if any(lm.shape != (21, 2) for lm in arrays):
                return False
            lms = tuple(np.array(lm, dtype=np.float64, copy=True) for lm in arrays)
            job = Snapshot(
                int(stamp_ns),
                actor_id,
                image,
                np.array(box, copy=True),
                lms,
                self.generation,
                self._context_revision,
                self._context,
                int(pear_stamp_ns),
                str(frame_id),
            )
            self.stats["replaced"] += self._job is not None
            self._job = job
            self.stats["posted"] += 1
            self.post_ms_max = max(self.post_ms_max, (time.perf_counter() - started) * 1000)
        except Exception:  # The optional handoff must never escape into the hand worker.
            self.stats["post_error"] += 1
            return False
        finally:
            self._lock.release()
        self._wake.set()
        return True

    def _invalidate(self, reason):
        if self._healthy:
            self.generation += 1
        self._healthy, self._ready_count = False, 0
        self._job, self._reason = None, reason

    def _next_meta(self, op, *, old=None, job=None, plan=None):
        self._request_seq += 1
        if old is not None:
            return replace(old, request_id=self._request_seq, op="retire", budget_ms=0.0, retire_through=old.request_id)
        values = dict(
            requester_boot_id=self.boot_id,
            producer_start_mono_ns=self.started_ns,
            stream_generation=self.generation,
            request_id=self._request_seq,
            op=op,
        )
        if job is not None:
            values.update(
                session_token=job.token,
                actor_id=job.actor_id,
                stamp_ns=job.stamp_ns,
                pear_stamp_ns=job.pear_stamp_ns,
                crop=plan.mode,
                roi=plan.roi,
                source_size=(job.image.shape[1], job.image.shape[0]),
                budget_ms=min(self.cfg.timeout_s, self.cfg.source_age_s - (self.ros_clock() - job.stamp_ns / 1e9))
                * 1000,
            )
        return RequestMeta(**values)

    def _begin_control(self, now, old=None):
        call = Call(self._next_meta("probe", old=old), now)
        self._control, self._next_probe = call, now + self.cfg.probe_s
        try:
            call.future = self.transport.begin(call.meta, None, "")
        except Exception:  # A control request has no image and can be retired/retried idempotently.
            self.stats["control_send_error"] += 1
            self._control = None

    def _response(self, call, *, allow_negative=False):
        response = call.future.result()
        if response.model.ready is not True:
            raise ValueError("remote_not_ready")
        if response.success is not True and not (allow_negative and response.success is False):
            raise ValueError("remote_not_ready")
        number(response.inference_time_ms, "inference_time_ms")
        digest(response.model.deployment_fingerprint, "deployment_fingerprint")
        result = ResultMeta.from_json(response.result_json)
        if result.request != call.meta or (self._engine and result.engine_sha256 != self._engine):
            raise ValueError("remote_identity")
        return response, result

    def _forget(self, call, *, cancelled=False):
        if call is not None and call.future is not None:
            self.transport.reap(call.future, cancel=cancelled)

    def _poll_control(self, now):
        call = self._control
        if call is None:
            return
        if call.future.done():
            try:
                _, result = self._response(call)
                if result.status not in ("ok", "retired") or not result.idle:
                    raise ValueError("remote_busy")
                if call.meta.op == "retire":
                    if result.retired_through < call.meta.retire_through:
                        raise ValueError("retirement_not_confirmed")
                    self._forget(self._call, cancelled=True)
                    self._call = None  # Only the matched, confirmed retirement may free a timed-out image call.
                with self._lock:
                    if self._server_boot and result.server_boot_id != self._server_boot:
                        self._invalidate("remote_restarted")
                    self._server_boot, self._engine = result.server_boot_id, result.engine_sha256
                    self._last_health = now
                    self._ready_count += 1
                    if self._ready_count >= self.cfg.ready_samples:
                        self._healthy, self._reason = True, ""
            except Exception:
                with self._lock:
                    self._invalidate("remote_not_ready")
                self.stats["control_rejected"] += 1
            finally:
                self._forget(call)
                self._control = None
        elif now - call.started >= self.cfg.probe_s:
            self._forget(
                call, cancelled=True
            )  # Controls never classify an image; cancelling one is not cancelling inference.
            self._control = None
            self.stats["control_timeout"] += 1

    def _poll_image(self, now):
        call = self._call
        if call is None:
            return
        with self._lock:
            context = self._context_now(self.ros_clock(), now)
            current = (
                context == call.meta.session_token
                and call.meta.stream_generation == self.generation
                and call.local_revision == self._context_revision
            )
        if not call.expired and (now - call.started > self.cfg.timeout_s or not current):
            call.expired = True
            self.stats["expired"] += 1
            if current:
                self._failures += 1
                if self._failures >= self.cfg.fail_limit:
                    with self._lock:
                        self._invalidate("inference_timeout")
        if call.future is None or not call.future.done() or call.response_checked:
            return
        call.response_checked = True
        if getattr(call.future, "cancelled", lambda: False)():
            call.expired = True
            return  # Local cancellation is not proof that the model stopped; require retirement.
        confirmed_idle = False
        try:
            response, result = self._response(call, allow_negative=True)
            if result.server_boot_id != self._server_boot or not result.idle:
                raise ValueError("remote_result_identity")
            if response.success is False:
                if result.status not in ("busy", "rejected", "error", "not_ready"):
                    raise ValueError("remote_result_status")
                confirmed_idle = True
                self.stats["negative_result"] += 1
                return  # Valid completion evidence is not a gesture vote or a retirement acknowledgement.
            if result.status != "ok":
                raise ValueError("remote_result_status")
            confirmed_idle = True
            age = self.ros_clock() - call.meta.stamp_ns / 1e9
            if not call.expired and current and -0.1 <= age <= self.cfg.source_age_s:
                with self._lock:
                    self._last_health, self._failures = now, 0
                self._publish_result(call, result, response.inference_time_ms)
            else:
                self.stats["stale_result"] += 1
        except Exception:
            call.expired = True
            self.stats["bad_result"] += 1
        finally:
            # A terminal Future alone cannot confirm that remote execution stopped.
            if confirmed_idle:
                self._forget(call)
                self._call = None

    def _publish_state(self, now, force=False):
        with self._lock:
            available = (
                self._healthy and self._context is not None and self._last_frame_actor >= 0 and not self._stop.is_set()
            )
            signature = (self.generation, available, self._context, self._server_boot, self._reason)
            if not force and signature == self._last_source and now - self._last_source_at < 1.0:
                return True
            self._state_seq += 1
            state = SourceState(
                self.boot_id,
                self.started_ns,
                self.generation,
                self._state_seq,
                available,
                int(now * 1e9),
                actor_id=self._last_frame_actor,
                session_token=self._context or "",
                server_boot_id=self._server_boot,
                engine_sha256=self._engine,
                reason=self._reason if not available else "",
            )
        self.publish_source(state.to_json())
        if not available and state.reason and state.reason != self._last_warning:
            self.warn(f"remote hand gesture unavailable ({state.reason}); mirroring continues")
            self._last_warning = state.reason
        elif available:
            self._last_warning = ""
        self._last_source, self._last_source_at = signature, now
        return True

    def _publish_result(self, call, result, inference_ms):
        now = self.clock()
        self._publish_state(now)
        with self._lock:
            now = self.clock()
            age = self.ros_clock() - call.meta.stamp_ns / 1e9
            context = self._context_now(self.ros_clock(), now)
            if (
                self._stop.is_set()
                or not self._healthy
                or context != call.meta.session_token
                or self.generation != call.meta.stream_generation
                or call.local_revision != self._context_revision
                or now - call.started > self.cfg.timeout_s
                or not -0.1 <= age <= self.cfg.source_age_s
            ):
                return
            self._result_seq += 1
            payload = dict(
                schema=GESTURE_SCHEMA,
                stamp_ns=call.meta.stamp_ns,
                actor_id=call.meta.actor_id,
                cls=result.gesture if result.score >= self.cfg.conf else "none",
                score=result.score if result.score >= self.cfg.conf else 0.0,
                crop=call.meta.crop,
                roi=call.meta.roi,
                infer_ms=call.crop_ms + inference_ms,
                requester_boot_id=self.boot_id,
                producer_start_mono_ns=self.started_ns,
                stream_generation=self.generation,
                result_seq=self._result_seq,
                published_mono_ns=int(now * 1e9),
                session_token=call.meta.session_token,
            )
        self.publish(dump_json(payload))
        self.stats["results"] += 1

    def step(self):
        """One bounded scheduling step; image serialization can only block this worker."""
        if self._stop.is_set():
            return
        now = self.clock()
        with self._lock:
            context = self._context_now(self.ros_clock(), now)
            if self._clock_jump_pending:
                self.generation += 1
                self._job, self._clock_jump_pending = None, False
            if context != self._context or self._applied_revision != self._context_revision:
                if self._context is not None and self._healthy:
                    self.generation += 1
                self._context, self._job = context, None
                self._applied_revision = self._context_revision
            if self._healthy and now - self._last_health > self.cfg.health_s:
                self._invalidate("health_stale")
        self._poll_control(now)
        self._poll_image(now)
        if not self.transport.ready():
            with self._lock:
                self._invalidate("service_missing")
        elif not self._stop.is_set() and self._control is None and now >= self._next_probe:
            if self._call is not None and self._call.expired:
                self._begin_control(now, self._call.meta)
            elif self._call is None and (not self._healthy or now - self._last_health >= self.cfg.probe_s):
                self._begin_control(now)
        self._publish_state(now)
        if self._stop.is_set() or not self._healthy or self._call is not None or self._control is not None:
            return
        with self._lock:
            job, self._job = self._job, None
        if job is None or job.generation != self.generation or job.token != self._context:
            return
        age = self.ros_clock() - job.stamp_ns / 1e9
        if not -0.1 <= age < self.cfg.source_age_s:
            self.stats["stale_input"] += 1
            return
        started = time.perf_counter()
        plan = gc.plan_crop(job.landmarks, job.box, self.crop, job.image.shape[:2])
        if not plan.run or not self.rate.allow(plan.mode, job.stamp_ns):
            self.stats["crop_or_rate_skip"] += 1
            return
        import cv2

        square = gc.extract_square(job.image, plan.x0, plan.y0, plan.side)
        square = cv2.resize(square, (320, 320), interpolation=cv2.INTER_LINEAR)
        with self._lock:
            current = self._context_now(self.ros_clock(), self.clock())
            if (
                self._stop.is_set()
                or job.generation != self.generation
                or job.local_revision != self._context_revision
                or current != job.token
            ):
                return
        meta = self._next_meta("infer", job=job, plan=plan)
        call = Call(
            meta, self.clock(), crop_ms=(time.perf_counter() - started) * 1000, local_revision=job.local_revision
        )
        self._call = call  # Occupy before send: an exception or a slow send cannot expose an empty slot.
        self.rate.mark(plan.mode, job.stamp_ns)
        try:
            call.future = self.transport.begin(meta, square, job.frame_id)
            self.stats["sent"] += 1
        except Exception:
            call.expired = True
            self.stats["send_error"] += 1

    def _main(self):
        while not self._stop.is_set():
            try:
                self.step()
                if self.clock() - self._last_log >= 5.0:
                    self.log(
                        f"remote gesture {dict(self.stats)} post_ms_max={self.post_ms_max:.2f} state={self._reason or 'ready'}"
                    )
                    self._last_log = self.clock()
            except Exception:
                self.stats["worker_error"] += 1
                with self._lock:
                    self._invalidate("worker_error")
            self._wake.wait(0.02)
            self._wake.clear()
