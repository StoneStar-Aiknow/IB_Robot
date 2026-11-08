"""Own one carried object until another executor claims it or it is put back.

The claim token fences stale/duplicate service calls. Expiry after a claim never
moves the robot: only the unclaimed pick pose has a known return path. An
unclaimed object that drops is a known state too: the hand opens and goes home.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import asdict, fields, is_dataclass

from .grasp_pipeline.contracts import GraspFailure, HeldObject

TOPIC = "/manipulation/held_object"
CLAIM = "/manipulation/claim_held_object"
RETURN = "/manipulation/return_held_object"
#: A drop whose hand opened (released, nothing left to fall) but whose way home
#: failed: the only uncertain custody a recovery skill may move out of.
DROP_HOME_FAILED = "DROP_HOME_FAILED"
#: How long an open, empty hand must stay back at home before such a custody ends.
SETTLED_S = 1.0
MANUAL_RESET_WINDOW_S = 60.0
AUTHORIZE_RESET = "/manipulation/authorize_held_reset"


def held_to_msg(held: HeldObject, *, state="picked", owner="pick", reason="", root_task_id=""):
    from ibrobot_msgs.msg import HeldObject as Message

    message = Message(
        holding=state in {"picked", "claimed", "recovering"},
        state=state,
        owner=owner,
        reason_code=reason,
        root_task_id=root_task_id,
    )
    for field in fields(HeldObject):
        if field.name not in {"lifted_at_ns", "frame_id"}:
            value = getattr(held, field.name)
            setattr(message, field.name, list(value) if isinstance(value, tuple) else value)
    message.header.frame_id = held.frame_id
    message.lifted_at.sec, message.lifted_at.nanosec = divmod(held.lifted_at_ns, 1_000_000_000)
    return message


def held_from_msg(message) -> HeldObject:
    values = {
        field.name: getattr(message, field.name)
        for field in fields(HeldObject)
        if field.name not in {"lifted_at_ns", "frame_id"}
    }
    for name in (
        "joint_names",
        "joint_positions",
        "home_positions",
        "points_in_palm",
        "return_positions",
        "return_times",
    ):
        values[name] = tuple(values[name])
    return HeldObject(
        **values,
        frame_id=message.header.frame_id,
        lifted_at_ns=message.lifted_at.sec * 1_000_000_000 + message.lifted_at.nanosec,
    )


class HeldCustody:
    def __init__(self, *, lease_s=30.0, clock=time.monotonic, changed=lambda: None, diagnostic=lambda report: None):
        if not 0 < lease_s < float("inf"):
            raise ValueError("held lease must be finite and positive")
        self.lease_s, self.clock, self.changed = lease_s, clock, changed
        self.diagnostic = diagnostic
        self.recovery_report = None
        self.lock = threading.RLock()
        self.held: HeldObject | None = None
        self.state, self.owner, self.reason, self.token = "empty", "", "", ""
        self.deadline = 0.0
        self.recover = self.release = self.holding = self.drop = self.settled = None
        self._unknown_since = None
        self._settled_since = None
        self.root_task_id = ""
        self.manual_reset = None

    @property
    def occupied(self):
        with self.lock:
            return self.state in {"picked", "claimed", "recovering", "uncertain"}

    def accept(self, held, *, recover, release, holding, root_task_id="", drop=None, settled=None):
        """``drop`` opens the hand and takes it home after an unclaimed object falls; True when it did.

        ``settled`` tells whether the hand is open and empty and the arm is back
        at ``held.home_positions``. After a drop whose hand opened but whose way
        home failed (DROP_HOME_FAILED), a recovery skill may bring the arm back;
        once ``settled`` holds for SETTLED_S the custody ends as an ordinary drop.
        """
        with self.lock:
            if self.occupied:
                raise GraspFailure("BUSY", "a held object already has custody")
            self.held, self.recover, self.release, self.holding = held, recover, release, holding
            self.drop, self.settled = drop, settled
            self.root_task_id = root_task_id
            self.state, self.owner, self.reason, self.token = "picked", "pick", "", ""
            self.deadline, self._unknown_since, self._settled_since = self.clock() + self.lease_s, None, None
            self.recovery_report = None
            self.manual_reset = None
            self.changed()

    def authorize_reset(self, custody_id, request_id, *, operator_confirmed_empty, operator_note):
        """Bind one operator's empty-hand check to this custody, without moving or clearing it."""
        with self.lock:
            if not request_id.strip() or not operator_confirmed_empty or not operator_note.strip():
                raise GraspFailure(
                    "OPERATOR_CHECK_REQUIRED", "an explicit empty-hand check and audit note are required"
                )
            if self.held is None or not custody_id or custody_id != self.held.custody_id:
                raise GraspFailure("STALE_CUSTODY", "the operator check does not match the current custody")
            if self.state != "uncertain":
                raise GraspFailure("RESET_NOT_REQUIRED", "only an uncertain custody can be reset")
            if self.settled is None:
                raise GraspFailure("RESET_VERIFICATION_UNAVAILABLE", "fresh empty-hand and home feedback are required")
            now = self.clock()
            if self.manual_reset is not None and self.manual_reset["request_id"] == request_id:
                if now >= self.manual_reset["deadline"]:
                    raise GraspFailure("RESET_AUTHORIZATION_EXPIRED", "a new operator check is required")
                return self.manual_reset["deadline"] - now  # Retries never extend authorization.
            self.manual_reset = {
                "request_id": request_id,
                "custody_id": custody_id,
                "original_reason": self.reason,
                "operator_note": operator_note,
                "started_at": now,
                "deadline": now + MANUAL_RESET_WINDOW_S,
            }
            self._settled_since = None
            self.diagnostic({**self.manual_reset, "operation": "MANUAL_RESET_AUTHORIZED", "succeeded": True})
            return MANUAL_RESET_WINDOW_S

    def claim(self, custody_id, target, claimant, *, token="", lease_s=90.0):
        with self.lock:
            if not claimant or not 0 < lease_s <= 120.0:
                raise GraspFailure("INVALID_CLAIM", "claim needs an owner and lease in (0, 120] seconds")
            if self.held is None or custody_id != self.held.custody_id:
                raise GraspFailure("NOT_HOLDING", "the requested custody no longer exists")
            if target.strip().casefold() != self.held.target_query.strip().casefold():
                raise GraspFailure("HELD_OBJECT_MISMATCH", "the held target differs from the place target")
            if self.state == "claimed" and token and token == self.token and claimant == self.owner:
                if self.clock() >= self.deadline:
                    raise GraspFailure("STALE_CLAIM", "claim has expired")
                self.deadline = self.clock() + lease_s
                return self.token, self.held
            if self.state != "picked" or self.clock() >= self.deadline:
                raise GraspFailure("BUSY" if self.occupied else "NOT_HOLDING", f"custody is {self.state}")
            grip = self.holding()
            if grip is None or grip:
                raise GraspFailure(
                    "GRASP_UNCERTAIN" if grip is None else "GRIP_LOST", "grip could not be confirmed at handover"
                )
            # Stop keepalives before acknowledging exclusive ownership to place.
            # Arm/hand streams must yield before place acquires them. Keep the
            # pick effector's frozen observation so a no-motion return can recover.
            self.release()
            self.token, self.owner, self.state = uuid.uuid4().hex, claimant, "claimed"
            self.deadline = self.clock() + lease_s
            # Grip debounce measures one continuous unknown stretch while pick
            # watches; a claim interrupts it, so the return starts a new one.
            self._unknown_since = None
            self.changed()
            return self.token, self.held

    def returned(self, custody_id, token, outcome, reason=""):
        with self.lock:
            if (
                self.held is None
                or custody_id != self.held.custody_id
                or not token
                or token != self.token
                or self.state != "claimed"
            ):
                raise GraspFailure("STALE_CLAIM", "return does not own the current custody")
            if outcome not in {"held", "released", "lost", "uncertain"}:
                raise GraspFailure("INVALID_RETURN", "unknown custody outcome")
            if self.clock() >= self.deadline:
                self.state, self.reason = "uncertain", "CLAIM_EXPIRED"
                self.changed()
                raise GraspFailure("STALE_CLAIM", "claim has expired")
            self.state = {"held": "picked", "released": "released", "lost": "lost", "uncertain": "uncertain"}[outcome]
            self.owner, self.reason = "pick" if outcome == "held" else "", reason
            # A terminal place failure returns the original, still-held
            # handover pose for put-back. Start that recovery at the next
            # tick instead of blocking new commands for another full lease.
            self.deadline = self.clock() if outcome == "held" and reason else self.clock() + self.lease_s
            self.token, self._unknown_since = "", None
            self.changed()

    def tick(self):
        # (motion, reason when it succeeds, reason when it does not), run outside the lock.
        recovery = None
        recovered = None
        with self.lock:
            if self.state == "claimed" and self.clock() >= self.deadline:
                self.state, self.reason = "uncertain", "CLAIM_EXPIRED"
                self.changed()
            elif self.state == "uncertain":
                if self.manual_reset is not None and self.clock() >= self.manual_reset["deadline"]:
                    self.diagnostic({**self.manual_reset, "operation": "MANUAL_RESET_EXPIRED", "succeeded": False})
                    self.manual_reset, self._settled_since = None, None
                if self.reason == DROP_HOME_FAILED or self.manual_reset is not None:
                    recovered = self._settle()
            elif self.state == "picked":
                grip = self.holding()
                if grip is None:
                    self._unknown_since = self.clock() if self._unknown_since is None else self._unknown_since
                    if self.clock() - self._unknown_since >= 1.0:
                        self.state, self.reason = "uncertain", "GRASP_UNCERTAIN"
                        self.changed()
                elif grip and self.drop is None:
                    self.state, self.reason = "uncertain", "GRIP_LOST"
                    self.changed()
                elif grip:
                    self.state, self.reason = "recovering", "GRIP_LOST"
                    recovery = self.drop, ("lost", "GRIP_LOST"), "DROP_RECOVERY_FAILED"
                    self.changed()
                else:
                    self._unknown_since = None
                    if self.clock() >= self.deadline:
                        self.state = "recovering"
                        done = "PLACE_FAILED_PUT_BACK" if self.reason else "LEASE_EXPIRED_PUT_BACK"
                        recovery = self.recover, ("released", done), "RECOVERY_FAILED"
                        self.changed()
        if recovery is not None:
            motion, done, failed = recovery
            report = {
                "custody_id": self.held.custody_id,
                "started_at": self.clock(),
                "operation": failed,
                "clock": "custody_monotonic",
            }
            try:
                outcome = motion()
                ok = outcome.succeeded if hasattr(outcome, "succeeded") else bool(outcome)
                report["result"] = asdict(outcome) if is_dataclass(outcome) else {"succeeded": ok}
            except Exception as exc:
                ok = False
                report["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            with self.lock:
                report.update(ended_at=self.clock(), succeeded=ok)
                self.recovery_report = report
                if not ok and failed == "DROP_RECOVERY_FAILED" and _released(report.get("result")):
                    failed = DROP_HOME_FAILED
                    report["operation"] = failed
                self.state, self.reason = done if ok else ("uncertain", failed)
                self._settled_since = None
                self.changed()
            self.diagnostic(report)
        if recovered is not None:
            self.diagnostic(recovered)

    def _settle(self):
        """Under the lock: end a DROP_HOME_FAILED custody once its hand stays open and home."""
        try:
            settled = self.settled is not None and bool(self.settled())
        except Exception:  # noqa: BLE001 - stale feedback is "not yet", never a reason to clear
            settled = False
        if not settled:
            self._settled_since = None
            return None
        now = self.clock()
        self._settled_since = now if self._settled_since is None else self._settled_since
        if now - self._settled_since < SETTLED_S:
            return None
        manual = self.manual_reset
        self.state, self.reason = ("released", "MANUAL_RESET_COMPLETE") if manual else ("lost", "GRIP_LOST")
        self.manual_reset, self._settled_since = None, None
        self.changed()
        return {
            **(manual or {}),
            "custody_id": self.held.custody_id,
            "started_at": now,
            "ended_at": now,
            "operation": "MANUAL_RESET_COMPLETE" if manual else "DROP_HOME_RECOVERED",
            "clock": "custody_monotonic",
            "succeeded": True,
        }


def _released(result) -> bool:
    """Whether a failed drop recovery still opened the hand (its release phase completed)."""
    return isinstance(result, dict) and "release" in (result.get("completed") or ())


class HeldCustodyServer:
    def __init__(self, node, *, lease_s=30.0):
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile

        from ibrobot_msgs.msg import HeldObject as Message
        from ibrobot_msgs.srv import AuthorizeHeldReset, ClaimHeldObject, ReturnHeldObject

        self.node = node
        self.publisher = node.create_publisher(
            Message, TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.custody = HeldCustody(lease_s=lease_s, changed=self.publish, diagnostic=self.log_recovery)
        group = MutuallyExclusiveCallbackGroup()
        self.claim_service = node.create_service(ClaimHeldObject, CLAIM, self._claim, callback_group=group)
        self.return_service = node.create_service(ReturnHeldObject, RETURN, self._return, callback_group=group)
        self.authorize_reset_service = node.create_service(
            AuthorizeHeldReset, AUTHORIZE_RESET, self._authorize_reset, callback_group=group
        )
        self.timer = node.create_timer(0.05, self.custody.tick, callback_group=group)
        # A (re)started pick executor holds nothing: replace any custody a
        # subscriber latched from the previous process, so the Gateway unblocks.
        empty = Message(state="empty")
        empty.header.stamp = node.get_clock().now().to_msg()
        self.publisher.publish(empty)

    def log_recovery(self, report):
        import json
        import math

        def finite(value):
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, list | tuple):
                return [finite(item) for item in value]
            # A sample without a middleware receive stamp is deliberately -inf.
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            return value

        text = json.dumps(finite(report), default=str, allow_nan=False)
        # rclpy binds severity to the call site; keep these on separate lines.
        if report["succeeded"]:
            self.node.get_logger().info(f"held custody recovery: {text}")
        else:
            self.node.get_logger().error(f"held custody recovery: {text}")

    def publish(self):
        custody = self.custody
        if custody.held is not None:
            message = held_to_msg(
                custody.held,
                state=custody.state,
                owner=custody.owner,
                reason=custody.reason,
                root_task_id=custody.root_task_id,
            )
            message.header.stamp = self.node.get_clock().now().to_msg()
            self.publisher.publish(message)
            if custody.state == "uncertain":
                self.node.get_logger().error(f"held custody needs manual check: {custody.reason}")
            elif custody.state == "lost":
                self.node.get_logger().warning(f"held object lost: {custody.reason}; the hand is open and home")

    def _claim(self, request, response):
        try:
            binding = request.dispatch_binding
            now_ns = self.node.get_clock().now().nanoseconds
            end = binding.task_budget.deadline.sec * 1_000_000_000 + binding.task_budget.deadline.nanosec
            if (
                binding.schema_version != 1
                or not binding.root_task_id
                or not binding.dispatch_nonce
                or end <= now_ns
                or binding.root_task_id != self.custody.root_task_id
            ):
                raise GraspFailure("EXECUTION_SCOPE_DENIED", "claim needs an active delegated dispatch")
            token, held = self.custody.claim(
                request.custody_id,
                request.target_query,
                request.claimant,
                token=request.claim_token,
                lease_s=request.lease_s,
            )
            response.success, response.claim_token = True, token
            response.held_object = held_to_msg(held, state="claimed", owner=request.claimant)
        except GraspFailure as failure:
            response.reason_code = failure.code
        return response

    def _return(self, request, response):
        try:
            self.custody.returned(request.custody_id, request.claim_token, request.outcome, request.reason_code)
            response.success = True
        except GraspFailure as failure:
            response.reason_code = failure.code
        return response

    def _authorize_reset(self, request, response):
        try:
            if request.schema_version != 1:
                raise GraspFailure("SKILL_SCHEMA_INVALID", "schema_version must be 1")
            remaining = self.custody.authorize_reset(
                request.custody_id,
                request.request_id,
                operator_confirmed_empty=request.operator_confirmed_empty,
                operator_note=request.operator_note,
            )
            response.success, response.custody_id = True, request.custody_id
            response.remaining_sec = remaining
            response.message = (
                "operator check recorded; custody clears only after fresh empty-hand and home verification"
            )
        except GraspFailure as failure:
            response.error_code, response.message = failure.code, str(failure)
        return response
