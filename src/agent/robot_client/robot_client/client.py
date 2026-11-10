"""Reusable robot access, with no local robot resources or execution authority.

The client is bound to one explicit :class:`GatewayTransport` for its whole
lifetime: it never discovers, selects or remembers targets. Callers resolve the
deployment configuration at their own boundary (the CLI and the Web workbench
do this from ``robot_config``) and hand the resulting transport in.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from contextlib import suppress

from .connection import bridge_kwargs, ros_environment_scope
from .errors import ClientError
from .transport import GatewayTransport

#: Status fields that identify one Gateway lifecycle and catalog revision.
#: A restart changes ``registry_epoch``; a reload or recompile changes
#: ``registry_generation``/the digests. The set is the preview freshness key.
_GATEWAY_IDENTITY_FIELDS = (
    "robot_name",
    "registry_epoch",
    "registry_generation",
    "registry_digest",
    "capability_digest",
)


def _transport_factory(**kwargs):
    from .transports.resident import ResidentBridge

    return ResidentBridge(**kwargs)


class RobotClient:
    def __init__(self, transport: GatewayTransport, *, transport_factory=None):
        if not isinstance(transport, GatewayTransport):
            raise ClientError(
                "INVALID_TRANSPORT", "bind an explicit GatewayTransport resolved from the deployed robot config"
            )
        self._transport = transport
        self._factory = transport_factory or _transport_factory
        self._connection_id = uuid.uuid4().hex
        self._target_key = f"{ros_environment_scope()}:{transport.identity_digest()}"
        self._bridge = None
        self._previews: dict[str, dict] = {}
        self._submitted: dict[str, dict] = {}
        self._unresolved: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._operations = threading.RLock()

    @property
    def transport(self) -> GatewayTransport:
        return self._transport

    @property
    def session(self):
        with self._lock:
            return {
                "connection_id": self._connection_id,
                "connection_version": 1,
                "robot_id": self._transport.robot_id,
                "robot_name": self._transport.target_name,
                "caller_scope": self._transport.caller_scope,
                "target_key": self._target_key,
                "read_only": False,
            }

    def _bridge_for_operations(self):
        with self._lock:
            if self._bridge is None:
                self._bridge = self._start_bridge()
            return self._bridge

    def _start_bridge(self):
        """Build and start the one transport this client operates through.

        The transport is created on the first operation, never when the client
        is bound, so constructing a client (the Web server startup path) does not
        initialize ROS or contact the Gateway. ``start()`` belongs to the
        transport contract: a transport that was never started answers every
        read with ``ROS_UNAVAILABLE`` (``ROS bridge is not started``) instead of
        the Gateway's status, and the Web adapter reports that as an internal
        error. A failed start is reported as the same ``ROS_UNAVAILABLE`` the CLI
        exits with, carries the transport's ``start_error`` diagnostic, and
        leaves no half-started bridge behind; the next operation retries.
        """
        bridge = self._factory(**bridge_kwargs(self._transport))
        target = self._transport.target_name or "the bound Gateway"
        try:
            started = bridge.start()
        except Exception as exc:
            with suppress(Exception):
                bridge.close()
            raise ClientError(
                "ROS_UNAVAILABLE",
                f"failed to initialize the ROS bridge for {target}: {type(exc).__name__}: {exc}",
            ) from exc
        if not started:
            detail = str(getattr(bridge, "start_error", "") or "").strip()
            with suppress(Exception):
                bridge.close()
            message = f"failed to initialize the ROS bridge for {target}"
            raise ClientError("ROS_UNAVAILABLE", f"{message}: {detail}" if detail else message)
        return bridge

    def _check_robot_name(self, value) -> None:
        expected = self._transport.target_name
        if expected and value != expected:
            raise ClientError(
                "TARGET_IDENTITY_CHANGED",
                "the Gateway reports a different robot than the bound configuration; check --config-name/--config-path",
            )

    def _gateway_identity(self, bridge) -> dict:
        """Read the business identity a preview is valid against, without discovery."""
        status = bridge.get_status(timeout_sec=5.0)
        self._check_robot_name(status.get("robot_name"))
        return {field: status.get(field) for field in _GATEWAY_IDENTITY_FIELDS}

    def get_status(self):
        bridge = self._bridge_for_operations()
        result = bridge.get_status(timeout_sec=5.0)
        self._check_robot_name(result.get("robot_name"))
        return result

    def catalog_view(self):
        from .catalog import capability_view_from_snapshot

        bridge = self._bridge_for_operations()
        for attempt in range(2):
            status = self.get_status()
            if not status.get("control_plane_ready"):
                raise ClientError(
                    "CONTROL_PLANE_NOT_READY", status.get("control_plane_error_code") or "catalog not ready"
                )
            snapshot = bridge.get_skill_snapshot(
                registry_epoch=status["registry_epoch"],
                generation=status["registry_generation"],
                timeout_sec=5.0,
            )
            if not snapshot.get("success"):
                if attempt == 0 and snapshot.get("error_code") in {
                    "SKILL_SNAPSHOT_NOT_FOUND",
                    "SKILL_REGISTRY_EPOCH_MISMATCH",
                    "SKILL_REGISTRY_NOT_READY",
                }:
                    continue
                raise ClientError(snapshot.get("error_code") or "SNAPSHOT_UNAVAILABLE", snapshot.get("message", ""))
            try:
                view = capability_view_from_snapshot(snapshot, status)
            except ValueError as exc:
                code, _, message = str(exc).partition(":")
                raise ClientError(code or "SKILL_SNAPSHOT_DIGEST_MISMATCH", message or str(exc)) from exc
            self._check_robot_name(view.get("robot_name"))
            readiness = {item["name"]: item for item in status.get("capabilities", [])}
            for skill in view.get("skills", []):
                observed = readiness.get(skill["name"], {})
                skill["ready"] = bool(observed.get("ready", False))
                skill["reason"] = observed.get("reason", "NOT_OBSERVED")
            return view
        raise ClientError("REGISTRY_CHANGED", "catalog changed during query")

    def list_skills(self):
        from .catalog import list_skills

        view = self.catalog_view()
        result = list_skills(view)
        by_name = {item["name"]: item for item in view["skills"]}
        for item in result.get("skills", []):
            source = by_name[item["name"]]
            for key in ("ready", "reason", "planner_visible", "parameters", "schema_version", "verification_status"):
                if key in source:
                    item[key] = copy.deepcopy(source[key])
        return result

    def describe_skill(self, name):
        from .catalog import UnknownSkillError, describe_skill

        view = self.catalog_view()
        try:
            result = describe_skill(view, name)
        except UnknownSkillError as exc:
            raise ClientError("UNKNOWN_SKILL", str(exc)) from exc
        source = next((skill for skill in view["skills"] if skill["name"] == result["name"]), {})
        for key in ("ready", "reason", "planner_visible", "verification_status"):
            if key in source:
                result[key] = copy.deepcopy(source[key])
        return result

    def list_poses(self):
        from .catalog import list_poses

        return list_poses(self.catalog_view())

    def prepare_workflow(self, request):
        from embodied_common.agent_execution_contract import IMMEDIATE_AFTER_PRESENTATION

        from .workflow import InteractiveController

        if not isinstance(request, dict) or set(request) - {"workflow_steps", "raw_command", "request_id"}:
            raise ClientError("INVALID_REQUEST", "unsupported workflow fields")
        if not isinstance(request.get("workflow_steps"), list) or not request["workflow_steps"]:
            raise ClientError("INVALID_REQUEST", "workflow_steps must be a nonempty list")
        raw = request.get("raw_command", "")
        request_id = request.get("request_id", uuid.uuid4().hex)
        if not isinstance(raw, str) or not isinstance(request_id, str) or not request_id.strip():
            raise ClientError("INVALID_REQUEST", "raw_command and request_id must be strings")
        with self._operations:
            bridge = self._bridge_for_operations()
            with self._lock:
                if self._unresolved:
                    raise ClientError("EXECUTION_UNRESOLVED", "observe the current task before preparing another")
                connection_id = self._connection_id
            controller = InteractiveController(bridge, timeout_policy={}, execution_mode=IMMEDIATE_AFTER_PRESENTATION)
            controller.discover()
            plan = controller.prepare_workflow(raw, request["workflow_steps"], request_id=request_id)
            preview = {
                "preview_id": uuid.uuid4().hex,
                "connection_id": connection_id,
                "connection_version": 1,
                "request_id": request_id,
                "target": {"robot_id": self._transport.robot_id, "robot_name": self._transport.target_name},
                "gateway_identity": self._gateway_identity(bridge),
                "plan": copy.deepcopy(plan),
            }
            with self._lock:
                self._previews.clear()  # Only the latest exact plan may be submitted.
                self._previews[preview["preview_id"]] = {
                    "public": copy.deepcopy(preview),
                    "controller": controller,
                }
            return preview

    def submit_prepared(self, preview_id, *, presenter, persist_reference):
        with self._operations:
            bridge = self._bridge_for_operations()
            with self._lock:
                if preview_id in self._submitted:
                    return copy.deepcopy(self._submitted[preview_id])
                prepared = self._previews.get(preview_id)
                if prepared is None or prepared["public"]["connection_id"] != self._connection_id:
                    raise ClientError("STALE_PREVIEW", "prepare and display a new plan")
                if self._unresolved:
                    raise ClientError("EXECUTION_UNRESOLVED", "another task is active or unobserved")
            current = self._gateway_identity(bridge)
            if current != prepared["public"]["gateway_identity"]:
                with self._lock:
                    self._previews.clear()
                raise ClientError(
                    "STALE_PREVIEW", "Gateway restarted or its catalog changed; prepare and display a new plan"
                )
            preview = copy.deepcopy(prepared["public"])
            presenter(copy.deepcopy(preview))
            confirmed = prepared["controller"].confirm_plan()
            if confirmed.get("stopped_before_execution"):
                raise ClientError("STOPPED_BEFORE_EXECUTION", "the prepared operation was stopped")
            if not bridge.wait_for_program_server(timeout_sec=5.0):
                raise ClientError("SERVER_UNAVAILABLE", "Program action is not ready")
            ref = {
                "target_key": self._target_key,
                "robot_id": self._transport.robot_id,
                "caller_scope": self._transport.caller_scope,
                "request_id": preview["request_id"],
                "execution_id": "",
                "legacy_task_id": "",
            }
            persist_reference(copy.deepcopy(ref))
            with self._lock:
                self._submitted[preview_id] = copy.deepcopy(ref)
                self._unresolved[ref["request_id"]] = copy.deepcopy(ref)
                self._previews.pop(preview_id, None)
            try:
                future = bridge.send_program_goal(
                    request_id=ref["request_id"],
                    workflow_steps=preview["plan"]["steps"],
                    registry=preview["plan"]["registry_identity"],
                    task_budget_sec=confirmed["confirmed_task_budget_sec"],
                )
                if not bridge.wait_future(future, timeout_sec=5.0):
                    raise ClientError(
                        "OBSERVATION_UNKNOWN", "goal receipt was not observed", reference=ref, observation_unknown=True
                    )
                handle = future.result()
                if handle is None or not handle.accepted:
                    with self._lock:
                        self._unresolved.pop(ref["request_id"], None)
                    raise ClientError("GOAL_REJECTED", "Gateway rejected the request", reference=ref)
            except ClientError:
                raise
            except Exception as exc:
                raise ClientError(
                    "OBSERVATION_UNKNOWN",
                    "submission outcome is unknown; query without replay",
                    reference=ref,
                    observation_unknown=True,
                ) from exc
            return copy.deepcopy(ref)

    def _reference(self, ref):
        allowed = {
            "request_id",
            "execution_id",
            "legacy_task_id",
            "target_key",
            "robot_id",
            "caller_scope",
        }
        string_fields = ("request_id", "execution_id", "legacy_task_id", "target_key", "robot_id", "caller_scope")
        if not isinstance(ref, dict) or set(ref) - allowed:
            raise ClientError("INVALID_REFERENCE", "an execution reference with request_id is required")
        if not isinstance(ref.get("request_id"), str) or not ref["request_id"] or len(ref["request_id"]) > 256:
            raise ClientError("INVALID_REFERENCE", "request_id must be a non-empty string of at most 256 characters")
        for field in string_fields[1:]:
            if field in ref and (not isinstance(ref[field], str) or len(ref[field]) > 256):
                raise ClientError("INVALID_REFERENCE", f"reference {field} must be a string of at most 256 characters")
        if ref.get("target_key") != self._target_key or ref.get("robot_id") != self._transport.robot_id:
            raise ClientError("WRONG_TARGET", "execution belongs to a different deployment")
        if ref.get("caller_scope") != self._transport.caller_scope or ref.get("legacy_task_id"):
            raise ClientError("INVALID_REFERENCE", "execution scope or identifier kind does not match")
        return {key: ref[key] for key in allowed if key in ref}

    def get_execution(self, ref):
        bridge = self._bridge_for_operations()
        checked = self._reference(ref)
        result = bridge.get_execution(
            request_id=checked["request_id"],
            execution_id=checked.get("execution_id", ""),
            timeout_sec=5.0,
        )
        with self._lock:
            if result.get("found") and (
                result.get("state") in {"SUCCEEDED", "FAILED", "CANCELED"}
                or (result.get("state") == "UNKNOWN" and result.get("reconciled"))
            ):
                self._unresolved.pop(checked["request_id"], None)
                for submitted in self._submitted.values():
                    if submitted["request_id"] == checked["request_id"]:
                        submitted["execution_id"] = result.get("execution_id", "")
            else:
                self._unresolved[checked["request_id"]] = copy.deepcopy(checked)
        return {**result, "observation_unknown": not bool(result.get("found"))}

    def get_execution_events(self, ref, after_sequence=0):
        bridge = self._bridge_for_operations()
        checked = self._reference(ref)
        if type(after_sequence) is not int or after_sequence < 0:
            raise ClientError("INVALID_CURSOR", "event sequence must be a nonnegative integer")
        result = bridge.get_execution_events(
            request_id=checked["request_id"],
            execution_id=checked.get("execution_id", ""),
            after_sequence=after_sequence,
            timeout_sec=5.0,
        )
        from .models import public_value

        try:
            result = public_value(result)
            if not isinstance(result, dict):
                raise TypeError("response is not an object")
            execution_id = result["execution_id"]
            events = result["events"]
            last_event_sequence = result["last_event_sequence"]
            has_more = result["has_more"]
            if not isinstance(execution_id, str):
                raise TypeError("execution_id is not a string")
            if not isinstance(events, list):
                raise TypeError("events is not a list")
            if type(last_event_sequence) is not int or last_event_sequence < 0:
                raise TypeError("last_event_sequence is not a nonnegative integer")
            if type(has_more) is not bool:
                raise TypeError("has_more is not a boolean")
        except ClientError:
            raise
        except Exception as exc:
            raise ClientError("INVALID_RESPONSE", "execution event response was invalid") from exc
        unique_events = {}
        for event in events:
            try:
                event_execution_id = event["execution_id"]
                sequence = event["sequence"]
            except (KeyError, TypeError) as exc:
                raise ClientError("INVALID_RESPONSE", "execution event response was invalid") from exc
            if not isinstance(event_execution_id, str) or type(sequence) is not int:
                raise ClientError("INVALID_RESPONSE", "execution event response was invalid")
            if event_execution_id != execution_id:
                raise ClientError("EVENT_IDENTITY_MISMATCH", "event belongs to another execution")
            if sequence > after_sequence:
                unique_events[sequence] = event
        ordered = sorted(unique_events)
        previous = after_sequence
        gap = False
        for sequence in ordered:
            gap = gap or sequence != previous + 1
            previous = sequence
        result["events"] = [unique_events[sequence] for sequence in ordered]
        last_returned = ordered[-1] if ordered else after_sequence
        result["event_gap"] = gap or (last_event_sequence > last_returned and not has_more)
        return result

    def cancel_execution(self, ref):
        bridge = self._bridge_for_operations()
        checked = self._reference(ref)
        return bridge.cancel_execution(
            request_id=checked["request_id"], execution_id=checked.get("execution_id", ""), timeout_sec=5.0
        )

    def wait_execution(self, ref, *, deadline):
        while time.monotonic() < deadline:
            observed = self.get_execution(ref)
            if observed.get("terminal"):
                return observed
            time.sleep(0.1)
        raise ClientError("TIMEOUT", "execution did not reach a terminal state before the deadline")

    def close(self):
        with self._lock:
            bridge = self._bridge
            self._bridge = None
        if bridge is not None:
            bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
