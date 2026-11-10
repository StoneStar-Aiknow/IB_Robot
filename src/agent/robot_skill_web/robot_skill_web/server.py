from __future__ import annotations

import copy
import json
import math
import re
import secrets
import threading
import traceback
from contextlib import suppress
from functools import wraps
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transports.ros_bridge import BridgeError

from .security import CSRF_COOKIE, HTTP_SESSION_COOKIE, SessionStore
from .serialization import (
    RequestError,
    error_payload,
    parse_json_body,
    require_object,
    validate_connection_version,
    validate_id,
)
from .submission_store import PendingSubmissionStore

MUTATING_METHODS = {"POST", "DELETE"}
STATIC_ROOT = files(__package__).joinpath("static")
SKILL_PATH = re.compile(r"/api/skills/([^/]+)\Z")
#: How long a submit response waits for a receipt before returning the durable
#: pending reference.  The reference is recorded before the client can send,
#: so the browser can always query instead of polling the session forever.
SUBMISSION_RESPONSE_TIMEOUT_SEC = 5.0


def bound_client(*, config_name: str | None = None, config_path: str | None = None) -> RobotClient:
    """Bind one server process to the deployed robot configuration.

    Only local files are read: resolving the configuration never contacts the
    Gateway, so starting the server cannot send an action. The resulting
    neutral transport is the single target every request is scoped to.

    The binding needs only the public ``embodied`` transport contract, so it
    uses the transport projection: a deployment that declares logical runtime
    interfaces (and therefore has no interface snapshot yet) still binds the
    same Gateway endpoints as its launch. This is not a full configuration
    validation; catalog and execution data always come from the Gateway.
    """
    from robot_client.transport import gateway_transport
    from robot_config.loader import load_transport_robot_config

    projection = load_transport_robot_config(config_name=config_name, config_path=config_path)
    transport = gateway_transport(projection.embodied, robot_id=projection.name, robot_name=projection.name)
    return RobotClient(transport)


def serialized_mutation(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        # Version checks and their mutation must observe the same connection.
        with self.state.lock:
            return method(self, *args, **kwargs)

    return invoke


def _text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _request_id(reference: Any) -> str:
    """Read the immutable request identity from a saved or incoming reference."""
    return _text_value(reference.get("request_id")) if isinstance(reference, dict) else ""


class WebState:
    def __init__(
        self,
        *,
        client: RobotClient,
        submission_directory: Any = None,
    ) -> None:
        self.client = client
        self.sessions = SessionStore()
        self.submissions = PendingSubmissionStore(submission_directory)
        self.lock = threading.RLock()
        self.connection_id: str | None = None
        self.previews: dict[str, dict[str, Any]] = {}
        self.submission_workers: dict[str, dict[str, Any]] = {}
        self.worker_lock = threading.Lock()
        self.recover_unsent()

    def require_connection(self, connection_version: Any = None) -> dict[str, Any]:
        with self.lock:
            session = self.client.session
            if not session:
                raise RequestError("the deployment is not bound to a target", status=409)
            if connection_version is not None and session["connection_version"] != validate_connection_version(
                connection_version
            ):
                raise RequestError("connection version is stale; refresh the workbench", status=409)
            if self.connection_id != session["connection_id"]:
                # A backend restart or a newly connected client invalidates previews.
                self.previews.clear()
                self.connection_id = session["connection_id"]
            return session

    def blocking_submission(self) -> dict[str, Any] | None:
        """Return the saved submission that owns admission, if any.

        A durable pending record is enough to block a second goal: the worker
        map is process-local and disappears on restart, while the record is
        the single source of truth for "one motion at a time".  The oldest
        unresolved record is returned so a caller can query its exact saved
        reference instead of guessing which task is active.

        Web admission lives here because this store is durable while the worker
        map is process-local.  The client lock is restored separately for
        observation only, and both a terminal snapshot and an unambiguous
        terminal event page release it, so recovery never leaves a permanent
        client lock behind.
        """
        unresolved = self.submissions.unresolved()
        if unresolved:
            return min(unresolved, key=lambda record: record.get("submitted_at", 0))
        with self.worker_lock:
            for worker in self.submission_workers.values():
                if not worker["done"] and worker["reference"].get("target_key"):
                    return {
                        "preview_id": "",
                        "phase": "submitting",
                        "state": "unknown",
                        "reference": copy.deepcopy(worker["reference"]),
                    }
        return None

    def unresolved_executions(self) -> list[dict[str, Any]]:
        return self.submissions.unresolved()

    def live_request_ids(self) -> set[str]:
        """Return the request identities this process still owns a worker for."""
        with self.worker_lock:
            workers = list(self.submission_workers.values())
        return {
            _request_id(worker.get("reference"))
            for worker in workers
            if not worker["done"] and _request_id(worker.get("reference"))
        }

    def recover_unsent(self) -> list[str]:
        """Release only the startup intents this process provably never sent.

        The worker map is empty when this runs at startup, so a live worker can
        only appear on a repeated call; it is passed as the exclusion set so a
        second recovery pass can never release an intent whose worker is still
        before the send boundary.  A corrupt or unreadable state directory is
        left to the usual fail-closed path on first use instead of being
        silently rewritten here.
        """
        try:
            session = self.client.session
            target_key = _text_value(session.get("target_key")) if isinstance(session, dict) else ""
            if not target_key:
                return []
            return self.submissions.recover_unsent(
                target_key=target_key,
                exclude_request_ids=self.live_request_ids(),
            )
        except (AttributeError, TypeError, OSError, ClientError):
            return []

    def restore_submission(self, reference: dict[str, Any]) -> bool:
        """Adopt a durable reference into the client's explicit admission lock.

        ``RobotClient.get_execution`` is read-only and never turns a foreign
        reference into a lock.  A Web process that recovers its own persisted
        record may adopt it explicitly, but only while the record is still
        unresolved and possibly sent, and only after the store lock proves that
        a concurrent rejection did not already finalize it.  A request this
        process still owns a live worker for is never adopted a second time:
        that client submission adds the same lock itself and removes it when
        the goal is rejected, so re-adding it here could revive a rejected
        request.  Client doubles without ``restore_ref`` keep relying on the
        Web store alone.
        """
        if _request_id(reference) in self.live_request_ids():
            return False

        def adopt(stored: dict[str, Any]) -> None:
            restore = getattr(self.client, "restore_ref", None)
            if restore is not None:
                restore(copy.deepcopy(stored))

        return self.submissions.adopt_if_unresolved(reference, adopt)

    def close(self) -> None:
        self.client.close()


class WebServer:
    def __init__(
        self,
        *,
        client: RobotClient | None = None,
        config_name: str | None = None,
        config_path: str | None = None,
        address: str = "127.0.0.1",
        port: int = 8765,
        submission_directory: Any = None,
    ) -> None:
        if address != "127.0.0.1" or not 1 <= port <= 65535:
            raise ValueError("WebServer only supports 127.0.0.1 and ports 1-65535")
        self.state = WebState(
            client=client or bound_client(config_name=config_name, config_path=config_path),
            submission_directory=submission_directory,
        )
        self.address = address
        self.port = port
        self.httpd: _Server | None = None

    def serve_forever(self) -> None:
        self.httpd = _Server((self.address, self.port), _RequestHandler, state=self.state)
        self.httpd.serve_forever(poll_interval=0.25)

    def close(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.state.close()

    def __enter__(self) -> WebServer:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Permit an orderly restart while old TCP connections remain in TIME_WAIT.
    # This does not share the port with an existing listener (no SO_REUSEPORT).
    allow_reuse_address = True

    def __init__(self, address, handler, *, state: WebState) -> None:
        super().__init__(address, handler)
        self.web_state = state

    def service_actions(self) -> None:
        state = self.web_state
        with state.worker_lock:
            finished = [key for key, worker in state.submission_workers.items() if worker["done"]]
            for key in finished:
                state.submission_workers.pop(key, None)


class _RequestHandler(SimpleHTTPRequestHandler):
    server_version = "robot-skill-web/0.1"
    sys_version = ""
    timeout = 15

    def __init__(self, request, client_address, server) -> None:
        self.state = server.web_state
        super().__init__(request, client_address, server, directory=str(STATIC_ROOT))

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep credentials and query strings out of local terminal logs.
        print(f"{self.address_string()} {self.command} {self.path.split('?', 1)[0]}", flush=True)

    def version_string(self) -> str:
        return self.server_version

    def end_headers(self) -> None:
        # HTML and scripts must not outlive each other during local UI updates.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, value: dict[str, Any], *, headers: list[tuple[str, str]] | None = None) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'",
        )
        for name, content in headers or []:
            self.send_header(name, content)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def cookies(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for part in self.headers.get("Cookie", "").split(";"):
            if "=" in part:
                key, value = part.strip().split("=", 1)
                result[key] = value
        return result

    def enforce_local_and_csrf(self, *, mutation: bool) -> None:
        self._enforce_host()
        host = self.headers.get("Host", "")
        if mutation:
            origin = self.headers.get("Origin")
            if origin != f"http://{host}":
                raise RequestError("Origin is not allowed", status=403)
            cookies = self.cookies()
            expected = cookies.get(CSRF_COOKIE, "")
            supplied = self.headers.get("X-CSRF-Token", "")
            if not expected or not supplied or not secrets.compare_digest(expected, supplied):
                raise RequestError("CSRF validation failed", status=403)
            token = cookies.get(HTTP_SESSION_COOKIE)
            if token:
                session = self.state.sessions.lookup(token)
                if not secrets.compare_digest(session.csrf, supplied):
                    raise RequestError("CSRF token does not belong to this session", status=403)

    def _enforce_host(self) -> None:
        host = self.headers.get("Host", "")
        if host != f"{self.server.server_address[0]}:{self.server.server_address[1]}":
            raise RequestError("Host is not allowed", status=403)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("DELETE")

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("GET", head_only=True)

    def _dispatch(self, method: str, *, head_only: bool = False) -> None:
        try:
            path = urlsplit(self.path).path
            if not path.startswith("/api/"):
                if method == "GET":
                    self._enforce_host()
                    if head_only:
                        super().do_HEAD()
                    else:
                        super().do_GET()
                    return
                self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
                return
            session_token = self.cookies().get(HTTP_SESSION_COOKIE)
            self.enforce_local_and_csrf(mutation=method in MUTATING_METHODS)
            if path == "/api/session":
                if method == "POST":
                    self._establish_session()
                    return
                if method == "GET":
                    session = None
                    if session_token is not None:
                        try:
                            session = self.state.sessions.lookup(session_token)
                        except PermissionError:
                            session_token = None
                    csrf = session.csrf if session else self.state.sessions.issue_csrf()
                    session_active = session_token is not None
                    headers = [("Set-Cookie", f"{CSRF_COOKIE}={csrf}; Path=/; SameSite=Strict; Max-Age=28800")]
                    if not session_active:
                        headers.append(
                            ("Set-Cookie", f"{HTTP_SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
                        )
                    self.send_json(
                        200,
                        {
                            "session_active": session_active,
                            "connection": self.state.client.session if session_active else None,
                            "unresolved_executions": self.state.unresolved_executions() if session_active else [],
                        },
                        headers=headers,
                    )
                    return
            elif session_token is not None:
                self.state.sessions.lookup(session_token)
                if method == "GET" and path == "/api/status":
                    self.state.require_connection()
                    self.send_json(200, self.state.client.get_status())
                    return
                if method == "GET" and (path == "/api/history" or path.startswith("/api/history/")):
                    self._get_history(path)
                    return
                if method == "GET" and path == "/api/skills":
                    self.state.require_connection()
                    listing = self.state.client.list_skills()
                    listing["skills"] = [
                        skill for skill in listing.get("skills", []) if skill.get("planner_visible") is True
                    ]
                    self.send_json(200, listing)
                    return
                skill_match = SKILL_PATH.fullmatch(path)
                if method == "GET" and skill_match:
                    self._send_skill(unquote(skill_match.group(1)))
                    return
                if method == "POST" and path == "/api/previews":
                    self._create_preview()
                    return
                if method == "POST" and path == "/api/executions":
                    self._start_execution()
                    return
                if method == "GET" and path.startswith("/api/executions/"):
                    self._get_execution(path)
                    return
                if method == "POST" and path.startswith("/api/executions/") and path.endswith("/cancel"):
                    self._cancel_execution(path)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._safe_error(exc)

    @serialized_mutation
    def _establish_session(self) -> None:
        """Bind this browser to the deployment the server was started with.

        The target was resolved at startup, so this only issues the session and
        CSRF cookies. It never contacts the Gateway and never sends an action.
        """
        # No credential or target field is accepted: the target was fixed at startup.
        require_object(parse_json_body(self), set())
        connection = self.state.client.session
        if not connection:
            raise ClientError("NOT_CONNECTED", "the Web server is not bound to a deployment robot config")
        self.state.sessions.check_connection_attempt()
        unresolved = self.state.submissions.unresolved()
        if any(record["reference"].get("target_key") != connection["target_key"] for record in unresolved):
            self.state.sessions.record_connection_failure()
            raise RequestError(
                "unresolved tasks belong to another deployment; observe them there before using this one",
                status=409,
            )
        for record in unresolved:
            try:
                self.state.restore_submission(record["reference"])
            except ClientError as exc:
                self.state.sessions.record_connection_failure()
                raise RequestError(
                    "a saved unresolved task cannot be adopted by this session; observe it in its original scope",
                    status=409,
                ) from exc
        token = self.state.sessions.issue_session()
        csrf = self.state.sessions.lookup(token).csrf
        self.send_json(
            201,
            {
                "session_active": True,
                "connection": connection,
                "unresolved_executions": self.state.unresolved_executions(),
            },
            headers=[
                ("Set-Cookie", f"{HTTP_SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=28800"),
                ("Set-Cookie", f"{CSRF_COOKIE}={csrf}; Path=/; SameSite=Strict; Max-Age=28800"),
            ],
        )

    def _send_skill(self, name: str) -> None:
        self.state.require_connection()
        try:
            self.send_json(200, self.state.client.describe_skill(name))
        except ClientError as exc:
            if exc.code == "UNKNOWN_SKILL":
                self.send_json(404, {"error": {"code": "UNKNOWN_SKILL", "message": str(exc)}})
            else:
                raise

    def _get_history(self, path: str) -> None:
        session = self.state.require_connection()
        scope = {
            "robot_id": session["robot_id"],
            "caller_scope": session["caller_scope"],
            "target_key": session["target_key"],
        }
        if path == "/api/history":
            self.send_json(200, self.state.submissions.history(**scope))
            return
        preview_id = validate_id(unquote(path[len("/api/history/") :]), "preview_id")
        record = self.state.submissions.history_record(preview_id, **scope)
        if record is None:
            self.send_json(404, {"error": {"code": "HISTORY_NOT_FOUND", "message": "history record not found"}})
            return
        self.send_json(200, {"record": record})

    @serialized_mutation
    def _create_preview(self) -> None:
        value = require_object(
            parse_json_body(self),
            {"skill_name", "raw_command", "parameters", "connection_version"},
            optional={"request_id"},
        )
        self.state.require_connection(value["connection_version"])
        blocking = self.state.blocking_submission()
        if blocking is not None:
            self.state.restore_submission(blocking["reference"])
            raise ClientError(
                "EXECUTION_UNRESOLVED",
                "observe the current task before preparing another",
                reference=copy.deepcopy(blocking["reference"]),
            )
        skill_name = validate_id(value["skill_name"], "skill_name")
        raw_command = value["raw_command"]
        if not isinstance(raw_command, str) or not raw_command.strip() or len(raw_command) > 4000:
            raise RequestError("raw_command must be nonempty text of at most 4000 characters")
        if not isinstance(value["parameters"], dict):
            raise RequestError("parameters must be an object")
        skill = self.state.client.describe_skill(skill_name)
        if skill.get("planner_visible") is not True:
            raise RequestError("this skill is not exposed to the Web planner", status=403)
        step = build_workflow_step(skill, value["parameters"])
        request: dict[str, Any] = {
            "workflow_steps": [step],
            "raw_command": raw_command,
        }
        if "request_id" in value:
            request["request_id"] = validate_id(value["request_id"], "request_id")
        preview = self.state.client.prepare_workflow(request)
        with self.state.lock:
            self.state.previews.clear()
            self.state.previews[preview["preview_id"]] = {
                **copy.deepcopy(preview),
                "command": {
                    "skill_name": skill_name,
                    "display_name": skill.get("display_name") or skill_name,
                    "raw_command": raw_command,
                    "parameters": copy.deepcopy(value["parameters"]),
                },
            }
        self.send_json(201, preview)

    @serialized_mutation
    def _start_execution(self) -> None:
        value = require_object(parse_json_body(self), {"preview_id", "connection_version"})
        session = self.state.require_connection(value["connection_version"])
        preview_id = validate_id(value["preview_id"], "preview_id")
        durable = self.state.submissions.get(preview_id)
        preview = self.state.previews.get(preview_id)
        if durable is not None:
            ref = durable["reference"]
            if (
                ref.get("target_key") != session["target_key"]
                or ref.get("caller_scope") != session.get("caller_scope", "")
                or ref.get("robot_id") != session["robot_id"]
            ):
                raise RequestError("submission belongs to another target or caller", status=409)
            if preview is not None and preview.get("request_id") != ref.get("request_id"):
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    "preview request_id conflicts with the saved submission",
                    reference=copy.deepcopy(ref),
                )
            if not (
                durable.get("state") in {"REJECTED", "SUCCEEDED", "FAILED", "CANCELED"}
                or (durable.get("state") == "UNKNOWN" and durable.get("reconciled") is True)
            ):
                self.state.restore_submission(ref)
            if durable.get("state") == "REJECTED":
                failure = durable.get("failure") if isinstance(durable.get("failure"), dict) else {}
                self.send_json(
                    200,
                    {
                        "execution": durable["reference"],
                        "already_submitted": True,
                        "state": durable.get("state"),
                        "submission_rejected": True,
                        "error": {
                            "code": failure.get("code", "GOAL_REJECTED"),
                            "message": failure.get("message") or "Gateway explicitly rejected the submitted goal",
                        },
                    },
                )
                return
            self.send_json(
                202,
                {
                    "execution": durable["reference"],
                    "already_submitted": True,
                    "state": durable.get("state"),
                    "phase": durable.get("phase"),
                },
            )
            return
        if preview is None:
            raise RequestError("preview is stale; generate a new plan", status=409)
        preview_request_id = preview.get("request_id")
        if not isinstance(preview_request_id, str) or not preview_request_id:
            raise RequestError("preview has no request identity; generate a new plan", status=409)
        # A repeated request identity is the same submission even when the UI
        # generated a fresh preview after a reload.  Return its saved reference
        # instead of replaying the goal.
        same_request = self.state.submissions.find_by_request(
            preview_request_id,
            target_key=session["target_key"],
            robot_id=session["robot_id"],
            caller_scope=session.get("caller_scope", ""),
        )
        if same_request is not None:
            self.send_json(
                202,
                {
                    "execution": same_request["reference"],
                    "already_submitted": True,
                    "state": same_request.get("state"),
                    "phase": same_request.get("phase"),
                },
            )
            return
        with self.state.worker_lock:
            existing = self.state.submission_workers.get(preview_id)
        if existing is not None:
            self._send_submission(existing, already_submitted=True)
            return
        blocking = self.state.blocking_submission()
        if blocking is not None:
            self.state.restore_submission(blocking["reference"])
            raise ClientError(
                "EXECUTION_UNRESOLVED",
                "another task is active or unobserved; query the saved execution before submitting again",
                reference=copy.deepcopy(blocking["reference"]),
            )

        # Record the request identity before the client starts its pre-goal
        # checks.  If the receipt response is lost (or this process dies before
        # the client calls back), the browser still has a durable reference to
        # query, and a restart cannot silently send a second goal.
        intent = {
            "target_key": session["target_key"],
            "robot_id": session["robot_id"],
            "caller_scope": session.get("caller_scope", ""),
            "request_id": preview_request_id,
            "execution_id": "",
            "legacy_task_id": "",
        }
        record = self.state.submissions.reserve_intent(
            preview_id,
            session["connection_id"],
            intent,
            command=preview.get("command"),
        )
        ready = threading.Condition()
        with self.state.worker_lock:
            worker: dict[str, Any] = {
                "state": "submitting",
                "reference": copy.deepcopy(record["reference"]),
                "result": None,
                "error": None,
                "accepted": False,
                "done": False,
                "ready": ready,
            }
            self.state.submission_workers[preview_id] = worker

        def run() -> None:
            persist_called = False
            try:

                def persist(reference: dict[str, Any]) -> None:
                    nonlocal persist_called
                    updated = self.state.submissions.mark_goal_may_be_sent(preview_id, reference)
                    worker["reference"] = copy.deepcopy(updated["reference"])
                    persist_called = True
                    with ready:
                        ready.notify_all()

                worker["result"] = self.state.client.submit_prepared(
                    preview_id,
                    presenter=presentation_complete,
                    persist_reference=persist,
                )
                if isinstance(worker.get("result"), dict):
                    worker["reference"] = copy.deepcopy(worker["result"])
                self.state.submissions.mark_accepted(preview_id, worker.get("result"))
                worker["accepted"] = True
                worker["state"] = "accepted"
            except ClientError as exc:
                goal_may_be_sent = persist_called or bool(exc.observation_unknown)
                rejected = (not goal_may_be_sent) or exc.code == "GOAL_REJECTED"
                worker["state"] = "rejected" if rejected else "unknown"
                worker["error"] = {
                    "code": exc.code,
                    "message": str(exc),
                    "reference": copy.deepcopy(exc.reference),
                }
                if exc.reference is not None:
                    worker["reference"] = copy.deepcopy(exc.reference)
                self.state.submissions.mark_failure(
                    preview_id,
                    error_code=exc.code,
                    message=str(exc),
                    rejected=rejected,
                )
            except Exception:
                worker["state"] = "unknown"
                worker["error"] = {"code": "INTERNAL_ERROR", "message": "submission outcome is unknown"}
                self.state.submissions.mark_failure(
                    preview_id,
                    error_code="INTERNAL_ERROR",
                    message="submission outcome is unknown",
                    rejected=False,
                )
            finally:
                worker["done"] = True
                with ready:
                    ready.notify_all()

        threading.Thread(target=run, name=f"submit-{preview_id[:12]}", daemon=True).start()
        self._send_submission(worker, already_submitted=False)

    def _send_submission(self, worker, *, already_submitted):
        # Wait for a receipt, not for task completion.  The durable intent is
        # already stored, so even a timeout returns a reference the browser can
        # query; it never turns into an unbounded session-polling loop.
        if not already_submitted:
            ready = worker["ready"]
            with ready:
                ready.wait_for(
                    lambda: worker.get("accepted") or worker["done"],
                    timeout=SUBMISSION_RESPONSE_TIMEOUT_SEC,
                )
        reference = copy.deepcopy(worker["reference"])
        if worker.get("accepted") or already_submitted:
            self.send_json(
                202,
                {
                    "execution": reference,
                    "already_submitted": already_submitted,
                    "state": worker.get("state"),
                },
            )
            return
        if worker["done"] and worker["state"] == "rejected" and worker["error"] is not None:
            error = worker["error"]
            status, payload = error_payload(ClientError(error["code"], error["message"]))
            self.send_json(status, payload)
            return
        if not reference.get("target_key"):
            # Only possible if the local intent record disappeared; never
            # claim a submittable reference in that state.
            self.send_json(
                503,
                {"error": {"code": "SUBMISSION_PENDING", "message": "submission reference is not yet observable"}},
            )
            return
        self.send_json(
            503,
            {
                "error": {
                    "code": "SUBMISSION_PENDING",
                    "message": "submission receipt is not observable yet; query without replay",
                },
                "execution": reference,
                "observation_unknown": True,
            },
        )

    def _get_execution(self, path: str) -> None:
        decoded = decode_execution(path[len("/api/executions/") :])
        session = self.state.require_connection()
        if decoded.get("target_key") != session["target_key"] or decoded.get("caller_scope") != session["caller_scope"]:
            raise RequestError("execution belongs to a different target or caller", status=409)
        # Match by immutable request identity, not by the mutable execution_id
        # association: a browser keeps the reference it received at submit time
        # even after the ledger links an execution id to it.  This GET is an
        # observation only: it may release an existing client lock when the
        # read-only ledger proves a terminal result, but it must never adopt a
        # new one.  Adopting here would let a poll that read a nonterminal
        # record re-add the client lock after the worker already cleared it on
        # GOAL_REJECTED, reviving a rejected request for the rest of the
        # process lifetime.
        durable = self.state.submissions.find_for_reference(decoded)
        if durable is not None and durable.get("state") == "REJECTED":
            failure = durable.get("failure") if isinstance(durable.get("failure"), dict) else {}
            self.send_json(
                200,
                {
                    "execution": {
                        "found": False,
                        "submission_rejected": True,
                        "error_code": failure.get("code", ""),
                        "message": failure.get("message") or "Gateway explicitly rejected the submitted goal",
                    },
                    "events": {"events": []},
                },
            )
            return
        query = dict(parse_qsl(urlsplit(self.path).query, keep_blank_values=True))
        after = 0
        if "after_sequence" in query:
            try:
                after = int(query["after_sequence"], 10)
            except ValueError as exc:
                raise RequestError("after_sequence must be an integer") from exc
            if after < 0:
                raise RequestError("after_sequence must be nonnegative")

        # Query the two read-only sources independently.  A snapshot timeout
        # must not prevent the durable ledger event page from associating a
        # pending submission or from reporting its terminal state.
        execution_error: ClientError | BridgeError | None = None
        try:
            result = self.state.client.get_execution(decoded)
        except (ClientError, BridgeError) as exc:
            execution_error = exc
            result = {
                "found": False,
                "observation_unknown": True,
                "error_code": exc.code,
                "message": str(exc),
            }
        try:
            events = self.state.client.get_execution_events(decoded, after_sequence=after)
        except (ClientError, BridgeError) as exc:
            events = {"events": [], "unavailable": True, "error_code": exc.code, "message": str(exc)}

        if durable is not None:
            reconciliation = self.state.submissions.reconcile(
                durable["preview_id"],
                result,
                events,
                reference=decoded,
            )
            if reconciliation.get("conflict"):
                raise ClientError(
                    "EXECUTION_IDENTITY_CONFLICT",
                    str(reconciliation["conflict"]),
                    reference=copy.deepcopy(durable["reference"]),
                )
            if reconciliation.get("terminal"):
                # The saved terminal observation is authoritative for a late
                # or unavailable snapshot; it must not downgrade the result.
                # Keep the response shape equal to the already cached raw
                # observation whenever the snapshot itself was terminal.
                if not result.get("found"):
                    result = {
                        **result,
                        "found": True,
                        "state": reconciliation["state"],
                        "observation_unknown": False,
                    }
                elif result.get("state") != reconciliation["state"]:
                    result = {**result, "state": reconciliation["state"]}
                if reconciliation.get("execution_id") and not result.get("execution_id"):
                    result["execution_id"] = reconciliation["execution_id"]
                if reconciliation["state"] == "UNKNOWN":
                    result["reconciled"] = bool(reconciliation.get("reconciled"))
                # Cache the same synthesized terminal view that the response
                # returns, so history and the next client poll agree even when
                # the terminal state came only from the ledger event page.
                self.state.submissions.observe(durable["preview_id"], result, events)
                execution_error = None
        if execution_error is not None:
            raise execution_error
        self.send_json(200, {"execution": result, "events": events})

    @serialized_mutation
    def _cancel_execution(self, path: str) -> None:
        value = require_object(parse_json_body(self), {"connection_version"})
        self.state.require_connection(value["connection_version"])
        decoded = decode_execution(path[len("/api/executions/") : -len("/cancel")])
        self.send_json(202, self.state.client.cancel_execution(decoded))

    def _safe_error(self, exc: Exception) -> None:
        status, payload = error_payload(exc)
        if status >= 500:
            traceback.print_exc()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.send_json(status, payload)


def presentation_complete(preview: dict[str, Any]) -> None:
    # The browser has already rendered this exact preview before invoking submit.
    if not isinstance(preview, dict) or not preview.get("preview_id"):
        raise RequestError("preview presentation is incomplete", status=409)


def build_workflow_step(skill: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    schema = skill.get("parameters")
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        raise RequestError("skill parameter schema is unavailable")
    properties = schema["properties"]
    required = set(schema.get("required", []))
    writable = {
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "arm_side",
        "imitation_duration_sec",
        "timeout_sec",
        "direction",
        "distance",
        "degree",
        "x",
        "y",
        "yaw",
    }
    unknown = set(parameters) - set(properties)
    if unknown:
        raise RequestError("unsupported parameters: " + ", ".join(sorted(unknown)))
    missing = required - set(parameters)
    if missing:
        raise RequestError("missing required parameters: " + ", ".join(sorted(missing)))
    unsupported = set(properties) - writable
    if unsupported:
        raise RequestError("the Web UI does not support parameters: " + ", ".join(sorted(unsupported)))
    version = skill.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise RequestError("unsupported WorkflowStep schema version")
    step: dict[str, Any] = {"skill_name": skill["name"], "schema_version": version}
    for name, value in parameters.items():
        property_schema = properties[name]
        if not isinstance(property_schema, dict) or property_schema.get("type") not in {"string", "number"}:
            raise RequestError(f"the Web UI does not support parameter type for {name}")
        expected = property_schema["type"]
        if expected == "string":
            if not isinstance(value, str):
                raise RequestError(f"{name} must be a string")
            values = property_schema.get("enum")
            if isinstance(values, list) and value not in values:
                raise RequestError(f"{name} must match the allowed values")
            step[name] = value
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RequestError(f"{name} must be a number")
            minimum = property_schema.get("exclusiveMinimum")
            if isinstance(minimum, int | float) and value <= minimum:
                raise RequestError(f"{name} must be greater than the schema minimum")
            maximum = property_schema.get("exclusiveMaximum")
            if isinstance(maximum, int | float) and value >= maximum:
                raise RequestError(f"{name} must be less than the schema maximum")
            if "minimum" in property_schema and value < property_schema["minimum"]:
                raise RequestError(f"{name} is below the schema minimum")
            if "maximum" in property_schema and value > property_schema["maximum"]:
                raise RequestError(f"{name} exceeds the schema maximum")
            step[name] = value
    navigation = {"direction", "distance", "degree", "x", "y", "yaw"}
    if navigation.intersection(step):
        if type(version) is not int or version < 2:
            raise RequestError("navigation parameters require contract schema_version 2")
        step["schema_version"] = version
        for axis in ("x", "y", "yaw"):
            if axis in step:
                step[f"has_{axis}"] = True
    return step


def decode_execution(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(unquote(value))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RequestError("execution reference is invalid") from exc
    if not isinstance(decoded, dict):
        raise RequestError("execution reference is invalid")
    return decoded
