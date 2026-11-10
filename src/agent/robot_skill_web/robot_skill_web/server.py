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
    SubmissionStore,
    error_payload,
    parse_json_body,
    require_object,
    validate_connection_version,
    validate_id,
)

MUTATING_METHODS = {"POST", "DELETE"}
STATIC_ROOT = files(__package__).joinpath("static")
SKILL_PATH = re.compile(r"/api/skills/([^/]+)\Z")


def bound_client(*, config_name: str | None = None, config_path: str | None = None) -> RobotClient:
    """Bind one server process to the deployed robot configuration.

    Only local files are read: resolving the configuration never contacts the
    Gateway, so starting the server cannot send an action. The resulting
    neutral transport is the single target every request is scoped to.
    """
    from robot_client.transport import gateway_transport
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict

    resolved = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved)
    robot_name = str(robot_config.get("name", ""))
    transport = gateway_transport(robot_config.get("embodied", {}), robot_id=robot_name, robot_name=robot_name)
    return RobotClient(transport)


def serialized_mutation(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        # Version checks and their mutation must observe the same connection.
        with self.state.lock:
            return method(self, *args, **kwargs)

    return invoke


class WebState:
    def __init__(
        self,
        *,
        client: RobotClient,
        submission_directory: Any = None,
    ) -> None:
        self.client = client
        self.sessions = SessionStore()
        self.submissions = SubmissionStore(submission_directory)
        self.lock = threading.RLock()
        self.connection_id: str | None = None
        self.previews: dict[str, dict[str, Any]] = {}
        self.submission_workers: dict[str, dict[str, Any]] = {}
        self.worker_lock = threading.Lock()

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

    def active_or_unknown(self) -> bool:
        if self.submissions.unresolved():
            return True
        with self.worker_lock:
            return any(not worker["done"] for worker in self.submission_workers.values())

    def unresolved_executions(self) -> list[dict[str, Any]]:
        return self.submissions.unresolved()

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
        if self.state.active_or_unknown():
            raise RequestError("observe the current task before preparing another", status=409)
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
        if durable is not None:
            ref = durable["reference"]
            if ref["target_key"] != session["target_key"] or ref["caller_scope"] != session["caller_scope"]:
                raise RequestError("submission belongs to another target or caller", status=409)
            self.send_json(202, {"execution": durable["reference"], "already_submitted": True})
            return
        preview = self.state.previews.get(preview_id)
        if preview is None:
            raise RequestError("preview is stale; generate a new plan", status=409)
        with self.state.worker_lock:
            existing = self.state.submission_workers.get(preview_id)
        if existing is not None:
            self._send_submission(existing, already_submitted=True)
            return
        if self.state.active_or_unknown():
            raise RequestError("another task is active or unobserved", status=409)
        worker_ref = {
            "target_key": "",
            "robot_id": session["robot_id"],
            "caller_scope": session.get("caller_scope", ""),
            "request_id": preview["request_id"],
            "execution_id": "",
            "legacy_task_id": "",
        }
        ready = threading.Condition()
        with self.state.worker_lock:
            worker: dict[str, Any] = {
                "state": "submitting",
                "reference": worker_ref,
                "result": None,
                "error": None,
                "done": False,
                "ready": ready,
            }
            self.state.submission_workers[preview_id] = worker

        def run() -> None:
            try:

                def persist(reference: dict[str, Any]) -> None:
                    record = self.state.submissions.reserve(
                        preview_id, session["connection_id"], reference, command=preview.get("command")
                    )
                    worker["reference"] = copy.deepcopy(record["reference"])
                    with ready:
                        ready.notify_all()

                worker["result"] = self.state.client.submit_prepared(
                    preview_id,
                    presenter=presentation_complete,
                    persist_reference=persist,
                )
                worker["state"] = "accepted"
                if isinstance(worker.get("result"), dict):
                    worker["reference"] = copy.deepcopy(worker["result"])
            except ClientError as exc:
                worker["state"] = "unknown" if exc.observation_unknown else "rejected"
                worker["error"] = {"code": exc.code, "message": str(exc), "reference": copy.deepcopy(exc.reference)}
                if exc.reference is not None:
                    worker["reference"] = copy.deepcopy(exc.reference)
                if exc.code == "GOAL_REJECTED" and exc.reference is not None:
                    self.state.submissions.resolve(preview_id, "REJECTED")
            except Exception:
                worker["state"] = "unknown"
                worker["error"] = {"code": "INTERNAL_ERROR", "message": "submission outcome is unknown"}
            finally:
                worker["done"] = True
                with ready:
                    ready.notify_all()

        threading.Thread(target=run, name=f"submit-{preview_id[:12]}", daemon=True).start()
        self._send_submission(worker, already_submitted=False)

    def _send_submission(self, worker, *, already_submitted):
        # Wait only for the pre-goal durable reference, never for task completion.
        ready = worker["ready"]
        with ready:
            ready.wait_for(lambda: worker["reference"].get("target_key") or worker["done"], timeout=5.0)
        if not worker["reference"].get("target_key"):
            if worker["error"] is not None:
                status, payload = error_payload(ClientError(worker["error"]["code"], worker["error"]["message"]))
                self.send_json(status, payload)
                return
            self.send_json(
                503,
                {"error": {"code": "SUBMISSION_PENDING", "message": "submission reference is not yet observable"}},
            )
            return
        self.send_json(202, {"execution": worker["reference"], "already_submitted": already_submitted})

    def _get_execution(self, path: str) -> None:
        decoded = decode_execution(path[len("/api/executions/") :])
        session = self.state.require_connection()
        if decoded.get("target_key") != session["target_key"] or decoded.get("caller_scope") != session["caller_scope"]:
            raise RequestError("execution belongs to a different target or caller", status=409)
        durable = self.state.submissions.by_request(decoded.get("request_id", ""), reference=decoded)
        if durable and durable["reference"] == decoded and durable.get("state") == "REJECTED":
            self.send_json(
                200,
                {
                    "execution": {
                        "found": False,
                        "submission_rejected": True,
                        "message": "Gateway explicitly rejected the submitted goal",
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
        result = self.state.client.get_execution(decoded)
        if result.get("found"):
            state = result.get("state", "")
            if durable is not None and (
                state in {"SUCCEEDED", "FAILED", "CANCELED"} or (state == "UNKNOWN" and result.get("reconciled"))
            ):
                self.state.submissions.resolve(durable["preview_id"], state, reconciled=bool(result.get("reconciled")))
        try:
            events = self.state.client.get_execution_events(decoded, after_sequence=after)
        except (ClientError, BridgeError) as exc:
            events = {"events": [], "unavailable": True, "error_code": exc.code, "message": str(exc)}
        if durable is not None and durable["reference"] == decoded:
            self.state.submissions.observe(durable["preview_id"], result, events)
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
