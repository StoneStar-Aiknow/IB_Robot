"""RT-Infer device-scheduler launch owner (Ascend 310P software scheduling).

This builder is the single launch-time owner of the RT-Infer device scheduler
on robots that opt in via ``robot.rt_infer.enable: true`` (only the
oee_summit_2026 X2 live entries do). Responsibilities, in order — the order
is load-bearing (review #F): the *coverage preflight* runs before anything
touches the device, so a launch with an unmapped Ascend model is refused
before the module is loaded, not after the device is taken over.

1. **Coverage preflight** (no device access): every Ascend deployment this
   launch will start must have an RT-Infer role. An unmapped Ascend model is
   a launch error here — a native process on a taken-over device is never
   allowed, and finding that out after insmod would be too late.
2. **Install** (idempotent, guarded): the bundle payload shipped with this
   repository (source tree or installed share — ``rt_infer_bundle`` resolves
   it) is installed to the canonical prefix ``/opt/rt-infer/<version>`` by
   the bundle's own bootstrap ``priorityctl.py`` (the prefix does not exist
   yet, so the bootstrap must come from the bundle itself). After the
   install, every control call uses the installed copy.
3. **Start before the models**: doctor + (create the periodic mapping) +
   ``start`` under an *exclusive owner lock* (flock; a second owner refuses
   conservatively instead of racing), then poll the vendor state file for a
   **live** scheduler identity. Any failure shuts the launch down with the
   reason — never a silent native fallback.
4. **Supervise** (review #3/E): ``rt_infer_scheduler_supervisor`` polls the
   vendor state for the *fixed startup identity* of the daemon this launch
   started; a missing/corrupt state file, a swapped pid or a dead daemon all
   exit the supervisor non-zero, and the launch shuts the dependent model
   processes down.
5. **Stop in order** (review #4/D): only the launch that started the
   scheduler stops it. The launch stamps a token into its wrapped clients'
   controlled environment (the vendor strips inherited RT_INFER_* variables,
   so the token rides the explicit ``/usr/bin/env`` argv, not the parent
   environment); shutdown cleanup signals *only* clients that are (a) still
   the same live process (pidfd-verified identity) and (b) carrying this
   launch's token. On any verification doubt the cleanup stops, reports and
   leaves the state alone — it never widens the kill. The launch shutdown
   callback does not wait; a launch-tracked asynchronous action performs the
   bounded drain and vendor stop while launch dispatches client signals.
6. **Retain unproven starts**: failed/timed-out vendor start does not prove
   daemon ownership. Release this launch's lock, retain the original state
   and require explicit recovery; never stop a possibly foreign daemon.

The vendor ``priorityctl.py`` remains the only lifecycle implementation; this
owner orchestrates it and adds the launch-side invariants (preflight, lock,
token-scoped cleanup, supervision) the vendor CLI cannot know.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from launch import Action
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch_ros.actions import Node

from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger
from robot_config.rt_infer_bundle import bootstrap_control_script, expected_toolkit_home, resolve_vendor_bundle
from robot_config.rt_infer_config import RtInferConfig, RtInferConfigError, parse_rt_infer_config

logger = get_colored_logger("robot_config.launch_builders.rt_infer_scheduler")

STATE_DIR = Path("/run/ibrobot-rtinfer")
STATE_FILE = STATE_DIR / "state.json"
OWNER_LOCK = STATE_DIR / "launch-owner.lock"


class _SchedulerOwnership:
    """What this launch found when it came up.

    ``started_here`` is the lease decision: only a launch that started the
    scheduler may stop it. ``start_attempted`` records an issued command;
    it does not establish ownership when readiness was not confirmed.
    """

    def __init__(self) -> None:
        self.started_here = False
        self.start_attempted = False
        self.verified_running = False
        self.lock_handle = None
        self.token = ""
        self.scheduler_identity: dict[str, Any] = {}
        self.boot_id = _boot_id()
        self.shutting_down = False
        self.roles: set[str] = set()


def rt_infer_enabled(context: LaunchBuildContext) -> bool:
    """Participate only when the merged YAML enables RT-Infer for this launch.

    Review #6: the gate sees the *resolved* launch target (CLI overrides
    included), so an inherited board config with ``use_sim:=true`` or a
    benchmark target never touches the device even though the YAML says
    enable. Parse errors are launch-fatal: fail loudly instead of running
    native while the configuration believes RT-Infer is on.
    """
    try:
        config = parse_rt_infer_config(context.robot_config)
    except RtInferConfigError as exc:
        raise RuntimeError(f"rt_infer configuration is invalid: {exc}") from exc
    if not config.enable:
        return False
    _refuse_simulated_target(context)
    return True


def _refuse_simulated_target(context: LaunchBuildContext) -> None:
    """Review #6: no simulated/benchmark launch may start the device owner."""
    if context.use_sim or (context.runtime_target is not None and context.runtime_target.value != "hardware"):
        raise RuntimeError(
            "rt_infer.enable=true is not compatible with this launch target "
            f"(use_sim={context.use_sim}, runtime_target={getattr(context.runtime_target, 'value', None)!r}); "
            "a simulated or benchmark launch must not start the device scheduler"
        )


def _run_ctl(script: Path, prefix: str, arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    # install declares its own required --prefix subcommand option.
    command = ["python3", str(script), *([] if arguments[0] == "install" else ["--prefix", prefix]), *arguments]
    environment = os.environ.copy()
    toolkit_home = expected_toolkit_home(script.parent.parent)
    if toolkit_home is not None:
        environment["ASCEND_TOOLKIT_HOME"] = toolkit_home
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def _installed_ctl(config: RtInferConfig, arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    return _run_ctl(Path(config.control_tool), config.prefix, arguments, timeout=timeout)


def _vendor_state() -> dict[str, Any]:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"RT-Infer state cannot be verified: {STATE_FILE}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"RT-Infer state must be an object: {STATE_FILE}")
    return state


def _identity_alive(identity: dict[str, Any]) -> bool:
    """A vendor identity (pid + start_ticks) still names a live process."""
    proc = Path("/proc") / str(identity.get("pid"))
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    if len(stat) <= 19 or stat[0] == "Z":
        return False
    return str(stat[19]) == str(identity.get("start_ticks"))


def _scheduler_alive(state: dict[str, Any]) -> bool:
    identity = state.get("scheduler")
    return bool(identity) and _identity_alive(identity)


def _process_identity(pid: int) -> dict[str, Any]:
    stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": stat[19]}


def _launch_token() -> str:
    """A per-launch identity carried by every client this launch wraps.

    The vendor's clean_environment strips inherited RT_INFER_*/RTINFER_*
    variables, so the token cannot ride the parent environment: the wrapper
    puts it in the explicit ``/usr/bin/env`` argv it already builds, and this
    owner reads it back from the vendor state's client list (the vendor
    records the client's environment-visible argv is not available to us —
    instead the wrapper writes its own token file under the vendor state
    dir, keyed by its pid, which cleanup verifies against the live process
    before signaling).
    """
    return f"launch-{os.getpid()}-{time.monotonic_ns()}"


def _acquire_owner_lock(token: str) -> Any:
    """Exclusive, non-blocking owner lock; a second owner refuses.

    Two concurrent launches must not both claim the scheduler: the loser
    would tear down a scheduler the winner still depends on. A stale lock is
    reclaimed only when no live launch process matches the recorded
    identity; anything unverifiable is refused conservatively.
    """
    import fcntl

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    handle = (OWNER_LOCK).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Someone holds the lock. Read the recorded holder from the TOP of
        # the file and decide conservatively; never block waiting for it.
        handle.seek(0)
        try:
            holder = json.loads(handle.read() or "{}")
        except ValueError:
            holder = {}
        handle.close()
        if _identity_alive(holder.get("identity") or {}):
            raise RuntimeError(
                f"another RT-Infer launch owner is active (pid {holder.get('identity', {}).get('pid')}); "
                "refusing to touch its scheduler"
            ) from None
        # The holder is dead but its fd is still open somewhere (a zombie
        # parent, an inherited descriptor): retrying would block, and we
        # never touch a scheduler a live holder's lock still covers.
        raise RuntimeError(
            "the RT-Infer owner lock is held by a stale file descriptor (its launch is dead, "
            "but the descriptor is still open); resolve the stale holder before starting RT-Infer"
        ) from None
    identity = _process_identity(os.getpid())
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"identity": identity, "token": token, "boot_id": _boot_id()}) + "\n")
    handle.flush()
    return handle


def _install_bundle(config: RtInferConfig) -> None:
    """Install the repository bundle to the canonical prefix (bootstrap).

    The prefix does not exist before this runs, so the install command is
    executed by the *bundle's own* bootstrap priorityctl.py (shipped with the
    repository payload); after it succeeds, the installed copy serves every
    later control call. The vendor install keeps every guard: root,
    trusted-tree, hash verification, and idempotence against a different
    prefix content (version discipline is a reboot rule, not a file swap).
    """
    bundle = resolve_vendor_bundle(config.bundle_version, config.bundle_dir)
    script = bootstrap_control_script(bundle)
    # Validate payload hashes before executing the supplied SVM probe. The
    # probe only binds this temporary process; it never submits NPU work or
    # replaces a driver. Unsupported SVM must fail before device takeover.
    verified = _run_ctl(
        script, config.prefix, ["verify-bundle", "--bundle", str(bundle)], timeout=config.verify_timeout_sec
    )
    if verified.returncode:
        raise RuntimeError(f"RT-Infer bundle verification failed: {verified.stderr or verified.stdout}")
    _probe_svm(bundle, config.verify_timeout_sec)
    _prepare_install_parent(config)
    result = _run_ctl(
        script,
        config.prefix,
        ["install", "--bundle", str(bundle), "--prefix", config.prefix],
        timeout=config.verify_timeout_sec,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RT-Infer bundle install to {config.prefix} failed: {(result.stderr or result.stdout).strip()}"
        )
    logger.info(f"RT-Infer bundle {config.bundle_version} verified/installed at {config.prefix}")


def _probe_svm(bundle: Path, timeout_sec: float) -> None:
    probe = subprocess.run(
        ["python3", str(bundle / "svm_bind_probe.py"), "--pid", "self", "--pasid", "-1"],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"RT-Infer SVM ABI guard failed: {probe.stderr or probe.stdout}")


def _prepare_install_parent(config: RtInferConfig) -> None:
    """Prepare only the canonical parent; vendor install verifies ownership."""
    Path(config.prefix).parent.mkdir(mode=0o755, parents=True, exist_ok=True)


def _create_periodic_mapping(config: RtInferConfig) -> dict[str, int]:
    """Create unarmed; the owner coordinator arms after model readiness."""
    import ctypes

    control_path = Path(config.pear_periodic.control_path)
    budget_ns = config.pear_periodic.budget_ns
    bridge = Path(config.prefix) / "lib" / "periodic_control_bridge.so"
    if not bridge.is_file():
        raise RuntimeError(f"RT-Infer periodic control bridge is missing from the installed prefix: {bridge}")
    library = ctypes.CDLL(str(bridge), use_errno=True)
    library.periodic_shared_create.restype = ctypes.c_void_p
    library.periodic_shared_create.argtypes = [ctypes.c_char_p]
    library.periodic_shared_close.argtypes = [ctypes.c_void_p]
    library.periodic_shared_close.restype = None
    handle = library.periodic_shared_create(str(control_path).encode())
    if not handle:
        raise RuntimeError(
            f"Creating the RT-Infer periodic control mapping failed at {control_path}: errno {ctypes.get_errno()}"
        )
    period_ns = (1_000_000_000 + config.pear_periodic.period_hz - 1) // config.pear_periodic.period_hz
    created_ns = time.monotonic_ns()
    library.periodic_shared_close(handle)
    logger.info(
        f"Created unarmed RT-Infer mapping at {control_path} "
        f"(period {period_ns} ns, budget {budget_ns} ns); waiting for runtime readiness"
    )
    return {
        "epoch_ns": 0,
        "created_ns": created_ns,
        "period_ns": period_ns,
        "budget_ns": budget_ns,
    }


def _write_periodic_session(
    config: RtInferConfig,
    ownership: _SchedulerOwnership,
    mapping: dict[str, int],
) -> None:
    """Publish the owner/scheduler identity clients must validate."""
    control_path = Path(config.pear_periodic.control_path)
    session_path = control_path.parent / "periodic-session.json"
    document = {
        **mapping,
        "boot_id": _boot_id(),
        "prefix": config.prefix,
        "control_path": str(control_path),
        "launch_token": ownership.token,
        "scheduler": ownership.scheduler_identity,
        "owner": _process_identity(os.getpid()),
        "expected_roles": sorted(ownership.roles),
        "phase": "initializing",
    }
    temporary = session_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(session_path)


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _wait_scheduler_ready(config: RtInferConfig) -> dict[str, Any]:
    """Poll the vendor state until the scheduler is *live* and running.

    Liveness (review #3) is the identity check, not the status string: a
    dead daemon with a stale file must never fake readiness.
    """
    deadline = time.monotonic() + config.readiness_timeout_sec
    last_status = None
    while time.monotonic() < deadline:
        state = _vendor_state()
        status = state.get("status")
        last_status = status
        if (
            status == "running"
            and state.get("prefix") == config.prefix
            and state.get("boot_id") == _boot_id()
            and _scheduler_alive(state)
        ):
            logger.info(
                f"RT-Infer scheduler ready and live (strategy={state.get('strategy')}, "
                f"cpu={state.get('cpu')}, pid={state.get('scheduler', {}).get('pid')})"
            )
            return state
        if status in ("start-failed", "recovery-required"):
            raise RuntimeError(
                f"RT-Infer scheduler entered status {status!r}; inspect {STATE_FILE} and the vendor "
                "scheduler log. A reboot is required before native device use."
            )
        time.sleep(0.2)
    raise RuntimeError(
        f"RT-Infer scheduler was not ready within {config.readiness_timeout_sec}s "
        f"(last status {last_status!r}); aborting launch instead of falling back to native"
    )


def _start_scheduler(config: RtInferConfig, ownership: _SchedulerOwnership) -> dict[str, Any]:
    """Verify, start and confirm the scheduler; returns the running state.

    Host gate first (review #6): this is the only place a config with
    ``enable: true`` meets real hardware, so a non-Ascend host aborts here
    with an actionable error instead of touching the device. The exclusive
    owner lock is taken before the module can be loaded: a second launch
    owner must be refused before insmod, not after (review #D/F).
    """
    import platform

    if platform.machine() != "aarch64":
        raise RuntimeError(
            f"rt_infer.enable=true requires an aarch64 Ascend host; this launch runs on "
            f"{platform.machine()!r}. Remove the rt_infer block on non-Ascend hosts instead "
            "of enabling it"
        )
    _require_idle_state()
    ownership.lock_handle = _acquire_owner_lock(ownership.token)
    # Recheck while holding the launch lease, closing the two-launch race.
    _require_idle_state()
    _install_bundle(config)
    doctor = _installed_ctl(config, ["doctor"], timeout=config.verify_timeout_sec)
    if doctor.returncode != 0:
        raise RuntimeError(f"RT-Infer environment check failed (doctor): {(doctor.stderr or doctor.stdout).strip()}")
    status = _installed_ctl(config, ["status"], timeout=config.verify_timeout_sec)
    if status.returncode != 0:
        raise RuntimeError(f"RT-Infer status preflight failed: {status.stderr or status.stdout}")
    state: dict[str, Any] = {}
    report: dict[str, Any] = {}
    if status.returncode == 0:
        try:
            report = json.loads(status.stdout)
            state = report.get("state", {})
        except (ValueError, AttributeError):
            raise RuntimeError("RT-Infer status preflight returned invalid JSON") from None
    if not isinstance(report, dict) or not isinstance(state, dict):
        raise RuntimeError("RT-Infer status preflight must contain object state")
    if report.get("device_users"):
        raise RuntimeError(f"Ascend device has compute users: {report['device_users']}; refusing takeover")
    if state.get("status") != "stopped" and (report.get("module_loaded") or report.get("ipc")):
        raise RuntimeError("Unmanaged soft scheduler state exists; refusing before mapping mutation")
    if state.get("status") == "running":
        if _scheduler_alive(state) and state.get("prefix") == config.prefix:
            raise RuntimeError(
                "An RT-Infer scheduler became active before this launch could start its own; refusing "
                "foreign scheduler adoption"
            )
        if state.get("prefix") != config.prefix:
            raise RuntimeError(
                f"An RT-Infer scheduler is already running for a different prefix "
                f"({state.get('prefix')!r}); stop it and reboot before launching with {config.prefix}"
            )
        raise RuntimeError(
            "The vendor state reports 'running' but the scheduler process is not alive; "
            "inspect the scheduler log. A reboot is required before native device use."
        )
    if state and state.get("status") not in (None, "stopped", "not-started-this-boot"):
        raise RuntimeError(
            f"RT-Infer state is {state.get('status')!r}; a clean stop or reboot is required "
            "before a new start (vendor rule)"
        )
    start_arguments = [
        "start",
        "--scheduler",
        "pfp",
        "--prefetch",
        "on",
        "--cpu",
        str(config.scheduler_cpu),
        "--pfp-exclusive",
        "on",
        "--pfp-batch-select",
        "on",
        "--local-pool",
        "on",
        "--pfp-min-cache",
        "on",
        "--pfp-hi-priority",
        "0",
    ]
    if config.pear_periodic is not None:
        # No clients/scheduler can own this mapping after the idle checks.
        Path(config.pear_periodic.control_path).unlink(missing_ok=True)
        (Path(config.pear_periodic.control_path).parent / "periodic-session.json").unlink(missing_ok=True)
        periodic_session = _create_periodic_mapping(config)
        start_arguments += ["--pfp-periodic", config.pear_periodic.control_path]
    else:
        periodic_session = None
    ownership.start_attempted = True
    try:
        start = _installed_ctl(
            config,
            start_arguments,
            timeout=config.readiness_timeout_sec + config.verify_timeout_sec,
        )
        if start.returncode != 0:
            raise RuntimeError(f"RT-Infer scheduler start failed: {(start.stderr or start.stdout).strip()}")
        state = _wait_scheduler_ready(config)
    except Exception:
        # External priorityctl callers do not take our launch lease. A
        # concurrent start can win the vendor lock after our idle check;
        # matching prefix/boot is therefore not proof that we spawned it.
        # Timeout also leaves the attempt's outcome unknown. Do not stop
        # anything or rewrite shared state without a confirmed identity.
        _release_owner_lock(ownership)
        logger.error(
            "RT-Infer start ownership is unconfirmed; no stop or shared-state cleanup was performed. "
            f"Manual recovery inspection is required at {STATE_FILE} before retrying or native device use."
        )
        raise
    ownership.verified_running = True
    ownership.scheduler_identity = dict(state.get("scheduler") or {})
    ownership.started_here = True
    if periodic_session is not None:
        _write_periodic_session(config, ownership, periodic_session)
    return state


def _release_owner_lock(ownership: _SchedulerOwnership) -> None:
    if ownership.lock_handle is not None:
        try:
            ownership.lock_handle.close()
        finally:
            ownership.lock_handle = None


def _require_idle_state() -> None:
    state = _vendor_state()
    if not state:
        return
    if state.get("boot_id") != _boot_id():
        raise RuntimeError("Stale RT-Infer state from another boot; refusing takeover")
    if (
        state.get("status") != "stopped"
        or _scheduler_alive(state)
        or any(_identity_alive(identity) for identity in state.get("clients", []))
    ):
        raise RuntimeError("RT-Infer is not idle; refusing to adopt or stop a foreign scheduler/client")


class _StopScheduler(Action):
    """Tracked asynchronous cleanup: launch keeps dispatching child signals."""

    def __init__(self, config: RtInferConfig, ownership: _SchedulerOwnership):
        super().__init__()
        self.config = config
        self.ownership = ownership
        self.future = None

    def execute(self, context):
        if self.future is None:
            self.future = context.asyncio_loop.run_in_executor(None, self._stop)

    def get_asyncio_future(self):
        return self.future

    def _stop(self):
        from robot_config.rt_infer_scheduler_supervisor import _teardown

        try:
            if not _teardown(
                token=self.ownership.token,
                prefix=self.config.prefix,
                control_tool=self.config.control_tool,
                timeout_sec=self.config.stop_timeout_sec,
                periodic_control_path=self.config.pear_periodic.control_path if self.config.pear_periodic else "",
                expected_identity=self.ownership.scheduler_identity,
                expected_boot_id=self.ownership.boot_id,
            ):
                raise RuntimeError("RT-Infer teardown incomplete; state retained, native recovery requires reboot")
        finally:
            _release_owner_lock(self.ownership)


def _supervise_action(config: RtInferConfig, ownership: _SchedulerOwnership):
    """Supervision node for a scheduler this launch started (review #3/#E).

    ``rt_infer_scheduler_supervisor`` (console script in this package) exits 2
    when the *fixed startup identity* of our daemon is no longer verifiable;
    the exit handler maps that to a launch Shutdown so dependent model
    processes stop instead of running unsupervised on a taken-over device.
    Foreign schedulers are refused before this action is created.
    """
    supervisor = Node(
        package="robot_config",
        executable="rt_infer_scheduler_supervisor",
        name="rt_infer_scheduler_supervisor",
        output="screen",
        parameters=[
            {
                "poll_interval_sec": config.client_poll_interval_sec,
                "expected_prefix": config.prefix,
                "expected_scheduler_pid": int(ownership.scheduler_identity["pid"]),
                "expected_scheduler_start_ticks": int(ownership.scheduler_identity["start_ticks"]),
                "expected_boot_id": ownership.boot_id,
                "launch_token": ownership.token,
                "control_tool": config.control_tool,
                "stop_timeout_sec": config.stop_timeout_sec,
                "periodic_control_path": (
                    config.pear_periodic.control_path if config.pear_periodic is not None else ""
                ),
            }
        ],
    )

    def _on_supervisor_exit(event, _ctx):
        # A signal/zero exit is normal only after an actual launch shutdown.
        if ownership.shutting_down:
            return []
        reason = (
            f"RT-Infer scheduler died while the launch was running (supervisor exit "
            f"{event.returncode}); shutting dependent model processes down. A reboot is "
            "required before native device use."
        )
        logger.error(reason)
        raise RuntimeError(reason)

    return RegisterEventHandler(OnProcessExit(target_action=supervisor, on_exit=_on_supervisor_exit)), supervisor


def build_rt_infer_scheduler(context: LaunchBuildContext) -> BuildResult:
    """Bring up the RT-Infer scheduler before any model process starts."""
    config = parse_rt_infer_config(context.robot_config)
    if not config.enable:
        return BuildResult(actions=[])
    _refuse_simulated_target(context)

    ownership = _SchedulerOwnership()

    # The launch identity every wrapped client will carry (the vendor strips
    # inherited RT_INFER_*/RTINFER_* variables, so builders embed it in the
    # controlled env argv; the owner keeps the value in this process for the
    # snapshot writers and the shutdown cleanup).
    ownership.token = _launch_token()

    # Review #F: the coverage preflight runs in the BUILD phase — before this
    # launch (or any of its nodes) touches the device. The model builders
    # register the same check for their own services; the owner re-checks
    # the whole effective config so an unmapped Ascend pipeline (for example
    # a policy pipeline on pipeline_policy_node) refuses the launch before
    # the module is ever loaded.
    roles = _coverage_preflight(context, config)
    ownership.roles = roles
    if not roles:
        logger.info("No local Ascend model roles; RT-Infer owner remains inactive")
        return BuildResult(actions=[])
    if "pear" not in roles:
        config = replace(config, pear_periodic=None)
    # Launch-local snapshots use the effective profile, including no PEAR
    # reservation for look/talk variants. Never inherit another launch token.
    context.robot_config["_rt_infer_launch"] = {"token": ownership.token, "periodic": config.pear_periodic is not None}

    def _start(_launch_context):
        try:
            _start_scheduler(config, ownership)
        except Exception as exc:  # noqa: BLE001 - launch must abort with the reason
            if not ownership.start_attempted:
                _release_owner_lock(ownership)
            logger.error(f"RT-Infer bring-up failed: {exc}")
            # Raising stops traversal before dependent actions can be queued
            # and makes launch return a failing exit status.
            raise RuntimeError(f"RT-Infer bring-up failed: {exc}") from exc
        if not ownership.started_here:
            return []
        registration, supervisor = _supervise_action(config, ownership)
        return [registration, supervisor]

    def _on_shutdown(_event, _ctx):
        ownership.shutting_down = True
        if ownership.started_here:
            return [_StopScheduler(config, ownership)]
        _release_owner_lock(ownership)
        return []

    actions = [
        RegisterEventHandler(OnShutdown(on_shutdown=_on_shutdown)),
        OpaqueFunction(function=_start),
    ]
    logger.info(
        f"RT-Infer launch owner enabled: version={config.bundle_version} prefix={config.prefix}, "
        f"scheduler cpu={config.scheduler_cpu}, "
        f"roles={ {role: (config.priority_for(role), config.cpu_for(role)) for role in ('pear', 'yolox', 'tts', 'vl')} }"
    )
    return BuildResult(actions=actions)


def _coverage_preflight(context: LaunchBuildContext, config: RtInferConfig) -> set[str]:
    """Resolve actual local deployments before any installation/device mutation."""
    from robot_config.perception_runtime_config import parse_perception_runtime_config
    from robot_config.rt_infer_config import deployment_uses_ascend, rt_infer_role_for_manifest
    from robot_config.utils import parse_bool

    robot_config = context.robot_config
    problems: list[str] = []
    roles: set[str] = set()
    if parse_bool(context.arguments.get("with_model_services", "true"), default=True):
        runtime = parse_perception_runtime_config(robot_config)
        mapping = robot_config.get("semantic_mapping") or {}
        excluded = set(mapping.get("query_model_service_ids", [])) if mapping.get("query_only") else set()
        for service in runtime.enabled_services:
            if service.instance_id in excluded:
                continue
            manifest = service.validated_manifest
            if not deployment_uses_ascend(manifest.deployment):
                continue
            role = rt_infer_role_for_manifest(manifest)
            if role is None:
                problems.append(f"perception service {service.instance_id!r}: {manifest.manifest.model.model_type}")
            else:
                roles.add(role)

    from robot_config.inference_config import parse_inference_config

    if context.with_inference:
        inference = parse_inference_config(
            robot_config,
            context.control_mode,
            scheduler_enabled=context.scheduler_enabled,
            runtime_target=context.runtime_target,
        )
        for pipeline in inference.pipelines.values():
            if deployment_uses_ascend(pipeline.validated_manifest.deployment):
                problems.append(
                    f"policy pipeline {pipeline.pipeline_id!r} (deployment {pipeline.deployment!r}) has no "
                    "RT-Infer role; the summit integration maps pear/yolox/tts/vl only"
                )

    tts = robot_config.get("voice_tts") or {}
    if tts.get("enabled", False):
        from inference_manifest import load_inference_manifest
        from robot_config.utils import resolve_ros_path

        bundle = Path(resolve_ros_path(tts["bundle_path"])).expanduser()
        if not bundle.is_absolute():
            workspace = os.environ.get("WORKSPACE", "")
            if not Path(workspace).is_absolute():
                raise RuntimeError("voice_tts.bundle_path requires an absolute WORKSPACE")
            bundle = Path(workspace) / bundle
        manifest = load_inference_manifest(bundle, tts["deployment"])
        if deployment_uses_ascend(manifest.deployment):
            profiles = list((manifest.deployment.role_runtime_profiles or {}).values())
            if manifest.deployment.runtime_profile is not None:
                profiles.append(manifest.deployment.runtime_profile)
            if any(profile.backend not in ("ascend", "onnx") for profile in profiles):
                problems.append("voice_tts uses an unsupported accelerator runtime")
            else:
                roles.add("tts")

    vl = robot_config.get("visual_interaction") or {}
    if vl.get("enabled", False) and (vl.get("vl") or {}).get("implementation") == "internvl":
        roles.add("vl")

    stream = robot_config.get("motion_imitation_stream") or {}
    if (
        "pear" in roles
        and config.pear_periodic
        and stream.get("enabled", False)
        and stream.get("max_pear_hz") != config.pear_periodic.period_hz
    ):
        problems.append("motion_imitation_stream.max_pear_hz must match rt_infer.pear_periodic.period_hz")

    if problems:
        raise RuntimeError(
            "rt_infer is enabled but this launch starts Ascend models without an RT-Infer role "
            "(a native process on a taken-over device is not supported). Refusing BEFORE loading "
            "the device module. Offenders: " + "; ".join(problems)
        )
    return roles


RT_INFER_SCHEDULER_BUILDER = BuilderSpec(
    id="rt_infer_scheduler",
    factory=f"{__name__}:build_rt_infer_scheduler",
    # The BUILTIN_BUILDERS entry places this before perception_models /
    # execution (which own the Ascend model processes), and the model
    # entries declare `after=["rt_infer_scheduler"]` so the registry orders
    # the wrapped model builds behind the owner when it participates.
    enabled=rt_infer_enabled,
)

__all__ = [
    "RT_INFER_SCHEDULER_BUILDER",
    "build_rt_infer_scheduler",
    "rt_infer_enabled",
]
