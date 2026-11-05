"""Launch one RT-Infer-managed Ascend inference role as a vendor client process.

This is the client half of the RT-Infer launch integration. The launch-side
owner (``robot_config.launch_builders.rt_infer_scheduler``) starts the device
scheduler before any model process; this wrapper then attaches one *whole
process* to it through the vendor's guarded ``priorityctl run`` entry, which
performs every client-side check (scheduler ready, IPC identity, live client
bookkeeping) and sets the documented per-process environment (LD_PRELOAD of
the managed libruntime plus libprofapi, RTINFER_STREAM_PRIORITY, RT_INFER_*
knobs). The wrapper adds nothing of its own on top.

Why a wrapper executable instead of a plain ``Node``: ``priorityctl run`` must
be the *parent* of the model process (it spawns the client and tracks its
identity in the vendor state), so the real node main has to run under it
rather than beside it. The wrapper stays a thin main so launch keeps using
``launch_ros.actions.Node`` with ``executable=rt_infer_client``.

Usage (installed console script, launched by the model-entry builders)::

    rt_infer_client --role pear \
        --rt-infer-config /path/to/snapshot.json \
        --node-main inference_service.model_service_node:main \
        -- [--ros-args ...node arguments...]

The role resolves priority/CPU from the launch-prepared JSON snapshot (the
wrapper never re-parses robot YAML). ``--list-args`` prints the exact
``priorityctl run`` argument vector without executing anything.

Node-main execution contract (review #1): ``module:function`` is resolved
with importlib *inside the child*, never by pasting Python source into a
``-c`` string — an annotation like ``module:main()`` there is silent dead
code and the node main never runs. The child rewrites ``sys.argv`` so ROS
argument parsing sees the real node name and remappings, calls the imported
function, and its exit code is what the vendor client bookkeeping records.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_config.rt_infer_config import RtInferConfig

_CLIENT_ROLES = ("pear", "yolox", "tts", "vl")


class RtInferClientError(RuntimeError):
    """The wrapper cannot attach this process to the RT-Infer scheduler."""


def _config_from_snapshot(snapshot: dict) -> RtInferConfig:
    """Rebuild the typed role config from the launch-prepared JSON snapshot.

    The snapshot is produced by the launch builders from the parsed
    ``robot.rt_infer`` block; the wrapper deliberately does not re-read robot
    YAML (the effective config may include live-bound interface data that is
    not re-derivable after launch preparation).
    """
    from robot_config.rt_infer_config import PearPeriodicConfig, RtInferConfig

    periodic = None
    if snapshot.get("pear_periodic"):
        periodic = PearPeriodicConfig(
            period_hz=int(snapshot["pear_periodic"]["period_hz"]),
            budget_ns=int(snapshot["pear_periodic"]["budget_ns"]),
            control_path=str(snapshot["pear_periodic"]["control_path"]),
        )
    return RtInferConfig(
        enable=True,
        bundle_version=str(snapshot["bundle_version"]),
        bundle_dir=str(snapshot.get("bundle_dir") or ""),
        role_priorities=dict(snapshot["role_priorities"]),
        role_cpus=dict(snapshot["role_cpus"]),
        pear_periodic=periodic,
        launch_token=str(snapshot.get("launch_token") or ""),
    )


def build_client_argv(config: RtInferConfig, role: str, command: list[str]) -> list[str]:
    """The exact priorityctl argument vector for one role's client process.

    The scheduler start passes ``--pfp-periodic`` when the typed config has a
    PEAR periodic profile, so the PEAR client must publish releases on the
    same mapping: the periodic environment is part of the *client* contract,
    and the node-side adapter (``inference_service.rt_infer_periodic``, armed
    by ``RT_INFER_PERIODIC_PATH``) publishes a real release per PEAR execute
    through the vendor bridge. Background roles carry no periodic
    environment.
    """
    priority = config.priority_for(role)
    cpu = config.cpu_for(role)
    argv = [
        sys.executable,
        config.control_tool,
        "--prefix",
        config.prefix,
        "run",
        "--priority",
        str(priority),
        "--cpu",
        str(cpu),
        "--",
    ]
    environment = {"RTINFER_PREFIX": config.prefix, "RTINFER_ROLE": role}
    from robot_config.rt_infer_bundle import expected_toolkit_home

    toolkit_home = expected_toolkit_home(Path(config.prefix))
    if toolkit_home is not None:
        environment["ASCEND_TOOLKIT_HOME"] = toolkit_home
    if getattr(config, "launch_token", ""):
        environment["RTINFER_LAUNCH_TOKEN"] = config.launch_token
    if role == "pear" and config.pear_periodic is not None:
        environment["RT_INFER_PERIODIC_PATH"] = config.pear_periodic.control_path
        environment["RT_INFER_PERIODIC_SESSION"] = str(
            Path(config.pear_periodic.control_path).parent / "periodic-session.json"
        )
        environment["RT_INFER_PERIODIC_PERIOD_HZ"] = str(config.pear_periodic.period_hz)
        environment["RT_INFER_PERIODIC_BUDGET_NS"] = str(config.pear_periodic.budget_ns)
        environment["RT_INFER_SAFE_ARGS_BATCH"] = "1"
    argv += ["/usr/bin/env", *(f"{key}={value}" for key, value in sorted(environment.items()))]
    return [*argv, *command]


def _write_client_token() -> None:
    """Record this client's ownership token where the launch owner can verify it.

    The vendor strips inherited RT_INFER_*/RTINFER_* variables from the
    client environment (clean_environment), so the launch token arrives via
    the explicit ``/usr/bin/env`` argv this wrapper builds. The node process
    re-exports it here as a file keyed by ITS pid: shutdown cleanup treats a
    client as "ours" only when the token matches and the pidfd-verified
    process identity agrees, and refuses to signal anything else.
    """
    import json
    import os
    from pathlib import Path

    token = os.environ.get("RTINFER_LAUNCH_TOKEN")
    if not token:
        raise RtInferClientError("The client requires an owner-generated launch token")
    state_dir = Path(os.environ.get("RTINFER_STATE_DIR", "/run/ibrobot-rtinfer"))
    try:
        fields = (Path("/proc") / str(os.getpid()) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        parent_pid = os.getppid()
        parent_ticks = (Path("/proc") / str(parent_pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        state_dir.mkdir(parents=True, exist_ok=True)
        temporary = state_dir / f"client-{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "token": token,
                    "pid": os.getpid(),
                    "start_ticks": fields[19],
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "role": os.environ.get("RTINFER_ROLE", ""),
                    "wrapper": {
                        "pid": parent_pid,
                        "start_ticks": parent_ticks,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    },
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(state_dir / f"client-{os.getpid()}.token")
    except OSError as exc:
        raise RtInferClientError(f"Cannot register managed client identity: {exc}") from exc


def _run_node_main() -> None:
    """Child entry: import the real node main and run it.

    Runs inside the process priorityctl spawned (this module executed with
    ``python -m``). ``sys.argv`` is rewritten so the node sees its own
    arguments (ROS remappings, launch parameters) exactly as a direct launch
    would pass them; the imported entry is called with no arguments, and any
    exception propagates as the process exit status the vendor records.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)
    _write_client_token()
    # The owner removes the token only after this PID has exited. Removing
    # it in a finally block creates a live-but-unverifiable shutdown race.
    try:
        _invoke_node_main()
    finally:
        # Once the real main has returned, an interrupt during interpreter
        # finalization must not turn its successful result into signal -2.
        # Errors still propagate unchanged; this never converts an exit code.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        path = Path(os.environ.get("RTINFER_STATE_DIR", "/run/ibrobot-rtinfer")) / f"client-{os.getpid()}.token"
        record = json.loads(path.read_text())
        record.update(closing=True, main_returned_ns=time.monotonic_ns())
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)
        print(
            "RT_INFER_CLIENT_EVENT "
            + json.dumps({"event": "main_returned", "pid": os.getpid(), "monotonic_ns": record["main_returned_ns"]}),
            flush=True,
        )


def mark_runtime_ready() -> None:
    """Report completed real model warmup to this launch's owner."""
    token = os.environ.get("RTINFER_LAUNCH_TOKEN")
    if not token:
        return
    path = Path(os.environ.get("RTINFER_STATE_DIR", "/run/ibrobot-rtinfer")) / f"client-{os.getpid()}.token"
    record = json.loads(path.read_text(encoding="utf-8"))
    ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (
        record.get("token") != token
        or record.get("pid") != os.getpid()
        or str(record.get("start_ticks")) != ticks
        or record.get("boot_id") != boot
    ):
        raise RtInferClientError("Cannot mark readiness for a different client identity")
    if record.get("ready") is True:
        return
    import time

    record.update(ready=True, ready_monotonic_ns=time.monotonic_ns())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _invoke_node_main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rt-infer-node-main", required=True)
    parser.add_argument("node_arguments", nargs=argparse.REMAINDER)
    known, _unused = parser.parse_known_args(sys.argv[1:])

    module_name, separator, attribute = known.rt_infer_node_main.partition(":")
    if not separator or not module_name or not attribute:
        raise RtInferClientError(f"--rt-infer-node-main must be 'module:function', got {known.rt_infer_node_main!r}")
    import importlib

    module = importlib.import_module(module_name)
    entry = getattr(module, attribute, None)
    if not callable(entry):
        raise RtInferClientError(f"{module_name}:{attribute} did not resolve to a callable node main")

    node_args = list(known.node_arguments)
    if node_args and node_args[0] == "--":
        node_args = node_args[1:]
    sys.argv = [attribute] + node_args
    result = entry()
    if isinstance(result, int):
        sys.exit(result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=_CLIENT_ROLES)
    parser.add_argument(
        "--rt-infer-config",
        required=True,
        help="Path to the launch-prepared rt_infer config JSON snapshot",
    )
    parser.add_argument("--node-main", required=True, help="Python entry 'module:function' of the real node")
    parser.add_argument("--list-args", action="store_true", help="Print the priorityctl argv without executing")
    parser.add_argument("node_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    node_args = list(args.node_arguments)
    if node_args and node_args[0] == "--":
        node_args = node_args[1:]

    try:
        snapshot = json.loads(Path(args.rt_infer_config).read_text(encoding="utf-8"))
        config = _config_from_snapshot(snapshot)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RtInferClientError(f"rt-infer config snapshot is unusable: {exc}") from exc

    # The child is this same module's _run_node_main executed with ``python
    # -m`` (imported and parsed by CPython, never pasted into a -c string):
    # it resolves module:function with importlib, rewrites sys.argv to the
    # node's own arguments and calls the real main.
    command = [
        sys.executable,
        "-m",
        "inference_service.rt_infer_client",
        "--rt-infer-node-main",
        args.node_main,
        "--",
        *node_args,
    ]
    argv_out = build_client_argv(config, args.role, command)
    if args.list_args:
        print(json.dumps(argv_out))
        return 0
    # The controller must wait for the actual child and finish unregistering.
    # The owner signals the token-bound child directly. The child restores its
    # normal Python/ROS handler above; terminal process-group SIGINT still works.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The wrapper execs the vendor client tool as its replacement: the node
    # process the launch supervises is the priorityctl child, so its exit and
    # the vendor client bookkeeping stay consistent on every shutdown path.
    os.execvp(argv_out[0], argv_out)
    return 0  # unreachable; execvp either succeeds or raises


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rt-infer-node-main":
        _run_node_main()
    else:
        sys.exit(main())
