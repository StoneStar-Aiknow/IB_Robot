#!/usr/bin/env python3
"""Priority experiment lifecycle, using the existing deployment primitives."""

import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import base_control as base


def live_clients(state):
    return [item for item in state.get("clients", []) if base.same_process(item)]


@contextlib.contextmanager
def client_locked():
    # Concurrent clients wait for short registration/update transactions.
    # The lock is released before inference starts.
    with base.state_locked(blocking=True):
        yield


def start(root, manifest, args):
    current = base.state_read()
    restarting = bool(current)
    if restarting:
        base.require(current.get("status") == "stopped", "Stop the current scheduler cleanly first")
        base.require(current.get("prefix") == str(root), "Reboot before switching bundle versions")
        base.require(not live_clients(current), "Clients are still running")
        base.require(not base.same_process(current.get("scheduler")), "Scheduler is still running")
        base.require(current.get("module_owned") and base.module_loaded(), "Managed module is missing")
        base.require(
            current.get("module_sha256") == manifest["artifacts"]["module"]["sha256"], "Module identity changed"
        )
    else:
        base.require(not base.module_loaded() and not base.ipc_segment(), "Unmanaged soft scheduler state exists")
    if args.pfp_periodic:
        base.require(
            args.scheduler == "pfp" and args.pfp_exclusive == "on", "Periodic admission requires exclusive PFP"
        )
    base.preflight(root, manifest)
    base.require(not base.device_users(), "Device has compute users; inspect doctor")
    base.require(args.cpu in os.sched_getaffinity(0), "Requested CPU is unavailable")
    if restarting:
        base.cleanup_stopped_ipc(current)
    base.require(not base.IPC_PATH.is_symlink(), "IPC directory is a symlink")
    base.IPC_PATH.mkdir(mode=0o755, exist_ok=True)
    state = {
        "boot_id": base.boot_id(),
        "prefix": str(root),
        "version": manifest["version"],
        "status": "starting",
        "module_owned": current.get("module_owned", False),
        "module_sha256": manifest["artifacts"]["module"]["sha256"],
        "strategy": args.scheduler,
        "prefetch": args.prefetch,
        "cpu": args.cpu,
        "pfp_exclusive": args.pfp_exclusive,
        "local_pool": args.local_pool,
        "pfp_batch_select": args.pfp_batch_select,
        "pfp_min_cache": args.pfp_min_cache,
        "pfp_hi_priority": args.pfp_hi_priority,
        "native_recovery": "reboot-required",
        "clients": [],
    }
    base.state_write(state)
    try:
        if not restarting:
            base.command(["insmod", str(base.artifact_path(root, manifest, "module"))], env=base.clean_environment())
            state["module_owned"] = True
            base.state_write(state)
        executable = base.artifact_path(root, manifest, "scheduler")
        log = base.STATE / "scheduler.log"
        if log.exists():
            log.rename(base.STATE / f"scheduler-{time.time_ns()}.log")
        env = base.clean_environment()
        if args.trace:
            trace = args.trace.resolve()
            base.require(
                trace.parent.is_dir() and not trace.exists(), "Trace needs an existing parent and new filename"
            )
            env["RTINFER_KERNEL_TRACE"] = str(trace)
        with log.open("wb") as output:
            child = subprocess.Popen(
                [
                    "taskset",
                    "-c",
                    str(args.cpu),
                    str(executable),
                    "--scheduler",
                    args.scheduler,
                    "--prefetch",
                    args.prefetch,
                    "--aicore-count",
                    "10",
                    "--pfp-exclusive",
                    args.pfp_exclusive,
                    "--local-pool",
                    args.local_pool,
                    "--pfp-batch-select",
                    args.pfp_batch_select,
                    "--pfp-min-cache",
                    args.pfp_min_cache,
                    "--pfp-hi-priority",
                    str(args.pfp_hi_priority),
                ]
                + (["--dispatch-audit", str(args.dispatch_audit)] if args.dispatch_audit else [])
                + (["--dispatch-timing", str(args.dispatch_timing)] if args.dispatch_timing else [])
                + (["--pfp-periodic", str(args.pfp_periodic)] if args.pfp_periodic else []),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        until = time.monotonic() + 15
        while time.monotonic() < until:
            base.require(child.poll() is None, "Scheduler exited; inspect scheduler.log")
            identity = base.process_identity(child.pid)
            if identity and identity["executable"] == str(executable):
                state["scheduler"] = identity
                base.state_write(state)
            segment = base.ipc_segment()
            if "Daemon running..." in log.read_text(errors="replace") and segment:
                base.require(state.get("scheduler") and int(segment["cpid"]) == child.pid, "IPC creator mismatch")
                base.require(int(segment["nattch"]) == 1, "Unexpected client during startup")
                state.update(status="running", shmid=segment["shmid"], ipc_identity=base.shm_identity(segment))
                base.state_write(state)
                print(f"Ready: {args.scheduler}, prefetch {args.prefetch}, CPU {args.cpu}, PID {child.pid}")
                return
            time.sleep(0.1)
        raise base.DeploymentError("Scheduler startup timed out")
    except BaseException:
        state["status"] = "start-failed"
        base.state_write(state)
        raise


def run(root, manifest, args):
    argv = args.command
    if argv and argv[0] == "--":
        argv = argv[1:]
    base.require(argv, "Supply a foreground application after --")
    with client_locked():
        state = base.state_read()
        base.require(state.get("prefix") == str(root) and state.get("status") == "running", "Scheduler is not ready")
        base.require(base.same_process(state.get("scheduler")), "Scheduler has exited")
        segment = base.ipc_segment()
        base.require(segment and segment["shmid"] == state["shmid"], "IPC identity mismatch")
        base.require(str(segment["cpid"]) == str(state["scheduler"]["pid"]), "IPC creator changed")
        env = base.clean_environment()
        runtime = str(base.artifact_path(root, manifest, "runtime"))
        env.update(
            LD_PRELOAD=runtime,
            RT_INFER_LIBRARY=runtime,
            TASK_QUEUE_ENABLE="0",
            AICORE_SCHED_ALLOCATOR="official",
            AICORE_SCHED_OFFICIAL_RUNTIME=manifest["compatibility"]["official_runtime_path"],
            RT_INFER_TASKGEN_MEMCPY="1",
            RT_INFER_ARGS_INDEX="1",
            RT_INFER_QUIET_RUNTIME="1",
            RTINFER_STREAM_PRIORITY=str(args.priority),
        )
        # Official pyACL static initialization needs libprofapi loaded before import acl.
        profiler = Path(manifest["compatibility"]["official_runtime_path"]).parent / "libprofapi.so"
        base.require(profiler.is_file(), "Official libprofapi.so is missing")
        env["LD_PRELOAD"] = f"{runtime}:{profiler}"
        if args.cpu is not None:
            base.require(args.cpu in os.sched_getaffinity(0), "Requested client CPU is unavailable")
            argv = ["taskset", "-c", str(args.cpu), *argv]
        child = subprocess.Popen(argv, env=env)
        identity = base.process_identity(child.pid)
        base.require(identity, "Application exited before registration")
        identity.pop("executable")
        state["clients"] = [*live_clients(state), identity]
        base.state_write(state)
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # Foreground processes share the terminal process group and receive SIGINT.
        print("Interrupted; inspect clients before stopping the scheduler", file=sys.stderr)
        return 130
    with client_locked():
        state = base.state_read()
        state["clients"] = [item for item in live_clients(state) if item != identity]
        state["last_application_exit"] = code
        if code:
            state["status"] = "recovery-required"
        base.state_write(state)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path)
    actions = parser.add_subparsers(dest="action", required=True)
    verify = actions.add_parser("verify-bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    install = actions.add_parser("install")
    install.add_argument("--bundle", type=Path, required=True)
    install.add_argument("--prefix", type=Path, required=True)
    for name in ("doctor", "status", "stop", "verify-native"):
        actions.add_parser(name)
    starter = actions.add_parser("start")
    starter.add_argument("--scheduler", choices=("fifo", "nice", "pfp"), required=True)
    starter.add_argument("--prefetch", choices=("on", "off"), default="off")
    starter.add_argument("--cpu", type=int, required=True)
    starter.add_argument("--trace", type=Path)
    starter.add_argument("--pfp-min-cache", choices=("on", "off"), default="off")
    starter.add_argument("--pfp-hi-priority", type=int, choices=range(8), default=1)
    starter.add_argument("--pfp-periodic", type=Path)
    starter.add_argument("--pfp-batch-select", choices=("on", "off"), default="off")
    starter.add_argument("--dispatch-timing", type=Path)
    starter.add_argument("--dispatch-audit", type=Path)
    starter.add_argument("--local-pool", choices=("on", "off"), default="off")
    starter.add_argument("--pfp-exclusive", choices=("on", "off"), default="off")
    runner = actions.add_parser("run")
    runner.add_argument("--priority", type=int, choices=range(8), required=True)
    runner.add_argument("--cpu", type=int)
    runner.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "verify-bundle":
            manifest = base.verify_bundle(args.bundle)
            print(f"Verified: {manifest['version']}")
            return 0
        if args.action == "install":
            base.install(args)
            (args.prefix / "bin/om_probe").chmod(0o755)
            return 0
        base.require(args.prefix, "Specify --prefix before the command")
        base.root_linux()
        root = args.prefix.resolve()
        base.trusted_tree(root)
        manifest = base.verify_bundle(root)
        if args.action == "run":
            return run(root, manifest, args)
        with base.locked():
            if args.action in ("doctor", "status"):
                base.report(root, manifest, doctor=args.action == "doctor")
            elif args.action == "start":
                start(root, manifest, args)
            elif args.action == "stop":
                state = base.state_read()
                base.require(state.get("prefix") == str(root), "Different managed prefix")
                base.require(not live_clients(state), "Applications are still running")
                base.stop()
            elif args.action == "verify-native":
                base.native_check()
        return 0
    except (base.DeploymentError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"priority-eval: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
