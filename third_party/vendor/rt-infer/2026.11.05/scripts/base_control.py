#!/usr/bin/env python3
"""Deploy optional, version-pinned RT-Infer binaries without replacing CANN.

Linux root-only lifecycle for the 310P RC bundle. State is global
because the supplied runtime uses a fixed ftok path and owns physical AI cores.
Stopping the daemon does NOT restore native device state: a reboot is required.
The deployment commands use only the Python standard library.
"""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

STATE = Path("/run/ibrobot-rtinfer")
IPC_PATH = Path("/var/run/aicore")
ROLES = {"scheduler", "runtime", "module"}


class DeploymentError(Exception):
    """A failed precondition; do not modify the device to work around it."""


def require(condition, message):
    if not condition:
        raise DeploymentError(message)


def compatible(field, observed, expected):
    require(
        observed == expected,
        f"Incompatible {field}: expected {expected!r}, found {observed!r}. "
        "Use a matching RT-Infer bundle/environment; IB-Robot dependencies will not be changed.",
    )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def member(root, name):
    relative = PurePosixPath(name)
    require(name and "\\" not in name and not relative.is_absolute(), f"Invalid bundle path: {name}")
    require(all(part not in ("..", ".") for part in name.split("/")), f"Invalid bundle path: {name}")
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes bundle: {name}")
    for parent in (path, *path.parents):
        if parent == root:
            break
        require(not parent.is_symlink(), f"Symlink not permitted: {parent}")
    require(path.is_file(), f"Missing bundle file: {name}")
    return path


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = load_json(member(root, "manifest.json"))
    require(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", manifest.get("version", "")), "Invalid version")
    require(set(manifest.get("artifacts", {})) == ROLES, "Expected scheduler, runtime and module")
    paths = set()
    for artifact in manifest["artifacts"].values():
        path = member(root, artifact["path"])
        require(artifact["path"] not in paths, "Duplicate artifact path")
        paths.add(artifact["path"])
        require(re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]), "Invalid SHA256")
        require(digest(path) == artifact["sha256"], f"Checksum mismatch: {path.name}")
        with path.open("rb") as stream:
            header = stream.read(20)
        require(header[:6] == b"\x7fELF\x02\x01" and header[18:20] == b"\xb7\x00", f"Not aarch64 ELF: {path}")
    for item in manifest.get("payload_files", []):
        require(item["path"] not in paths, "Duplicate payload path")
        path = member(root, item["path"])
        require(digest(path) == item["sha256"], f"Checksum mismatch: {item['path']}")
        paths.add(item["path"])
    payload = {item["path"]: item["sha256"] for item in manifest.get("payload_files", [])}
    for item in manifest.get("graph_files", []):
        require(item["path"].startswith("graphs/"), "Invalid precompiled graph path")
        require(payload.get(item["path"]) == item["sha256"], "Graph identity is not pinned in the payload")
    require(manifest.get("native_recovery") == "reboot-required", "Unvalidated native recovery policy")
    require(manifest.get("compatibility", {}).get("architecture") == "aarch64", "Unsupported architecture")
    for name in manifest.get("metadata_files", []):
        require(name not in paths and name != "manifest.json", "Duplicate metadata/artifact path")
        member(root, name)
        paths.add(name)
    return manifest


def artifact_path(root, manifest, role):
    return member(root, manifest["artifacts"][role]["path"])


def trusted_tree(root):
    """Root executes and preloads these files: reject writable or foreign-owned inputs."""
    for path in (root, *root.parents, *root.rglob("*")):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"Symlink in installed tree: {path}")
        require(info.st_uid == 0 and not info.st_mode & 0o022, f"Root ownership and non-writable path required: {path}")


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def root_linux():
    require(platform.system() == "Linux", "Lifecycle commands require Linux")
    require(os.geteuid() == 0, "This release requires root, including run/stop")


@contextlib.contextmanager
def locked():
    import fcntl

    root_linux()
    require(not STATE.is_symlink(), "State directory must not be a symlink")
    STATE.mkdir(mode=0o700, exist_ok=True)
    trusted_tree(STATE)
    with (STATE / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def state_read():
    path = STATE / "state.json"
    if not path.exists():
        return {}
    state = load_json(path)
    require(state.get("boot_id") == boot_id(), "Stale state directory: use the host /run, not a persistent mount")
    return state


def state_write(state):
    save_json(STATE / "state.json", state)


def process_identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19], "executable": str((proc / "exe").resolve(strict=True))}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    if not identity:
        return False
    actual = process_identity(identity["pid"])
    return bool(actual) and all(actual.get(key) == value for key, value in identity.items())


def module_loaded():
    return any(line.split()[0] == "aicore_device" for line in Path("/proc/modules").read_text().splitlines())


def shm_rows():
    lines = Path("/proc/sysvipc/shm").read_text().splitlines()
    return [dict(zip(lines[0].split(), line.split(), strict=True)) for line in lines[1:]]


def ipc_segment():
    if not IPC_PATH.exists():
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ftok.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.ftok.restype = ctypes.c_int
    key = libc.ftok(os.fsencode(IPC_PATH), ord("X"))
    require(key != -1, "Cannot determine RT-Infer shared-memory key")
    return next((row for row in shm_rows() if int(row["key"]) == key), None)


def compute_device(target):
    # Management-only descriptors are held by the stock logging/health services.
    # They do not imply an inference context and must not require stopping CANN services.
    return bool(re.fullmatch(r"/dev/davinci\d+", target)) or target.startswith(
        ("/dev/aicore", "/dev/devmm", "/dev/hisi_hdc")
    )


def device_users(exclude=(), management_only=False):
    """Separate compute-device holders from stock manager-only services."""
    users = set()
    for proc in Path("/proc").glob("[0-9]*"):
        pid = int(proc.name)
        if pid in {*exclude, os.getpid()}:
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            if comm.startswith("aicore_sched") and not management_only:
                users.add(pid)
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                matches = target == "/dev/davinci_manager" if management_only else compute_device(target)
                if matches:
                    users.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise DeploymentError(f"Cannot inspect device users: {proc}") from error
    return sorted(users)


def clean_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"LD_PRELOAD", "RT_INFER_LIBRARY", "IBROBOT_PI05_TORCHAIR_CACHE_HOME"}
        and not key.startswith(("RT_INFER_", "RTINFER_", "AICORE_SCHED_"))
    }


def preflight(root, manifest):
    compatibility = manifest["compatibility"]
    for tool in ("taskset", "modinfo", "insmod", "ipcrm", "ldd"):
        require(shutil.which(tool), f"Missing required system tool: {tool}")
    compatible("CPU architecture", platform.machine(), compatibility["architecture"])
    compatible("kernel release", platform.release(), compatibility["kernel_release"])
    require(Path("/dev/davinci0").exists(), "Ascend device /dev/davinci0 missing")
    info = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    compatible("OS ID", info.get("ID", "").strip('"'), compatibility["os_id"])
    compatible("OS image BUILD_ID", info.get("BUILD_ID", "").strip('"'), compatibility["os_build_id"])
    official = Path(compatibility["official_runtime_path"])
    compatible(
        "official CANN runtime SHA256",
        digest(official) if official.is_file() else "missing",
        compatibility["official_runtime_sha256"],
    )
    compatible(
        "ASCEND_TOOLKIT_HOME",
        str(Path(os.environ.get("ASCEND_TOOLKIT_HOME", "/missing")).resolve()),
        str(official.parents[2]),
    )
    module = artifact_path(root, manifest, "module")
    compatible("module vermagic", command(["modinfo", "-F", "vermagic", str(module)]), compatibility["module_vermagic"])
    for role in ("scheduler", "runtime"):
        output = command(["ldd", str(artifact_path(root, manifest, role))], env=clean_environment())
        require("not found" not in output, f"Missing dependencies for {role}: {output}")
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Python/kernel pidfd support required")


def install(args):
    root_linux()
    source = args.bundle.resolve()
    manifest = verify_bundle(source)
    # Validate this optional bundle before any installation/device mutation.
    # Do not install dependencies or rewrite IB-Robot's global setup requirements.
    preflight(source, manifest)
    target = args.prefix.absolute()
    require(target.parent.is_dir(), "Installation parent must already exist")
    require(not target.is_symlink(), "Installation prefix must not be a symlink")
    with locked():
        require(not same_process(state_read().get("scheduler")), "Stop the managed scheduler before installation")
        if target.exists():
            require(
                verify_bundle(target) == manifest,
                "Version directory exists with different content; choose a new prefix",
            )
            trusted_tree(target)
            print(f"Already installed and verified: {target}")
            return
        stage = Path(tempfile.mkdtemp(prefix=".rtinfer-", dir=target.parent))
        try:
            # Copy only the manifest and explicitly listed artifacts/metadata, never an arbitrary source tree.
            names = [
                "manifest.json",
                *[item["path"] for item in manifest["artifacts"].values()],
                *[item["path"] for item in manifest.get("payload_files", [])],
                *manifest.get("metadata_files", []),
            ]
            for name in names:
                source_path = member(source, name)
                output = stage / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, output)
                output.chmod(0o755 if name == manifest["artifacts"]["scheduler"]["path"] else 0o644)
            verify_bundle(stage)
            trusted_tree(stage)
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    print(f"Installed {manifest['version']} at {target}; this command did not start a scheduler")


def start(root, manifest, args):
    require(
        args.experimental,
        "Use --exclusive-device to acknowledge exclusive device ownership and reboot-required recovery",
    )
    current = state_read()
    restarting = bool(current)
    if restarting:
        require(current.get("status") == "stopped", "Scheduler must be stopped cleanly before another start")
        require(
            current.get("prefix") == str(root) and current.get("version") == manifest["version"],
            "Restart requires the same installed bundle; reboot before changing versions",
        )
        require(
            not same_process(current.get("scheduler")) and not same_process(current.get("client")),
            "Previous scheduler or client is still running",
        )
    preflight(root, manifest)
    if restarting:
        require(module_loaded() and current.get("module_owned"), "Managed module is missing; do not reload in place")
        require(current.get("module_sha256") == manifest["artifacts"]["module"]["sha256"], "Module version changed")
    else:
        require(not module_loaded(), "Unmanaged aicore_device module is loaded; refusing takeover")
        require(not ipc_segment(), "Existing RT-Infer shared memory; refusing takeover")
    require(not device_users(), "Ascend device is in use; stop applications and inspect doctor output")
    require(args.cpu in os.sched_getaffinity(0), "Scheduler CPU is outside the available CPU affinity")
    if restarting:
        cleanup_stopped_ipc(current)
    require(not IPC_PATH.is_symlink() and (not IPC_PATH.exists() or IPC_PATH.is_dir()), "Invalid IPC path")
    IPC_PATH.mkdir(mode=0o755, exist_ok=True)
    state = {
        "boot_id": boot_id(),
        "prefix": str(root),
        "version": manifest["version"],
        "status": "starting",
        "native_recovery": "reboot-required",
        "module_owned": current.get("module_owned", False),
        "module_sha256": manifest["artifacts"]["module"]["sha256"],
    }
    # Persist before insmod: failed/partial initialization may already alter hardware state.
    state_write(state)
    try:
        if not restarting:
            command(["insmod", str(artifact_path(root, manifest, "module"))], env=clean_environment())
            state["module_owned"] = True
            state_write(state)
        executable = artifact_path(root, manifest, "scheduler")
        if (STATE / "scheduler.log").exists():
            (STATE / "scheduler.log").rename(STATE / f"scheduler-{time.time_ns()}.log")
        with (STATE / "scheduler.log").open("wb") as log:
            child = subprocess.Popen(
                ["taskset", "-c", str(args.cpu), str(executable), "--scheduler", "fifo", "--prefetch", "on"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=clean_environment(),
                start_new_session=True,
            )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            require(child.poll() is None, "Scheduler exited; see /run/ibrobot-rtinfer/scheduler.log")
            identity = process_identity(child.pid)
            if identity and identity["executable"] == str(executable):
                state["scheduler"] = identity
                state_write(state)
            segment = ipc_segment()
            if "Daemon running..." in (STATE / "scheduler.log").read_text(errors="replace") and segment:
                require(state.get("scheduler") and int(segment["cpid"]) == child.pid, "Scheduler/IPC identity mismatch")
                require(int(segment["nattch"]) == 1, "Unexpected client attached during startup")
                state.update(status="running", shmid=segment["shmid"], ipc_identity=shm_identity(segment))
                state_write(state)
                print(f"Running PID {child.pid}, FIFO, prefetch on, CPU {args.cpu}; reboot before native use")
                return
            time.sleep(0.1)
        raise DeploymentError("Scheduler readiness timed out; inspect status/logs, then stop; no forced kill")
    except BaseException:
        state["status"] = "start-failed"
        state_write(state)
        raise


def shm_identity(segment):
    return {key: segment[key] for key in ("key", "shmid", "cpid", "ctime", "uid", "size")}


def cleanup_stopped_ipc(state):
    """Remove only our detached segment; never infer hardware recovery from IPC removal."""
    segment = ipc_segment()
    if segment is None:
        return
    require(state.get("ipc_identity") == shm_identity(segment), "Shared-memory identity changed; refusing cleanup")
    require(int(segment["nattch"]) == 0, "Shared memory still has attached clients")
    command(["ipcrm", "-m", segment["shmid"]])
    require(ipc_segment() is None, "Shared memory still exists after cleanup")


def stop():
    state = state_read()
    require(state, "No managed scheduler state; refusing to stop unrelated processes")
    identity = state.get("scheduler")
    require(not same_process(state.get("client")), "Managed application is still running")
    if state.get("status") == "stopped":
        require(not same_process(identity), "Stopped state disagrees with live scheduler")
        print("Scheduler already stopped; soft restart allowed, reboot before native use")
        return
    if not same_process(identity):
        state["status"] = "recovery-required"
        state_write(state)
        print("Scheduler exited unexpectedly; inspect the failure before recovery")
        return
    segment = ipc_segment()
    if segment:
        require(int(segment["cpid"]) == identity["pid"], "Shared-memory creator changed; refusing stop")
        require(not state.get("shmid") or segment["shmid"] == state["shmid"], "Shared-memory identity changed")
        require(int(segment["nattch"]) == 1, "Other clients are attached; refusing stop")
    else:
        require(state["status"] == "start-failed", "Shared memory disappeared; refusing stop")
    require(not device_users(exclude=(identity["pid"],)), "Other device users exist; refusing stop")
    # pidfd binds the signal to this process even if its numeric PID is subsequently reused.
    with contextlib.closing(os.fdopen(os.pidfd_open(identity["pid"]), "rb")) as handle:
        require(same_process(identity), "Scheduler identity changed; refusing signal")
        signal.pidfd_send_signal(handle.fileno(), signal.SIGTERM)
        deadline = time.monotonic() + 15
        while same_process(identity) and time.monotonic() < deadline:
            time.sleep(0.1)
    require(not same_process(identity), "Scheduler did not exit; not forcing termination or unloading driver")
    # A clean stop of a healthy daemon permits soft restart, not native recovery.
    state["status"] = "stopped" if state["status"] == "running" else "recovery-required"
    state_write(state)
    print(f"Scheduler stopped ({state['status']}). Driver retained; reboot before native use.")


def check_pi05_policy(policy):
    """Validate the external model bundle's declared files and measured fast preset."""
    policy = policy.resolve()
    model_manifest = load_json(policy / "inference_manifest.json")
    require(model_manifest.get("model", {}).get("model_type") == "pi05", "Expected a PI0.5 model bundle")
    fast = model_manifest.get("deployments", {}).get("torch-npu-fast", {})
    settings = fast.get("extra_parameters", {})
    expected = {
        "denoising": {"solver": "ab2", "schedule": "uniform", "steps": 3},
        "language_capacity": 48,
        "vision_ffn": True,
        "image_preprocessing": "layout_aware",
        "action_output_device": "cpu",
    }
    require(
        all(settings.get(key) == value for key, value in expected.items()),
        "PI0.5 fast preset differs from validated configuration",
    )
    names = {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"}
    names.update(entry["path"] for entry in model_manifest.get("bundle", {}).get("files", []))
    for name in names:
        relative = PurePosixPath(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "Invalid model asset path"
        )
        require((policy / name).is_file(), f"Missing model bundle asset: {name}")
    return {
        "deployment": "torch-npu-fast",
        "declared_files_checked": len(names),
        "validation": "presence and preset only; weight identity and numerical accuracy not checked",
    }


def report(root, manifest, doctor=False, policy=None, check_model_env=False):
    state = state_read()
    result = {
        "version": manifest["version"],
        "prefix": str(root),
        "state": state or {"status": "not-started-this-boot"},
        "module_loaded": module_loaded(),
        "ipc": ipc_segment(),
        "device_users": device_users(),
        "management_device_users": device_users(management_only=True),
    }
    if doctor:
        preflight(root, manifest)
        result["compatibility"] = "passed"
        if check_model_env or policy:
            model_environment(manifest)
            result["model_environment"] = "passed"
        if policy:
            result["pi05_policy"] = check_pi05_policy(policy)
    print(json.dumps(result, indent=2))


def model_environment(manifest):
    """Check package metadata and compiler binaries without initializing the NPU."""
    compatibility = manifest["compatibility"]
    compatible("Python", f"{sys.version_info.major}.{sys.version_info.minor}", compatibility["python"])
    for package in ("torch", "torch_npu", "transformers"):
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = "not installed"
        compatible(package, observed, compatibility[package])
    distribution = importlib.metadata.distribution("torch_npu")
    for relative, expected in compatibility.get("torchair_core_files", {}).items():
        path = Path(distribution.locate_file(relative))
        require(path.is_file() and digest(path) == expected, f"TorchAir core library missing or different: {relative}")


def prepare_application_cache(root, manifest, cache, profile):
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "rt-infer-cache.json"
    fingerprint = {
        "version": manifest["version"],
        "runtime": manifest["artifacts"]["runtime"]["sha256"],
        "profile": profile,
    }
    if marker.exists():
        require(load_json(marker) == fingerprint, "Cache belongs to a different runtime or application profile")
    else:
        require(not any(cache.iterdir()), "Soft cache must be new/empty; do not reuse native graph caches")
        if profile == "pi05-765":
            graphs = manifest.get("graph_files", [])
            require(graphs, "PI0.5 binary deployment requires its pinned precompiled graph assets")
            for item in graphs:
                source = member(root, item["path"])
                require(item["path"].startswith("graphs/"), "Invalid precompiled graph path")
                target = cache / item["path"][len("graphs/") :]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o600)
        save_json(marker, fingerprint)
    if profile == "pi05-765":
        require(manifest.get("graph_files"), "Missing precompiled graph identity")
        for item in manifest["graph_files"]:
            path = member(cache, item["path"][len("graphs/") :])
            require(digest(path) == item["sha256"], f"Precompiled graph changed: {path.name}")


def run_application(root, manifest, args):
    argv = args.command
    if argv and argv[0] == "--":
        argv = argv[1:]
    require(argv, "Supply an application command after --")
    model_environment(manifest)
    with locked():
        state = state_read()
        preflight(root, manifest)
        require(
            state.get("prefix") == str(root) and state.get("status") == "running", "This bundle has no ready scheduler"
        )
        require(same_process(state.get("scheduler")), "Scheduler is no longer running")
        require(not same_process(state.get("client")), "Only one managed application is supported")
        segment = ipc_segment()
        require(
            segment and segment["shmid"] == state["shmid"] and int(segment["nattch"]) == 1,
            "Unexpected shared-memory clients",
        )
        require(not device_users(exclude=(state["scheduler"]["pid"],)), "Other device users are active")
        environment = clean_environment()
        runtime = str(artifact_path(root, manifest, "runtime"))
        cache = args.cache.resolve()
        prepare_application_cache(root, manifest, cache, args.profile)
        environment.update(
            LD_PRELOAD=runtime,
            RT_INFER_LIBRARY=runtime,
            TASK_QUEUE_ENABLE="0",
            AICORE_SCHED_ALLOCATOR="official",
            AICORE_SCHED_OFFICIAL_RUNTIME=manifest["compatibility"]["official_runtime_path"],
            RT_INFER_TASKGEN_MEMCPY="1",
            RT_INFER_ARGS_INDEX="1",
            RT_INFER_QUIET_RUNTIME="1",
            IBROBOT_PI05_TORCHAIR_CACHE_HOME=str(cache),
        )
        if args.profile == "pi05-765":
            require(
                manifest.get("application_profile") == "pi05-765-v1", "Bundle does not support PI0.5 765-task profile"
            )
            environment.update(
                RT_INFER_RECORD_API_TYPE="1",
                RT_INFER_FIXED_VISION_ARGS="1",
                RT_INFER_FROZEN_VISION="1",
                RT_INFER_FROZEN_PREFILL="1",
                RT_INFER_MANUAL_RMS="1",
                RT_INFER_DENOISE_TASK_COUNT="765",
            )
            cann_cache = cache / "cann"
            cann_cache.mkdir(exist_ok=True)
            environment["ASCEND_CACHE_PATH"] = str(cann_cache)
            vendors = ("attention", "mlp_pair", "linear_nz", "qkv", "gated", "rms")
            environment["ASCEND_CUSTOM_OPP_PATH"] = ":".join(str(root / "operators" / f"rtinfer_{v}") for v in vendors)
            profiler = Path(manifest["compatibility"]["official_runtime_path"]).parents[2] / "lib64/libprofapi.so"
            require(profiler.is_file(), "Missing official libprofapi.so")
            environment["LD_PRELOAD"] = f"{runtime}:{profiler}"
        child = subprocess.Popen(argv, env=environment)
        identity = process_identity(child.pid)
        if identity:
            # exec() is allowed in application wrappers; the process start time remains stable.
            identity.pop("executable")
            state["client"] = identity
        state_write(state)
    # Do not hold the control lock across inference; stop uses both PID and IPC checks.
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # Foreground child receives the same terminal SIGINT; never stop the scheduler here.
        print("Interrupted; inspect status before stop", file=sys.stderr)
        return 130
    with locked():
        state = state_read()
        if state.get("client") == identity:
            state.pop("client", None)
            state["last_application_exit"] = code
            if code:
                state["status"] = "application-failed-reboot-required"
            state_write(state)
    return code if code >= 0 else 128 - code


def native_check():
    require(not state_read(), "This boot has used/attempted soft scheduling; reboot before native verification")
    require(not module_loaded() and not ipc_segment(), "Unmanaged soft module/IPC present; native state is not clean")
    require(not device_users(), "Device is busy")
    require(
        not os.environ.get("LD_PRELOAD") and not os.environ.get("RT_INFER_LIBRARY"), "Remove inherited soft preload"
    )
    print(
        "No soft state detected this boot. Run IB-Robot numerical validation with a separate native cache; this check is not an accuracy test."
    )


def uninstall(root, manifest):
    state = state_read()
    require(not state and not module_loaded() and not ipc_segment(), "Reboot to clear soft state before uninstall")
    require(not device_users(), "Device is busy")
    names = [
        "manifest.json",
        *[item["path"] for item in manifest["artifacts"].values()],
        *[item["path"] for item in manifest.get("payload_files", [])],
        *manifest.get("metadata_files", []),
    ]
    allowed = set(names)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == allowed, "Installation contains extra/missing files; refusing recursive removal")
    for name in names:
        member(root, name).unlink()
    for path in sorted(
        (path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True
    ):
        path.rmdir()
    root.rmdir()
    print("Removed this version's files; no CANN files were changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, help="Installed version directory")
    sub = parser.add_subparsers(dest="action", required=True)
    installer = sub.add_parser("install", help="Install a local, checksum-pinned bundle; does not start it")
    installer.add_argument("--bundle", type=Path, required=True)
    installer.add_argument("--prefix", type=Path, required=True)
    verifier = sub.add_parser("verify-bundle", help="Offline artifact integrity check, no device access")
    verifier.add_argument("--bundle", type=Path, required=True)
    starter = sub.add_parser("start")
    starter.add_argument("--exclusive-device", "--experimental", dest="experimental", action="store_true")
    starter.add_argument("--cpu", type=int, required=True)
    starter.add_argument(
        "--device-profile",
        choices=["ascend310p1-rc"],
        required=True,
        help="Operator-confirmed board profile; this is not automatic chip detection",
    )
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--model-env", action="store_true", help="Check Python packages and pinned TorchAir core files")
    doctor.add_argument("--policy", type=Path, help="Check model environment and PI0.5 fast bundle assets/preset")
    for action in ("status", "stop", "verify-native", "uninstall"):
        sub.add_parser(action)
    runner = sub.add_parser("run")
    runner.add_argument("--profile", choices=("standard", "pi05-765"), default="standard")
    runner.add_argument("--cache", type=Path, required=True)
    runner.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.action == "verify-bundle":
            print(json.dumps(verify_bundle(args.bundle), indent=2))
            return 0
        if args.action == "install":
            install(args)
            return 0
        require(args.prefix is not None, "Specify --prefix before the subcommand")
        root_linux()
        root = args.prefix.resolve()
        trusted_tree(root)
        manifest = verify_bundle(root)
        if args.action == "run":
            return run_application(root, manifest, args)
        with locked():
            if args.action in ("status", "doctor"):
                report(
                    root,
                    manifest,
                    args.action == "doctor",
                    getattr(args, "policy", None),
                    getattr(args, "model_env", False),
                )
            elif args.action == "start":
                start(root, manifest, args)
            elif args.action == "stop":
                require(state_read().get("prefix") == str(root), "Managed scheduler belongs to another prefix")
                stop()
            elif args.action == "verify-native":
                native_check()
            elif args.action == "uninstall":
                uninstall(root, manifest)
        return 0
    except (DeploymentError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"rt-infer: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
