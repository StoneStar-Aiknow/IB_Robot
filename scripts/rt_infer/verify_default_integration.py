#!/usr/bin/env python3
"""Model-only acceptance harness for the default owner/wrapper/session path.

No cameras, audio playback, runtime providers, motion or robot launch graph.
Native cases must precede software scheduling in the same clean boot. Each
role retains its models for all rounds. CSV accounts for every absolute due
time; background completion outside the common window does not count.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

FIELDS = ["model", "event", "index", "release_ns", "start_ns", "finish_ns", "response_ms", "service_ms", "checked"]
CHAINS = {
    "pear": ("pear",),
    "yolox": ("yolox",),
    "tts": ("te", "fm", "fm", "fm"),
    "vl": ("vision", "prefill", *(["decode"] * 15)),
}
RATES = {"pear": Fraction(18), "yolox": Fraction(1), "tts": Fraction(2, 15), "vl": Fraction(2, 5)}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def releases(start, duration, rate, stop_requested=lambda: False):
    """Same latest-frame/drop policy as periodic/latency_periodic10.py."""
    count = math.ceil(duration * rate)
    index = 0
    end = start + int(duration * 1e9)
    while index < count:
        if stop_requested():
            while index < count:
                yield index, start + index * 1_000_000_000 * rate.denominator // rate.numerator, False
                index += 1
            return
        due = start + index * 1_000_000_000 * rate.denominator // rate.numerator
        remaining = (due - time.monotonic_ns()) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        now = time.monotonic_ns()
        latest = (
            count
            if now >= end
            else min(count - 1, (now - start) * rate.numerator // (1_000_000_000 * rate.denominator))
        )
        while index < latest:
            yield index, start + index * 1_000_000_000 * rate.denominator // rate.numerator, False
            index += 1
        if index < count:
            yield index, start + index * 1_000_000_000 * rate.denominator // rate.numerator, True
            index += 1


class RoleExecutor:
    """ZipVoice's host role boundary, excluding vocoder/playback work.

    ZipVoice310pSession calls _run_role and reshapes the FM output itself;
    the OM reports a flat/unknown shape although the manifest fixes it.
    Preserve that production adapter behavior for this TE+3FM workload.
    """

    def __init__(self, session):
        self.session = session

    def execute(self, request, context):
        role = request.metadata["execution_role"]
        deployment = self.session._loaded_deployment()
        outputs = self.session._run_role(
            deployment.execution.index(role), role, dict(request.inputs), request=request, context=context
        )
        for binding in deployment.bindings[role].outputs:
            if all(dimension > 0 for dimension in binding.shape):
                outputs[binding.semantic] = outputs[binding.semantic].reshape(binding.shape)
        return outputs


class VLSession:
    """The same ais_bench API used by the external InternVL adapter."""

    def __init__(self, path):
        self.path = path
        self.session = None

    def load(self, _context=None):
        from ais_bench.infer.interface import InferSession

        self.session = InferSession(device_id=0, model_path=self.path)

    def execute(self, request, context):
        context.check("vl_execute")
        outputs = self.session.infer(list(request.inputs.values()))
        context.check("vl_execute")
        return {f"o{i}": value.copy() for i, value in enumerate(outputs)}

    def close(self):
        if self.session is not None:
            self.session.free_resource()
            self.session = None


class Model:
    def __init__(self, name, config, shared=None):
        import numpy as np

        from inference_manifest import load_inference_manifest
        from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
        from inference_service.backends.types import RuntimeContext
        from inference_service.model_sessions.ascend import build_ascend_model_session
        from inference_service.unified_runtime import ModelRuntimeHandle, ResultAdapter, RuntimeAssembly

        self.name, self.config, self.calls, self.checks = name, config, 0, 0
        self.last = None
        self.last_case = 0
        self.owns_handle = shared is None
        self.execution_role = None
        reference = Path(config["references"]) / name
        with np.load(reference / "inputs.npz") as data:
            original = [data[f"i{i}"].copy() for i in range(len(data.files))]
        with np.load(reference / "outputs.npz") as data:
            self.original_expected = [data[f"o{i}"].copy() for i in range(len(data.files))]
        self.inputs = [original]
        for scale in (0.99, 1.01):
            changed = [value.copy() for value in original]
            for value in changed:
                if value.dtype.kind == "f" and value.ndim >= 2:
                    value *= scale
                    break
                if value.dtype.kind in "iu" and value.ndim >= 2 and np.unique(value).size > 1:
                    value[:] = np.roll(value, 1 if scale < 1 else 2, axis=-1)
                    break
            self.inputs.append(changed)
        self.expected = []
        expected_root = Path(config["numeric_cases"])
        expected_root.mkdir(exist_ok=True, parents=True)
        self.manager = None
        if name in ("pear", "yolox", "te", "fm"):
            bundle = config["bundles"]["tts" if name in ("te", "fm") else name]
            validated = load_inference_manifest(bundle, "ascend_310p")
            runtime_context = RuntimeContext(validated, runtime_options={"device_id": 0})
            self.manager = AclRuntimeManager()
            from types import SimpleNamespace

            session = (
                shared.session
                if shared is not None
                else build_ascend_model_session(
                    runtime_context, providers=SimpleNamespace(acl_runtime_provider=self.manager)
                )
            )
            if name in ("te", "fm"):
                role = "text_encoder" if name == "te" else "flow_decoder_1537"
                self.execution_role = role
                bindings = validated.deployment.bindings[role]
                self.input_names = [entry.semantic for entry in bindings.inputs]
                self.output_names = [entry.semantic for entry in bindings.outputs]
                executor = RoleExecutor(session)
            else:
                self.input_names = [entry.semantic for entry in validated.manifest.model.inputs]
                self.output_names = [entry.semantic for entry in validated.manifest.model.outputs]
                executor = session
            self.session = session
        else:
            runtime_context = None
            self.session = VLSession(config["models"][name])
            executor = self.session
            self.input_names = [f"i{i}" for i in range(len(original))]
            self.output_names = [f"o{i}" for i in range(len(self.original_expected))]
        self.handle = (
            shared.handle
            if shared is not None
            else ModelRuntimeHandle(
                RuntimeAssembly(
                    runtime_executor=executor,
                    session=self.session,
                    result_adapter=ResultAdapter(),
                    runtime_id=f"verify-{name}",
                    max_active_executions=1,
                )
            )
        )
        if shared is None:
            self.handle.load(runtime_context)
        for case in range(3):
            output = self.execute(case)
            path = expected_root / f"{name}-{case}.npz"
            if config["create_numeric_cases"]:
                if case == 0:
                    self.compare(output, self.original_expected)
                if path.exists():
                    with np.load(path) as data:
                        self.compare(output, [data[f"o{i}"].copy() for i in range(len(data.files))])
                else:
                    np.savez(path, **{f"o{i}": value for i, value in enumerate(output)})
                expected = [value.copy() for value in output]
            else:
                with np.load(path) as data:
                    expected = [data[f"o{i}"].copy() for i in range(len(data.files))]
                self.compare(output, expected)
            self.expected.append(expected)
        self.buffer_identity = self.buffers()

    def buffers(self):
        return {
            role: {
                "input_dataset": str(model.input_dataset),
                "output_dataset": str(model.output_dataset),
                "device_inputs": [str(item.pointer) for item in model.input_buffers],
                "device_outputs": [str(item.pointer) for item in model.output_buffers],
                "host_outputs": [str(item) for item in model.output_host_buffers],
            }
            for role, model in getattr(self.session, "_models", {}).items()
        }

    def execute(self, case, release=None):
        from inference_service.unified_runtime import ExecutionContext, ModelRequest

        metadata = {} if release is None else {"rt_infer_release_monotonic_ns": release}
        if self.execution_role is not None:
            metadata["execution_role"] = self.execution_role
        request = ModelRequest(inputs=dict(zip(self.input_names, self.inputs[case], strict=True)), metadata=metadata)
        result = self.handle.execute(request, ExecutionContext(f"verify-{self.name}-{self.calls}"))
        return [result.outputs[name] for name in self.output_names]

    def compare(self, outputs, expected):
        import numpy as np

        if len(outputs) != len(expected):
            raise AssertionError(f"{self.name}: output count")
        for index, (actual, wanted) in enumerate(zip(outputs, expected, strict=True)):
            actual, wanted = np.asarray(actual), np.asarray(wanted)
            if self.name == "fm":
                actual, wanted = actual.reshape(-1)[:153700], wanted.reshape(-1)[:153700]
            if actual.shape != wanted.shape or not np.allclose(actual, wanted, rtol=1e-4, atol=1e-5):
                raise AssertionError(f"{self.name}: numeric mismatch output {index}, {actual.shape} vs {wanted.shape}")
        self.checks += 1

    def call(self, index, release, writer):
        case = self.calls % 3
        previous = self.last
        previous_digest = None if previous is None else hashlib.sha256(previous[0].tobytes()).hexdigest()
        start = time.monotonic_ns()
        output = self.execute(case, release if self.name == "pear" else None)
        finish = time.monotonic_ns()
        checked = self.calls % 100 == 0
        if checked:
            self.compare(output, self.expected[case])
            if previous is not None:
                assert hashlib.sha256(previous[0].tobytes()).hexdigest() == previous_digest, (
                    "previous output overwritten"
                )
            assert self.buffers() == self.buffer_identity, "persistent ACL buffers changed"
        self.last, self.last_case = output, case
        self.calls += 1
        writer.writerow(
            dict(
                model=self.name,
                event="complete",
                index=index,
                release_ns=release,
                start_ns=start,
                finish_ns=finish,
                response_ms=(finish - release) / 1e6,
                service_ms=(finish - start) / 1e6,
                checked=int(checked),
            )
        )

    def close(self):
        if self.last is not None:
            self.compare(self.last, self.expected[self.last_case])
        if self.owns_handle:
            self.handle.close()


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload-role", choices=CHAINS, required=True)
    args, _ros = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    root, role = Path(config["output"]), args.workload_role

    def trace(stage, **details):
        with (root / f"events-{role}.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"stage": stage, "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), **details}) + "\n"
            )

    stop_requested = [False]
    signal.signal(signal.SIGINT, lambda *_args: stop_requested.__setitem__(0, True))

    def stopping():
        return stop_requested[0] or (root / "stop-requested.json").exists()

    models = {}
    try:
        for name in dict.fromkeys(CHAINS[role]):
            models[name] = Model(name, config, models.get("te") if name == "fm" else None)
        for model in models.values():
            model.buffer_identity = model.buffers()
        if role == "vl":
            from inference_service.rt_infer_client import mark_runtime_ready

            mark_runtime_ready()
        maps = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libruntime.so" in line and len(line.split()) >= 6
            }
        )
        save(
            root / f"ready-{role}.json",
            {
                "pid": os.getpid(),
                "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "runtime_libraries": maps,
                "token": os.environ.get("RTINFER_LAUNCH_TOKEN"),
                "priority": os.environ.get("RTINFER_STREAM_PRIORITY"),
                "buffers": {name: model.buffer_identity for name, model in models.items()},
                "input_sha256": {
                    name: [hashlib.sha256(b"".join(v.tobytes() for v in case)).hexdigest() for case in model.inputs]
                    for name, model in models.items()
                },
            },
        )
        for round_index in range(config["rounds"]):
            folder = root / f"round-{round_index + 1}"
            deadline = time.monotonic() + config["duration"] + 300
            while not (folder / "go.json").exists():
                if stopping():
                    raise RuntimeError("Launch stopped before the measurement GO")
                if time.monotonic() > deadline:
                    raise TimeoutError("GO did not arrive")
                time.sleep(0.02)
            start = json.loads((folder / "go.json").read_text())["release_ns"]
            end = start + int(config["duration"] * 1e9)
            with (folder / f"{role}.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for index, release, execute in releases(start, config["duration"], RATES[role], stopping):
                    if not execute:
                        writer.writerow(dict(model=role, event="drop", index=index, release_ns=release))
                        continue
                    begin = time.monotonic_ns()
                    completed = True
                    for step, name in enumerate(CHAINS[role]):
                        if stopping() or time.monotonic_ns() >= end:
                            completed = False
                            break
                        models[name].call(
                            index if len(CHAINS[role]) == 1 else index * len(CHAINS[role]) + step, release, writer
                        )
                    if len(CHAINS[role]) > 1:
                        finish = time.monotonic_ns()
                        writer.writerow(
                            dict(
                                model=role,
                                event="complete" if completed else "incomplete",
                                index=index,
                                release_ns=release,
                                start_ns=begin,
                                finish_ns=finish,
                                response_ms=(finish - release) / 1e6,
                                service_ms=(finish - begin) / 1e6,
                            )
                        )
                    elif not completed:
                        writer.writerow(dict(model=role, event="drop", index=index, release_ns=release))
                    stream.flush()
            for model in models.values():
                if model.last is not None:
                    model.compare(model.last, model.expected[model.last_case])
            save(
                folder / f"done-{role}.json",
                {name: {"calls": m.calls, "checks": m.checks} for name, m in models.items()},
            )
        if role == "pear" and config.get("lifecycle_probe"):
            before = models["pear"].session._periodic_admission._last_release_ns
            models.pop("pear").close()
            idle_start = time.monotonic_ns()
            time.sleep(0.25)  # Explicit lifecycle test interval, never an admission delay.
            reloaded = Model("pear", config)
            models["pear"] = reloaded
            for index in range(10):
                output = reloaded.execute(index % 3)
                reloaded.compare(output, reloaded.expected[index % 3])
            after = reloaded.session._periodic_admission._last_release_ns
            assert after > before
            save(
                root / "pear-session-reopen.json",
                {
                    "pid": os.getpid(),
                    "idle_start_ns": idle_start,
                    "last_release_before": before,
                    "last_release_after": after,
                    "new_session_calls": 10,
                    "numeric_pass": True,
                    "scope": "same process ModelRuntimeHandle/ModelSession close and reopen; owner stays active",
                },
            )
        for model in reversed(tuple(models.values())):
            trace("handle_close_begin", model=model.name)
            model.close()
            trace("handle_close_end", model=model.name)
        models.clear()
        save(root / f"finished-{role}.json", {"success": True})
        trace("worker_finished")
    except BaseException:
        import traceback

        save(root / f"failed-{role}.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        for model in reversed(tuple(models.values())):
            model.handle.close()


async def campaign(config, *, failed=lambda: False, end_window_only=False):
    root, roles = Path(config["output"]), config["roles"]

    async def wait_for(paths, timeout):
        until = time.monotonic() + timeout
        while not all(path.exists() for path in paths):
            if failed() or list(root.glob("failed-*.json")):
                raise RuntimeError("worker/owner failed; inspect first failure logs")
            if time.monotonic() > until:
                raise TimeoutError(f"Timed out waiting for {[str(path) for path in paths]}")
            await asyncio.sleep(0.05)

    await wait_for([root / f"ready-{role}.json" for role in roles], 240)
    if config["mode"] == "soft":
        until = time.monotonic() + 10
        while json.loads(Path("/run/ibrobot-rtinfer/periodic-session.json").read_text()).get("phase") != "active":
            if failed() or time.monotonic() > until:
                raise RuntimeError("Owner did not activate periodic admission after real warmup")
            await asyncio.sleep(0.05)
    for index in range(config["rounds"]):
        folder = root / f"round-{index + 1}"
        folder.mkdir()
        release = time.monotonic_ns() + 500_000_000
        if config["mode"] == "soft":
            periodic = json.loads(Path("/run/ibrobot-rtinfer/periodic-session.json").read_text())
            epoch, period = periodic["epoch_ns"], periodic["period_ns"]
            release = epoch + max(0, math.ceil((release - epoch) / period)) * period
            save(folder / "periodic-session.json", periodic)
            save(folder / "vendor-state-start.json", json.loads(Path("/run/ibrobot-rtinfer/state.json").read_text()))
        save(folder / "go.json", {"release_ns": release, "duration": config["duration"], "trim_sec": 5})
        if end_window_only:
            await asyncio.sleep(max(0, (release + int(config["duration"] * 1e9) - time.monotonic_ns()) / 1e9))
            save(folder / "window-closed.json", {"closed_monotonic_ns": time.monotonic_ns()})
            if config.get("lifecycle_probe"):
                await wait_for([root / "pear-session-reopen.json"], 30)
            save(
                root / "stop-requested.json",
                {"monotonic_ns": time.monotonic_ns(), "reason": "measurement window closed"},
            )
            return
        await wait_for([folder / f"done-{role}.json" for role in roles], config["duration"] + 180)
        if config["mode"] == "soft":
            save(folder / "vendor-state-end.json", json.loads(Path("/run/ibrobot-rtinfer/state.json").read_text()))
    await wait_for([root / f"finished-{role}.json" for role in roles], 30)


def run_campaign(args):
    from robot_config.loader import load_robot_config_dict
    from robot_config.rt_infer_config import parse_rt_infer_config

    robot = load_robot_config_dict(args.robot_config, defer_interface_binding=True)
    profile = parse_rt_infer_config(robot)
    if args.mode == "soft" and args.rounds != 1:
        raise ValueError("Run independent soft launches with --rounds 1: final teardown disarms each owner session")
    args.output.mkdir(parents=True, exist_ok=False)
    asset_root = args.assets
    models = {
        name: str(next((asset_root / "models" / name).glob("*.om")))
        for name in ("pear", "yolox", "te", "fm", "vision", "prefill", "decode")
    }
    if "b4" not in Path(models["vision"]).name:
        raise RuntimeError("Acceptance requires the b4 vision model")
    config = {
        "mode": args.mode,
        "output": str(args.output),
        "duration": args.duration,
        "rounds": args.rounds,
        "roles": ["pear"] if args.pose_only else list(CHAINS),
        "models": models,
        "references": str(asset_root / "references"),
        "numeric_cases": str(args.numeric_cases),
        "create_numeric_cases": args.create_numeric_cases,
        "lifecycle_probe": args.lifecycle_probe,
        "bundles": {
            "pear": str(args.bundles / "pear_parameter_network"),
            "yolox": str(args.bundles / "yolox_x_640"),
            "tts": str(args.bundles / "zipvoice"),
        },
    }
    save(args.output / "config.json", config)
    if args.mode == "native":
        state = Path("/run/ibrobot-rtinfer/state.json")
        if state.exists() or "aicore_device" in Path("/proc/modules").read_text():
            raise RuntimeError("Native acceptance requires a clean boot before soft scheduling")
        children, logs = [], []
        try:
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("RTINFER_", "RT_INFER_", "AICORE_SCHED_")) and key != "LD_PRELOAD"
            }
            for role in config["roles"]:
                log = (args.output / f"{role}.log").open("x")
                logs.append(log)
                children.append(
                    subprocess.Popen(
                        [
                            "taskset",
                            "-c",
                            str(profile.cpu_for(role)),
                            sys.executable,
                            "-m",
                            "scripts.rt_infer.verify_default_integration",
                            "worker",
                            "--config",
                            str(args.output / "config.json"),
                            "--workload-role",
                            role,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                )
            asyncio.run(campaign(config, failed=lambda: any(child.poll() not in (None, 0) for child in children)))
            for child in children:
                assert child.wait(timeout=30) == 0
            save(
                args.output / "campaign-complete.json",
                {
                    "status": "executed",
                    "numeric_checks": "passed",
                    "roles": config["roles"],
                    "process_exits": [
                        {"role": role, "pid": child.pid, "returncode": child.returncode}
                        for role, child in zip(config["roles"], children, strict=True)
                    ],
                },
            )
        finally:
            for child in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            for log in logs:
                log.close()
        return 0

    from launch import Action, LaunchDescription, LaunchService
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import Node

    from inference_service.launch_builders.perception_models import write_rt_infer_snapshot
    from robot_config.launch_builders.rt_infer_scheduler import build_rt_infer_scheduler
    from robot_config.launch_registry.registry import LaunchBuildContext

    class Campaign(Action):
        def execute(self, context):
            async def run():
                await campaign(config, failed=lambda: context.is_shutdown, end_window_only=True)
                context.emit_event_sync(Shutdown(reason="measurement window closed; drain owned clients"))

            self.future = context.asyncio_loop.create_task(run())

        def get_asyncio_future(self):
            return getattr(self, "future", None)

    result = build_rt_infer_scheduler(LaunchBuildContext(robot_config=robot, use_sim=False))
    snapshot = write_rt_infer_snapshot(robot)
    nodes = [
        Node(
            package="inference_service",
            executable="rt_infer_client",
            sigterm_timeout=str(profile.stop_timeout_sec + 1),
            name=f"rtinfer_verify_{role}",
            output="screen",
            arguments=[
                "--role",
                role,
                "--rt-infer-config",
                str(snapshot),
                "--node-main",
                "scripts.rt_infer.verify_default_integration:worker_main",
                "--",
                "--config",
                str(args.output / "config.json"),
                "--workload-role",
                role,
            ],
        )
        for role in config["roles"]
    ]
    service = LaunchService()
    exits = []

    def record_exit(event, _context):
        exits.append({"name": event.process_name, "pid": event.pid, "returncode": event.returncode, "cmd": event.cmd})
        return []

    service.include_launch_description(
        LaunchDescription(
            [
                RegisterEventHandler(OnProcessExit(on_exit=record_exit)),
                *result.actions,
                *nodes,
                Campaign(),
            ]
        )
    )
    code = asyncio.run(service.run_async())
    save(args.output / "process-exits.json", exits)
    if (
        code
        or len(exits) != len(config["roles"]) + 1
        or any(entry["returncode"] for entry in exits)
        or not all((args.output / f"finished-{role}.json").exists() for role in config["roles"])
    ):
        raise RuntimeError(f"Soft launch/drain failed (launch exit {code}); inspect first error logs")
    save(
        args.output / "campaign-complete.json",
        {"status": "executed", "numeric_checks": "passed", "roles": config["roles"], "process_exits": exits},
    )
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        worker_main()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "soft"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--numeric-cases", type=Path, required=True)
    parser.add_argument("--create-numeric-cases", action="store_true")
    parser.add_argument("--pose-only", action="store_true")
    parser.add_argument("--lifecycle-probe", action="store_true")
    parser.add_argument("--duration", type=float, default=95)
    parser.add_argument("--rounds", type=int, default=3)
    return run_campaign(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
