#!/usr/bin/env python3
"""Periodic latest-frame PEAR and bounded, deterministic seven-OM interference."""

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path

PREFIX = Path(os.environ["RTINFER_PREFIX"]).resolve()
REFERENCES = Path(os.environ["RTINFER_REFERENCES"]).resolve()
KEEPALIVE = []
FIELDS = ["model", "event", "index", "release_ns", "start_ns", "finish_ns", "response_ms", "service_ms", "checked"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_until(target):
    remaining = (target - time.monotonic_ns()) / 1e9
    if remaining > 0:
        time.sleep(remaining)


def periodic_releases(start, duration, hz, now=time.monotonic_ns, sleep=wait_until):
    """Account for every release; retain the latest frame, never burst catch up."""
    total = math.ceil(duration * hz)
    numerator, denominator = hz.numerator, hz.denominator
    index = 0
    end = start + int(duration * 1e9)
    while index < total:
        due = start + (index * 1_000_000_000 * denominator) // numerator
        sleep(due)
        current = now()
        latest = (
            total
            if current >= end
            else min(total - 1, ((current - start) * numerator) // (1_000_000_000 * denominator))
        )
        while index < latest:
            yield index, start + (index * 1_000_000_000 * denominator) // numerator, False
            index += 1
        if index < total:
            yield index, start + (index * 1_000_000_000 * denominator) // numerator, True
            index += 1


def worker(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baseline import OM, InferSession, np

    folder = args.folder
    config = json.loads((folder / "config.json").read_text())
    periodic = None
    if args.role == "pose" and config.get("periodic_path"):
        from periodic_control import PeriodicControl

        periodic = PeriodicControl(config["periodic_path"])
    role = args.role
    errors = []

    def mark_ready():
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libruntime.so" in line and len(line.split()) >= 6
            }
        )
        save(
            folder / f"ready-{role}.json",
            {
                "pid": os.getpid(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "persistent_io": os.environ.get("RT_INFER_PERSISTENT_IO"),
                "safe_args_batch": os.environ.get("RT_INFER_SAFE_ARGS_BATCH"),
                "stream_priority": os.environ.get("RTINFER_STREAM_PRIORITY"),
                "runtime_libraries": libraries,
            },
        )

    class Model:
        def __init__(self, name):
            self.name = name
            with np.load(REFERENCES / name / "inputs.npz") as archive:
                self.inputs = [archive[f"i{i}"].copy() for i in range(len(archive.files))]
            with np.load(REFERENCES / name / "outputs.npz") as archive:
                self.expected = [archive[f"o{i}"].copy() for i in range(len(archive.files))]
            self.persistent = name == "pear" and os.environ.get("RT_INFER_PERSISTENT_IO") == "1"
            if self.persistent:
                from persistent_acl import PersistentSession

                self.session = PersistentSession(OM[name], self.expected)
            else:
                self.session = InferSession(device_id=0, model_path=OM[name])
            if name == "fm":
                self.session.set_custom_outsize([1048576])
            self.file = (folder / f"{role}-{name}.csv").open("x", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
            self.writer.writeheader()
            self.calls = self.checks = self.drops = 0
            self.last = None
            KEEPALIVE.append(self)
            for _ in range(3):
                self.validate(self.execute())

        def execute(self):
            return (
                self.session.run(self.inputs, out_array=True)
                if self.name in ("te", "fm")
                else self.session.infer(self.inputs)
            )

        def validate(self, output):
            if len(output) != len(self.expected):
                raise RuntimeError(f"{self.name}: changed output count")
            for number, (actual, expected) in enumerate(zip(output, self.expected, strict=True)):
                actual = np.asarray(actual)
                if self.name == "fm" and number == 0:
                    actual, expected = actual.reshape(-1)[:153700], expected.reshape(-1)[:153700]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"{self.name}: output {number} differs from native reference")
            self.checks += 1

        def call(self, index, release):
            start = time.monotonic_ns()
            if periodic and self.name == "pear":
                periodic.release(release)
            output = self.execute()
            finish = time.monotonic_ns()
            checked = self.calls % 100 == 0
            if checked:
                self.validate(output)
            self.last = output
            self.calls += 1
            self.writer.writerow(
                {
                    "model": self.name,
                    "event": "complete",
                    "index": index,
                    "release_ns": release,
                    "start_ns": start,
                    "finish_ns": finish,
                    "response_ms": (finish - release) / 1e6,
                    "service_ms": (finish - start) / 1e6,
                    "checked": int(checked),
                }
            )
            if self.calls % 100 == 0:
                self.file.flush()

        def drop(self, index, release):
            self.drops += 1
            self.writer.writerow({"model": self.name, "event": "drop", "index": index, "release_ns": release})

    go_deadline = time.monotonic() + 180

    def go():
        while not (folder / "go.json").exists():
            if time.monotonic() > go_deadline:
                raise TimeoutError("GO did not arrive")
            time.sleep(0.01)
        return json.loads((folder / "go.json").read_text())["release_ns"]

    if role in ("pose", "yolo"):
        ready = threading.Barrier(2)

        def pose_thread(name, hz):
            try:
                model = Model(name)
                ready.wait(timeout=180)
                start = go()
                for index, release, execute in periodic_releases(start, config["duration"], hz):
                    if execute:
                        model.call(index, release)
                    else:
                        model.drop(index, release)
            except Exception:  # noqa: BLE001 -- propagate worker failures to the joining main thread
                errors.append(traceback.format_exc())
                ready.abort()

        threads = [
            threading.Thread(target=pose_thread, args=("pear", config["hz"]) if role == "pose" else ("yolox", 1))
        ]
        for thread in threads:
            thread.start()
        ready.wait(timeout=180)
        mark_ready()
        for thread in threads:
            thread.join(timeout=config["duration"] + 180)
        if errors or any(thread.is_alive() for thread in threads):
            raise RuntimeError("\n".join(errors) or "Pose worker hung")
    else:
        names = ("te", "fm") if role == "tts" else ("vision", "prefill", "decode")
        models = {name: Model(name) for name in names}
        mark_ready()
        start = go()
        # Same absolute offered schedule for every strategy. One outstanding chain.
        hz = Fraction(2, 15) if role == "tts" else Fraction(2, 5)
        sequence = ("te", "fm", "fm", "fm") if role == "tts" else ("vision", "prefill", *(["decode"] * 15))
        with (folder / f"{role}-chains.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for index, release, execute in periodic_releases(start, config["duration"], hz):
                if not execute:
                    writer.writerow({"model": role, "event": "drop", "index": index, "release_ns": release})
                    continue
                begin = time.monotonic_ns()
                for name in sequence:
                    models[name].call(models[name].calls, release)
                finish = time.monotonic_ns()
                writer.writerow(
                    {
                        "model": role,
                        "event": "complete",
                        "index": index,
                        "release_ns": release,
                        "start_ns": begin,
                        "finish_ns": finish,
                        "response_ms": (finish - release) / 1e6,
                        "service_ms": (finish - begin) / 1e6,
                    }
                )
                file.flush()

    if periodic:
        periodic.disarm()
        periodic.close()
    for model in KEEPALIVE:
        if model.last is not None:
            model.validate(model.last)
        model.file.close()
        if model.persistent:
            model.session.close()
    save(
        folder / f"completed-{role}.json",
        {
            "pid": os.getpid(),
            "numerical": True,
            "models": {
                model.name: {"calls": model.calls, "drops": model.drops, "checks": model.checks} for model in KEEPALIVE
            },
        },
    )


def run_case(args):
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=False)
    config = {
        "periodic_path": os.environ.get("RT_INFER_PERIODIC_PATH"),
        "high_budget_ns": 38_000_000,
        "persistent_io": os.environ.get("RT_INFER_PERSISTENT_IO"),
        "safe_args_batch": os.environ.get("RT_INFER_SAFE_ARGS_BATCH"),
        "mode": args.mode,
        "scene": args.scene,
        "hz": args.hz,
        "duration": args.duration,
        "trim": args.trim,
        "priorities": {
            "pose": args.pose_priority,
            "yolo": args.background_priority,
            "tts": args.background_priority,
            "vl": args.background_priority,
        },
        "cpu": {"pose": 13, "yolo": 10, "tts": 12, "vl": 11, "scheduler": 14},
        "prefetch": "off",
        "deadline_ms": 1000 / args.hz,
        "drop_policy": "latest frame; all offered releases counted",
        "output_policy": "host arrays in every path",
        "correctness": "3 warmups, first/every100 completed calls and final output; native references",
    }
    config["measurement_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(folder / "config.json", config)
    ctl = [sys.executable, str(PREFIX / "scripts/priorityctl.py"), "--prefix", str(PREFIX)]
    roles = ["pose"] if args.scene == "pose" else ["pose", "yolo", "tts", "vl"]
    if args.mode != "native":
        state = json.loads(Path("/run/ibrobot-rtinfer/state.json").read_text())
        assert state["status"] == "running" and state["prefix"] == str(PREFIX) and state["strategy"] == args.mode
        assert state["prefetch"] in ("on", "off")
        config["prefetch"] = state["prefetch"]
        config["cpu"]["scheduler"] = int(state["cpu"])
        assert os.sched_getaffinity(state["scheduler"]["pid"]) == {int(state["cpu"])}
        config["scheduler_pid"] = state["scheduler"]["pid"]
    else:
        result = subprocess.run([*ctl, "verify-native"], capture_output=True, text=True, check=True)
        (folder / "native-state.log").write_text(result.stdout + result.stderr)
    config["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    save(folder / "config.json", config)
    children, logs = [], []
    try:
        for role in roles:
            argv = [sys.executable, str(Path(__file__).resolve()), "--role", role, "--folder", str(folder)]
            cpu = config["cpu"][role]
            if args.mode == "native":
                argv = ["taskset", "-c", str(cpu), *argv]
            else:
                argv = [
                    *ctl,
                    "run",
                    "--priority",
                    str(config["priorities"][role]),
                    "--cpu",
                    str(cpu),
                    "--",
                    "/usr/bin/env",
                    "RT_INFER_SAFE_ARGS_BATCH=" + os.environ["RT_INFER_SAFE_ARGS_BATCH"],
                    "RT_INFER_PERSISTENT_IO=" + os.environ.get("RT_INFER_PERSISTENT_IO", "0"),
                    *argv,
                ]
            log = (folder / f"{role}.log").open("x")
            logs.append(log)
            children.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        until = time.monotonic() + 180
        while not all((folder / f"ready-{role}.json").exists() for role in roles):
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Worker failed during initialization; inspect role logs")
            if time.monotonic() > until:
                raise TimeoutError("Load/warmup timeout")
            time.sleep(0.05)
        release_ns = time.monotonic_ns() + 200_000_000
        if config.get("periodic_path"):
            from periodic_control import PeriodicControl

            publisher = PeriodicControl(config["periodic_path"])
            publisher.arm(release_ns, (1_000_000_000 + args.hz - 1) // args.hz, config["high_budget_ns"])
            publisher.close()
        save(folder / "go.json", {"release_ns": release_ns})
        until = time.monotonic() + args.duration + 180
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Worker failed; inspect role logs")
            if time.monotonic() > until:
                raise TimeoutError("Inference completion timeout")
            time.sleep(0.2)
        codes = [child.returncode for child in children]
        assert codes == [0] * len(roles)
        reports = {role: json.loads((folder / f"completed-{role}.json").read_text()) for role in roles}
        assert all(report["numerical"] for report in reports.values())
        save(folder / "completed.json", {"exit_codes": codes, "reports": reports, "status": "PASS"})
        print("CASE PASS", folder.name, flush=True)
    except BaseException:
        save(folder / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("pose", "yolo", "tts", "vl"))
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "fifo", "nice", "pfp"), default="native")
    parser.add_argument("--scene", choices=("pose", "full"), default="full")
    parser.add_argument("--hz", type=int, choices=(18, 20), default=18)
    parser.add_argument("--duration", type=int, default=80)
    parser.add_argument("--trim", type=int, default=20)
    parser.add_argument("--pose-priority", type=int, default=1)
    parser.add_argument("--background-priority", type=int, default=3)
    options = parser.parse_args()
    if options.role:
        worker(options)
    else:
        assert 0 <= options.trim < options.duration
        run_case(options)
