#!/usr/bin/env python3
"""Finite p/t/v NPU-only combination: every completed call checks native output."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

WORK = Path(os.environ["RTINFER_RESULTS"]).resolve()
PREFIX = Path(os.environ["RTINFER_PREFIX"]).resolve()
REFERENCES = Path(os.environ["RTINFER_REFERENCES"]).resolve()
KEEPALIVE = []


def wait_go(folder):
    deadline = time.monotonic() + 180
    while not (folder / "go.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Other roles did not become ready")
        time.sleep(0.02)
    return json.loads((folder / "go.json").read_text())["release_ns"]


def worker(role, folder):
    import numpy as np
    from ais_bench.infer.interface import InferSession

    OM = json.loads(Path(os.environ["RTINFER_MODELS"]).read_text())

    class Model:
        def __init__(self, name):
            self.name = name
            with np.load(REFERENCES / name / "inputs.npz") as archive:
                self.inputs = [archive[f"i{i}"].copy() for i in range(len(archive.files))]
            with np.load(REFERENCES / name / "outputs.npz") as archive:
                self.expected = [archive[f"o{i}"].copy() for i in range(len(archive.files))]
            self.session = InferSession(device_id=0, model_path=OM[name])
            if name == "fm":
                self.session.set_custom_outsize([1048576])
            self.output = (folder / f"{role}-{name}.jsonl").open("x")
            self.completed = 0
            KEEPALIVE.append(self)
            for _ in range(3):
                self.call(record=False)

        def call(self, record=True, release=None):
            start = time.monotonic_ns()
            result = (
                self.session.run(self.inputs, out_array=True)
                if self.name in ("te", "fm")
                else self.session.infer(self.inputs)
            )
            finish = time.monotonic_ns()
            errors = []
            if len(result) != len(self.expected):
                raise RuntimeError(f"{self.name}: output count changed")
            for i, (actual, expected) in enumerate(zip(result, self.expected, strict=True)):
                actual = np.asarray(actual)
                if self.name == "fm" and i == 0:
                    actual, expected = actual.reshape(-1)[:153700], expected.reshape(-1)[:153700]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"{self.name}: output {i} differs from native reference")
                errors.append(float(np.max(np.abs(actual.astype(np.float64) - expected))))
            if record:
                self.completed += 1
                self.output.write(
                    json.dumps(
                        {
                            "model": self.name,
                            "iteration": self.completed,
                            "release_ns": release if release is not None else start,
                            "start_ns": start,
                            "finish_ns": finish,
                            "execute_ms": (finish - start) / 1e6,
                            "numerical": True,
                            "max_abs": errors,
                        }
                    )
                    + "\n"
                )
                self.output.flush()

    counts = {}
    errors = []
    if role == "pose":
        barrier = threading.Barrier(3)

        def pose_thread(name, iterations, period):
            try:
                model = Model(name)
                barrier.wait(timeout=180)
                release = wait_go(folder)
                for i in range(iterations):
                    due = release + i * period
                    time.sleep(max(0, (due - time.monotonic_ns()) / 1e9))
                    model.call(release=due)
                counts[name] = model.completed
            except BaseException:
                errors.append(traceback.format_exc())
                barrier.abort()

        threads = [
            threading.Thread(target=pose_thread, args=("pear", 36, 55556000)),
            threading.Thread(target=pose_thread, args=("yolox", 3, 1000000000)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=180)
        (folder / "ready-pose.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
        for thread in threads:
            thread.join(timeout=180)
        if errors or any(thread.is_alive() for thread in threads):
            raise RuntimeError("\n".join(errors) or "Pose thread timeout")
    else:
        names = ("te", "fm") if role == "tts" else ("vision", "prefill", "decode")
        models = {name: Model(name) for name in names}
        (folder / f"ready-{role}.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
        release = wait_go(folder)
        time.sleep(max(0, (release - time.monotonic_ns()) / 1e9))
        sequence = ("te", "fm", "fm", "fm") if role == "tts" else ("vision", "prefill", *(["decode"] * 15))
        for name in sequence:
            models[name].call()
        counts = {name: model.completed for name, model in models.items()}
    for model in KEEPALIVE:
        model.output.close()
    (folder / f"completed-{role}.json").write_text(
        json.dumps({"pid": os.getpid(), "counts": counts, "numerical": True}) + "\n"
    )
    print("ROLE PASS", role, counts, flush=True)


def orchestrate(strategy, label):
    state = json.loads(Path("/run/ibrobot-rtinfer/state.json").read_text())
    assert state["status"] == "running" and state["prefix"] == str(PREFIX) and state["strategy"] == strategy
    assert f"Scheduler={strategy} prefetch=off" in Path("/run/ibrobot-rtinfer/scheduler.log").read_text()
    folder = WORK / "combinations" / label
    folder.mkdir(parents=True, exist_ok=False)
    ctl = [sys.executable, str(PREFIX / "scripts/priorityctl.py"), "--prefix", str(PREFIX)]
    children = []
    logs = []
    try:
        for role, priority, cpu in (("pose", 1, 13), ("tts", 3, 12), ("vl", 3, 11)):
            log = (folder / f"{role}.log").open("x")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        *ctl,
                        "run",
                        "--priority",
                        str(priority),
                        "--cpu",
                        str(cpu),
                        "--",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--role",
                        role,
                        "--folder",
                        str(folder),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 180
        while not all((folder / f"ready-{role}.json").exists() for role in ("pose", "tts", "vl")):
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A role exited during load/warmup; inspect role logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Role initialization timeout")
            time.sleep(0.05)
        (folder / "go.json").write_text(json.dumps({"release_ns": time.monotonic_ns() + 100000000}) + "\n")
        codes = [child.wait(timeout=180) for child in children]
        if any(codes):
            raise RuntimeError(f"Combination failed: {codes}")
        expected = {
            "pose": {"pear": 36, "yolox": 3},
            "tts": {"te": 1, "fm": 3},
            "vl": {"vision": 1, "prefill": 1, "decode": 15},
        }
        reports = {role: json.loads((folder / f"completed-{role}.json").read_text()) for role in expected}
        assert all(report["numerical"] and report["counts"] == expected[role] for role, report in reports.items())
        result = {
            "strategy": strategy,
            "boot_id": state["boot_id"],
            "scheduler_pid": state["scheduler"]["pid"],
            "prefetch": state["prefetch"],
            "scope": "finite synthetic p/t/v combination; not a tail-latency acceptance",
            "exit_codes": codes,
            "reports": reports,
            "numerical": "PASS",
        }
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("Owned test group still active:", child.pid, file=sys.stderr)
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("pose", "tts", "vl"))
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--strategy", default="fifo", choices=("fifo", "nice", "pfp"))
    parser.add_argument("--label", default="fifo-01")
    args = parser.parse_args()
    if args.role:
        worker(args.role, args.folder)
    else:
        orchestrate(args.strategy, args.label)
