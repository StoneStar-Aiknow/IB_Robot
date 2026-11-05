"""Run the candidate10 periodic workload without overwriting previous evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from metrics import analyze_case
from periodic_control import PeriodicControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prefix = Path(os.environ["RTINFER_PREFIX"]).resolve()
    references = Path(os.environ["RTINFER_REFERENCES"]).resolve()
    models = json.loads(Path(os.environ["RTINFER_MODELS"]).read_text())
    assets = json.loads(Path(__file__).with_name("assets.json").read_text())

    def verify(path, key):
        expected = assets[key]
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Asset differs from measured workload: {key}")

    if set(models) != {"pear", "yolox", "te", "fm", "vision", "prefill", "decode"}:
        raise ValueError("Expected exactly seven named models")
    for name, model in models.items():
        if not Path(model).is_absolute() or not Path(model).is_file():
            raise ValueError(f"Missing absolute model path: {name}")
        keys = [key for key in assets if key.startswith(f"models/{name}/") and key.endswith(".om")]
        if len(keys) != 1:
            raise ValueError(f"Expected one pinned model for {name}")
        verify(Path(model), keys[0])
        for filename in ("inputs.npz", "outputs.npz"):
            if not (references / name / filename).is_file():
                raise ValueError(f"Missing reference: {name}/{filename}")
            verify(references / name / filename, f"references/{name}/{filename}")
    if not {0, 10, 11, 12, 13, 15}.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("This reproduction requires CPUs 0,10,11,12,13,15")
    ctl = [sys.executable, str(prefix / "scripts/priorityctl.py"), "--prefix", str(prefix)]
    subprocess.run(ctl + ["verify-bundle", "--bundle", str(prefix)], check=True)
    subprocess.run(ctl + ["doctor"], check=True)
    status = json.loads(subprocess.check_output(ctl + ["status"], text=True))
    if status.get("device_users") or status.get("status") == "running":
        raise RuntimeError("Device or scheduler is busy; coordinate an idle test window")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cmdline": Path("/proc/cmdline").read_text().strip(),
        "isolated": Path("/sys/devices/system/cpu/isolated").read_text().strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    summaries = {}
    for label, scene, duration in (("alone-periodic", "pose", 30), ("full-periodic", "full", 90)):
        path = output / f"{label}-control.bin"
        control = PeriodicControl(path, create=True)
        started = False
        try:
            subprocess.run(
                ctl
                + [
                    "start",
                    "--scheduler",
                    "pfp",
                    "--prefetch",
                    "on",
                    "--cpu",
                    "15",
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
                    "--pfp-periodic",
                    str(path),
                ],
                check=True,
            )
            started = True
            env = os.environ.copy()
            env.update(RT_INFER_SAFE_ARGS_BATCH="1", RT_INFER_PERSISTENT_IO="1", RT_INFER_PERIODIC_PATH=str(path))
            with (output / f"{label}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("latency_periodic10.py")),
                        "--folder",
                        str(output / label),
                        "--mode",
                        "pfp",
                        "--pose-priority",
                        "0",
                        "--scene",
                        scene,
                        "--hz",
                        "18",
                        "--duration",
                        str(duration),
                        "--trim",
                        "5",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            summaries[label] = analyze_case(output / label)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2))
        finally:
            control.disarm()
            control.close()
            if started:
                subprocess.run(ctl + ["stop"], check=True)
    print("Completed. Inspect deadline misses AND background progress. Reboot before native recovery.")


if __name__ == "__main__":
    main()
