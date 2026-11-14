"""Find and stop only processes owned by this interaction checkout (Linux)."""

import argparse
import os
from pathlib import Path
import signal
import time


def snapshot(root, launch, config, all_x2_runtimes=False):
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal() or int(directory.name) == os.getpid():
            continue
        try:
            args = directory.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            stat = directory.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
            if stat[0] == "Z":
                continue
            # Match complete argv tokens, never a shell's command string.
            node = any(a.startswith((root + "/install/lib/", root + "/build/")) for a in args[:2])
            leader = (
                "ros2" in [Path(a).name for a in args[:2]]
                and "launch" in args and launch in args and "config_path:=" + config in args
            )
            environment = directory.joinpath("environ").read_bytes().split(b"\0")
            tagged = ("IBROBOT_INTERACTION_OWNER=" + root).encode() in environment
            # Explicit opt-in for duplicate X2 runtimes from other checkouts.
            runtime = all_x2_runtimes and (
                any(a.endswith("/install/lib/aimdk_robot/aimdk_runtime") for a in args[:2])
                or (
                    "ros2" in [Path(a).name for a in args[:2]]
                    and any(args[i:i + 3] == ["launch", "aimdk_robot", "runtime.launch.py"] for i in (1, 2))
                    and any(a.startswith("profile:=x2") for a in args)
                )
            )
            if node or leader or tagged or runtime:
                found[int(directory.name)] = (stat[19], args)
        except (OSError, ValueError, UnicodeError, IndexError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "stop"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--all-x2-runtimes", action="store_true", help="include aimdk runtimes from other checkouts")
    options = parser.parse_args()
    root = str(Path(options.root).resolve())
    config = root + "/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
    owned = lambda: snapshot(root, options.launch, config, options.all_x2_runtimes)
    found = owned()
    if options.action == "stop":
        # Re-scan between phases to catch late-spawned nodes. Recheck start time
        # before every signal so PID reuse cannot target another process.
        for sig, budget in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            for pid, (started, _) in found.items():
                current = owned().get(pid)
                if current is not None and current[0] == started:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            until = time.monotonic() + budget
            while time.monotonic() < until:
                found = owned()
                if not found:
                    break
                time.sleep(0.2)
            if not found:
                break
    found = owned()
    for pid, (_, args) in found.items():
        print(f"remaining pid={pid}: {' '.join(args)}", flush=True)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
