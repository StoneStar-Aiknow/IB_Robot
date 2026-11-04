"""Read-only Linux PHC sampling and LinuxPTP status acquisition."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .clock_calibration import (
    CalibrationPolicy,
    CaptureIdentity,
    ClockSample,
    PtpStatus,
    load_calibration,
)

_DATASETS = ("DEFAULT_DATA_SET", "PORT_DATA_SET", "TIME_STATUS_NP", "CURRENT_DATA_SET", "PORT_PROPERTIES_NP")


def parse_pmc(text: str, observed_monotonic_ns: int) -> PtpStatus:
    """Parse one local, single-port PMC response; reject duplicate/missing TLVs."""
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"RESPONSE MANAGEMENT (\w+)\s*$", line)
        if match:
            name = match[1]
            if name in sections:
                raise ValueError(f"ambiguous PMC response: repeated {name}")
            current = sections[name] = {}
        elif current is not None:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                if fields[0] in current:
                    raise ValueError(f"duplicate PMC field: {fields[0]}")
                current[fields[0]] = fields[1]
    if set(sections) != set(_DATASETS):
        raise ValueError("incomplete or unexpected PMC datasets")
    defaults, port, timing, path, properties = (sections[name] for name in _DATASETS)
    if port["portIdentity"].rsplit("-", 1)[0] != defaults["clockIdentity"]:
        raise ValueError("PMC clock/port mismatch")
    if properties["portIdentity"] != port["portIdentity"]:
        raise ValueError("PMC port properties mismatch")
    if properties["portState"] != port["portState"]:
        raise ValueError("PMC port changed state during status acquisition")
    if timing["gmPresent"] not in {"true", "false"}:
        raise ValueError("invalid PMC gmPresent")
    delay = float(path["meanPathDelay"])
    if not math.isfinite(delay):
        raise ValueError("non-finite PMC meanPathDelay")
    return PtpStatus(
        observed_monotonic_ns=observed_monotonic_ns,
        domain=int(defaults["domainNumber"]),
        clock_identity=defaults["clockIdentity"],
        grandmaster_identity=timing["gmIdentity"],
        port_state=port["portState"],
        timestamping=properties["timestamping"],
        interface=properties["interface"],
        gm_present=timing["gmPresent"] == "true",
        master_offset_ns=int(timing["master_offset"]),
        mean_path_delay_ns=math.ceil(delay),
        ingress_time_ns=int(timing["ingress_time"]),
        gm_time_base_indicator=int(timing["gmTimeBaseIndicator"]),
    )


def sample_clocks(clock_id: int, *, attempts: int = 9) -> dict[str, int]:
    """All clock reads lie inside a measured MONOTONIC bracket."""
    if not 1 <= attempts <= 100:
        raise ValueError("attempts must be in 1..100")
    best = None
    for _ in range(attempts):
        before = time.monotonic_ns()
        realtime = time.time_ns()
        raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        phc = time.clock_gettime_ns(clock_id)
        after = time.monotonic_ns()
        sample = {
            "monotonic_ns": (before + after) // 2,
            "monotonic_raw_ns": raw,
            "realtime_ns": realtime,
            "boottime_ns": boot,
            "phc_ns": phc,
            "bracket_ns": after - before,
        }
        if best is None or sample["bracket_ns"] < best["bracket_ns"]:
            best = sample
    return best


class PmcMonitor:
    def __init__(self, socket_path: str, domain: int, client_directory: str):
        self.command = [
            "pmc",
            "-u",
            "-b",
            "0",
            "-d",
            str(domain),
            "-s",
            socket_path,
            "-i",
            str(Path(client_directory) / "pmc.sock"),
            *[f"GET {name}" for name in _DATASETS],
        ]
        self.lock = threading.Lock()
        self.latest = (None, "", "not polled", 0)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, name="clock-pmc", daemon=True)

    def _poll(self):
        while not self.stop.is_set():
            started = time.monotonic_ns()
            raw, error, status = "", "", None
            try:
                result = subprocess.run(self.command, capture_output=True, text=True, timeout=3, check=True)
                raw = result.stdout
                if len(raw) > 65536:
                    raise ValueError("oversized PMC reply")
                status = parse_pmc(raw, started)
            except (OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
                error = str(exc)
            with self.lock:
                self.latest = (status, raw, error, started)
            self.stop.wait(0.1)

    def snapshot(self):
        with self.lock:
            return self.latest

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)


def capture(args) -> None:
    if not math.isfinite(args.duration) or not 0 < args.duration <= 600:
        raise ValueError("duration must be in (0, 600] seconds")
    if not math.isfinite(args.rate) or not 1 <= args.rate <= 100:
        raise ValueError("rate must be in 1..100 Hz")
    phc = Path(args.phc).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", args.interface):
        raise ValueError("invalid network interface name")
    ethtool = subprocess.check_output(["ethtool", "-T", args.interface], text=True, timeout=5)
    indices = re.findall(r"^\s*PTP Hardware Clock:\s*(\d+)\s*$", ethtool, re.MULTILINE)
    if len(indices) != 1 or phc.name != f"ptp{indices[0]}":
        raise ValueError("PHC does not belong to the selected network interface")
    identity = CaptureIdentity(
        session_id=args.session_id,
        machine_id=Path("/etc/machine-id").read_text().strip(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        interface=args.interface,
        phc=str(phc),
        grandmaster_identity=args.grandmaster,
        domain=args.domain,
    )
    policy = CalibrationPolicy()
    header = {
        "kind": "clock_calibration",
        "schema_version": 1,
        "identity": asdict(identity),
        "policy": asdict(policy),
        "provenance": {
            "pmc_socket": args.pmc_socket,
            "sample_rate_hz": args.rate,
            "read_attempts": 9,
            "pmc_version": subprocess.check_output(["pmc", "-v"], text=True).strip(),
            "ethtool_timestamp_capabilities": ethtool,
            "reference": "experiment PHC; not UTC",
            "quality": "conditional engineering estimate, not certified accuracy",
        },
    }
    fd = os.open(phc, os.O_RDONLY | os.O_CLOEXEC)
    try:
        # Linux FD_TO_CLOCKID: read access is sufficient; no clock adjustment.
        clock_id = ((~fd) << 3) | 3
        with Path(args.output).open("x") as stream, tempfile.TemporaryDirectory(prefix="ib-clock-") as directory:
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            with PmcMonitor(args.pmc_socket, args.domain, directory) as monitor:
                next_sample = time.monotonic()
                deadline = next_sample + args.duration
                previous_status = None
                try:
                    while time.monotonic() < deadline:
                        status, raw, error, status_id = monitor.snapshot()
                        if status_id != previous_status:
                            stream.write(
                                json.dumps(
                                    {"kind": "ptp_status", "monotonic_ns": status_id, "raw_pmc": raw, "error": error}
                                )
                                + "\n"
                            )
                            previous_status = status_id
                        sample = ClockSample(**sample_clocks(clock_id), status=status)
                        stream.write(json.dumps({"kind": "sample", "sample": asdict(sample)}, sort_keys=True) + "\n")
                        stream.flush()
                        next_sample += 1 / args.rate
                        time.sleep(max(0, next_sample - time.monotonic()))
                except KeyboardInterrupt:
                    pass
    finally:
        os.close(fd)
    print(json.dumps(load_calibration(args.output).summary(), indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture", help="Read PHC and PMC; does not start PTP or set any clocks")
    collect.add_argument("--phc", required=True)
    collect.add_argument("--interface", required=True)
    collect.add_argument("--pmc-socket", required=True)
    collect.add_argument("--domain", type=int, required=True)
    collect.add_argument("--grandmaster", required=True, help="Pinned expected grandmaster clockIdentity")
    collect.add_argument("--session-id", required=True, help="Shared experiment ID; use a new ID after PTP restarts")
    collect.add_argument("--duration", type=float, default=60)
    collect.add_argument("--rate", type=float, default=50)
    collect.add_argument("--output", required=True, help="New JSONL path; never overwrites a capture")
    inspect = commands.add_parser("inspect", help="Summarize qualified/invalid clock mapping intervals")
    inspect.add_argument("source")
    args = parser.parse_args(argv)
    if args.command == "capture":
        capture(args)
    else:
        print(json.dumps(load_calibration(args.source).summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
