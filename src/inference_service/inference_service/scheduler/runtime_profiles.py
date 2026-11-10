"""Runtime-execution profiles; deliberately separate from legacy global_proxy."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import time
from pathlib import Path

from .runtime_protocol import (
    DEFAULT_RUNTIME_ENDPOINT,
    MAX_INPUT_BUCKET_BYTES,
    RUNTIME_WORK_TYPES,
    ControlError,
    identifier,
)

MAX_PROFILE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_BUCKETS = 4096
MAX_PROFILE_SAMPLES = 1_000_000
PRODUCTION_MIN_SAMPLES = 10000
DEFAULT_MIN_SAMPLES = PRODUCTION_MIN_SAMPLES


def validate_profile_options(min_samples, max_age_days):
    if type(min_samples) is not int or min_samples < 20:
        raise ValueError("at least 20 warm profile samples are required")
    if type(max_age_days) is not int or max_age_days <= 0:
        raise ValueError("profile_max_age_days must be a positive integer")


class RuntimeProfileRegistry:
    def __init__(self, target, *, min_samples=DEFAULT_MIN_SAMPLES, max_age_days=30, now=time.time):
        validate_profile_options(min_samples, max_age_days)
        path = Path(target.profile_path)
        # O_NONBLOCK also protects the open against a path replaced by a FIFO.
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("runtime profile must be a regular file")
            if info.st_size > MAX_PROFILE_BYTES:
                raise ValueError("runtime profile exceeds size limit")
            payload = source.read(MAX_PROFILE_BYTES + 1)
        if len(payload) > MAX_PROFILE_BYTES:
            raise ValueError("runtime profile exceeds size limit")
        document = json.loads(payload)
        if set(document) != {"schema_version", "scope", "identity", "measured_at", "buckets"}:
            raise ValueError("invalid runtime profile fields")
        if document["schema_version"] != 2 or document["scope"] != "runtime_execution":
            raise ValueError(
                "runtime_execution v2 profile required; v1/global_proxy is not compatible; remeasure inputs"
            )
        expected = self.identity(target)
        if document["identity"] != expected:
            raise ValueError("runtime profile identity mismatch")
        measured = document["measured_at"]
        if (
            type(measured) not in (int, float)
            or not math.isfinite(measured)
            or not 0 <= now() - measured <= max_age_days * 86400
        ):
            raise ValueError("runtime profile is stale or future dated")
        self._measured_at = measured
        self._max_age_seconds = max_age_days * 86400
        self._now = now
        self.estimates = {}
        buckets = document["buckets"]
        if not isinstance(buckets, list) or not 1 <= len(buckets) <= MAX_PROFILE_BUCKETS:
            raise ValueError("invalid profile buckets")
        for bucket in buckets:
            if set(bucket) != {"work", "input_bucket", "samples", "upper_bound_ns"}:
                raise ValueError("invalid profile bucket fields")
            if (
                bucket["work"] not in RUNTIME_WORK_TYPES
                or not isinstance(bucket["input_bucket"], str)
                or len(bucket["input_bucket"].encode()) > MAX_INPUT_BUCKET_BYTES
            ):
                raise ValueError("invalid profile work or input bucket")
            if type(bucket["samples"]) is not int or bucket["samples"] < min_samples:
                raise ValueError("insufficient runtime profile samples")
            estimate = bucket["upper_bound_ns"]
            if type(estimate) is not int or estimate <= 0:
                raise ValueError("invalid runtime occupancy estimate")
            key = (bucket["work"], bucket["input_bucket"])
            if key in self.estimates:
                raise ValueError("duplicate runtime profile bucket")
            self.estimates[key] = estimate

    @staticmethod
    def identity(target):
        return {
            "profile_observation_inputs": list(target.profile_observation_inputs),
            **{
                name: getattr(target, name)
                for name in (
                    "deployment_fingerprint",
                    "composition_fingerprint",
                    "hardware_resource_id",
                    "physical_device",
                    "resource_mode",
                    "execution_kind",
                    "hardware_profile_fingerprint",
                    "runtime_version",
                )
            },
        }

    def check_freshness(self):
        if not 0 <= self._now() - self._measured_at <= self._max_age_seconds:
            raise ControlError("PROFILE_EXPIRED")

    def estimate(self, submission):
        self.check_freshness()
        value = self.estimates.get((submission.work, submission.input_bucket))
        if value is None:
            raise ControlError("PROFILE_INPUT_OUT_OF_RANGE")
        return value


def _profiling_output(output):
    output = Path(output).resolve()
    if (
        not any(output.parts[index : index + 2] == ("models", "_work") for index in range(len(output.parts) - 1))
        or "profiling" not in output.parts
    ):
        raise ValueError("profiling artifacts must live under models/_work/<bundle>/profiling")
    return output


def write_runtime_profile(target, samples, output, *, min_samples=DEFAULT_MIN_SAMPLES, margin=1.2):
    """Build a workload profile from successful Global occupancy samples.

    Callers collect warm measurements after startup. The tool never converts
    kernel-only or legacy proxy samples into runtime-execution estimates.
    Declared observation inputs must be sampled across representative and
    expensive contents, not just repeated frames. Every class uses its maximum
    observed occupancy plus margin; this is empirical, not a worst-case proof.
    """
    output = _profiling_output(output)
    if type(min_samples) is not int or min_samples < 20 or not math.isfinite(margin) or margin < 1:
        raise ValueError("at least 20 warm samples and a margin >= 1 are required")
    identity = RuntimeProfileRegistry.identity(target)
    groups = {}
    seen = set()
    measured_at = math.inf
    for index, sample in enumerate(samples):
        if index >= MAX_PROFILE_SAMPLES:
            raise ValueError("runtime profile sample limit exceeded")
        if sample["identity"] != identity or sample["target_id"] != target.target_id:
            raise ValueError("measurement identity mismatch")
        identifier(sample["operation_id"], "operation measurement")
        if sample["operation_id"] in seen:
            raise ValueError("duplicate operation measurement")
        seen.add(sample["operation_id"])
        duration = sample["occupancy_ns"]
        if type(duration) is not int or duration <= 0:
            raise ValueError("invalid measured occupancy")
        measured = sample["measured_at"]
        if type(measured) not in (int, float) or not math.isfinite(measured) or measured > time.time():
            raise ValueError("invalid measurement time")
        if sample["work"] not in RUNTIME_WORK_TYPES:
            raise ValueError("invalid profile work")
        bucket = sample["input_bucket"]
        if not isinstance(bucket, str) or len(bucket.encode()) > MAX_INPUT_BUCKET_BYTES:
            raise ValueError("invalid input bucket")
        key = (sample["work"], bucket)
        count, maximum = groups.get(key, (0, 0))
        groups[key] = (count + 1, max(maximum, duration))
        if len(groups) > MAX_PROFILE_BUCKETS:
            raise ValueError("too many workload buckets")
        measured_at = min(measured_at, measured)
    if not groups or any(count < min_samples for count, _ in groups.values()):
        raise ValueError("insufficient measurements for each workload bucket")
    document = {
        "schema_version": 2,
        "scope": "runtime_execution",
        "identity": identity,
        "measured_at": measured_at,
        "buckets": [
            {
                "work": work,
                "input_bucket": bucket,
                "samples": count,
                "upper_bound_ns": math.ceil(maximum * margin),
            }
            for (work, bucket), (count, maximum) in sorted(groups.items())
        ],
    }
    payload = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > MAX_PROFILE_BYTES:
        raise ValueError("runtime profile exceeds size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), output.stat().st_mode & 0o777 if output.exists() else 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def read_runtime_samples(path):
    """Stream arbitrarily many measurements while bounding each input record."""
    with Path(path).open("rb") as source:
        while line := source.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024:
                raise ValueError("sample record exceeds 1 MiB")
            if line.strip():
                yield json.loads(line)


def collect_runtime_samples(target, call, output, *, count=DEFAULT_MIN_SAMPLES):
    """Tail one target's new measurements; reject gaps and epoch changes.

    Warm up before starting this collector. Each run creates a new JSONL file;
    errors leave its collected prefix for diagnosis, never a completed profile.
    ``call`` is the existing request/reply transport, not a sampling backend.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("sample count must be positive")
    output = _profiling_output(output)
    description = call({"verb": "describe"})
    epoch = description["epoch"]
    # The fixed catalogue exposes one target per page. Validate before creating
    # a readiness file or waiting for samples that this epoch can never produce.
    for offset in range(description["target_count"]):
        page = description if offset == 0 else call({"verb": "describe", "offset": offset})
        if page["epoch"] != epoch:
            raise ControlError("EPOCH_MISMATCH")
        if any(entry["target_id"] == target.target_id for entry in page["targets"]):
            if page["targets"] != [target.to_dict()]:
                raise ValueError("target configuration mismatch")
            break
    else:
        raise ValueError(f"unknown target in Global catalogue: {target.target_id}")

    def read(cursor):
        return call({"verb": "measurements", "epoch": epoch, "after_sequence": cursor})

    cursor = read(0)["latest_sequence"]
    identity = RuntimeProfileRegistry.identity(target)
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        while written < count:
            page = read(cursor)
            if page["dropped"]:
                raise ValueError(f"measurement overrun: {page['dropped']} samples dropped; repeat collection")
            for sample in page["samples"]:
                if sample["target_id"] != target.target_id:
                    continue
                if sample["identity"] != identity:
                    raise ValueError("measurement identity mismatch")
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
                written += 1
                if written == count:
                    break
            stream.flush()
            cursor = page["next_sequence"]
            if not page["samples"]:
                time.sleep(0.05)
    return written


def collect_main():
    import argparse

    import rclpy

    from .runtime_protocol import RuntimeExecutionTarget
    from .runtime_ros import _reply, _request, _service_type

    parser = argparse.ArgumentParser(description="Continuously collect warm Global measurements for one target")
    parser.add_argument("--target", required=True, help="Resolved target JSON used by the Host and Global")
    parser.add_argument("--output", required=True, help="New models/_work/<bundle>/profiling/<samples>.jsonl")
    parser.add_argument("--endpoint", default=DEFAULT_RUNTIME_ENDPOINT)
    parser.add_argument("--count", type=int, default=DEFAULT_MIN_SAMPLES)
    args = parser.parse_args()
    target = RuntimeExecutionTarget(**json.loads(Path(args.target).read_text()))
    rclpy.init(args=[])
    node = rclpy.create_node("runtime_profile_collector")
    try:
        client = node.create_client(_service_type(), args.endpoint)
        if not client.wait_for_service(timeout_sec=10):
            raise TimeoutError("Global measurement service unavailable")

        def call(payload):
            future = client.call_async(_request(payload))
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if not future.done():
                client.remove_pending_request(future)
                raise TimeoutError("Global measurement response timed out")
            return _reply(future.result())

        written = collect_runtime_samples(target, call, args.output, count=args.count)
        print(f"Collected {written} samples for {target.target_id} into {args.output}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    import argparse

    from .runtime_protocol import RuntimeExecutionTarget

    parser = argparse.ArgumentParser(description="Build runtime_execution v2 profiles from warm Global measurements")
    parser.add_argument("--target", required=True, help="Resolved target JSON from robot_config")
    parser.add_argument("--samples", required=True, help="JSONL occupancy samples collected from Global")
    parser.add_argument("--output", required=True, help="models/_work/<bundle>/profiling/<profile>.json")
    parser.add_argument("--margin", type=float, default=1.2)
    parser.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="Match runtime_scheduler.profile_min_samples"
    )
    args = parser.parse_args()
    target = RuntimeExecutionTarget(**json.loads(Path(args.target).read_text()))
    write_runtime_profile(
        target, read_runtime_samples(args.samples), args.output, min_samples=args.min_samples, margin=args.margin
    )


if __name__ == "__main__":
    main()
