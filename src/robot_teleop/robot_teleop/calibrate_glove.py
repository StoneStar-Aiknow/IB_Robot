"""Interactive calibration CLI for mHandPro glove retargeting."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import queue
import statistics
import sys
import time
from pathlib import Path

from .devices.aero_compact_retarget import (
    THUMB_FLEX_SWEEP_LOWER_QUANTILE,
    THUMB_FLEX_SWEEP_UPPER_QUANTILE,
    THUMB_ROOT_SWEEP_LOWER_QUANTILE,
    THUMB_ROOT_SWEEP_UPPER_QUANTILE,
    build_aero_compact_calibration,
)
from .devices.aero_hand_retarget import (
    fit_task_space_thresholds,
    fit_thumb_quaternion_thresholds,
    validate_fitted_task_space,
)
from .devices.glove_calibration import (
    calibration_document,
    load_calibration,
    write_calibration_atomic,
    write_raw_capture_atomic,
)
from .devices.mhandpro_sdk import CS_SUCCEEDED
from .devices.mhandpro_source import RealMHandProSource, SharedRealMHandProSource
from .devices.mocap_retarget import (
    CHANNEL_NAMES,
    FEATURE_SCHEMA_AERO_COMPACT,
    extract_features,
    extract_thumb_kinematics,
)
from .hand_state import detect_open_frames


def _default_output(side: str) -> Path:
    return Path.home() / ".calibrate" / f"aero_hand_{side}_calibrate.json"


def _sides_for(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("left", "right")
    return (side,)


def _side_paths(side: str, path: Path | None, *, raw: bool = False) -> dict[str, Path]:
    """Resolve one output path per side without allowing ambiguous dual output files."""
    sides = _sides_for(side)
    if len(sides) == 1:
        return {sides[0]: path or _default_output(sides[0])}
    if path is None:
        directory = Path.home() / ".calibrate"
    else:
        if path.exists() and not path.is_dir():
            raise ValueError(f"--{'raw-output' if raw else 'output'} must be a directory with --side both")
        if path.suffix:
            raise ValueError(f"--{'raw-output' if raw else 'output'} must be a directory with --side both")
        directory = path
    filename = "aero_hand_{side}_capture.json" if raw else "aero_hand_{side}_calibrate.json"
    return {current_side: directory / filename.format(side=current_side) for current_side in sides}


def collect_frames(source, side: str, *, duration: float, minimum: int, timeout: float) -> list:
    """Collect positions from distinct callback frames (legacy compatibility wrapper)."""
    return [
        frame.positions
        for frame in collect_glove_frames(source, side, duration=duration, minimum=minimum, timeout=timeout)
    ]


def collect_glove_frames(source, side: str, *, duration: float, minimum: int, timeout: float) -> list:
    """Collect complete callback frames without blocking the SDK callback thread."""
    frames = []
    initial = source.latest_frame(side)
    seen_sequence = initial.sequence if initial is not None else -1
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        frame = source.latest_frame(side)
        if frame is not None and frame.sequence != seen_sequence:
            frames.append(frame)
            seen_sequence = frame.sequence
        if time.monotonic() - started >= duration and len(frames) >= minimum:
            return frames
        time.sleep(0.002)
    raise RuntimeError(f"Captured only {len(frames)} {side} frames; at least {minimum} are required")


def collect_glove_frames_multi(source, sides, *, duration: float, minimum: int, timeout: float) -> dict[str, list]:
    """Collect all requested sides in one shared time window."""
    requested = tuple(sides)
    if not requested or set(requested) != {"left", "right"}:
        raise ValueError("Dual capture requires both left and right sides")
    frames = {side: [] for side in requested}
    seen_sequences = {
        side: (frame.sequence if (frame := source.latest_frame(side)) is not None else -1) for side in requested
    }
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        for side in requested:
            frame = source.latest_frame(side)
            if frame is not None and frame.sequence != seen_sequences[side]:
                frames[side].append(frame)
                seen_sequences[side] = frame.sequence
        if time.monotonic() - started >= duration and all(len(values) >= minimum for values in frames.values()):
            return frames
        time.sleep(0.002)
    counts = ", ".join(f"{side}={len(frames[side])}" for side in requested)
    raise RuntimeError(f"Captured insufficient dual-hand frames ({counts}); each side needs at least {minimum}")


def _persistence_probe_worker(queue, lib_path: str, side: str, duration: float, minimum: int, timeout: float):
    """Load the vendor library in a fresh interpreter and return held-pose features."""
    probe_source = RealMHandProSource(lib_path, side)
    try:
        probe_source.connect()
        frames = collect_glove_frames(probe_source, side, duration=duration, minimum=minimum, timeout=timeout)
        queue.put(("ok", _persistence_features(frames, side)))
    except Exception as exc:
        queue.put(("error", str(exc)))
    finally:
        probe_source.disconnect()


def verify_p_pose_persistence(source, side: str, args) -> list[float]:
    """Reload the SDK in a fresh process and compare a held P-pose."""
    print("Keep the P-pose stable while cross-process persistence is checked automatically.")
    before = _persistence_features(
        collect_glove_frames(
            source,
            side,
            duration=args.duration,
            minimum=args.minimum_frames,
            timeout=args.timeout,
        ),
        side,
    )
    source.disconnect()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_persistence_probe_worker,
        args=(result_queue, args.lib_path, side, args.duration, args.minimum_frames, args.timeout),
    )
    process.start()
    process.join(args.timeout + 10.0)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Cross-process P-pose persistence probe timed out")
    try:
        status, result = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(f"Cross-process P-pose persistence probe exited with code {process.exitcode}") from exc
    finally:
        result_queue.close()
    if status != "ok":
        raise RuntimeError(f"Cross-process P-pose persistence probe failed: {result}")
    after = result
    deltas = [abs(first - second) for first, second in zip(before, after, strict=True)]
    max_delta = max(deltas)
    if max_delta > math.radians(args.persistence_tolerance_deg):
        raise RuntimeError(
            "P-pose calibration did not survive an SDK process restart: "
            f"maximum feature delta is {math.degrees(max_delta):.1f} degrees"
        )
    print(f"P-pose persistence verified (max feature delta {math.degrees(max_delta):.1f} degrees).")
    source.connect()
    return deltas


def _persistence_features(frames, side: str) -> list[float]:
    """Include SDK virtual-tip geometry in the cross-process comparison."""
    rows = []
    for frame in frames:
        thumb = extract_thumb_kinematics(frame.positions, frame.virtual_positions, side)
        rows.append(
            [
                *extract_features(frame.positions, side),
                thumb.root_yaw,
                thumb.root_pitch,
                thumb.mcp_flex,
                thumb.ip_flex,
            ]
        )
    return [statistics.median(row[index] for row in rows) for index in range(len(rows[0]))]


def run_runtime_p_pose(service_name: str, side: str, timeout: float) -> int:
    """Align the running SDK worker; reusable mapping ranges stay on disk."""
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger

    rclpy.init(args=[])
    node = Node(f"mhandpro_{side}_runtime_calibration")
    try:
        client = node.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"Runtime calibration service is unavailable: {service_name}")
        input(
            "Make a P-pose: arms level and forward, palms down, wrists straight, all fingers straight, "
            "and each thumb straight about 45 degrees from the index finger. Press Enter when stable: "
        )
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout + 5.0)
        if not future.done():
            raise RuntimeError("Runtime P-pose service timed out")
        response = future.result()
        if response is None or not response.success:
            message = response.message if response is not None else "service returned no response"
            raise RuntimeError(f"Runtime P-pose failed: {message}")
        print(response.message)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate an mHandPro glove for Aero Hand retargeting")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--lib-path", help="External libVDMocapSDK_mHandPro.so path")
    parser.add_argument(
        "--runtime-service",
        help="Calibrate the SDK worker owned by a running teleoperation node instead of rebuilding mapping ranges",
    )
    parser.add_argument(
        "--runtime-neutral-service",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Calibration JSON destination; with --side both, use a directory for both side files",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw capture path; with --side both, use a directory for both side files",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--duration", type=float, default=0.8, help=argparse.SUPPRESS)
    parser.add_argument("--sweep-duration", type=float, default=15.0, help="Continuous full-range sweep duration")
    parser.add_argument("--minimum-frames", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--sdk-calibration-timeout", type=float, default=30.0)
    parser.add_argument("--persistence-tolerance-deg", type=float, default=10.0)
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "--task-space-only",
        action="store_true",
        help="Fit task-space shape ranges into an existing calibration without repeating P-pose",
    )
    update_group.add_argument(
        "--aero-compact-only",
        "--sdk-skeleton-only",
        dest="aero_compact_only",
        action="store_true",
        help="Fit Aero compact geometry from complete virtual-tip frames without repeating P-pose",
    )
    parser.add_argument(
        "--skip-p-pose-persistence-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-p-pose-persistence",
        action="store_true",
        help="Diagnostic check for SDK state across process restart; not required with runtime calibration",
    )
    args = parser.parse_args(argv)
    if not args.runtime_service and not args.lib_path:
        parser.error("--lib-path is required unless --runtime-service is used")
    if args.duration <= 0 or args.sweep_duration < 5.0 or args.minimum_frames <= 0 or args.timeout <= 0:
        parser.error("capture duration, frame count, and timeout must be positive")
    if args.persistence_tolerance_deg <= 0:
        parser.error("persistence tolerance must be positive")
    if args.side == "both" and args.verify_p_pose_persistence:
        parser.error("--verify-p-pose-persistence is currently supported only for one side")
    if args.side == "both":
        for option, path in (("--output", args.output), ("--raw-output", args.raw_output)):
            if path is not None and ((path.exists() and not path.is_dir()) or path.suffix):
                parser.error(f"{option} must be a directory with --side both")
    return args


def capture_open_and_sweep(source, side: str, args):
    """Capture one unlabeled sweep and infer its stable open-hand frames."""
    input(
        f"After Enter, move continuously for {args.sweep_duration:.0f} seconds through your full comfortable "
        "hand range. Repeatedly make a relaxed open hand and a natural fist without forcing any joint to its "
        "end range; spread, oppose and curl the thumb; and include a few brief relaxed open moments. "
        "No pose order is required. Press Enter to start: "
    )
    sweep_captured = collect_glove_frames(
        source,
        side,
        duration=args.sweep_duration,
        minimum=max(args.minimum_frames, int(args.sweep_duration * 20.0)),
        timeout=args.sweep_duration + args.timeout,
    )
    open_captured = detect_open_frames(sweep_captured, side, minimum_frames=args.minimum_frames)
    print(
        f"Automatically selected {len(open_captured)} stable open-hand frames from {len(sweep_captured)} sweep frames."
    )
    return open_captured, sweep_captured


def capture_open_and_sweep_multi(source, args):
    """Capture both hands concurrently and infer each side's stable open frames."""
    input(
        f"After Enter, move both hands continuously for {args.sweep_duration:.0f} seconds through your full "
        "comfortable range. Repeatedly make relaxed open hands and natural fists without forcing any joint "
        "to its end range; spread, oppose and curl both thumbs; and include brief relaxed open moments. "
        "No pose order is required. Press Enter to start: "
    )
    captured = collect_glove_frames_multi(
        source,
        ("left", "right"),
        duration=args.sweep_duration,
        minimum=max(args.minimum_frames, int(args.sweep_duration * 20.0)),
        timeout=args.sweep_duration + args.timeout,
    )
    result = {}
    for side, sweep_captured in captured.items():
        open_captured = detect_open_frames(sweep_captured, side, minimum_frames=args.minimum_frames)
        print(
            f"{side}: automatically selected {len(open_captured)} stable open-hand frames "
            f"from {len(sweep_captured)} sweep frames."
        )
        result[side] = (open_captured, sweep_captured)
    return result


def fit_captured_task_space(open_captured, sweep_captured, side: str) -> dict:
    """Fit and validate all task-space ranges available in a complete capture."""
    thresholds = fit_task_space_thresholds(
        [frame.positions for frame in open_captured],
        [frame.positions for frame in sweep_captured],
        side,
    )
    all_frames = [*open_captured, *sweep_captured]
    if all(frame.quaternions is not None for frame in all_frames):
        thresholds.update(fit_thumb_quaternion_thresholds(open_captured, sweep_captured))
    validate_fitted_task_space(thresholds)
    return thresholds


def update_task_space_calibration(source, side: str, output: Path, args) -> int:
    """Add task-space ranges to an existing verified calibration."""
    load_calibration(output, side, require_persistence=False)
    document = json.loads(output.read_text(encoding="utf-8"))
    open_captured, sweep_captured = capture_open_and_sweep(source, side, args)
    if args.raw_output is not None:
        written_raw = write_raw_capture_atomic(
            args.raw_output,
            side,
            source.sdk_version,
            {"open": open_captured, "sweep": sweep_captured},
        )
        print(f"Wrote raw capture: {written_raw}")
    document["task_space"] = fit_captured_task_space(open_captured, sweep_captured, side)
    acquisition = document.setdefault("acquisition", {})
    acquisition["task_space_method"] = "automatic_open_from_free_sweep"
    acquisition["task_space_sweep_duration_s"] = args.sweep_duration
    written = write_calibration_atomic(output, document)
    print(f"Wrote task-space ranges: {written}")
    return 0


def _update_task_space_from_capture(output: Path, side: str, open_captured, sweep_captured, args) -> Path:
    """Update one existing calibration using frames captured in a shared window."""
    load_calibration(output, side, require_persistence=False)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["task_space"] = fit_captured_task_space(open_captured, sweep_captured, side)
    acquisition = document.setdefault("acquisition", {})
    acquisition["task_space_method"] = "automatic_open_from_free_sweep"
    acquisition["task_space_sweep_duration_s"] = args.sweep_duration
    return write_calibration_atomic(output, document)


def update_aero_compact_calibration(source, side: str, output: Path, args) -> int:
    """Fit Aero compact neutral geometry without repeating P-pose."""
    load_calibration(output, side, require_persistence=False)
    document = json.loads(output.read_text(encoding="utf-8"))
    open_captured, sweep_captured = capture_open_and_sweep(source, side, args)
    if args.raw_output is not None:
        written_raw = write_raw_capture_atomic(
            args.raw_output,
            side,
            source.sdk_version,
            {"open": open_captured, "sweep": sweep_captured},
        )
        print(f"Wrote complete SDK geometry capture: {written_raw}")
    endpoints = build_aero_compact_calibration(open_captured, sweep_captured, side)
    document["low"] = endpoints["low"]
    document["high"] = endpoints["high"]
    document["feature_schema"] = FEATURE_SCHEMA_AERO_COMPACT
    document["thumb_endpoints"] = endpoints["thumb_endpoints"]
    document["finger_endpoints"] = endpoints["finger_endpoints"]
    document.pop("thumb_neutral", None)
    acquisition = document.setdefault("acquisition", {})
    acquisition.update(
        {
            "method": "automatic_open_from_free_sweep",
            "automatic_open_frame_count": len(open_captured),
            "sweep_duration_s": args.sweep_duration,
            "robust_quantiles": [THUMB_FLEX_SWEEP_LOWER_QUANTILE, THUMB_FLEX_SWEEP_UPPER_QUANTILE],
            "thumb_root_quantiles": [THUMB_ROOT_SWEEP_LOWER_QUANTILE, THUMB_ROOT_SWEEP_UPPER_QUANTILE],
            "thumb_flex_quantiles": [THUMB_FLEX_SWEEP_LOWER_QUANTILE, THUMB_FLEX_SWEEP_UPPER_QUANTILE],
            "finger_active_trim_fraction": 0.08,
        }
    )
    written = write_calibration_atomic(output, document)
    print(f"Wrote Aero compact calibration: {written}")
    return 0


def _aero_compact_document(
    side: str,
    open_captured,
    sweep_captured,
    *,
    sdk_version: str,
    persistence_verified: bool,
    existing: dict | None = None,
    persistence_deltas: list[float] | None = None,
) -> dict:
    endpoints = build_aero_compact_calibration(open_captured, sweep_captured, side)
    acquisition = dict(existing.get("acquisition", {})) if existing is not None else {}
    acquisition.update(
        {
            "method": "automatic_open_from_free_sweep",
            "automatic_open_frame_count": len(open_captured),
            "sweep_duration_s": None,
            "robust_quantiles": [THUMB_FLEX_SWEEP_LOWER_QUANTILE, THUMB_FLEX_SWEEP_UPPER_QUANTILE],
            "thumb_root_quantiles": [THUMB_ROOT_SWEEP_LOWER_QUANTILE, THUMB_ROOT_SWEEP_UPPER_QUANTILE],
            "thumb_flex_quantiles": [THUMB_FLEX_SWEEP_LOWER_QUANTILE, THUMB_FLEX_SWEEP_UPPER_QUANTILE],
            "finger_active_trim_fraction": 0.08,
        }
    )
    if persistence_deltas is not None:
        acquisition["reconnect_feature_deltas_rad"] = persistence_deltas
    if existing is None:
        return calibration_document(
            side,
            endpoints["low"],
            endpoints["high"],
            sdk_version=sdk_version,
            persistence_verified=persistence_verified,
            feature_schema=FEATURE_SCHEMA_AERO_COMPACT,
            thumb_endpoints=endpoints["thumb_endpoints"],
            finger_endpoints=endpoints["finger_endpoints"],
            acquisition=acquisition,
        )
    existing["low"] = endpoints["low"]
    existing["high"] = endpoints["high"]
    existing["feature_schema"] = FEATURE_SCHEMA_AERO_COMPACT
    existing["thumb_endpoints"] = endpoints["thumb_endpoints"]
    existing["finger_endpoints"] = endpoints["finger_endpoints"]
    existing.pop("thumb_neutral", None)
    existing["acquisition"] = acquisition
    return existing


def _write_aero_compact_from_capture(
    source,
    side: str,
    output: Path,
    open_captured,
    sweep_captured,
    args,
    *,
    update_existing: bool,
) -> Path:
    existing = None
    if update_existing:
        load_calibration(output, side, require_persistence=False)
        existing = json.loads(output.read_text(encoding="utf-8"))
    document = _aero_compact_document(
        side,
        open_captured,
        sweep_captured,
        sdk_version=source.sdk_version,
        persistence_verified=False,
        existing=existing,
    )
    document["acquisition"]["sweep_duration_s"] = args.sweep_duration
    written = write_calibration_atomic(output, document)
    print(f"Wrote Aero compact calibration: {written}")
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.runtime_service:
        try:
            return run_runtime_p_pose(
                args.runtime_service,
                args.side,
                args.sdk_calibration_timeout,
            )
        except (RuntimeError, KeyboardInterrupt) as exc:
            print(f"Runtime calibration failed: {exc}", file=sys.stderr)
            return 1
    sides = _sides_for(args.side)
    outputs = _side_paths(args.side, args.output)
    raw_outputs = _side_paths(args.side, args.raw_output, raw=True) if args.raw_output is not None else None
    source = (
        SharedRealMHandProSource(args.lib_path, sides)
        if args.side == "both"
        else RealMHandProSource(args.lib_path, args.side)
    )
    try:
        source.connect()
        _calibrate_connected_source(source, args)

        persistence_verified = False
        persistence_deltas = [0.0] * len(CHANNEL_NAMES)
        if args.verify_p_pose_persistence and not args.skip_p_pose_persistence_check:
            persistence_deltas = verify_p_pose_persistence(source, args.side, args)
            persistence_verified = True

        captures = (
            capture_open_and_sweep_multi(source, args)
            if args.side == "both"
            else {args.side: capture_open_and_sweep(source, args.side, args)}
        )
        for side in sides:
            open_captured, sweep_captured = captures[side]
            print(f"{side}: captured {len(open_captured)} open-reference frames.")
            print(f"{side}: captured {len(sweep_captured)} full-range sweep frames.")
            if raw_outputs is not None:
                written_raw = write_raw_capture_atomic(
                    raw_outputs[side],
                    side,
                    source.sdk_version,
                    {"open": open_captured, "sweep": sweep_captured},
                )
                print(f"{side}: wrote raw capture: {written_raw}")

            if args.task_space_only:
                written = _update_task_space_from_capture(outputs[side], side, open_captured, sweep_captured, args)
                print(f"{side}: wrote task-space ranges: {written}")
                continue

            written = _write_aero_compact_from_capture(
                source,
                side,
                outputs[side],
                open_captured,
                sweep_captured,
                args,
                update_existing=args.aero_compact_only,
            )
            if not args.aero_compact_only:
                document = json.loads(outputs[side].read_text(encoding="utf-8"))
                document["sdk"]["p_pose_cross_process_persistence_verified"] = persistence_verified
                document["acquisition"]["reconnect_feature_deltas_rad"] = persistence_deltas
                written = write_calibration_atomic(outputs[side], document)
            print(f"{side}: wrote calibration: {written}")

        if not persistence_verified:
            print(
                "This mapping is reusable, but real teleoperation must complete runtime P-pose before output unlocks."
            )
        return 0
    except (ConnectionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCalibration cancelled.", file=sys.stderr)
        return 130
    finally:
        source.disconnect()


def _calibrate_connected_source(source, args) -> None:
    input(
        "Make a P-pose: arms level and forward, palms down, wrists straight, all fingers straight, "
        "and each thumb straight about 45 degrees from the index finger. Press Enter when stable: "
    )
    state, progress = source.calibrate_p_pose(args.sdk_calibration_timeout)
    if state != CS_SUCCEEDED:
        raise RuntimeError(f"mHandPro P-pose calibration failed (state={state}, progress={progress:.1f})")
    print("P-pose SDK calibration succeeded for this SDK process.")


if __name__ == "__main__":
    raise SystemExit(main())
