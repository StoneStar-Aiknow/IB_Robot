"""Isolated process entry point for the crash-prone mHandPro vendor SDK."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import select
import sys
import threading
import time
from pathlib import Path


def _load_sdk_module():
    module_path = Path(__file__).with_name("mhandpro_sdk.py")
    spec = importlib.util.spec_from_file_location("_mhandpro_sdk_isolated", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load mHandPro bindings from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _emit(stream, payload: dict) -> None:
    stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-path", required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), required=True)
    parser.add_argument("--failure-policy", choices=("require_all", "allow_available"), default="require_all")
    parser.add_argument("--event-fd", required=True, type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    events = os.fdopen(args.event_fd, "w", encoding="utf-8", buffering=1)
    sdk_module = _load_sdk_module()
    sdk = None
    frame_lock = threading.Lock()
    latest_frames: dict[str, dict] = {}
    sequences = {"left": 0, "right": 0}
    break_state = None

    def data_callback(right_data, left_data):
        now = time.monotonic()
        with frame_lock:
            for side, glove_data in (("right", right_data), ("left", left_data)):
                if not glove_data.isUpdate:
                    continue
                sequences[side] += 1
                latest_frames[side] = {
                    "type": "frame",
                    "side": side,
                    "sequence": sequences[side],
                    "timestamp": now,
                    "sdk_frame_index": int(glove_data.frameIndex),
                    "device_power": float(glove_data.devicePower),
                    "frequency": int(glove_data.frequency),
                    "positions": sdk_module.mocap_data_to_list(glove_data),
                    "quaternions": sdk_module.mocap_quaternions_to_list(glove_data),
                    "virtual_positions": sdk_module.mocap_virtual_positions_to_list(glove_data),
                    "sensor_states": sdk_module.mocap_sensor_states_to_list(glove_data),
                    "gyroscope": sdk_module.mocap_vectors_to_list(glove_data, "gyr"),
                    "accelerations": sdk_module.mocap_vectors_to_list(glove_data, "acc"),
                    "velocities": sdk_module.mocap_vectors_to_list(glove_data, "velocity"),
                }

    def break_callback(glove_mode):
        nonlocal break_state
        with frame_lock:
            break_state = int(glove_mode)

    try:
        sdk = sdk_module.MHandProSDK(args.lib_path)
        sdk.initial()
        sdk.set_break_callback(break_callback)
        sdk.set_data_with_virtual_callback(data_callback)
        requested_sides = ("left", "right") if args.side == "both" else (args.side,)

        state = sdk_module.CONNECTED_NONE
        for attempt in range(3):
            state = sdk.connect()
            if sdk_module.connection_satisfies_policy(state, requested_sides, args.failure_policy):
                break
            if attempt < 2:
                time.sleep(1.0)
        if not sdk_module.connection_satisfies_policy(state, requested_sides, args.failure_policy):
            raise ConnectionError(
                f"mHandPro connection state {state} does not satisfy {args.failure_policy} for {requested_sides}"
            )
        sdk.set_hand_dimension(True)
        sdk.set_tremor()
        _emit(events, {"type": "ready", "state": state, "sdk_version": "unknown"})

        emitted_sequences = {side: -1 for side in requested_sides}
        emitted_break_state = None
        running = True
        while running:
            with frame_lock:
                frames = {side: latest_frames.get(side) for side in requested_sides}
                current_break = break_state
            for side, frame in frames.items():
                if frame is not None and frame["sequence"] != emitted_sequences[side]:
                    _emit(events, frame)
                    emitted_sequences[side] = frame["sequence"]
            if current_break is not None and current_break != emitted_break_state:
                _emit(events, {"type": "break", "state": current_break})
                emitted_break_state = current_break

            readable, _, _ = select.select([sys.stdin], [], [], 0.005)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            command = json.loads(line)
            command_name = command.get("command")
            if command_name == "shutdown":
                running = False
            elif command_name == "calibrate":
                request_id = command.get("request_id")
                state, progress = sdk.start_calibration(
                    int(command["mode"]),
                    timeout=float(command["timeout"]),
                )
                _emit(
                    events,
                    {
                        "type": "calibration_result",
                        "request_id": request_id,
                        "state": state,
                        "progress": progress,
                    },
                )
            else:
                raise ValueError(f"Unknown mHandPro worker command: {command_name!r}")
        return 0
    except Exception as exc:
        _emit(events, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if sdk is not None:
            sdk.disconnect()
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
