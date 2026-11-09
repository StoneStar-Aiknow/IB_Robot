"""Read-only, ROS-free registration of a separately certified fixed hold.

The deployment owns the three input paths. Both launch and runtime reconstruct
the same private profile overlay; the shared profile and assets are never written.
Synthetic tests of this reader are not physical robot geometry certificates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

from aimdk_robot.preset_geometry import NamedHoldGeometryGuard
from aimdk_robot.projection import DEXTEROUS_JOINT_ORDER, VENDOR_ARM_ORDER
from aimdk_robot.trajectory import load_trajectory

MOTION_NAME = "overhead_heart"
HAND_ORDER = tuple(f"{side}_{joint}_joint" for side in ("L", "R") for joint in DEXTEROUS_JOINT_ORDER)
NATIVE_ORDER = VENDOR_ARM_ORDER + HAND_ORDER
_KEYS = {"motion_name", "template_file", "geometry_file"}


def _finite(values, count, label):
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
    ):
        raise ValueError(f"named hold {label} requires {count} finite native radian values")
    return list(map(float, values))


def _file(value, label, *, relative_to=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"named hold {label} is required")
    path = Path(value)
    if relative_to is not None and not path.is_absolute():
        path = relative_to / path
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"named hold {label} must resolve to an existing absolute file")
    return path.resolve()


def register_named_hold(profile, payload_json):
    """Return a private overlay, or the identical profile for an empty opt-in.

    The geometry binds the visual version, exact native order and target digest.
    It is still checked mathematically by the runtime guard before registration.
    No general profile overrides, mode widening or path generation are accepted.
    """
    if not payload_json:
        return profile
    payload = json.loads(payload_json)
    if not isinstance(payload, dict) or set(payload) != _KEYS or payload["motion_name"] != MOTION_NAME:
        raise ValueError("named hold registration requires the fixed name and exactly three deployment fields")
    template_path = _file(payload["template_file"], "template_file")
    geometry_path = _file(payload["geometry_file"], "geometry_file")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or template.get("name") != MOTION_NAME:
        raise ValueError("named hold template identity mismatch")
    if (
        type(template.get("schema_version")) is not int
        or template.get("schema_version") != 1
        or template.get("units") != "rad"
    ):
        raise ValueError("named hold template requires schema 1 and native radian units")
    if (
        template.get("validated") is not True
        or template.get("motion_enabled") is not True
        or template.get("stop_bridge_certified") is not True
    ):
        raise ValueError("named hold requires a validated template and certified stop bridge")
    approval = template.get("visual_approval")
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        raise ValueError("named hold requires independent visual approval")
    candidate_sha256 = approval.get("source_candidate_sha256")
    if not isinstance(candidate_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", candidate_sha256) is None:
        raise ValueError("named hold visual approval must bind its source candidate digest")
    version = template.get("visual_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("named hold visual version is missing")
    if template.get("arm_joint_names") != list(VENDOR_ARM_ORDER) or template.get("hand_joint_names") != list(
        HAND_ORDER
    ):
        raise ValueError("named hold template must preserve the exact 14-arm plus 20-hand native order")
    arms = _finite(template.get("arm_target_rad"), 14, "arm target")
    hands = _finite(template.get("hand_target_rad"), 20, "hand target")
    target = arms + hands
    identity = hashlib.sha256(json.dumps(target, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if not isinstance(geometry, dict) or geometry.get("name") != MOTION_NAME:
        raise ValueError("named hold geometry identity mismatch")
    if (
        geometry.get("visual_version") != version
        or geometry.get("native_joint_names") != list(NATIVE_ORDER)
        or geometry.get("native_target_sha256") != identity
    ):
        raise ValueError("named hold geometry does not bind the approved native pose identity")
    for record in (template, geometry):
        clearance = record.get("required_head_clearance_m")
        if type(clearance) not in (int, float) or not math.isfinite(clearance) or clearance < 0.20:
            raise ValueError("named hold requires the unchanged 0.20 m head clearance")
    guard = NamedHoldGeometryGuard(geometry)
    if guard.invalid_reason:
        raise ValueError(f"named hold geometry is not certified: {guard.invalid_reason}")
    ok, reason = guard.validate_path(name=MOTION_NAME, times=[0.0], positions=[target])
    if not ok:
        raise ValueError(f"named hold target geometry rejected: {reason}")

    vendor = profile.get("vendor") or {}
    config = vendor.get("named_motion") or {}
    capabilities = profile.get("capabilities") or {}
    named = capabilities.get("motion.named")
    if not isinstance(named, dict) or profile.get("arm_joints") != list(VENDOR_ARM_ORDER):
        raise ValueError("named hold requires the existing named capability and native arm order")
    existing = set(vendor.get("preset_motions") or {}) | set(vendor.get("postures") or {})
    existing |= set(config.get("trajectories") or {}) | set(named.get("names") or [])
    existing |= set(config.get("cancel_return_motions") or [])
    existing |= set((config.get("display_completion") or {}).get("motions") or [])
    if MOTION_NAME in existing or config.get("hold_motions") or config.get("hold_geometry_file"):
        raise ValueError("named hold registration conflicts with an existing motion or hold geometry owner")
    # model_inference is the deployment control mode; its runtime mode is the
    # existing policy_stream mode. Never add a general replay mode for this hold.
    if "policy_stream" not in profile.get("modes", {}) or "policy_stream" not in config.get("trajectory_modes", []):
        raise ValueError("named hold requires the existing policy_stream replay mode")
    blend = config.get("trajectory_blend_speed_radps", 0.5)
    maximum = config.get("trajectory_max_speed_radps", 3.0)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (blend, maximum)):
        raise ValueError("named hold requires finite positive existing arm bridge speeds")
    if any(speed < min(blend, maximum) for speed in guard.speed[:14]):
        raise ValueError("named hold geometry speed bounds cannot support the unchanged arm bridge")
    arm_limits = (vendor.get("arm") or {}).get("limits") or {}
    for index, name in enumerate(VENDOR_ARM_ORDER):
        band = _finite(arm_limits.get(name), 2, f"{name} limits")
        if band[0] >= band[1] or guard.lower[index] < band[0] or guard.upper[index] > band[1]:
            raise ValueError("named hold geometry cannot widen the runtime's native arm limits")

    recording_path = _file(template.get("recording_file"), "recording_file", relative_to=template_path.parent)
    recording_bytes = recording_path.read_bytes()
    raw = json.loads(recording_bytes)
    if (
        not isinstance(raw, dict)
        or raw.get("format") != "ibrobot.fixed_motion.delivery"
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("validated") is not True
    ):
        raise ValueError("named hold requires a validated full arm-and-hand delivery recording")
    if (
        raw.get("name") != MOTION_NAME
        or raw.get("visual_version") != version
        or raw.get("native_target_sha256") != identity
        or raw.get("source_candidate_sha256") != candidate_sha256
    ):
        raise ValueError("named hold recording identity mismatch")
    raw_arms, raw_hands = raw.get("arms") or {}, raw.get("hands") or {}
    expected_hand_order = [f"{side}_{joint}" for side in ("left", "right") for joint in DEXTEROUS_JOINT_ORDER]
    if raw_arms.get("joints") != list(VENDOR_ARM_ORDER) or raw_hands.get("joints") != expected_hand_order:
        raise ValueError("named hold recording must preserve the exact native arm-and-hand order")
    times = raw.get("time_from_start_s")
    if not isinstance(times, list) or len(times) < 2:
        raise ValueError("named hold recording needs at least two fixed samples")
    _finite(times, len(times), "recording times")
    for values, count, expected in ((raw_arms.get("positions_rad"), 14, arms), (raw_hands.get("positions"), 20, hands)):
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError("named hold recording sample counts differ")
        for row in values:
            if _finite(row, count, "recording sample") != expected:
                raise ValueError("named hold recording differs from the frozen approved target")
    recording = load_trajectory(recording_path, max_speed_radps=maximum)
    if (
        recording.name != MOTION_NAME
        or recording.target != "both"
        or any(list(point) != arms for point in recording.positions)
        or any(list(point) != hands for point in recording.hands)
    ):
        raise ValueError("named hold recording must name the fixed both-hand target")

    result = copy.deepcopy(profile)
    output = result["vendor"].setdefault("named_motion", {})
    output.setdefault("trajectories", {})[MOTION_NAME] = str(recording_path)
    output["hold_motions"] = [MOTION_NAME]
    output["hold_geometry_file"] = str(geometry_path)
    output["hold_registered_native_target"] = target
    output["hold_registered_geometry_sha256"] = geometry["sha256"]
    output["hold_registered_recording_sha256"] = hashlib.sha256(recording_bytes).hexdigest()
    hand_limits = [[float(low), float(high)] for low, high in zip(guard.lower[14:], guard.upper[14:], strict=True)]
    existing_limits = config.get("hold_hand_limits", hand_limits)
    if not isinstance(existing_limits, list) or len(existing_limits) != 20:
        raise ValueError("existing named hold hand limits must contain 20 native bands")
    for index, band in enumerate(existing_limits):
        low, high = _finite(band, 2, "existing hand limits")
        hand_limits[index] = [max(low, hand_limits[index][0]), min(high, hand_limits[index][1])]
        if low > high or not hand_limits[index][0] <= hands[index] <= hand_limits[index][1]:
            raise ValueError("named hold target exceeds existing or certified native hand limits")
    output["hold_hand_limits"] = hand_limits
    hand_speeds = config.get("hold_hand_speed_radps", [float(speed) for speed in guard.speed[14:]])
    if type(hand_speeds) in (int, float):
        hand_speeds = [hand_speeds] * 20
    hand_speeds = _finite(hand_speeds, 20, "hand speeds")
    if any(speed <= 0 for speed in hand_speeds):
        raise ValueError("named hold hand speeds must be positive")
    output["hold_hand_speed_radps"] = [
        min(speed, float(bound)) for speed, bound in zip(hand_speeds, guard.speed[14:], strict=True)
    ]
    for key, default in (
        ("hold_lease_timeout_s", 1.0),
        ("hold_max_duration_s", 60.0),
        ("hold_release_timeout_s", 0.3),
        ("hold_feedback_skew_s", 0.04),
    ):
        value = output.setdefault(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"named hold {key} must be finite and positive")
    if output["hold_lease_timeout_s"] > 1.0 or output["hold_max_duration_s"] > 60.0:
        raise ValueError("named hold cannot widen the lease or absolute duration cap")
    interfaces = result.setdefault("interfaces", {})
    for interface_id, key, topic, direction in (
        ("motion.named_hold_lease", "hold_lease_topic", "/runtime/named_motion_hold_lease", "subscribe"),
        ("motion.named_hold_lease_ack", "hold_lease_ack_topic", "/runtime/named_motion_hold_lease_ack", "publish"),
    ):
        endpoint = output.setdefault(key, topic)
        if (
            not isinstance(endpoint, str)
            or re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", endpoint) is None
        ):
            raise ValueError("named hold lease endpoints must be absolute ROS names")
        spec = {
            "capability": "motion.named",
            "kind": "topic",
            "direction": direction,
            "endpoint": endpoint,
            "message_type": "std_msgs/msg/String",
            "qos": {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10},
        }
        if interface_id in interfaces and interfaces[interface_id] != spec:
            raise ValueError("named hold lease interface conflicts with the runtime profile")
        if any(item.get("endpoint") == endpoint for name, item in interfaces.items() if name != interface_id):
            raise ValueError("named hold lease endpoint conflicts with another interface")
        interfaces[interface_id] = spec
    result["capabilities"]["motion.named"].setdefault("names", []).append(MOTION_NAME)
    return result


def verify_loaded_named_hold(profile, recordings, guard):
    """Pin the final readers to the assets reviewed during registration.

    The action server and guard each read their asset after registration. This
    check runs before the runtime joins its executor, so a concurrent deployment
    cannot replace a fixed pose between those reads and open goal admission.
    """
    config = (profile.get("vendor") or {}).get("named_motion") or {}
    expected = config.get("hold_registered_native_target")
    if expected is None:
        return
    expected = _finite(expected, 34, "registered target")
    recording = recordings.get(MOTION_NAME)
    if (
        recording is None
        or recording.name != MOTION_NAME
        or recording.target != "both"
        or not recording.positions
        or not recording.hands
        or any(list(point) != expected[:14] for point in recording.positions)
        or any(list(point) != expected[14:] for point in recording.hands)
        or len(recording.positions) != len(recording.hands)
    ):
        raise ValueError("named hold recording changed between registration and final loading")
    recording_path = config["trajectories"][MOTION_NAME]
    if hashlib.sha256(Path(recording_path).read_bytes()).hexdigest() != config.get("hold_registered_recording_sha256"):
        raise ValueError("named hold recording asset changed after registration")
    if (
        guard is None
        or guard.invalid_reason
        or guard.record.get("sha256") != config.get("hold_registered_geometry_sha256")
    ):
        raise ValueError("named hold geometry changed between registration and final loading")
