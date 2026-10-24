"""Record the hand data path for the left/right comparison. Read only.

    python3 hand_rec.py <out dir> <group>

One process per group (a slow writer must not delay another topic's receipt);
each topic goes to <out dir>/<name>.jsonl, one line per message, "t" = receipt
time on this machine (time.time()), the clock every stage is compared on.

  frame    /hri/hand/frame                              perception, both people's hands
  preview  /motion_imitation_mock/hand_command_preview  retarget output, 20 values
           /mirror_session/hand_command_preview         session output after HandOverlay
  mc       /aima/hal/joint/hand/command                 what MC is commanding
           /aima/hal/joint/hand/state                   measured hands
  status   /motion_imitation_mock/status, /mirror_session/status, /mirror_session/state

20 values = robot left thumb_roll .. pinky_pip, then robot right.
"""

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)


def r(xs, nd=4):
    return [round(float(v), nd) for v in xs]


def stamp(h):
    s = getattr(h, "stamp", None)
    return None if s is None else int(s.sec) * 1_000_000_000 + int(s.nanosec)


def obs(o):
    return {
        "detected": bool(o.detected),
        "score": round(float(o.score), 4),
        "curl_rad": r(o.curl_rad),
        "curl_norm": r(o.curl_norm),
        "image_landmarks": r(o.image_landmarks, 1),
        "gesture": o.gesture,
        "gesture_score": round(float(o.gesture_score), 4),
    }


def hand_frame(m):
    return {
        "stamp_ns": stamp(m.header),
        "frame_id": int(m.frame_id),
        "person_valid": bool(m.person_valid),
        "bbox": r(m.person_bbox_xyxy, 1),
        "hands_in_frame": int(m.hands_in_frame),
        "latency_ms": round(float(m.latency_ms), 1),
        "left": obs(m.left),
        "right": obs(m.right),
    }


def array(m):
    return {"data": r(m.data)}


def string(m):
    try:
        return {"json": json.loads(m.data)}
    except ValueError:
        return {"data": m.data}


class Vendor:
    """aimdk HandCommandArray / HandStateArray: names once, then numbers per side."""

    def __init__(self):
        self.named = False

    def __call__(self, m):
        out = {"stamp_ns": stamp(m.header)}
        for side in ("left", "right"):
            js = getattr(m, f"{side}_hands")
            if not self.named:
                out[f"{side}_names"] = [j.name for j in js]
            out[side] = r([j.position for j in js])
            if hasattr(js[0] if js else None, "faultcode"):
                out[f"{side}_effort"] = r([j.effort for j in js], 3)
                out[f"{side}_state"] = [int(j.state) for j in js]
                out[f"{side}_fault"] = [int(j.faultcode) for j in js]
        self.named = True
        return out


GROUPS = {
    "frame": [("hand_frame", "/hri/hand/frame", "ibrobot_msgs/msg/HandFrame", hand_frame)],
    "preview": [
        ("retarget_preview", "/motion_imitation_mock/hand_command_preview", "std_msgs/msg/Float64MultiArray", array),
        ("session_preview", "/mirror_session/hand_command_preview", "std_msgs/msg/Float64MultiArray", array),
    ],
    "mc": [
        ("mc_command", "/aima/hal/joint/hand/command", "aimdk_msgs/msg/HandCommandArray", Vendor()),
        ("mc_state", "/aima/hal/joint/hand/state", "aimdk_msgs/msg/HandStateArray", Vendor()),
    ],
    "status": [
        ("retarget_status", "/motion_imitation_mock/status", "std_msgs/msg/String", string),
        ("session_status", "/mirror_session/status", "std_msgs/msg/String", string),
        ("session_state", "/mirror_session/state", "std_msgs/msg/String", string),
    ],
}


def main():
    out_dir, group = Path(sys.argv[1]), sys.argv[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node(f"x2_dexhand_rec_{group}")
    files, counts = [], {}
    for name, topic, typ, conv in GROUPS[group]:
        f = open(out_dir / f"{name}.jsonl", "a", buffering=1 << 16)
        files.append(f)
        counts[name] = 0

        def cb(m, f=f, conv=conv, name=name):
            rec = {"t": time.time(), **conv(m)}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts[name] += 1

        node.create_subscription(get_message(typ), topic, cb, QOS)
    last = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if time.monotonic() - last >= 30.0:
                last = time.monotonic()
                print(time.strftime("%H:%M:%S"), counts, flush=True)
                for f in files:
                    f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        for f in files:
            f.close()
        print("final", counts, flush=True)


if __name__ == "__main__":
    main()
