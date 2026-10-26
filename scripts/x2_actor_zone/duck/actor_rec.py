"""Record the actor-select topics for offline analysis / replay. Read only.

    python3 actor_rec.py <out dir> <group>

Written after hand_rec.py: one process per group (a slow writer must not delay
another topic's receipt); each topic goes to <out dir>/<name>.jsonl, one line per
message, "t" = receipt time on this machine (time.time()).

  state  /motion_imitation_stream/actor_state       ibrobot_msgs/ActorState, every field
  cand   /motion_imitation_stream/actor_candidates  std_msgs/String, data is JSON

ActorState is walked by its own field list, so a field added to the message is
recorded without changing this file. Times (header.stamp, detections_stamp)
become integer ns: "stamp_ns", "detections_stamp_ns".
A candidates line that is not JSON is kept as {"raw": ..., "parse_error": ...}.
"""

import json
import math
import numbers
import sys
import time
from pathlib import Path

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)


def ns(t):
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


def num(v, nd=4):
    v = float(v)
    return round(v, nd) if math.isfinite(v) else v


def value(v):
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, numbers.Integral):  # also numpy ints
        return int(v)
    if isinstance(v, numbers.Real):  # also numpy float32 (array fields come back as numpy)
        return num(v)
    if hasattr(v, "sec") and hasattr(v, "nanosec"):
        return ns(v)
    if hasattr(v, "get_fields_and_field_types"):
        return fields(v)
    try:
        return [value(x) for x in v]
    except TypeError:
        return repr(v)


def fields(m):
    out = {}
    for name in m.get_fields_and_field_types():
        v = getattr(m, name)
        if name == "header":
            out["stamp_ns"] = ns(v.stamp)
            out["frame_id"] = v.frame_id
        elif hasattr(v, "sec") and hasattr(v, "nanosec"):
            out[f"{name}_ns"] = ns(v)
        else:
            out[name] = value(v)
    return out


def candidates(m):
    try:
        return {"msg": json.loads(m.data)}
    except ValueError as exc:
        return {"raw": m.data, "parse_error": f"{type(exc).__name__}: {exc}"}


GROUPS = {
    "state": [("actor_state", "/motion_imitation_stream/actor_state", "ibrobot_msgs/msg/ActorState", fields)],
    "cand": [("actor_candidates", "/motion_imitation_stream/actor_candidates", "std_msgs/msg/String", candidates)],
}


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in GROUPS:
        sys.exit(f"usage: python3 actor_rec.py <out dir> <{'|'.join(GROUPS)}>")
    out_dir, group = Path(sys.argv[1]), sys.argv[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node(f"x2_actor_rec_{group}")
    files, counts = [], {}
    for name, topic, typ, conv in GROUPS[group]:
        f = open(out_dir / f"{name}.jsonl", "a", buffering=1 << 16)
        files.append(f)
        counts[name] = 0

        def cb(m, f=f, conv=conv, name=name):
            t = time.time()
            try:
                rec = {"t": t, **conv(m)}
            except Exception as exc:  # never let one odd message stop the recorder
                rec = {"t": t, "convert_error": f"{type(exc).__name__}: {exc}"}
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
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
