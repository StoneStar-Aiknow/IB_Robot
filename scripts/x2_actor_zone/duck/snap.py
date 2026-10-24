#!/usr/bin/env python3
"""S1：存头部相机的截图，给 zll 画站位区用。只订阅，不发任何东西。

用法（duck，域 0，source /root/x2-actor-run/env_actor.sh 之后）：
    python3 -u /root/x2-actor-run/snap.py <输出目录> [张数 20] [间隔秒 3]
每隔 <间隔> 秒存一张 half 流的原始 jpeg（不解码、不缩放），文件名 snap_<序号>_<相机 stamp>.jpg。
画面里有人脸：只留在板上/本机，不进 git。
"""

import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

TOPIC = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    node = Node("x2_actor_snap")
    latest = {}
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(CompressedImage, TOPIC, lambda m: latest.__setitem__("m", m), qos)
    saved = 0
    next_t = time.monotonic()
    deadline = time.monotonic() + count * every + 15.0
    try:
        while saved < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() < next_t or "m" not in latest:
                continue
            m = latest.pop("m")
            data = bytes(m.data)
            ext = "jpg" if data[:2] == b"\xff\xd8" else "bin"
            stamp = f"{m.header.stamp.sec}.{m.header.stamp.nanosec:09d}"
            path = os.path.join(out, f"snap_{saved:02d}_{stamp}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
            saved += 1
            print(f"{saved}/{count} {path} {len(data)} B format={m.format!r}", flush=True)
            next_t = time.monotonic() + every
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"saved {saved}/{count}")
    return 0 if saved == count else 1


if __name__ == "__main__":
    sys.exit(main())
