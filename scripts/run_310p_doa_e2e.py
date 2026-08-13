#!/usr/bin/env python3
"""310P 离线 wav DOA 端到端验证：构造 speech_direction 节点 + 订阅 /voice/speech_direction，
等 wav 实时回放处理完，查收到的方向消息 + doa_state 段历史。

用法（310P 上）：
    source /opt/ros/humble/setup.bash
    source <repo>/install/setup.bash
    cd <repo>
    python3 scripts/run_310p_doa_e2e.py \\
        --ros-args --params-file src/voice_asr_service/config/speech_direction.yaml

wav_path 由 yaml 注入；建议用含人声的 raw6ch wav（如 run_20260807-115633）。
返回码：0=收到方向消息；2=wav 处理完但无方向输出（可能 wav 无人声）；1=节点降级。

不走 ros2 CLI（DDS 发现易卡），同进程订阅更可靠。
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

# 孤立 ROS 域，避免被同机其他 ROS 节点干扰
os.environ.setdefault("ROS_DOMAIN_ID", "99")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import rclpy

from ibrobot_msgs.msg import SpeechDirection
from voice_asr_service.speech_direction.node import SpeechDirectionNode


def main() -> int:
    # wav_path 由 yaml 的 wav_path 参数注入（--ros-args --params-file）
    rclpy.init(args=sys.argv[1:])  # 透传 --ros-args 给 rclpy
    node = SpeechDirectionNode()
    if node._degraded:
        print(f"!!! 节点降级: {node._degrade_reason}", flush=True)
        rclpy.shutdown()
        return 1

    # 订阅节点自己发布的话题（同进程，绕开 ros2 topic echo 的 DDS 发现）
    received: list[SpeechDirection] = []

    def on_dir(msg: SpeechDirection) -> None:
        received.append(msg)
        deg = math.degrees(msg.azimuth_rad)
        print(
            f">>> 收到方向 seq={msg.seq_id} azimuth={deg:.1f}deg ({msg.azimuth_rad:.3f}rad) t={time.time():.2f}",
            flush=True,
        )

    node.create_subscription(SpeechDirection, "/voice/speech_direction", on_dir, 10)

    rt = node._runtime
    dur = len(node._wav_input.audio) / node._wav_input.sample_rate
    print(f"wav 时长 {dur:.1f}s, 实时回放约 {dur + 3:.0f}s 后处理完", flush=True)
    print(f"=== spin {dur + 8:.0f}s ===", flush=True)

    # 监控线程：每 2s 打印处理进度和当前 DOA 状态
    def mon() -> None:
        t0 = time.time()
        while time.time() - t0 < dur + 8:
            time.sleep(2)
            sp = getattr(rt.pipeline, "_samples_processed", 0)
            full = rt.doa_state.get_full()
            print(
                f"  t={time.time() - t0:.0f}s samples={sp}/{int(dur * 16000)} "
                f"is_speech={full.get('is_speech')} angle={full.get('angle')} "
                f"recv={len(received)}",
                flush=True,
            )
        print("=== 监控结束, shutdown ===", flush=True)
        rclpy.shutdown()

    threading.Thread(target=mon, daemon=True).start()

    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"spin 结束: {e}", flush=True)

    print("\n=== 最终结果 ===", flush=True)
    print(f"收到方向消息数: {len(received)}", flush=True)
    for m in received:
        print(f"  seq={m.seq_id} azimuth={math.degrees(m.azimuth_rad):.1f}deg ({m.azimuth_rad:.3f}rad)", flush=True)
    hist = getattr(rt.pipeline, "_history", [])
    print(f"pipeline._history 段数: {len(hist)}", flush=True)
    for h in hist[-5:]:
        print(f"  {h}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if received else 2


if __name__ == "__main__":
    raise SystemExit(main())
