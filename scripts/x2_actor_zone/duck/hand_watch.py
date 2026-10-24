"""Watch the dexterous hands in the mirror. Read only.

Every 5 s one line from the two status topics:
  retarget node /motion_imitation_mock/status  "hands": perception frames,
      detections per side, hand age, stale flags, gesture modes;
  session      /mirror_session/status          "hands": following ticks,
      start source, unjams, hand feedback age / lost, last 20 values
      (left thumb roll .. right pinky pip).
Plus the session state, and the hand perception rate measured here.
"""

import json
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from ibrobot_msgs.msg import HandFrame

PERIOD_S = 5.0


def main():
    rclpy.init()
    node = rclpy.create_node("x2_dexhand_hand_watch")
    latest = {"retarget": None, "session": None}
    frames = [0, 0, 0]  # all, left detected, right detected

    def on_retarget(msg):
        latest["retarget"] = json.loads(msg.data).get("hands")

    def on_session(msg):
        latest["session"] = json.loads(msg.data)

    def on_frame(msg):
        frames[0] += 1
        frames[1] += int(msg.left.detected)
        frames[2] += int(msg.right.detected)

    node.create_subscription(String, "/motion_imitation_mock/status", on_retarget, qos_profile_sensor_data)
    node.create_subscription(String, "/mirror_session/status", on_session, qos_profile_sensor_data)
    node.create_subscription(HandFrame, "/hri/hand/frame", on_frame, qos_profile_sensor_data)
    last = time.monotonic()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
        now = time.monotonic()
        if now - last < PERIOD_S:
            continue
        dt, last = now - last, now
        rate = [round(v / dt, 1) for v in frames]
        frames[:] = [0, 0, 0]
        session = latest["session"] or {}
        hands = session.get("hands") or {}
        retarget = latest["retarget"] or {}
        print(
            time.strftime("%H:%M:%S"),
            f"state={session.get('state')} phase={session.get('phase')}",
            f"frame_hz(all/L/R)={rate}",
            f"perc={{age_ms:{retarget.get('hand_age_ms')}, stale:{retarget.get('hand_stale')}, "
            f"gesture:{retarget.get('gesture')}}}",
            f"overlay={{follow:{hands.get('following')}, start:{hands.get('start_source')}, "
            f"unjams:{hands.get('unjams')}, unjam:{hands.get('unjam')}, fb_age:{hands.get('state_age_s')}, "
            f"fb_lost:{hands.get('state_lost')}, held:{hands.get('held_state')}, "
            f"follow_age:{hands.get('follow_age_s')}}}",
            f"last={hands.get('last')}",
            flush=True,
        )
        latest["retarget"] = latest["session"] = None


if __name__ == "__main__":
    main()
