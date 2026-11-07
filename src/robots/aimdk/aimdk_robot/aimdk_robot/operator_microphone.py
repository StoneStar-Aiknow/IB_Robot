"""Operator microphone selection with readback. Never sends a motion command."""

import argparse
import json
import time

SOURCES = {"builtin": 0, "external": 1}


def select_microphone(backend, source, *, pause=time.sleep):
    """Write at most once; a timeout or failed readback never causes a retry."""
    target = SOURCES[source]
    before = backend.get_source()
    if before != target:
        backend.set_source(target)
    for _ in range(6):
        current = backend.get_source()
        if current == target:
            return {"before": before, "selected": current, "source": source, "changed": before != target}
        pause(0.3)
    raise RuntimeError(f"source readback is {current}, expected {target}; no write retry was attempted")


class RosBackend:
    def __init__(self):
        import rclpy
        from aimdk_msgs.msg import CommonState
        from aimdk_msgs.srv import GetMicSourceRequest, SetMicSourceRequest

        self.ros = rclpy
        self.node = rclpy.create_node("operator_microphone_selector")
        self.success = CommonState.SUCCESS
        self.get_type = GetMicSourceRequest
        self.set_type = SetMicSourceRequest
        self.get_client = self.node.create_client(GetMicSourceRequest, "/aimdk_5Fmsgs/srv/GetMicSourceRequest")
        self.set_client = self.node.create_client(SetMicSourceRequest, "/aimdk_5Fmsgs/srv/SetMicSourceRequest")

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError("vendor microphone service unavailable; check HAL on 10.0.1.42 and ROS domain")
        request.header.header.stamp = self.node.get_clock().now().to_msg()
        future = client.call_async(request)
        self.ros.spin_until_future_complete(self.node, future, timeout_sec=8)
        if not future.done() or future.result() is None:
            raise RuntimeError("service result unknown; query status before issuing another switch")
        response = future.result()
        if response.header.status.value != self.success:
            raise RuntimeError(f"vendor did not confirm success: {response.header}")
        return response

    def get_source(self):
        return int(self.call(self.get_client, self.get_type.Request()).mic_source)

    def set_source(self, value):
        request = self.set_type.Request()
        request.mic_source = value
        self.call(self.set_client, request)

    def verify_audio(self, source):
        from aimdk_msgs.msg import AudioCapture
        from rclpy.qos import qos_profile_sensor_data

        expected = (4, 2) if source == "builtin" else (1, 1)
        frames = []
        observed = set()

        def receive(message):
            shape = (int(message.mic_channels), int(message.ref_channels))
            observed.add(shape)
            if shape == expected and message.info.sample_rate == 16000 and len(message.data.data) > 0:
                frames.append(len(message.data.data))

        subscription = self.node.create_subscription(
            AudioCapture, "/aima/hal/audio/capture", receive, qos_profile_sensor_data
        )
        try:
            deadline = time.monotonic() + 10
            while len(frames) < 3 and time.monotonic() < deadline:
                self.ros.spin_once(self.node, timeout_sec=0.1)
            if len(frames) < 3:
                raise RuntimeError(
                    f"source changed, but audio was not verified (expected mic/ref={expected}, "
                    f"observed={sorted(observed)}). Check receiver on PC3/42, USB support and transmitter pairing"
                )
            return {
                "frames": len(frames),
                "mic_channels": expected[0],
                "reference_channels": expected[1],
                "sample_rate": 16000,
                "note": "PCM packets verified; audibility/ASR still requires speaking test",
            }
        finally:
            self.node.destroy_subscription(subscription)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("status", "builtin", "external"))
    args = parser.parse_args(argv)
    import rclpy

    rclpy.init()
    backend = None
    try:
        backend = RosBackend()
        if args.source == "status":
            print(json.dumps({"mic_source": backend.get_source(), "builtin": 0, "external": 1}), flush=True)
        else:
            print(json.dumps(select_microphone(backend, args.source)), flush=True)
            print(json.dumps({"audio": backend.verify_audio(args.source)}), flush=True)
        return 0
    except RuntimeError as exc:
        print(json.dumps({"success": False, "message": str(exc)}), flush=True)
        return 1
    finally:
        if backend is not None:
            backend.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
