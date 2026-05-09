#!/usr/bin/env python3

import argparse
import threading
from http import server

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def image_to_mat(msg: Image) -> np.ndarray:
    if msg.step * msg.height != len(msg.data):
        raise ValueError(
            f"Invalid image buffer: step={msg.step}, height={msg.height}, size={len(msg.data)}"
        )

    if msg.encoding == "rgb8":
        mat = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    if msg.encoding == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)

    if msg.encoding == "rgba8":
        mat = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    if msg.encoding == "bgra8":
        mat = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(mat, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


class MjpegHandler(server.BaseHTTPRequestHandler):
    streamer = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (
                "<html><head><title>ROS2 Camera Viewer</title></head>"
                "<body style='margin:0;background:#111;'>"
                "<img src='/stream.mjpg' style='display:block;width:100vw;height:100vh;"
                "object-fit:contain;background:#111;' />"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.streamer.get_jpeg_frame()
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


class MjpegStreamer:
    def __init__(self, host: str, port: int):
        self._condition = threading.Condition()
        self._jpeg_frame = None
        self._server = server.ThreadingHTTPServer((host, port), MjpegHandler)
        MjpegHandler.streamer = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return
        with self._condition:
            self._jpeg_frame = encoded.tobytes()
            self._condition.notify_all()

    def get_jpeg_frame(self):
        with self._condition:
            while self._jpeg_frame is None and rclpy.ok():
                self._condition.wait(timeout=0.5)
            return self._jpeg_frame

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class CameraTopicViewer(Node):
    def __init__(self, topic: str, window_name: str, mode: str, host: str, port: int):
        super().__init__("camera_topic_viewer")
        self._window_name = window_name
        self._topic = topic
        self._mode = mode
        self._gui_available = None
        self._streamer = None

        self.create_subscription(
            Image,
            topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribed to {topic}")
        if mode in ("auto", "mjpeg"):
            self._streamer = MjpegStreamer(host, port)
            self.get_logger().info(
                f"MJPEG preview available at http://{host}:{port}"
            )

    def _image_callback(self, msg: Image) -> None:
        try:
            image = image_to_mat(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        if self._mode != "mjpeg":
            if self._gui_available is not False:
                try:
                    cv2.imshow(self._window_name, image)
                    self._gui_available = True
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self.get_logger().info("Exit requested from viewer window")
                        rclpy.shutdown()
                        return
                except cv2.error as exc:
                    self._gui_available = False
                    self.get_logger().warning(
                        "OpenCV GUI is unavailable, falling back to MJPEG preview. "
                        f"Details: {exc}"
                    )

        if self._streamer is not None:
            self._streamer.update(image)

    def close(self) -> None:
        if self._streamer is not None:
            self._streamer.shutdown()
        if self._gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to a ROS 2 image topic and show the live image.",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="ROS 2 image topic, for example /camera/top/image_raw",
    )
    parser.add_argument(
        "--window-name",
        default="ROS2 Camera Viewer",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gui", "mjpeg"],
        default="auto",
        help="Display mode. 'auto' tries OpenCV GUI first, then falls back to MJPEG.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the MJPEG preview server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the MJPEG preview server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    viewer = CameraTopicViewer(
        topic=args.topic,
        window_name=args.window_name,
        mode=args.mode,
        host=args.host,
        port=args.port,
    )

    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
