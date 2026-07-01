"""Save Grounded-SAM2 input and segmentation visualizations for inspection."""

import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from ibrobot_msgs.srv import DetectSegment

PALETTE_BGR = [
    (38, 114, 236),
    (76, 175, 80),
    (233, 150, 43),
    (156, 39, 176),
    (0, 188, 212),
    (244, 67, 54),
    (121, 85, 72),
    (63, 81, 181),
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return slug[:48] or "prompt"


def make_run_dir(root: Path, prompt: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(prompt)}"
    for suffix in [""] + [f"_{idx:02d}" for idx in range(1, 100)]:
        run_dir = root / f"{base_name}{suffix}"
        if not run_dir.exists():
            run_dir.mkdir()
            return run_dir
    raise RuntimeError(f"Could not create a unique output directory under {root}")


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(text_h + baseline + 4, y)
    cv2.rectangle(
        image,
        (x, y - text_h - baseline - 6),
        (x + text_w + 8, y + baseline + 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def image_msg_to_gray(bridge: CvBridge, msg: Image, size: tuple[int, int]) -> np.ndarray:
    mask = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    mask = np.asarray(mask, dtype=np.uint8)
    expected_h, expected_w = size
    if mask.shape[:2] != (expected_h, expected_w):
        mask = cv2.resize(mask, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
    return mask


def depth_to_meters(depth_image: np.ndarray, depth_scale: float) -> np.ndarray:
    depth = np.asarray(depth_image)
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)
    return depth.astype(np.float32) / float(depth_scale)


def save_depth_preview(path: Path, depth_m: np.ndarray, depth_trunc: float) -> None:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m, 0.0, depth_trunc)
        depth_vis = (255.0 * (1.0 - clipped / depth_trunc)).astype(np.uint8)
        depth_vis[~valid] = 0
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_color[~valid] = (0, 0, 0)
    save_image(path, depth_color)


def pointcloud_from_depth(
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    mask: np.ndarray | None,
    depth_trunc: float,
    max_points: int,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_trunc)
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        valid &= mask > 0

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    if max_points > 0 and ys.size > max_points:
        idx = np.linspace(0, ys.size - 1, max_points, dtype=np.int64)
        ys = ys[idx]
        xs = xs[idx]

    z = depth_m[ys, xs].astype(np.float32)
    k = np.asarray(camera_info.k, dtype=np.float32).reshape(3, 3)
    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    rgb = image_bgr[ys, xs, ::-1].astype(np.float32)
    return np.column_stack([x, y, z, rgb]).astype(np.float32)


def save_ply(path: Path, cloud: np.ndarray) -> None:
    cloud = np.asarray(cloud, dtype=np.float32)
    vertex_count = int(cloud.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(vertex_count, dtype=dtype)
    if vertex_count:
        vertices["x"] = cloud[:, 0]
        vertices["y"] = cloud[:, 1]
        vertices["z"] = cloud[:, 2]
        colors = np.clip(cloud[:, 3:6], 0, 255).astype(np.uint8)
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

    with path.open("wb") as f:
        f.write(header)
        vertices.tofile(f)


def camera_info_to_metadata(camera_info: CameraInfo) -> dict:
    k = [float(v) for v in camera_info.k]
    return {
        "frame_id": camera_info.header.frame_id,
        "width": int(camera_info.width),
        "height": int(camera_info.height),
        "distortion_model": camera_info.distortion_model,
        "d": [float(v) for v in camera_info.d],
        "k": k,
        "r": [float(v) for v in camera_info.r],
        "p": [float(v) for v in camera_info.p],
        "fx": k[0],
        "fy": k[4],
        "cx": k[2],
        "cy": k[5],
    }


def build_pointcloud_outputs(
    bridge: CvBridge,
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    camera_info: CameraInfo,
    response: DetectSegment.Response,
    detections_meta: list[dict],
    run_dir: Path,
    depth_scale: float,
    depth_trunc: float,
    max_full_cloud_points: int,
    max_object_cloud_points: int,
) -> dict:
    depth_m = depth_to_meters(depth_image, depth_scale)
    np.save(run_dir / "depth_raw.npy", depth_image)
    save_depth_preview(run_dir / "depth.png", depth_m, depth_trunc)

    full_cloud_file = "cloud_full.ply"
    full_cloud = pointcloud_from_depth(
        image_bgr,
        depth_m,
        camera_info,
        mask=None,
        depth_trunc=depth_trunc,
        max_points=max_full_cloud_points,
    )
    save_ply(run_dir / full_cloud_file, full_cloud)

    cloud_meta = {
        "enabled": True,
        "depth": "depth_raw.npy",
        "depth_preview": "depth.png",
        "full_cloud": full_cloud_file,
        "full_cloud_points": int(full_cloud.shape[0]),
        "depth_scale": depth_scale,
        "depth_trunc": depth_trunc,
        "camera_frame_id": camera_info.header.frame_id,
        "camera_info": camera_info_to_metadata(camera_info),
    }

    for det, meta in zip(response.detections.detections, detections_meta, strict=False):
        mask = image_msg_to_gray(bridge, det.mask, image_bgr.shape[:2])
        cloud = pointcloud_from_depth(
            image_bgr,
            depth_m,
            camera_info,
            mask=mask,
            depth_trunc=depth_trunc,
            max_points=max_object_cloud_points,
        )
        cloud_file = f"cloud_{meta['index']:02d}_{slugify(meta['label'])}.ply"
        save_ply(run_dir / cloud_file, cloud)
        meta["pointcloud"] = cloud_file
        meta["pointcloud_points"] = int(cloud.shape[0])

    return cloud_meta


def build_visualizations(
    bridge: CvBridge,
    image_bgr: np.ndarray,
    response: DetectSegment.Response,
    run_dir: Path,
    alpha: float,
) -> tuple[np.ndarray, list[dict]]:
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]
    detections_meta = []

    if not response.detections.detections:
        draw_text_box(overlay, "no detections", (12, 28), (64, 64, 64))
        return overlay, detections_meta

    for idx, det in enumerate(response.detections.detections):
        color = PALETTE_BGR[idx % len(PALETTE_BGR)]
        mask = image_msg_to_gray(bridge, det.mask, (h, w))
        mask_bool = mask > 0
        mask_file = f"mask_{idx:02d}_{slugify(det.label)}.png"
        save_image(run_dir / mask_file, np.where(mask_bool, 255, 0).astype(np.uint8))

        if np.any(mask_bool):
            color_arr = np.array(color, dtype=np.float32)
            overlay_pixels = overlay[mask_bool].astype(np.float32)
            overlay[mask_bool] = ((1.0 - alpha) * overlay_pixels + alpha * color_arr).astype(np.uint8)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        draw_text_box(overlay, f"{idx}: {det.confidence:.2f}", (x1, max(18, y1)), color)

        detections_meta.append(
            {
                "index": idx,
                "label": det.label,
                "confidence": float(det.confidence),
                "bbox_xyxy": [float(v) for v in det.bbox],
                "centroid_xyz": [
                    float(det.centroid_xyz.x),
                    float(det.centroid_xyz.y),
                    float(det.centroid_xyz.z),
                ],
                "point_count": int(det.point_count),
                "mask": mask_file,
            }
        )

    return overlay, detections_meta


def make_comparison(input_image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    left = input_image.copy()
    right = overlay.copy()
    draw_text_box(left, "input", (12, 28), (64, 64, 64))
    draw_text_box(right, "segmentation", (12, 28), (64, 64, 64))
    separator = np.full((left.shape[0], 8, 3), 240, dtype=np.uint8)
    return np.hstack([left, separator, right])


def write_index_html(run_dir: Path, metadata: dict) -> None:
    rows = []
    for det in metadata["detections"]:
        pointcloud_cell = ""
        if det.get("pointcloud"):
            pointcloud_cell = f'<a href="{html.escape(det["pointcloud"])}">{html.escape(det["pointcloud"])}</a>'
        rows.append(
            "<tr>"
            f"<td>{det['index']}</td>"
            f"<td>{html.escape(det['label'])}</td>"
            f"<td>{det['confidence']:.3f}</td>"
            f"<td>{html.escape(str([round(v, 1) for v in det['bbox_xyxy']]))}</td>"
            f"<td>{det['point_count']}</td>"
            f'<td><a href="{html.escape(det["mask"])}">{html.escape(det["mask"])}</a></td>'
            f"<td>{pointcloud_cell}</td>"
            "</tr>"
        )

    rows_html = "\n".join(rows) or '<tr><td colspan="7">No detections</td></tr>'
    pointcloud_html = "<p>Point cloud export disabled or unavailable.</p>"
    pointcloud = metadata.get("pointcloud", {})
    if pointcloud.get("enabled"):
        pointcloud_html = (
            f"<p><strong>Full cloud:</strong> "
            f'<a href="{html.escape(pointcloud["full_cloud"])}">'
            f"{html.escape(pointcloud['full_cloud'])}</a> "
            f"({pointcloud['full_cloud_points']} points)</p>"
            f"<p><strong>Depth preview:</strong> "
            f'<a href="{html.escape(pointcloud["depth_preview"])}">'
            f"{html.escape(pointcloud['depth_preview'])}</a></p>"
            f'<img src="{html.escape(pointcloud["depth_preview"])}" alt="Depth preview">'
        )
    elif pointcloud.get("reason"):
        pointcloud_html = f"<p>{html.escape(pointcloud['reason'])}</p>"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grounded-SAM2 Snapshot</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
    th {{ background: #f4f4f4; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Grounded-SAM2 Snapshot</h1>
  <p><strong>Prompt:</strong> <code>{html.escape(metadata["prompt"])}</code></p>
  <p><strong>Message:</strong> {html.escape(metadata["message"])}</p>
  <p><strong>Inference:</strong> {metadata["inference_time_ms"]:.1f} ms</p>
  <h2>Comparison</h2>
  <img src="comparison.png" alt="Input and segmentation comparison">
  <h2>Input</h2>
  <img src="input.png" alt="Input image">
  <h2>Overlay</h2>
  <img src="overlay.png" alt="Segmentation overlay">
  <h2>Point Cloud</h2>
  {pointcloud_html}
  <h2>Detections</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Label</th><th>Confidence</th><th>BBox xyxy</th><th>Points</th><th>Mask</th><th>Point Cloud</th></tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(page, encoding="utf-8")


class SnapshotNode(Node):
    def __init__(self, rgb_topic: str, depth_topic: str, camera_info_topic: str, service_name: str):
        super().__init__("grounded_sam2_snapshot")
        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_camera_info: CameraInfo | None = None
        self.latest_frame_id = ""
        self.latest_stamp = None

        self.create_subscription(
            Image,
            rgb_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.client = self.create_client(DetectSegment, service_name)
        self.get_logger().info(f"Waiting for RGB topic: {rgb_topic}")
        self.get_logger().info(f"Waiting for depth topic: {depth_topic}")
        self.get_logger().info(f"Waiting for camera info topic: {camera_info_topic}")

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert RGB image: {exc}")
            return
        self.latest_image = image.copy()
        self.latest_frame_id = msg.header.frame_id
        self.latest_stamp = msg.header.stamp

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert depth image: {exc}")
            return
        self.latest_depth = np.asarray(depth).copy()

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def wait_for_image(self, timeout_sec: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.latest_image is None:
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for an RGB image")
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_image.copy()

    def wait_for_depth_inputs(self, timeout_sec: float) -> tuple[np.ndarray, CameraInfo]:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and (self.latest_depth is None or self.latest_camera_info is None):
            if time.monotonic() > deadline:
                missing = []
                if self.latest_depth is None:
                    missing.append("depth image")
                if self.latest_camera_info is None:
                    missing.append("camera info")
                raise RuntimeError(f"Timed out waiting for {' and '.join(missing)}")
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_depth.copy(), self.latest_camera_info

    def call_detection(
        self,
        prompt: str,
        confidence_threshold: float,
        service_timeout_sec: float,
    ) -> DetectSegment.Response:
        if not self.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("DetectSegment service is not available")

        request = DetectSegment.Request()
        request.text_prompt = prompt
        request.confidence_threshold = confidence_threshold
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=service_timeout_sec)

        if not future.done():
            raise RuntimeError("Timed out waiting for DetectSegment response")
        result = future.result()
        if result is None:
            raise RuntimeError("DetectSegment service returned no result")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Grounded-SAM2 with a prompt and save input/output images.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for detection")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Override service box confidence threshold; use 0 to keep node default",
    )
    parser.add_argument(
        "--service",
        default="/grounded_sam2/detect_and_segment",
        help="DetectSegment service name",
    )
    parser.add_argument(
        "--rgb-topic",
        default="/camera/wrist/image_raw",
        help="RGB topic to save as input.png",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/wrist/aligned_depth_to_color/image_raw",
        help="Aligned depth topic used for point cloud export",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/camera/wrist/aligned_depth_to_color/camera_info",
        help="CameraInfo topic matching the RGB/aligned-depth frame",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/grounded_sam2",
        help="Root directory for generated snapshot folders",
    )
    parser.add_argument(
        "--image-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for one RGB frame",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for Grounded-SAM2 inference",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Segmentation overlay opacity",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open OpenCV windows after saving images",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Skip depth and PLY point cloud export",
    )
    parser.add_argument(
        "--depth-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for depth and camera info",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale for integer depth images; RealSense Z16 uses 1000 for meters",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Discard depth points farther than this many meters",
    )
    parser.add_argument(
        "--max-full-cloud-points",
        type=int,
        default=250000,
        help="Max points saved in cloud_full.ply; use 0 for all valid points",
    )
    parser.add_argument(
        "--max-object-cloud-points",
        type=int,
        default=0,
        help="Max points saved per object point cloud; use 0 for all valid points",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir).expanduser()
    run_dir = make_run_dir(out_root, args.prompt)

    rclpy.init()
    node = SnapshotNode(args.rgb_topic, args.depth_topic, args.camera_info_topic, args.service)
    try:
        input_image = node.wait_for_image(args.image_timeout)
        depth_image = None
        camera_info = None
        if not args.no_pointcloud:
            depth_image, camera_info = node.wait_for_depth_inputs(args.depth_timeout)
        response = node.call_detection(
            args.prompt,
            args.confidence_threshold,
            args.service_timeout,
        )

        overlay, detections = build_visualizations(
            node.bridge,
            input_image,
            response,
            run_dir,
            max(0.0, min(1.0, args.alpha)),
        )
        comparison = make_comparison(input_image, overlay)

        save_image(run_dir / "input.png", input_image)
        save_image(run_dir / "overlay.png", overlay)
        save_image(run_dir / "comparison.png", comparison)

        pointcloud_meta = {
            "enabled": False,
            "reason": "point cloud export disabled",
        }
        if not args.no_pointcloud and depth_image is not None and camera_info is not None:
            pointcloud_meta = build_pointcloud_outputs(
                node.bridge,
                input_image,
                depth_image,
                camera_info,
                response,
                detections,
                run_dir,
                args.depth_scale,
                args.depth_trunc,
                args.max_full_cloud_points,
                args.max_object_cloud_points,
            )

        metadata = {
            "prompt": args.prompt,
            "confidence_threshold": args.confidence_threshold,
            "rgb_topic": args.rgb_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "service": args.service,
            "success": bool(response.success),
            "message": response.message,
            "inference_time_ms": float(response.inference_time_ms),
            "frame_id": node.latest_frame_id,
            "camera_info": camera_info_to_metadata(camera_info) if camera_info is not None else None,
            "pointcloud": pointcloud_meta,
            "detections": detections,
        }
        (run_dir / "result.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_index_html(run_dir, metadata)

        print(f"output_dir: {run_dir.resolve()}")
        print(f"success: {response.success}")
        print(f"message: {response.message}")
        print(f"inference_time_ms: {response.inference_time_ms:.1f}")
        print(f"detection_count: {len(detections)}")
        if pointcloud_meta.get("enabled"):
            print(f"full_pointcloud: {pointcloud_meta['full_cloud']} points={pointcloud_meta['full_cloud_points']}")
        for det in detections:
            bbox = [round(v, 1) for v in det["bbox_xyxy"]]
            centroid = [round(v, 3) for v in det["centroid_xyz"]]
            pointcloud_file = det.get("pointcloud", "")
            print(
                f"detection[{det['index']}]: label={det['label']!r} "
                f"conf={det['confidence']:.3f} bbox={bbox} "
                f"points={det['point_count']} centroid={centroid} "
                f"cloud={pointcloud_file}"
            )

        if args.show:
            cv2.imshow("input", input_image)
            cv2.imshow("grounded_sam2_overlay", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0 if response.success else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
