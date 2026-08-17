"""Render offline test overlays from a body-to-camera result."""

from pathlib import Path

import cv2
import numpy as np
import yaml


def _read_pcd(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.rstrip() == "DATA ascii":
                return np.loadtxt(stream, dtype=np.float32, usecols=(0, 1, 2), ndmin=2)
    raise ValueError("point cloud must be an ASCII XYZ PCD")


def render_test_overlay(result: Path, exported_scene: Path, output: Path) -> int:
    """Project the independent test dense body cloud onto its RGB frame."""
    transform = yaml.safe_load(result.read_bytes())
    info = yaml.safe_load((exported_scene / "camera_info.yaml").read_bytes())
    image = cv2.imread(str(exported_scene / "image.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("test RGB image is unreadable")
    points = _read_pcd(exported_scene / "cloud_dense_body_from_livox.pcd")
    rotation = np.asarray(transform["rotation_matrix"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    camera = points @ rotation.T + translation
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)]
    pixels_float = camera @ np.asarray(info["K"], dtype=float).reshape(3, 3).T
    pixels = np.rint(pixels_float[:, :2] / pixels_float[:, 2:3]).astype(int)
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
    )
    pixels, depths = pixels[inside], camera[inside, 2]
    if not len(pixels):
        raise ValueError("no test cloud points project inside the image")
    near, far = np.percentile(depths, [5, 95])
    normalized = np.clip((depths - near) / max(far - near, 1e-9), 0.0, 1.0)
    colors = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    stride = max(1, len(pixels) // 12000)
    for (x, y), color in zip(pixels[::stride], colors[::stride], strict=True):
        cv2.circle(image, (int(x), int(y)), 1, tuple(int(value) for value in color), -1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or not cv2.imwrite(str(output), image):
        raise OSError(f"unable to exclusively write test overlay: {output}")
    return len(pixels)
