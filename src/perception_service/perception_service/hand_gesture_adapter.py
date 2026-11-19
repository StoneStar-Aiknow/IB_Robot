"""Fixed 320 BGR crop adapter for the existing YOLOv10 one-to-one head.

Postprocessing follows motion-template-hand/post_cut.py, as already used by
manipulation_execution.hand_perception.gesture_om. No resizing, cropping or NMS.
"""

from __future__ import annotations

import numpy as np

from .model_contracts import validate_image_message
from .perception_adapter import AdapterIdentity, PerceptionAdapter

INPUT_SEMANTIC = "observation.image"
OUTPUT_SEMANTIC = "gesture.raw"
INPUT_SHAPE = (1, 3, 320, 320)
OUTPUT_SHAPE = (1, 38, 2100)
OUTPUT_NAME = "/model.23/Concat_3_output_0"
VERIFIED_ENGINE_SHA256 = "421c325669b665e3331430f98365fe3a126bce92a4588097e9faeaa89fd7d639"


def validate_control_image(image):
    if image.width or image.height or image.step or image.encoding or image.is_bigendian or len(image.data):
        raise ValueError("probe/retire require an empty image")


class HandGestureAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        model_type="hand_gesture",
        operation="classify",
        preprocessing="x2-hand-gesture-bgr320-rgb-float32-nchw-v1",
        postprocessing="yolov10-cut34-top300-heart8-9-holy2-xsign7-v1",
        supported_deployments=frozenset({"tensorrt_orin_fp32"}),
    )

    def preprocess(self, image):
        validate_image_message(image)
        if (image.width, image.height, image.encoding, image.step, image.is_bigendian) != (
            320,
            320,
            "bgr8",
            960,
            0,
        ) or len(image.data) != 307200:
            raise ValueError("gesture image must be exactly 320x320 bgr8, step 960, 307200 bytes")
        bgr = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(320, 320, 3)
        x = bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        x *= np.float32(1.0 / 255.0)
        return {INPUT_SEMANTIC: np.ascontiguousarray(x)}

    def postprocess(self, result, **options):
        y = np.asarray(result.outputs[OUTPUT_SEMANTIC])
        if y.shape != OUTPUT_SHAPE or y.dtype != np.float32 or not np.isfinite(y).all():
            raise ValueError("gesture output must be finite float32 [1,38,2100]")
        t = y.reshape(38, 2100)
        s = t[4:]
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("gesture sigmoid scores must be in [0,1]")
        conf = float(options.get("confidence", 0.25))
        # Same stable top-300 selection as the deployed cut-head post_fast.
        ms = s.max(0)
        a = np.flatnonzero(ms >= conf)
        if a.size > 300:
            a = a[np.argsort(-ms[a], kind="stable")[:300]]
        sub = s[:, a]
        c, j = np.nonzero(sub >= conf)
        sc = sub[c, j]
        order = np.argsort(-sc, kind="stable")[:300]
        best = ("none", 0.0)
        for name, ids in (("heart", (8, 9)), ("holy", (2,)), ("xsign", (7,))):
            rows = np.flatnonzero(np.isin(c[order], ids))
            if rows.size:
                score = float(sc[order[rows[0]]])
                if score > best[1]:
                    best = name, score
        return best
