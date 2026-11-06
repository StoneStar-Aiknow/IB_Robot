"""MediaPipe 手部流水线（palm detection → ROI → hand landmark → 跟踪）的 numpy/OpenCV 重写版，
推理后端可换：笔记本上用 onnxruntime 跑 ONNX，310P 上用 ais_bench 跑 OM。

为什么要重写：官方 mediapipe 的 aarch64 包在鲲鹏上 SIGILL（无 RCPC 指令），只能把模型拿出来自己搭流水线。
算法照 MediaPipe 官方图（hand_landmarker / palm_detection / hand_landmark 的 calculator 配置）实现：
  - SSD anchors：192 输入、4 层、strides [8,16,16,16]、固定尺寸 → 2016 个
  - 解码：x/y/w/h/7 个关键点都按 192 缩放加 anchor；sigmoid 分数 ≥ 0.5；加权 NMS（IoU 0.3）
  - 检测 → ROI：旋转 = 90° − atan2(−(kp2−kp0)) ；放大 2.6、y 向偏移 −0.5、取长边成正方形
  - 关键点 → 下一帧 ROI（跟踪）：旋转由腕 + 食/中/无名 MCP 定；放大 2.0、偏移 −0.1
  - 关键点模型输入 224，输出 21×3（输入像素）、手存在分数、左右手、21×3 世界坐标（米）
本模块不依赖 mediapipe 包，只依赖 numpy、cv2 和一个 backend（infer(name, x) -> list[np.ndarray]）。
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

DET_SIZE, LM_SIZE = 192, 224


# --------------------------------------------------------------------------- anchors / 解码
def ssd_anchors():
    strides, n_layers = [8, 16, 16, 16], 4
    anchors, layer = [], 0
    while layer < n_layers:
        last = layer
        n_per_cell = 0
        while last < n_layers and strides[last] == strides[layer]:
            n_per_cell += 2                       # 1 个宽高比 + 1 个插值尺度
            last += 1
        g = math.ceil(DET_SIZE / strides[layer])
        for y in range(g):
            for x in range(g):
                for _ in range(n_per_cell):
                    anchors.append(((x + 0.5) / g, (y + 0.5) / g))
        layer = last
    a = np.array(anchors, np.float32)
    assert a.shape == (2016, 2)
    return a


ANCHORS = ssd_anchors()


def decode(raw_boxes, raw_scores, thr=0.5):
    """→ (N,4) [xmin,ymin,xmax,ymax]，(N,7,2) 关键点，(N,) 分数；坐标归一化到 192 输入。"""
    b = raw_boxes.reshape(-1, 18)
    s = 1.0 / (1.0 + np.exp(-np.clip(raw_scores.reshape(-1), -100, 100)))
    keep = s >= thr
    if not keep.any():
        return np.zeros((0, 4)), np.zeros((0, 7, 2)), np.zeros(0)
    b, s, an = b[keep], s[keep], ANCHORS[keep]
    cx = b[:, 0] / DET_SIZE + an[:, 0]
    cy = b[:, 1] / DET_SIZE + an[:, 1]
    w, h = b[:, 2] / DET_SIZE, b[:, 3] / DET_SIZE
    box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    kp = b[:, 4:].reshape(-1, 7, 2) / DET_SIZE + an[:, None, :]
    return box, kp, s


def iou(a, b):
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def weighted_nms(box, kp, s, thr=0.3):
    order = np.argsort(-s)
    box, kp, s = box[order], kp[order], s[order]
    out = []
    used = np.zeros(len(s), bool)
    for i in range(len(s)):
        if used[i]:
            continue
        o = iou(box[i], box) > thr
        o &= ~used
        used |= o
        w = s[o][:, None]
        out.append(((box[o] * w).sum(0) / w.sum(), (kp[o] * w[:, :, None]).sum(0) / w.sum(), s[i]))
    return out


# --------------------------------------------------------------------------- ROI 几何（像素坐标）
def norm_rad(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def transform_rect(cx, cy, w, h, rot, scale, shift_y):
    """RectTransformation：先在旋转坐标里平移，再取长边成正方形并放大。"""
    cx += -h * shift_y * math.sin(rot)
    cy += h * shift_y * math.cos(rot)
    s = max(w, h) * scale
    return cx, cy, s, rot


def roi_from_detection(box_px, kp_px):
    x0, y0 = kp_px[0]
    x1, y1 = kp_px[2]
    rot = norm_rad(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    cx, cy = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
    return transform_rect(cx, cy, box_px[2] - box_px[0], box_px[3] - box_px[1], rot, 2.6, -0.5)


PARTIAL = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


def roi_from_landmarks(lm_px):
    """HandLandmarksToRect：腕 → (食MCP+无名MCP)/2 与中MCP 的中点 定方向；在旋转坐标里取外接框。"""
    p = lm_px[PARTIAL, :2]
    x0, y0 = lm_px[0, :2]
    xm = ((lm_px[5, 0] + lm_px[13, 0]) / 2 + lm_px[9, 0]) / 2
    ym = ((lm_px[5, 1] + lm_px[13, 1]) / 2 + lm_px[9, 1]) / 2
    rot = norm_rad(math.pi / 2 - math.atan2(-(ym - y0), xm - x0))
    c = (p.min(0) + p.max(0)) / 2
    R = np.array([[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]])
    q = (p - c) @ R.T
    mn, mx = q.min(0), q.max(0)
    cc = c + ((mn + mx) / 2) @ np.linalg.inv(R).T
    return transform_rect(cc[0], cc[1], mx[0] - mn[0], mx[1] - mn[1], rot, 2.0, -0.1)


def roi_affine(roi, size):
    """ROI(cx,cy,s,rot) → 把原图映射到 size×size 的 2×3 仿射矩阵。"""
    cx, cy, s, rot = roi
    c, si = math.cos(rot), math.sin(rot)
    # 输出 (u,v) → 原图：center + R·((u/size−0.5)s, (v/size−0.5)s)
    src = np.array([[cx + (-0.5 * s) * c - (-0.5 * s) * si, cy + (-0.5 * s) * si + (-0.5 * s) * c],
                    [cx + (0.5 * s) * c - (-0.5 * s) * si, cy + (0.5 * s) * si + (-0.5 * s) * c],
                    [cx + (-0.5 * s) * c - (0.5 * s) * si, cy + (-0.5 * s) * si + (0.5 * s) * c]], np.float32)
    dst = np.array([[0, 0], [size, 0], [0, size]], np.float32)
    return cv2.getAffineTransform(src, dst)


# --------------------------------------------------------------------------- 规则（与 analyze.py 相同阈值）
TIPS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20)}
GESTURES = {   # 1 = 伸直；顺序 thumb,index,middle,ring,pinky
    "open": (1, 1, 1, 1, 1), "thumbs_up": (1, 0, 0, 0, 0), "thumb_index": (1, 1, 0, 0, 0),
    "thumb_index_middle": (1, 1, 1, 0, 0), "six": (1, 0, 0, 0, 1)}


def bend_sum(w, chain, root):
    pts = np.concatenate([w[root:root + 1], w[list(chain)]])
    seg = np.diff(pts, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
    return np.degrees(np.arccos(np.clip(np.sum(seg[:-1] * seg[1:], 1), -1, 1))).sum()


def finger_states(w):
    st = {}
    for f, ch in TIPS.items():
        if f == "thumb":
            palm = np.linalg.norm(w[5] - w[17]) + 1e-9
            st[f] = bool(bend_sum(w, ch[1:], ch[0]) < 60 and np.linalg.norm(w[4] - w[9]) / palm > 1.1)
        else:
            st[f] = bool(bend_sum(w, ch, 0) < 80)
    return st


def rule_class(st):
    four = [st[x] for x in ("index", "middle", "ring", "pinky")]
    if not any(four) and not st["thumb"]:
        return "fist"
    key = tuple(int(st[x]) for x in ("thumb", "index", "middle", "ring", "pinky"))
    for g, t in GESTURES.items():
        if key == t:
            return g
    return "other"


def curl_deg(w):
    """连续弯曲度（度）：四指 腕→MCP→PIP→DIP→尖 夹角和；拇指 CMC→MCP→IP→尖。"""
    return {f: float(bend_sum(w, ch, 0) if f != "thumb" else bend_sum(w, ch[1:], ch[0])) for f, ch in TIPS.items()}


# --------------------------------------------------------------------------- 流水线
class HandPipeline:
    """每帧输入：原图（BGR）+ 人框；输出两只手的关键点、世界坐标、手势。

    track=True：沿用上一帧关键点推出的 ROI，只有跟踪的手少于 2 只时才跑手掌检测（官方 VIDEO 模式的做法）。
    """

    def __init__(self, backend, track=True, presence_thr=0.5, det_thr=0.5):
        self.be, self.track = backend, track
        self.presence_thr, self.det_thr = presence_thr, det_thr
        self.rois = []
        self.t = {}

    def _tick(self, k, t0):
        self.t[k] = self.t.get(k, 0.0) + (time.perf_counter() - t0) * 1e3

    def detect(self, img, person_box):
        t0 = time.perf_counter()
        H, W = img.shape[:2]
        x0, y0, x1, y1 = person_box
        w, h = x1 - x0, y1 - y0
        cx0, cy0 = int(max(0, x0 - 0.3 * w)), int(max(0, y0 - 0.05 * h))
        cx1, cy1 = int(min(W, x1 + 0.3 * w)), int(min(H, y0 + 0.7 * h))
        crop = img[cy0:cy1, cx0:cx1]
        ch, cw = crop.shape[:2]
        sc = DET_SIZE / max(ch, cw)
        nw, nh = int(round(cw * sc)), int(round(ch * sc))
        px, py = (DET_SIZE - nw) // 2, (DET_SIZE - nh) // 2
        lb = np.zeros((DET_SIZE, DET_SIZE, 3), np.uint8)
        lb[py:py + nh, px:px + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self._tick("det_pre", t0)
        t0 = time.perf_counter()
        boxes, scores = self.be.infer("det", x)
        self._tick("det_npu", t0)
        t0 = time.perf_counter()
        box, kp, s = decode(boxes, scores, self.det_thr)
        rois = []
        for b, k, _ in weighted_nms(box, kp, s):
            # 192 输入坐标 → 原图像素
            to_img = lambda u, v: (cx0 + (u * DET_SIZE - px) / sc, cy0 + (v * DET_SIZE - py) / sc)
            bx0, by0 = to_img(b[0], b[1]); bx1, by1 = to_img(b[2], b[3])
            kpp = np.array([to_img(*q) for q in k])
            rois.append(roi_from_detection((bx0, by0, bx1, by1), kpp))
        self._tick("det_post", t0)
        return rois

    def landmarks(self, img, roi):
        t0 = time.perf_counter()
        M = roi_affine(roi, LM_SIZE)
        patch = cv2.warpAffine(img, M, (LM_SIZE, LM_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self._tick("lm_pre", t0)
        t0 = time.perf_counter()
        lm, pres, hand, wlm = self.be.infer("lm", x)
        self._tick("lm_npu", t0)
        t0 = time.perf_counter()
        lm = lm.reshape(21, 3)
        Mi = cv2.invertAffineTransform(M)
        xy = np.c_[lm[:, :2], np.ones(21)] @ Mi.T
        out = {"lm": np.c_[xy, lm[:, 2] * roi[2] / LM_SIZE], "wlm": wlm.reshape(21, 3).astype(np.float64),
               "presence": float(pres.reshape(-1)[0]), "handed": float(hand.reshape(-1)[0]), "roi": roi}
        self._tick("lm_post", t0)
        return out

    def __call__(self, img, person_box):
        self.t = {}
        t_all = time.perf_counter()
        rois = list(self.rois) if self.track else []
        ran_det = False
        if len(rois) < 2:
            ran_det = True
            new = self.detect(img, person_box)
            for r in new:                      # 已在跟踪的手附近的检测丢掉（去重）
                if all(math.hypot(r[0] - q[0], r[1] - q[1]) > 0.5 * max(r[2], q[2]) for q in rois):
                    rois.append(r)
        hands = []
        for r in rois[:2]:
            hd = self.landmarks(img, r)
            if hd["presence"] >= self.presence_thr:
                hands.append(hd)
        t0 = time.perf_counter()
        for hd in hands:
            hd["states"] = finger_states(hd["wlm"])
            hd["gesture"] = rule_class(hd["states"])
            hd["curl_deg"] = curl_deg(hd["wlm"])
        self.rois = [roi_from_landmarks(hd["lm"]) for hd in hands] if self.track else []
        self._tick("rules_track", t0)
        self.t["total"] = (time.perf_counter() - t_all) * 1e3
        self.t["ran_det"] = ran_det
        return hands, dict(self.t)

    def reset(self):
        self.rois = []


class OrtBackend:
    def __init__(self, det_onnx, lm_onnx):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 4
        self.s = {"det": ort.InferenceSession(det_onnx, o, providers=["CPUExecutionProvider"]),
                  "lm": ort.InferenceSession(lm_onnx, o, providers=["CPUExecutionProvider"])}

    def infer(self, name, x):
        s = self.s[name]
        outs = s.run(None, {s.get_inputs()[0].name: x})
        if name == "det":                       # Identity [1,2016,18] 框，Identity_1 [1,2016,1] 分数
            return outs[0], outs[1]
        return self._lm_order(outs)

    @staticmethod
    def _lm_order(outs):
        """关键点模型输出顺序（ONNX 与 OM 相同，已核对）：Identity [1,63] 关键点（输入像素）、
        Identity_1 [1,1] 手存在分数（已 sigmoid）、Identity_2 [1,1] 左右手、Identity_3 [1,63] 世界坐标（米）。"""
        assert [o.size for o in outs] == [63, 1, 1, 63], [o.shape for o in outs]
        return outs[0], outs[1], outs[2], outs[3]


class AclBackend:
    """310P：ais_bench 跑 OM。输出顺序与 ONNX 一致（ATC 按图输出顺序），排序规则同 ORT。"""

    def __init__(self, det_om, lm_om, device=0):
        from ais_bench.infer.interface import InferSession
        self.s = {"det": InferSession(device, det_om), "lm": InferSession(device, lm_om)}

    def infer(self, name, x):
        outs = [np.asarray(o) for o in self.s[name].infer([np.ascontiguousarray(x)])]
        if name == "det":
            return outs[0].reshape(1, 2016, 18), outs[1].reshape(1, 2016, 1)
        return OrtBackend._lm_order([o.reshape(1, -1) for o in outs])
