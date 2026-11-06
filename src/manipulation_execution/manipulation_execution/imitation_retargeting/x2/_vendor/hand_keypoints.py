"""手部关键点 → 每指弯曲度。**纯 numpy，没有一行 ROS**，也不 import mediapipe。

H1 离线标定（tools/dexhand_h1.py）和 H3 实时节点用的是这同一个模块，所以
「实时看到的弯曲度」和「标定时的弯曲度」是同一个函数算出来的。

输入是 MediaPipe 手部关键点的约定（21 点：0 腕，1–4 拇指，5–8 食指，9–12 中指，
13–16 无名指，17–20 小指）。只认数组，不认 mediapipe 的对象 —— 调用方自己把
`hand_landmarks` 换成像素 (21,2)、`hand_world_landmarks` 换成 (21,3)。
WiLoR（H7）的 21 点是同一个顺序，可以直接喂。

  1. **弯曲度** `finger_curls(wlm)`：每指一条链「腕 → 根 → 中 → 末 → 尖」，
     相邻骨段的夹角（不带符号）求和，弧度。四指是 MCP/PIP/DIP 三个角；拇指链从腕
     起到 1–4，是 CMC/MCP/IP 三个角。用 world landmarks（米制 3D，手心为原点），
     所以和手在画面里多大、离多远无关。
     - 0 不是「伸直」：张开的手也有 0.5–1 rad（MCP 处骨段并不共线）。
       端点要按人标定（H1 标定文件），hand_core 再归一化。
     - 夹角不带符号：过伸会算成弯（真人过伸只有十几度，落在张开端点附近）；
       侧摆会漏进 MCP 角（五指张大时多几度）。两者都不会把「伸」读成「握」。
     - 视角：world landmarks 的深度是单目估出来的，侧视时手指互相遮挡，
       握拳会读浅一些，量化见 docs/dexhand/H1_REPORT.md。
  2. **左右手** `assign_sides(...)`：按画面位置分，不用 MediaPipe 的 handedness
     （D1 上只有约 80% 对）。相机对着人、画面不镜像，所以**画面左边是人的右手**。
     两只手 → 按 x 排；一只手 → 跟人框中心比。两只手几乎重合 → 当成同一只手检了两次。
  3. **运行时不抛异常**：缺手、NaN、形状不对、退化骨段 → 该手（或该指）给 NaN，
     下游 hand_core 会剔除 NaN 行、保持上一次的有效值。
"""

from __future__ import annotations

import numpy as np

N_LANDMARKS = 21
WRIST = 0
# 每指一条链（不含腕）。顺序 = hand_core.FINGERS + 拇指
CHAINS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
    "thumb": (1, 2, 3, 4),
}
CURL_FINGERS = ("index", "middle", "ring", "pinky", "thumb")
N_CURL = len(CURL_FINGERS)
_PATHS = np.array([(WRIST,) + CHAINS[f] for f in CURL_FINGERS])     # (5,5)
_MIN_SEG_M = 1e-5            # 骨段短于 0.01 mm 当退化（关键点重合）

HUMAN_SIDES = ("left", "right")      # 人手侧
_DUP_FRAC = 0.3              # 两只手中心距 < 0.3 × 手的尺寸 → 同一只手检了两次


def _as_array(x, shape):
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.shape != shape:
        return None
    return a


def finger_curls(wlm) -> np.ndarray:
    """(21,3) world landmarks → (5,) 弧度，CURL_FINGERS 顺序（食、中、无名、小、拇）。

    整只手无效 → 5 个 NaN；某指有退化骨段 → 该指 NaN。不原地改输入。
    """
    a = _as_array(wlm, (N_LANDMARKS, 3))
    if a is None or not np.all(np.isfinite(a)):
        return np.full(N_CURL, np.nan)
    pts = a[_PATHS]                                 # (5,5,3)
    seg = np.diff(pts, axis=1)                      # (5,4,3)
    ln = np.linalg.norm(seg, axis=2)
    bad = np.any(ln < _MIN_SEG_M, axis=1)
    u = seg / np.maximum(ln, _MIN_SEG_M)[..., None]
    cos = np.clip(np.sum(u[:, :-1] * u[:, 1:], axis=2), -1.0, 1.0)
    out = np.arccos(cos).sum(axis=1)
    out[bad] = np.nan
    return out


THUMB_NEAR = (6, 7, 10, 11)  # 食指、中指的 PIP / DIP
PALM = (WRIST, 9)            # 手掌长：腕 → 中指根


def thumb_tuck(wlm) -> float:
    """(21,3) world landmarks（或 HandObservation 里展平的 63 个）→ 拇指尖到食指、中指 PIP / DIP
    的最近距离 ÷ 手掌长，无单位。

    拇指跟随（hand_core.ThumbFollowConfig）用它判断人的拇指是扣在四指上，还是伸在外面：
    D1 上握拳中位 0.17，竖拇指 0.63，「六」0.78（docs/dexhand/THUMB_FOLLOW.md）。
    四指张开时这个距离也大，但那时 curve(四指均值) ≈ 0，取 min 以后不起作用。
    无效（形状不对、NaN、手掌长退化）→ NaN。不抛异常。
    """
    a = _as_array(wlm, (N_LANDMARKS, 3))
    if a is None:
        a = _as_array(wlm, (N_LANDMARKS * 3,))
        a = None if a is None else a.reshape(N_LANDMARKS, 3)
    if a is None or not np.all(np.isfinite(a)):
        return float("nan")
    palm = float(np.linalg.norm(a[PALM[1]] - a[PALM[0]]))
    if palm < _MIN_SEG_M:
        return float("nan")
    return float(np.linalg.norm(a[list(THUMB_NEAR)] - a[4], axis=1).min() / palm)


def hand_center(lm) -> np.ndarray | None:
    """(21,2) 像素 → 中心；无效 → None。"""
    a = _as_array(lm, (N_LANDMARKS, 2))
    if a is None or not np.all(np.isfinite(a)):
        return None
    return a.mean(axis=0)


def assign_sides(lms, person_cx) -> dict:
    """每只检出的手的像素关键点 → {"left": 下标或 None, "right": 下标或 None}（人手侧）。

    lms：list，每项 (21,2) 像素（无效项跳过）。person_cx：人框中心的 x（像素）。
    画面左边 = 人的右手。两只有效手 → 按 x 排；一只 → 跟 person_cx 比；
    person_cx 无效且只有一只手 → 分不出，两边都 None。多于两只 → 只看最左、最右两只。
    """
    out = {"left": None, "right": None}
    hands = []
    for i, lm in enumerate(lms if lms is not None else ()):
        c = hand_center(lm)
        if c is not None:
            size = float(np.ptp(np.asarray(lm, dtype=np.float64), axis=0).max())
            hands.append((float(c[0]), float(c[1]), size, i))
    if len(hands) >= 2:
        hands.sort(key=lambda h: h[0])
        a, b = hands[0], hands[-1]
        if np.hypot(a[0] - b[0], a[1] - b[1]) < _DUP_FRAC * max(a[2], b[2], 1e-9):
            hands = [a]                             # 同一只手检了两次 → 按一只手处理
        else:
            out["right"], out["left"] = a[3], b[3]
            return out
    if len(hands) == 1:
        try:
            cx = float(person_cx)
        except (TypeError, ValueError):
            return out
        if np.isfinite(cx):
            out["right" if hands[0][0] < cx else "left"] = hands[0][3]
    return out


def upper_body_box(bbox, W, H):
    """PEAR 人框 → 喂 MediaPipe 的上半身裁图框（整数像素，已夹进图内）。

    左右各扩 30%，从框顶往上 5% 到框高 70%。H1 离线（tools/dexhand_mp_d1.py）和
    H3 实时节点用的都是这一份，换了它两边的结果就对不上了。
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = x1 - x0, y1 - y0
    bx0, by0 = max(0, int(x0 - 0.3 * w)), max(0, int(y0 - 0.05 * h))
    bx1, by1 = min(int(W), int(x1 + 0.3 * w)), min(int(H), int(y0 + 0.7 * h))
    return bx0, by0, bx1, by1


def hand_curls(hands, person_cx) -> dict:
    """一帧的全部检出 → {"left": (5,), "right": (5,)}，人手侧，缺手 = 5 个 NaN。

    hands：list，每项是 (lm (21,2) 像素, wlm (21,3) 米) 的二元组或带这两个键的 dict。
    """
    lms, wlms = [], []
    for h in hands if hands is not None else ():
        try:
            lm, wlm = (h["lm"], h["wlm"]) if isinstance(h, dict) else h
        except (TypeError, ValueError, KeyError):
            lm, wlm = None, None
        lms.append(lm)
        wlms.append(wlm)
    idx = assign_sides(lms, person_cx)
    return {s: (np.full(N_CURL, np.nan) if idx[s] is None else finger_curls(wlms[idx[s]]))
            for s in HUMAN_SIDES}
