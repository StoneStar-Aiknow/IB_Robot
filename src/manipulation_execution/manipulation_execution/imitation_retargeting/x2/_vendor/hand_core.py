"""灵巧手的流式核心。**这个模块里没有一行 ROS**，也不 import vendor 和 followme_core。

设计稿：docs/dexhand/S3_DESIGN.md。这里只记代码读者必须知道的几条：

  1. **平行缓冲，不碰手臂的 `_buf`。** 每个 PEAR 帧算一行 8 个数（机器人左手四指 +
     机器人右手四指的归一化弯曲度，未夹紧），按与手臂**同一个 t、同一套插入排序
     和裁剪规则**入队。选窗函数 `select_window` 逐字照抄 followme_core
     `_live_target` 的选窗段（含 ±1e-9、右窗撑开、<8 帧饿死）。两边一致由
     tools/dexhand_core_check.py 的 W1 截获手臂真实窗口逐 tick 钉住——**改了手臂
     的选窗，这里必须跟着改，W1 会红。**

  2. **参数空间。** 每只手 5 个参数 p = (index, middle, ring, pinky, thumb) ∈ [0,1]。
     平滑、指间混合、相位淡入淡出、夹紧、限速全在 p 上做，最后一步才仿射映射
     `q = open + p·(fist − open)`。于是 B1 ⇔ p∈[0,1]，B2 ⇔ |Δp| ≤ 1/(T·hz)，
     拇指 3 个关节永远在「张开↔握拳」那条线段上（按关节分别限速会把它扯离线段）。

  3. **镜像在入口做完。** `curl_rows` 按机器人手取人手：镜像时机器人左手 ← 人右手，
     轴表和端点**跟人手走**（S2 Q7：交换即可，不翻符号）。机器人侧的左右符号差全在
     `dexhand_robot.json` 的两套预设里。

  4. **park/final 逐位等于 `q_neutral`。** 靠 `emit` 里 `force_neutral` 那一行直接拷贝，
     不靠「中性 + 0·(…)」恰好算回中性。与手臂 followme_core `_emit` 的 park/final 覆盖同理。

  5. **运行时不抛异常。** 坏输入（NaN、非正交、None、形状不对）一律变成 NaN 行，
     窗口回归剔除；某只手整窗无效就保持该手上一次的有效值。配置错误只在构造时 raise。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# HANDOFF_DEXHAND.md §2.4，hand_pos 每只手 10 个的下标顺序
HAND_JOINTS = ("thumb_roll", "thumb_abad", "thumb_mcp", "index_abad", "index_pip",
               "middle_pip", "ring_abad", "ring_pip", "pinky_abad", "pinky_pip")
HAND_CHANNEL_NAMES = tuple(f"{s}_hand_{j}_joint" for s in ("left", "right") for j in HAND_JOINTS)
HAND_DOF = len(HAND_CHANNEL_NAMES)          # 20

# PEAR / SMPL-X 的 15 个手部关节（S2 Q1：小指在无名指前面）
_PEAR_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
PEAR_JOINT_ORDER = tuple(f"{f}{i}" for f in _PEAR_FINGERS for i in (1, 2, 3))

# 参数空间里四指的顺序 = 机器人 pip 的顺序
FINGERS = ("index", "middle", "ring", "pinky")
_PEAR_GROUP = tuple(_PEAR_FINGERS.index(f) for f in FINGERS)      # (0, 1, 3, 2)
N_PARAM = 5                                  # 四指 + 拇指
THUMB = 4
# 每个关节由哪个参数驱动；5 = 常量（侧摆），对应 p6 末尾补的 0
_DRIVE = np.array([THUMB, THUMB, THUMB, 5, 0, 1, 5, 2, 5, 3])
_ABAD = (3, 6, 8)
_THUMB_JOINTS = (0, 1, 2)

MIRROR_RV = np.array([1.0, -1.0, -1.0])      # 右手轴 = 左手轴 ⊙ (1,−1,−1)，S2 Q1
SIDES = ("left", "right")

DATA_DIR = Path(__file__).resolve().parent / "data"
CALIB_PATH = DATA_DIR / "dexhand_calib.json"
# H1：关键点（MediaPipe）版标定，配 curl_rows_external 用；默认仍是上面那份（PEAR）
CALIB_MEDIAPIPE_PATH = DATA_DIR / "dexhand_calib_mediapipe.json"
# WiLoR 后端（hand_source=wilor）的四指标定：tools/dexhand_calib_wilor.py，D1 已有 WiLoR 结果
CALIB_WILOR_PATH = DATA_DIR / "dexhand_calib_wilor.json"
ROBOT_PATH = DATA_DIR / "dexhand_robot.json"

_FINGER_MODES = ("per_finger", "shared", "shrink")
_RIGHT_AXIS_MAX_DEG = 15.0                   # S2 实测 0.4–2.8°；整根轴写反会是 ~180°
_TIMING_RING = 1 << 14


class HandConfigError(ValueError):
    """标定或预设文件不合格。只在构造时抛。"""


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------

def so3_log(R: np.ndarray) -> np.ndarray:
    """(...,3,3) → (...,3) 旋转向量。搬自 tools/dexhand_explore.py（S2）。

    手指关节转角远小于 π，不处理 θ≈π 的分支。trace 先夹进 [−1,1]，
    所以非正交的坏矩阵给出的是有限的垃圾值，不是 NaN；那些值后面会被夹到 [0,1]。
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    th = np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    v = np.stack((R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]), axis=-1)
    s = np.sin(th)
    k = np.where(s > 1e-6, th / (2.0 * np.maximum(s, 1e-12)), 0.5)
    return v * k[..., None]


def so3_exp(rv: np.ndarray) -> np.ndarray:
    """(...,3) → (...,3,3)。只给测试造合成手用。"""
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    u = rv / np.maximum(th, 1e-12)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -u[..., 2], u[..., 1]
    K[..., 1, 0], K[..., 1, 2] = u[..., 2], -u[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -u[..., 1], u[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def curl_raw(hand_pose: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """做法 (a)（S2 Q3）：每节旋转向量投影到弯曲轴，每指三节求和。

    hand_pose (...,15,3,3)，axes (15,3)（**该人手自己的**轴表）。
    返回 (...,5)，PEAR 顺序 index/middle/pinky/ring/thumb，弧度，0 = 平手。
    """
    proj = np.einsum("...jk,jk->...j", so3_log(hand_pose), axes)
    return proj.reshape(proj.shape[:-1] + (5, 3)).sum(-1)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThumbCurve:
    """拇指参数 s = f(四指均值)。起步晚（start）、斜率陡（1/span），中性时微弯。"""
    kind: str
    start: float
    span: float

    def __call__(self, m):
        x = np.clip((np.asarray(m, dtype=np.float64) - self.start) / self.span, 0.0, 1.0)
        if self.kind == "smoothstep":
            return x * x * (3.0 - 2.0 * x)
        return x


@dataclass(frozen=True)
class ThumbFollowConfig:
    """拇指跟随（2026-10-23）：s = min(curve(四指均值), s_own)，s_own 由人拇指自己的特征算。

    特征 d = hand_keypoints.thumb_tuck（拇指尖到食指、中指 PIP / DIP 的最近距离 ÷ 手掌长）。
    d ≤ d_tuck → s_own = 1（扣在四指上，不限制），d ≥ d_out → s_own = 0（伸在外面），中间线性。
    只会让拇指比原来**更早张开**，闭合方向仍由 curve(四指均值) 封顶（B3 不变）。
    阈值由 D1 定（tools/dexhand_thumb_follow.py，docs/dexhand/THUMB_FOLLOW.md）。
    """
    d_tuck: float = 0.40
    d_out: float = 0.55

    def __post_init__(self):
        if not (0.0 < self.d_tuck < self.d_out):
            raise HandConfigError(f"拇指跟随阈值要 0 < d_tuck < d_out，给的是 {self.d_tuck} / {self.d_out}")

    def s_own(self, d):
        """特征 → s_own ∈ [0,1]；NaN 原样传出（调用方剔除）。"""
        return np.clip((self.d_out - np.asarray(d, dtype=np.float64)) / (self.d_out - self.d_tuck), 0.0, 1.0)


def parse_on_off(value, name: str) -> bool:
    """开关参数 → bool。`-p x:=on` 会被 ROS 按 YAML 读成布尔 true，`-p x:='on'` 才是字符串，
    所以 on / off / true / false（不分大小写）和 bool 都收；别的值 raise（构造时）。"""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("on", "true"):
        return True
    if v in ("off", "false"):
        return False
    raise ValueError(f"{name} 只能是 on / off（或 true / false），收到 {value!r}")


@dataclass(frozen=True)
class HandConfig:
    axes: dict                 # 人手侧 "left"/"right" → (15,3)
    endpoints: dict            # 人手侧 "left"/"right" → (4,2)，FINGERS 顺序 [c_open, c_fist]
    shrink: float              # per_finger → 0，shared → 1，shrink → 给定值
    finger_mode: str
    thumb_curve: ThumbCurve
    sigma_s: float
    neutral_c: float
    q_open: dict               # 机器人侧 "left"/"right" → (10,)
    q_fist: dict
    full_travel_s: float       # 四指，B2
    full_travel_s_thumb: float
    source: str
    sha256: str                # 标定文件
    robot_sha256: str
    gestures: dict = field(default_factory=dict)   # H6：名字 → (2,10) 机器人左右手预设（X2 接口）


def _fail(path, msg):
    raise HandConfigError(f"{path}: {msg}")


def _finite_array(path, name, value, shape):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail(path, f"{name} 不是数值数组")
    if a.shape != shape:
        _fail(path, f"{name} 形状 {a.shape}，应为 {shape}")
    if not np.all(np.isfinite(a)):
        _fail(path, f"{name} 含非有限值")
    return a


def _number(path, name, value, lo=-np.inf, hi=np.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"{name} 不是数值：{value!r}")
    v = float(value)
    if not np.isfinite(v) or not lo <= v <= hi:
        _fail(path, f"{name}={v} 不在 [{lo}, {hi}]")
    return v


def parse_calib(doc: dict, path="<calib>") -> dict:
    """校验人手侧标定（S4 产出 docs/dexhand/dexhand_calib.json 的格式）。"""
    if not isinstance(doc, dict):
        _fail(path, "顶层不是对象")
    if doc.get("version") != 1:
        _fail(path, f"version={doc.get('version')!r}，只认 1")
    if list(doc.get("joint_order", [])) != list(PEAR_JOINT_ORDER):
        _fail(path, f"joint_order 应为 {list(PEAR_JOINT_ORDER)}")
    axes, endpoints = {}, {}
    for side in SIDES:
        a = _finite_array(path, f"axes.{side}", doc.get("axes", {}).get(side), (15, 3))
        norm = np.linalg.norm(a, axis=1)
        if np.any(np.abs(norm - 1.0) > 1e-2):
            _fail(path, f"axes.{side} 不是单位向量（模长 {norm.min():.4f}–{norm.max():.4f}）")
        axes[side] = a
        ep = doc.get("endpoints", {}).get(side)
        if not isinstance(ep, dict):
            _fail(path, f"endpoints.{side} 缺失")
        rows = []
        for f in FINGERS:
            e = _finite_array(path, f"endpoints.{side}.{f}", ep.get(f), (2,))
            if not e[1] > e[0]:
                _fail(path, f"endpoints.{side}.{f}：c_fist {e[1]} 必须大于 c_open {e[0]}")
            rows.append(e)
        endpoints[side] = np.stack(rows)
    # 右手轴必须近似等于左手轴的镜像。整根轴写反（c 整体变负 → 手永远张开，
    # S2 Q7 点名的隐蔽故障）会在这里以 ~180° 被拦下。只查四指：拇指 S2 只差 1–8°，
    # 但拇指轴本来就不进计算。
    mirrored = axes["left"] * MIRROR_RV
    for j in range(12):
        cos = float(np.dot(mirrored[j], axes["right"][j])
                    / (np.linalg.norm(mirrored[j]) * np.linalg.norm(axes["right"][j])))
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if ang > _RIGHT_AXIS_MAX_DEG:
            _fail(path, f"axes.right[{PEAR_JOINT_ORDER[j]}] 与镜像左手轴差 {ang:.1f}° "
                        f"> {_RIGHT_AXIS_MAX_DEG}°（右手轴应 ≈ 左手轴 ⊙ (1,−1,−1)）")
    fm = doc.get("finger_mode", {})
    kind = fm.get("kind") if isinstance(fm, dict) else None
    if kind not in _FINGER_MODES:
        _fail(path, f"finger_mode.kind={kind!r}，应为 {_FINGER_MODES} 之一")
    if kind == "per_finger":
        shrink = 0.0
    elif kind == "shared":
        shrink = 1.0
    else:
        shrink = _number(path, "finger_mode.shrink", fm.get("shrink"), 0.0, 1.0)
    tc = doc.get("thumb_curve", {})
    if not isinstance(tc, dict) or tc.get("kind") not in ("smoothstep", "linear"):
        _fail(path, "thumb_curve.kind 应为 smoothstep 或 linear")
    curve = ThumbCurve(kind=tc["kind"],
                       start=_number(path, "thumb_curve.start", tc.get("start"), 0.0, 1.0),
                       span=_number(path, "thumb_curve.span", tc.get("span"), 1e-3, 1.0))
    flt = doc.get("filter", {})
    if not isinstance(flt, dict) or flt.get("kind") != "window_gauss_loclin":
        _fail(path, "filter.kind 只支持 window_gauss_loclin")
    sigma = _number(path, "filter.sigma_s", flt.get("sigma_s"), 1e-3, 1.0)
    neutral = _number(path, "neutral_c", doc.get("neutral_c"), 0.0, 1.0)
    return dict(axes=axes, endpoints=endpoints, shrink=shrink, finger_mode=kind,
                thumb_curve=curve, sigma_s=sigma, neutral_c=neutral,
                source=str(doc.get("source", "?")))


def parse_robot(doc: dict, path="<robot>") -> dict:
    """校验机器人侧预设（data/dexhand_robot.json）。"""
    if not isinstance(doc, dict):
        _fail(path, "顶层不是对象")
    if doc.get("version") != 1:
        _fail(path, f"version={doc.get('version')!r}，只认 1")
    if list(doc.get("joint_order", [])) != list(HAND_JOINTS):
        _fail(path, f"joint_order 应为 {list(HAND_JOINTS)}")
    q_open, q_fist = {}, {}
    for side in SIDES:
        o = _finite_array(path, f"open.{side}", doc.get("open", {}).get(side), (10,))
        f = _finite_array(path, f"fist.{side}", doc.get("fist", {}).get(side), (10,))
        for j in _ABAD:
            # 侧摆是常量：两端必须相等，否则「侧摆固定」这条已定决定被配置悄悄推翻
            if o[j] != f[j]:
                _fail(path, f"{side} {HAND_JOINTS[j]}：张开 {o[j]} ≠ 握拳 {f[j]}，侧摆必须固定")
        q_open[side], q_fist[side] = o, f
    gestures = {}
    for name, g in (doc.get("gestures") or {}).items():
        if not isinstance(g, dict):
            _fail(path, f"gestures.{name} 不是对象")
        gestures[name] = np.stack([_finite_array(path, f"gestures.{name}.{side}", g.get(side), (10,))
                                   for side in SIDES])
    return dict(q_open=q_open, q_fist=q_fist, gestures=gestures,
                full_travel_s=_number(path, "full_travel_s", doc.get("full_travel_s"), 0.1, 10.0),
                full_travel_s_thumb=_number(path, "full_travel_s_thumb",
                                            doc.get("full_travel_s_thumb"), 0.1, 10.0))


def load_config(calib_path=None, robot_path=None) -> HandConfig:
    calib_path = Path(calib_path) if calib_path else CALIB_PATH
    robot_path = Path(robot_path) if robot_path else ROBOT_PATH
    raw_c, raw_r = calib_path.read_bytes(), robot_path.read_bytes()
    try:
        doc_c, doc_r = json.loads(raw_c), json.loads(raw_r)
    except json.JSONDecodeError as e:
        raise HandConfigError(f"JSON 解析失败：{e}") from e
    c = parse_calib(doc_c, calib_path)
    r = parse_robot(doc_r, robot_path)
    return HandConfig(**c, **r, sha256=hashlib.sha256(raw_c).hexdigest(),
                      robot_sha256=hashlib.sha256(raw_r).hexdigest())


# ---------------------------------------------------------------------------
# 逐帧：手部姿态 → 一行 8 个归一化弯曲度
# ---------------------------------------------------------------------------

_NAN4 = np.full(4, np.nan)


def curl_rows(left_pose, right_pose, cfg: HandConfig, mirror: bool) -> np.ndarray:
    """→ [8]：机器人左手四指、机器人右手四指，归一化但**未夹紧**。

    缺失或任何非有限值 → 该机器人手 4 个 NaN。**不原地改输入**，不抛异常。
    """
    human = (right_pose, left_pose) if mirror else (left_pose, right_pose)
    sides = ("right", "left") if mirror else ("left", "right")
    out = np.empty(8)
    for r, (pose, side) in enumerate(zip(human, sides)):
        c = _NAN4
        if pose is not None:
            try:
                a = np.asarray(pose, dtype=np.float64)
            except (TypeError, ValueError):
                a = None
            if a is not None and a.size == 135:
                raw = curl_raw(a.reshape(15, 3, 3), cfg.axes[side])
                ep = cfg.endpoints[side]
                c = (raw[list(_PEAR_GROUP)] - ep[:, 0]) / (ep[:, 1] - ep[:, 0])
                if not np.all(np.isfinite(c)):
                    c = _NAN4
        out[4 * r:4 * r + 4] = c
    return out


def curl_rows_external(left_curl, right_curl, cfg: HandConfig, mirror: bool) -> np.ndarray:
    """外部弯曲度入口（H1）：每指原始弯曲度已经在别处算好（hand_keypoints.finger_curls）。

    → [8]，与 `curl_rows` **同一行格式**（机器人左手四指、机器人右手四指，归一化、未夹紧），
    之后进同一个平行缓冲、走同一条下游。left_curl / right_curl 是**人手侧**的，
    FINGERS 顺序至少 4 个，单位与 cfg.endpoints 相同（配 dexhand_calib_mediapipe.json）。
    第 5 个及以后（拇指）**不读**：拇指仍由 thumb_curve(四指均值) 走防撞路径。
    镜像、端点跟人手走，与 `curl_rows` 相同。缺失或非有限 → 该机器人手 4 个 NaN。
    不原地改输入，不抛异常。
    """
    human = (right_curl, left_curl) if mirror else (left_curl, right_curl)
    sides = ("right", "left") if mirror else ("left", "right")
    out = np.empty(8)
    for r, (curl, side) in enumerate(zip(human, sides)):
        c = _NAN4
        if curl is not None:
            try:
                a = np.asarray(curl, dtype=np.float64)
            except (TypeError, ValueError):
                a = None
            if a is not None and a.ndim == 1 and a.size >= 4:
                ep = cfg.endpoints[side]
                c = (a[:4] - ep[:, 0]) / (ep[:, 1] - ep[:, 0])
                if not np.all(np.isfinite(c)):
                    c = _NAN4
        out[4 * r:4 * r + 4] = c
    return out


# ---------------------------------------------------------------------------
# 选窗与平滑
# ---------------------------------------------------------------------------

def select_window(ts: np.ndarray, t_out: float, history_s: float,
                  lookahead_s: float, min_frames: int = 8) -> np.ndarray | None:
    """**逐字照抄** followme_core.FollowMeStream._live_target 的选窗段，返回下标。

    ts 是平行缓冲的时间戳（已按相机 stamp 升序，与手臂 `_buf` 逐位相同）。
    不累加 stretched_frames —— 那个计数归手臂。
    """
    lo, hi = t_out - history_s, t_out + lookahead_s
    idx = np.flatnonzero((lo - 1e-9 <= ts) & (ts <= hi + 1e-9))
    if idx.size and ts[idx[-1]] < t_out and ts.size:
        newest = ts[-1]
        if newest > t_out:
            idx = np.flatnonzero((lo - 1e-9 <= ts) & (ts <= newest + 1e-9))
    if idx.size < min_frames:
        return None
    if ts[idx[0]] > t_out or ts[idx[-1]] < t_out:
        return None
    return idx


def window_bounds(ts: np.ndarray, t_out: float, history_s: float,
                  lookahead_s: float) -> tuple[float, float]:
    """手臂这一 tick 选窗的**时间边界** (lo, hi)，含右窗撑开（H4）。

    撑开条件逐字照抄 `select_window`（= followme_core `_live_target`）：窗口里最后一帧
    早于 t_out、而缓冲里有晚于 t_out 的帧时，右边界撑到最新一帧。边界与饿不饿死无关。
    异步手部流（MediaPipe）的帧和 PEAR 帧不是同一批 t，所以手部按**这组边界**选自己的帧，
    而不是在自己的缓冲上另算一次撑开 —— 同一个 t_out 下两边看的是同一段时间（W1）。
    """
    lo, hi = t_out - history_s, t_out + lookahead_s
    idx = np.flatnonzero((lo - 1e-9 <= ts) & (ts <= hi + 1e-9))
    if idx.size and ts[idx[-1]] < t_out and ts.size:
        newest = ts[-1]
        if newest > t_out:
            hi = float(newest)
    return lo, hi


def select_in_bounds(ts: np.ndarray, t_out: float, lo: float, hi: float,
                     min_frames: int = 8) -> np.ndarray | None:
    """在给定边界里选帧，饿死规则同 `select_window`（< min_frames 帧、t_out 不在窗口内部 → None）。
    min_frames 默认 8；只有 WiLoR（约 5 fps）传 4（docs/dexhand/WILOR_LAPTOP_BENCH.md §6）。"""
    idx = np.flatnonzero((lo - 1e-9 <= ts) & (ts <= hi + 1e-9))
    if idx.size < min_frames:
        return None
    if ts[idx[0]] > t_out or ts[idx[-1]] < t_out:
        return None
    return idx


def window_loclin(ts: np.ndarray, rows: np.ndarray, t_out: float,
                  sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """σ 高斯加权局部线性回归，在 t_out 处取值（S2 Q6 的 `window_gauss`）。

    rows (n,8)。每只手单独剔除含 NaN 的行。返回 ((2,4) 值, (2,) 是否有效)。
    有效行 < 2、权重和几乎为 0，或该手的有效行**没有跨过 t_out 两侧** → 该手无效（调用方保持上一值）。
    最后一条是 H4 加的：异步手部流里一只手消失（detected=false → NaN 行）时，整窗仍可能跨过 t_out，
    但那只手的有效行全在左边 —— 不拦的话局部线性回归会沿最后的趋势**外插**，而这里绝不外插。
    """
    u = ts - t_out
    w_all = np.exp(-0.5 * (u / sigma) ** 2)
    val = np.zeros((2, 4))
    ok = np.zeros(2, dtype=bool)
    for h in range(2):
        x = rows[:, 4 * h:4 * h + 4]
        good = np.all(np.isfinite(x), axis=1)
        if good.sum() < 2:
            continue
        w, uu, xx = w_all[good], u[good], x[good]
        if uu.min() > 1e-9 or uu.max() < -1e-9:
            continue
        sw = w.sum()
        if not sw > 1e-6:
            continue
        su, suu = (w * uu).sum(), (w * uu * uu).sum()
        swx, suwx = w @ xx, (w * uu) @ xx
        det = sw * suu - su * su
        if det < 1e-12 * sw * sw:      # 退化（只剩一侧/一帧）→ 加权平均
            val[h] = swx / sw
        else:
            val[h] = (suu * swx - su * suwx) / det
        ok[h] = np.all(np.isfinite(val[h]))
    return val, ok


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# H6：预制手势层（关节空间，跟在弯曲跟踪后面）
# ---------------------------------------------------------------------------

GESTURE_LABELS = ("Victory", "Thumb_Up", "Pointing_Up")   # MediaPipe 自带标签里只认这三个（YEAH / 点赞 / one）
_PIPS, _THUMB_ABAD = (4, 5, 7, 9), 1
_THUMB_IDX = (0, 1, 2)
_FINGER_IDX = (3, 4, 5, 6, 7, 8, 9)


# 3d 路径的硬约束（不依赖安全表，2026-10-23 真机接触 real_20261023_200035：拇指 (−0.199, 1.0, −0.686) 横在掌心时，
# 四指 pip 命令 0.67、实测卡在 0.50 弯不动 → thumb_roll fault；安全表在这个姿态给 +10 mm，偏乐观）：
#   拇指 |abad| > T3_ABAD_IN 时，四指 pip 命令 ≤ T3_PIP_CAP；四指要再弯，拇指先退到 |abad| ≤ T3_ABAD_IN（沿到张开的直线等比缩回）
T3_ABAD_IN = 0.6
T3_PIP_CAP = 0.4
THUMB_3D_HANDS = {"left": (True, False), "right": (False, True), "both": (True, True)}   # 机器人手 [左, 右]
T3_MEDIAN = 3                  # 3d 拇指：最近几帧 map 输出取中值（5 fps 下约 0.4 s 的帧）


def _sync_step(d: np.ndarray, step: np.ndarray) -> np.ndarray:
    """拇指三关节同步：按最慢的关节等比缩放，关节空间走直线（逐关节截断时 mcp 先到位，
    拇指远节会扫过已握住的中指 —— 会话 1 fe5ca66）。每个关节仍 ≤ 各自步长。只在 3d 路径用。"""
    nz = np.abs(d) > 1e-12
    if not nz.any():
        return d.copy()
    r = float(np.min(step[nz] / np.abs(d[nz])))
    return d * min(1.0, r)


@dataclass(frozen=True)
class GestureConfig:
    """granger 2026-10-23 定：置信度超过阈值就 blend 过去，到了手势以后置信度低了就退回弯曲跟踪。"""
    theta_in: float = 0.7          # 进入：同一标签 score ≥ theta_in …
    theta_out: float = 0.5         # 退出：score < theta_out（或标签变了、手无效）…
    enter_s: float = 0.15          # … 连续这么久才进入（防单帧误判）
    exit_s: float = 0.3            # … 连续这么久才退出
    # 拇指顺序（按预设自动判断，不按名字写死）：预设里拇指往掌心折（|abad| > exit_thumb_abad）时，
    # 进入：预设里弯到握拳的手指都到目标的 fold_frac，拇指才开始折；退出：拇指 |abad| 回到 exit_thumb_abad 以内，四指才动
    fold_frac: float = 0.9
    exit_thumb_abad: float = 0.3

    def __post_init__(self):
        if not 0.0 <= self.theta_out < self.theta_in <= 1.0:
            raise HandConfigError(f"手势阈值要 0 ≤ theta_out < theta_in ≤ 1（迟滞），给的是 "
                                  f"{self.theta_out} / {self.theta_in}")
        if not (self.enter_s >= 0 and self.exit_s >= 0):
            raise HandConfigError("enter_s / exit_s 必须 ≥ 0")


class _GestureHand:
    """一只机器人手的手势状态。"""

    def __init__(self):
        self.obs: deque = deque()      # (t, label, score, valid)，按 t 升序
        self.active: str | None = None
        self.pend: str | None = None
        self.pend_since = 0.0
        self.bad_since: float | None = None
        self.last_t = -np.inf          # 已处理到的最后一条观测
        self.mode = "track"            # track（直通）/ gesture（走向或停在预设）/ exit（限速回跟踪值）
        self.exit_from: str | None = None
        self.cur: str | None = None    # 正在走向或停在的那个手势
        self.q: np.ndarray | None = None
        self.late = 0                  # 比已处理的还早的观测（晚到），丢弃计数
        self.n_enter = 0


class HandStream:
    """把平行缓冲里的弯曲度变成 50 Hz 的 [20] 关节角。由 FollowMeStream 持有。

    `emit` 只收三个标量（t_out / weight / force_neutral），**不持有任何手臂状态**。
    """

    def __init__(self, cfg: HandConfig, *, mirror: bool, history_s: float,
                 lookahead_s: float, out_hz: float, buf_span: float, stale_s: float = 1.0,
                 gesture: GestureConfig | None = None,
                 thumb_follow: ThumbFollowConfig | None = None, min_frames: int = 8,
                 thumb_3d=None, thumb_3d_hands: str = "left",
                 gesture_labels=GESTURE_LABELS) -> None:
        if cfg.sigma_s > lookahead_s / 2 + 1e-12:
            # S2 Q6：σ 超过前瞻一半时窗口右侧截断会引入滞后，手会晚于手臂
            raise HandConfigError(f"sigma_s={cfg.sigma_s} > lookahead_s/2={lookahead_s / 2}")
        self.cfg = cfg
        self.min_frames = int(min_frames)      # 选窗饿死线；8 = 原样，WiLoR 传 4
        # 3D 拇指（WiLoR 冲刺，thumb_mode:=3d）：会话 1 的 thumb_map.ThumbMapper；None = 关，行为逐位同前
        self.thumb_3d = thumb_3d
        # 哪只机器人手走 3d 拇指（left / right / both）；其余的手拇指保持线段 / 拇指跟随的值，逐位同 follow。
        # 默认 left：右手的镜像取反和 guard_pose 从没在真机上验证过（2026-10-23 第一次装右手）
        if thumb_3d_hands not in THUMB_3D_HANDS:
            raise HandConfigError(f"thumb_3d_hands 只能是 {tuple(THUMB_3D_HANDS)}，收到 {thumb_3d_hands!r}")
        self._t3_on = np.array(THUMB_3D_HANDS[thumb_3d_hands], dtype=bool)
        self._kbuf: deque = deque()            # (t, kp (2,21,3) 机器人手顺序, chirality (2,))
        self._t3_prev: np.ndarray | None = None
        self._t3_cprev: np.ndarray | None = None       # 上一 tick 发出的四指均值（guard 查 tick 之间那一段）
        self._t3_pprev: np.ndarray | None = None       # 上一 tick 3d 层发出的四指 pip (2,4)（guard_pose 用）
        self._t3_final: np.ndarray | None = None       # 上一 tick 手势层之后最终发出的 (2,10)
        self.t3_gguard = np.zeros(2, dtype=np.int64)    # 手势层之后 guard_pose 又改写拇指的 tick
        self.t3_capped = np.zeros(2, dtype=np.int64)    # 硬约束：拇指在掌心、四指 pip 被压到上限的 tick
        self.t3_retract = np.zeros(2, dtype=np.int64)   # 硬约束：四指要弯、拇指被要求先退的 tick
        self._t3_want = np.zeros(2, dtype=bool)         # 本 tick 四指目标是否超过 pip 上限（限速前）
        self.t3_fcap = np.zeros(2, dtype=np.int64)      # finger_caps（会话 1）压了四指的 tick（跟踪 + 手势后）
        self._finger_caps = None
        if thumb_3d is not None:
            from .thumb_safety import finger_caps
            self._finger_caps = finger_caps
        self.t3_ok = np.zeros(2, dtype=np.int64)        # 3D 映射生效的 tick
        self.t3_fallback = np.zeros(2, dtype=np.int64)  # 退回线段 / 拇指跟随的 tick
        self.t3_guard = np.zeros(2, dtype=np.int64)     # guard 改写了要发的拇指的 tick
        if thumb_3d is not None:
            trav = np.abs(np.stack([cfg.q_fist[s_] - cfg.q_open[s_] for s_ in SIDES]))[:, list(_THUMB_IDX)]
            self._t3_step = np.maximum(trav, 1e-6) / (cfg.full_travel_s_thumb * out_hz)
        self.mirror = mirror
        self.history_s = history_s
        self.lookahead_s = lookahead_s
        self.buf_span = buf_span
        # H4：某只手的数据无效（流断 / 没检出）超过 stale_s → 该手目标改成中性，经 B2 限速回去。
        # 「无效多久」在输出时刻看：t_out − 该手 stamp ≤ t_out 的最新一帧有效数据。只看 ≤ t_out 的，
        # 不看已经到了、但还在 t_out 之后的帧：否则断流后新数据一到、还没进窗口，
        # 这只手就会从中性跳回保持的旧姿态，再去追新数据（一次真实的抽动）。
        if not stale_s > 0:
            raise HandConfigError(f"stale_s={stale_s} 必须 > 0")
        self.stale_s = float(stale_s)
        # H6：手势层（None = 关，行为与之前逐位相同）。预设和关节步长在构造时算好。
        self.gesture = gesture
        # IB-Robot patch: the recognised labels come from the caller (default: upstream's three)
        self.gesture_labels = tuple(gesture_labels)
        if gesture is not None:
            missing = [n for n in self.gesture_labels if n not in cfg.gestures]
            if missing:
                raise HandConfigError(f"开了手势层，但 dexhand_robot.json 的 gestures 里没有 {missing}")
            self._g_preset = {n: cfg.gestures[n] for n in self.gesture_labels}          # (2,10)
            # 关节步长：和播放端 hand_limits.hand_step_limit 同一个公式（全行程 / B2 时间），侧摆按 pip 的速度
            travel = np.abs(np.stack([cfg.q_fist[s_] - cfg.q_open[s_] for s_ in SIDES]))
            travel[:, list(_ABAD)] = np.abs(travel[:, 4:5]).max(axis=1, keepdims=True)
            tt = np.array([cfg.full_travel_s_thumb if j in _THUMB_IDX else cfg.full_travel_s for j in range(10)])
            self._g_step = travel / (tt * out_hz)
            # 每个预设、每只手：拇指折不折进掌心；折的话要等哪几根手指（预设里弯到握拳 ≥ 90% 的 pip）
            fist = np.stack([cfg.q_fist[s_] for s_ in SIDES])
            self._g_fold = {n: [([j for j in _PIPS if abs(p[h][j]) >= 0.9 * abs(fist[h][j])]
                                 if abs(p[h][_THUMB_ABAD]) > gesture.exit_thumb_abad else None)
                                for h in range(2)] for n, p in self._g_preset.items()}
        # 拇指跟随（None = 关，行为与之前逐位相同）。特征走自己的平行缓冲 _tbuf，选窗、平滑同四指
        self.thumb_follow = thumb_follow
        self.step = np.array([1.0 / (cfg.full_travel_s * out_hz)] * 4
                             + [1.0 / (cfg.full_travel_s_thumb * out_hz)])

        self._q_open = np.stack([cfg.q_open[s] for s in SIDES])            # (2,10)
        self._q_span = np.stack([cfg.q_fist[s] - cfg.q_open[s] for s in SIDES])
        self._q_lo = np.minimum(self._q_open, self._q_open + self._q_span)
        self._q_hi = np.maximum(self._q_open, self._q_open + self._q_span)
        self._q_lo = np.minimum(self._q_lo, np.stack([cfg.q_fist[s] for s in SIDES]))
        self._q_hi = np.maximum(self._q_hi, np.stack([cfg.q_fist[s] for s in SIDES]))

        pn = np.empty((2, N_PARAM))
        pn[:, :4] = cfg.neutral_c
        pn[:, THUMB] = cfg.thumb_curve(cfg.neutral_c)
        self.p_neutral = pn
        self.q_neutral = self.map(pn)          # [20]，之后不再重算

        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None
        self.emit_ns = np.zeros(_TIMING_RING, dtype=np.int64)
        self.push_ns = np.zeros(_TIMING_RING, dtype=np.int64)
        self.n_emit = 0
        self.n_push = 0
        self.reset()

    def reset(self) -> None:
        """会话级状态全清（同 FollowMeStream.reset 的 `_buf` 归类）。"""
        self._buf.clear()
        self._arrays = None
        self._tbuf: deque[tuple[float, np.ndarray]] = deque()
        self._tarrays: tuple[np.ndarray, np.ndarray] | None = None
        self.s_own_last: np.ndarray | None = None      # 只给检查工具看：本 tick 平滑后的 s_own（NaN = 无效）
        self._last_live: list[np.ndarray | None] = [None, None]
        self._prev = self.p_neutral.copy()
        self.live_last: np.ndarray | None = None
        self.starved_ticks = 0
        self.hand_hold_ticks = np.zeros(2, dtype=np.int64)
        self.limit_trips = np.zeros(N_PARAM, dtype=np.int64)
        self.stale_ticks = np.zeros(2, dtype=np.int64)
        self.stale_last = np.zeros(2, dtype=bool)      # 只给检查工具看
        self._g = [_GestureHand(), _GestureHand()] if self.gesture is not None else None

    # -- 输入 -----------------------------------------------------------------
    def curl(self, left_pose, right_pose) -> np.ndarray:
        t0 = time.perf_counter_ns()
        row = curl_rows(left_pose, right_pose, self.cfg, self.mirror)
        self.push_ns[self.n_push % _TIMING_RING] = time.perf_counter_ns() - t0
        self.n_push += 1
        return row

    def curl_external(self, left_curl, right_curl) -> np.ndarray:
        """`curl` 的外部弯曲度版（H1）：人手侧每指原始弯曲度 → 一行 [8]，交给 `push`。"""
        t0 = time.perf_counter_ns()
        row = curl_rows_external(left_curl, right_curl, self.cfg, self.mirror)
        self.push_ns[self.n_push % _TIMING_RING] = time.perf_counter_ns() - t0
        self.n_push += 1
        return row

    def push(self, t: float, row: np.ndarray | None) -> None:
        """与 followme_core.FollowMeStream.push_pose 的 `_buf` 操作**同一套规则、同一个 t**。

        row 为 None（老工具直接 push_pose、SourceFrame 不带手）也要入一行 NaN，
        否则两个缓冲的长度就对不上了。
        """
        row = np.full(8, np.nan) if row is None else np.asarray(row, dtype=np.float64)
        if self._buf and t < self._buf[-1][0]:
            items = list(self._buf)
            items.append((t, row))
            items.sort(key=lambda it: it[0])
            self._buf = deque(items)
        else:
            self._buf.append((t, row))
        cutoff = t - self.buf_span
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        self._arrays = None

    def push_thumb(self, t: float, d) -> None:
        """拇指跟随：一帧的拇指特征（**机器人手**顺序 [2]，镜像已在调用方换好；无效给 NaN）。
        入 `_tbuf`，插入排序和裁剪规则同 `push`。拇指跟随关着时什么都不做。"""
        if self.thumb_follow is None:
            return
        try:
            d = np.asarray(d, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            d = np.full(2, np.nan)
        row = self.thumb_follow.s_own(d)
        if self._tbuf and t < self._tbuf[-1][0]:
            items = sorted(list(self._tbuf) + [(t, row)], key=lambda it: it[0])
            self._tbuf = deque(items)
        else:
            self._tbuf.append((t, row))
        cutoff = t - self.buf_span
        while self._tbuf and self._tbuf[0][0] < cutoff:
            self._tbuf.popleft()
        self._tarrays = None

    def _s_own(self, t_out: float, bounds=None) -> np.ndarray:
        """(2,) 平滑后的 s_own：和四指同一套选窗、同一个 σ 的局部线性回归（不外插）；无效 → NaN。"""
        if self._tarrays is None:
            if self._tbuf:
                self._tarrays = (np.array([t for t, _ in self._tbuf], dtype=np.float64),
                                 np.stack([r for _, r in self._tbuf]))
            else:
                self._tarrays = (np.zeros(0), np.zeros((0, 2)))
        ts, rows = self._tarrays
        out = np.full(2, np.nan)
        idx = self._select(ts, t_out, bounds)
        if idx is not None:
            # 每只手的 s_own 复制成 4 列，借 window_loclin 的「一只手 4 个数」口径，算术逐位同四指
            val, ok = window_loclin(ts[idx], np.repeat(rows[idx], 4, axis=1), t_out, self.cfg.sigma_s)
            out[ok] = np.clip(val[ok, 0], 0.0, 1.0)
        return out

    def buffer_times(self) -> list[float]:
        return [t for t, _ in self._buf]

    def _window_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is None:
            if self._buf:
                ts = np.array([t for t, _ in self._buf], dtype=np.float64)
                rows = np.stack([r for _, r in self._buf])
            else:
                ts, rows = np.zeros(0), np.zeros((0, 8))
            self._arrays = (ts, rows)
        return self._arrays

    # -- 输出 -----------------------------------------------------------------
    def _select(self, ts, t_out, bounds):
        """bounds=None：在自己的缓冲上按手臂规则选（S3，手部帧与手臂帧同一批 t）；
        给了 (lo, hi)：按手臂这一 tick 的边界选（H4 异步手部流）。"""
        if bounds is None:
            return select_window(ts, t_out, self.history_s, self.lookahead_s, self.min_frames)
        return select_in_bounds(ts, t_out, bounds[0], bounds[1], self.min_frames)

    def window(self, t_out: float, bounds=None) -> np.ndarray | None:
        """t_out 处选中的窗口时间戳（W1 用）；饿死 → None。"""
        ts, _ = self._window_arrays()
        idx = self._select(ts, t_out, bounds)
        return None if idx is None else ts[idx]

    def _mix(self, c: np.ndarray) -> np.ndarray:
        m = c.mean()
        return m + (1.0 - self.cfg.shrink) * (c - m)

    def _live(self, t_out: float, bounds=None) -> np.ndarray:
        """(2,5) 的活动目标。饿死/某手无效 → 保持该手上一次的有效值，没有就用中性。"""
        ts, rows = self._window_arrays()
        idx = self._select(ts, t_out, bounds)
        if idx is None:
            self.starved_ticks += 1
            ok = np.zeros(2, dtype=bool)
            val = None
        else:
            val, ok = window_loclin(ts[idx], rows[idx], t_out, self.cfg.sigma_s)
        s_own = self._s_own(t_out, bounds) if self.thumb_follow is not None else None
        self.s_own_last = s_own
        live = np.empty((2, N_PARAM))
        for h in range(2):
            if ok[h]:
                c = self._clip_params(self._mix(val[h]))
                live[h, :4] = c
                live[h, THUMB] = self.cfg.thumb_curve(c.mean())
                if s_own is not None and np.isfinite(s_own[h]):
                    # 拇指跟随：只往张开方向改（≤ 原值）；s_own 无效 → 原值
                    live[h, THUMB] = min(live[h, THUMB], s_own[h])
                self._last_live[h] = live[h].copy()
            else:
                self.hand_hold_ticks[h] += 1
                held = self._last_live[h]
                live[h] = held if held is not None else self.p_neutral[h]
        # 数据无效太久 → 这只手的目标改成中性（限速在 _limit 里，所以是单调地回去、恢复时也不跳）。
        # 写在保持之后：_last_live 不动，数据回来时从当前位置在限速内追回 live 值。
        stale = (t_out - self._last_valid_le(ts, rows, t_out)) > self.stale_s
        for h in np.flatnonzero(stale):
            live[h] = self.p_neutral[h]
            self.stale_ticks[h] += 1
        self.stale_last = stale
        self.live_last = live          # 只给检查工具看（C2/D2），不参与计算
        return live

    @staticmethod
    def _last_valid_le(ts: np.ndarray, rows: np.ndarray, t_out: float) -> np.ndarray:
        """(2,)：每只机器人手 stamp ≤ t_out 的最新一帧有效数据的 t；缓冲里没有 → −inf。"""
        out = np.full(2, -np.inf)
        if ts.size:
            past = ts <= t_out + 1e-9
            for h in range(2):
                ok = past & np.all(np.isfinite(rows[:, 4 * h:4 * h + 4]), axis=1)
                if ok.any():
                    out[h] = ts[ok].max()
        return out

    def _limit(self, target: np.ndarray) -> np.ndarray:
        """参数空间限速。先限四指，拇指目标再压到 curve(限速后四指均值) 以下，然后限拇指
        → 四指被限速卡住时拇指不会在闭合方向抢到前面（B3）。"""
        prev = self._prev
        out = np.empty_like(target)
        d = target[:, :4] - prev[:, :4]
        hit = np.abs(d) > self.step[:4]
        self.limit_trips[:4] += hit.sum(axis=0)
        out[:, :4] = prev[:, :4] + np.clip(d, -self.step[:4], self.step[:4])
        thumb_goal = np.minimum(target[:, THUMB], self.cfg.thumb_curve(out[:, :4].mean(axis=1)))
        d = thumb_goal - prev[:, THUMB]
        self.limit_trips[THUMB] += int((np.abs(d) > self.step[THUMB]).sum())
        out[:, THUMB] = prev[:, THUMB] + np.clip(d, -self.step[THUMB], self.step[THUMB])
        self._prev = out
        return out

    def map(self, p: np.ndarray) -> np.ndarray:
        """(2,5) 参数 → [20] 关节角，按机器人左右手各自的预设；结果夹在两端预设之间。"""
        p6 = np.concatenate([p, np.zeros((2, 1))], axis=1)
        q = self._q_open + p6[:, _DRIVE] * self._q_span
        return self._clip_q(q).reshape(HAND_DOF)

    # 两处夹紧单独成方法：B1 的全部保证都在这里，检查工具的 no_clip 变异就打在这两处
    @staticmethod
    def _clip_params(p: np.ndarray) -> np.ndarray:
        return np.clip(p, 0.0, 1.0)

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self._q_lo, self._q_hi)

    def emit(self, t_out: float, weight: float, force_neutral: bool, bounds=None) -> np.ndarray:
        """bounds：手臂这一 tick 的选窗边界 (lo, hi)（`window_bounds`），followme_core 每 tick 给；
        None = 旧口径，在自己的缓冲上按手臂规则选（离线工具和 S3 的单元门用）。"""
        t0 = time.perf_counter_ns()
        if force_neutral:
            # park / final：**逐位**等于 q_neutral，靠这一行，不靠算术恰好算回来
            self._prev = self.p_neutral.copy()
            q = self.q_neutral.copy()
            self._t3_prev = None
            self._t3_cprev = None
            self._t3_pprev = None
            self._t3_final = None
        else:
            if weight <= 0.0:
                target = self.p_neutral.copy()
            else:
                live = self._live(t_out, bounds)
                # 与手臂 `base + weight·(live − base)` 同一个公式、同一个 weight
                target = self.p_neutral + weight * (live - self.p_neutral)
            target = self._clip_params(target)          # B1；clip 会把 NaN 原样放过，上游已剔除
            if self.thumb_3d is not None:
                target = self._t3_finger_cap(target)
            q = self.map(self._limit(target))
            if self.thumb_3d is not None:
                q = self._thumb3d_layer(t_out, q, weight)
        if self._g is not None:
            q = self._gesture_layer(t_out, q, weight, force_neutral)
            if self.thumb_3d is not None and not force_neutral:
                q = self._thumb3d_post(q)
        self.emit_ns[self.n_emit % _TIMING_RING] = time.perf_counter_ns() - t0
        self.n_emit += 1
        return q

    # -- 3D 拇指（WiLoR 冲刺）---------------------------------------------------
    def push_kp(self, t: float, kps, chirality) -> None:
        """一帧 21 点 3D 关键点，**机器人手**顺序（镜像已在调用方换好）：kps (2,21,3)（没检出 = NaN），
        chirality = 每只机器人手对应的人手侧 "left"/"right"（thumb_map 的约定）。"""
        if self.thumb_3d is None:
            return
        self._kbuf.append((float(t), np.asarray(kps, dtype=np.float64).reshape(2, 21, 3), tuple(chirality)))
        if len(self._kbuf) > 1 and t < self._kbuf[-2][0]:
            self._kbuf = deque(sorted(self._kbuf, key=lambda it: it[0]))
        cutoff = t - self.buf_span
        while self._kbuf and self._kbuf[0][0] < cutoff:
            self._kbuf.popleft()

    def _thumb3d_layer(self, t_out: float, q: np.ndarray, weight: float) -> np.ndarray:
        """线段 / 拇指跟随算出的拇指 q 之后：每只手取 stamp ≤ t_out 的最新一帧关键点（不外插），
        thumb_map 出安全格点上的 (roll, abad, mcp)，按 weight 和线段值混合，关节空间限速（拇指 B2 步长）。
        映射不了（没检出 / 超过 stale_s / 不 ok）→ 目标退回线段值，同样限速过去。"""
        out = q.reshape(2, 10).copy()
        line = out[:, list(_THUMB_IDX)].copy()
        c = self._prev[:, :4].mean(axis=1)
        c_prev = self._t3_cprev if self._t3_cprev is not None else c
        # stamp ≤ t_out、未超时的最近 T3_MEDIAN 帧，各自 map 后逐关节取中值：WiLoR 偶尔单帧把拇指看成
        # 横在掌心（D1 张开段 4–27% 的帧），单帧翻转不该被拇指 B2 追过去（会话 1 fcf3945 的分析）
        frames = [it for it in self._kbuf if it[0] <= t_out + 1e-9 and t_out - it[0] <= self.stale_s][-T3_MEDIAN:]
        pips = out[:, list(_PIPS)]
        pips_prev = self._t3_pprev if self._t3_pprev is not None else pips
        prev = self._t3_prev if self._t3_prev is not None else line.copy()
        for h in range(2):
            if not self._t3_on[h]:
                prev[h] = line[h]                  # 这只手不走 3d：原样用线段 / 拇指跟随的值
                continue
            tgt = line[h]
            ths = []
            for fr in frames:
                if not np.all(np.isfinite(fr[1][h])):
                    continue
                try:
                    res = self.thumb_3d.map(fr[1][h], fr[2][h], float(np.clip(c[h], 0.0, 1.0)))
                except Exception:      # noqa: BLE001 —— 映射出错：这一帧不算
                    res = None
                if res is not None and res.ok and res.thumb is not None:
                    ths.append(np.asarray(res.thumb, dtype=np.float64))
            if ths:
                th = np.median(np.stack(ths), axis=0)
                if SIDES[h] == "right":
                    th = -th           # 机器人右手：X2 左手约定的下标 0/1/2 取反（HANDOFF §2.5，未验证）
                tgt = line[h] + weight * (th - line[h])
                self.t3_ok[h] += 1
            else:
                self.t3_fallback[h] += 1
            # 硬约束的拇指一半：四指想弯过上限、或已经弯过上限 → 拇指目标 |abad| ≤ T3_ABAD_IN（沿到张开的直线缩回）
            if (self._t3_want[h] or np.any(pips[h] > T3_PIP_CAP + 1e-9)) and abs(tgt[1]) > T3_ABAD_IN:
                tgt = tgt * (T3_ABAD_IN / abs(tgt[1]))
                self.t3_retract[h] += 1
            last = prev[h].copy()
            send = prev[h] + _sync_step(tgt - prev[h], self._t3_step[h])
            # 会话 1 的 guard（b034f88）：限速 + 混合之后，按**这一 tick 要发的**四指均值再判一次；
            # 不安全 → 朝当前 c 下最近的安全格点退（每 tick ≤ 0.12 rad，可超 B2，刻意的）；None → 线段值。
            # 安全表是 X2 左手约定，机器人右手进出各取反一次。
            sg = -1.0 if SIDES[h] == "right" else 1.0
            try:
                # fcf3945：guard_pose 按四指逐根 pip 判（guard 用均值 c，W6 A 段漏掉 5 个 tick）
                g = self.thumb_3d.guard_pose(sg * send, pips[h], sg * last, pips_prev[h])
            except Exception:      # noqa: BLE001
                g = None
            if g is None:
                self.t3_guard[h] += 1
                send = line[h].copy()
            else:
                g = sg * np.asarray(g, dtype=np.float64)
                if not np.array_equal(g, send):
                    self.t3_guard[h] += 1
                send = g
            prev[h] = send
        self._t3_prev = prev
        self._t3_cprev = c.copy()
        self._t3_pprev = pips.copy()
        out[:, list(_THUMB_IDX)] = prev
        return self._clip_q(out).reshape(HAND_DOF)

    def _pip_of(self, p4: np.ndarray, h: int) -> np.ndarray:
        """四指参数 → 四个 pip 的 |关节角|（map 的同一个换算）。"""
        return np.abs(self._q_open[h, list(_PIPS)] + p4 * self._q_span[h, list(_PIPS)])

    def _t3_finger_cap(self, target: np.ndarray) -> np.ndarray:
        """硬约束的四指一半（参数空间、限速之前，所以压下来也按 B2 走）：上一 tick 发出的拇指 |abad| > T3_ABAD_IN
        → 这只手四指目标 pip ≤ T3_PIP_CAP。同时记下四指「想弯过上限」，给拇指那一半用。"""
        out = target.copy()
        for h in range(2):
            if not self._t3_on[h]:
                continue
            want = self._pip_of(target[h, :4], h)
            self._t3_want[h] = bool(np.any(want > T3_PIP_CAP + 1e-9))
            if self._finger_caps is not None and self._t3_final is not None:
                # 会话 1 的 finger_caps（d41a89b）：按上一 tick **实际发出**的拇指和四指 pip 给上限（已弯过去的不限）。
                # 加在参数空间、限速之前，所以压下来、放开都按 B2 走
                sg = -1.0 if SIDES[h] == "right" else 1.0
                fin = self._t3_final[h]
                caps = self._finger_caps(sg * fin[list(_THUMB_IDX)], np.abs(fin[list(_PIPS)]))
                cp = (caps - np.abs(self._q_open[h, list(_PIPS)])) / np.abs(self._q_span[h, list(_PIPS)])
                if np.any(out[h, :4] > cp + 1e-12):
                    self.t3_fcap[h] += 1
                out[h, :4] = np.minimum(out[h, :4], cp)
            if self._t3_prev is not None and abs(self._t3_prev[h][1]) > T3_ABAD_IN + 1e-9:
                span = np.abs(self._q_span[h, list(_PIPS)])
                cap_p = (T3_PIP_CAP - np.abs(self._q_open[h, list(_PIPS)])) / span
                if np.any(out[h, :4] > cap_p):
                    self.t3_capped[h] += 1
                out[h, :4] = np.minimum(out[h, :4], cap_p)
        return out

    def _thumb3d_post(self, q: np.ndarray) -> np.ndarray:
        """3d 路径：手势层可能又改了拇指 / 四指，最终要发的值再过一次 guard_pose（相对上一 tick 最终值）。"""
        out = q.reshape(2, 10).copy()
        prev = self._t3_final if self._t3_final is not None else out.copy()
        ti, pi = list(_THUMB_IDX), list(_PIPS)
        for h in range(2):
            if not self._t3_on[h]:
                continue
            sg = -1.0 if SIDES[h] == "right" else 1.0
            try:
                g = self.thumb_3d.guard_pose(sg * out[h, ti], out[h, pi], sg * prev[h, ti], prev[h, pi])
            except Exception:      # noqa: BLE001
                g = None
            if g is not None:
                g = sg * np.asarray(g, dtype=np.float64)
                if not np.array_equal(g, out[h, ti]):
                    self.t3_gguard[h] += 1
                    out[h, ti] = g
        # 手势预设也过 finger_caps（3d 路径）：按这一 tick 要发的拇指、上一 tick 实际发出的 pip。只往下压四指；
        # 压了就写回手势层的状态，免得下一 tick 手势层从没压过的值接着走（放开时一步跳回去）
        if self._finger_caps is not None and self._g is not None:
            for h in range(2):
                if not self._t3_on[h] or self._g[h].mode == "track":
                    continue
                sg = -1.0 if SIDES[h] == "right" else 1.0
                caps = self._finger_caps(sg * out[h, ti], np.abs(prev[h, pi]))
                cur = out[h, pi]
                capped = np.sign(cur) * np.minimum(np.abs(cur), caps)
                if np.any(capped != cur):
                    self.t3_fcap[h] += 1
                    out[h, pi] = capped
                    self._g[h].q[pi] = capped
        # 这里不做 _clip_q：手势预设本来就可以在「张开↔握拳」区间之外（点赞的侧摆、拇指 roll −0.562），
        # 夹回去会改掉预设（thumb_3d_hands 的 follow 对照门抓到的）。guard_pose 的返回值是查找表格点，
        # 本身在硬件范围内；播放端还会按 hand_limits 再夹一次
        self._t3_final = out.copy()
        return out.reshape(HAND_DOF)

    # -- H6 手势层 ------------------------------------------------------------
    def push_gesture(self, t: float, obs) -> None:
        """一帧的手势观测，**机器人手**顺序（镜像已在调用方换好）：[(label, score, valid)] × 2。
        按 t 插入；比已处理的还早的（晚到）丢掉并计数。"""
        if self._g is None:
            return
        for h in range(2):
            label, score, valid = obs[h]
            g = self._g[h]
            if t <= g.last_t:
                g.late += 1
                continue
            item = (float(t), str(label or ""), float(score) if np.isfinite(score) else 0.0, bool(valid))
            if g.obs and t < g.obs[-1][0]:
                g.obs = deque(sorted(list(g.obs) + [item], key=lambda x: x[0]))
            else:
                g.obs.append(item)

    def _gesture_update(self, h: int, t_out: float, allowed: bool) -> None:
        """处理 t ≤ t_out 的观测，更新该手的 active（迟滞 + 持续时间）。"""
        c, g = self.gesture, self._g[h]
        while g.obs and g.obs[0][0] <= t_out + 1e-9:
            t, label, score, valid = g.obs.popleft()
            g.last_t = t
            cand = label if (valid and label in self.gesture_labels and score >= c.theta_in) else None
            if g.active is None:
                if cand is None:
                    g.pend = None
                elif cand != g.pend:
                    g.pend, g.pend_since = cand, t
                if allowed and g.pend is not None and t - g.pend_since >= c.enter_s - 1e-9:
                    g.active, g.pend, g.bad_since = g.pend, None, None
                    g.n_enter += 1
            else:
                keep = valid and label == g.active and score >= c.theta_out
                if keep:
                    g.bad_since = None
                else:
                    if g.bad_since is None:
                        g.bad_since = t
                    if t - g.bad_since >= c.exit_s - 1e-9:
                        g.active, g.bad_since = None, None
                        g.pend = cand
                        g.pend_since = t
        # 没有新观测（流断了）也要能退出：最后一条观测之后 exit_s 还没来新的 → 退出
        if g.active is not None and t_out - g.last_t >= c.exit_s - 1e-9:
            g.active, g.bad_since, g.pend = None, None, None
        if not allowed and g.active is not None:
            g.active, g.bad_since, g.pend = None, None, None

    def _gesture_layer(self, t_out: float, q: np.ndarray, weight: float, force_neutral: bool) -> np.ndarray:
        """跟在弯曲跟踪后面：手势 active → 关节空间限速走向预设；退出 → 限速回到跟踪值，追上后直通。
        park/final、非 live 相位（weight < 1）、该手数据超时都不允许手势。"""
        out = q.copy()
        for h in range(2):
            g = self._g[h]
            sl = slice(10 * h, 10 * h + 10)
            allowed = (not force_neutral) and weight >= 1.0 and not bool(self.stale_last[h])
            self._gesture_update(h, t_out, allowed)
            if force_neutral:
                g.mode, g.q, g.exit_from = "track", q[sl].copy(), None       # park/final：逐位直通
                continue
            if g.mode == "track" and g.active is None:
                g.q = q[sl].copy()
                continue
            if g.q is None:
                g.q = q[sl].copy()     # IB-Robot patch: a gesture already active on the very first emit
            if g.active is not None:
                g.mode, g.cur, g.exit_from = "gesture", g.active, None
                goal = self._g_preset[g.active][h]
            else:
                if g.mode == "gesture":
                    g.mode, g.exit_from = "exit", g.cur
                goal = q[sl]
            step = self._g_step[h]
            d = goal - g.q
            move = np.clip(d, -step, step)
            if self.thumb_3d is not None and self._t3_on[h]:
                move[list(_THUMB_IDX)] = _sync_step(d[list(_THUMB_IDX)], step[list(_THUMB_IDX)])
            fold = self._g_fold[g.cur][h] if g.cur is not None else None
            if g.mode == "gesture" and fold:
                frac = self.gesture.fold_frac
                if not all(abs(g.q[j]) >= frac * abs(goal[j]) - 1e-12 for j in fold):
                    move[list(_THUMB_IDX)] = 0.0                    # 被拇指压住的手指先弯到 90%，拇指才开始折
            if g.mode == "exit" and self._g_fold[g.exit_from][h] is not None:
                if abs(g.q[_THUMB_ABAD]) > self.gesture.exit_thumb_abad + 1e-12:
                    move[list(_FINGER_IDX)] = 0.0                   # 拇指先离开（|abad| ≤ 0.3），四指再动
            # 一步够得着就直接落到目标（逐位），否则走一个步长；被顺序规则卡住的关节不动
            reach = (np.abs(d) <= step) & (move == d)
            nq = np.where(reach, goal, g.q + move)
            if g.mode == "exit" and np.array_equal(nq, q[sl]):
                g.mode, g.exit_from = "track", None             # 追上跟踪值：之后逐位直通
            g.q = nq
            out[sl] = nq
        return out

    def gesture_state(self) -> list:
        """[(mode, active)] × 2，只给检查工具和日志看。"""
        return [] if self._g is None else [(g.mode, g.active) for g in self._g]

    def timing_us(self) -> dict:
        def stats(a, n):
            a = a[:min(n, _TIMING_RING)] / 1e3
            if not a.size:
                return dict(n=0)
            return dict(n=int(n), p50=float(np.percentile(a, 50)),
                        p99=float(np.percentile(a, 99)), max=float(a.max()))
        return dict(emit=stats(self.emit_ns, self.n_emit), curl=stats(self.push_ns, self.n_push))
