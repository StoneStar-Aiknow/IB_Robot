"""followme 的流式核心。**这个模块里没有一行 ROS**。

拆出来的理由和 viz_node 拆出推理是同一个：能不能在没有相机、没有 DDS、
没有机器人的条件下，把「流式管线和整段离线管线出一样的东西」这件事**证明**出来。
M1 的五条验收全部跑在这个文件上（见 tools/followme_replay.py），M2 的 ROS 节点
只是给它接上话题。

设计压在三条**实测**结论上，全部可复现（tools/followme_replay.py --probe）：

  1. **前缀等价是精确的。** readout B 下 `retarget(前 k 帧)` 的前 k 行与
     `retarget(全 N 帧)` 的前 k 行**逐位相同** —— 22 段真片子，最大偏差
     0.000e+00 rad。所以逐帧映射不是近似，是恒等。

  2. **唯一的时间耦合是 source.py:90 的 np.unwrap**，而它在 22 段真片子上
     **一次都没触发**（在 `body_pose` 局部矩阵的那一轴上，改动恒为 0）。
     但 `continuous_intrinsic_xyz` 的 docstring 写着「the reason a clip must be
     converted as a whole rather than frame by frame」——它是对的，所以这里
     不赌，见 BranchGuard。

  3. **readout A 有第二条整段耦合，readout B 没有。** `readouts.rest_direction`
     的 `.mean(0)` 是**整段统计**，而 `twist_indicators:180` 无条件调用它——
     只是结果只喂给 `"spin"`（A），默认的 `"plane"`（B）不读。
     **谁要把 readout A 窗口化，会静默拿到错的零点。** 这里只支持 B。

不移植的东西（PLAN_FOLLOWME.md §8）：`fit_velocity_budget` / `retime` /
`to_playback_rate` 的整段时间重标定。一个全局标量回头改写整条时钟，没有窗口
能近似它。流式侧由逐帧限速器取代——**这是替换，不是移植**。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_DOF = 14


def _import_vendor():
    """只读 import vendor 的算术。一个字节都不重写。

    `retarget` / `smooth` / `despike` / `quintic_blend` / `clip_to_limits`
    全部原样调用，followme 只负责**喂什么、什么时候喂**。

    IB-Robot patch: the vendored copy beside this file, not sys.path.
    """
    from .smplx2robot import postprocess, readouts  # noqa: F401
    from .smplx2robot.ledger import Ledger
    from .smplx2robot.retarget import READOUT_B, retarget
    from .smplx2robot.robot import ARM_JOINT_NAMES, X2Kinematics
    from .smplx2robot.source import SourceAnimation
    return dict(postprocess=postprocess, Ledger=Ledger, retarget=retarget,
                READOUT_B=READOUT_B, X2Kinematics=X2Kinematics,
                ARM_JOINT_NAMES=ARM_JOINT_NAMES, SourceAnimation=SourceAnimation)


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceFrame:
    """一帧 PEAR 拟合结果。字段名与 SourceAnimation 的同名字段逐字对应。

    `t_camera` 是**相机采集时刻**（秒）。绝不用本机接收时刻——node.py:166-184
    的单槽丢弃已经是按相机 stamp 判的，这里沿用同一个判据。
    """
    t_camera: float
    frame_id: int
    points: np.ndarray          # [145,3]
    body_pose: np.ndarray       # [21,3,3]
    global_orient: np.ndarray   # [3,3]


# ---------------------------------------------------------------------------
# 分支哨兵：唯一时间耦合的看守
# ---------------------------------------------------------------------------

class BranchGuard:
    """守住 np.unwrap 的分支，O(1)/帧。

    逐帧映射（窗口 = 1 帧）只在 unwrap **没有跨过 ±180°** 时与整段等价。
    22 段真片子上它一次都没触发，但这是数据的性质、不是算法的保证，
    而代价是腕部一次静默的 360° 反转——正是 `continuous_intrinsic_xyz`
    的 docstring 点名警告的后果（"a full-speed reversal on the robot"）。

    所以这里独立地、因果地把真分支算出来：
        unwrapped[k] = unwrapped[k-1] + wrap_to_pi(principal[k] - principal[k-1])
    这与 np.unwrap 的定义逐字相同（默认 discont=pi, period=2pi），因此不是近似。
    再与主值比：相等 → 单帧窗口精确；不等 → 报警并交出锚点。

    锚点取「该通道分支最后一次为 0 的帧」：从那一帧起的窗口，其内部 unwrap
    会重建出真分支，因为增量与整段的增量逐点相同。
    """

    # ledger 里 source_type == "euler" 的通道，(body_index, euler_axis) 去重后
    # 只有 6 个标量序列（左右腕 × pitch/roll/yaw），所以这整件事是 O(1)。
    def __init__(self, channels: list[tuple[int, int]]) -> None:
        self.channels = channels
        self._prev_principal: dict[tuple[int, int], float] = {}
        self._branch: dict[tuple[int, int], int] = {k: 0 for k in channels}
        self._last_zero_index: dict[tuple[int, int], int] = {k: 0 for k in channels}
        self.trips = 0
        self.worst_branch = 0

    @staticmethod
    def _principal(body_pose: np.ndarray, body_index: int, axis: int) -> float:
        from scipy.spatial.transform import Rotation
        ang = Rotation.from_matrix(body_pose[body_index]).as_euler("XYZ", degrees=False)
        return float(ang[axis])

    def observe(self, frame_index: int, body_pose: np.ndarray) -> int:
        """吃一帧，返回窗口锚点的帧号。等于 frame_index 就是「单帧窗口够用」。"""
        anchor = frame_index
        for key in self.channels:
            body_index, axis = key
            principal = self._principal(body_pose, body_index, axis)
            prev = self._prev_principal.get(key)
            if prev is not None:
                # wrap 到 (-pi, pi]，这就是 np.unwrap 每一步做的事
                step = principal - prev
                turns = int(np.round((step - np.mod(step + np.pi, 2 * np.pi) + np.pi)
                                     / (2 * np.pi)))
                self._branch[key] += turns
            self._prev_principal[key] = principal
            if self._branch[key] == 0:
                self._last_zero_index[key] = frame_index
            else:
                self.trips += 1
                self.worst_branch = max(self.worst_branch, abs(self._branch[key]))
                anchor = min(anchor, self._last_zero_index[key])
        return anchor


# ---------------------------------------------------------------------------
# 映射
# ---------------------------------------------------------------------------

class StreamRetargeter:
    """逐帧把 PEAR 拟合结果映射成 14 路关节角。

    **算术一行没重写**：调的就是 `smplx2robot.retarget.retarget`，只是喂给它
    一个长度为 1 的 SourceAnimation。前缀等价保证这与整段调用逐位相同。

    实测 1 帧窗口 1.34 ms（固定开销占大头，边际 0.155 ms/帧），
    20 Hz 输入下 **2.7% 单核**。
    """

    def __init__(self, vendor: dict, kinematics=None, ledger=None,
                 max_anchor_frames: int = 1200, mirror: bool = False) -> None:
        self._v = vendor
        # 照镜子：她的右臂驱动机器人左臂。**不是这里新写的算术**，调的就是
        # `Trajectory.mirrored()`（retarget.py:158），离线 `--mirror` 走的同一个。
        #
        # 位置也和离线一致：`clip_pipeline._convert` 把 --mirror 挂在 convert
        # 这一步，即 retarget 之后、postprocess（去离群/平滑/限位）之前。
        # 顺序不能换：镜像是"翻转+交换+常偏置"的仿射映射，跟线性的平滑可交换，
        # 但跟**逐通道不对称的限位夹紧**不可交换——先夹后镜，夹的是另一条臂的限位。
        # `mirrored()` 内部就是先对 raw 做、再夹（retarget.py:189-192），所以把它
        # 套在 trajectory 上、而不是套在已经夹过的 arm_pos 上，才等于离线。
        #
        # 逐行独立，实测：单帧窗口 mirror 与整段 mirror 的同一行差 **0.0**。
        # 所以流式逐帧做 = 离线整段做，跟前缀等价那条性质是一回事。
        self.mirror = mirror
        self.kinematics = kinematics if kinematics is not None else vendor["X2Kinematics"]()
        self.ledger = ledger if ledger is not None else vendor["Ledger"].load()
        self.max_anchor_frames = max_anchor_frames

        euler: set[tuple[int, int]] = set()
        for name, channel in self.ledger.channels.items():
            if channel.source_type == "euler" and channel.body_index is not None:
                for side in ("left", "right"):
                    euler.add((int(channel.body_index[side]), int(channel.euler_axis)))
        self.guard = BranchGuard(sorted(euler))

        self._frames: list[SourceFrame] = []      # 只在哨兵触发时才会用到多于 1 帧
        # IB-Robot patch: the guard keeps absolute frame numbers, so frames trimmed
        # off the front must be counted; upstream used list positions, and after the
        # first trim (max_anchor_frames pushes) every anchor equalled the current
        # frame and the guard never widened the window again.
        self._trimmed = 0
        self.limits_rad = np.array(self.kinematics.limits_rad, dtype=np.float64)
        self.notes: list[str] = []

    def _animation(self, frames: list[SourceFrame]):
        SourceAnimation = self._v["SourceAnimation"]
        return SourceAnimation(
            path=Path("<stream>"),
            frame_id=np.array([f.frame_id for f in frames], dtype=np.int64),
            timestamp_ns=np.array([int(f.t_camera * 1e9) for f in frames], dtype=np.int64),
            points=np.stack([f.points for f in frames]),
            body_pose=np.stack([f.body_pose for f in frames]),
            global_orient=np.stack([f.global_orient for f in frames]),
        )

    def push(self, frame: SourceFrame) -> np.ndarray:
        """吃一帧，出 [14] 弧度（已按关节限位夹紧，与 retarget 的 arm_pos 同义）。"""
        self._frames.append(frame)
        index = self._trimmed + len(self._frames) - 1
        anchor = self.guard.observe(index, frame.body_pose)

        if anchor < index:
            span = index - anchor + 1
            if span > self.max_anchor_frames:
                # 说到做到：不静默降级。窗口截断会改分支，改分支就是反转。
                anchor = index - self.max_anchor_frames + 1
                self.notes.append(
                    f"帧 {index}: unwrap 分支跨度 {span} 帧超过上限 "
                    f"{self.max_anchor_frames}，窗口被截断，腕部可能出现 360° 台阶")
                # IB-Robot patch: no later window starts before this anchor (the
                # clamp only moves forward and a newly tripped channel anchors
                # later), so drop those frames; upstream kept every frame for as
                # long as the guard stayed tripped.
                del self._frames[:anchor - self._trimmed]
                self._trimmed = anchor
            window = self._frames[anchor - self._trimmed:]
        else:
            window = self._frames[-1:]
            # 单帧窗口够用时不留历史，内存恒定
            if len(self._frames) > self.max_anchor_frames:
                self._trimmed += len(self._frames) - self.max_anchor_frames
                del self._frames[:-self.max_anchor_frames]

        trajectory = self._v["retarget"](
            self._animation(window), kinematics=self.kinematics,
            ledger=self.ledger, readout=self._v["READOUT_B"], verify=False)
        if self.mirror:
            # **FK 自检开着**（传 kinematics）。实测 0.20 ms/帧，而 retarget 本身
            # 2.3 ms/帧 —— 9% 的开销，买的是"符号错了会抛异常"。符号错一个，
            # 机器人那条关节就是满速反向，这不是能靠事后看日志兜住的事。
            # 超过 MIRROR_CHECK_LIMIT_MM 它自己 raise，我们不接。
            # IB-Robot patch: the per-frame FK self-check is off (no `kinematics=`);
            # x2.followme.create runs it once at startup instead. The signs are
            # constants, so a frame can only fail it if the table is wrong, and a
            # 20 mm trip on one live frame would drop that frame for nothing.
            trajectory = trajectory.mirrored(self.ledger)
        return trajectory.arm_pos[-1].copy()


# ---------------------------------------------------------------------------
# 去离群（因果）
# ---------------------------------------------------------------------------

class CausalDespiker:
    """`despike`（postprocess.py:312）的因果版。**判据的结构原样保留。**

    离线版三个统计里有一个是整段的：步长检验用**全片**一阶差分的 median/MAD
    （:331-334）。流式拿不到全片，换成 EWMA 估计；另外两个——Hampel 的滚动
    中位数（局部，半窗 5 = ±167 ms）和「物理不可能」的额定速度检验（逐帧）——
    **本来就是局部的，一字不改**。

    交集规则同样保留，而且保留的理由比在离线时更强：
    `despike` 的 docstring 说「false deletion is risk #1，因为被错删的动作
    留下的结果平滑、可信、而且没人看得出少了东西」。流式看不到整段，更没人能
    发现少了什么。

    预热期（默认 1.5 s）内 EWMA 还没收敛，**只做「物理不可能」那一条**——
    它不是统计判据，不需要第二意见（docstring 原话）。
    """

    def __init__(self, rated_rad_s: np.ndarray, dt: float,
                 warmup_s: float = 1.5, threshold: float = 3.0,
                 mad_floor_rad: float = np.radians(0.05),
                 impossible_factor: float = 3.0, half_life_s: float = 2.0) -> None:
        self.rated = np.asarray(rated_rad_s, dtype=np.float64)
        self.threshold = threshold
        self.mad_floor = mad_floor_rad
        self.impossible_factor = impossible_factor
        self.warmup_frames = max(1, int(round(warmup_s / max(dt, 1e-6))))
        self.alpha = 1.0 - 0.5 ** (max(dt, 1e-6) / half_life_s)
        self._center = np.zeros(ARM_DOF)                      # rad/s
        self._spread = np.full(ARM_DOF, mad_floor_rad / max(dt, 1e-6))
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self._seen = 0
        self.flagged = np.zeros(ARM_DOF, dtype=np.int64)

    def update(self, pose: np.ndarray, t: float) -> None:
        """喂一帧**原始**映射结果，推进 EWMA。不改数据。

        统计量是**速度**（rad/s）不是步长。离线 `despike` 跑在已经重采样过的
        均匀栅格上，步长和速度只差一个常数；流式这边输入间隔是 33–233 ms 的
        实测分布（§1.5），而窗口重采样到 33.3 ms，两者不是同一个尺度。
        存速度，比的时候各自乘各自的 dt——否则丢一帧就会被自己的统计判成离群。
        """
        if self._prev is not None and t > self._prev_t:
            step_dt = t - self._prev_t
            v = (pose - self._prev) / step_dt
            a = self.alpha
            if self.warm:
                # **温莎化**：离群点不许把「多大算离群」这把尺子自己顶上去。
                # 离线 `despike` 用的是全片的 median/MAD，天然抗污染（:331-334）；
                # EWMA 不是——一个 ±30° 单帧点就是 ~16 rad/s，直接进 `_spread`
                # 的话，阈值被它自己抬高，**后面的离群点就判不出来了**。
                # 实测：不温莎化时，单个注入点残留 0.59°，同一段注入 6 个后
                # 最差残留 9.38°；差别全在这里，不在判据结构上。
                # 一帧同时穿过两条路：进统计的那份被夹住，进 `_prev` 的那份不夹
                # —— 夹了就等于悄悄改数据，而定罪与重建只能在 `scrub` 里发生。
                scale = np.maximum(1.4826 * self._spread, self.mad_floor / step_dt)
                lo = self._center - self.threshold * scale
                hi = self._center + self.threshold * scale
                v = np.clip(v, lo, hi)
            self._center = (1 - a) * self._center + a * v
            self._spread = (1 - a) * self._spread + a * np.abs(v - self._center)
        self._prev = pose.copy()
        self._prev_t = t
        self._seen += 1

    @property
    def warm(self) -> bool:
        return self._seen >= self.warmup_frames

    def scrub(self, window: np.ndarray, dt: float, vendor: dict) -> tuple[np.ndarray, np.ndarray]:
        """在一个窗口上定罪并重建。返回 (修好的窗口, 标记)。"""
        postprocess = vendor["postprocess"]
        level = postprocess.hampel_flags(window, 5, self.threshold, self.mad_floor)

        v = np.diff(window, axis=0, prepend=window[:1]) / dt   # rad/s，与 EWMA 同域
        scale = np.maximum(1.4826 * self._spread, self.mad_floor / dt)[None, :]
        step = np.abs(v - self._center[None, :]) > self.threshold * scale
        impossible = np.abs(v) > self.impossible_factor * self.rated[None, :]

        seeds = ((level & step) | impossible) if self.warm else impossible
        flags = seeds.copy()
        # 判决沿 level 掩码生长：交集能判"这里有事"，判不了"多宽"。
        # 步长检验在进出一段异常时各响一次、异常内部静默，只删首帧会把峰留下，
        # 然后拿被删的帧去对着那个幸存的峰插值——比什么都不做更糟（:352-360）。
        for j in range(window.shape[1]):
            column = level[:, j]
            for i in np.where(seeds[:, j])[0]:
                k = i
                while k + 1 < len(column) and column[k + 1]:
                    k += 1
                    flags[k, j] = True
                k = i
                while k - 1 >= 0 and column[k - 1]:
                    k -= 1
                    flags[k, j] = True

        out = window.copy()
        index = np.arange(window.shape[0])
        for j in range(window.shape[1]):
            bad = flags[:, j]
            if not bad.any() or bad.all():
                continue
            # 不外插、不丢弃：用前后有效帧线性插值。缓冲里有 850 ms，
            # 后向邻居必定已到——这是 850 ms 买到的东西之一。
            out[bad, j] = np.interp(index[bad], index[~bad], window[~bad, j])
        return out, flags


# ---------------------------------------------------------------------------
# 限速
# ---------------------------------------------------------------------------

class RateLimiter:
    """逐通道限制 |Δθ| ≤ v_max·dt。**每次触发都要能报出来。**

    `animation_player` 没有这个东西，而它必须有：22 段真片子的裸映射输出峰值
    5647 °/s，平滑后典型 200–350 °/s，**其中一段平滑后仍有单通道超额定**。
    所以不能指望平滑兜住。

    但限速器**不是解药**：持续触发意味着上游平滑没做够。它是信号。
    M1 的验收 2 要求**零触发**——限速器在那一步的作用是当探针，不是当补丁。
    """

    def __init__(self, rated_rad_s: np.ndarray, dt: float, scale: float = 1.0) -> None:
        self.step = np.asarray(rated_rad_s, dtype=np.float64) * scale * dt
        self.trips = np.zeros(ARM_DOF, dtype=np.int64)
        self.worst_excess_rad = np.zeros(ARM_DOF)
        self.dt = dt
        self._peak = np.zeros(ARM_DOF)
        self._prev: np.ndarray | None = None

    def apply(self, pose: np.ndarray, clamp: bool = True) -> np.ndarray:
        """`clamp=False` 只统计不改数据——M1 验收 2 要的是「本来就没超」，
        所以那一步必须让限速器当**探针**，不能让它一边补救一边报零触发。"""
        if self._prev is None:
            self._prev = pose.copy()
            return pose
        delta = pose - self._prev
        self._peak = np.maximum(self._peak, np.abs(delta) / self.dt)
        excess = np.abs(delta) - self.step
        hit = excess > 0
        if hit.any():
            self.trips += hit
            self.worst_excess_rad = np.maximum(self.worst_excess_rad,
                                               np.where(hit, excess, 0.0))
            if clamp:
                delta = np.clip(delta, -self.step, self.step)
        out = self._prev + delta
        self._prev = out
        return out

    @property
    def peak_deg_s(self) -> np.ndarray:
        return np.degrees(self._peak) if self._peak is not None else np.zeros(ARM_DOF)


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------

# D / L 的**唯一权威值**。followme_node 的 declare_parameter、
# tools/followme_replay.py 的 CLI 默认值都引用这两个名字，不许再抄字面量 ——
# 2026-09-29 之前它们散在三处（core 签名 850/600、node 850/600、
# run_followme.sh 覆盖成 400/250），谁在读哪个全靠猜。
# 定档理由、四档实测、以及"L 不是免费的"那段账，见 FollowMeStream.__init__。
DEFAULT_LATENCY_S = 0.400
DEFAULT_LOOKAHEAD_S = 0.250

PARK, BLEND_IN, LIVE, BLEND_OUT, FINAL, DONE = (
    "park", "blend_in", "live", "blend_out", "final", "done")


@dataclass
class Session:
    """会话时序。**起点和终点是同一个 base 数组，逐位相同。**

    用户的要求是：从中性位开始 → 可以从任意 pose 起录 → 约定时长 →
    2 s 混回初始位 → 起点终点相同。

    `blend_in` 的**起点永远是 base**，不是人的 pose：人站成什么样不影响
    机器人从哪出发，人的姿态只决定 blend_in 的终点。
    """
    live_s: float = 10.0
    park_s: float = 0.5
    blend_s: float = 2.0
    out_hz: float = 50.0

    # `live_s <= 0` = **不限时长**（followme 的 forever 模式，2026-09-29 granger
    # 提的新要求）：一直跟，直到 Ctrl-C 才反应式收尾。非 --followme 的录制链
    # 不受影响，它那条仍然是约定时长。
    #
    # 为什么 forever 只需要把 n_live 设大、不需要改 `phase()`：
    # 收尾用的是**交叉淡入**，`out = base + w(k)·(live(t) − base)`，而
    # w'(1) = 0（quintic），所以在 live → blend_out 的接缝上
    # d(out)/dt = w·live' = live' —— **速度本来就是连续的**。
    # 排期的 blend_out（提前 2 s 就知道要收）和反应式的 blend_out
    # （Ctrl-C 那一刻才知道）因此**一样干净**，不产生台阶。
    # 这就是 `request_stop()` 可以在任意 k 上截断 n_live 的全部依据。
    FOREVER_N = 1 << 62   # 不用 math.inf：int(round(inf * 50)) 会 OverflowError

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.out_hz
        self.n_park = max(1, int(round(self.park_s * self.out_hz)))
        self.n_blend = max(1, int(round(self.blend_s * self.out_hz)))
        self.forever = self.live_s <= 0.0
        self.n_live = (self.FOREVER_N if self.forever
                       else max(1, int(round(self.live_s * self.out_hz))))
        self.n_total = 2 * self.n_park + 2 * self.n_blend + self.n_live

    def request_stop(self, k: int) -> bool:
        """反应式收尾：让第 k 个输出帧成为 blend_out 的第一帧。

        只在 forever 模式下有意义（定长会话的收尾时刻是排期好的，截断它等于
        偷偷改用户约定的时长）。返回是否真的截断了。

        截断只动 `n_live`/`n_total` 两个数，`phase()` 一个字不改 —— 依据见
        上面 FOREVER_N 处的交叉淡入论证。若 Ctrl-C 来得比 blend_in 结束还早，
        `max(1, ...)` 保证 live 至少走一帧，会话仍然走完
        park → blend_in → live → blend_out → final，起点终点仍逐位相同。
        """
        if not self.forever:
            return False
        self.n_live = max(1, k - self.n_park - self.n_blend)
        self.n_total = 2 * self.n_park + 2 * self.n_blend + self.n_live
        self.forever = False
        return True

    def phase(self, k: int) -> tuple[str, float]:
        """第 k 个输出帧的 (相位, 活动目标的权重 0..1)。

        权重取自 vendor 的 `quintic_blend`（postprocess.py:451），零速零加速度
        进出。它 "excludes both endpoints"，正好用来接缝而不重复帧——所以
        park 的最后一帧和 live 的第一帧都不会被 blend 复制一遍。

        这里用的是**交叉淡入**而不是点对点混合：`quintic_blend(base, 第一帧
        活动目标)` 在因果上做不到——发 blend_in 的时候，live 的第一帧还没到。
        交叉淡入 out = base + w(k)·(live(t) − base) 是良定义的，而且端点性质
        一样好：w'(0)=0 保证零速起步，w(1)=1、w'(1)=0 保证接进 live 时
        速度就是 live 自己的速度，不产生台阶。
        """
        n_park, n_blend, n_live = self.n_park, self.n_blend, self.n_live
        if k < n_park:
            return PARK, 0.0
        k -= n_park
        if k < n_blend:
            return BLEND_IN, self._weights[k]
        k -= n_blend
        if k < n_live:
            return LIVE, 1.0
        k -= n_live
        if k < n_blend:
            return BLEND_OUT, self._weights[n_blend - 1 - k]
        k -= n_blend
        if k < n_park:
            return FINAL, 0.0
        return DONE, 0.0

    def bind(self, vendor: dict) -> "Session":
        quintic_blend = vendor["postprocess"].quintic_blend
        self._weights = quintic_blend(np.zeros(1), np.ones(1), self.n_blend)[:, 0]
        return self


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """一个 50 Hz 输出帧。"""
    t_out: float          # 输出时钟（= 相机时刻 − latency_s）
    t_wall: float         # 发出这一帧的挂钟时刻
    t_camera_latest: float
    phase: str
    pose: np.ndarray      # [14] rad，已限位（+限速）
    pose_raw: np.ndarray  # [14] rad，限位之前
    buffered: int
    starved: bool         # 窗口里没有足够的料，靠保持上一个活动目标顶过去


class FollowMeStream:
    """把一串 SourceFrame 变成一串 50 Hz 的 Sample。

    输出时钟：`t_out = t_camera_latest − latency_s`。所有输出都在这条时钟上
    按 out_hz 产生。**只要输入的断流不超过 latency_s，输出永远有料可发**——
    「保证一直出流」是这条时钟的结构性质，不是靠重试或外插换来的。

    每个输出时刻的处理顺序与离线 `process()`（postprocess.py:602）逐段对应：

        离线  resample_uniform → despike → smooth → repair_limits → retime → add_ramps
        流式  窗口 resample    → 因果 despike → 窗口 smooth → 逐帧限位 → (不做) → 会话相位

    `retime` 那一格是空的，而且必须是空的：它是一个全局标量回头改写整条时钟，
    没有任何窗口能近似它（PLAN_FOLLOWME.md §8）。流式侧由 RateLimiter 取代。
    """

    def __init__(self, *, vendor: dict | None = None, kinematics=None, ledger=None,
                 session: Session | None = None,
                 # D = 400 / L = 250 ms（2026-09-29 定档，granger 看过实机效果
                 # 后主观验收通过）。四档实测手感延迟：850/600 → 1002 ms、
                 # 700/600 → 847、550/400 → 702、**400/250 → 628**；
                 # 四档都是饿死 0 / 乱序 0 / 输出 3250 帧不缺。
                 #
                 # ⚠️ 撤回先前写在这里的「L 不花延迟」。那句话把**两个不同的量**
                 # 都叫了 p99，必须分开：
                 #   ① 「相对离线整段 filtfilt 的偏差」—— L 越大越小。
                 #      全 22 段（PLAN §7 验收 6 的对照表）L 400→600：
                 #      中位 0.060°→0.027°、p99 0.865°→0.470°。
                 #      **这是保真度**，也正是 M1 验收 1 那条门（p99<1.0°/max<2.0°）。
                 #   ② 「逐 tick 单步跨度 p99」—— L 越**小**它也变小
                 #      （6.390°→5.106°）。那不是变好：L 小 → filtfilt 的边缘
                 #      填充逼近 t_out → 响应被压平 → 幅度衰减 → 跨度自然小。
                 #      **这个量结构上看不见 L 的代价**，不能拿它给 L 定档。
                 # 按 ① 量，L=233 是 p99 1.60° / max 3.70°，**在 M1 的门外**；
                 # L=250 就落在那条线附近。所以 400/250 是拿 1~3° 的幅度衰减
                 # 换 374 ms 手感延迟 —— 这笔交易 granger 接受了，但它**不是免费的**。
                 #
                 # 还没被那次四档扫描覆盖、**别删**的一条：输入出现 800 ms 空洞时，
                 # L 400→600 让最大单步 68.85°→18.71°、饿死帧 215→108。
                 # 四档实测全是「饿死 0」= **从没进入那个工况**，所以这条没被证伪，
                 # 只是没被触发。L=250 在真有空洞时的最大单步是**未知量**。
                 #
                 # 越界预算 = D − L = 150 ms：窗口要 t_out+0.25 的料，所以约束的是
                 # PEAR 端到端 age 相对自身中位的**上偏** ≤ 150 ms（不是 age 本身，
                 # age 的常量部分被 `_offset` 吸收掉了）。实测 pear 帧净年龄
                 # 中位 ≈150、干净环境 p99 223（上偏 73，放得下），
                 # **脏环境 p99 329（上偏 179 > 150，放不下）**。
                 # 超了不出错，会**优雅退化**成"有多少右臂用多少"，但那一下就是「卡一下」。
                 latency_s: float = DEFAULT_LATENCY_S,
                 lookahead_s: float = DEFAULT_LOOKAHEAD_S,
                 history_s: float = 1.0, grid_dt: float = 1.0 / 30.0,
                 base_pose: np.ndarray | None = None,
                 cutoff_hz: float = 2.0, vel_scale: float = 1.0,
                 limit_rate: bool = False, despike: bool = True,
                 mirror: bool = False) -> None:
        self._v = vendor if vendor is not None else _import_vendor()
        postprocess = self._v["postprocess"]

        # base **不镜像**：中性位是机器人自己的姿态，跟人站成什么样无关，
        # 会话的起点终点仍然是同一个 BASE_POSE_RAD 逐位相同。离线也是这样——
        # mirror 在 convert 做完，postprocess 的混合照样收敛到未镜像的 BASE_POSE_RAD。
        # （顺带：BASE_POSE 自己镜像一下会差 0.35°，即 mirrored() docstring 里
        #  说的"三分之一度的常数"，所以这个选择是看得见的，不是无所谓。）
        self.mirror = mirror
        self.retargeter = StreamRetargeter(self._v, kinematics, ledger,
                                           mirror=mirror)
        self.kinematics = self.retargeter.kinematics
        self.session = (session or Session()).bind(self._v)
        self.latency_s = latency_s
        self.lookahead_s = lookahead_s
        self.history_s = history_s
        self.grid_dt = grid_dt
        self.cutoff_hz = cutoff_hz
        self.base = (np.asarray(base_pose, dtype=np.float64).copy()
                     if base_pose is not None else postprocess.BASE_POSE_RAD.copy())
        self.rated = np.radians(postprocess.RATED_SPEED_DEG_S)

        self.despiker = CausalDespiker(self.rated, grid_dt)
        # 限速器每场会话重建（见 reset），所以 scale 要留一份。
        self.vel_scale = vel_scale
        self.limiter = RateLimiter(self.rated, self.session.dt, vel_scale)
        self.limit_rate = limit_rate
        self.despike = despike

        # 环形缓冲：(t_camera, 映射后的 [14])。容量按 latency + 历史 + 余量。
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._buf_span = latency_s + history_s + 1.5
        self._k = 0                       # 已发出的输出帧计数
        self._t0: float | None = None     # 会话第 0 帧的输出时刻
        self._offset = 0.0                # 挂钟 − 输出时钟，标定一次
        # 标定用的 age 样本：闸没开的那段时间里每次 poll 采一个（见 poll）。
        self._warm_ages: list[float] = []
        self.offset_samples = 0           # 最终用了几个样本取中位数
        self._last_live: np.ndarray | None = None
        self.starved_frames = 0
        self.stretched_frames = 0   # 右窗撑开过几次，见 _live_target
        self.notes: list[str] = []

    def reset(self, session: "Session") -> None:
        """清掉**会话级**状态，准备再起一场。输入侧的状态一律不动。

        这是 followme_node 的 start/stop 轮转（每 2 分钟收尾、10 s 后重起，
        为 6 小时连续展示而设）唯一需要 core 配合的地方。它**不重建
        FollowMeStream 本身** —— 那要重新建运动学/ledger（日志里那个「建图 N ms」），
        一天几百轮全是白花的。

        分界线是「这个状态属于会话，还是属于输入流」：

        **清**（会话级，下一场必须从头来）
          `session` `_k` `_t0` `_offset` `_warm_ages` `offset_samples`
          `_last_live` `starved_frames` `stretched_frames` `notes`，外加
          重建 `limiter`（`_prev`/`trips`/`_peak` 都是逐场的账）。
          `_offset` 是重点：它是**热身期 age 中位数**标定出来的挂钟偏置，
          沿用上一场就等于假设人和 PEAR 的延迟这十几分钟没变过。每场重标
          反而是轮转的真实收益 —— 一次标定撞上尖峰的损失从「整场」压到「一轮」。

        **不清**（输入级，跟着人走，断了反而更差）
          `despiker` —— EWMA 预热 1.5 s，两场之间输入根本没停，清了等于让
          新一场开头 1.5 s 退回「只查物理不可能」那一档，白丢保护。
          `retargeter` —— `BranchGuard` 守的是 unwrap 的分支连续性，那是**人**
          的连续性，不是会话的。在轮次边界上重置它，正好是它存在要防的那个
          360° 腕部反转。

        `_buf` 归第一类：清掉之后重启走的是和冷启动**逐字相同**的路径，
        没有第二条少测过的分支。代价是新一场要重新灌满缓冲，而热身 3 s +
        park 0.5 s + blend_in 2 s = 5.5 s 远够（10 Hz 输入灌满 latency_s
        只要 0.4 s）。
        """
        self.session = session.bind(self._v)
        self._buf.clear()
        self._k = 0
        self._t0 = None
        self._offset = 0.0
        self._warm_ages.clear()
        self.offset_samples = 0
        self._last_live = None
        self.starved_frames = 0
        self.stretched_frames = 0
        self.notes.clear()
        self.limiter = RateLimiter(self.rated, self.session.dt, self.vel_scale)

    # -- 输入 ------------------------------------------------------------
    def push(self, frame: SourceFrame) -> None:
        self.push_pose(frame.t_camera, self.retargeter.push(frame))

    def push_pose(self, t: float, pose: np.ndarray) -> None:
        """绕开映射直接入队。

        存在的理由只有一个：M1 的验收 4（注入离群点）和 5（人为空洞）要能在
        **不伪造 body_pose 矩阵**的前提下做。前缀等价（模块 docstring 第 1 条）
        保证 `retargeter.push` 与整段 `retarget` 逐位相同，所以从姿态灌进来
        和从拟合结果灌进来，经过的是同一条后半程。
        """
        pose = np.asarray(pose, dtype=np.float64)
        self.despiker.update(pose, t)
        # 按相机 stamp 插入排序：BEST_EFFORT 下帧会乱序。**绝不用本机时钟。**
        if self._buf and t < self._buf[-1][0]:
            items = list(self._buf)
            items.append((t, pose))
            items.sort(key=lambda it: it[0])
            self._buf = deque(items)
        else:
            self._buf.append((t, pose))
        cutoff = t - self._buf_span
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    # -- 输出 ------------------------------------------------------------
    def observe_age(self, now: float) -> None:
        """会话时钟起步**之前**，按输出 tick 的节奏采一个 age 样本。

        `poll` 自己也采（闸关着的那几次），但节点在热身期**根本不调 poll**
        （followme_node._tick 在 warmup 分支里就 return 了），所以热身结束时
        缓冲早就灌满、闸一次就开 —— 只剩一个样本，中位数退化回原来的单次读数。
        这个方法让热身期的每个 tick 都贡献一个样本：50 Hz × 3 s = 150 个。

        它估的必须和 poll 里那一行是**同一个量** —— tick 时刻的
        `now − t_latest`，含 10 Hz 出帧带来的锯齿 —— 所以两处逐字相同。
        锯齿本身也该被中位数吃掉：刚收到一帧时 reach 大、下一帧前 reach 小，
        居中的那个才是输出时钟该挂的位置。
        """
        if self._t0 is None and self._buf:
            self._warm_ages.append(now - self._buf[-1][0])

    def poll(self, now: float) -> list[Sample]:
        """挂钟走到 `now`，把到期的 50 Hz 帧全部吐出来。

        **输出时钟挂在挂钟上，不挂在最后一帧的相机时刻上。** 这个区别只在
        输入断流时才显出来，而那正是它存在的理由：若 `t_out` 跟着
        `t_camera_latest` 走，输入一停输出就跟着停——断流会**如实地变成输出
        断流**，850 ms 缓冲一点忙都帮不上。挂在挂钟上，缓冲才真的是缓冲。

        `_offset` 全程只标定一次（会话时钟不许中途跳），但**用的是 age 的
        中位数，不是某一次 poll 的瞬时值**：

            offset = median(age 样本) + latency_s ,  age = now − t_camera_latest

        这里换掉了原来的单次读数 `offset = now − (t_latest − latency_s)`。
        那个写法在代数上就是 `age₀ + D`，于是**第一帧恰好赶上一次 age 尖峰，
        整场会话的手感延迟就被锁在高位**：实测 400/250 档下多出 +78 ms
        （`手感延迟 − D` 从 150 变成 228）。这正是 granger 说的"有时候整场
        都慢，下一场又正常"。而那 78 ms 是从 `D − L = 150` 的越界预算里出的。
        中位数不花任何启动时间：样本取自**闸本来就关着**的那段
        （缓冲要灌到 latency_s 才开闸，50 Hz 下 ≥ 20 次 poll，
        带 `--warmup` 则是几百次），闸一开就有现成的分布。

        offset 同时吸收掉 PEAR 推理 + 传输的**常量部分**，所以预算约束的是
        age 相对自身中位的上偏，不是 age（见模块 docstring）。
        """
        if self._t0 is None:
            # **会话时钟等缓冲灌满才起步。** 少了这道闸，第 0 帧的 t_out 就是
            # `t_first − D`，而那个时刻**没有任何数据**——整个 park 和
            # blend_in 的前半段都在窗口外，逐帧饿死、靠 base 顶着。
            # 行为上不出错（那一段权重本来就近 0），但它把「饿死」这个信号
            # 磨钝了：真正断流时的饿死帧数会淹在启动噪声里。
            if self._buf:
                self._warm_ages.append(now - self._buf[-1][0])
            if not self._buf or self._buf[-1][0] - self._buf[0][0] < self.latency_s:
                return []
            self.offset_samples = len(self._warm_ages)
            self._offset = float(np.median(self._warm_ages)) + self.latency_s
            self._warm_ages.clear()
            self._t0 = now - self._offset
        t_now = now - self._offset
        t_latest = self._buf[-1][0] if self._buf else float("nan")
        out: list[Sample] = []
        while True:
            phase, weight = self.session.phase(self._k)
            if phase == DONE:
                break
            t_out = self._t0 + self._k * self.session.dt
            if t_out > t_now + 1e-9:
                break
            out.append(self._emit(t_out, t_out + self._offset, t_latest, phase, weight))
            self._k += 1
        return out

    @property
    def finished(self) -> bool:
        return self.session.phase(self._k)[0] == DONE

    def _emit(self, t_out: float, t_wall: float, t_latest: float,
              phase: str, weight: float) -> Sample:
        starved = False
        if weight <= 0.0:
            pose = self.base.copy()
        else:
            live = self._live_target(t_out)
            if live is None:
                starved = True
                self.starved_frames += 1
                # 断流：保持上一个活动目标。绝不外插——外插的是我们自己编的
                # 速度，而这正是 smooth 的 docstring 拒绝 filtfilt 反射填充的
                # 同一个理由（"we would then be smoothing our own invention"）。
                live = self._last_live if self._last_live is not None else self.base
            else:
                self._last_live = live
            pose = self.base + weight * (live - self.base)

        raw = pose.copy()
        pose = self.kinematics.clip_to_limits(pose)
        pose = self.limiter.apply(pose, clamp=self.limit_rate)
        # park / final 必须**逐位**等于 base：验收 3 靠的是这一行，
        # 不是靠 clip_to_limits 恰好没动它。
        if phase in (PARK, FINAL):
            pose = raw = self.base.copy()
            self.limiter._prev = pose.copy()
        return Sample(t_out=t_out, t_wall=t_wall, t_camera_latest=t_latest,
                      phase=phase, pose=pose, pose_raw=raw,
                      buffered=len(self._buf), starved=starved)

    def _live_target(self, t_out: float) -> np.ndarray | None:
        """窗口 resample → 去离群 → 平滑 → 在 t_out 处取值。

        窗口左 1.0 s 是刻意的：正好等于 `smooth()` 自己的
        `pad = max(12, round(1/dt))`（postprocess.py:388），所以左边界
        **不引入额外误差**。右 0.4 s 是实测选出来的前瞻（p99 0.62°/max 1.16°）。
        """
        lo, hi = t_out - self.history_s, t_out + self.lookahead_s
        window = [(t, p) for t, p in self._buf if lo - 1e-9 <= t <= hi + 1e-9]

        # 右窗空了 → 把右边界**临时**撑到缓冲里最新的那一帧。
        #
        # 这一条不是优化，是断流那一条验收的要害。名义前瞻 0.4 s 比延迟档
        # 0.85 s 短，中间那 0.45 s 的料**本来就已经躺在缓冲里**（PLAN §4 里
        # 「乱序/丢帧补偿 200 ms」+「50 Hz 重采样前瞻 150 ms」两格买的就是它）。
        # 输入出现 500 ms 空洞时，t_out 落进空洞的头 0.1 s 里，[t_out, t_out+0.4]
        # 一帧都没有——可缓冲里明明有空洞之后的帧。不撑开就得走「保持上一个
        # 目标」，等空洞过去再一步跳回来：实测单步 37.5°，正是验收 5 的
        # 「不跳变」要禁的东西。撑开之后前瞻只是**变长**，而更长的前瞻在
        # §1.4 的表里只会让误差更小，不会更大。
        # 左边界不动：往回多看没有代价，但改左边界会动到 smooth 的 pad 语义。
        if window and window[-1][0] < t_out and self._buf:
            newest = self._buf[-1][0]
            if newest > t_out:
                window = [(t, p) for t, p in self._buf
                          if lo - 1e-9 <= t <= newest + 1e-9]
                self.stretched_frames += 1

        if len(window) < 8:               # smooth 需要足够的料，和 process() 同一条线
            return None
        if window[0][0] > t_out or window[-1][0] < t_out:
            return None                   # t_out 不在窗口内部，只能靠保持

        t = np.array([w[0] for w in window])
        x = np.stack([w[1] for w in window])
        postprocess = self._v["postprocess"]
        grid, xr = postprocess.resample_uniform(t, x, self.grid_dt)
        if self.despike:
            xd, flags = self.despiker.scrub(xr, self.grid_dt, self._v)
            # 窗口是重叠的，每个输出 tick 都会重扫同一批帧。只在 t_out 落的那一格
            # 上记账，才数得出「这条流里有多少帧被判过离群」。
            self.despiker.flagged += flags[int(np.argmin(np.abs(grid - t_out)))]
        else:
            xd = xr                      # 只在验收台分解误差预算时用
        xs = postprocess.smooth(xd, self.grid_dt, self.cutoff_hz)

        # 三次样条，不是线性插值——因为**离线侧就是三次样条**：
        # `delivery.to_playback_rate:123-130` 用 `CubicSpline(..., "natural")`
        # 把后处理栅格铺到 50 Hz。线性插值在 30 Hz 栅格上对一条 2 Hz 的曲线
        # 会留下**弦误差**：(1/8)·a·h² 量级，实测能到 5°，比窗口化误差本身
        # 大一个数量级——那不是流式的误差，是我自己换了插值方法造的。
        # PLAN §4 预算表里「50 Hz 重采样前瞻 150 ms / 三次样条节点影响每节点
        # 衰减 0.268」这一行，说的就是这里用的东西。
        from scipy.interpolate import CubicSpline
        return CubicSpline(grid, xs, axis=0, bc_type="natural")(t_out)
