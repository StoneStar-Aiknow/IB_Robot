"""X2 肩部 pitch/roll/yaw 映射：把人的上臂朝向按机器人关节链解析求解（single_A）。

只依赖 numpy，按帧流式调用，用于实时链路。设计原理见同目录 HANDOFF_SHOULDER_IK.md。

一句话：上臂朝向 = 球面上一个点。X2 肩是 pitch -> roll -> yaw 三个串联转轴，每个朝向有两种关节写法
（A：|roll| <= 90；B：|roll| > 90，pitch 翻半圈），两者是同一个物理姿态，只在「上臂正侧平举」处连通。
同时用两种写法就一定会遇到「必须在远处换写法」= 手臂甩一圈。本模块只用 A：远处换写法按构造不可能；
A 唯一的奇异点是侧平举，在那里 pitch 怎么转手臂指向都几乎不变，经过时顺手转过去就是安全切换。

用法::

    geo = {side: ShoulderGeometry.from_dict(d) for side, d in json.load(open("geometry_x2.json"))["sides"].items()}
    solver = ShoulderSolver(geo["left"])          # 每只手臂一个，跨帧保留状态；换人/中断后 solver.reset()
    pitch, roll, yaw = solver.step(arm_dir, dt, yaw_need)   # 全是 rad；yaw_need 可为 None（此时 yaw 返回 None）

arm_dir：人上臂单位方向（肩 -> 肘），躯干系 (前, 左, 上)，和机器人躯干系同一约定。
yaw_need(pitch, roll) -> (yaw_cmd_rad, weight)：在这组 pitch/roll 下，要还原人的上臂扭转 yaw 需要转到多少，
以及扭转读数的可信度 0..1（肘伸直时扭转无定义 -> 0）。参考实现见 reference/yaw_need_reference.py。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

__all__ = ["ShoulderGeometry", "ShoulderParams", "ShoulderSolver"]


# ---------------------------------------------------------------- 小工具

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# IB-Robot patch: step() runs per frame in the live chain, where numpy's generic calls on 3-vectors
# and scalars cost more than the arithmetic. _cross and _clip are np.cross and np.clip with the same
# products and comparisons, so they return the same floats.
def _cross(u, v) -> np.ndarray:
    return np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """绕 axis 转 angle（rad）的旋转矩阵（Rodrigues）。"""
    a = _unit(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """u 绕 axis 转到 v 的角度（两者先投影到垂直于 axis 的平面）。"""
    u = u - (u @ axis) * axis
    v = v - (v @ axis) * axis
    return float(np.arctan2(axis @ _cross(u, v), u @ v))


# ---------------------------------------------------------------- 几何

@dataclass(frozen=True)
class ShoulderGeometry:
    """一侧肩的几何常数：躯干系 (前, 左, 上)，取自机器人零位（全部关节为 0）。

    rest_dir 必须是「roll 关节 -> 肘关节」的方向，不能从 pitch 关节量起：
    pitch 轴和 roll 轴不交于一点，从 pitch 关节量会差约 1.4 度。
    """
    pitch_axis: np.ndarray
    roll_axis: np.ndarray                 # 零位时的 roll 轴
    rest_dir: np.ndarray                  # 零位时上臂方向（roll 关节 -> 肘关节）
    pitch_limit: tuple[float, float]      # rad
    roll_limit: tuple[float, float]       # rad
    yaw_limit: tuple[float, float]        # rad

    @classmethod
    def from_dict(cls, d: dict) -> "ShoulderGeometry":
        return cls(pitch_axis=_unit(np.asarray(d["pitch_axis"], float)),
                   roll_axis=_unit(np.asarray(d["roll_axis"], float)),
                   rest_dir=_unit(np.asarray(d["rest_dir"], float)),
                   pitch_limit=tuple(d["pitch_limit_rad"]), roll_limit=tuple(d["roll_limit_rad"]),
                   yaw_limit=tuple(d["yaw_limit_rad"]))

    def direction(self, pitch: float, roll: float) -> np.ndarray:
        """正运动学：上臂方向 = R(pitch 轴, pitch) R(roll 轴, roll) rest_dir。yaw 不改变上臂方向。"""
        return rot(self.pitch_axis, pitch) @ rot(self.roll_axis, roll) @ self.rest_dir

    @cached_property
    def _direction_terms(self) -> tuple[list[float], list[float], list[float], list[float]]:
        ar, v0 = _unit(self.roll_axis), self.rest_dir
        return _unit(self.pitch_axis).tolist(), v0.tolist(), np.cross(ar, v0).tolist(), (ar * (ar @ v0)).tolist()

    def pointing_error_deg(self, pitch: float, roll: float, h: list[float]) -> float:
        """IB-Robot patch: direction(pitch, roll) 与 h 的夹角（度），Rodrigues 展开成浮点运算，只用于候选打分。"""
        a, v0, c0, k0 = self._direction_terms
        s, c = math.sin(roll), math.cos(roll)
        v = [v0[i] * c + c0[i] * s + k0[i] * (1.0 - c) for i in range(3)]
        s, c = math.sin(pitch), math.cos(pitch)
        axv = (a[1] * v[2] - a[2] * v[1], a[2] * v[0] - a[0] * v[2], a[0] * v[1] - a[1] * v[0])
        av = (a[0] * v[0] + a[1] * v[1] + a[2] * v[2]) * (1.0 - c)
        dot = sum((v[i] * c + axv[i] * s + a[i] * av) * h[i] for i in range(3))
        return math.degrees(math.acos(_clip(dot, -1.0, 1.0)))

    def exact_solutions(self, h: np.ndarray) -> list[tuple[float, float]]:
        """direction(pitch, roll) == h 的解析解（Paden-Kahan 子问题 2）。

        一般有两组：一组 |roll| <= 90（A），另一组 pitch+180、roll -> 180-roll（B）。
        h 不可达（几何上落在两个锥面之外）时，返回最接近的一组。
        """
        ap, ar, v0 = self.pitch_axis, self.roll_axis, self.rest_dir
        d = ap @ ar
        A, B = ap @ h, ar @ v0
        alpha = (A - d * B) / (1.0 - d * d)
        beta = (B - d * A) / (1.0 - d * d)
        n = _cross(ap, ar)
        g2 = (1.0 - alpha * alpha - beta * beta - 2.0 * alpha * beta * d) / (n @ n)
        gammas = (np.sqrt(g2), -np.sqrt(g2)) if g2 > 0 else (0.0,)
        out = []
        for g in gammas:
            c = _unit(alpha * ap + beta * ar + g * n)       # R(ar,roll) v0 == c == R(ap,-pitch) h
            out.append((-_signed_angle(h, c, ap), _signed_angle(v0, c, ar)))
        return out

    def roll_for_pitch(self, h: np.ndarray, pitch: float) -> float:
        """pitch 固定时，让朝向最接近 h 的 roll（子问题 1）。"""
        return _signed_angle(self.rest_dir, rot(self.pitch_axis, -pitch) @ h, self.roll_axis)


# ---------------------------------------------------------------- 参数

@dataclass
class ShoulderParams:
    max_speed_deg_s: float = 360.0       # pitch/roll/yaw 每个关节的最大角速度（兜底）。granger 11-08 接受 360 的效果
    limit_margin_deg: float = 1.0        # 关节限位往里收
    median3: bool = True                 # 输入方向 3 帧中值，去 PEAR 单帧闪烁（1 帧延迟）
    median_window_s: float = 0.10        # 中值只取这么长时间窗里的帧（丢帧空档不能拿来混）
    min_dt_s: float = 1.0 / 60.0         # dt 下限：时间戳重复（dt=0）不能让限速变 0、把手臂冻住
    overhead_back_deg: float | None = 15.0   # 举过头顶最多比正上方往后倒这么多（granger 11-08 定）；None = 不限
    twist_weight: float = 1.0            # 候选打分：yaw 超限 1 度 ≈ 指向误差 1 度（再乘扭转可信度）
    select_tol_deg: float = 2.0          # 候选误差在最优 +这么多以内的，按离上一帧最近来挑


# ---------------------------------------------------------------- 求解器

@dataclass
class ShoulderSolver:
    """一侧肩。每帧调用 step()；状态（上一帧、中值历史）跨帧保留。"""
    geo: ShoulderGeometry
    params: ShoulderParams = field(default_factory=ShoulderParams)
    stats: dict = field(default_factory=lambda: {"frames": 0, "rate_limited": 0, "at_limit": 0,
                                                 "overhead_back_clamped": 0})
    _prev: tuple[float, float] | None = None
    _prev_yaw: float | None = None
    _hist: list = field(default_factory=list)
    _clock: float = 0.0

    def reset(self) -> None:
        """换人、跟随中断、长时间丢帧后调用：下一帧不做连续性约束和限速。"""
        self._prev, self._prev_yaw, self._hist = None, None, []

    # ---- 限位：关节限位留余量，再和 A 写法（|roll| <= 90）取交集
    def _limits(self) -> tuple[np.ndarray, np.ndarray]:
        m = np.radians(self.params.limit_margin_deg)
        lo = np.array([self.geo.pitch_limit[0] + m, self.geo.roll_limit[0] + m])
        hi = np.array([self.geo.pitch_limit[1] - m, self.geo.roll_limit[1] - m])
        lo[1], hi[1] = max(lo[1], -np.pi / 2), min(hi[1], np.pi / 2)
        return lo, hi

    # ---- 输入滤波
    def _filtered_input(self, h: np.ndarray, dt: float) -> np.ndarray:
        if not self.params.median3:
            return h
        self._clock += dt
        self._hist = [(t, v) for t, v in self._hist if self._clock - t <= self.params.median_window_s]
        self._hist.append((self._clock, h))
        self._hist = self._hist[-3:]
        if len(self._hist) < 3:
            return h
        return _unit(np.array([sorted(c)[1] for c in zip(*(v.tolist() for _, v in self._hist), strict=True)]))

    def _limit_overhead_back(self, h: np.ndarray) -> np.ndarray:
        """头顶后方禁区：上臂高于水平且越过「正上方往后倒 θ」的平面时，投影回平面。

        平面经过左右轴，法向 n = (cosθ, 0, sinθ)；n·h < 0 表示在平面后面。只管上半球，
        正常的往后下方摆臂不受影响；投影只去掉沿 n 的分量，左右分量不变。
        """
        th = self.params.overhead_back_deg
        if th is None or h[2] <= 0.0:
            return h
        n = np.array([np.cos(np.radians(th)), 0.0, np.sin(np.radians(th))])
        k = h @ n
        if k >= 0.0:
            return h
        self.stats["overhead_back_clamped"] += 1
        return _unit(h - k * n)

    # ---- yaw 辅助
    def _yaw_overshoot(self, y: float) -> float:
        """y 换到离限位区间最近的 2π 分支后，还超出限位多少（rad，>=0）。"""
        lo, hi = self.geo.yaw_limit
        return min(max(lo - v, 0.0, v - hi) for v in (y - 2 * np.pi, y, y + 2 * np.pi))

    def _migrate_roll(self, prev, h: np.ndarray, pitch_now: float) -> float:
        """pitch 被限速截住时：在当前 pitch 下重解 roll 去贴人的朝向（取离上一帧最近的 2π 分支）。"""
        lo, hi = self._limits()
        r = self.geo.roll_for_pitch(h, pitch_now)
        r = min((r + 2 * np.pi * k for k in (-1, 0, 1)), key=lambda v: abs(v - prev[1]))
        return float(_clip(r, lo[1], hi[1]))

    # ---- 主入口
    def step(self, arm_dir: np.ndarray, dt: float, yaw_need=None) -> tuple[float, float, float | None]:
        p, st = self.params, self.stats
        st["frames"] += 1
        dt = max(float(dt), p.min_dt_s)
        h = self._filtered_input(_unit(np.asarray(arm_dir, dtype=np.float64)), dt)
        h = self._limit_overhead_back(h)
        lo, hi = self._limits()
        prev = self._prev

        # 1. 候选：解析解（各自再试 ±2π），夹进限位。打分 = 指向误差 + 扭转代价（yaw 超出限位的量）
        cands, seen = [], set()
        (lo_p, lo_r), (hi_p, hi_r), h_list = lo.tolist(), hi.tolist(), h.tolist()
        for pitch, roll in self.geo.exact_solutions(h):
            for dp in (0.0, 2 * np.pi, -2 * np.pi):
                for dr in (0.0, 2 * np.pi, -2 * np.pi):
                    q = (_clip(pitch + dp, lo_p, hi_p), _clip(roll + dr, lo_r, hi_r))
                    key = (round(q[0], 6), round(q[1], 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    cands.append((self.geo.pointing_error_deg(*q, h_list), q))
        if yaw_need is not None:
            # IB-Robot patch: yaw_need runs a full FK. The twist cost is never negative, so a
            # candidate whose pointing error alone is past the best total + select_tol_deg can
            # neither win nor enter `ok` below; the rest are not asked. Results are unchanged.
            # Step 4 asks again at the final pitch/roll, which is a scored candidate unless
            # the rate limit moved it, so the answers are kept (yaw_need is fixed within a step).
            pointing, cands, low, asked = sorted(cands, key=lambda c: c[0]), [], float("inf"), {}
            for err, q in pointing:
                if err > low + p.select_tol_deg:
                    break
                y, w = asked[q] = yaw_need(q[0], q[1])
                err += w * p.twist_weight * np.degrees(self._yaw_overshoot(y))
                low = min(low, err)
                cands.append((err, q))

        # 2. 选解：严格先准、再连续。只有一种写法，没有分支歧义；连续性只用来在 ±2π 等价解之间挑。
        #    （不能用「误差 + λ·距离」：经过侧平举时会选中被限位夹住、方向错的近解，pitch 反向转到限位卡死。）
        best = min(c[0] for c in cands)
        if prev is None:
            pitch, roll = min(cands, key=lambda c: c[0])[1]
        else:
            ok = [c for c in cands if c[0] <= best + p.select_tol_deg]
            pitch, roll = min(ok, key=lambda c: np.hypot(c[1][0] - prev[0], c[1][1] - prev[1]))[1]

        # 3. 兜底限速。pitch 被截住时（典型：经过侧平举，pitch 要转很多），roll 不线性插值，
        #    而是在当前 pitch 下重解去贴人的朝向——侧平举附近 pitch 几乎不影响朝向，所以手臂不甩。
        if prev is not None:
            cap = np.radians(p.max_speed_deg_s) * dt
            p_new = float(_clip(pitch, prev[0] - cap, prev[0] + cap))
            if abs(p_new - pitch) > 1e-9:
                st["rate_limited"] += 1
                r_goal = self._migrate_roll(prev, h, p_new)
            else:
                r_goal = roll
            r_new = float(_clip(r_goal, prev[1] - cap, prev[1] + cap))
            if abs(r_new - r_goal) > 1e-9:
                st["rate_limited"] += 1
            pitch, roll = p_new, r_new

        q = (_clip(pitch, lo_p, hi_p), _clip(roll, lo_r, hi_r))
        # np.isclose(atol=1e-6) 的判据（默认 rtol=1e-5）
        if any(abs(v - b) <= 1e-6 + 1e-5 * abs(b) for v, b in zip(q * 2, (lo_p, lo_r, hi_p, hi_r), strict=True)):
            st["at_limit"] += 1
        self._prev = (float(q[0]), float(q[1]))
        if yaw_need is None:
            return self._prev[0], self._prev[1], None

        # 4. yaw：在最终的 pitch/roll 下取需要的 yaw -> 离上一帧最近的 2π 分支 -> 夹限位 -> 限速
        y, _ = asked[self._prev] if self._prev in asked else yaw_need(*self._prev)
        ylo, yhi = self.geo.yaw_limit
        m = np.radians(p.limit_margin_deg)
        ref = self._prev_yaw if self._prev_yaw is not None else 0.0
        y = min((y + 2 * np.pi * k for k in (-1, 0, 1)), key=lambda v: abs(v - ref))
        y = float(_clip(y, ylo + m, yhi - m))
        if self._prev_yaw is not None:
            cap = np.radians(p.max_speed_deg_s) * dt
            y = float(_clip(y, self._prev_yaw - cap, self._prev_yaw + cap))
        self._prev_yaw = y
        return self._prev[0], self._prev[1], y
