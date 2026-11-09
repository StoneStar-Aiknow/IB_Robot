"""shoulder_ik 的验收测试。只依赖 numpy + geometry_x2.json，不需要 mujoco / PEAR / 人体数据。

    python -m pytest -q test_shoulder_ik.py

IB-Robot: copied from the port package beside _vendor/shoulder_ik.py (see _vendor/SOURCES.md);
only the import and the data directory changed.

每个测试对应一条几何结论或一个踩过的坑（见 HANDOFF_SHOULDER_IK.md 第 5 节）。
移植到 summit 后，把 import 换成新位置，这些测试必须原样通过；golden 测试不过 = 行为变了，先查清再说。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manipulation_execution.imitation_retargeting.x2._vendor.shoulder_ik import (
    ShoulderGeometry,
    ShoulderParams,
    ShoulderSolver,
)

HERE = Path(__file__).resolve().parent / "data" / "x2_shoulder_ik"
GEO = {s: ShoulderGeometry.from_dict(d)
       for s, d in json.loads((HERE / "geometry_x2.json").read_text())["sides"].items()}
SIDES = ("left", "right")


# ---------------------------------------------------------------- 合成路径（golden 用）

def sph(elev_deg, azim_deg, side="left"):
    """抬高（0 下垂/180 正上）+ 方位（0 正前/90 外侧/180 正后）-> 躯干系 (前, 左, 上)。右臂左右镜像。"""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    d = np.array([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), -np.cos(e)])
    return d if side == "left" else d * np.array([1, -1, 1])


def path(keys, n, side="left"):
    """沿球面大圆段插值，n 帧。"""
    pts = [sph(e, a, side) for e, a in keys]
    seg = np.array([np.arccos(np.clip(a @ b, -1, 1)) for a, b in zip(pts, pts[1:])])
    cum = np.r_[0, np.cumsum(seg)]
    out = []
    for x in np.linspace(0, cum[-1], n):
        i = min(np.searchsorted(cum[1:], x, side="right"), len(seg) - 1)
        a, b, th, x0 = pts[i], pts[i + 1], seg[i], x - cum[i]
        out.append(a if th < 1e-9 else (np.sin(th - x0) * a + np.sin(x0) * b) / np.sin(th))
    return np.array(out) / np.linalg.norm(out, axis=1, keepdims=True)


PATHS = {
    "front_raise_overhead":   ([(0, 0), (90, 0), (170, 0), (90, 0), (0, 0)], 90),
    "side_raise_overhead":    ([(0, 90), (90, 90), (170, 90), (90, 90), (0, 90)], 90),
    "codman_front_side_down": ([(0, 0), (90, 0), (90, 90), (90, 100), (0, 100)], 90),
    "side_overhead_front_down": ([(0, 90), (90, 90), (165, 60), (90, 0), (0, 0)], 90),
    "circle_at_side":         ([(90, 90), (120, 60), (90, 30), (60, 60), (90, 90)], 60),
    "wave_high":              ([(130, 60), (150, 90), (130, 120), (150, 90), (130, 60)], 60),
}


def run(side, H, dt=1 / 30, params=None):
    s = ShoulderSolver(GEO[side], params or ShoulderParams())
    dts = np.broadcast_to(dt, (len(H),)) if np.isscalar(dt) else dt
    return np.array([s.step(h, d)[:2] for h, d in zip(H, dts)]), s


# ---------------------------------------------------------------- 几何

@pytest.mark.parametrize("side", SIDES)
def test_exact_solutions_roundtrip(side):
    """解析解是 FK 的精确逆：任意可达 (pitch, roll) -> 方向 -> 解里一定有原来那组（mod 2π）。"""
    g, rng = GEO[side], np.random.default_rng(0)
    for _ in range(300):
        p, r = rng.uniform(-3.0, 2.0), rng.uniform(-1.5, 1.5)
        sols = g.exact_solutions(g.direction(p, r))
        d = [np.hypot((sp - p + np.pi) % (2 * np.pi) - np.pi, (sr - r + np.pi) % (2 * np.pi) - np.pi)
             for sp, sr in sols]
        assert min(d) < 1e-6


@pytest.mark.parametrize("side", SIDES)
def test_two_solutions_same_direction(side):
    """两组解（A/B）指向完全相同 —— 它们是同一个上臂朝向的两种关节写法。"""
    g, rng = GEO[side], np.random.default_rng(1)
    for _ in range(200):
        h = rng.normal(size=3)
        h /= np.linalg.norm(h)
        sols = g.exact_solutions(h)
        if len(sols) == 2:
            for s in sols:
                assert np.degrees(np.arccos(np.clip(g.direction(*s) @ h, -1, 1))) < 1e-4   # arccos 在 1 附近的浮点精度约 1e-6 度


# ---------------------------------------------------------------- 只用 A 写法

@pytest.mark.parametrize("side", SIDES)
def test_never_uses_branch_b(side):
    """|roll| 永远不超过 90 度（只用一张地图 -> 远处换写法按构造不可能）。"""
    rng = np.random.default_rng(2)
    for _ in range(10):
        keys = [(rng.uniform(0, 170), rng.uniform(-20, 150)) for _ in range(6)]
        q, _ = run(side, path(keys, 120, side))
        assert np.abs(q[:, 1]).max() <= np.pi / 2 + 1e-9


@pytest.mark.parametrize("side", SIDES)
def test_side_raise_spins_the_right_way(side):
    """坑：侧举经过侧平举时 pitch 要转到 ≈ −170（往前翻上头顶）。
    旧的「误差 + λ·距离」选解会让 pitch 往反方向转、撞 +116.9 的限位卡死。"""
    q, _ = run(side, path([(0, 90), (90, 90), (170, 90)], 90, side))
    assert np.degrees(q[-1, 0]) < -150, f"pitch 应接近 −170，实际 {np.degrees(q[-1, 0]):.1f}"
    # 经过奇异点附近时 pitch 本来就会短暂摆动（实测最高约 +79 度）；bug 的症状是撞上 +116.9 的上限并卡住
    upper = np.degrees(GEO[side].pitch_limit[1])
    assert np.degrees(q[:, 0]).max() < upper - 10, "pitch 撞到了上限位（往反方向转的老 bug）"


@pytest.mark.parametrize("side", SIDES)
def test_pointing_accuracy_when_slow(side):
    """动作够慢（不触发限速）时，A 可达区域里指向误差 < 2 度。"""
    g = GEO[side]
    H = path([(10, 60), (60, 30), (90, 10), (60, 60), (30, 80)], 300, side)
    q, _ = run(side, H, params=ShoulderParams(median3=False))
    err = [np.degrees(np.arccos(np.clip(g.direction(*x) @ h, -1, 1))) for x, h in zip(q, H)]
    assert max(err) < 2.0


# ---------------------------------------------------------------- 兜底

@pytest.mark.parametrize("side", SIDES)
def test_rate_limit(side):
    """每帧每个关节不超过 max_speed * dt。"""
    p = ShoulderParams(max_speed_deg_s=360.0)
    q, _ = run(side, path([(0, 0), (170, 0), (0, 90), (170, 90)], 40, side), params=p)
    assert np.degrees(np.abs(np.diff(q, axis=0))).max() <= 360.0 / 30 + 1e-6


@pytest.mark.parametrize("side", SIDES)
def test_duplicate_timestamp_does_not_freeze(side):
    """坑：dt=0（时间戳重复）时限速不能变 0。实测 rec_202609xx 里有 5–19 处。"""
    H = path([(0, 60), (90, 60)], 30, side)
    q, _ = run(side, H, dt=np.zeros(len(H)))
    assert np.degrees(q[-1, 1] - q[0, 1]) * (1 if side == "left" else -1) > 30


def test_median_ignores_stale_frames():
    """坑：中值只取 median_window_s 以内的帧；丢帧空档后不能拿 0.3 s 前的帧来混。"""
    s = ShoulderSolver(GEO["left"])
    a, b = sph(10, 60), sph(80, 60)
    s.step(a, 1 / 30)
    s.step(a, 1 / 30)
    q = s.step(b, 0.3)                      # 空档 0.3 s 后突然变到 b
    assert np.degrees(np.arccos(np.clip(GEO["left"].direction(*q[:2]) @ b, -1, 1))) < 5


def test_overhead_back_limit():
    """头顶后方禁区：人上臂比正上方往后倒 40 度时，机器人最多往后倒约 15 度。"""
    g = GEO["left"]
    h = np.array([-np.sin(np.radians(40)), 0.0, np.cos(np.radians(40))])
    s = ShoulderSolver(g, ShoulderParams(max_speed_deg_s=1e9, median3=False))
    for _ in range(5):
        q = s.step(h, 1 / 30)
    d = g.direction(*q[:2])
    assert np.degrees(np.arctan2(-d[0], d[2])) < 17


def test_yaw_overshoot_steers_choice():
    """有 yaw_need 时，yaw 被夹进限位并限速；返回三个值。"""
    g = GEO["left"]
    s = ShoulderSolver(g)
    ys = [s.step(sph(40, 60), 1 / 30, lambda p, r: (3.0, 1.0))[2] for _ in range(60)]
    assert abs(ys[-1]) <= g.yaw_limit[1] and np.degrees(abs(ys[1] - ys[0])) <= 12 + 1e-6


# ---------------------------------------------------------------- golden：行为不能悄悄变

@pytest.mark.parametrize("name", list(PATHS))
@pytest.mark.parametrize("side", SIDES)
def test_golden(name, side):
    keys, n = PATHS[name]
    q, _ = run(side, path(keys, n, side))
    gold = np.load(HERE / "golden_shoulder_ik.npz")[f"{name}_{side}"]
    assert np.allclose(q, gold, atol=1e-9), f"{name}/{side} 最大差 {np.degrees(np.abs(q - gold).max()):.4g} 度"


if __name__ == "__main__":       # 生成 golden（只在确认行为正确之后）
    out = {}
    for name, (keys, n) in PATHS.items():
        for side in SIDES:
            out[f"{name}_{side}"] = run(side, path(keys, n, side))[0]
    np.savez(HERE / "golden_shoulder_ik.npz", **out)
    print("golden written:", len(out))
