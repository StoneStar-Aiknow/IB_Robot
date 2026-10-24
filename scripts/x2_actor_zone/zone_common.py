"""zone_click.py / actor_overlay.py / zone_composite.py 共用：读写 actor_zone.yaml、校验、画区域。

正式校验是本仓 src/manipulation_execution 里的 manipulation_execution.actor_zone_config（节点读文件用的就是它）。
temp_check 只是照设计 9.1 的「校验」列表做的临时检查，只在 actor_overlay 里用来提示，不代替正式校验。
ACTOR_TREE 默认是本文件所在的仓库根，可用环境变量换成别的树。
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2

ACTOR_TREE = Path(os.environ.get("ACTOR_TREE", Path(__file__).resolve().parents[2]))
OFFICIAL_PKG = ACTOR_TREE / "src/manipulation_execution"
OFFICIAL_MOD = OFFICIAL_PKG / "manipulation_execution/actor_zone_config.py"

KEYS = ("version", "image_size", "crop_x", "floor_quad", "near_edge_margin_px", "note")
CORNERS = ("near_left", "near_right", "far_right", "far_left")
MARGIN_DEFAULT = 30
MARGIN_MIN = 16


def zone_yaml(w, h, crop_x, quad, margin, note):
    """设计 9.1 的格式，键名、顺序、注释照抄。"""
    L, R = crop_x
    q = [[int(round(x)), int(round(y))] for x, y in quad]
    return (
        "version: 1\n"
        "# 画区域时用的图像尺寸。运行时帧尺寸不同，就按比例缩放，并 warn 一次。\n"
        f"image_size: [{int(w)}, {int(h)}]\n"
        "# 画面带：框整个落在这个区间外面的人，完全不管。\n"
        f"crop_x: [{int(round(L))}, {int(round(R))}]\n"
        "# 地面区域：图上 4 个脚的位置，顺序固定。\n"
        "#   近左、近右、远右、远左（近 = 靠机器人 = 图上偏下）\n"
        "floor_quad:\n"
        f"  - [{q[0][0]}, {q[0][1]}]   # near_left\n"
        f"  - [{q[1][0]}, {q[1][1]}]   # near_right\n"
        f"  - [{q[2][0]}, {q[2][1]}]   # far_right\n"
        f"  - [{q[3][0]}, {q[3][1]}]   # far_left\n"
        "# 可选，默认 30，最小 16。近边两个点离画面底至少这么多 px。\n"
        f"near_edge_margin_px: {int(margin)}\n"
        "# 谁、什么时候、对着哪张图画的\n"
        f"note: {json.dumps(note, ensure_ascii=False)}\n"
    )


def load_zone(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _num(v):
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def temp_check(z):
    """临时检查（设计 9.1 校验列表）。返回出错列表，空 = 通过。"""
    if not isinstance(z, dict):
        return ["整个文件不是一个映射"]
    errs = []
    for k in z:
        if k not in KEYS:
            errs.append(f"{k}: 不认识的键")
    if z.get("version") != 1:
        errs.append(f"version: 必须是 1，现在是 {z.get('version')!r}")
    size = z.get("image_size")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
    ):
        errs.append(f"image_size: 必须是两个正整数，现在是 {size!r}")
        return errs
    w, h = size
    cx = z.get("crop_x")
    if not (isinstance(cx, list) and len(cx) == 2 and all(_num(v) for v in cx)):
        errs.append(f"crop_x: 必须是两个数，现在是 {cx!r}")
        return errs
    L, R = cx
    if not (0 <= L < R <= w):
        errs.append(f"crop_x: 要满足 0 ≤ L < R ≤ {w}，现在是 [{L}, {R}]")
    margin = z.get("near_edge_margin_px", MARGIN_DEFAULT)
    if not (_num(margin) and margin >= MARGIN_MIN):
        errs.append(f"near_edge_margin_px: 必须 ≥ {MARGIN_MIN}，现在是 {margin!r}")
        margin = MARGIN_MIN
    q = z.get("floor_quad")
    if not (
        isinstance(q, list)
        and len(q) == 4
        and all(isinstance(p, list) and len(p) == 2 and all(_num(v) for v in p) for p in q)
    ):
        errs.append(f"floor_quad: 必须是 4 个 [x, y]，现在是 {q!r}")
        return errs
    for name, (x, y) in zip(CORNERS, q, strict=True):
        if not (0 <= x < w and 0 <= y < h):
            errs.append(f"floor_quad.{name}: ({x}, {y}) 不在图内 {w}x{h}")
        if not (L <= x <= R):
            errs.append(f"floor_quad.{name}: x={x} 不在 crop_x [{L}, {R}] 里")
    nl, nr, fr, fl = q
    if not nl[0] < nr[0]:
        errs.append(f"floor_quad: near_left.x {nl[0]} 要小于 near_right.x {nr[0]}")
    if not fl[0] < fr[0]:
        errs.append(f"floor_quad: far_left.x {fl[0]} 要小于 far_right.x {fr[0]}")
    if not nl[1] > fl[1]:
        errs.append(f"floor_quad: near_left.y {nl[1]} 要大于 far_left.y {fl[1]}")
    if not nr[1] > fr[1]:
        errs.append(f"floor_quad: near_right.y {nr[1]} 要大于 far_right.y {fr[1]}")
    cross = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
        errs.append(f"floor_quad: 不是凸四边形或顺序不对（叉积 {cross}）")
    for name, (_x, y) in (("near_left", nl), ("near_right", nr)):
        if y > h - margin:
            errs.append(f"floor_quad.{name}: y={y} 要 ≤ 图高 {h} − near_edge_margin_px {margin} = {h - margin}")
    return errs


def check(path):
    """先用正式校验；没有时用临时检查并说明。返回 True = 通过。"""
    if OFFICIAL_MOD.exists():
        env = dict(os.environ)
        env["PYTHONPATH"] = str(OFFICIAL_PKG) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        p = subprocess.run(
            [sys.executable, "-m", "manipulation_execution.actor_zone_config", "--zone", str(path)],
            env=env,
            capture_output=True,
            text=True,
        )
        print(f"[正式校验] {OFFICIAL_MOD} rc={p.returncode}")
        for line in (p.stdout + p.stderr).strip().splitlines():
            print("  " + line)
        return p.returncode == 0
    print(f"[临时检查] 用的是临时检查，不是正式校验（{OFFICIAL_MOD} 还不存在）")
    errs = temp_check(load_zone(path))
    for e in errs:
        print(f"  {path}: {e}")
    print("  临时检查通过" if not errs else f"  临时检查不通过：{len(errs)} 处")
    return not errs


def dashed_hline(img, y, x0, x1, color, thick=1, dash=12):
    for x in range(int(x0), int(x1), dash * 2):
        cv2.line(img, (x, int(y)), (min(x + dash, int(x1)), int(y)), color, thick, cv2.LINE_AA)


# 颜色（BGR）
C_CROP = (255, 200, 0)
C_QUAD = (0, 255, 0)
C_MARGIN = (0, 0, 255)
C_BOX = (0, 200, 255)
C_FOOT = (255, 0, 255)


def draw_zone(img, crop_x=None, quad=(), margin=None, s=1.0, closed=True):
    """在 img 上画区域。坐标是原图像素，s = 显示缩放。quad 可以不满 4 个点。"""
    h, w = img.shape[:2]
    t = max(1, int(round(2 * s)))
    if margin is not None:
        y = h - margin * s
        dashed_hline(img, y, 0, w, C_MARGIN, t)
        cv2.putText(
            img, f"near margin {margin}px", (8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MARGIN, 1, cv2.LINE_AA
        )
    for x in crop_x or ():
        cv2.line(img, (int(round(x * s)), 0), (int(round(x * s)), h), C_CROP, t, cv2.LINE_AA)
    pts = [(int(round(x * s)), int(round(y * s))) for x, y in quad]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], C_QUAD, t, cv2.LINE_AA)
    if closed and len(pts) == 4:
        cv2.line(img, pts[3], pts[0], C_QUAD, t, cv2.LINE_AA)
    for name, p, (x, y) in zip(CORNERS, pts, quad, strict=False):  # quad 可以不满 4 个点
        cv2.circle(img, p, 5, C_QUAD, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{name} ({int(round(x))},{int(round(y))})",
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            C_QUAD,
            1,
            cv2.LINE_AA,
        )
