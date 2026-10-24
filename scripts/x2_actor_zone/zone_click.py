"""在截图上点画面带和 4 个脚点，写出 actor_zone.yaml（设计 9.1 格式），然后校验。

    python3 zone_click.py <截图.jpg> <输出 actor_zone.yaml> [--margin 30] [--note "..."]
    python3 zone_click.py <截图> <输出> --points "L,R,x1,y1,x2,y2,x3,y3,x4,y4"   # 不开窗口

点的顺序：
  1. 两下：画面带左线 x、右线 x（只取 x）；
  2. 四下：近左、近右、远右、远左（近 = 图上偏下）。
按键：u 撤销上一点，r 重来，s 或回车 写文件并校验，q 或 Esc 退出。
截图大于窗口时缩放显示，点击坐标换回原图像素。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from zone_common import CORNERS, MARGIN_DEFAULT, check, draw_zone, zone_yaml

STEPS = ["crop left x", "crop right x"] + list(CORNERS)


def write(out, w, h, pts, margin, note):
    L, R = pts[0][0], pts[1][0]
    if L > R:
        print(f"注意：先点的左线 x={L:.0f} 在右线 x={R:.0f} 右边，已按大小换过来")
        L, R = R, L
    quad = [p for p in pts[2:6]]
    out.write_text(zone_yaml(w, h, (L, R), quad, margin, note), encoding="utf-8")
    print(f"已写 {out}")
    return check(out)


def gui(img, out, margin, note, max_w, max_h):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    base = (
        cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    )
    pts = []
    msg = [""]

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 6:
            px, py = min(max(x / s, 0), w - 1), min(max(y / s, 0), h - 1)
            pts.append((int(round(px)), int(round(py))))
            msg[0] = ""

    win = "zone_click"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        sys.exit(f"这个 python（{sys.executable}）的 cv2 没有窗口支持；用 /usr/bin/python3 跑（系统 cv2 4.5.4 带 GTK）")
    cv2.setMouseCallback(win, on_mouse)
    print(f"图 {w}x{h}，显示缩放 {s:.3f}")
    while True:
        view = base.copy()
        draw_zone(view, [p[0] for p in pts[:2]], pts[2:6], margin, s)
        nxt = STEPS[len(pts)] if len(pts) < 6 else "done: s/Enter save"
        lines = [f"next: {nxt}   [u]ndo [r]eset [s]ave [q]uit", msg[0]]
        for i, t in enumerate(lines):
            if t:
                cv2.putText(view, t, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, t, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            print("退出（这次没写文件）")
            return 1
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("r"):
            pts.clear()
        elif k in (ord("s"), 13, 10):
            if len(pts) < 6:
                msg[0] = f"need 6 clicks, have {len(pts)}"
                continue
            if write(out, w, h, pts, margin, note):
                cv2.destroyAllWindows()
                return 0
            msg[0] = "check FAILED, see terminal; u/r to fix, s to save again"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--margin", type=int, default=MARGIN_DEFAULT, help="near_edge_margin_px，默认 30，最小 16")
    ap.add_argument("--note", default="", help="写进 note，前面自动加时间和截图名")
    ap.add_argument("--points", help="非交互：L,R,x1,y1,x2,y2,x3,y3,x4,y4（近左、近右、远右、远左）")
    ap.add_argument("--max-w", type=int, default=1280, help="窗口最大宽")
    ap.add_argument("--max-h", type=int, default=900, help="窗口最大高")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"读不了图：{a.image}")
    h, w = img.shape[:2]
    out = Path(a.out)
    note = f"{time.strftime('%Y-%m-%d %H:%M')} zone_click 对着 {Path(a.image).name}（{w}x{h}）画"
    if a.note:
        note += "；" + a.note

    if a.points:
        v = [float(t) for t in a.points.split(",")]
        if len(v) != 10:
            sys.exit(f"--points 要 10 个数，给了 {len(v)} 个")
        pts = [(v[0], 0), (v[1], 0)] + [(v[i], v[i + 1]) for i in range(2, 10, 2)]
        sys.exit(0 if write(out, w, h, pts, a.margin, note) else 1)
    sys.exit(gui(img, out, a.margin, note, a.max_w, a.max_h))


if __name__ == "__main__":
    main()
