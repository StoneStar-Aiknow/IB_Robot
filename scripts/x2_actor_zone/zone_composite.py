"""把一次采样（snap.py 存的一组截图）叠成一张图，每帧里人的脚点标成红点和帧号，给 zone_click 点角用。

    python3 zone_composite.py <截图目录> <输出.jpg> [--empty 5] [--zone 旧区域.yaml]

做法：前 --empty 张是空场（没人），取中值当背景；其余每张和背景做差，
差得多的像素当成人（--thresh 每像素差、--min-px 像素数下限，不够就算这张没人）。
脚点 = 人那块最低处（99.5 分位）往上 30 px 内像素 x 的中值。人那块半透明叠到背景上。
同时写 <输出>.feet.json：[{"frame": 帧号, "file": 文件名, "foot": [x, y] 或 null, "pixels": n}, ...]。

这只是帮着看：背景里有人走动、灯光变了，差出来的块就不准。不准时直接拿单张截图
（谁站在角上的那张）给 zone_click 点，效果一样。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from zone_common import draw_zone, load_zone

C_DOT = (0, 0, 255)


def frame_no(path):
    m = re.match(r"snap_(\d+)_", path.name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snap_dir")
    ap.add_argument("out")
    ap.add_argument("--empty", type=int, default=5, help="开头几张是空场，默认 5")
    ap.add_argument("--thresh", type=int, default=45, help="每像素差多少算变了（0–255），默认 45")
    ap.add_argument("--min-px", type=int, default=3000, help="变了的像素少于这么多就算这张没人，默认 3000")
    ap.add_argument("--zone", help="可选：把这个区域文件也画上去（比如会场之前的旧区域）")
    a = ap.parse_args()

    files = sorted(Path(a.snap_dir).glob("snap_*.jpg"), key=lambda p: (frame_no(p) is None, frame_no(p), p.name))
    if len(files) <= a.empty:
        sys.exit(f"{a.snap_dir} 里只有 {len(files)} 张 snap_*.jpg，至少要比 --empty {a.empty} 多")
    imgs = [cv2.imread(str(p)) for p in files]
    bad = [p.name for p, im in zip(files, imgs, strict=True) if im is None]
    if bad:
        sys.exit(f"读不了：{bad}")
    sizes = {im.shape for im in imgs}
    if len(sizes) != 1:
        sys.exit(f"截图尺寸不一样：{sizes}")
    h, w = imgs[0].shape[:2]
    empty = np.median(np.stack(imgs[: a.empty]), axis=0).astype(np.uint8)
    out = empty.copy()
    kernel = np.ones((5, 5), np.uint8)
    feet = []
    marks = []
    for path, img in zip(files, imgs, strict=True):
        n = frame_no(path)
        diff = cv2.absdiff(img, empty).max(axis=2) > a.thresh
        mask = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        pixels = int(mask.sum())
        foot = None
        if pixels >= a.min_px:
            ys, xs = np.nonzero(mask)
            foot_y = float(np.percentile(ys, 99.5))
            foot_x = float(np.median(xs[ys > foot_y - 30]))
            foot = [int(round(foot_x)), int(round(foot_y))]
            out[mask] = (0.5 * out[mask] + 0.5 * img[mask]).astype(np.uint8)
            marks.append((n, foot))
        feet.append({"frame": n, "file": path.name, "foot": foot, "pixels": pixels})
        shown = f"脚点 ({foot[0]:4d}, {foot[1]:4d})" if foot else "没人            "
        print(f"帧 {n:>3}  {shown}  变化像素 {pixels}")
    if a.zone:
        z = load_zone(a.zone)
        zw, zh = z["image_size"]
        sx, sy = w / zw, h / zh
        draw_zone(out, [v * sx for v in z["crop_x"]], [(x * sx, y * sy) for x, y in z["floor_quad"]])
    for n, (x, y) in marks:
        cv2.circle(out, (x, y), 5, C_DOT, -1, cv2.LINE_AA)
        cv2.putText(out, str(n), (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(n), (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DOT, 1, cv2.LINE_AA)
    if not cv2.imwrite(a.out, out):
        sys.exit(f"写不了：{a.out}")
    feet_path = Path(a.out + ".feet.json")
    feet_path.write_text(json.dumps(feet, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"空场 {a.empty} 张，有人 {len(marks)} 张，共 {len(files)} 张（{w}x{h}）")
    print(f"已写 {a.out} 和 {feet_path}")


if __name__ == "__main__":
    main()
