"""把 actor_zone.yaml 画到截图上给 zll 核对；可选把一组人框和脚点画上去。

    python3 actor_overlay.py <截图> <actor_zone.yaml> <输出.png> [--boxes boxes.json]

--boxes：[{"bbox":[x1,y1,x2,y2],"conf":0.9,"label":"person"}, ...]，坐标是截图像素。
脚点 = 框底边中点（框中线，不是节点里的粘性脚点）。
截图尺寸和 image_size 不同时，区域按比例缩放到截图上，并打印一行提示。
"""

import argparse
import json
import sys

import cv2
from zone_common import C_BOX, C_FOOT, MARGIN_DEFAULT, OFFICIAL_MOD, draw_zone, load_zone, temp_check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("zone")
    ap.add_argument("out")
    ap.add_argument("--boxes")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"读不了图：{a.image}")
    z = load_zone(a.zone)
    errs = temp_check(z)
    if errs:
        # 只提示，照样画出来方便看哪里错；正式校验请用 zone_click 或 actor_zone_config
        print(
            f"注意：区域文件没过临时检查（不是正式校验{'' if not OFFICIAL_MOD.exists() else '，正式校验已存在请另跑'}）："
        )
        for e in errs:
            print(f"  {a.zone}: {e}")
    zw, zh = z["image_size"]
    h, w = img.shape[:2]
    sx, sy = w / zw, h / zh
    if (w, h) != (zw, zh):
        print(f"提示：截图 {w}x{h} 和 image_size {zw}x{zh} 不同，区域按 x×{sx:.4f}、y×{sy:.4f} 缩放")
    crop = [v * sx for v in z["crop_x"]]
    quad = [(x * sx, y * sy) for x, y in z["floor_quad"]]
    margin = z.get("near_edge_margin_px", MARGIN_DEFAULT) * sy
    draw_zone(img, crop, quad, margin)

    if a.boxes:
        with open(a.boxes, encoding="utf-8") as f:
            boxes = json.load(f)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), C_BOX, 2, cv2.LINE_AA)
            fx, fy = (x1 + x2) / 2, y2
            cv2.circle(img, (int(round(fx)), int(round(fy))), 6, C_FOOT, -1, cv2.LINE_AA)
            tag = f"#{i} {b.get('label', '')} {b.get('conf', float('nan')):.2f} foot({fx:.0f},{fy:.0f})"
            cv2.putText(
                img, tag, (int(x1) + 2, max(14, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BOX, 1, cv2.LINE_AA
            )
        print(f"画了 {len(boxes)} 个框")
    if not cv2.imwrite(a.out, img):
        sys.exit(f"写不了：{a.out}")
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
