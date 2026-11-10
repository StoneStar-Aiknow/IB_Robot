#!/usr/bin/env python3
"""X2 照镜子拍照：在上位机上把 inbox 做成一页 review.html，逐张看「拍得对不对」。只用标准库，不改 inbox 里的照片。

    python3 review.py                     默认 ./x2_photos，写 ./x2_photos/review.html
    python3 review.py D:/x2_photos        指定 inbox

每张照片上画出人框（红框，就是发给 AIGC 的 bbox），旁边列出：
本轮怎么选的（source/label、score、total、scoring）、本轮多长、有几个候选、各来源各几个。
页首汇总每种 quality、每种来源各几张，以及账本里收到几次、重发几次。
页面用相对路径引用同目录的照片，直接用浏览器打开；照片不复制、不上传。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

LEDGER = "ledger.jsonl"
PAGE = "review.html"


def load_photos(inbox: Path) -> list[dict]:
    """inbox 里每张照片的元数据（有 .jpg 才算），新的在前。"""
    photos = []
    for meta_path in inbox.glob("*.json"):
        image = meta_path.with_suffix(".jpg")
        if not image.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            meta["_image"] = image.name
            photos.append(meta)
    photos.sort(key=lambda meta: (str(meta.get("received_at") or ""), meta["_image"]), reverse=True)
    return photos


def ledger_counts(inbox: Path) -> tuple[int, int]:
    """账本里 (新照片次数, 重发次数)。"""
    new = again = 0
    try:
        lines = (inbox / LEDGER).read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, 0
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            if entry.get("duplicate"):
                again += 1
            else:
                new += 1
    return new, again


def box_style(meta: dict) -> str | None:
    """人框在图上的位置（百分比）；框不可用时 None。"""
    try:
        x1, y1, x2, y2 = (float(value) for value in meta.get("bbox") or ())
        width, height = float(meta["width"]), float(meta["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return (
        f"left:{100 * x1 / width:.2f}%;top:{100 * y1 / height:.2f}%;"
        f"width:{100 * (x2 - x1) / width:.2f}%;height:{100 * (y2 - y1) / height:.2f}%"
    )


def round_seconds(meta: dict) -> str:
    try:  # "2026-11-10T12:18:53.123+0800": fromisoformat takes no "+0800" before Python 3.11
        started = datetime.strptime(meta["round_started_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
        ended = datetime.strptime(meta["round_ended_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
    except (KeyError, TypeError, ValueError):
        return "-"
    return f"{(ended - started).total_seconds():.1f} s"


def _row(name: str, value) -> str:
    if value is None or value == "":
        value = "-"
    elif isinstance(value, dict):
        value = ", ".join(f"{key} {count}" for key, count in sorted(value.items())) or "-"
    return f"<tr><th>{html.escape(name)}</th><td>{html.escape(str(value))}</td></tr>"


def card(meta: dict) -> str:
    style = box_style(meta)
    box = f'<div class="box" style="{style}"></div>' if style else '<div class="nobox">没有人框</div>'
    picked = "/".join(str(meta.get(key) or "-") for key in ("source", "label"))
    rows = [
        _row("quality", meta.get("quality")),
        _row("选中", picked),
        _row("score / weight", f"{meta.get('score', '-')} / {meta.get('weight', '-')}"),
        _row("total", meta.get("total")),
        _row("scoring", meta.get("scoring")),
        _row("speed", meta.get("speed")),
        _row("本轮时长", round_seconds(meta)),
        _row("结束原因", meta.get("end_reason")),
        _row("候选数", meta.get("candidates")),
        _row("候选来源", meta.get("candidate_sources")),
        _row("bbox", meta.get("bbox")),
        _row("尺寸", f"{meta.get('width')}x{meta.get('height')}, {meta.get('bytes')} bytes"),
        _row("拍摄", meta.get("captured_at")),
        _row("收到", meta.get("received_at")),
    ]
    if meta.get("error"):
        rows.append(_row("error", meta.get("error")))
    name = html.escape(meta["_image"])
    return (
        f'<div class="card q-{html.escape(str(meta.get("quality")))}"><h3>{html.escape(str(meta.get("round_id")))}</h3>'
        f'<div class="pic"><img src="{name}" alt="{name}">{box}</div>'
        f"<table>{''.join(rows)}</table></div>"
    )


def render(inbox: Path) -> tuple[str, int]:
    photos = load_photos(inbox)
    new, again = ledger_counts(inbox)
    qualities = Counter(str(meta.get("quality")) for meta in photos)
    sources = Counter(f"{meta.get('source') or '-'}/{meta.get('label') or '-'}" for meta in photos)
    summary = (
        f"<p>{len(photos)} 张照片；账本：新照片 {new} 次、重发 {again} 次。生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p>quality：{html.escape(', '.join(f'{k} {v}' for k, v in sorted(qualities.items())) or '-')}</p>"
        f"<p>选中来源：{html.escape(', '.join(f'{k} {v}' for k, v in sorted(sources.items())) or '-')}</p>"
    )
    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>X2 照镜子拍照 review</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
.card {{ display: inline-block; vertical-align: top; margin: 8px; padding: 8px; border: 1px solid #ccc; width: 660px; }}
.card.q-fallback, .card.q-recovered {{ background: #fff4e0; }}
.card.q-test {{ background: #eee; }}
.pic {{ position: relative; display: inline-block; }}
.pic img {{ display: block; width: 640px; }}
.box {{ position: absolute; border: 3px solid red; box-sizing: border-box; }}
.nobox {{ position: absolute; left: 4px; top: 4px; background: #c00; color: #fff; padding: 2px 6px; }}
th {{ text-align: left; padding-right: 12px; font-weight: normal; color: #555; }}
h3 {{ margin: 0 0 6px; font-size: 14px; }}
</style></head><body>
<h2>X2 照镜子拍照 review（{html.escape(str(inbox))}）</h2>
{summary}
{"".join(card(meta) for meta in photos)}
</body></html>
"""
    return page, len(photos)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Make review.html for the photos in a receiver inbox.")
    parser.add_argument("inbox", nargs="?", default="x2_photos")
    args = parser.parse_args(argv)
    inbox = Path(args.inbox).resolve()
    if not inbox.is_dir():
        print(f"没有这个目录：{inbox}", file=sys.stderr)
        return 1
    page, count = render(inbox)
    out = inbox / PAGE
    part = out.with_name(out.name + ".part")
    part.write_text(page, encoding="utf-8")
    os.replace(part, out)
    print(f"{count} 张 → {out}（用浏览器打开）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
