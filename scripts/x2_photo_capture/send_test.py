#!/usr/bin/env python3
"""X2 照镜子拍照：不带 ROS、不连机器人，按拍照节点的同一套代码往上位机发图。在 .5 上（或任何有本仓的机器上）跑。

    python3 send_test.py health [<ip>[:端口]]       本机会从哪个地址发出（只查路由，不发包），再访问一次 /health
    python3 send_test.py send [<ip>[:端口]] [-n 3]  发 n 张合成图（64x48 色块，没有人像），每张都要等到回执

不写地址时用拍照节点的目标文件（默认 /root/x2-feat-photo-capture-run/photo_target，环境变量 PHOTO_TARGET_FILE 可换）。
发送用的就是节点的 manipulation_execution.mirror_photo.uploader.post_photo（从本仓 src 导入，不需要 source ROS）。
测试图的 quality 是 "test"、轮次 id 以 sendtest- 开头，上位机工作流可以据此跳过。

退出码：0 全部成功；1 对方没回或没确认；2 地址写错（photo_target.sh 据此决定写不写）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import secrets
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src" / "manipulation_execution"), str(REPO / "src" / "embodied_common")]

from manipulation_execution.mirror_photo.config import normalize_target  # noqa: E402
from manipulation_execution.mirror_photo.uploader import Refused, post_photo, read_target  # noqa: E402

TARGET_FILE = os.environ.get("PHOTO_TARGET_FILE", "/root/x2-feat-photo-capture-run/photo_target")
TIMEOUT_SEC = 3.0
# 64x48 合成色块（渐变），没有人像；bbox 随便框中间一块，给 --crop 用。
TEST_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdB"
    "RkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09P"
    "T09PT09PT09PT09PT0//wAARCAAwAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA"
    "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
    "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE"
    "xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDzpYfapVh9quLD7VMsPtSp1SYzKaw+1SrD"
    "7VcWH2qVYfau+nVNozKaw+1SrD7VcWH2qVYfau+nVN4zKaw+1SrD7VdWH2qVYfau+nVNozMxYfapVh9quLD7VKsPtX5/Tqnh"
    "RmU1h9qlWH2q6sPtUqw+1d9OqbRmU1h9qlWH2q4sPtUqw+1ehTqm8ZlNYfapVh9quLD7VMsPtXfTqm0ZmYsPtUqw+1XFh9ql"
    "WH2r8/p1TwozKaw+1SrD7VcWH2qZYfau+nVNozKSw+1TLD7VcWH2qVYfau+nVN4zKaw+1SrD7VcWH2qVYfavQp1TaMz/2Q=="
)
TEST_SIZE = (64, 48)
TEST_BBOX = [16.0, 8.0, 48.0, 40.0]

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def resolve_target(text: str | None) -> str:
    """命令行的地址，或目标文件里的；格式不对就以退出码 2 结束。"""
    if text:
        try:
            return normalize_target(text)
        except ValueError as exc:
            print(f"地址写错了：{exc}（要 <ip>[:端口]，例如 192.168.3.24 或 192.168.3.24:8765）", file=sys.stderr)
            sys.exit(2)
    target, problem = read_target(TARGET_FILE)
    if not target:
        print(f"没有目标：{problem}", file=sys.stderr)
        sys.exit(2)
    return target


def source_address(target: str) -> str:
    """发往 target 时本机用哪个地址（UDP connect 只查路由表，不发包）。"""
    parts = urllib.parse.urlsplit(target)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.connect((parts.hostname, parts.port or 80))
            return udp.getsockname()[0]
    except OSError as exc:
        return f"无路由（{exc}）"


def health(target: str) -> int:
    print(f"目标 {target}")
    print(f"本机发出地址 {source_address(target)}")
    try:
        with _OPENER.open(target + "/health", timeout=TIMEOUT_SEC) as response:
            answer = json.loads(response.read(4096))
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f"/health 没有回应：{exc}")
        return 1
    if not isinstance(answer, dict) or answer.get("ok") is not True:
        print(f"/health 回了，但不是接收端：{answer}")
        return 1
    mode = "只记账本（--discard）" if answer.get("discard") else "存照片"
    print(f"/health 正常：{answer.get('service')}，{mode}，本次已收 {answer.get('received')} 张")
    return 0


def send(target: str, count: int) -> int:
    failed = 0
    for index in range(count):
        round_id = f"sendtest-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}-{index}"
        meta = {
            "schema": "x2_mirror_photo/1",
            "round_id": round_id,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "quality": "test",
            "source": "send_test",
            "label": None,
            "bbox": TEST_BBOX,
            "width": TEST_SIZE[0],
            "height": TEST_SIZE[1],
            "bytes": len(TEST_JPEG),
            "sha256": hashlib.sha256(TEST_JPEG).hexdigest(),
        }
        started = time.monotonic()
        try:
            post_photo(target, TEST_JPEG, meta, TIMEOUT_SEC)
        except Refused as exc:
            failed += 1
            print(f"{round_id}: 对方回了但没确认（{exc}）")
            continue
        except (OSError, http.client.HTTPException) as exc:
            failed += 1
            print(f"{round_id}: 没有回应（{exc}）")
            continue
        print(f"{round_id}: 已确认（sha256 对得上），{(time.monotonic() - started) * 1000:.0f} ms")
    print(f"{count - failed}/{count} 张已确认")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Send synthetic photos to the upper PC like the photo node does.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("health", "send"):
        command = sub.add_parser(name)
        command.add_argument("target", nargs="?", help=f"<ip>[:port]; default: the first line of {TARGET_FILE}")
        if name == "send":
            command.add_argument("-n", "--count", type=int, default=3)
    args = parser.parse_args(argv)
    target = resolve_target(args.target)
    if args.command == "health":
        return health(target)
    return send(target, max(1, args.count))


if __name__ == "__main__":
    sys.exit(main())
