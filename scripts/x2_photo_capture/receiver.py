#!/usr/bin/env python3
"""X2 照镜子拍照：上位机接收端。只用 Python >= 3.8 标准库，Ubuntu 和 Windows 同一份代码。

    python receiver.py --discard                    开发阶段一律这样跑：只记账本，不存照片字节
    python receiver.py --inbox D:/x2_photos         现场：照片存进 inbox（<轮次id>.jpg + <轮次id>.json）
        [--crop]                                    另存 <轮次id>_crop.jpg（按人框外扩一圈，需 pip install pillow）
        [--on-photo "python aigc.py {path}"]        原图落盘后在后台拉起一条命令，不等它跑完
        [--allow 192.168.3.10]                      只收这个来源地址（可写多次），其它来源 403
        [--port 8765] [--max-mb 5]

协议（与 .5 上 manipulation_execution.mirror_photo.uploader 一致）：

* ``POST /x2/photo``：请求体是相机给的原始 JPEG，元数据是请求头 ``X-X2-Photo-Meta`` 里的一份 JSON
  （round_id、bbox ``[x1, y1, x2, y2]`` 原图像素、width/height、quality、sha256 ...）。
  存好以后回 200 ``{"ok": true, "sha256": ...}``；.5 只有看到 sha256 对得上才删本地那张，否则一直重发。
* ``GET /health``：连通性检查。没有别的路径：不能下载、不能列目录。

同一轮次、同样字节的重发只回执、不多存一张（按账本 ledger.jsonl 去重，重启后也认）。
同一轮次 id 来了不同的字节，另存为 ``<轮次id>_<sha 前 8 位>``，一张也不丢。
先写 .json 再写 .jpg（都是临时文件 + fsync + 换名），盯着 inbox 里 *.jpg 的工作流看到图时元数据一定已经在。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENDPOINT = "/x2/photo"
HEALTH = "/health"
META_HEADER = "X-X2-Photo-Meta"
DEFAULT_PORT = 8765
SERVICE = "x2-photo-receiver"
LEDGER = "ledger.jsonl"
# 轮次 id 直接做文件名：只许字母数字和 . _ -，不许以点开头（挡住 ../ 和隐藏文件）。
ROUND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
JPEG_START = b"\xff\xd8\xff"
MAX_META_CHARS = 16 * 1024
CROP_MARGIN = 0.15


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def _write_atomic(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def crop_box(bbox, width: int, height: int, margin: float = CROP_MARGIN):
    """人框外扩 ``margin``（框宽高的比例）再裁到图内；``(left, top, right, bottom)`` 或 None（框不可用）。"""
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy)))
    right, bottom = min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def command_for(template: str, values: dict) -> str:
    """``--on-photo`` 的命令：把 {path} {meta} {crop} {round_id} 换成加好引号的值。"""
    if os.name == "nt":
        quote = lambda value: subprocess.list2cmdline([value])  # noqa: E731
    else:
        quote = shlex.quote
    command = template
    for key, value in values.items():
        command = command.replace("{" + key + "}", quote(value))
    return command


class Rejected(Exception):
    def __init__(self, code: int, why: str):
        super().__init__(why)
        self.code = code
        self.why = why


class Inbox:
    """照片落到哪里。一把锁：重发和它的第一份同时到，也只存一张。"""

    def __init__(self, path: Path, *, discard: bool, crop: bool = False, on_photo: str = ""):
        self.path = path
        self.discard = discard
        self.crop = crop and not discard
        self.on_photo = "" if discard else on_photo
        self.path.mkdir(parents=True, exist_ok=True)
        self.received = 0
        self.duplicates = 0
        self._lock = threading.Lock()
        self._seen = self._read_ledger()
        self._told_no_pillow = False

    def _read_ledger(self) -> dict:
        seen: dict = {}
        try:
            lines = (self.path / LEDGER).read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return seen
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # 一行写坏了（断电）：只少认一次去重，照片本身不受影响
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("round_id"), str)
                and isinstance(entry.get("sha256"), str)
            ):
                seen.setdefault(entry["round_id"], set()).add(entry["sha256"])
        return seen

    def known_rounds(self) -> int:
        with self._lock:
            return len(self._seen)

    def receive(self, body: bytes, meta_text: str, peer: str) -> dict:
        try:
            meta = json.loads(meta_text) if meta_text else None
        except ValueError:
            raise Rejected(400, f"{META_HEADER} is not JSON") from None
        if not isinstance(meta, dict):
            raise Rejected(400, f"no {META_HEADER} JSON object")
        round_id = meta.get("round_id")
        if not isinstance(round_id, str) or not ROUND_ID.match(round_id):
            raise Rejected(400, "round_id missing or not a plain name")
        if not body.startswith(JPEG_START):
            raise Rejected(415, "not a JPEG")
        sha = hashlib.sha256(body).hexdigest()
        if meta.get("sha256") not in (None, sha):
            raise Rejected(400, "the body does not match the metadata's sha256 (cut short?)")
        with self._lock:
            known = self._seen.get(round_id, set())
            duplicate = sha in known
            stem = round_id if not known else f"{round_id}_{sha[:8]}"
            stored = None
            if not duplicate and not self.discard:
                stored = self._store(stem, body, meta, peer, sha)
            self._append_ledger(meta, round_id, sha, len(body), peer, duplicate, stored)
            # Only now: a photo the disk refused is not taken for a duplicate when it comes again.
            self._seen.setdefault(round_id, set()).add(sha)
            if duplicate:
                self.duplicates += 1
            else:
                self.received += 1
        self._say(meta, round_id, len(body), peer, duplicate, stored)
        if stored is not None and (self.crop or self.on_photo):
            threading.Thread(target=self._after, args=(stored, body, meta), daemon=True).start()
        return {"ok": True, "sha256": sha, "round_id": round_id, "duplicate": duplicate}

    def _store(self, stem: str, body: bytes, meta: dict, peer: str, sha: str) -> Path:
        record = dict(meta)
        record.update(sha256=sha, received_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), peer=peer)
        _write_atomic(self.path / f"{stem}.json", json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8"))
        image = self.path / f"{stem}.jpg"
        _write_atomic(image, body)
        return image

    def _append_ledger(self, meta, round_id, sha, size, peer, duplicate, stored) -> None:
        entry = {
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "round_id": round_id,
            "sha256": sha,
            "bytes": size,
            "peer": peer,
            "duplicate": duplicate,
            "stored": stored.name if stored is not None else None,
        }
        for key in ("quality", "source", "label", "bbox", "width", "height", "end_reason"):
            entry[key] = meta.get(key)
        with open(self.path / LEDGER, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _say(self, meta, round_id, size, peer, duplicate, stored) -> None:
        if duplicate:
            log(f"photo {round_id} from {peer} again (a resend), confirmed, not stored twice")
            return
        where = stored.name if stored is not None else "discarded (--discard)"
        log(
            f"photo {round_id} from {peer}: {size} bytes {meta.get('width')}x{meta.get('height')}, "
            f"{meta.get('quality')} {meta.get('source') or '-'}/{meta.get('label') or '-'}, "
            f"bbox {meta.get('bbox')} -> {where}"
        )

    # -- after the receipt (a thread per photo; nothing here can cost the photo) -----------------

    def _after(self, image: Path, body: bytes, meta: dict) -> None:
        crop = ""
        if self.crop:
            try:
                crop = self._crop(image, body, meta)
            except Exception as exc:  # noqa: BLE001 - the original is in, the crop is optional
                log(f"photo {image.stem}: no crop ({type(exc).__name__}: {exc}); the original is kept")
        if self.on_photo:
            values = {"path": str(image), "meta": str(image.with_suffix(".json")), "crop": crop, "round_id": image.stem}
            command = command_for(self.on_photo, values)
            try:
                process = subprocess.Popen(command, shell=True)
            except OSError as exc:
                log(f"photo {image.stem}: --on-photo did not start ({exc}): {command}")
                return
            code = process.wait()
            if code != 0:
                log(f"photo {image.stem}: --on-photo exited {code}: {command}")

    def _crop(self, image: Path, body: bytes, meta: dict) -> str:
        try:
            from PIL import Image
        except ImportError:
            if not self._told_no_pillow:
                self._told_no_pillow = True
                log("--crop needs Pillow (pip install pillow); originals only until then")
            return ""
        with Image.open(io.BytesIO(body)) as picture:
            box = crop_box(meta.get("bbox"), picture.width, picture.height)
            if box is None:
                return ""
            out = io.BytesIO()
            picture.crop(box).save(out, format="JPEG", quality=95)
        path = image.with_name(f"{image.stem}_crop.jpg")
        _write_atomic(path, out.getvalue())
        return str(path)


class Handler(BaseHTTPRequestHandler):
    server_version = "x2-photo-receiver/1"
    timeout = 15  # a sender that stops halfway does not hold a thread for ever

    def do_GET(self) -> None:
        if not self._allowed():
            return
        if self.path.split("?", 1)[0] != HEALTH:
            self._answer(404, {"ok": False, "error": "not found"})
            return
        inbox = self.server.inbox
        self._answer(
            200,
            {"ok": True, "service": SERVICE, "discard": inbox.discard, "received": inbox.received},
        )

    def do_POST(self) -> None:
        if not self._allowed():
            return
        if self.path != ENDPOINT:
            self._answer(404, {"ok": False, "error": "not found"})
            return
        length_text = self.headers.get("Content-Length")
        if length_text is None or not length_text.isdigit():
            self._answer(411, {"ok": False, "error": "Content-Length required"})
            return
        length = int(length_text)
        if length > self.server.max_bytes:
            self.close_connection = True
            self._answer(413, {"ok": False, "error": f"larger than {self.server.max_bytes} bytes"})
            log(f"refused {length} bytes from {self.client_address[0]}: larger than --max-mb")
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._answer(400, {"ok": False, "error": "body cut short"})
            return
        meta_text = self.headers.get(META_HEADER, "")
        if len(meta_text) > MAX_META_CHARS:
            self._answer(400, {"ok": False, "error": f"{META_HEADER} too long"})
            return
        try:
            answer = self.server.inbox.receive(body, meta_text, self.client_address[0])
        except Rejected as exc:
            log(f"refused a photo from {self.client_address[0]}: {exc.why}")
            self._answer(exc.code, {"ok": False, "error": exc.why})
            return
        except OSError as exc:
            # Not confirmed, so the board keeps it and sends it again.
            log(f"ERROR cannot store a photo from {self.client_address[0]}: {exc}")
            self._answer(500, {"ok": False, "error": f"cannot store: {exc}"})
            return
        self._answer(200, answer)

    def _allowed(self) -> bool:
        allow = self.server.allow
        peer = self.client_address[0]
        if not allow or peer in allow:
            return True
        if peer not in self.server.refused_peers:
            self.server.refused_peers.add(peer)
            log(f"refused {peer}: not in --allow {sorted(allow)}")
        self._answer(403, {"ok": False, "error": "not allowed"})
        return False

    def _answer(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args) -> None:  # noqa: A002 - BaseHTTPRequestHandler's name
        pass  # one line per photo from Inbox instead of one per request


class Receiver(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, inbox: Inbox, *, allow=(), max_mb: float = 5.0):
        super().__init__(address, Handler)
        self.inbox = inbox
        self.allow = frozenset(allow)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.refused_peers: set = set()


def local_addresses() -> list:
    """本机 IPv4 地址，给现场照着往 .5 上填。UDP connect 只查路由，不发包。"""
    found = []
    with contextlib.suppress(OSError):
        found += [info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    for probe in ("192.168.3.1", "10.42.0.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                udp.connect((probe, 9))
                found.append(udp.getsockname()[0])
        except OSError:
            pass
    return sorted({address for address in found if not address.startswith(("127.", "0."))})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X2 mirror photo receiver (the upper PC end).")
    parser.add_argument("--inbox", default="x2_photos", help="where photos and ledger.jsonl go (default ./x2_photos)")
    parser.add_argument("--discard", action="store_true", help="keep only the ledger, never the photo bytes")
    parser.add_argument("--crop", action="store_true", help="also save <round>_crop.jpg around the person box")
    parser.add_argument("--on-photo", default="", help='run this for each new photo, e.g. "python aigc.py {path}"')
    parser.add_argument("--allow", action="append", default=[], help="accept only this sender address (repeatable)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-mb", type=float, default=5.0, help="largest photo accepted (default 5)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inbox = Inbox(Path(args.inbox).resolve(), discard=args.discard, crop=args.crop, on_photo=args.on_photo)
    try:
        server = Receiver((args.host, args.port), inbox, allow=args.allow, max_mb=args.max_mb)
    except OSError as exc:
        print(f"cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    mode = "--discard: ledger only, no photo bytes" if args.discard else "photos kept"
    log(
        f"X2 photo receiver on port {args.port}, inbox {inbox.path} ({mode}); {inbox.known_rounds()} rounds in the ledger"
    )
    if args.discard and (args.crop or args.on_photo):
        log("--discard: --crop / --on-photo do nothing (there is no file)")
    if args.allow:
        log(f"accepting only {', '.join(args.allow)}")
    addresses = local_addresses()
    log("this PC: " + (", ".join(addresses) if addresses else "no IPv4 address found"))
    for address in addresses:
        log(f"  on the board: bash <tree>/scripts/x2_photo_capture/photo_target.sh {address}:{args.port}")
    log(f"Windows: allow inbound TCP {args.port} in the firewall. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log(f"stopped: {inbox.received} new photo(s), {inbox.duplicates} resend(s) this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
