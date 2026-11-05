"""Inspect and maintain the pose memory.

    pose_memory --path <db> list [--all]
    pose_memory --path <db> show <pose_id>
    pose_memory --path <db> audit [--limit N]
    pose_memory --path <db> export <file.yaml>
    pose_memory --path <db> pin|unpin|delete <pose_id>
    pose_memory --path <db> ban <pose_id> <phrase>
    pose_memory --path <db> import <file.yaml>

Reading works any time. Writing needs the memory's lock, so stop the pose
composer first; imported poses are stored stale and re-solved against the
robot's joint travel when the composer next starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from pose_composer.memory import MemoryLocked, PoseMemory

WRITES = {"pin", "unpin", "delete", "ban", "import"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").add_argument("--all", action="store_true")
    sub.add_parser("show").add_argument("pose_id")
    sub.add_parser("audit").add_argument("--limit", type=int, default=50)
    sub.add_parser("export").add_argument("file")
    for name in ("pin", "unpin", "delete"):
        sub.add_parser(name).add_argument("pose_id")
    ban = sub.add_parser("ban")
    ban.add_argument("pose_id")
    ban.add_argument("phrase")
    sub.add_parser("import").add_argument("file")
    args = parser.parse_args(argv)
    try:
        memory = PoseMemory(args.path, fingerprint="", writer=args.command in WRITES)
    except MemoryLocked as exc:
        print(f"{exc}: stop the pose composer first", file=sys.stderr)
        return 2
    try:
        if args.command == "list":
            for record in memory.list(include_retired=args.all):
                phrases = [a["phrase"] for a in memory.aliases(record.pose_id) if a["state"] == "active"]
                print(f"{record.pose_id:24s} {record.state:9s} ok={record.successes} good={record.good} "
                      f"bad={record.bad} v{record.version} {record.source:9s} {'/'.join(phrases)}")  # fmt: skip
        elif args.command == "show":
            record = memory.get(args.pose_id)
            print(json.dumps({"record": record.__dict__ if record else None, "aliases": memory.aliases(args.pose_id)},
                             ensure_ascii=False, indent=1, default=str))  # fmt: skip
        elif args.command == "audit":
            for row in memory.audit_log(args.limit):
                print(json.dumps(row, ensure_ascii=False))
        elif args.command == "export":
            with open(args.file, "w", encoding="utf-8") as fh:
                yaml.safe_dump({"version": "export", "poses": memory.export()}, fh, allow_unicode=True, sort_keys=False)
        elif args.command in ("pin", "unpin"):
            memory.set_pinned(args.pose_id, args.command == "pin")
        elif args.command == "delete":
            memory.delete(args.pose_id)
        elif args.command == "ban":
            memory.ban(args.phrase, args.pose_id)
        elif args.command == "import":
            entries = (yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}).get("poses") or []
            pinned = [e for e in entries if e.get("pinned")]
            plain = [e for e in entries if not e.get("pinned")]
            count = memory.add_described(pinned, source="import", pinned=True)
            count += memory.add_described(plain, source="import", pinned=False)
            print(f"imported {count} poses; they are re-solved when the pose composer starts")
    finally:
        memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
