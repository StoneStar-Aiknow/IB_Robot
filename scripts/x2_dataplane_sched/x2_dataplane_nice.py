#!/usr/bin/env python3
"""Give the running aimdk_dataplane a better nice (and optionally CPUs), and undo it.

Off unless asked: without X2_DATAPLANE_NICE=1, ``hook`` only stops an earlier
watcher and puts back anything it changed. Only process attributes of the
dataplane's own threads change (setpriority, sched_setaffinity); no kernel,
sysctl or network setting. The launch line stays as it is: the start script
adds one line before it.

    x2_dataplane_nice.py status              the dataplane's threads: nice, CPUs; the watcher
    x2_dataplane_nice.py apply [--nice N] [--cpus 0-3,6]
                                             once, every thread; the originals are saved first
    x2_dataplane_nice.py restore             stop the watcher, put every saved thread back
    x2_dataplane_nice.py watch [...]         apply now and to every new thread or new dataplane,
                                             until the dataplane is gone for --linger seconds
    x2_dataplane_nice.py hook                for start scripts: X2_DATAPLANE_NICE=1 starts the
                                             watcher in the background; otherwise as restore
    x2_dataplane_nice.py apply-when-up [--nice N] [--wait S] [--settle S] [--result FILE]
                                             for start scripts, after the launch: wait for the
                                             dataplane, apply twice, read every thread back;
                                             one line "ok ..." or "FAIL ..." to FILE

Every thread gets the same nice: the intake threads and the command tick share
one lock, so lifting only the tick thread could leave it waiting on a starved
intake thread. Linux nice is per thread; ``renice -p PID`` alone moves only the
main thread. The status topic's tick_timing.tick_nice reads the command
thread's nice back from inside the process.

Environment: X2_DATAPLANE_NICE (hook: 1 = on), X2_DATAPLANE_NICE_VALUE (default -5),
X2_DATAPLANE_CPUS (default: leave the CPUs), X2_DATAPLANE_NICE_LOG (the watcher's log),
X2_DATAPLANE_NICE_STATE (where the originals are saved), X2_DATAPLANE_EXE (the
dataplane's argv[0] suffix; tests only).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EXE = os.environ.get("X2_DATAPLANE_EXE", "/lib/aimdk_dataplane/aimdk_dataplane")
DEFAULT_NICE = int(os.environ.get("X2_DATAPLANE_NICE_VALUE", "-5"))
DEFAULT_CPUS = os.environ.get("X2_DATAPLANE_CPUS", "")
STATE = Path(
    os.environ.get(
        "X2_DATAPLANE_NICE_STATE",
        "/run/x2-dataplane-nice" if os.geteuid() == 0 else f"/tmp/x2-dataplane-nice-{os.getuid()}",
    )
)
LOG = Path(os.environ.get("X2_DATAPLANE_NICE_LOG", str(STATE / "watch.log")))
SCRIPT = Path(__file__).resolve()


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S ") + text, flush=True)


def parse_cpus(text: str) -> set[int]:
    cpus = set()
    for part in filter(None, text.split(",")):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    if not cpus:
        raise ValueError(f"no CPUs in {text!r}")
    return cpus


def cpus_text(cpus) -> str:
    cpus, out = sorted(cpus), []
    start = prev = None
    for cpu in cpus + [None]:
        if start is not None and cpu == prev + 1:
            prev = cpu
            continue
        if start is not None:
            out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    return ",".join(out)


def start_time(pid: int) -> str | None:
    """The process's start time in clock ticks: tells a reused pid apart."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def is_dataplane(pid: int) -> bool:
    """argv[0] is the dataplane executable (not a shell naming it)."""
    try:
        argv0 = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
    except OSError:
        return False
    return argv0.endswith(EXE)


def dataplanes() -> list[int]:
    return sorted(int(e.name) for e in Path("/proc").iterdir() if e.name.isdigit() and is_dataplane(int(e.name)))


def threads(pid: int) -> list[int]:
    try:
        return sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return []


def thread_name(tid: int) -> str:
    try:
        return Path(f"/proc/{tid}/comm").read_text().strip()
    except OSError:
        return "?"


def read_thread(tid: int) -> dict | None:
    try:
        return {"nice": os.getpriority(os.PRIO_PROCESS, tid), "cpus": sorted(os.sched_getaffinity(tid))}
    except OSError:
        return None  # it ended


def state_file(pid: int, started: str) -> Path:
    return STATE / f"{pid}-{started}.json"


def one_dataplane(pid: int | None) -> int | None:
    if pid is not None:
        if not is_dataplane(pid):
            say(f"停：进程 {pid} 不是 dataplane（argv[0] 不以 {EXE} 结尾）")
            return None
        return pid
    found = dataplanes()
    if len(found) > 1:
        say(f"停：找到 {len(found)} 个 dataplane {found}，用 --pid 指定一个")
        return None
    return found[0] if found else None


def apply(pid: int, nice: int, cpus: set[int] | None, started: str | None = None) -> tuple[int, int]:
    """Set every thread of ``pid``; save each thread's original first. Returns (changed, failed).

    ``started``: the start time ``pid`` had when it was chosen; nothing changes if
    it is not that process any more, or not a dataplane.
    """
    started = started or start_time(pid)
    if started is None or start_time(pid) != started or not is_dataplane(pid):
        return 0, 0
    STATE.mkdir(parents=True, exist_ok=True)
    path = state_file(pid, started)
    if path.exists():
        saved, first = json.loads(path.read_text()), False
    else:
        main = read_thread(pid)
        if main is None:
            return 0, 0
        saved, first = {"pid": pid, "threads": {}, "default": main}, True
    changed = failed = 0
    for tid in threads(pid):
        before = read_thread(tid)
        if before is None:
            continue
        if str(tid) not in saved["threads"]:
            # Only threads there before the first change have their own original.
            # One born later inherited the change from the thread that made it:
            # its original is the main thread's.
            saved["threads"][str(tid)] = before if first else saved["default"]
            path.write_text(json.dumps(saved))
        wanted = {"nice": nice, "cpus": sorted(cpus) if cpus is not None else before["cpus"]}
        if before == wanted:
            continue
        ok = True
        for name, put in (("nice", lambda v: os.setpriority(os.PRIO_PROCESS, tid, v)),
                          ("cpus", lambda v: os.sched_setaffinity(tid, v))):
            if before[name] == wanted[name]:
                continue
            if start_time(pid) != started or not Path(f"/proc/{pid}/task/{tid}").exists():
                break  # it ended; the tid is not this dataplane's any more
            try:
                put(wanted[name])
            except OSError as error:
                if error.errno == errno.ESRCH:
                    break
                ok = False
                say(f"线程 {tid}（{thread_name(tid)}）的 {name} 改不了：{error}")
        changed += ok
        failed += not ok
    return changed, failed


def restore_one(path: Path) -> tuple[int, int]:
    saved = json.loads(path.read_text())
    pid = saved["pid"]
    started = path.stem.split("-", 1)[1]
    restored = failed = 0
    if start_time(pid) == started:
        for tid in threads(pid):
            original = saved["threads"].get(str(tid), saved.get("default"))
            now = read_thread(tid)
            if original is None or now is None or now == original:
                continue
            # Each attribute on its own: the CPUs come back even where the nice cannot.
            done = True
            for name, put in (("cpus", lambda v: os.sched_setaffinity(tid, v)),
                              ("nice", lambda v: os.setpriority(os.PRIO_PROCESS, tid, v))):
                if now[name] == original[name]:
                    continue
                try:
                    put(original[name])
                except OSError as error:
                    if error.errno == errno.ESRCH:
                        break
                    done = False
                    say(f"线程 {tid}（{thread_name(tid)}）的 {name} 还原不了：{error}")
            restored += done
            failed += not done
    if failed == 0:
        path.unlink()
    return restored, failed


def watcher_pid() -> int | None:
    """The running watcher, from the pid it wrote under its lock; checked to be this script."""
    try:
        pid = int((STATE / "watch.pid").read_text())
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):
        return None
    if any(a.endswith(SCRIPT.name.encode()) for a in argv) and b"watch" in argv:
        return pid
    return None


def stop_watcher() -> bool:
    pid = watcher_pid()
    if pid is None:
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if watcher_pid() is None or not Path(f"/proc/{pid}").exists():
            break
        time.sleep(0.1)
    say(f"已停监视进程 {pid}")
    return True


def cmd_status(args) -> int:
    pids = [args.pid] if args.pid else dataplanes()
    if not pids:
        say("没有在跑的 dataplane")
    for pid in pids:
        per = {}
        for tid in threads(pid):
            now = read_thread(tid)
            if now:
                key = f"nice={now['nice']} cpus={cpus_text(now['cpus'])}"
                per.setdefault(key, []).append(tid)
        started = start_time(pid)
        saved = started is not None and state_file(pid, started).exists()
        say(f"dataplane {pid}：{sum(map(len, per.values()))} 个线程，{'有' if saved else '没有'}保存的原值")
        for key, tids in sorted(per.items()):
            say(f"  {key}: {len(tids)} 个线程")
    watcher = watcher_pid()
    say(f"监视进程：{watcher if watcher else '没有'}")
    return 0


def cmd_apply(args) -> int:
    pid = one_dataplane(args.pid)
    started = start_time(pid) if pid is not None else None
    if started is None:
        say("没有可改的 dataplane")
        return 1
    changed, failed = apply(pid, args.nice, args.cpus, started)
    say(f"dataplane {pid}：改了 {changed} 个线程，失败 {failed}（nice={args.nice}"
        f"{' cpus=' + cpus_text(args.cpus) if args.cpus else ''}）")
    return 1 if failed else 0


def cmd_apply_when_up(args) -> int:
    """The verdict is read back from the threads, not taken from apply's counts.

    A start script runs this in the background and goes on whatever it says.
    """
    wanted = f"nice={args.nice}{' cpus=' + cpus_text(args.cpus) if args.cpus else ''}"

    def verdict(ok: bool, text: str) -> int:
        line = f"{'ok' if ok else 'FAIL'} {wanted}：{text}"
        say(line)
        if args.result:
            Path(args.result).write_text(line + "\n")
        return 0 if ok else 1

    end = time.monotonic() + args.wait
    found = dataplanes()
    while not found and time.monotonic() < end:
        time.sleep(args.interval)
        found = dataplanes()
    if not found:
        return verdict(False, f"{args.wait:.0f} s 内没等到 dataplane，不改")
    if len(found) > 1:
        return verdict(False, f"找到 {len(found)} 个 dataplane {found}，不改")
    pid = found[0]
    started = start_time(pid)
    say(f"dataplane {pid} 已起，每隔 {args.settle:g} s 改两遍")
    # Twice: a thread started during the first pass by a thread not yet changed
    # keeps its own nice; threads started after both inherit the change.
    for _ in range(2):
        time.sleep(args.settle)
        if started is None or start_time(pid) != started:
            return verdict(False, f"dataplane {pid} 已退出，没改完")
        changed, failed = apply(pid, args.nice, args.cpus, started)
        say(f"dataplane {pid}：改了 {changed} 个线程，失败 {failed}")
    seen = off = 0
    for tid in threads(pid):
        now = read_thread(tid)
        if now is None:
            continue
        seen += 1
        off += now["nice"] != args.nice or (args.cpus is not None and set(now["cpus"]) != args.cpus)
    if start_time(pid) != started or seen == 0:
        return verdict(False, f"dataplane {pid} 已退出，没改完")
    if off:
        return verdict(False, f"dataplane {pid} 有 {off}/{seen} 个线程没改成")
    return verdict(True, f"dataplane {pid} 全部 {seen} 个线程")


def cmd_restore(args) -> int:
    stop_watcher()
    failed_any = False
    paths = sorted(STATE.glob("*-*.json")) if STATE.exists() else []
    for path in paths:
        restored, failed = restore_one(path)
        failed_any |= failed > 0
        say(f"{path.name}：还原 {restored} 个线程，失败 {failed}")
    if not paths:
        say("没有保存的原值，不用还原")
    return 1 if failed_any else 0


def cmd_watch(args) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    lock = open(STATE / "watch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        say("已有监视进程在跑，退出")
        return 0
    (STATE / "watch.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    say(f"监视开始：nice={args.nice}{' cpus=' + cpus_text(args.cpus) if args.cpus else ''}"
        f"，dataplane 消失 {args.linger:.0f} s 后退出")
    seen: set[tuple[int, str]] = set()
    gone_since = None
    try:
        while True:
            found = dataplanes()
            if len(found) > 1:
                say(f"找到 {len(found)} 个 dataplane {found}，这一轮不改")
            elif found:
                pid = found[0]
                key = (pid, start_time(pid) or "")
                changed, failed = apply(pid, args.nice, args.cpus, key[1] or None)
                if key not in seen or changed or failed:
                    say(f"dataplane {pid}：改了 {changed} 个线程，失败 {failed}")
                    seen.add(key)
                gone_since = None
            if not found:
                gone_since = gone_since if gone_since is not None else time.monotonic()
                if (seen or not args.wait_first) and time.monotonic() - gone_since >= args.linger:
                    say("dataplane 已退出，监视结束" if seen else "一直没有 dataplane，监视结束")
                    return 0
            time.sleep(args.interval)
    finally:
        try:
            if (STATE / "watch.pid").read_text() == str(os.getpid()):
                (STATE / "watch.pid").unlink()
        except OSError:
            pass


def cmd_hook(args) -> int:
    # An earlier run's watcher must not touch this run's dataplane, off or with other values.
    stop_watcher()
    if os.environ.get("X2_DATAPLANE_NICE", "0") != "1":
        say("x2_dataplane_nice: 关（X2_DATAPLANE_NICE 不是 1），不改 dataplane")
        # A dataplane still running from an earlier run gets its originals back. Never
        # fails the start script: a failure is printed.
        cmd_restore(args)
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(SCRIPT), "watch", "--nice", str(args.nice), "--wait-first"]
    if args.cpus:
        command += ["--cpus", cpus_text(args.cpus)]
    with open(LOG, "a") as log:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    say(f"x2_dataplane_nice: 开，nice={args.nice}{' cpus=' + cpus_text(args.cpus) if args.cpus else ''}，"
        f"日志 {LOG}；还原：{SCRIPT} restore")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "apply", "restore", "watch", "hook", "apply-when-up"])
    parser.add_argument("--pid", type=int, help="this dataplane (default: the only one running)")
    parser.add_argument("--nice", type=int, default=DEFAULT_NICE)
    parser.add_argument("--cpus", type=parse_cpus, default=parse_cpus(DEFAULT_CPUS) if DEFAULT_CPUS else None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--linger", type=float, default=30.0)
    parser.add_argument("--wait-first", action="store_true", help="watch: wait for the first dataplane however long")
    parser.add_argument("--wait", type=float, default=60.0, help="apply-when-up: for the dataplane, at most")
    parser.add_argument("--settle", type=float, default=3.0, help="apply-when-up: before each pass")
    parser.add_argument("--result", help="apply-when-up: write the verdict line here")
    args = parser.parse_args(argv)
    if not -20 <= args.nice <= 19:
        parser.error("--nice is -20..19")
    return {"status": cmd_status, "apply": cmd_apply, "restore": cmd_restore, "watch": cmd_watch,
            "hook": cmd_hook, "apply-when-up": cmd_apply_when_up}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
