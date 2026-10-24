"""看一趟的选人读数：从 <趟目录>/rec/ 里列出选人核心和 session 的状态时间线，以及 GRACE/放手原因、目标位移分布。
用法（duck 上，不用 source 环境）：python3 /root/x2-actor-run/run_read.py $(cat /root/x2-actor-run/LATEST)
只读 start.sh（zone 模式）记下的 rec/actor_candidates.jsonl 和 rec/session_state.jsonl。
"""

import collections
import contextlib
import json
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def lines(name):
    with open(run / "rec" / name) as f:
        for line in f:
            with contextlib.suppress(ValueError):
                yield json.loads(line)


print("== 选人核心（只列状态/actor_id/原因变化，以及有 events/warnings 的刷新）")
prev = None
counts = collections.Counter()
grace_reasons = collections.Counter()
release = []
persons = collections.Counter()
dx, dy, hr = [], [], []
n = 0
unknown = 0
first = last = None
for rec in lines("actor_candidates.jsonl"):
    m = rec["msg"]
    if m.get("kind") == "unknown":
        unknown += 1
        continue
    n += 1
    first = first or m["stamp"]
    last = m["stamp"]
    persons[m["persons_count"]] += 1
    counts[m["state"]] += 1
    reason = m["decision"]["reason"]
    if m["state"] == "GRACE":
        grace_reasons[reason] += 1
    key = (m["state"], m["actor_id"], reason)
    if key != prev or m["events"] or m["warnings"]:
        tgt = next((t for t in m["tracks"] if t["track"] == m["target_track"]), None)
        extra = ""
        if tgt:
            extra = f" uv={tgt.get('uv')} in_zone={tgt.get('in_zone')} occl={tgt.get('occluded_by')}"
        print(
            f"{hms(m['stamp'])} {m['state']:<9} actor={m['actor_id']:>2} bad={m['bad']} n={m['persons_count']} "
            f"reason={reason} ev={m['events']} warn={m['warnings']} handoff={m['handoff_suspected']}{extra}"
        )
        prev = key
    if m["state"] in ("LOCKED", "GRACE"):
        for t in m["tracks"]:
            if t["track"] == m["target_track"] and t.get("dx") is not None:
                dx.append(abs(t["dx"]))
                dy.append(abs(t["dy"]))
                hr.append(t["h_ratio"])
print(
    f"刷新 {n} 次（{hms(first)}–{hms(last)}，平均间隔 {(last - first) / max(n - 1, 1):.2f} s），unknown 记录 {unknown}"
)
print("各状态刷新数", dict(counts))
print("GRACE 原因", dict(grace_reasons))
print("人数分布", dict(sorted(persons.items())))


def q(v, ps=(0.5, 0.9, 0.99, 1.0)):
    v = sorted(v)
    return {p: round(v[min(len(v) - 1, int(p * len(v)))], 3) for p in ps} if v else {}


print(
    f"目标逐刷新位移（LOCKED/GRACE，{len(dx)} 个）|dx| {q(dx)} |dy| {q(dy)} 框高比 min {min(hr, default=None)} {q(hr)}"
)

print("== session 状态变化")
prev = None
for rec in lines("session_state.jsonl"):
    j = rec["json"]
    key = (j["state"], j.get("wait_reason"), j.get("cancel_reason"), j.get("phase"))
    if key != prev:
        print(
            f"{hms(j['stamp'])} {j['state']:<10} wait={j.get('wait_reason')} cancel={j.get('cancel_reason')} "
            f"phase={j.get('phase')} last={j.get('last_transition')}"
        )
        prev = key
