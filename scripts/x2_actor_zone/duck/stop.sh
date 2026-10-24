#!/bin/bash
# 停掉 /root/x2-actor-run/start.sh 起的整套（照 /root/x2-dexhand-run/stop.sh，只换目录和环境脚本）。顺序是固定的：
#   1. 整套还活着时，先对模仿流程发 stop（同长捂）：在模仿中会先收尾，
#      在 WAIT 里（比如出过故障）会直接请求 idle，厂商从 URS 回到 STAND_DEFAULT。
#      手臂会回到厂商站立姿势：周围别站人，别碰头。
#   2. 确认厂商回到 STAND_DEFAULT 后，再对整组进程发 SIGINT（= Ctrl-C），等它收尾。
#   3. 停掉后台记录（rt_watch、fb_watch、带手时还有 hand_watch）。
# 顺序反了（先杀进程），runtime 先退，就没人能请求 idle，只能重启机器人。
RUN=/root/x2-actor-run
STATE=$RUN/run
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || { echo "停：另一个 start/stop 正在执行"; exit 1; }
ENV="env -i HOME=/root TERM=xterm bash --noprofile --norc -c"

alive() { [ -s "$STATE/$1.pid" ] && kill -0 "$(cat "$STATE/$1.pid")" 2>/dev/null; }
vendor() { $ENV 'source /root/x2-actor-run/env_actor.sh >/dev/null 2>&1; python3 /root/x2-retarget-run/mc_state.py' 2>&1; }
stop_group() {  # name, seconds to wait after SIGINT
    alive "$1" || return 0
    local pgid; pgid=$(cat "$STATE/$1.pid")
    kill -INT -- -"$pgid" 2>/dev/null
    for _ in $(seq "$2"); do pgrep -g "$pgid" >/dev/null || break; sleep 1; done
    if pgrep -g "$pgid" >/dev/null; then
        echo "!! $1 在 $2 秒内没退完，发 SIGTERM"; kill -TERM -- -"$pgid" 2>/dev/null; sleep 5
    fi
    if pgrep -g "$pgid" >/dev/null; then
        echo "!! $1 还有残留，发 SIGKILL"; kill -KILL -- -"$pgid" 2>/dev/null; sleep 1
    fi
    rm -f "$STATE/$1.pid"
    echo "== $1 已停"
}

LOG=$(cat "$STATE/log_dir" 2>/dev/null)
if alive launch; then
    echo "== 1/3 让机器人退出 URS（发 stop，同长捂）"
    v=$(vendor); echo "   之前：$v"
    $ENV 'source /root/x2-actor-run/env_actor.sh >/dev/null 2>&1; timeout 15 ros2 service call /mirror_session/set_session std_srvs/srv/SetBool "{data: false}"' 2>&1 | grep -E "success|message" | sed "s/^/   /"
    ok=0
    for _ in 1 2 3 4 5; do
        v=$(vendor)
        if echo "$v" | grep -q "^action STAND_DEFAULT"; then ok=1; break; fi
        sleep 1
    done
    echo "   之后：$v"
    if [ $ok = 1 ]; then
        echo "== 厂商已回到 STAND_DEFAULT"
    else
        echo "!! 厂商没回到 STAND_DEFAULT。整套先不停，保留现场，告诉 Claude。"
        echo "   确认要强行停（之后多半要重启机器人）：bash $RUN/stop.sh --force"
        [ "$1" = --force ] || exit 1
    fi
    echo "== 2/3 停整套（SIGINT 整组，等它收尾）"
    stop_group launch 30
else
    echo "== 整套没在跑"
fi
echo "== 3/3 停记录"
for w in $(cat "$STATE/watchers" 2>/dev/null || echo rt_watch fb_watch hand_watch actor_state actor_cand); do stop_group "$w" 5; done
echo "== 最后厂商状态：$(vendor)"
if [ -n "$LOG" ]; then echo "== 本次日志 $LOG"; fi
