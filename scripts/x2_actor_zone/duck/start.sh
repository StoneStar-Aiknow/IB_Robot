#!/bin/bash
# 起照镜子 + 灵巧手 + 站位区选人（手臂真动，不看图不说话），后台跑，不占终端。
#   bash /root/x2-actor-run/start.sh            带区域选人（默认），带调试记录
#   bash /root/x2-actor-run/start.sh nozone     不带区域：配置换成 dexhand_rec（= PR #5 灵巧手），其余一样
#   bash /root/x2-actor-run/start.sh nozone norec   不带区域、不做调试记录（配置用 dexhand）
#   停止：bash /root/x2-actor-run/stop.sh   （先让机器人退出 URS，再停进程）
# 照 /root/x2-dexhand-run/start.sh 改（那份不改，回退就用它）。ros2 launch 那一行与它相同，只换配置文件。
# 区域文件 $RUN/zone/actor_zone.yaml、参数文件 $RUN/zone/actor_tuning.yaml：改完重跑本脚本生效，不用构建。
# 带区域时启动前先校验两个文件，把校验输出和两个文件的副本、sha256 记进本趟日志目录。
# 带调试记录时：手部节点照旧把每帧的关键点和手的小裁图写到 dexhand 的 dump 目录（裁图可能拍到脸，
# 只留在板上，分析完删）；hand_rec 4 组 + （带区域时）actor_rec 2 组记到 日志目录/rec/。
# 另起 rt_watch / fb_watch（只读，用 /root/visual-talk-run 里的原件，不改）。
RUN=/root/x2-actor-run
STATE=$RUN/run
TREE=/root/x2-dexhand-wt; BRANCH=actor-select; ENVF=$RUN/env_actor.sh
ZONE=$RUN/zone/actor_zone.yaml; TUNING=$RUN/zone/actor_tuning.yaml
CFGDIR=$TREE/src/robot_config/config/robots
MODE=zone; REC=1
for a in "$@"; do
    case "$a" in
        zone|nozone) MODE=$a ;;
        norec) REC=0 ;;
        *) echo "用法：bash $RUN/start.sh [zone|nozone] [norec]"; exit 1 ;;
    esac
done
if [ "$MODE" = zone ] && [ $REC = 0 ]; then echo "停：zone 模式只有带记录的配置（dexhand_rec_zone），不能 norec"; exit 1; fi
if [ "$MODE" = zone ]; then CFG=$CFGDIR/aimdk_x2_magicmirror_live_quiet_dexhand_rec_zone.yaml
elif [ $REC = 1 ]; then CFG=$CFGDIR/aimdk_x2_magicmirror_live_quiet_dexhand_rec.yaml
else CFG=$CFGDIR/aimdk_x2_magicmirror_live_quiet_dexhand.yaml; fi
[ -f "$CFG" ] || { echo "停：没有配置 $CFG"; exit 1; }
if [ "$MODE" = zone ]; then
    for f in "$ZONE" "$TUNING" "$RUN/actor_rec.py"; do [ -f "$f" ] || { echo "停：没有 $f"; exit 1; }; done
    # 配置档里写死的路径必须就是这两个文件，否则校验的和节点读的不是同一份
    [ "$(grep -c "^ *actor_zone_file: *\"\?$ZONE\"\?$" "$CFG")" = 1 ] || { echo "停：$CFG 里 actor_zone_file 不是 $ZONE"; exit 1; }
    [ "$(grep -c "^ *actor_tuning_file: *\"\?$TUNING\"\?$" "$CFG")" = 1 ] || { echo "停：$CFG 里 actor_tuning_file 不是 $TUNING"; exit 1; }
fi
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || { echo "停：另一个 start/stop 正在执行"; exit 1; }

alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
if alive "$STATE/launch.pid"; then
    echo "停：整套已经在跑（进程组 $(cat "$STATE/launch.pid")，日志 $(cat "$STATE/log_dir")）。先 stop.sh"
    exit 1
fi
if [ -s /root/x2-dexhand-run/run/launch.pid ] && kill -0 "$(cat /root/x2-dexhand-run/run/launch.pid)" 2>/dev/null; then
    echo "停：dexhand 那套 start.sh 起的整套还在跑，先 bash /root/x2-dexhand-run/stop.sh"; exit 1
fi

LOG=$RUN/logs/run-$(date +%m%d-%H%M%S)-$MODE$([ $REC = 1 ] && echo -rec)
mkdir -p "$LOG"
ENV="env -i HOME=/root TERM=xterm TREE=$TREE BRANCH=$BRANCH ENVF=$ENVF CFG=$CFG MODE=$MODE ZONE=$ZONE TUNING=$TUNING LOG=$LOG bash --noprofile --norc -c"

$ENV '
source "$ENVF" >/dev/null 2>&1 || { echo "停：环境脚本出错"; exit 1; }
[ -n "$FASTRTPS_DEFAULT_PROFILES_FILE" ] || { echo "停：DDS 没限到机器人网卡，告诉 Claude"; exit 1; }
[ "$(git -C "$TREE" rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "停：$TREE 不在 $BRANCH"; exit 1; }
echo "== 树 $TREE @ $(git -C "$TREE" rev-parse --short HEAD)"
echo "== manipulation_execution 来自 $(ros2 pkg prefix manipulation_execution)"
echo "== runtime(aimdk_robot) 来自 $(ros2 pkg prefix aimdk_robot)（应为 $TREE/install）"
for p in manipulation_execution aimdk_robot robot_config embodied_bringup ibrobot_msgs; do
    [ "$(ros2 pkg prefix $p)" = "$TREE/install" ] || { echo "停：$p 来自 $(ros2 pkg prefix $p)，不是 $TREE/install"; exit 1; }
done
echo "== 配置 $CFG"
ls "$(ros2 pkg prefix aimdk_robot)/share/aimdk_robot/profiles/x2_ultra_omnihand_dexhand.yaml" >/dev/null 2>&1 \
    || { echo "停：装好的 aimdk_robot 里没有 x2_ultra_omnihand_dexhand 档"; exit 1; }
ros2 pkg executables manipulation_execution | grep hand_perception_node >/dev/null \
    || { echo "停：manipulation_execution 里没有 hand_perception_node"; exit 1; }
[ -d /root/x2-dexhand-run/models ] || { echo "停：没有 /root/x2-dexhand-run/models"; exit 1; }
if [ "$MODE" = zone ]; then
    echo "== 校验区域和参数文件"
    python3 -m manipulation_execution.actor_zone_config --zone "$ZONE" --tuning "$TUNING" >"$LOG/zone_check.txt" 2>&1
    rc=$?; sed "s/^/   /" "$LOG/zone_check.txt"
    [ $rc = 0 ] || { echo "停：区域/参数文件校验没过（rc=$rc），改好再跑"; exit 1; }
    cp "$ZONE" "$LOG/actor_zone.yaml"; cp "$TUNING" "$LOG/actor_tuning.yaml"
fi
if pgrep -af "^[^ ]*python[0-9.]* .*ros2 launch"; then echo "停：已有 ros2 launch 在跑（不是本脚本起的）"; exit 1; fi
echo "== 厂商状态（必须是 action STAND_DEFAULT）"
s=$(python3 /root/x2-retarget-run/mc_state.py 2>&1); echo "$s"
echo "$s" | grep -q "^action STAND_DEFAULT" || { echo "停：厂商不在 STAND_DEFAULT。上次没有正常退出 URS，告诉 Claude"; exit 1; }
' || exit 1

set -m
start_bg() {  # name, command
    nohup setsid bash -c "echo \$\$ > '$STATE/$1.pid'; exec $2" </dev/null >"$LOG/$1.log" 2>&1 9>&- &
}
start_bg launch "$ENV 'source $ENVF >/dev/null 2>&1; exec ros2 launch embodied_bringup embodied_pipeline.launch.py robot_config:=aimdk_x2_magicmirror config_path:=$CFG control_mode:=model_inference authorize_motion:=true'"
start_bg rt_watch "$ENV 'source $ENVF >/dev/null 2>&1; exec python3 -u /root/visual-talk-run/rt_watch.py'"
start_bg fb_watch "$ENV 'source $ENVF >/dev/null 2>&1; exec python3 -u /root/visual-talk-run/fb_watch.py'"
start_bg hand_watch "$ENV 'source $ENVF >/dev/null 2>&1; exec python3 -u $RUN/hand_watch.py'"
WATCHERS="rt_watch fb_watch hand_watch"
if [ $REC = 1 ]; then
    for g in frame preview mc status; do
        start_bg rec_$g "$ENV 'source $ENVF >/dev/null 2>&1; exec python3 -u $RUN/hand_rec.py $LOG/rec $g'"
        WATCHERS="$WATCHERS rec_$g"
    done
fi
if [ "$MODE" = zone ]; then
    for g in state cand; do
        start_bg actor_$g "$ENV 'source $ENVF >/dev/null 2>&1; exec python3 -u $RUN/actor_rec.py $LOG/rec $g'"
        WATCHERS="$WATCHERS actor_$g"
    done
fi
sleep 2
echo "$LOG" > "$STATE/log_dir"; echo "$LOG" > "$RUN/LATEST"
echo "$WATCHERS" > "$STATE/watchers"
{
    echo "mode=$MODE rec=$REC tree=$TREE cfg=$CFG start=$(date +%s)"
    if [ "$MODE" = zone ]; then
        echo "zone_sha256=$(sha256sum "$ZONE" | cut -d" " -f1) tuning_sha256=$(sha256sum "$TUNING" | cut -d" " -f1)"
    fi
} > "$LOG/mode.txt"

for name in launch $WATCHERS; do
    if alive "$STATE/$name.pid"; then echo "== $name 已起（进程组 $(cat "$STATE/$name.pid")）"; else echo "!! $name 没起来，看 $LOG/$name.log"; fi
done
echo "== 模式 $MODE；日志目录 $LOG"
echo "   跟启动日志：tail -f $LOG/launch.log"
echo "   看手：tail -f $LOG/hand_watch.log"
[ "$MODE" = zone ] && echo "   看选人：tail -f $LOG/rec/actor_state.jsonl"
echo "   停止：bash $RUN/stop.sh"
echo "   提醒：别碰机器人的头"
