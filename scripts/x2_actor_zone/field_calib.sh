#!/usr/bin/env bash
# 现场标定站位区（地面四边形 + 左右蓝线），写进 duck 上运行时读的 actor_zone.yaml。在笔记本上跑。
#
#   bash field_calib.sh                 全套：采样 → 叠图 → 点角 → 核对 → 上传（每步都会停下来问）
#   bash field_calib.sh snap            只采样并拉回本机，打印本机目录
#   bash field_calib.sh pick <目录> [图] 对已拉回的采样叠图、点角、核对；不上传。[图] 换成某张单图来点
#   bash field_calib.sh push <区域.yaml> 校验后上传到 duck（旧文件先备份成 .bak-<时刻>）
#   bash field_calib.sh deploy          把本目录 duck/ 下的脚本装到 duck 的 /root/x2-actor-run（换新板子时用）
#
# 环境变量：DUCK（ssh 主机名，默认 duck）、CALIB_DIR（本机存采样的目录，默认 ~/x2_actor_calib，别放进 git：
# 截图里有人脸）、EMPTY（空场张数，默认 5）、HOLD（每个位置几张，默认 4）、EVERY（隔几秒一张，默认 3）、
# PY（开窗口用的 python，默认 /usr/bin/python3，要带 GUI 的 cv2）。
# 运行时读的区域文件只在 start.sh 启动时读一次：上传后要 stop.sh 再 start.sh 才生效（不用构建）。
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DUCK=${DUCK:-duck}
CALIB_DIR=${CALIB_DIR:-$HOME/x2_actor_calib}
EMPTY=${EMPTY:-5}; HOLD=${HOLD:-4}; EVERY=${EVERY:-3}
PY=${PY:-/usr/bin/python3}
RUN=/root/x2-actor-run
RZ=$RUN/zone/actor_zone.yaml; RT=$RUN/zone/actor_tuning.yaml
DENV="env -i HOME=/root TERM=xterm bash --noprofile --norc -c"
POSITIONS=("近左角" "近右角" "远右角" "远左角" "中间")

say() { printf '\n\033[1m== %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m停：%s\033[0m\n' "$*" >&2; exit 1; }
ask() { local r; read -r -p "$1 [y/N] " r; [[ $r == y || $r == Y ]]; }

do_snap() {
    local ts count rdir ldir
    ts=$(date +%m%d-%H%M%S)
    count=$((EMPTY + HOLD * ${#POSITIONS[@]}))
    rdir=$RUN/calib/snap-$ts
    ldir=$CALIB_DIR/snap-$ts
    ssh "$DUCK" "test -f $RUN/snap.py && test -f $RUN/env_actor.sh" || die "$DUCK 上没有 $RUN/snap.py 或 env_actor.sh（新板子先 deploy）"
    say "采样：共 $count 张，每 $EVERY 秒一张，约 $((count * EVERY)) 秒"
    {
        echo "  1. 开头 $EMPTY 张：画面里不要有人（站位区和两边都清空）"
        for p in "${POSITIONS[@]}"; do echo "  然后 $HOLD 张：一个人站在【$p】（角 = 想要的区域角上），站稳别动"; done
        echo "  屏幕上会提示什么时候换位置。机器人不用起 start.sh，只要相机在出图。"
    } >&2
    read -r -p "人准备好、画面清空后按回车开始" _
    local phase="保持空场"
    printf '\033[1;33m>>> 现在：%s\033[0m\n' "$phase" >&2
    ssh "$DUCK" "$DENV 'source $RUN/env_actor.sh >/dev/null 2>&1; exec python3 -u $RUN/snap.py $rdir $count $EVERY'" 2>&1 |
        while IFS= read -r line; do
            if [[ $line =~ ^([0-9]+)/([0-9]+)\  ]]; then
                local i=${BASH_REMATCH[1]} want
                if ((i < EMPTY)); then want="保持空场"
                elif ((i < count)); then want="站到【${POSITIONS[$(((i - EMPTY) / HOLD))]}】"
                else want="完成"; fi
                echo "   已存 $i/$count" >&2
                if [[ $want != "$phase" ]]; then printf '\033[1;33m>>> 现在：%s\033[0m\n' "$want" >&2; phase=$want; fi
            elif [[ $line == saved* ]]; then echo "   $line" >&2
            fi
        done || die "采样没跑完（相机没出图？看上面的输出）"
    mkdir -p "$CALIB_DIR"
    scp -q -r "$DUCK:$rdir" "$ldir" || die "拉回 $rdir 失败"
    local n; n=$(find "$ldir" -name 'snap_*.jpg' | wc -l)
    [ "$n" = "$count" ] || die "$ldir 里只有 $n 张，应为 $count"
    echo "   已拉回 $n 张到 $ldir（duck 上的 $rdir 可以留着，也可以删）" >&2
    echo "$ldir"
}

do_pick() {
    local ldir=$1 img=${2:-} new ov
    [ -d "$ldir" ] || die "没有目录 $ldir"
    new=$ldir/actor_zone.yaml; ov=$ldir/zone_overlay.png
    {  # 只有最后一行（区域文件路径）走 stdout，给 all 接
    if [ -z "$img" ]; then
        say "叠图（前 $EMPTY 张当空场）"
        "$PY" "$HERE/zone_composite.py" "$ldir" "$ldir/composite.jpg" --empty "$EMPTY" || die "叠图失败"
        img=$ldir/composite.jpg
    fi
    say "点区域：先点左蓝线、右蓝线（只取 x），再点 近左、近右、远右、远左 四个脚点"
    echo "   红点 + 数字 = 那一帧人的脚。u 撤销，r 重来，s/回车 保存并校验，q 放弃"
    echo "   蓝线以外（整个框都在外面）的人完全不管；四个角是站进去才会被选的地面范围"
    "$PY" "$HERE/zone_click.py" "$img" "$new" --note "现场标定 $(basename "$ldir")" || die "没写成区域文件（放弃或校验没过）"
    "$PY" "$HERE/actor_overlay.py" "$img" "$new" "$ov" >/dev/null || die "画核对图失败"
    say "核对图 $ov"
    (xdg-open "$ov" >/dev/null 2>&1 &) || true
    cat "$new"
    } >&2
    echo "$new"
}

do_push() {
    local new=$1 ts sha_local out
    [ -f "$new" ] || die "没有 $new"
    say "本机正式校验 $new"
    PYTHONPATH="$REPO/src/manipulation_execution${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m manipulation_execution.actor_zone_config --zone "$new" || die "本机校验没过"
    ts=$(date +%m%d-%H%M%S)
    sha_local=$(sha256sum "$new" | cut -d' ' -f1)
    ssh "$DUCK" "mkdir -p $RUN/zone" || die "连不上 $DUCK"
    scp -q "$new" "$DUCK:$RZ.new-$ts" || die "上传失败"
    say "duck 上用装好的那份校验（连同参数文件），通过才替换；旧文件备份成 $RZ.bak-$ts"
    out=$(ssh "$DUCK" "$DENV '
        source $RUN/env_actor.sh >/dev/null 2>&1 || { echo \"FAIL 环境脚本出错\"; exit 1; }
        python3 -m manipulation_execution.actor_zone_config --zone $RZ.new-$ts --tuning $RT || { echo FAIL 校验没过，duck 上没换; rm -f $RZ.new-$ts; exit 1; }
        if [ -f $RZ ]; then cp -p $RZ $RZ.bak-$ts && echo BACKUP $RZ.bak-$ts; fi
        mv $RZ.new-$ts $RZ && echo SHA \$(sha256sum $RZ | cut -d\" \" -f1)
        if [ -s $RUN/run/launch.pid ] && kill -0 \$(cat $RUN/run/launch.pid) 2>/dev/null; then echo RUNNING; fi
    '" 2>&1) || { echo "$out" | grep -E '^(OK|ERROR|WARN|FAIL)' >&2 || true; die "duck 上没换"; }
    echo "$out" | grep -E '^(OK|zone|tuning|values|WARN|BACKUP|SHA|RUNNING)' | sed 's/^/   /'
    [ "$(echo "$out" | sed -n 's/^SHA //p')" = "$sha_local" ] || die "duck 上的 sha256 和本机不一致，去 duck 上看 $RZ"
    say "已换好，sha256 ${sha_local:0:12} 与本机一致"
    if echo "$out" | grep -q '^RUNNING'; then echo "   注意：整套正在跑，用的还是旧区域。"; fi
    echo "   生效：ssh $DUCK 后执行"
    echo "     bash $RUN/stop.sh"
    echo "     bash $RUN/start.sh"
    echo "   退回旧区域：ssh $DUCK 'cp $RZ.bak-$ts $RZ'，再 stop/start"
}

do_deploy() {
    local ts stage
    ts=$(date +%m%d-%H%M%S); stage=$RUN/.deploy-$ts
    say "把 $HERE/duck/ 装到 $DUCK:$RUN（不同的旧文件备份成 .bak-$ts；区域/参数文件已有就不动）"
    # 目标目录不存在时 scp -r 把 duck/ 拷成它；不能写 duck/.（新版 scp 拒收文件名 "."）
    scp -q -r "$HERE/duck" "$DUCK:$stage" || die "上传失败"
    ssh "$DUCK" "bash -s" <<EOF
set -e
cd $stage
mkdir -p $RUN/zone
for f in start.sh stop.sh env_actor.sh snap.py actor_rec.py hand_rec.py hand_watch.py run_read.py; do
    if [ -f $RUN/\$f ] && cmp -s \$f $RUN/\$f; then echo "same    \$f"; continue; fi
    [ -f $RUN/\$f ] && cp -p $RUN/\$f $RUN/\$f.bak-$ts && echo "backup  \$f -> \$f.bak-$ts"
    cp \$f $RUN/\$f && echo "install \$f"
done
if [ -f $RT ]; then echo "keep    zone/actor_tuning.yaml"; else cp zone/actor_tuning.yaml $RT && echo "install zone/actor_tuning.yaml"; fi
if [ -f $RZ ]; then echo "keep    zone/actor_zone.yaml"; else echo "MISSING zone/actor_zone.yaml：先现场标定（field_calib.sh），实验室那份在 zone/actor_zone.lab-1024.yaml 只能在实验室用"; fi
cp zone/actor_zone.lab-1024.yaml $RUN/zone/
rm -rf $stage
EOF
    echo "   start.sh 还依赖：/root/x2-dexhand-wt（actor-select 分支已构建）、/root/x2-dexhand-run/{models,pydeps,extern}、"
    echo "   /root/visual-talk-run/{rt_watch,fb_watch}.py、/root/x2-retarget-run/mc_state.py。见 README 第 5 节"
}

case "${1:-all}" in
    snap) do_snap ;;
    pick) do_pick "${2:?用法：field_calib.sh pick <采样目录> [图]}" "${3:-}" ;;
    push) do_push "${2:?用法：field_calib.sh push <区域.yaml>}" ;;
    deploy) do_deploy ;;
    all)
        dir=$(do_snap)
        new=$(do_pick "$dir")
        ask "核对图里的区域对吗？上传到 $DUCK？" || { echo "没上传。以后可以：bash $0 push $new"; exit 0; }
        do_push "$new"
        ;;
    *) sed -n '2,15p' "$0"; exit 1 ;;
esac
