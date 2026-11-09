#!/usr/bin/env bash
# X2 照镜子拍照：现场改上位机地址。在 .5 上跑。拍照节点每次发图前都重读目标文件，改完不用重启、不用重新部署，
# 排队中的照片自动发往新地址。
#
#   bash photo_target.sh                 显示当前目标、outbox 排队张数，并访问一次目标的 /health
#   bash photo_target.sh <ip>[:端口]      检查格式 → 打印 .5 从哪个本机地址发出 → 访问 /health → 写入目标文件
#
# 填的是上位机在同一张网上的地址（不是 .5 自己的 192.168.3.10）。上位机先起好 receiver.py。
# /health 不通时照样写入（照片在 outbox 里等着，上位机一起来就补发），但会提示；地址格式不对则不写。
# 只查路由表（UDP connect 不发包），不改 .5 的任何网络设置。
#
# 环境变量：PHOTO_RUN（默认 /root/x2-feat-photo-capture-run，与 yaml 里 photos_dir / target_file 一致）、PY（默认 python3）。
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=${PHOTO_RUN:-/root/x2-feat-photo-capture-run}
TARGET_FILE=$RUN/photo_target
OUTBOX=$RUN/photos/outbox
PY=${PY:-python3}

die() { printf '\033[31m停：%s\033[0m\n' "$*" >&2; exit 1; }
command -v "$PY" >/dev/null || die "找不到 $PY"

show_outbox() {
    if [[ ! -d $OUTBOX ]]; then
        echo "outbox $OUTBOX 还不存在（拍照节点没起过）"
        return
    fi
    local n
    n=$(find "$OUTBOX" -maxdepth 1 -name '*.jpg' ! -name 'current.jpg' | wc -l)
    echo "outbox 排队 $n 张（$OUTBOX）"
    [[ -f $OUTBOX/current.jpg ]] && echo "  另有进行中一轮的暂存 current.jpg（这一轮结束或节点重启后发出）"
    return 0
}

current_target() {
    [[ -f $TARGET_FILE ]] || return 0
    grep -v '^[[:space:]]*#' "$TARGET_FILE" | grep -m1 -v '^[[:space:]]*$' || true
}

if [[ $# -eq 0 ]]; then
    target=$(current_target)
    if [[ -n $target ]]; then
        echo "目标文件 $TARGET_FILE：$target"
    else
        echo "目标文件 $TARGET_FILE 不存在或为空：照片在 outbox 里等，直到写入目标"
    fi
    show_outbox
    if [[ -n $target ]]; then
        echo
        PHOTO_TARGET_FILE=$TARGET_FILE "$PY" "$HERE/send_test.py" health || true
    fi
    exit 0
fi
[[ $# -eq 1 && $1 != -* ]] || die "用法：bash photo_target.sh [<ip>[:端口]]"

set +e
PHOTO_TARGET_FILE=$TARGET_FILE "$PY" "$HERE/send_test.py" health "$1"
rc=$?
set -e
case $rc in
    0) ;;
    1) printf '\033[33m上位机现在没回 /health：照样写入，照片在 outbox 里等它起来。看上位机 receiver.py 起了没、防火墙放行没、填的是不是同一张网上的地址。\033[0m\n' ;;
    2) die "地址格式不对，没有写入" ;;
    *) die "send_test.py 没跑成（退出码 $rc），没有写入" ;;
esac

[[ -d $RUN ]] || die "$RUN 不存在：拍照的运行目录还没部署"
tmp=$(mktemp "$RUN/.photo_target.XXXXXX")
printf '%s\n' "$1" >"$tmp"
mv -f "$tmp" "$TARGET_FILE"
echo "已写入 $TARGET_FILE：$(current_target)"
show_outbox
