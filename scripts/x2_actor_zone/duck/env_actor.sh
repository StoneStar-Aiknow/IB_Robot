# Robot-network environment for /root/x2-dexhand-wt on branch actor-select (zone selector).
# Source inside `env -i HOME=/root TERM=$TERM bash --noprofile --norc`.
# Copy of /root/x2-dexhand-run/env_dexhand.sh (that file is not touched). Only differences:
#   RUN is /root/x2-actor-run (so ROS_LOG_DIR lands here), the hand node's NPU runtime
#   still comes from the dexhand run dir (DEXRUN, read only), and the CHECK prefix.
MAIN_REPO=/root/IB-ROBOT-X2
WORKTREE=/root/x2-dexhand-wt
RUN=/root/x2-actor-run
DEXRUN=/root/x2-dexhand-run
cd "$WORKTREE" || return 1
source "$MAIN_REPO/venv/bin/activate"
_PY_VER="$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$MAIN_REPO/venv/lib/${_PY_VER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset _PY_VER
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}/usr/local/lib64/python3.11/site-packages
# The hand node's NPU runtime: only ais_bench, linked from /usr/local/lib/python3.11/site-packages
# (that whole directory has its own lerobot, accelerate, ... and stays off the path).
export PYTHONPATH=$PYTHONPATH:$DEXRUN/pydeps
[ -f "$WORKTREE/.shrc_local" ] && source "$WORKTREE/.shrc_local"
source "$WORKTREE/install/local_setup.bash"
source /root/aimdk/install/local_setup.bash
export ROS_LOG_DIR="$RUN/ros_log"
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
_xml=/root/x2-mirror-dds-robot-lan.xml
if [[ -f "$_xml" && "$(ip -o -4 addr show 2>/dev/null)" == *" 10.0.1.50/"* ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$_xml"
else
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  echo "actor-env: 本机没有 10.0.1.50 或缺 $_xml，DDS 不限网卡 —— 停下告诉 Claude" >&2
fi
unset _xml
echo "CHECK: $ROS_DOMAIN_ID $ROS_LOCALHOST_ONLY ${FASTRTPS_DEFAULT_PROFILES_FILE:-NONE} $(git -C "$WORKTREE" --no-optional-locks rev-parse --short HEAD) $(git -C "$WORKTREE" --no-optional-locks rev-parse --abbrev-ref HEAD)"
