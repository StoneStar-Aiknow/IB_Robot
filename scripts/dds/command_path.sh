# Register the workspace-owned DDS CLI for a sourced IB_Robot environment.
# This file is sourced by .shrc_local; it must not be executed as a subprocess.

# Remove only our previous registration and duplicate copies of this target.
# Preserve the order and empty components of every unrelated PATH entry.
_IB_DDS_TARGET="${WORKSPACE_DIR}/scripts/dds"
_IB_DDS_REST="${PATH}"
_IB_DDS_RESULT=""
_IB_DDS_SEPARATOR=""
while :; do
    case "${_IB_DDS_REST}" in
        *:*) _IB_DDS_PART="${_IB_DDS_REST%%:*}"; _IB_DDS_REST="${_IB_DDS_REST#*:}"; _IB_DDS_MORE=1 ;;
        *) _IB_DDS_PART="${_IB_DDS_REST}"; _IB_DDS_MORE=0 ;;
    esac
    if [ "${_IB_DDS_PART}" != "${_IB_DDS_TARGET}" ] &&
       { [ -z "${_IBROBOT_DDS_COMMAND_BIN:-}" ] || [ "${_IB_DDS_PART}" != "${_IBROBOT_DDS_COMMAND_BIN}" ]; }; then
        _IB_DDS_RESULT="${_IB_DDS_RESULT}${_IB_DDS_SEPARATOR}${_IB_DDS_PART}"
        _IB_DDS_SEPARATOR=:
    fi
    [ "${_IB_DDS_MORE}" = 1 ] || break
done
if [ -x "${_IB_DDS_TARGET}/dds-config" ]; then
    PATH="${_IB_DDS_TARGET}${_IB_DDS_SEPARATOR}${_IB_DDS_RESULT}"
    export _IBROBOT_DDS_COMMAND_BIN="${_IB_DDS_TARGET}"
else
    PATH="${_IB_DDS_RESULT}"
    unset _IBROBOT_DDS_COMMAND_BIN
fi
export PATH
# Refresh shell command caches after switching workspaces.
hash -r 2>/dev/null || :
unset _IB_DDS_TARGET _IB_DDS_REST _IB_DDS_RESULT _IB_DDS_SEPARATOR _IB_DDS_PART _IB_DDS_MORE
