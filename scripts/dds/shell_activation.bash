# Current-shell DDS activation/restoration. No generated shell code/eval.
_ib_dds_names=(RMW_IMPLEMENTATION CYCLONEDDS_URI FASTRTPS_DEFAULT_PROFILES_FILE
    FASTDDS_DEFAULT_PROFILES_FILE FASTDDS_BUILTIN_TRANSPORTS RMW_FASTRTPS_USE_QOS_FROM_XML
    RMW_FASTRTPS_PUBLICATION_MODE SKIP_DEFAULT_XML_FILE ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT
    FASTDDS_ENVIRONMENT_FILE IBROBOT_DDS_STATE IBROBOT_DDS_STATUS)

_ib_dds_attribute() {
    local declaration
    declaration=$(declare -p "$1" 2>/dev/null) || { printf absent; return; }
    declaration=${declaration#declare }
    printf '%s' "${declaration%% *}"
}

_ib_dds_restore_map() {
    # namerefs are internal only; user-controlled variable attributes are rejected.
    local -n values=$1 attrs=$2
    local name
    for name in "${_ib_dds_names[@]}"; do
        case ${attrs[$name]} in
            absent) unset "$name" || return ;;
            *) printf -v "$name" '%s' "${values[$name]}" || return
               if [[ ${attrs[$name]} == -x ]]; then export "$name" || return
               else export -n "$name" || return; fi ;;
        esac
    done
}

_ib_dds_error() {
    # Runtime errors honor --json even when detected before entering Python.
    if [[ ${json_output:-0} == 1 ]]; then
        python3 -c 'import json,sys; print(json.dumps({"error":sys.argv[1], "code":4, "advice":sys.argv[2]}), file=sys.stderr)' "$1" "$2"
    else
        printf 'DDS：%s\n建议：%s\n' "$1" "$2" >&2
    fi
    return 4
}

_ib_dds_activate() {
    local mode=$1 name attrs directory action result=0 present value argument json_output=0 repairing_source=0
    shift
    for argument in "$@"; do [[ $argument != --json ]] || json_output=1; done
    # A saved default can fail on source because its DDS packages are missing.
    # Permit explicit set to repair that known status-only failure without
    # treating the tool's own degraded marker as an inherited user record.
    if [[ $mode == set && ${_ib_dds_owned:-0} != 1 && ${_ib_dds_failed_source:-0} == 1 &&
          ${IBROBOT_DDS_STATUS-} == degraded && ! -v IBROBOT_DDS_STATE ]]; then
        repairing_source=1
    fi
    if [[ $mode == source && ${_ib_dds_owned:-0} != 1 && ! -v IBROBOT_DDS_STATE && ! -v IBROBOT_DDS_STATUS &&
          ! -e $HOME/.config/ibrobot/dds/config.yaml && ! -L $HOME/.config/ibrobot/dds/config.yaml ]]; then
        return 0
    fi
    local -A before_values=() before_attrs=()
    for name in "${_ib_dds_names[@]}"; do
        attrs=$(_ib_dds_attribute "$name")
        case $attrs in absent|--|-x) ;; *) _ib_dds_error "不支持的变量属性：$name ($attrs)" "移除 readonly/数组等属性的设置来源，再在新终端初始化。"; return 4;; esac
        before_attrs[$name]=$attrs
        before_values[$name]=${!name-}
        if [[ ${_ib_dds_owned:-0} == 1 ]]; then
            # set/source explicitly select or restore a configuration. Allow
            # runtime-value drift here, but never guess a changed identity or
            # baseline. The before maps still capture actual values for rollback.
            if [[ $HOME != "$_ib_dds_home" ]]; then
                _ib_dds_error "HOME 已改变" "请在新终端重新初始化。"
                return 4
            fi
            if [[ ( $name == IBROBOT_DDS_STATE || $name == IBROBOT_DDS_STATUS ) &&
                  ( $attrs != "${_ib_dds_last_attrs[$name]}" || ${!name-} != "${_ib_dds_last_values[$name]}" ) ]]; then
                _ib_dds_error "dds-config 配置状态被修改：$name" "请在新终端重新初始化。"
                return 4
            fi
        fi
    done
    if [[ $mode == unset && ${_ib_dds_owned:-0} != 1 &&
          ( -v IBROBOT_DDS_STATE || -v IBROBOT_DDS_STATUS ) &&
          ! ( ${_ib_dds_failed_source:-0} == 1 && ${IBROBOT_DDS_STATUS-} == degraded &&
             ! -v IBROBOT_DDS_STATE ) ]]; then
        _ib_dds_error "本终端仅继承了 DDS 配置，缺少接管前的环境" "请在原设置终端执行 unset。"
        return 4
    fi
    if [[ $mode == unset && ${_ib_dds_owned:-0} == 1 &&
          ( ${_ib_dds_base_attrs[IBROBOT_DDS_STATE]} != absent ||
            ${_ib_dds_base_attrs[IBROBOT_DDS_STATUS]} != absent ) ]]; then
        _ib_dds_error "本终端接管前已继承 DDS 配置，无法恢复到未接管环境" "请在原设置终端执行 unset。"
        return 4
    fi
    directory=$(mktemp -d "${TMPDIR:-/tmp}/ibrobot-dds-shell.XXXXXXXX") || return 6
    if [[ $repairing_source == 1 ]]; then
        # The degraded marker was written by our failed source, not by the
        # caller. Do not inherit it as a baseline on a successful repair.
        unset IBROBOT_DDS_STATUS
        before_attrs[IBROBOT_DDS_STATUS]=absent
        before_values[IBROBOT_DDS_STATUS]=''
    fi
    if [[ $mode == unset && ( ${_ib_dds_owned:-0} == 1 || ${_ib_dds_failed_source:-0} == 1 ) ]]; then
        : > "$directory/restore-owned"
    fi
    # Capture shell values without exporting/changing them in the parent.
    for name in "${_ib_dds_names[@]}" ROS_DOMAIN_ID ROS_LOCALHOST_ONLY; do
        # Our status-only failure is not an inherited runtime ownership record.
        if [[ ${_ib_dds_owned:-0} == 1 && ! -v IBROBOT_DDS_STATE && $name == IBROBOT_DDS_STATUS ]]; then
            printf '%s\0%s\0%s\0' "$name" 0 ''
            continue
        fi
        present=0; [[ -v $name ]] && present=1
        printf '%s\0%s\0%s\0' "$name" "$present" "${!name-}"
    done > "$directory/input"
    action=$(python3 "${_IBROBOT_DDS_COMMAND_BIN}/shell_entry.py" "$mode" "$directory" "$@" < "$directory/input") || result=$?
    if (( result != 0 )); then
        rm -rf -- "$directory"
        if [[ $mode == source && $result != 4 ]]; then
            # Keep the prior runtime environment; warn even in noninteractive shells.
            if [[ ${_ib_dds_owned:-0} != 1 ]]; then
                declare -gA _ib_dds_base_values=() _ib_dds_base_attrs=()
                for name in "${_ib_dds_names[@]}"; do
                    _ib_dds_base_values[$name]=${before_values[$name]}
                    _ib_dds_base_attrs[$name]=${before_attrs[$name]}
                done
            fi
            export IBROBOT_DDS_STATUS=degraded
            _ib_dds_failed_source=1
            result=0
        else
            if [[ $repairing_source == 1 ]]; then export IBROBOT_DDS_STATUS=degraded; fi
            return "$result"
        fi
    elif [[ $action == unset ]]; then
        if [[ ${_ib_dds_owned:-0} == 1 || ${_ib_dds_failed_source:-0} == 1 ]]; then
            _ib_dds_restore_map _ib_dds_base_values _ib_dds_base_attrs || result=$?
        fi
        if (( result == 0 )); then
            python3 "${_IBROBOT_DDS_COMMAND_BIN}/shell_entry.py" commit "$directory" || result=$?
        fi
        rm -rf -- "$directory"
        if (( result != 0 )); then
            _ib_dds_restore_map before_values before_attrs || return 4
            if [[ $json_output != 1 ]]; then
                printf 'DDS：unset 失败，已恢复操作前的终端环境；请处理上述错误后重试。\n' >&2
            fi
            return "$result"
        fi
        unset _ib_dds_owned _ib_dds_failed_source _ib_dds_home _ib_dds_base_values _ib_dds_base_attrs _ib_dds_last_values _ib_dds_last_attrs
        return 0
    elif [[ $action == restore ]]; then
        rm -rf -- "$directory"
        if [[ ${_ib_dds_owned:-0} == 1 ]]; then
            _ib_dds_restore_map _ib_dds_base_values _ib_dds_base_attrs || return 4
            unset _ib_dds_owned _ib_dds_home _ib_dds_base_values _ib_dds_base_attrs _ib_dds_last_values _ib_dds_last_attrs
        fi
        return 0
    elif [[ $action == apply ]]; then
        while IFS= read -r -d '' name && IFS= read -r -d '' present && IFS= read -r -d '' value; do
            case " ${_ib_dds_names[*]} " in *" $name "*) ;; *) result=6; break;; esac
            if [[ $present == 1 ]]; then
                printf -v "$name" '%s' "$value" && export "$name" || { result=4; break; }
            else unset "$name" || { result=4; break; }; fi
        done < "$directory/values"
        if (( result == 0 )); then
            python3 "${_IBROBOT_DDS_COMMAND_BIN}/shell_entry.py" commit "$directory" || result=$?
        fi
        rm -rf -- "$directory"
        if (( result != 0 )); then
            _ib_dds_restore_map before_values before_attrs || return 4
            if [[ $repairing_source == 1 ]]; then export IBROBOT_DDS_STATUS=degraded; fi
            return "$result"
        fi
        if [[ ${_ib_dds_owned:-0} != 1 ]]; then
            declare -gA _ib_dds_base_values=() _ib_dds_base_attrs=()
            for name in "${_ib_dds_names[@]}"; do
                _ib_dds_base_values[$name]=${before_values[$name]}
                _ib_dds_base_attrs[$name]=${before_attrs[$name]}
            done
        fi
    else rm -rf -- "$directory"; return 6; fi
    unset _ib_dds_failed_source
    _ib_dds_owned=1
    _ib_dds_home=$HOME
    declare -gA _ib_dds_last_values=() _ib_dds_last_attrs=()
    for name in "${_ib_dds_names[@]}"; do
        _ib_dds_last_values[$name]=${!name-}
        _ib_dds_last_attrs[$name]=$(_ib_dds_attribute "$name")
    done
}

dds-config() {
    if [[ ${1:-} == set || ${1:-} == unset ]]; then
        local argument operation=$1
        for argument in "$@"; do
            if [[ $argument == --help || $argument == -h ]]; then
                command "${_IBROBOT_DDS_COMMAND_BIN}/dds-config" "$operation" --help
                return $?
            fi
        done
        shift
        _ib_dds_activate "$operation" "$@"
    else
        command "${_IBROBOT_DDS_COMMAND_BIN}/dds-config" "$@"
    fi
}
