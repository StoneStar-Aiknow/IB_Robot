# Bash activation: source always follows the saved default.
# Zsh support is not claimed until its variable-attribute tests are available.
if [ -z "${BASH_VERSION:-}" ]; then
    if [ -e "$HOME/.config/ibrobot/dds/config.yaml" ] || [ -L "$HOME/.config/ibrobot/dds/config.yaml" ] ||
       [ -n "${IBROBOT_DDS_STATE+x}" ] || [ -n "${IBROBOT_DDS_STATUS+x}" ]; then
        printf '%s\n' 'DDS 配置目前仅支持 Bash；本终端未应用保存的默认配置，也不能使用 set/unset。' >&2
    fi
    return 0
fi
source "${WORKSPACE_DIR}/scripts/dds/shell_activation.bash" || return $?
_ib_dds_activate source
