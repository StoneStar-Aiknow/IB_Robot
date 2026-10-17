#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
# Verify interpreter selection without CANN, an NPU or a compiler.
set -euo pipefail
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
for name in workspace/venv active explicit setup; do
    mkdir -p "${test_root}/${name}/bin"
    ln -s "$(command -v python3)" "${test_root}/${name}/bin/python"
done

check_python() (
    local expected="$1"
    shift
    unset GRASPGENX_OPS_VENV_PYTHON VENV_PYTHON VIRTUAL_ENV
    export WORKSPACE="${test_root}/workspace"
    for setting in "$@"; do export "${setting}"; done
    source "${repo_root}/scripts/setup/install_graspgenx_ascend_ops.sh"
    # Only the selection/validation and cache stamping are under test.
    graspgenx_ops_stamp() { printf '%s\n' "${GRASPGENX_OPS_VENV_PYTHON}"; }
    graspgenx_ops_prepare_env() { :; }
    graspgenx_ops_build_one() { :; }
    install_graspgenx_ascend_ops --root "${test_root}/unrelated-cache" >/dev/null
    [[ "${GRASPGENX_OPS_VENV_PYTHON}" == "${expected}" ]]
)

check_python "${test_root}/workspace/venv/bin/python"
check_python "${test_root}/active/bin/python" "VIRTUAL_ENV=${test_root}/active"
check_python "${test_root}/setup/bin/python" "VIRTUAL_ENV=${test_root}/active" "VENV_PYTHON=${test_root}/setup/bin/python"
check_python "${test_root}/explicit/bin/python" "VENV_PYTHON=${test_root}/setup/bin/python" "GRASPGENX_OPS_VENV_PYTHON=${test_root}/explicit/bin/python"
printf 'GraspGenX operator interpreter selection: 4 cases passed\n'
