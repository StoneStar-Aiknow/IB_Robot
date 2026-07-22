#!/bin/sh
# Houmo XH2 1.3.0 runtime environment for RoboPi/OpenHarmony.
#
# Source this before running HMM inference:
#   . /data/roboframe/scripts/setup/houmo_hmm_env.sh
#
# This is the OpenHarmony-native musl package. It runs directly with the
# system loader and does not use the legacy /data/houmo glibc launcher.
HOUMO_ROOT="${HOUMO_ROOT:-/data/local/houmo}"
HOUMO_PATH="${HOUMO_PATH:-${HOUMO_ROOT}}"
HOUMO_SDK_PATH="${HOUMO_SDK_PATH:-/data/local/houmo-sdk}"
TCIM_RUNTIME_PATH="${TCIM_RUNTIME_PATH:-${HOUMO_ROOT}/lib}"

export HOUMO_ROOT HOUMO_PATH HOUMO_SDK_PATH TCIM_RUNTIME_PATH
export TCIM_BACKEND=xh2
export HOUMO_TARGET=xh2
export HOUMO_VERSION="${HOUMO_VERSION:-1.3.0}"

# Keep existing RoboFrame paths while making the native runtime visible.
export LD_LIBRARY_PATH="${HOUMO_ROOT}/lib:${HOUMO_SDK_PATH}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HOUMO_ROOT}/python:${PYTHONPATH}"

if [ ! -f "${HOUMO_ROOT}/lib/libtcim_runtime_lite.so" ]; then
    echo "[houmo_hmm_env] missing TCIM runtime: ${HOUMO_ROOT}/lib/libtcim_runtime_lite.so" >&2
elif [ ! -f "${HOUMO_SDK_PATH}/lib/libhal_xh2a.so" ]; then
    echo "[houmo_hmm_env] missing Houmo HAL: ${HOUMO_SDK_PATH}/lib/libhal_xh2a.so" >&2
else
    echo "[houmo_hmm_env] HOUMO_VERSION=${HOUMO_VERSION} TCIM_BACKEND=${TCIM_BACKEND}"
fi
