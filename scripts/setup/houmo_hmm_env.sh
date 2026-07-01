#!/bin/bash
# Houmo LQ50/M50 (xh2) HMM runtime environment for the board (openEuler Embedded / aarch64).
#
# Source this before running any `device:=hmm` inference:
#   source scripts/setup/houmo_hmm_env.sh
#
# Verified working on OPi_20T + LQ50-24G (houmo-drv-xh2 1.3.0, tcim_runtime_lite 1.3.0):
#   - device_num=1, vision.hmm forward returns [1,64,960] fp16.
#
# Why TCIM_BACKEND=Xh2HalBackend (NOT HDPL_PLATFORM=ASIC):
#   Setting HDPL_PLATFORM=ASIC makes tcim_lite dlopen `libhdplrt_asic.so`, which is
#   NOT shipped in the aarch64 Runtime SDK tarball. The HAL backend path instead uses
#   `libhal_xh2a.so` (present in houmo-sdk/hal/lib) and correctly initializes the device.

# --- Houmo runtime SDK root (extracted houmo_tcim_runtime_xh2_linux_aarch64-1.3.0) ---
# Adjust this path if your Runtime SDK is unpacked elsewhere.
TCIM_RUNTIME_PATH="${TCIM_RUNTIME_PATH:-/root/houmo_tcim_runtime_xh2_linux_aarch64-1.3.0}"
export TCIM_RUNTIME_PATH

# --- System software (driver) SDK root ---
export HOUMO_SDK_PATH="${HOUMO_SDK_PATH:-/usr/local/houmo-sdk}"

# --- Backend selection: HAL backend (uses libhal_xh2a.so, no libhdplrt_asic.so needed) ---
export TCIM_BACKEND="${TCIM_BACKEND:-Xh2HalBackend}"
export HOUMO_TARGET="${HOUMO_TARGET:-xh2}"

# --- Library path: runtime SDK lib + HAL lib ---
export LD_LIBRARY_PATH="${TCIM_RUNTIME_PATH}/lib:${HOUMO_SDK_PATH}/hal/lib:${LD_LIBRARY_PATH}"

# --- Python path: tcim_lite lives under the runtime SDK's python/ dir ---
export PYTHONPATH="${TCIM_RUNTIME_PATH}/python:${PYTHONPATH}"

# Sanity check (non-fatal): can we see the device?
if command -v python3 >/dev/null 2>&1; then
    _DEV_NUM=$(python3 -c "import tcim_lite.runtime as r; print(r.get_device_num())" 2>/dev/null | tail -1)
    if [ -n "$_DEV_NUM" ]; then
        echo "[houmo_hmm_env] LQ50 device_num=${_DEV_NUM} (TCIM_BACKEND=${TCIM_BACKEND})"
    fi
fi
