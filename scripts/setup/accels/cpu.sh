#!/bin/bash

# CPU-only accelerator profile. Sourced by detect.sh::load_accel_impl when
# SETUP_ACCEL_ID is "cpu", which is also the fallback when no accelerator is
# detected. Every hook below is part of the accelerator contract consumed by
# python_venv.sh; keep the three profiles in sync.

# Whether the Torch GraspGen stack (grasp_gen plus the pointnet2_ops CUDA
# extension) can be installed on this accelerator.
accel_supports_torch_grasp() {
    # pointnet2_ops is a CUDA extension, so the Torch grasp stack has no
    # backend here. The workspace still builds and its CPU tests still run.
    return 1
}

# Whether GraspGenX can be installed on this accelerator; see the note in
# accels/cann.sh for why it is asked separately from the GraspGen stack.
accel_supports_torch_graspgenx() {
    # Its only deployments are cuda and npu, so a CPU-only host has no backend
    # to serve it even though the wheel itself would install.
    return 1
}

# Workspace pip constraint file that protects this accelerator's Torch ABI
# during every dependency install, or no output when none is needed.
accel_pip_constraints() {
    :
}

# Workspace requirements file holding this accelerator's own Python
# dependencies, or no output when it has none.
accel_requirements() {
    # The CPU profile adds nothing: Torch comes from the lerobot editable
    # install, and no vendor runtime is involved.
    :
}

# Post-install validation of the accelerator runtime. $1 is the venv Python.
accel_verify_runtime() {
    :
}

# Runs right after accel_requirements, with the pip runner as "$@". Use it to
# remove packages LeRobot's own resolution installs that the accelerator's
# Torch ABI cannot support.
accel_post_requirements() {
    # The CPU profile installs no accelerator Torch, so nothing to remove.
    :
}
