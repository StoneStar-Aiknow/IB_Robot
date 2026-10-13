#!/bin/bash

# NVIDIA CUDA accelerator profile. Sourced by detect.sh::load_accel_impl when
# SETUP_ACCEL_ID is "cuda". Every hook below is part of the accelerator
# contract consumed by python_venv.sh; keep the three profiles in sync.

# Whether the Torch GraspGen stack (grasp_gen plus the pointnet2_ops CUDA
# extension) can be installed on this accelerator.
accel_supports_torch_grasp() {
    return 0
}

# Workspace pip constraint file that protects this accelerator's Torch ABI
# during every dependency install, or no output when none is needed.
accel_pip_constraints() {
    # CUDA hosts follow the Torch build selected by lerobot, narrowed to the
    # pointnet2_ops wheel ABI by install_graspgen_torch_abi when grasp is
    # installed. No global constraint is required.
    :
}

# Workspace requirements file holding this accelerator's own Python
# dependencies, or no output when it has none.
accel_requirements() {
    # The CUDA Torch build comes from the lerobot editable install plus the
    # pointnet2_ops ABI pin, not from a requirements file of its own.
    :
}

# Post-install validation of the accelerator runtime. $1 is the venv Python.
accel_verify_runtime() {
    # Deliberately not a torch.cuda.is_available() probe: --accel cuda is also
    # used to install the CUDA dependency set in containers without --gpus,
    # where no device is visible. The benchmark profile, which does require a
    # driver, runs its own CUDA smoke test.
    :
}

# Runs right after accel_requirements, with the pip runner as "$@". Use it to
# remove packages LeRobot's own resolution installs that the accelerator's
# Torch ABI cannot support.
accel_post_requirements() {
    # CUDA follows LeRobot's own resolution; nothing to remove afterwards.
    :
}
