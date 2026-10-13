#!/bin/bash
# python_venv.sh - Python environment setup, dependency installation, and lerobot management

check_lerobot_python_compat() {
    local toml_path="${WORKSPACE}/libs/lerobot/pyproject.toml"
    local required_python=">=3.10"
    
    if [[ -f "${toml_path}" ]]; then
        local extracted_req
        extracted_req=$(grep -oP '^requires-python\s*=\s*"\K[^"]+' "${toml_path}" || true)
        if [[ -n "${extracted_req}" ]]; then
            required_python="${extracted_req}"
        fi
    fi

    if ! "${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
req = '${required_python}'
if not req:
    sys.exit(0)
spec = SpecifierSet(req)
if version not in spec:
    print(f'ERROR: Python {version} does not satisfy lerobot requirement {req}')
    sys.exit(1)
" 2>/dev/null; then
        local err_msg
        err_msg=$("${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
req = '${required_python}'
if not req:
    sys.exit(0)
spec = SpecifierSet(req)
if version not in spec:
    print(f'ERROR: Python {version} does not satisfy lerobot requirement {req}')
    sys.exit(1)
" 2>&1)
        log_error "${err_msg}"
        return 1
    fi
    return 0
}

check_lerobot_ros_numpy_compat() {
    if ! "${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
spec = SpecifierSet('<3.12')
if version not in spec:
    print(f'WARN: Python {version} detected. ROS 2 Humble binary packages (cv_bridge, image_transport)')
    print('WARN: are compiled against NumPy 1.x. Python 3.12+ typically requires NumPy 2.x, which')
    print('WARN: breaks ABI compatibility. C++ extensions may crash at runtime.')
" 2>/dev/null; then
        local warn_msg
        warn_msg=$("${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
spec = SpecifierSet('<3.12')
if version not in spec:
    print(f'WARN: Python {version} detected. ROS 2 Humble binary packages (cv_bridge, image_transport)')
    print('WARN: are compiled against NumPy 1.x. Python 3.12+ typically requires NumPy 2.x, which')
    print('WARN: breaks ABI compatibility. C++ extensions may crash at runtime.')
" 2>&1)
        while IFS= read -r line; do
            log_warn "${line}"
        done <<< "${warn_msg}"
    fi
    return 0
}

# detect_cann_version lives in setup/detect.sh: it feeds the accelerator axis
# (SETUP_ACCEL_ID), which has to be resolved before any Torch build is chosen.
# The Torch ABI that CANN dictates is no longer selected here: it belongs to the
# accelerator axis, so accels/cann.sh picks it from requirements/accel-cann-*.txt
# and install_accel_requirements installs it on whichever OS carries the NPU.
install_openeuler_python_dependencies() {
    local pip_runner=("$@")

    run_cmd "${pip_runner[@]}" -r "${WORKSPACE}/requirements/openeuler-24.03.txt" --quiet
}

# Accelerator-axis Python dependencies, installed on every OS. The OS axis
# installs its own list separately; neither may carry the other's packages.
install_accel_requirements() {
    local pip_runner=("$@")
    local requirements_file rc=0

    # Capture the status explicitly: a failing command substitution inside a
    # bare assignment does not trip errexit, and the hook contract still
    # allows an accelerator to refuse outright (a broken runtime, not merely
    # an unvalidated release).
    requirements_file="$(accel_requirements)" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        log_error "Accelerator ${SETUP_ACCEL_ID} cannot select its Python dependencies; see above."
        return 1
    fi
    if [[ -z "${requirements_file}" ]]; then
        # An empty result means "no accelerator packages to install". For cann
        # that is a deferral, not a no-op: either no CANN toolkit was detected
        # or the detected release has no validated Torch ABI here, so no
        # torch_npu landed in the venv. The hook ran inside the $( ) subshell
        # above and cannot set variables here, so the CALLER records the
        # pending state from the empty result; setup.sh's end summary reads it
        # and prints the manual-configuration guidance.
        if [[ "${SETUP_ACCEL_ID}" == "cann" ]]; then
            ACCEL_NPU_PENDING_MANUAL_CONFIG=true
            log_warn "No NPU backend installed for cann; NPU inference stays unavailable until it is configured manually."
        else
            log_info "No accelerator-specific Python dependencies for ${SETUP_ACCEL_ID}."
        fi
        return 0
    fi
    log_info "Installing ${SETUP_ACCEL_ID} accelerator dependencies..."
    run_cmd "${pip_runner[@]}" -r "${requirements_file}" --quiet
}

# The CANN release selects the exact Torch triple it requires. 8.1 is
# hardware-validated on Ascend310P1; 8.3 follows Ascend's compatibility table
# (torch_npu 2.8.0 pairs with CANN 8.3.RC1 and requires exactly torch 2.8.0,
# torchvision 0.23.0 matches).
verify_cann_torch_abi() {
    local python_path="$1"
    local cann_version="" release torch_pin torch_npu_pin torchvision_pin

    cann_version="$(detect_cann_version 2>/dev/null || true)"
    case "${cann_version}" in
        8.1*) release="8.1"; torch_pin="2.5.1"; torch_npu_pin="2.5.1"; torchvision_pin="0.20.1" ;;
        8.3*) release="8.3"; torch_pin="2.8.0"; torch_npu_pin="2.8.0"; torchvision_pin="0.23.0" ;;
        *) return 0 ;;
    esac
    "${python_path}" - "${release}" "${torch_pin}" "${torch_npu_pin}" "${torchvision_pin}" <<'PY'
import sys

release, torch_pin, torch_npu_pin, torchvision_pin = sys.argv[1:5]
import torch
import torch_npu
import torchvision

expected = {"torch": torch_pin, "torch_npu": torch_npu_pin, "torchvision": torchvision_pin}
actual = {
    "torch": torch.__version__.split("+", maxsplit=1)[0],
    "torch_npu": torch_npu.__version__.split("+", maxsplit=1)[0],
    "torchvision": torchvision.__version__.split("+", maxsplit=1)[0],
}
if actual != expected:
    raise SystemExit(f"CANN {release} requires the exact Torch ABI {expected}, got {actual}")
print(f"CANN {release} Torch ABI verified: {actual}")
PY
}

verify_cann_runtime_versions() {
    local python_path="$1"
    local cann_version="" release
    cann_version="$(detect_cann_version 2>/dev/null || true)"
    case "${cann_version}" in
        8.1*) release="8.1" ;;
        8.3*) release="8.3" ;;
        *) return 0 ;;
    esac
    verify_cann_torch_abi "${python_path}" || return 1
    "${python_path}" - "${release}" "${WORKSPACE}/requirements/constraints-cann-${release}.txt" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement

release, constraints_path = sys.argv[1], Path(sys.argv[2])
for raw in constraints_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        # The GUI wheel is optional; the headless wheel is installed by compat.
        if requirement.name == "opencv-python":
            continue
        raise
    if installed not in requirement.specifier:
        raise SystemExit(f"CANN {release} requires {requirement}, got {installed}")
    print(f"CANN {release} runtime verified: {requirement.name}=={installed}")
numpy_version = importlib.metadata.version("numpy")
if numpy_version != "1.26.4":
    raise SystemExit(f"CANN {release} final ROS ABI requires numpy==1.26.4, got {numpy_version}")
PY
}

install_lerobot_editable() {
    local pip_runner=("$@")

    if ! check_lerobot_python_compat; then
        log_error "Cannot install lerobot: Python version is incompatible."
        log_error "Ensure patches 0001/0002 from scripts/setup/lerobot_patches.sh are applied"
        log_error "to downgrade the requirement to >=3.10 if you are on an older system."
        exit 1
    fi

    check_lerobot_ros_numpy_compat

    # [smolvla,pi] extras pull in policy-specific deps; kinematics pulls in
    # placo for SO-101 Placo Cartesian teleop; diffusion pulls in diffusers
    # for Diffusion Policy training/inference; dataset pulls in datasets +
    # torchcodec (video decoding) for training/dataset loading; deepdiff-dep
    # supplies the deepdiff package lerobot's motors_bus needs (v0.6.0+). We
    # deliberately use deepdiff-dep rather than the feetech extra because
    # so101_hardware already provides the Python feetech-servo-sdk via its
    # setup.py install_requires, and the C++ ftservo_sdk is built by
    # so101_hardware's CMake for the ros2_control node — neither should be
    # re-installed via pip here. See libs/lerobot/pyproject.toml. LIBERO is
    # Ubuntu-only and opt-in. Remove the legacy editable ``libero``
    # distribution before installing hf-libero so the two providers never
    # coexist under the same top-level Python package.
    local lerobot_extras=(smolvla pi diffusion dataset deepdiff-dep)
    if [[ "${SETUP_PROFILE:-full}" != "inference" ]]; then
        lerobot_extras+=(kinematics)
    fi
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID:-unknown}" == "ubuntu-22.04" ]]; then
        log_info "Removing the legacy libs/libero editable provider, if installed..."
        "${pip_runner[@]}" uninstall -y libero >/dev/null 2>&1 || true
        lerobot_extras+=(libero)
        log_info "Enabling the local LeRobot libero extra for Ubuntu benchmark setup."
    fi

    local lerobot_extras_csv
    lerobot_extras_csv="$(IFS=,; printf '%s' "${lerobot_extras[*]}")"
    local constraint_args=()
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && -n "${BENCHMARK_PIP_CONSTRAINTS:-}" ]]; then
        constraint_args+=(--constraint "${BENCHMARK_PIP_CONSTRAINTS}")
    fi
    run_cmd "${pip_runner[@]}" install "${constraint_args[@]}" -e \
        "${WORKSPACE}/libs/lerobot[${lerobot_extras_csv}]"
}

# TorchCodec release matching the pinned Torch 2.7.1 ABI. Upstream's
# compatibility table pairs 0.4/0.5 with torch 2.7 and 0.6/0.7 with torch 2.8;
# 0.5 is also the version the benchmark profile validates for the same Torch
# build (BENCHMARK_TORCHCODEC_VERSION in benchmark_profile.sh). Keep both on
# the same version for as long as they ride the same Torch.
GRASPGEN_TORCHCODEC_VERSION="0.5"

install_graspgen_torch_abi() {
    # Both wheel paths use this exact ABI. A skip plan must leave the runtime
    # alone, especially on GPUs that the pinned Torch itself cannot run on.
    case "${1:-skip:no GraspGen install plan}" in
        ok:*|build:*) ;;
        *) return 0 ;;
    esac

    local torch_version torchvision_version torchcodec_version
    torch_version="$("${VENV_PYTHON}" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
    torchvision_version="$("${VENV_PYTHON}" -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || true)"
    if [[ "${torch_version}" != "2.7.1+cu126" || "${torchvision_version}" != "0.22.1+cu126" ]]; then
        log_info "Pinning Torch 2.7.1+cu126 for the precompiled pointnet2_ops ABI..."
        run_cmd "${VENV_PYTHON}" -m pip install --force-reinstall \
            "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" \
            --index-url "https://download.pytorch.org/whl/cu126" \
            --extra-index-url "${SETUP_PIP_INDEX_URL}" --quiet
    fi

    # torchcodec is ABI-bound to a Torch minor but declares no torch
    # requirement, so pip cannot see the conflict: LeRobot's [dataset] extra
    # allows >=0.3.0,<0.12.0 and the resolver takes the newest, which links
    # against a newer Torch (0.11.x wants libnvrtc.so.13, i.e. the CUDA 13
    # runtime shipped with Torch 2.11). Install the matching version so
    # LeRobot's own resolution finds the requirement already satisfied and
    # leaves it alone. This check deliberately runs on its own instead of
    # behind the Torch pin above: environments set up before it existed
    # (master's no-nvcc pin path) already carry 2.7.1+cu126 with a
    # resolver-picked torchcodec, and a re-run must heal them rather than
    # short-circuit past them.
    torchcodec_version="$("${VENV_PYTHON}" -c 'import importlib.metadata as m; print(m.version("torchcodec"))' 2>/dev/null || true)"
    if [[ "${torchcodec_version}" != "${GRASPGEN_TORCHCODEC_VERSION}" ]]; then
        log_info "Pinning torchcodec ${GRASPGEN_TORCHCODEC_VERSION} for the Torch 2.7.1 ABI..."
        # --no-deps, like the benchmark runtime ABI install: re-resolving here
        # could replace either side of the ABI contract with an incompatible
        # release.
        run_cmd "${VENV_PYTHON}" -m pip install --force-reinstall --no-deps \
            "torchcodec==${GRASPGEN_TORCHCODEC_VERSION}" --index-url "${SETUP_PIP_INDEX_URL}" --quiet
    fi
    # The ABI cannot be expressed as a constraint for pip to enforce, so verify
    # unconditionally — also on hosts where both versions already match — that
    # the extension loads and exposes its public decoder API.
    if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import torchcodec._core; from torchcodec.decoders import VideoDecoder' >/dev/null 2>&1; then
        log_error "torchcodec ${GRASPGEN_TORCHCODEC_VERSION} does not load against the pinned Torch 2.7.1+cu126."
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import torchcodec._core; from torchcodec.decoders import VideoDecoder' || true
        return 1
    fi
}

setup_python_venv() {
    if ! platform_supports_local_workspace_build; then
        log_info "Skipping workspace venv setup on ${SETUP_PLATFORM_ID}."
        log_info "Use the board ROS runtime directly after sourcing $(platform_ros_setup_path)."
        log_info "Cross-build RoboFrame (IB_Robot) OpenHarmony artifacts on the host with scripts/openharmony/build_roboframe_oh.sh."
        PYTHON_ENV_STATUS="skipped"
        log_skipped "Workspace Python virtual environment"
        return 0
    fi

    local venv_path="${WORKSPACE}/venv"
    local lerobot_dir="${WORKSPACE}/libs/lerobot"

    # Convenience flag for profile gating below: the inference profile skips
    # every dependency group that is not required to load policy bundles and
    # serve inference requests (hardware, teleop, perception, grasp, voice,
    # sim, dev tooling).
    local full_profile=true
    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        full_profile=false
    fi

    # 0. Python interpreter preflight
    local host_python_path host_python_version host_py_major host_py_minor
    host_python_path="$(command -v python3 || true)"
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        host_python_path="${SETUP_BOOTSTRAP_PYTHON_BIN:-${host_python_path}}"
    fi
    if [[ -z "${host_python_path}" ]]; then
        log_error "No bootstrap Python was selected. Install a platform-supported Python before running setup.sh."
        exit 1
    fi
    host_python_version="$(${host_python_path} -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
    log_info "Using setup bootstrap Python: ${host_python_path} (version ${host_python_version})"
    host_py_major="$(${host_python_path} -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
    host_py_minor="$(${host_python_path} -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
    if (( host_py_major < 3 )) || { (( host_py_major == 3 )) && (( host_py_minor < 10 )); }; then
        log_error "Python ${host_python_version} is too old. setup.sh requires Python >= 3.10."
        log_error "On openEuler: 'sudo dnf install -y python3.10 python3.10-devel' and re-run."
        exit 1
    fi

    # Ensure system-level venv tools are installed (done by platform scripts during install_system_deps)

    # Create the virtual environment (must include --system-site-packages to use system rclpy)
    if [[ ! -d "${venv_path}" ]]; then
        run_with_live_output "Creating virtual environment at ${venv_path} with --system-site-packages..." "${SETUP_BOOTSTRAP_PYTHON_BIN:-python3}" -m venv --system-site-packages "${venv_path}"
    else
        log_info "Virtual environment already exists at ${venv_path}."
    fi
    touch "${venv_path}/COLCON_IGNORE"

    # Activate the virtual environment and install dependencies
    log_info "Configuring Python environment and dependencies..."
    source "${venv_path}/bin/activate"

    if [[ -n "${PYTHONPATH:-}" ]]; then
        log_warn "Clearing inherited PYTHONPATH for isolated package installation inside ${venv_path}."
        unset PYTHONPATH
    fi

    if [[ -n "${PYTHONHOME:-}" ]]; then
        log_warn "Clearing inherited PYTHONHOME for isolated package installation inside ${venv_path}."
        unset PYTHONHOME
    fi

    if [[ -z "${VENV_PYTHON}" ]]; then
        log_error "No working virtual environment python was found under ${venv_path}/bin."
        exit 1
    fi

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        local bootstrap_python_mm venv_python_mm
        bootstrap_python_mm="$(${host_python_path} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        venv_python_mm="$(${VENV_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "${venv_python_mm}" != "${bootstrap_python_mm}" ]]; then
            log_error "Existing workspace venv uses Python ${venv_python_mm}, but Benchmark setup requires Python ${bootstrap_python_mm}."
            log_error "Remove ${venv_path} and rerun setup; setup will not mutate a foreign-ABI venv in place."
            exit 1
        fi
    fi

    # A function-local exported constraint applies to all pip invocations and
    # build subprocesses, and is restored on return. CUDA activates its GraspGen
    # ABI constraint before LeRobot resolves only when the preflight confirms
    # that pointnet2_ops will be installed. Other accelerators resolve LeRobot
    # first, install their runtime override, and activate their constraint after
    # that override; this preserves the CANN 8.1 metadata exception.
    local -x PIP_CONSTRAINT="${PIP_CONSTRAINT:-}"
    local pip_install=("${VENV_PYTHON}" -m pip install)
    local ros_abi_constraints="${venv_path}/ros_abi_constraints.txt"
    local ros_abi_pin_packages=("numpy==1.26.4" "opencv-python-headless<4.12")
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "ubuntu-22.04" ]]; then
        # hf-libero requires the GUI OpenCV distribution. Keep both wheels on
        # the same NumPy-1-compatible line in benchmark environments.
        ros_abi_pin_packages+=("opencv-python<4.12")
    fi

    # Upgrade pip
    run_cmd "${VENV_PYTHON}" -m pip install --upgrade pip --quiet

    # Pin and force-reinstall setuptools to a version that satisfies both
    # LeRobot (>=71,<81) and colcon-core (<80), while retaining the legacy
    # `setup.py develop --editable` option used by colcon's symlink install.
    run_cmd "${VENV_PYTHON}" -m pip install --force-reinstall "setuptools==75.8.2" --quiet

    # ------------------------------------------------------------------
    # LeRobot patch stack — MUST run before install_lerobot_editable.
    #
    # Why here: install_lerobot_editable invokes check_lerobot_python_compat
    # which reads libs/lerobot/pyproject.toml. On a fresh clone with py3.10/py3.11
    # hosts, pyproject.toml carries upstream's `requires-python>=3.12`
    # until the patch stack lowers it. Running patches first ensures
    # the compat gate reads the patched pyproject.toml.
    # ------------------------------------------------------------------
    log_info "Installing PyYAML for the lerobot patch dispatcher..."
    run_cmd "${VENV_PYTHON}" -m pip install pyyaml --quiet
    if [[ -d "${lerobot_dir}" ]]; then
        ensure_lerobot_patch_stack_applied
    fi

    # Preflight before any GraspGen Torch pin or LeRobot dependency resolution.
    # The same plan drives the later wheel install so a skipped stack never
    # downgrades the runtime merely for an extension that will not be installed.
    local pointnet2ops_plan="skip:GraspGen is disabled for this setup profile"
    if [[ "${full_profile}" == true && "${INSTALL_BENCHMARK_DEPS:-false}" != true ]] &&
        accel_supports_torch_grasp; then
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/setup/install_graspgen_pip.sh"
        unset GRASPGEN_POINTNET2OPS_WHEEL
        pointnet2ops_plan="$(graspgen_pointnet2ops_plan)"
    fi

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "ubuntu-22.04" ]]; then
        benchmark_prepare_torch_profile "${VENV_PYTHON}" "${venv_path}/benchmark_torch_constraints.txt"
        # Protect the validated Torch/TorchVision/TorchCodec ABI during every
        # subsequent dependency install, not only the LeRobot editable step.
        pip_install+=(--constraint "${BENCHMARK_PIP_CONSTRAINTS}")
    else
        install_graspgen_torch_abi "${pointnet2ops_plan}"
        if [[ "${pointnet2ops_plan%%:*}" != "skip" ]]; then
            local grasp_torch_constraints="${venv_path}/graspgen_torch_constraints.txt"
            printf '%s\n' 'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126' \
                "torchcodec==${GRASPGEN_TORCHCODEC_VERSION}" > "${grasp_torch_constraints}"
            PIP_CONSTRAINT="${grasp_torch_constraints}${PIP_CONSTRAINT:+ ${PIP_CONSTRAINT}}"
        fi
    fi

    # Install LeRobot in editable mode
    # Note: Do not pass the -c numpy==1.26.4 constraint. The lerobot dependency graph
    # (rerun-sdk, opencv, datasets, etc.) resolves under numpy>=2. A hard constraint
    # causes pip to fail with resolution-too-deep. We allow numpy 2.x here, then force-reinstall 1.26.4 + opencv<4.12 at the end.
    if [[ -d "${lerobot_dir}" ]]; then
        log_info "Installing LeRobot in editable mode with kinematics extra..."
        install_lerobot_editable "${VENV_PYTHON}" -m pip
    fi

    # Accelerator-axis dependencies, on every OS. torch_npu used to live in the
    # openEuler list, which is why an Ubuntu host with an Ascend NPU could never
    # get it; the axis decides this now, not the OS.
    #
    # This runs immediately after LeRobot so the override lands on top of
    # LeRobot's resolution, and the constraint is activated right after it so no
    # later step can replace the selected ABI.
    install_accel_requirements "${pip_install[@]}"
    accel_post_requirements "${pip_install[@]}"

    local accel_constraints
    accel_constraints="$(accel_pip_constraints)"
    if [[ -n "${accel_constraints}" ]]; then
        PIP_CONSTRAINT="${accel_constraints}${PIP_CONSTRAINT:+ ${PIP_CONSTRAINT}}"
        log_info "Protecting the validated ${SETUP_ACCEL_ID} Torch runtime for every later dependency install."
    fi

    # Install base Python dependencies
    log_info "Installing base Python dependencies..."
    run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/base.txt" --quiet

    # Install hardware dependencies
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing hardware dependencies..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/hardware.txt" --quiet
    else
        log_info "Skipping hardware dependencies (inference profile)."
    fi

    # Install the optional built-in WebPhone dependency
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping WebPhone teleoperation dependency (inference profile)."
    elif [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes mode: installing WebPhone dependency (websockets)..."
        run_cmd "${pip_install[@]}" websockets --quiet
        log_done "Phone teleoperation dependency installed (websockets)"
    else
        echo ""
        echo "  WebPhone teleoperation (optional):"
        echo "    1) Install websockets (browser WebXR AR + optical-flow fallback)"
        echo "    0) Skip WebPhone"
        echo ""
        while true; do
            read -r -p "  Enter your choice [0-1]: " PHONE_CHOICE
            case "${PHONE_CHOICE}" in
                1)
                    run_cmd "${pip_install[@]}" websockets --quiet
                    log_done "Phone dependencies installed: websockets (WebPhone)"
                    break
                    ;;
                0)
                    log_info "Skipping WebPhone teleoperation dependency."
                    break
                    ;;
                *)
                    echo "  Invalid choice. Please enter 0 or 1."
                    ;;
            esac
        done
    fi

    # Install development and training tools
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing dev-tools (tensorboard, rerun, gitlint, ruff, pre-commit)..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/dev-tools.txt" --quiet
    else
        log_info "Skipping dev-tools (inference profile)."
    fi

    # Install OS-axis dependencies (ONNX tooling, etc.)
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if [[ "${full_profile}" == true ]]; then
                run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/ubuntu-22.04.txt" --quiet
            else
                # Inference profile: only the ONNX toolchain; skip mujoco (sim).
                # AtomGit collaboration now uses the official CLI (no pip package).
                log_info "Installing inference ONNX tooling (inference profile)..."
                run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/inference.txt" --quiet
            fi
            ;;
        openeuler-embedded-24.03)
            # Installed whole in both profiles: onnx/onnxruntime (ONNX policy
            # path) and pygraphviz (required by verify_env on this platform)
            # are both inference-relevant.
            install_openeuler_python_dependencies "${pip_install[@]}"
            ;;
    esac

    if [[ "${full_profile}" == true ]] && [[ -f "${WORKSPACE}/.pre-commit-config.yaml" ]]; then
        "${VENV_PYTHON}" -m pre_commit install
    fi

    # ------------------------------------------------------------------
    # Environment Patches & Overrides
    # ------------------------------------------------------------------
    
    log_info "Installing colcon-common-extensions + colcon-mixin into the workspace venv..."
    run_cmd "${pip_install[@]}" --ignore-installed --upgrade colcon-common-extensions colcon-mixin --quiet

    if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import setuptools.command.develop as develop

raise SystemExit(
    0 if hasattr(develop, "develop") and hasattr(develop.develop, "install_for_development")
    else 1
)
PY
    then
        log_warn "colcon symlink installation will fail because setuptools>=71 removed setup.py develop."
        log_warn "Please ensure setuptools is downgraded or wait for colcon-core updates."
    fi

    log_info "Pinning Empy 3.3.4 for ROS 2 Humble rosidl compatibility..."
    run_cmd "${pip_install[@]}" --force-reinstall "empy==3.3.4" --quiet

    # rosdep was already installed into this same venv by the early
    # ensure_workspace_venv + ensure_rosdep step. Re-running pip install
    # here is a no-op when the package is current, and acts as a safety net
    # in case the venv was recreated between the two steps.
    log_info "Ensuring rosdep is present in the workspace venv..."
    run_cmd "${pip_install[@]}" rosdep --quiet

    # Force NumPy/OpenCV back to ROS 2 Humble ABI-compatible versions.
    # The lerobot installation brings in numpy 2.x. We unconditionally overwrite it
    # here to ensure ROS packages (cv_bridge, image_transport, etc.) do not trigger binary incompatibility errors at runtime.
    # Only install the headless OpenCV wheel by default; keep opencv-python in
    # the constraints file below so optional dependencies cannot pull 4.12+ and
    # force NumPy 2.x back into the ROS environment.
    log_info "Pinning NumPy/OpenCV to the ROS 2 Humble ABI-compatible versions..."
    run_cmd "${pip_install[@]}" --force-reinstall "${ros_abi_pin_packages[@]}" --quiet
    benchmark_install_runtime_abi "${VENV_PYTHON}" -m pip install

    cat > "${ros_abi_constraints}" <<'EOF'
numpy==1.26.4
opencv-python<4.12
opencv-python-headless<4.12
EOF
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        cat "${BENCHMARK_PIP_CONSTRAINTS}" >> "${ros_abi_constraints}"
    fi

    # Install the ZipVoice frontend after creating the ROS ABI constraints.
    # Vocos itself is maintained in voice_tts_service.vocos_backend because
    # the PyPI package can replace the workspace Torch/torchaudio ABI.
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing voice TTS frontend dependencies..."
        run_cmd "${pip_install[@]}" --constraint "${ros_abi_constraints}" \
            -r "${WORKSPACE}/requirements/voice-tts.txt" --quiet
    else
        log_info "Skipping voice TTS frontend dependencies (inference profile)."
    fi

    # Perception runtime dependencies (SAM2, Grounding-DINO, RAM++, SigLIP2) are
    # part of the default install contract: perception_service and
    # semantic_mapping already build in the default workspace, and the audited
    # RAM++ / GroundingDINO wheels ship in third_party/. Install them on every
    # platform that runs the local workspace build, including openEuler Embedded
    # (the Ascend OM path does not replace the Torch perception runtime).
    # The inference profile skips them: edge-cloud inference verification uses
    # the perception ROS mock instead of the real perception stack.
    local ram_wheel_root="${WORKSPACE}/third_party/wheels/recognize-anything/7cb804a"
    local ram_wheel="${ram_wheel_root}/ibrobot_ram-0.0.1+ibrobot.1-py3-none-any.whl"
    local gdino_wheel_root="${WORKSPACE}/third_party/wheels/groundingdino/313392a"
    local gdino_wheel="${gdino_wheel_root}/ibrobot_groundingdino-0.1.0+ibrobot.1-py3-none-any.whl"
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping perception dependencies (inference profile)."
    else
        log_info "Installing perception dependencies (SAM2, Grounding-DINO, RAM++, SigLIP2)..."
        run_cmd env SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" SAM2_BUILD_ALLOW_ERRORS=1 \
            "${pip_install[@]}" --no-build-isolation --constraint "${ros_abi_constraints}" \
            -r "${WORKSPACE}/requirements/perception.txt" --quiet
        if ! (cd "${ram_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "RAM++ wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${ram_wheel}" --quiet
        if ! (cd "${gdino_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "GroundingDINO wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${gdino_wheel}" --quiet
    fi

    # FullSubNet speech enhancement model (audio_zen + model.py) as an audited
    # wheel. Pure-Python; the Torch backend loads Model from the installed
    # package instead of cloning the upstream source tree into models/.
    local fullsubnet_wheel_root="${WORKSPACE}/third_party/wheels/fullsubnet/e97448375"
    local fullsubnet_wheel="${fullsubnet_wheel_root}/ibrobot_fullsubnet-0.0.1+ibrobot.1-py3-none-any.whl"
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping FullSubNet wheel (inference profile)."
    elif [[ -f "${fullsubnet_wheel_root}/SHA256SUMS" ]]; then
        if ! (cd "${fullsubnet_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "FullSubNet wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${fullsubnet_wheel}" --quiet
    else
        log_warn "FullSubNet wheel not found at ${fullsubnet_wheel_root}; skipping (speech_direction Torch backend unavailable)."
    fi

    # GraspGen and GraspGenX runtime dependencies are part of the default
    # install contract: manipulation_service is part of the default workspace
    # build. The stack is gated on the accelerator, not the OS: pointnet2_ops is
    # a CUDA extension, and the Ascend grasp path is the OM runtime instead.
    # install_graspgen_pip installs the audited grasp_gen and pointnet2_ops
    # wheels and validates the latter's Torch/Python ABI contract.
    # Record whether grasp was actually installed and let the smoke test below
    # key off that fact rather than re-deriving the condition. Re-deriving is
    # how the smoke test kept an OS test after the install moved to the
    # accelerator axis, which made a full setup on Ubuntu + Ascend or Ubuntu +
    # CPU skip the install and then fail importing grasp_gen.
    local grasp_installed=false
    local graspgenx_wheel_root="${WORKSPACE}/third_party/wheels/graspgenx/b942909"
    local graspgenx_wheel="${graspgenx_wheel_root}/graspgenx-1.0.0+ibrobot.2-py3-none-any.whl"
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping grasp dependencies (inference profile)."
    elif [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        log_info "Skipping GraspGen/manipulation dependencies for Benchmark-only setup."
    elif ! accel_supports_torch_grasp; then
        log_warn "Skipping grasp dependencies on accelerator ${SETUP_ACCEL_ID}; Torch GraspGen needs CUDA."
        log_warn "Re-run with --accel cuda to install them regardless, e.g. on a host with a CUDA toolkit but no visible GPU."
    else
        local pointnet2ops_action pointnet2ops_arg
        pointnet2ops_action="${pointnet2ops_plan%%:*}"
        pointnet2ops_arg="${pointnet2ops_plan#*:}"
        local install_grasp=true
        if [[ "${pointnet2ops_action}" == "skip" ]]; then
            log_warn "Skipping grasp dependencies: ${pointnet2ops_arg}"
            install_grasp=false
        elif [[ "${pointnet2ops_action}" == "build" ]] && ! graspgen_pointnet2ops_build "${pointnet2ops_arg}"; then
            # The build is best-effort: it needs GitHub and a working CUDA
            # toolchain, and its failure must degrade to a skipped grasp
            # instead of aborting the whole setup.
            log_warn "Skipping grasp dependencies: the local pointnet2_ops wheel build failed; resolve the error above and re-run setup."
            install_grasp=false
        fi
        if [[ "${install_grasp}" == true ]]; then
            log_info "Installing grasp dependencies (GraspGen)..."
            export ROS_ABI_CONSTRAINTS="${ros_abi_constraints}"
            install_graspgen_pip "${VENV_PYTHON}" -m pip install
            # GraspGenX ships as an audited pure-Python wheel built from its pinned
            # upstream commit (scripts/build_graspgenx_wheel.sh). --no-deps: upstream
            # pins torch<2.7, diffusers==0.11.1 and huggingface-hub==0.25.2 only for
            # its authors' driver; it reuses GraspGen's pointnet2_ops and runtime
            # requirements plus requirements/graspgenx.txt, so it is installed, or
            # skipped, together with GraspGen.
            log_info "Installing grasp dependencies (GraspGenX)..."
            run_cmd "${pip_install[@]}" --constraint "${ros_abi_constraints}" \
                -r "${WORKSPACE}/requirements/graspgenx.txt" --quiet
            if ! (cd "${graspgenx_wheel_root}" && sha256sum --check SHA256SUMS); then
                log_error "GraspGenX wheel checksum verification failed."
                return 1
            fi
            run_cmd "${pip_install[@]}" --no-deps "${graspgenx_wheel}" --quiet
            grasp_installed=true
        fi
    fi

    # Optional speech_direction report dependencies (Plotly/Matplotlib). Pure-Python.
    # Only the `speech_direction_report` offline CLI imports these; the runtime node
    # does not, so the default install (without this flag) stays lightweight.
    if [[ "${INSTALL_DIAGNOSTICS_DEPS:-false}" == true ]]; then
        log_info "Installing optional speech_direction report dependencies (Plotly, Matplotlib)..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/diagnostics.txt" --quiet
        installed_diagnostics_deps=true
    else
        log_info "Skipping optional speech_direction report dependencies. Re-run setup with --with-diagnostics if needed."
    fi

    # ------------------------------------------------------------------
    # Benchmark (LIBERO) optional resource configuration.
    #
    # The conditional LeRobot libero extra above installs the sole hf-libero
    # provider. This layer only prepares the workspace-owned config/assets;
    # provider identity and runtime API checks belong to the LIBERO adapter.
    # ------------------------------------------------------------------
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        if [[ "${SETUP_PLATFORM_ID}" != "ubuntu-22.04" ]]; then
            log_warn "Skipping optional benchmark dependencies on ${SETUP_PLATFORM_ID}; LIBERO/MuJoCo are validated on Ubuntu only."
        else
            log_info "Configuring optional benchmark (LIBERO) runtime..."
            export WORKSPACE="${WORKSPACE}"

            # Generate workspace-owned config and fetch/reuse official assets
            # before importing hf-libero. This prevents the provider's legacy
            # first-import prompt and makes get_assets_path() deterministic.
            local libero_config_script="${SCRIPT_DIR}/setup/libero_config.py"
            local libero_config_dir="${venv_path}/ibrobot_libero"
            log_info "Preparing workspace-owned hf-libero config and assets..."
            if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${libero_config_script}" \
                > "${venv_path}/benchmark_libero_config_output.txt" 2>&1; then
                log_error "Failed to prepare workspace-owned hf-libero config/assets."
                cat "${venv_path}/benchmark_libero_config_output.txt" 2>/dev/null || true
                exit 1
            fi
            export LIBERO_CONFIG_PATH="${libero_config_dir}"
            log_info "hf-libero config/assets ready at ${libero_config_dir}"
        fi
    fi

    # Optional perception/grasp dependencies can pull OpenCV wheels whose latest
    # releases require NumPy 2.x. Re-apply the final ROS 2 ABI pin before smoke tests.
    log_info "Re-applying NumPy/OpenCV ROS 2 ABI pins after optional dependencies..."
    run_cmd "${pip_install[@]}" --force-reinstall "${ros_abi_pin_packages[@]}" --quiet

    log_info "Running NumPy/OpenCV dependency smoke test..."
    PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import cv2
import numpy

if not numpy.__version__.startswith("1.26"):
    raise SystemExit(f"Expected NumPy 1.26.x after setup, got {numpy.__version__}")
print(f"NumPy/OpenCV smoke test passed: numpy={numpy.__version__}, cv2={cv2.__version__}")
PY
    # Only assert the import when this run actually installed GraspGen.
    if [[ "${grasp_installed}" == true ]]; then
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib

try:
    importlib.import_module("grasp_gen")
except Exception as exc:
    raise SystemExit(f"Missing grasp_gen after default grasp install: {exc}")
try:
    importlib.import_module("pointnet2_ops")
except Exception as exc:
    raise SystemExit(f"Missing pointnet2_ops after default grasp install: {exc}")
print("Grasp dependencies smoke test passed")
PY
    fi
    # gitlint ships in dev-tools.txt, which the inference profile skips.
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping gitlint commit-msg hook (inference profile)."
    else
        local commit_msg_hook
        commit_msg_hook="$(git rev-parse --git-path hooks/commit-msg 2>/dev/null || true)"
        if [[ -f "${commit_msg_hook}" ]] && grep -qi "gitlint" "${commit_msg_hook}"; then
            log_warn "gitlint commit-msg hook already exists at ${commit_msg_hook}; keeping it."
        else
            log_info "Installing gitlint commit-msg hook..."
            # gitlint is installed in the venv, which is currently activated
            printf 'y\n' | gitlint install-hook || log_warn "Failed to install gitlint hook"
        fi
    fi

    # Venv summary: print the key facts users need to debug "wrong python /
    # wrong colcon" issues without having to source the venv themselves.
    local venv_numpy_ver="unknown" venv_colcon_path="missing" venv_py_ver="unknown"
    venv_py_ver="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    venv_numpy_ver="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo unknown)"
    venv_colcon_path="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import colcon, os; print(os.path.dirname(colcon.__file__))' 2>/dev/null || echo missing)"

    # User-site inspection: even though build.sh sets PYTHONNOUSERSITE=1 and
    # we install colcon into the venv, a stale ~/.local/lib/.../colcon left
    # by an old system-wide pip install can still shadow the venv colcon when
    # users run `colcon` interactively outside of build.sh. We explicitly
    # suppress the user-site in build.sh to guarantee isolation, but a user's
    # system-level colcon would silently no-op. We surface this state in
    # the summary and warn explicitly when colcon shadows are detected.
    local user_site user_site_status="not-present" user_site_colcon=""
    user_site="$("${VENV_PYTHON}" -m site --user-site 2>/dev/null || true)"
    if [[ -n "${user_site}" && -d "${user_site}" ]]; then
        local user_pkg_count
        user_pkg_count="$(find "${user_site}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
        user_site_status="active (${user_pkg_count} packages)"
        if find "${user_site}" -maxdepth 1 -name "colcon*" | grep -q .; then
            user_site_colcon="DETECTED"
        fi
    fi

    echo -e "\n${YELLOW}Python Environment Summary:${NC}"
    echo "  Python Version:     ${venv_py_ver} (${VENV_PYTHON})"
    echo "  NumPy Version:      ${venv_numpy_ver} (Target: 1.26.x for ROS 2 ABI)"
    echo "  Colcon Path:        ${venv_colcon_path}"
    echo "  User Site-Packages: ${user_site_status} (${user_site})"

    if [[ -n "${user_site_colcon}" ]]; then
        log_warn "User site-packages contain a colcon installation that may shadow the workspace venv."
        log_warn "If 'colcon build' fails outside of build.sh, consider clearing the user site colcon:"
        log_warn "    rm -rf ${user_site}/colcon* ${user_site%/lib/*}/bin/colcon*"
    fi

    # Optional full-profile installs run after the platform dependencies.
    # Validate at the very end so no later pip transaction can invalidate it.
    # Gated on the accelerator, not the OS: an Ascend NPU needs the same ABI
    # check on Ubuntu as it does on openEuler.
    accel_verify_runtime "${VENV_PYTHON}" || return 1
    PYTHON_ENV_STATUS="done"

}
