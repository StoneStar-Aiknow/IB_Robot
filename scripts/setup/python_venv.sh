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

install_lerobot_editable() {
    local pip_runner=("$@")
    
    if ! check_lerobot_python_compat; then
        log_error "Cannot install lerobot: Python version is incompatible."
        log_error "Ensure patches 0001/0002 from scripts/setup/lerobot_patches.sh are applied"
        log_error "to downgrade the requirement to >=3.10 if you are on an older system."
        exit 1
    fi

    check_lerobot_ros_numpy_compat

    # [smolvla,pi] extras pull in transformers-dep + smolvla-specific deps
    # (num2words, accelerate, safetensors) so PI05/SmolVLA policies load
    # without manual pip installs. See libs/lerobot/pyproject.toml.
    "${pip_runner[@]}" install -e "${WORKSPACE}/libs/lerobot[smolvla,pi]"
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

    # 0. Python interpreter preflight
    local host_python_path host_python_version host_py_major host_py_minor
    host_python_path="$(command -v python3 || true)"
    if [[ -z "${host_python_path}" ]]; then
        log_error "python3 not found on PATH. Install python3 (>=3.10) before running setup.sh."
        exit 1
    fi
    host_python_version="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
    log_info "Using host python3: ${host_python_path} (version ${host_python_version})"
    host_py_major="$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
    host_py_minor="$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
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

    local pip_install=("${VENV_PYTHON}" -m pip install)
    local installed_perception_deps=false
    local installed_grasp_deps=false
    local ros_abi_constraints="${venv_path}/ros_abi_constraints.txt"

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

    # Install LeRobot in editable mode
    # Note: Do not pass the -c numpy==1.26.4 constraint. The lerobot dependency graph
    # (rerun-sdk, opencv, datasets, etc.) resolves under numpy>=2. A hard constraint
    # causes pip to fail with resolution-too-deep. We allow numpy 2.x here, then force-reinstall 1.26.4 + opencv<4.12 at the end.
    if [[ -d "${lerobot_dir}" ]]; then
        log_info "Installing LeRobot in editable mode..."
        install_lerobot_editable "${VENV_PYTHON}" -m pip
    fi

    # Install base Python dependencies
    log_info "Installing base Python dependencies..."
    run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/base.txt" --quiet

    # Install hardware dependencies
    log_info "Installing hardware dependencies..."
    run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/hardware.txt" --quiet

    # Install optional dependencies for mobile teleop (iOS: hebi-py, Android: teleop)
    log_info "Installing optional phone teleoperation dependencies..."
    if [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes mode: installing both phone backends (hebi-py + teleop)..."
        run_cmd "${pip_install[@]}" hebi-py teleop --quiet
        log_done "Phone teleoperation dependencies installed (hebi-py + teleop)"
    else
        echo ""
        echo "  Phone teleoperation backends (optional):"
        echo "    1) iOS only  — hebi-py  (HEBI Mobile I/O + ARKit)"
        echo "    2) Android only — teleop  (WebXR WebSocket)"
        echo "    3) Both (iOS + Android)"
        echo "    0) Skip phone backends"
        echo ""
        while true; do
            read -r -p "  Enter your choice [0-3]: " PHONE_CHOICE
            case "${PHONE_CHOICE}" in
                1)
                    run_cmd "${pip_install[@]}" hebi-py --quiet
                    log_done "Phone dependencies installed: hebi-py (iOS)"
                    break
                    ;;
                2)
                    run_cmd "${pip_install[@]}" teleop --quiet
                    log_done "Phone dependencies installed: teleop (Android)"
                    break
                    ;;
                3)
                    run_cmd "${pip_install[@]}" hebi-py teleop --quiet
                    log_done "Phone dependencies installed: hebi-py + teleop (iOS + Android)"
                    break
                    ;;
                0)
                    log_info "Skipping phone teleoperation dependencies."
                    break
                    ;;
                *)
                    echo "  Invalid choice. Please enter 0-3."
                    ;;
            esac
        done
    fi

    # Install development and training tools
    log_info "Installing dev-tools (tensorboard, rerun, gitlint, ruff, pre-commit)..."
    run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/dev-tools.txt" --quiet

    # Install platform-specific dependencies (ONNX tooling, etc.)
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/ubuntu-22.04.txt" --quiet
            ;;
        openeuler-embedded-24.03)
            run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/openeuler-24.03.txt" --quiet
            ;;
    esac

    if [[ -f "${WORKSPACE}/.pre-commit-config.yaml" ]]; then
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
    log_info "Pinning NumPy 1.26.4 + opencv-python-headless<4.12 (ROS 2 Humble ABI)..."
    run_cmd "${pip_install[@]}" --force-reinstall "numpy==1.26.4" \
        "opencv-python-headless<4.12" --quiet

    cat > "${ros_abi_constraints}" <<'EOF'
numpy==1.26.4
opencv-python<4.12
opencv-python-headless<4.12
EOF

    # Optional perception and manipulation dependencies are installed after the
    # core dependency set and ABI pins so later setup steps cannot overwrite them.
    if [[ "${INSTALL_PERCEPTION_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "Skipping optional perception dependencies on openEuler; SAM2/Grounding-DINO are validated on Ubuntu only."
    elif [[ "${INSTALL_PERCEPTION_DEPS:-false}" == true ]]; then
        log_info "Installing optional perception dependencies (SAM2, Grounding-DINO)..."
        run_cmd env SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" SAM2_BUILD_ALLOW_ERRORS=1 \
            "${pip_install[@]}" --no-build-isolation --constraint "${ros_abi_constraints}" \
            -r "${WORKSPACE}/requirements/perception.txt" --quiet
        installed_perception_deps=true
    else
        log_info "Skipping optional perception dependencies. Re-run setup with --with-perception if needed."
    fi

    if [[ "${INSTALL_GRASP_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "Skipping optional grasp dependencies on openEuler; GraspGen CUDA extensions are validated on Ubuntu only."
    elif [[ "${INSTALL_GRASP_DEPS:-false}" == true ]]; then
        log_info "Installing optional grasp dependencies (GraspGen)..."
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/setup/install_graspgen_pip.sh"
        export ROS_ABI_CONSTRAINTS="${ros_abi_constraints}"
        install_graspgen_pip "${VENV_PYTHON}" -m pip install
        installed_grasp_deps=true
    else
        log_info "Skipping optional grasp dependencies. Re-run setup with --with-grasp if needed."
    fi

    # Optional perception/grasp dependencies can pull OpenCV wheels whose latest
    # releases require NumPy 2.x. Re-apply the final ROS 2 ABI pin before smoke tests.
    log_info "Re-applying NumPy/OpenCV ROS 2 ABI pins after optional dependencies..."
    run_cmd "${pip_install[@]}" --force-reinstall "numpy==1.26.4" \
        "opencv-python-headless<4.12" --quiet

    log_info "Running Python dependency smoke tests..."
    PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import cv2
import numpy

if not numpy.__version__.startswith("1.26"):
    raise SystemExit(f"Expected NumPy 1.26.x after setup, got {numpy.__version__}")
print(f"NumPy/OpenCV smoke test passed: numpy={numpy.__version__}, cv2={cv2.__version__}")
PY
    if [[ "${installed_perception_deps}" == true ]]; then
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib.util

missing = [name for name in ("groundingdino", "sam2") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing perception modules after optional install: {', '.join(missing)}")
print("Perception optional dependencies smoke test passed")
PY
    fi
    if [[ "${installed_grasp_deps}" == true ]]; then
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib.util

if importlib.util.find_spec("grasp_gen") is None:
    raise SystemExit("Missing grasp_gen after optional GraspGen install")
print("GraspGen optional dependencies smoke test passed")
PY
    fi

    local commit_msg_hook
    commit_msg_hook="$(git rev-parse --git-path hooks/commit-msg 2>/dev/null || true)"
    if [[ -f "${commit_msg_hook}" ]] && grep -qi "gitlint" "${commit_msg_hook}"; then
        log_warn "gitlint commit-msg hook already exists at ${commit_msg_hook}; keeping it."
    else
        log_info "Installing gitlint commit-msg hook..."
        # gitlint is installed in the venv, which is currently activated
        printf 'y\n' | gitlint install-hook || log_warn "Failed to install gitlint hook"
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

    PYTHON_ENV_STATUS="done"
}
