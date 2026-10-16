"""Exercise the setup shell gate in real children, without simulator/assets/GPU."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

VERIFY_SCRIPT = Path(__file__).resolve().with_name("verify_env.sh")


@pytest.fixture
def gate_workspace(tmp_path: Path) -> Path:
    probe = tmp_path / "src/benchmark/adapters/libero/benchmark_libero/version_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "assert 'libero' not in sys.modules, 'mixed providers'\n"
        "def prepare_libero_plus_import():\n"
        "    assert os.environ['PYTHONNOUSERSITE'] == '1'\n"
        "    assert os.environ['MUJOCO_GL'] == 'osmesa'\n"
        "    os.environ['LIBERO_CONFIG_PATH'] = 'plus-config'\n"
        "    sys.path.insert(0, str(Path(os.environ['WORKSPACE']) / 'provider'))\n"
        "def probe_libero_plus_provider(allow_missing_assets=False):\n"
        "    assert allow_missing_assets is True\n"
        "    assert 'libero.libero.benchmark' in sys.modules\n"
        "    assert 'libero.libero.envs' in sys.modules\n"
        "    if os.environ['ASSETS_STATUS'] == 'invalid':\n"
        "        raise RuntimeError('invalid asset tree')\n"
        "    return SimpleNamespace(distribution_name='libero-plus', module_path='plus-provider')\n"
    )
    provider = tmp_path / "provider/libero/libero"
    provider.mkdir(parents=True)
    (provider.parent / "__init__.py").write_text("")
    (provider / "__init__.py").write_text(
        "import os\nassert os.environ['LIBERO_CONFIG_PATH'] == 'plus-config', 'prepare must run first'\n"
    )
    (provider / "benchmark.py").write_text("def get_benchmark_dict():\n    return {}\n")
    (provider / "envs.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['WORKSPACE'], 'plus.pid').write_text(str(os.getpid()))\n"
        "failure = os.environ.get('IMPORT_FAILURE', '')\n"
        "if failure == 'python':\n    raise ImportError('missing perturbation dependency')\n"
        "if failure == 'native':\n    raise OSError('libMagickWand.so: cannot open shared object file')\n"
        "if failure == 'exit':\n    raise SystemExit(17)\n"
        "class OffScreenRenderEnv:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('import gate must not construct an environment or load assets')\n"
    )
    return tmp_path


@pytest.mark.parametrize("assets_status", ["missing", "ready", "invalid"])
@pytest.mark.parametrize("import_failure", ["", "python", "native", "exit"])
def test_plus_import_gate_is_independent_and_fail_closed(
    gate_workspace: Path, assets_status: str, import_failure: str
) -> None:
    python = shlex.quote(sys.executable)
    # Stub only the existing, separately tested hf/GPU body. The plus body must
    # actually execute with the interpreter, never in hf's import namespace.
    shell = f"""
set -eu
source {shlex.quote(str(VERIFY_SCRIPT))}
log_info() {{ :; }}
log_done() {{ :; }}
benchmark_python() {{
    local script
    script="$(</dev/stdin)"
    if [[ "$script" == *MINIMAL_BENCHMARK_VERIFY_OK* ]]; then
        {python} -c 'import os, sys, types; from pathlib import Path; sys.modules["libero"] = types.ModuleType("hf_libero"); Path(os.environ["WORKSPACE"], "hf.pid").write_text(str(os.getpid()))'
    else
        {python} -c "$script"
    fi
}}
VENV_PYTHON=benchmark_python
verify_benchmark_minimal
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        env={
            **os.environ,
            "WORKSPACE": str(gate_workspace),
            "VENV_PYTHON": sys.executable,
            "SETUP_PLATFORM_ID": "ubuntu-22.04",
            "ROSDEP_BIN": "/unused/rosdep",
            "USE_SUDO": "false",
            "SETUP_ROS_SETUP_PATH": "/unused/ros/setup.bash",
            "LIBERO_CONFIG_PATH": "hf-config",
            "MUJOCO_GL": "osmesa",
            "ASSETS_STATUS": assets_status,
            "IMPORT_FAILURE": import_failure,
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert (gate_workspace / "hf.pid").read_text() != (gate_workspace / "plus.pid").read_text()
    success = not import_failure and assets_status != "invalid"
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    assert ("LIBERO_PLUS_IMPORT_VERIFY_OK" in result.stdout) == success
    if import_failure == "python":
        assert "missing perturbation dependency" in result.stderr
    elif import_failure == "native":
        assert "libMagickWand.so" in result.stderr
    elif assets_status == "invalid" and not import_failure:
        assert "invalid asset tree" in result.stderr
