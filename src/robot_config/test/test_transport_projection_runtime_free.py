"""The transport projection must not pull in ``robot_runtime``.

Review discussion 22320fc2: ``load_transport_robot_config`` is the consumer
entry point for the bound Gateway transport (closed-loop driver, Hermes
preflight, Web bridge). Importing it used to import ``robot_runtime`` at module
scope, so a host without the runtime/model layer could not even bind endpoints.

The check runs in a clean isolated interpreter (``-I`` ignores cwd, PYTHONPATH
and user site-packages) and inserts only the repository source trees that the
projection legitimately needs. A regression imports the repository
``robot_runtime`` package and fails the assertion; the test does not fake
``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import ament_index_python

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "src" / "robot_config" / "config" / "robots" / "unitree_g1_arm_presets.yaml"

_PROBE = r"""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
config = Path(sys.argv[2]).resolve()
ros_site = Path(sys.argv[3]).resolve()
runtime_src = sys.argv[4]
sys.path.insert(0, str(ros_site))
for relative in ("robot_config", "embodied_common"):
    sys.path.insert(0, str(repo / "src" / relative))
if runtime_src:
    sys.path.insert(0, runtime_src)

import robot_config.loader  # noqa: F401 - importing the projection module must be light

def runtime_modules():
    return sorted(
        name
        for name in sys.modules
        if name == "robot_runtime" or name.startswith("robot_runtime.")
    )

after_import = runtime_modules()
from robot_config.loader import load_transport_robot_config

projection = load_transport_robot_config(config_path=config)
after_projection = runtime_modules()
print(
    json.dumps(
        {
            "name": projection.name,
            "catalog_profile": projection.embodied.get("skill_catalog_profile"),
            "loader_file": sys.modules["robot_config.loader"].__file__,
            "after_import": after_import,
            "after_projection": after_projection,
        }
    )
)
raise SystemExit(3 if after_import or after_projection else 0)
"""


def _run_probe(tmp_path, runtime_src: Path | None) -> dict:
    ros_site = Path(ament_index_python.__file__).resolve().parent.parent
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _PROBE,
            str(REPO_ROOT),
            str(CONFIG),
            str(ros_site),
            "" if runtime_src is None else str(runtime_src),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert process.returncode == 0, f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_transport_projection_imports_no_robot_runtime(tmp_path):
    payload = _run_probe(tmp_path, REPO_ROOT / "src" / "robot_runtime")

    assert payload["name"] == "unitree_g1_arm_presets"
    assert payload["catalog_profile"] == "unitree_g1_arm_presets"
    assert Path(payload["loader_file"]).resolve().is_relative_to(REPO_ROOT / "src" / "robot_config")
    assert payload["after_import"] == [], "importing the projection module loaded robot_runtime"
    assert payload["after_projection"] == [], "calling the projection loaded robot_runtime"


def test_transport_projection_does_not_need_robot_runtime_installed(tmp_path):
    """A host without the runtime layer can still bind the Gateway transport."""

    payload = _run_probe(tmp_path, None)

    assert payload["name"] == "unitree_g1_arm_presets"
    assert payload["after_import"] == []
    assert payload["after_projection"] == []
