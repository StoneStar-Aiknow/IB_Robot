"""The transport-only CLI context must not drag in ``robot_runtime``.

Review discussion 22320fc2 covers ``robot-skill-closed-loop`` and the Hermes
preflight too: they bind the Gateway through ``load_transport_context`` and must
not need the runtime/model layer before the Gateway answers. This probe imports
the real CLI catalog module and projects the real repository G1 configuration in
a fresh isolated interpreter, then asserts no ``robot_runtime`` module was
imported by that path.

The CLI imports generated ROS message packages (``ibrobot_msgs``) by design, so
the probe receives their installed location explicitly instead of inheriting
``PYTHONPATH``; it never manipulates ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import ament_index_python
import rosidl_parser

import ibrobot_msgs

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG = REPO_ROOT / "src" / "robot_config" / "config" / "robots" / "unitree_g1_arm_presets.yaml"

_PROBE = r"""
import json
import sys

config, paths = sys.argv[1], sys.argv[2:]
for path in reversed(paths):
    sys.path.insert(0, path)

import robot_config.loader
import robot_skill_cli.catalog

context, transport = robot_skill_cli.catalog.load_transport_context(config_path=config)
runtime_modules = sorted(
    name
    for name in sys.modules
    if name == "robot_runtime" or name.startswith("robot_runtime.")
)
print(
    json.dumps(
        {
            "robot_name": context.view.get("robot_name"),
            "status_service": transport.status_service,
            "loader_file": robot_config.loader.__file__,
            "catalog_file": robot_skill_cli.catalog.__file__,
            "runtime_modules": runtime_modules,
        }
    )
)
raise SystemExit(3 if runtime_modules else 0)
"""


def _isolated_paths() -> list[Path]:
    """Return only the paths this CLI entry point legitimately needs."""
    ros_paths = [
        Path(ament_index_python.__file__).resolve().parent.parent,
        Path(rosidl_parser.__file__).resolve().parent.parent,
    ]
    workspace_paths = [Path(ibrobot_msgs.__file__).resolve().parent.parent]
    source_paths = [
        REPO_ROOT / "src" / "agent" / "robot_skill_cli",
        REPO_ROOT / "src" / "agent" / "robot_client",
        REPO_ROOT / "src" / "agent" / "skill_catalog",
        REPO_ROOT / "src" / "robot_config",
        REPO_ROOT / "src" / "embodied_common",
        REPO_ROOT / "src" / "robot_runtime",
    ]
    # Source trees win over any installed copy; the generated message package
    # is the only workspace artifact this path genuinely needs.
    return [*source_paths, *workspace_paths, *ros_paths]


def test_transport_context_projects_the_g1_transport_without_runtime(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _PROBE,
            str(CONFIG),
            *(str(path) for path in _isolated_paths()),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert process.returncode == 0, f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    assert payload["robot_name"] == "unitree_g1_arm_presets"
    assert payload["status_service"] == "/embodied/get_skill_gateway_status"
    assert Path(payload["loader_file"]).resolve().is_relative_to(REPO_ROOT / "src" / "robot_config")
    assert Path(payload["catalog_file"]).resolve().is_relative_to(REPO_ROOT / "src" / "agent" / "robot_skill_cli")
    assert payload["runtime_modules"] == [], "the transport-only CLI context imported robot_runtime"
