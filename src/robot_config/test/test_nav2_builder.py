from pathlib import Path

import pytest

from robot_config.launch_builders import nav2


def test_nav2_builder_rejects_missing_explicit_params_file(monkeypatch, tmp_path):
    share = tmp_path / "robot_navigation"
    (share / "launch").mkdir(parents=True)
    monkeypatch.setattr(nav2, "get_package_share_directory", lambda _package: str(share))
    monkeypatch.setattr(nav2, "resolve_ros_path", lambda path: path)

    with pytest.raises(RuntimeError, match="Invalid Nav2 params_file"):
        nav2.generate_nav2_nodes(
            {
                "nav2_bringup": {
                    "enabled": True,
                    "params_file": str(tmp_path / "missing.yaml"),
                }
            }
        )


def test_nav2_builder_accepts_supported_params_file(monkeypatch, tmp_path):
    share = tmp_path / "robot_navigation"
    launch_dir = share / "launch"
    config_dir = share / "config"
    launch_dir.mkdir(parents=True)
    config_dir.mkdir()
    (launch_dir / "nav2_bringup.launch.py").touch()
    params_file = config_dir / "nav2_params.yaml"
    params_file.write_text("controller_server: {}\n", encoding="utf-8")
    monkeypatch.setattr(nav2, "get_package_share_directory", lambda _package: str(share))

    actions = nav2.generate_nav2_nodes(
        {
            "nav2_bringup": {
                "enabled": True,
                "params_file": str(params_file),
            }
        }
    )

    assert len(actions) == 1
    assert Path(params_file).is_file()
