from pathlib import Path


def test_robot_config_does_not_export_embodied_launch_builder():
    import robot_config.launch_builders as launch_builders

    assert "generate_embodied_nodes" not in launch_builders.__all__
    assert not hasattr(launch_builders, "generate_embodied_nodes")


def test_robot_config_launch_builder_package_does_not_import_embodied_runtime():
    import robot_config.launch_builders as launch_builders

    exported_names = set(launch_builders.__all__)

    assert "generate_embodied_nodes" not in exported_names
    assert all("embodied" not in name for name in exported_names)


def test_robot_config_launch_does_not_import_embodied_bringup():
    launch_file = Path(__file__).parents[1] / "launch" / "robot.launch.py"

    assert "from embodied_bringup" not in launch_file.read_text()
