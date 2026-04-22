from launch.substitutions import TextSubstitution

from robot_config.launch_builders.recording import generate_rerun_viewer_node


def _text(substitutions):
    return "".join(item.text if isinstance(item, TextSubstitution) else str(item) for item in substitutions)


def test_generate_rerun_viewer_node_forces_pythonnousesite():
    nodes = generate_rerun_viewer_node({"_config_path": "/tmp/robot.yaml"})

    assert len(nodes) == 1
    assert dict((_text(key), _text(value)) for key, value in nodes[0].additional_env) == {"PYTHONNOUSERSITE": "1"}
