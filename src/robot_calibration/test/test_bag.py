import pytest
import yaml

from robot_calibration.bag import REQUIRED_TOPIC_TYPES, camera_coefficients, validate_fast_calib_bag


def _bag(root, *, duration_ns=10_000_000_000):
    storage = root / "scene_0.mcap"
    root.mkdir()
    storage.write_bytes(b"mcap")
    metadata = {
        "rosbag2_bagfile_information": {
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": duration_ns},
            "relative_file_paths": [storage.name],
            "topics_with_message_count": [
                {"topic_metadata": {"name": name, "type": type_name}, "message_count": 1}
                for name, type_name in REQUIRED_TOPIC_TYPES.items()
            ],
        }
    }
    (root / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return root


def test_validate_fast_calib_bag_accepts_complete_mcap(tmp_path):
    bag = validate_fast_calib_bag(_bag(tmp_path / "scene"))

    assert bag.duration_s == pytest.approx(10.0)
    assert bag.storage_files[0].name == "scene_0.mcap"
    assert set(bag.topic_counts) == set(REQUIRED_TOPIC_TYPES)


def test_validate_fast_calib_bag_rejects_storage_traversal(tmp_path):
    root = _bag(tmp_path / "unsafe")
    metadata = yaml.safe_load((root / "metadata.yaml").read_text())
    metadata["rosbag2_bagfile_information"]["relative_file_paths"] = ["../outside.mcap"]
    (root / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="storage path"):
        validate_fast_calib_bag(root)


def test_validate_fast_calib_bag_accepts_slow_board_duration(tmp_path):
    bag = validate_fast_calib_bag(_bag(tmp_path / "slow", duration_ns=8_000_000_000))

    assert bag.duration_s == pytest.approx(8.0)


def test_camera_coefficients_are_yaml_serializable_python_floats():
    import numpy as np

    values = camera_coefficients(np.array([606.5, 1.0], dtype=np.float32))

    assert all(type(value) is float for value in values)
    assert yaml.safe_load(yaml.safe_dump({"K": values}))["K"] == pytest.approx([606.5, 1.0])
