import copy
from typing import Any

import pytest
import yaml

from robot_config.loader import (
    load_robot_config_dict,
    robot_config_digest,
    robot_execution_endpoints,
)

_MISSING = object()
_ENDPOINT = "/robot/navigation/execute"


def _robot_config(
    *, action_name: Any = _ENDPOINT, legacy_owner: Any = _MISSING, command_server: Any = _MISSING
) -> dict:
    execution: dict[str, Any] = {}
    if legacy_owner is not _MISSING:
        execution["navigation_action_name"] = legacy_owner
    navigation: dict[str, Any] = {"enabled": True}
    if command_server is not _MISSING:
        navigation["command_server"] = copy.deepcopy(command_server)
        if isinstance(navigation["command_server"], dict) and action_name is not _MISSING:
            navigation["command_server"]["action_name"] = action_name
    return {
        "name": "navigation_robot",
        "embodied": {"execution": execution},
        "navigation": navigation,
    }


def _write_config(tmp_path, robot_config: dict, *, name: str = "robot.yaml"):
    config_path = tmp_path / name
    config_path.write_text(yaml.safe_dump({"robot": robot_config}), encoding="utf-8")
    return config_path


def _load(tmp_path, robot_config: dict, *, nav_stage: str = ""):
    return load_robot_config_dict(_write_config(tmp_path, robot_config), nav_stage=nav_stage)


def test_enabled_navigation_command_server_requires_endpoint_owner(tmp_path):
    config = _robot_config(action_name=_MISSING, command_server={"enabled": True})

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config)

    message = str(exc_info.value)
    assert "required" in message


@pytest.mark.parametrize(
    ("owner", "expected_error"),
    [
        ("", "non-empty"),
        ("navigation/execute", "absolute ROS name"),
        ("/navigation//execute", "valid ROS name"),
    ],
)
def test_navigation_endpoint_owner_must_be_a_valid_absolute_ros_name(tmp_path, owner, expected_error):
    config = _robot_config(action_name=owner, command_server={"enabled": True})

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config)

    message = str(exc_info.value)
    assert expected_error in message


def test_legacy_endpoint_owner_is_rejected_even_when_values_match(tmp_path):
    config = _robot_config(
        legacy_owner=_ENDPOINT,
        command_server={"enabled": True},
    )

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config)

    message = str(exc_info.value)
    assert "embodied.execution.navigation_action_name" in message


def test_equal_legacy_command_server_endpoint_is_rejected(tmp_path):
    config = _robot_config(
        legacy_owner=_ENDPOINT,
        command_server={"enabled": True, "action_name": _ENDPOINT},
    )

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config)

    assert "embodied.execution.navigation_action_name" in str(exc_info.value)


def test_v1_config_without_command_server_has_no_navigation_endpoint(tmp_path):
    config = _robot_config(command_server=_MISSING)

    loaded = _load(tmp_path, config)

    assert "navigation_action" not in robot_execution_endpoints(loaded)


def test_navigation_endpoint_owner_rejects_disabled_command_server_mapping(tmp_path):
    config = _robot_config(command_server={"enabled": False})

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config)

    message = str(exc_info.value)
    assert "action_name must be omitted" in message


def test_mapping_stage_rejects_navigation_endpoint_owner_leakage(tmp_path):
    config = {
        "name": "staged_navigation_robot",
        "default_nav_stage": "navigation",
        "nav_stages": {
            "mapping": {
                "embodied": {"execution": {"navigation_action_name": _ENDPOINT}},
                "navigation": {"command_server": {"enabled": True}},
            },
            "navigation": {},
        },
    }

    with pytest.raises(ValueError) as exc_info:
        _load(tmp_path, config, nav_stage="mapping")

    message = str(exc_info.value)
    assert "mapping" in message
    assert "embodied.execution.navigation_action_name" in message


def test_non_default_navigation_endpoint_is_projected_and_changes_digest(tmp_path):
    custom = _robot_config(command_server={"enabled": True})
    default = copy.deepcopy(custom)
    default["navigation"]["command_server"]["action_name"] = "/navigation/execute"

    loaded = _load(tmp_path, custom)

    assert loaded["navigation"]["command_server"]["action_name"] == _ENDPOINT
    assert robot_execution_endpoints(loaded)["navigation_action"] == _ENDPOINT
    assert robot_config_digest(custom) != robot_config_digest(default)
