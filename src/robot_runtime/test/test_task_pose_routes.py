"""Route identity and schema are independent of endpoint approval."""

import copy

import pytest

from robot_runtime.task_poses import normalize_task_poses


def config():
    return {
        "schema_version": 1,
        "robot_type": "test",
        "joint_names": ["joint"],
        "poses": {
            "ready": {
                "joint_positions": {"joint": 0.5},
                "hardware_approved": True,
                "transition_route": {
                    "waypoints": [{"joint": 0.2}],
                    "supported_starts": [{"joint_positions": {"joint": 0.0}, "tolerance_rad": 0.1}],
                    "hardware_approved": False,
                    "source": {"path": "recording.json", "sha256": "a" * 64, "sample_indices": [1]},
                },
            }
        },
    }


@pytest.mark.parametrize("part", ["waypoint", "start", "approval", "source", "wait"])
def test_route_content_and_wait_are_bound_to_snapshot_digest(part):
    raw = config()
    before = normalize_task_poses(raw)
    pose = raw["poses"]["ready"]
    route = pose["transition_route"]
    if part == "waypoint":
        route["waypoints"][0]["joint"] += 0.01
    elif part == "start":
        route["supported_starts"][0]["tolerance_rad"] += 0.01
    elif part == "approval":
        route["hardware_approved"] = True
    elif part == "source":
        route["source"]["sha256"] = "b" * 64
    else:
        pose["arrival_wait_sec"] = 0.5
    assert normalize_task_poses(raw)["digest"] != before["digest"]
    assert before["poses"]["ready"]["transition_route"]["hardware_approved"] is False


@pytest.mark.parametrize("field,value", [("waypoints", []), ("supported_starts", []), ("hardware_approved", 1)])
def test_malformed_route_is_rejected(field, value):
    raw = copy.deepcopy(config())
    raw["poses"]["ready"]["transition_route"][field] = value
    with pytest.raises(ValueError):
        normalize_task_poses(raw)


def test_legacy_snapshot_digest_does_not_change_on_renormalization():
    raw = config()
    del raw["poses"]["ready"]["transition_route"]
    normalized = normalize_task_poses(raw)
    assert normalize_task_poses(normalized) == normalized
