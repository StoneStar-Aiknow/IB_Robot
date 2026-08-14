from pathlib import Path

import yaml

SCENE_ROOT = Path(__file__).parents[1] / "config" / "fast_calib" / "scenes"

EXPECTED = {
    "current_installation_scene01.yaml": {
        "roi": (-2.0, 2.0, -5.2, -4.3, 0.10, 1.60, 0.01),
        "normal_search_radius": 0.06,
        "boundary_search_radius": 0.06,
    },
    "current_installation_scene02.yaml": {
        "roi": (-2.0, 2.0, -5.2, -4.3, 0.10, 1.60, 0.01),
        "normal_search_radius": 0.06,
        "boundary_search_radius": 0.06,
    },
    "current_installation_scene03.yaml": {
        "roi": (-2.0, 2.0, -5.2, -4.3, 0.10, 1.60, 0.01),
        "normal_search_radius": 0.06,
        "boundary_search_radius": 0.06,
    },
    "current_installation_scene04_test.yaml": {
        "roi": (-2.0, 2.0, -5.2, -4.3, 0.10, 1.60, 0.01),
        "normal_search_radius": 0.06,
        "boundary_search_radius": 0.06,
    },
}


def test_tuned_scene_profile_preserves_current_installation_parameters():
    for filename, expected in EXPECTED.items():
        document = yaml.safe_load((SCENE_ROOT / filename).read_text(encoding="utf-8"))
        params = document["fast_calib"]["ros__parameters"]

        assert (
            tuple(params[name] for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"))
            + (params["plane_distance_threshold"],)
            == expected["roi"]
        )
        assert params["normal_search_radius"] == expected["normal_search_radius"]
        assert params["boundary_search_radius"] == expected["boundary_search_radius"]
        assert params["prefer_vertical_plane"] is True
        assert params["circle_radius"] == 0.13
        assert params["delta_width_circles"] == 0.50
        assert params["delta_height_circles"] == 0.40
        assert params["observation_only"] is True
        assert params["lidar_topic"] == "/cloud_dense_body"
