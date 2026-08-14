import numpy as np

from robot_calibration.export import _resolve_body_from_livox


def test_resolve_body_from_livox_accepts_recorded_direct_static_edge():
    expected = np.eye(4)
    expected[:3, 3] = [0.1, -0.2, 0.3]

    resolved = _resolve_body_from_livox({("body", "livox_frame"): expected})

    assert np.allclose(resolved, expected)


def test_resolve_body_from_livox_accepts_legacy_base_link_edges():
    base_from_body = np.eye(4)
    base_from_body[:3, 3] = [1.0, 0.0, 0.0]
    base_from_livox = np.eye(4)
    base_from_livox[:3, 3] = [1.0, 2.0, 0.0]

    resolved = _resolve_body_from_livox(
        {
            ("base_link", "body"): base_from_body,
            ("base_link", "livox_frame"): base_from_livox,
        }
    )

    assert np.allclose(resolved[:3, 3], [0.0, 2.0, 0.0])
