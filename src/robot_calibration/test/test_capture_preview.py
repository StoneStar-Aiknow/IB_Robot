import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from robot_calibration.capture_preview import create_preview_cloud, parser, select_preview_points


def test_capture_preview_defaults_are_low_bandwidth():
    args = parser().parse_args([])

    assert args.image_topic == "/camera/front/image_raw"
    assert args.cloud_topic == "/cloud_registered_body"
    assert args.output_image_topic == "/calib/preview/image/compressed"
    assert args.output_cloud_topic == "/calib/preview/cloud"
    assert args.max_fps == 8.0
    assert args.jpeg_quality == 70
    assert args.max_points == 6000


def test_capture_preview_evenly_bounds_cloud_size():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    selected = select_preview_points(points, 4)

    assert selected.shape == (4, 3)
    assert selected[:, 0].tolist() == [0.0, 9.0, 18.0, 27.0]


def test_capture_preview_preserves_intensity_field():
    fields = [
        PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(("x", "y", "z", "intensity"))
    ]
    source = point_cloud2.create_cloud(
        Header(frame_id="body"),
        fields,
        [(1.0, 2.0, 3.0, 10.0), (4.0, 5.0, 6.0, 20.0)],
    )

    preview = create_preview_cloud(source, max_points=6000)
    values = point_cloud2.read_points_numpy(preview, field_names=["x", "y", "z", "intensity"])

    assert [field.name for field in preview.fields] == ["x", "y", "z", "intensity"]
    np.testing.assert_allclose(values, [[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]])
