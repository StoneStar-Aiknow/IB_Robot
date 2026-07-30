"""PointCloud2 conversion helpers."""

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def xyz_to_pointcloud2(points: np.ndarray, stamp: Time, frame_id: str) -> PointCloud2:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    message = PointCloud2()
    message.header = Header(stamp=stamp, frame_id=frame_id)
    message.height = 1
    message.width = points.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.is_dense = bool(np.isfinite(points).all())
    message.data = points.tobytes()
    return message
