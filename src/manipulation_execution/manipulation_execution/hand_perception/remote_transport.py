"""ROS adapter for the existing model-service host; imported only in remote mode."""

from __future__ import annotations


class ModelServiceTransport:
    def __init__(self, node, endpoint):
        from ibrobot_msgs.srv import ClassifyHandGesture

        self.service_type = ClassifyHandGesture
        self.client = node.create_client(ClassifyHandGesture, endpoint)

    def ready(self):
        return self.client.service_is_ready()

    def begin(self, meta, pixels, frame_id):
        request = self.service_type.Request()
        request.metadata_json = meta.to_json()
        if pixels is not None:
            if pixels.shape != (320, 320, 3) or str(pixels.dtype) != "uint8":
                raise ValueError("remote gesture needs a 320 x 320 BGR uint8 crop")
            image = request.image
            image.header.stamp.sec, image.header.stamp.nanosec = divmod(meta.stamp_ns, 1_000_000_000)
            image.header.frame_id = frame_id
            image.height, image.width, image.step = 320, 320, 960
            image.encoding, image.is_bigendian = "bgr8", 0
            image.data = pixels.tobytes()
        return self.client.call_async(request)

    def reap(self, future, *, cancel=False):
        # A normal image timeout never calls this. Only a real terminal result
        # or the server's matched retirement acknowledgement releases its slot.
        self.client.remove_pending_request(future)
        if cancel and not future.done():
            future.cancel()
