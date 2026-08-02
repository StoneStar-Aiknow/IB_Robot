# tensormsg

`tensormsg` converts between ROS messages, NumPy arrays, and `ibrobot_msgs/VariantsList`. It remains the DDS tensor
serialization boundary for non-image distributed observations and for images whose contract explicitly uses
`transport.mode: dds`.

RTP observation video does not place H.264 payloads in `VariantsList`. The video data plane is owned by
`inference_service`; `tensormsg.converter` only provides shared image reconstruction primitives for that path:

- validate padded ROS image rows and convert `rgb8`, `bgr8`, or `mono8` to contiguous HWC `uint8`
- convert HWC `uint8` to limited/full-range BT.709 NV12 and back
- convert decoded HWC frames to canonical resized CHW float tensors
- preserve the existing DDS tensor conversion behavior

NV12 conversion requires positive even dimensions. Unsupported channel layouts, encodings, strides, color ranges,
or malformed payload lengths fail explicitly rather than being guessed. Stream negotiation, RTP packetization,
timestamp mapping, codec lifecycle, and transport fallback policy are not responsibilities of this package.
