# Observation transport codec extensions

The default registry and the `ascend` FFmpeg H.264 backend are unchanged.
`AscendFfmpegH264Decoder` consumes H.264 and returns RGB/BGR; JPEG is not H.264.

`ascend_jpeg_codec.register_ascend_jpeg_backend(registry, lease_factory=...)`
adds an explicit, opt-in JPEG decoder to an existing `VideoCodecRegistry`.
Select it with `registry.resolve("ascend_jpeg", "decoder", codec="jpeg")`.
Registry calls default to `codec="h264"`; auto resolution filters capability
declarations, so a JPEG backend cannot replace an H.264 decoder by priority.
The caller supplies an ACL context lease using its runtime owner; this package
does not import `inference_service`, initialize ACL globally, or reset unrelated
devices. Registration and capability probing do not acquire a lease.

The decoder uses the existing `VideoDecoder`, `EncodedPacket`, `VideoFrame`,
`CodecMetrics`, and `CodecLifecycleState` contracts. Each complete JPEG packet
produces one packed NV12 frame with `color_space="bt601"`, `color_range="full"`,
and the packet's capture timestamp. Hardware descriptor strides are checked and
padding is removed. Consumers must retain these color semantics when converting.
No generic H.264 auto policy is changed by this opt-in registration.

Call `decode` from one worker. A failed decode enters `FAILED`; `reset` releases
DVPP resources and allows another frame while retaining the injected lease.
Stop and join the worker before `close`. Stream synchronization must succeed
before in-flight buffers are freed. Failed DVPP cleanup retains its handles for
retry; lease release is delegated to its existing runtime owner. `close` only
marks the decoder closed after its owned resources are released.

The HRI adapter in `manipulation_execution` registers this decoder using the
mainline `AclRuntimeManager`, translates ROS JPEG messages to `EncodedPacket`,
and translates the resulting frame to its explicitly full-range NV12 message.
Models' resize, normalization, crop and color conversion remain in their
perception adapters rather than in this transport layer.
