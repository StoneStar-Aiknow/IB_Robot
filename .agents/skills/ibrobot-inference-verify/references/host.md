# Host Verification Commands

## When to Read

- Running tiers 1–4 of the inference verification matrix
- Need the exact pytest, build, and ROS mock launch commands
- Need the perception typed-service call scripts

## Tier 1 — Unit & Contract Tests

```bash
source .shrc_local && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q src/inference_service/tests
```

Pass criteria: all tests pass (no collection errors).

## Tier 2 — ROS Overlay Build

Never invoke `colcon build` directly; always build through the project script
with a clean cache:

```bash
source .shrc_local && ./scripts/build.sh --clean
```

Pass criteria: all packages finished, 0 failures.

## Tier 3 — Policy ROS Mock (ACT CPU)

### Generate mock YAML

```python
import yaml, copy
c = copy.deepcopy(yaml.safe_load(open("src/robot_config/config/robots/so101_single_arm.yaml")))
c["robot"]["simulation"]["platform"] = "mock"
c["robot"]["simulation"]["scene"] = None
c["robot"]["default_control_mode"] = "model_inference"
c["robot"]["control_modes"]["model_inference"]["scheduler_enabled"] = False
c["robot"]["voice_tts"]["enabled"] = False
with open("/tmp/opencode/so101_mock_verify.yaml", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

### Launch and verify

```bash
source .shrc_local && export ROS_DOMAIN_ID=99 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    config_path:=/tmp/opencode/so101_mock_verify.yaml \
    control_mode:=model_inference use_sim:=true
```

Pass criteria (check launch output):
- `hardware_mock contract_mock active`
- `Unified pipeline started: id=policy ... backend=torch`
- `First inference received: chunk=100, latency=...ms`

### DDS stale discovery fix

If `action_dispatcher` shows `in_progress=True gen=0` and never recovers, the
previous launch left stale DDS discovery entries. Use a fresh `ROS_DOMAIN_ID`
(try 93–99, never reuse 42 for tests).

## Tier 4 — Perception ROS Typed Service

### Generate perception mock YAML

Add `perception_services` under `robot:` (NOT top-level) with one service per
model. Disable `voice_tts` and `scheduler_enabled` to reduce noise.

```python
import yaml, copy
c = copy.deepcopy(yaml.safe_load(open("src/robot_config/config/robots/so101_single_arm.yaml")))
c["robot"]["simulation"]["platform"] = "mock"
c["robot"]["simulation"]["scene"] = None
c["robot"]["default_control_mode"] = "model_inference"
c["robot"]["control_modes"]["model_inference"]["scheduler_enabled"] = False
c["robot"]["voice_tts"]["enabled"] = False
c["robot"]["perception_services"] = {
    "services": [
        {"id": "ram_plus_tags", "enabled": True, "required": True,
         "bundle_path": "models/ram_plus_swin_large_14m", "deployment": "torch_cpu",
         "adapter_class": "perception_service.model_service_plugins:RAMPlusRecognizeTagsPlugin",
         "service_type": "ibrobot_msgs/srv/RecognizeTags",
         "endpoint": "/perception/ram_plus/recognize_tags", "node_name": "ram_plus_tags"},
        {"id": "sam2_masks", "enabled": True, "required": True,
         "bundle_path": "models/sam2.1_hiera_tiny", "deployment": "torch_cpu",
         "adapter_class": "perception_service.model_service_plugins:SAM2GenerateMasksPlugin",
         "service_type": "ibrobot_msgs/srv/GenerateMasks",
         "endpoint": "/perception/sam2/generate_masks", "node_name": "sam2_masks"},
        {"id": "siglip2_image", "enabled": True, "required": True,
         "bundle_path": "models/siglip2_so400m_patch14_384", "deployment": "torch_cpu",
         "adapter_class": "perception_service.model_service_plugins:SigLIP2EncodeEmbeddingsPlugin",
         "service_type": "ibrobot_msgs/srv/EncodeEmbeddings",
         "endpoint": "/perception/siglip2/encode_embeddings", "node_name": "siglip2_image"},
    ]
}
with open("/tmp/opencode/so101_perception_mock.yaml", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

### Launch and call services

Launch the same way as Tier 3 but with the perception YAML. Wait 30–40 s for
all model_service_node instances to load, then call each service:

```python
import rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from ibrobot_msgs.srv import RecognizeTags

rclpy.init()
node = Node("perception_test")
cli = node.create_client(RecognizeTags, "/perception/ram_plus/recognize_tags")
while not cli.wait_for_service(timeout_sec=5): pass
img = Image()
img.header.stamp = node.get_clock().now().to_msg()
img.header.frame_id = "camera"
img.height = 480; img.width = 640; img.encoding = "rgb8"
img.is_bigendian = False; img.step = 1920
data = np.zeros((480, 640, 3), dtype=np.uint8)
data[120:360, 180:460] = [220, 40, 30]  # red rectangle
img.data = data.tobytes()
req = RecognizeTags.Request(); req.image = img; req.score_threshold = 0.5
future = cli.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=30)
result = future.result()
print(f"RAM++ success={result.success} tags={len(result.tags)} "
      f"infer_ms={result.inference_time_ms:.1f} state={result.model.runtime_state}")
node.destroy_node(); rclpy.shutdown()
```

Pass criteria per service:
- Service registered (`ros2 service list` shows the endpoint)
- `success=True` (or `success=False` with a clear non-runtime error like
  "RGB image data is truncated" if test data is incomplete)
- `runtime_state=ready`

### Common: SAM2 CPU timeout

SAM2 automatic mask generation on CPU takes ~100 s. Use `timeout_sec=120` in
`spin_until_future_complete` or skip SAM2 in service tests and verify via script.
