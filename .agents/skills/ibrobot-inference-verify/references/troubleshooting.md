# Troubleshooting

## When to Read

- A verification tier failed with an unfamiliar error
- DDS or action_dispatcher behaves unexpectedly
- Model service plugin fails to initialize
- Board SSH or CANN environment issues

## Pass / Fail Criteria

| Tier | Pass criteria |
|------|----------------|
| 1 (pytest) | All tests pass, 0 collection errors |
| 2 (build) | All packages finished, 0 failures |
| 3 (policy mock) | `First inference received: chunk=100` in launch output |
| 4 (perception service) | Service registered + `success=True` or `runtime_state=ready` |
| 5 (board OM script) | `status=passed` with correct shapes + `finite=True` |
| 6 (board ROS mock) | Pipeline started + service `success=True` |
| 7 (board NPU) | `action_shape=(100, 6) finite=True` (slow, ~610 s) |

## Known Issues

### DDS stale action-server discovery

**Symptom**: `action_dispatcher` shows `in_progress=True gen=0` forever; pipeline
says "already processing another operation" even after the first launch exits.

**Cause**: Previous `ros2 launch` left DDS discovery entries; the new
`action_dispatcher` discovers a dead action server and dispatches a goal that
never completes, permanently blocking the pipeline.

**Fix**: Use a fresh `ROS_DOMAIN_ID` for each test run (e.g., 93, 94, 95, …).
Never reuse `ROS_DOMAIN_ID=42` for tests. If already stuck, kill all ROS
processes and wait 10 s before relaunching.

### adapter.json identity mismatch

**Symptom**: `model_service_node` crashes with "adapter identity mismatch:
expected {'interface': 'tensor_model', 'model_type': 'ram_plus', ...}, got
{'family': 'ram_plus', ...}".

**Cause**: `models/<bundle>/assets/adapter.json` still has the legacy `family`
field instead of v3 `interface` and `model_type`.

**Fix**: Add `interface: "tensor_model"` and `model_type: "<model_type>"` and
`operation: "<operation>"` to each `adapter.json`. Remove `family` if it
duplicates `model_type`.

Note: `models/` is `.gitignore`-d, so this is a local-only fix. Board bundles
may already have the correct fields.

### builder_options not unwrapped (SessionBuilderRegistry)

**Symptom**: `model_service_node` crashes with "missing required keyword-only
argument 'adapter'" when launching perception services.

**Cause**: `SessionBuilderRegistry.create()` filtered out `builder_options`
because the builder signature doesn't have a `builder_options` parameter.

**Fix**: Ensure `registry.py` has the `builder_options` unwrapping code:
```python
builder_options = kwargs.pop("builder_options", None)
if builder_options:
    kwargs.update(dict(builder_options))
```

### LeRobot `typing.Self` import error

**Symptom**: `ImportError: cannot import name 'Self' from 'typing'` on Python
3.10.

**Cause**: LeRobot v0.6.0 uses `typing.Self` (Python 3.11+); the v0.6.0 patch
stack backports it via `typing_compat` but the patch may not be applied.

**Fix**: Run `bash scripts/setup.sh --only-patch --yes` to apply the LeRobot
patch stack. On the board, use `IBR_LEROBOT_FORCE_REBUILD=1` if the board
is still on v0.5.1 patched branch.

### ACT policy "stack expects each tensor to be equal size"

**Symptom**: `RuntimeError: stack expects each tensor to be equal size, but
got [3, 1024] at entry 0 and [1024] at entry 1`.

**Cause**: LeRobot preprocessor outputs tensors without batch dimension;
`predict_action_chunk` treats the channel dimension as batch, causing
encoder layers to produce mismatched shapes.

**Fix**: Ensure `LeRobotTorchModelSession._execute` adds batch dimension
when the input tensor's dimension count equals the manifest-declared shape
dimension count (via `unsqueeze(0)`).

### SAM2 operation mismatch on board

**Symptom**: "plugin requires tensor_model/sam2/automatic, got
tensor_model/sam2/prompt".

**Cause**: Board's `sam2_hiera_tiny_ascend` bundle has `operation: prompt`
(box-prompt mode), but `SAM2GenerateMasksPlugin` expects `operation: automatic`.

**Fix**: Remove SAM2 from the board's `perception_services` list, or use a
bundle with `operation: automatic` for the automatic mask generator.

### Board contract_mock crash with minimal YAML

**Symptom**: `contract_mock` crashes with `AttributeError: 'str' object has
no attribute 'get'` when using a hand-written minimal YAML.

**Cause**: The minimal YAML has `peripherals: {cameras: []}` (a dict with
empty list), but the loader expects `peripherals` to be a list of peripheral
dicts (copied from the production YAML).

**Fix**: Always start from a copy of the production `so101_single_arm.yaml`
and override fields, rather than writing a minimal YAML from scratch.

### Board RTPS_READER_HISTORY payload size error

**Symptom**: `RTPS_READER_HISTORY Error: Change payload size of '32' bytes is
larger than the history payload size of '19' bytes`.

**Cause**: DDS FastRTPS on openEuler Embedded has a small default payload
history; some message types exceed it.

**Impact**: Non-fatal warning; does not block inference or service calls.
