# Action Dispatch

A pull-based action distribution layer between inference models and ros2_control.

## Overview

This package provides an efficient action dispatching mechanism for distributing actions output by embodied AI models to robot controllers. It supports cross-frame temporal smoothing for Action Chunking models (e.g., ACT, Diffusion Policy), ensuring smooth transitions between consecutive inference outputs.

## System Architecture

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IB Robot System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│  │  Inference       │         │  Action          │         │  ros2_      │ │
│  │  Service         │         │  Dispatch        │         │  control    │ │
│  │                  │         │                  │         │             │ │
│  │ ┌──────────────┐ │         │ ┌──────────────┐ │         │ ┌─────────┐ │ │
│  │ │ Model        │ │         │ │ Action       │ │         │ │ Joint   │ │ │
│  │ │ (ACT/Diff)   │ │         │ │ Dispatcher   │ │         │ │ State   │ │ │
│  │ └──────────────┘ │         │ │   Node       │ │         │ │ Pub/Sub │ │ │
│  │                  │         │ └──────────────┘ │         │ └─────────┘ │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Temporal     │ │         │             │ │
│  │                  │         │ │ Smoother     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Topic        │───────────▶│ Controllers│ │
│  │                  │         │ │ Executor     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  └──────────────────┘         └──────────────────┘         └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 Communication                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Inference Service │                              │ Action Dispatch  │     │
│  │                   │                              │                  │     │
│  │                   │    DispatchInfer Action      │                  │     │
│  │                   │◀─────────────────────────────│                  │     │
│  │                   │    (ibrobot_msgs/action)     │                  │     │
│  │                   │                              │                  │     │
│  │                   │    VariantsList (Result)     │                  │     │
│  │                   │─────────────────────────────▶│                  │     │
│  │                   │    (action chunk tensor)     │                  │     │
│  └──────────────────┘                              └────────┬─────────┘     │
│                                                             │                │
│                                                             │                │
│  ┌──────────────────┐                              ┌────────▼─────────┐     │
│  │ ros2_control     │                              │ TopicExecutor    │     │
│  │                  │◀─────────────────────────────│                  │     │
│  │ /joint_commands  │   Float64MultiArray /        │                  │     │
│  │ /arm_commands    │   JointTrajectory            │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Sensor Layer     │                              │ Action Dispatch  │     │
│  │                  │                              │                  │     │
│  │ /joint_states    │─────────────────────────────▶│ (subscription)   │     │
│  │ (JointState)     │   optional                   │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Internal Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ActionDispatcherNode Internal Flow                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────┐                                                          │
│    │ Inference   │                                                          │
│    │ Request     │                                                          │
│    │ (watermark) │                                                          │
│    └──────┬──────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ Record      │      │ Send        │      │ Wait for    │               │
│    │ Current     │─────▶│ DispatchInfer─────▶│ Inference   │               │
│    │ Queue Len   │      │ Goal        │      │ Result      │               │
│    └─────────────┘      └─────────────┘      └──────┬──────┘               │
│                                                      │                      │
│                                                      ▼                      │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ Calculate   │      │ Time        │      │ Decode      │               │
│    │ Actions     │◀─────│ Alignment   │◀─────│ VariantsList│               │
│    │ Executed    │      │ (skip done) │      │ to Tensor   │               │
│    └──────┬──────┘      └─────────────┘      └─────────────┘               │
│           │                                                                  │
│           ▼                                                                  │
│    ┌─────────────────────────────────────────────────────────┐             │
│    │                    TemporalSmoother                      │             │
│    │  ┌─────────────────────────────────────────────────┐    │             │
│    │  │  Smoothing Enabled:                               │    │             │
│    │  │    old_actions + new_actions → blended_actions   │    │             │
│    │  │    (exponential weighted smoothing)               │    │             │
│    │  └─────────────────────────────────────────────────┘    │             │
│    │  ┌─────────────────────────────────────────────────┐    │             │
│    │  │  Smoothing Disabled:                              │    │             │
│    │  │    new_actions → direct queue replacement         │    │             │
│    │  └─────────────────────────────────────────────────┘    │             │
│    └───────────────────────────┬─────────────────────────────┘             │
│                                │                                            │
│                                ▼                                            │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ Control     │      │ Pop Next    │      │ TopicExecutor│               │
│    │ Loop        │─────▶│ Action      │─────▶│ Publish to  │               │
│    │ (100Hz)     │      │             │      │ Topics      │               │
│    └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ActionDispatcherNode

The main ROS2 node responsible for:
- Maintaining an action queue
- Triggering inference requests based on watermark thresholds
- Publishing actions to ros2_control at a fixed frequency
- Optional cross-frame temporal smoothing

### 2. TemporalSmoother

A cross-frame exponential smoother for handling Action Chunking model outputs:
- Maintains a smoothed action plan
- Performs temporal alignment when new inference results arrive
- Applies exponential weighted smoothing to overlapping regions

### 3. TopicExecutor

A topic-based action executor:
- Routes actions to correct topics based on Contract specifications
- Supports `Float64MultiArray` and `JointTrajectory` message types
- High-frequency position control

## Installation

```bash
cd ~/ibrobot_ws
colcon build --packages-select action_dispatch
source install/setup.bash
```

## Usage

### Launch Node

```bash
ros2 run action_dispatch action_dispatcher_node
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queue_size` | int | 100 | Maximum action queue length |
| `watermark_threshold` | int | 20 | Watermark threshold to trigger inference |
| `control_frequency` | double | 100.0 | Control frequency (Hz) |
| `inference_action_server` | string | `/act_inference_node/DispatchInfer` | Inference service Action name |
| `inference_reset_service` | string | `/act_inference_node/reset_policy_state` | Inference-side policy reset service; called best-effort during reset |
| `policy_reset_timeout_sec` | double | 2.0 | Max wait time for inference-side policy reset completion |
| `contract_path` | string | `''` | Contract file path |
| `joint_state_topic` | string | `/joint_states` | Joint state topic |
| `navigation_mode` | bool | false | Navigation mode (stopped at startup, waiting for external trigger) |
| `temporal_smoothing_enabled` | bool | false | Enable cross-frame smoothing |
| `temporal_ensemble_coeff` | double | 0.01 | Smoothing coefficient |
| `chunk_size` | int | 100 | Action chunk size |
| `smoothing_device` | string | `''` | Device for smoothing computation (empty=auto-detect) |

### Launch File Example

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='action_dispatch',
            executable='action_dispatcher_node',
            name='action_dispatcher',
            parameters=[{
                'queue_size': 100,
                'watermark_threshold': 20,
                'control_frequency': 100.0,
                'temporal_smoothing_enabled': True,
                'temporal_ensemble_coeff': 0.01,
                'chunk_size': 100,
                'contract_path': '/path/to/contract.yaml',
            }]
        )
    ])
```

## Cross-Frame Temporal Smoothing

### Principle

Embodied models typically output in Action Chunk format, producing n actions per inference. Cross-frame smoothing solves the following problem:

```
First inference: produces n action chunks
After executing l actions (l < n), second inference completes
New inference results need to be smoothed and aligned with remaining n-l actions
```

### Cross-Frame Smoothing Diagram

```
Timeline ──────────────────────────────────────────────────────────────────────▶

                    ┌─ Inference Start ─┐                ┌─ Inference End ─┐
                    │                   │                │                 │
                    ▼                   │                ▼                 │
                                                                                                  
T1: First Inference [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]  (n=10 actions)
                    │                                               │
                    │  Executing actions...                        │
                    ▼                                               ▼
T2: During Exec     [a4, a5, a6, a7, a8, a9, a10]                Remaining 7
                    │     ▲                                       ▲
                    │     │                                       │
                    │     └─ 3 actions executed during inference ┘
                    │
                    ▼
T3: Second Inference [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]  (new n=10)
                    │     │
                    │     └─ First 3 are outdated, skip
                    ▼
T4: Aligned New     [b4, b5, b6, b7, b8, b9, b10]              Relevant (n-l=7)
                    │
                    │  Smooth overlap with old actions
                    ▼
T5: Smoothed Result [blend, blend, blend, blend, b8, b9, b10]
                    │  └───────┬───────┘  │
                    │    Overlap Region   New Tail
                    │    (7 old + 7 new → 7 blended)
                    ▼
                    Final: 7 blended + 3 new = 10 actions
```

### Smoothing Process Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Cross-Frame Smoothing Calculation                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Original Action Queue (first inference result):                             │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─┐ ┌────┬────┬────┬────┬────┬────┬────┐                │
│  │  a1   a2   a3   │ │ a4 │ a5 │ a6 │ a7 │ a8 │ a9 │a10 │                │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─┘ └────┴────┴────┴────┴────┴────┴────┘                │
│  ╎                    │    │    │    │    │    │    │                      │
│  ╎ Executed (skip)    │    │    │    │    │    │    │   Remaining Queue   │
│  ╎ (3 during infer)   │    │    │    │    │    │    │   count: [1,1,1,1,1,1,1]│
│  ╎                    ▼    ▼    ▼    ▼    ▼    ▼    ▼                      │
│  ╎                                                                     │
│  ╎  New Inference Result (complete):                                    │
│  ╎  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─┐ ┌────┬────┬────┬────┬────┬────┬────┐            │
│  ╎  │  b1   b2   b3   │ │ b4 │ b5 │ b6 │ b7 │ b8 │ b9 │b10 │            │
│  ╎  └ ─ ─ ─ ─ ─ ─ ─ ─ ─┘ └────┴────┴────┴────┴────┴────┴────┴────┘        │
│  ╎  Outdated (skip)      │    │    │    │    │    │    │                  │
│  ╎                       │    │    │    │    └────┴────┴──▶ New tail      │
│  ╎                       │    │    │    │          (direct append)        │
│  ╎                       └────┴────┴────┴──▶ Overlap (needs smoothing)    │
│  ╎                                                                     │
│  ╎  Weight: w = exp(-0.01 * k),  Cumsum: [1.00, 1.99, 2.97, ...]       │
│                                                                              │
│  Smoothing Calculation (overlap region):                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  blended[i] = (old[i] * cumsum[count-1] + new[i] * weight[count])   │   │
│  │                         cumsum[count]                                 │   │
│  │                                                                       │   │
│  │  Example (i=0, count=1):                                              │   │
│  │    blended = (a4 * 1.00 + b4 * 0.99) / 1.99                          │   │
│  │           = 0.502 * a4 + 0.498 * b4                                  │   │
│  │                                                                       │   │
│  │  After multiple smoothings (count=k):                                 │   │
│  │    Old action weights accumulate, new action weights decrease         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Final Smoothed Result:                                                      │
│  ┌────────┬────────┬────────┬────────┬────┬────┬────┬────┬────┬────┐      │
│  │blend(4)│blend(5)│blend(6)│blend(7)│ b8 │ b9 │b10 │    │    │    │      │
│  └────────┴────────┴────────┴────────┴────┴────┴────┴────┴────┴────┘      │
│    └──────────┬──────────┘   └──┬──┘                                        │
│        Smoothed Region        New Tail                                      │
│                                                                              │
│  Legend: ╎ ╎ ╎ = Dashed lines show executed/outdated actions, not smoothed   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Smoothing Formula

```python
blended[i] = (old[i] * cumsum[count[i]-1] + new[i] * weight[count[i]]) / cumsum[count[i]]
```

Where:
- `old[i]`: The i-th action in the old action plan
- `new[i]`: The i-th action in the new inference result
- `weight[k]`: Weight for k-th contribution = exp(-coeff * k)
- `cumsum[k]`: Cumulative weight sum

### Smoothing Coefficient

| Coefficient Value | Effect |
|-------------------|--------|
| `0.0` | Uniform weighting, no preference for old/new |
| `Positive` | More weight to older actions (stable, conservative) |
| `Negative` | More weight to newer actions (responsive, may cause jitter) |

Default value `0.01` is from the original ACT paper.

### Runtime Toggle

```bash
# Toggle smoothing on/off
ros2 service call /action_dispatcher/toggle_smoothing std_srvs/srv/Empty

# Reset state
ros2 service call /action_dispatcher/reset std_srvs/srv/Empty
```

## Navigation Mode

When `navigation_mode=true`, the system starts in a stopped state and waits for an external trigger to begin execution. This mode is used when nav2 reaches the destination, then triggers the ACT model to execute grasping tasks.

### Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Navigation Mode Workflow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. System Startup                                                           │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  At startup: _is_running = False                        │
│     │ [NAV] Mode  │  System ready, waiting for trigger                       │
│     └─────────────┘                                                          │
│                                                                              │
│  2. Nav2 Navigation                                                          │
│     ┌─────────────┐                                                          │
│     │   Nav2      │  Navigate to target position                             │
│     │ Navigating  │  Dispatcher does not execute actions                     │
│     └─────────────┘                                                          │
│                                                                              │
│  3. Arrival at Destination                                                   │
│     ┌─────────────┐                                                          │
│     │  Nav2 Done  │  Call /action_dispatcher/start_evaluate                  │
│     └─────────────┘                                                          │
│                                                                              │
│  4. ACT Execution                                                            │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  _is_running = True                                      │
│     │ Executing   │  Trigger inference, execute ACT action sequence          │
│     └─────────────┘                                                          │
│                                                                              │
│  5. Task Complete                                                            │
│     ┌─────────────┐                                                          │
│     │    Call     │  Call /action_dispatcher/stop_evaluate                   │
│     │stop_evaluate│  Stop execution, set base velocity to zero               │
│     └─────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```bash
# Launch system (when navigation_mode=true)
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_navi control_mode:=navi

# After Nav2 reaches destination, start execution
ros2 service call /action_dispatcher/start_evaluate std_srvs/srv/Trigger

# After task completion, stop execution
ros2 service call /action_dispatcher/stop_evaluate std_srvs/srv/Trigger

# Query current status
ros2 service call /action_dispatcher/get_status std_srvs/srv/Trigger
```

### Configuration Example

Enable in robot configuration YAML:

```yaml
control_modes:
  navi:
    executor:
      navigation_mode: true    # Enable navigation mode
      watermark_threshold: 20
      control_frequency: 30.0
```

## Topics and Services

### Communication with Inference Service

| Direction | Topic/Action | Message Type | Description |
|-----------|--------------|--------------|-------------|
| Request | `/act_inference_node/DispatchInfer` | `ibrobot_msgs/action/DispatchInfer` | Send inference request |
| Response | `result.action_chunk` | `ibrobot_msgs/msg/VariantsList` | Receive action chunk (Tensor) |

### Published Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `~/queue_size` | `std_msgs/Int32` | Current queue length |
| `~/smoothing_enabled` | `std_msgs/Bool` | Whether smoothing is enabled |

### Subscribed Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Joint states (optional) |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `~/reset` | `std_srvs/Empty` | Reset queue and state; also best-effort calls `inference_reset_service` to reset inference-side policy state |
| `~/toggle_smoothing` | `std_srvs/Empty` | Toggle smoothing on/off |
| `~/start_evaluate` | `std_srvs/Trigger` | Start execution (only when navigation_mode=true) |
| `~/stop_evaluate` | `std_srvs/Trigger` | Stop execution and stop base (only when navigation_mode=true) |
| `~/get_status` | `std_srvs/Trigger` | Get running status (running/stopped) |

### Communication with ros2_control

| Direction | Topic | Message Type | Description |
|-----------|-------|--------------|-------------|
| Publish | `/joint_commands` | `std_msgs/Float64MultiArray` | Joint position commands |
| Publish | `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Trajectory commands |

## API Usage

### Using TemporalSmoother Directly

```python
from action_dispatch import TemporalSmoother, TemporalSmootherConfig

# Create configuration
config = TemporalSmootherConfig(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# Create smoother
smoother = TemporalSmoother(config)

# First inference
actions1 = model.inference(obs)  # shape: (100, action_dim)
smoother.update(actions1, actions_executed=0)

# Get actions one by one
for _ in range(30):
    action = smoother.get_next_action()
    robot.execute(action)

# Second inference (30 actions executed during inference)
actions2 = model.inference(obs)
smoother.update(actions2, actions_executed=30)

# Continue executing smoothed actions
while smoother.plan_length > 0:
    action = smoother.get_next_action()
    robot.execute(action)
```

### Using TemporalSmootherManager

```python
from action_dispatch import TemporalSmootherManager

manager = TemporalSmootherManager(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# Runtime toggle
manager.set_enabled(False)  # Disable smoothing
manager.set_enabled(True)   # Enable smoothing

# Check status
print(f"Plan length: {manager.plan_length}")
print(f"Smoothing enabled: {manager.is_enabled}")
```

## Dependencies

- ROS2 Humble
- Python 3.10+
- PyTorch
- NumPy
- ibrobot_msgs
- tensormsg

## License

Apache License 2.0
