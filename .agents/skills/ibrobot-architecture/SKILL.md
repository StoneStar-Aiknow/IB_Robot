---
name: ibrobot-architecture
description: "Provides deep knowledge of IB-Robot's architecture. Use when user needs to 'understand architecture', 'explain design', 'check SSOT', 'modify robot_config', 'check contract', 'architecture', '架构', '设计说明', '配置加载', '数据流', '契约设计'. Triggers for 'how does it work?', '架构设计', '系统原理', or when modifying core robot parameters and single source of truth files."
---

# IB-Robot Architecture Skill

This skill provides comprehensive knowledge of IB-Robot's layered architecture, design principles, and core components.

**Reference Documentation**: https://deepwiki.com/wuxiaoqiang12/IB_Robot

## Three Architectural Pillars

### 1. Single Source of Truth (robot_config YAML)

The `robot_config` YAML file serves as the **single authoritative source** for all robot specifications:

| Traditional Approach | IB-Robot Approach |
|---------------------|-------------------|
| Separate configs for ros2_control, cameras, ML contracts | One YAML defines everything |
| Manual synchronization between systems | Auto-propagation to all subsystems |
| Configuration drift over time | Guaranteed consistency |

**Key Files**:
- `src/robot_config/config/robots/so101_single_arm.yaml` - Robot configuration (SSOT)
- `src/robot_config/robot_config/loader.py` - Config loader and validator
- `src/robot_config/robot_config/config.py` - `RobotConfig` dataclass

### 2. Contract-Driven Design

A **Contract** defines the observation-action interface between robot and policy:

```python
@dataclass
class Contract:
    name: str
    rate_hz: int
    max_duration_s: float
    observations: list[ObservationSpec]  # Sensors → ML tensors
    actions: list[ActionSpec]            # ML tensors → Actuators
```

**Contract Consumers** (identical processing):
1. `episode_recorder` - Records data during teleoperation
2. `bag_to_lerobot` - Converts rosbag to LeRobot dataset
3. `lerobot_policy_node` - Live inference

**Key Files**:
- `src/robot_config/robot_config/contract_utils.py` - Contract data structures
- `src/robot_config/robot_config/contract_builder.py` - Contract synthesis
- `src/robot_config/robot_config/config.py` - `to_contract()` method

### 3. Control Mode Architecture

Three control modes converge on the same `ros2_control` hardware interface:

| Mode | Controllers | Interface | Frequency | Use Case |
|------|-------------|-----------|-----------|----------|
| `teleop` | `JointGroupPositionController` | Topic | 50 Hz | Human teleoperation |
| `model_inference` | `JointGroupPositionController` | Topic | 100 Hz | AI policy control |
| `moveit_planning` | `JointTrajectoryController` | Action | Variable | Motion planning |

**Key Files**:
- `src/robot_config/launch/robot.launch.py` - Mode selection logic
- `src/robot_config/robot_config/launch_builders/` - Modular launch builders

## Package Architecture

```
src/
├── robot_config/        # Configuration center (SSOT)
│   ├── config/robots/   # YAML configurations
│   ├── launch/          # robot.launch.py orchestrator
│   └── robot_config/    # Python modules
│       ├── loader.py           # Config loading
│       ├── contract_builder.py # Contract synthesis
│       └── launch_builders/    # Modular node generators
├── tensormsg/           # ROS↔Tensor protocol conversion
├── inference_service/   # Policy inference (monolithic/distributed)
├── action_dispatch/     # Action execution with temporal smoothing
├── dataset_tools/       # Episode recording & dataset conversion
├── robot_teleop/        # Teleoperation interfaces
├── robot_moveit/        # Motion planning integration
├── robot_description/   # URDF, SRDF, meshes
└── so101_hardware/      # ros2_control hardware plugin
```

For package responsibilities, README-as-contract rules, and the full Key Files Reference table, see `references/key-files.md`.

## Data Flow Overview

```
Observation Flow: Camera/JointState → ROS Topic → decode_value() → StreamBuffer → sample() → Preprocessor → Model
Action Flow:      Model → VariantsList → TemporalSmoother → Queue → TopicExecutor → Controller Topic → Hardware
```

For detailed code paths, inference execution modes (monolithic / distributed), and temporal smoothing internals, see `references/data-flow.md`.

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| Detailed data flow diagrams, Key Code Paths, Inference Execution Modes (Monolithic + Distributed), Temporal Smoothing | `references/data-flow.md` |
| Launch System (Modular Launch Builders, Key Launch Arguments), Common Patterns (Launching, Adding New Robot, Debugging Contracts), Troubleshooting (3 Issues) | `references/launch-and-troubleshooting.md` |
| Package Responsibilities table, README as Local Architecture Contract, Key Files Reference table | `references/key-files.md` |

Do not expose these references as separate skills.

## DeepWiki References

- [IB-Robot Overview](https://deepwiki.com/wuxiaoqiang12/IB_Robot/1-ib-robot-overview)
- [Core Concepts](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3-core-concepts)
- [Single Source of Truth Pattern](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.1-single-source-of-truth-pattern)
- [Contract System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.2-contract-system)
- [Control Mode Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.3-control-mode-architecture)
- [System Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/4-system-architecture)
- [Configuration System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/5-configuration-system-(robot_config))
- [Inference Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/7-inference-pipeline)
- [Action Dispatch](https://deepwiki.com/wuxiaoqiang12/IB_Robot/8-action-dispatch)
- [Data Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/9-data-pipeline)
