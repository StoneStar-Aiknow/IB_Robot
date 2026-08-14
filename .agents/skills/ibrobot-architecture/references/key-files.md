# Key Files and Package Responsibilities

## When to Read

- 需要快速定位某个职责对应的包或源文件（如推理、动作分发、契约合成、记录转换等）
- 评估某项代码改动是否会影响包的职责边界、公开 API、配置键、跨包依赖或运行约束
- 进行架构审查时需要判断 package README 是否与代码行为存在漂移

## Package Responsibilities

| Package | Primary Responsibility |
|---------|----------------------|
| `robot_config` | Configuration management, launch orchestration |
| `ibrobot_msgs` | Interface definitions (Actions, Messages) |
| `tensormsg` | ROS↔Tensor protocol conversion |
| `inference_service` | Policy inference (monolithic/distributed) |
| `action_dispatch` | Action execution with temporal smoothing |
| `dataset_tools` | Episode recording and dataset conversion |
| `robot_teleop` | Teleoperation interfaces |
| `robot_moveit` | Motion planning integration |
| `so101_hardware` | Hardware drivers (ros2_control plugin) |

## README as Local Architecture Contract

Each package-level `README.md` is treated as the package's local architecture contract. It should describe the package's responsibilities, public entry points, launch/configuration usage, data flow, dependency boundaries, and known constraints.

When code changes alter any of the following, the package README must be checked and updated if needed:

1. Package responsibilities or prohibited responsibilities
2. Public APIs, CLIs, launch arguments, topics, services, or actions
3. Configuration keys, defaults, or SSOT sources
4. Data flow, tensor/ROS message contracts, or control mode behavior
5. Cross-package dependencies or layer boundaries
6. Operational limitations, required hardware, or setup steps

Architecture reviews should flag README drift as an architecture issue when code behavior and documentation diverge. A stale README is not a minor documentation style problem; it invalidates the package contract and makes future architecture reviews unreliable.

## Key Files Reference

| File | Purpose |
|------|---------|
| `robot_config/config/robots/so101_single_arm.yaml` | Robot configuration (SSOT) |
| `robot_config/launch/robot.launch.py` | Main orchestrator |
| `robot_config/robot_config/loader.py` | Config loading |
| `robot_config/robot_config/contract_builder.py` | Contract synthesis |
| `robot_config/robot_config/contract_utils.py` | Contract data structures |
| `robot_config/robot_config/launch_builders/execution.py` | Inference nodes |
| `inference_service/lerobot_policy_node.py` | Policy inference |
| `action_dispatch/action_dispatcher_node.py` | Action dispatch |
| `action_dispatch/temporal_smoother.py` | Temporal smoothing |
| `dataset_tools/episode_recorder.py` | Episode recording |
| `dataset_tools/bag_to_lerobot.py` | Dataset conversion |
