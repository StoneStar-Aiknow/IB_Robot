# Launch System and Troubleshooting

## When to Read

- 需要了解 launch builder 注册机制，或查找某个业务的启动条目在哪个包（control/perception/simulation/execution/navigation/voice/recording/tracing 等）
- 需要查阅 `robot.launch.py` 支持的 launch arguments 及默认值
- 第一次启动系统、添加新机器人、调试契约合成时
- 遇到配置校验失败、控制器不匹配、推理 pipeline 启动失败等问题需要排查

## Launch System

`src/robot_config/launch/robot.launch.py` 是薄入口，只做三件事：声明参数 → `prepare_launch`（配置加载、CLI 覆盖、
hook、安全检查、冻结参与计划）→ `build_prepared_launch`（按计划执行条目，必要时经 readiness barrier 与 runtime
live-binding 延迟构造）。入口不导入任何业务模块。

启动条目是**显式注册的 `BuilderSpec`**（`robot_config/launch_registry/registry.py`，框架模块见下节），由两部分组成：

- **内建条目**：`builtin_registry.BUILTIN_BUILDERS` 中的有序引用表，只保留框架、基础设施与通用条目；
- **扩展条目**：已安装的业务包经 Python entry point 组 `ibrobot.launch_builders` 贡献，按 entry point 名称排序追加在
  内建条目之后。`build_launch_registry()` = 内建 + 扩展。

每个条目自己声明 id、构建函数（`module:attr` 惰性解析）、`after` / `requires`、`timing`（`immediate` /
`runtime_ready`）、启用条件（`enabled` / `mock_skip_key`）、自带的 CLI 参数（`arguments`）与 hook（`hooks`，挂在
`config_overrides` / `preflight` 等准备阶段）。类型化输出（control / simulation / benchmark 的 startup）只能由固定
的内建条目产生，扩展不得声明。**包未安装时，它贡献的条目和 CLI 参数都不出现**。仿真平台经
`sim_backend.get_sim_backend(platform)` 分发到 `GazeboAdapter` / `MujocoAdapter` / `MockAdapter`。

新增业务的写法、校验规则与示例见 `src/robot_config/README.md` 的「Launch builder 注册表」一节。

### 框架模块（`robot_config/launch_registry/`）

注册框架独立成包（openspec change `split-launch-registry-package`）：`registry.py`（BuilderSpec / 注册表 /
拓扑排序）、`preparation.py`（准备流水线）、`orchestrator.py`（执行与 live binding）、`launch_options.py`
（核心参数表与参数合成）、`builtin_registry.py`（内建表 + entry point 发现）、`builder_support.py`（扩展可用的
公共 helper）、`import_budget.py`（导入开销检查，供各包测试使用）。业务包定义启动条目时从
`robot_config.launch_registry` 导入框架 API；旧路径 `robot_config.launch_builders.<框架模块>`
只在本仓库历史中存在过，已删除，仓库外扩展需改用新路径。

`robot_config/launch_builders/` 只保留 robot_config 自己贡献的启动条目（见下表）与基础设施辅助模块：
`readiness.py`（controller readiness barrier，同时拥有 `controller_startup` 内建条目）、`tracing.py`
（LTTng session）、`description.py`、`runtime.py`、`hardware_mock.py`、`sim_backend/` 等。

### 内建条目（robot_config，10 个）

| 条目 | 模块 | 说明 |
|------|------|------|
| `control`、`auxiliary_actuators` | `control.py` | ros2_control 节点、controller spawner、被控辅助执行器（如 aero hand）、资源占用检查 |
| `simulation` | `simulation.py` | 仿真分发（`sim_backend/`、`sim_peripheral_bridge.py`、`hardware_mock.py`） |
| `controller_startup` | `readiness.py` | controller 就绪门控 |
| `perception` | `perception.py` | 无 provider 部署的相机/雷达驱动；有 `runtime.provider` 时物理外设由机器人 runtime 启动，只保留虚拟相机转发 |
| `benchmark` | `benchmark.py` | benchmark 物化，为后续条目固化配置快照 |
| `audio_io` | `audio_io.py` | 无 provider 部署的 ALSA 采集/播放；有 provider 时让路给 runtime（见下） |
| `execution` | `execution.py` | 推理 pipeline + 动作分发（inference_service / action_dispatch / robot_evaluate 节点） |
| `motion_planning` | `task_execution.py` | `with_moveit` 解析与 provider 校验（MoveIt 服务由 runtime provider 提供） |
| `embodied` | `embodied.py` | embodied 相关配置覆盖；embodied 运行时节点由 `embodied_bringup` 的入口启动 |

`description.py`（URDF xacro + 相机注入）、`runtime.py`（runtime provider 启动与 live binding）、`cmd_vel.py`、
`fast_lio.py`、`static_tf.py` 是内建条目与扩展共用的辅助模块（后三者也被 `sensor_calibration.launch.py` 使用）。

### 扩展条目（由业务包经 entry point 贡献，14 个）

| 条目 | 所属包 / 模块 | 说明 |
|------|--------------|------|
| `teleop`、`hand_sources` | `robot_teleop.launch_builders.{teleop,hand_sources}` | 遥操作设备；MHandPro 等手部输入源、`hand_profile` 与启动前 P 位姿确认 |
| `voice_asr`、`speech_direction` | `voice_asr_service.launch_builders.*` | 语音识别、声源定向 |
| `voice_tts` | `voice_tts_service.launch_builders.voice_tts` | 语音合成与播放 |
| `navigation` | `robot_navigation.launch_builders.navigation` | 导航（同包的 `nav2`、`localization`、`navigation_command`、`voice_nav` 子模块） |
| `task_executor` | `task_dispatch.launch_builders.task_execution` | `task_executor_node` |
| `recording`、`rerun_visualizer` | `dataset_tools.launch_builders.recording` | 连续 rosbag / episodic 录制、rerun viewer |
| `semantic_mapping` | `semantic_mapping.launch_builders.semantic_mapping` | 语义建图 |
| `perception_models` | `inference_service.launch_builders.perception_models` | `perception_services.services` 中的模型服务插件 |
| `motion_imitation_stream`、`mirror_session` | `manipulation_execution.launch_builders.*` | 动作模仿流、X2 镜像会话 |
| `interaction_demo` | `robot_interaction_demo.launch_builders.interaction_demo` | 交互演示 |

查看当前环境实际注册了哪些条目（内建表 + 已安装包声明的 entry point）：

```bash
source .shrc_local && python3 -c "
from importlib.metadata import entry_points
from robot_config.launch_registry.builtin_registry import BUILTIN_BUILDERS, LAUNCH_BUILDER_ENTRY_POINT_GROUP
print('builtin:', [ref.rsplit(':', 1)[1] for ref in BUILTIN_BUILDERS])
print('extensions:', sorted(f'{e.name} = {e.value}' for e in entry_points(group=LAUNCH_BUILDER_ENTRY_POINT_GROUP)))"
```

### 音频归属

设置了 `runtime.provider` 的部署，麦克风/扬声器由机器人 runtime 启动（`robot_runtime.peripherals` 的 `microphone` /
`speaker` 外设），并以 `audio.capture` / `audio.play` 接口（带采样率、通道数、采样格式）暴露；语音使用方通过
`voice_asr.audio_input`、`speech_direction.audio_input`、`voice_tts.audio_output` 绑定。这类部署不得写 `audio_io` 段。
无 provider 部署仍由 robot_config 的 `audio_io` 条目启动 ALSA 节点。

### Key Launch Arguments (25)

前 14 个是核心参数（`launch_options.CORE_ARGUMENT_SPECS`），其余由所属条目声明；所属包未安装时不会声明。

| Argument | Owner | Purpose | Default |
|----------|-------|---------|---------|
| `robot_config` | core | Configuration name (without .yaml) | `so101_single_arm` |
| `config_path` | core | Full path override | `""` |
| `use_sim` | core | Simulation flag（空则由 runtime target 推断） | `""` |
| `sim_platform` | core | Override `simulation.platform` (gazebo/mujoco/mock) | `""` |
| `runtime_target` | core | hardware/simulation/benchmark override | `""` |
| `auto_start_controllers` | core | Spawn controllers at startup | `true` |
| `control_mode` | core | teleop / model_inference / moveit_planning（空 → `default_control_mode`） | `""` |
| `nav_stage` | core | mapping/navigation/grasp/hybrid | `""` |
| `with_inference` | core | Force enable/disable inference（空自动检测） | `""` |
| `inference_pipeline` | core | Named pipeline ID for overrides | `""` |
| `inference_execution_mode` | core | monolithic / distributed | `""` |
| `moveit_display` | core | RViz | `true` |
| `enable_tracing` | core | ros2_tracing/LTTng | `false` |
| `trace_session_name` | core | Trace session name | `ib_robot_trace` |
| `with_moveit` | `motion_planning` | 空 → 由控制模式自动检测 | `""` |
| `with_embodied` | `embodied` | 空 → `robot.embodied.enabled` | `""` |
| `with_perception` | `embodied` | 空 → `robot.embodied.perception.enabled` | `""` |
| `hand_profile` | `hand_sources`（robot_teleop） | Hand profile selection（right/left/dual） | `""` |
| `with_navigation` | `navigation`（robot_navigation） | 空 → `robot.navigation.enabled` | `""` |
| `navigation_mode` | `navigation`（robot_navigation） | full / odom_only / imu_only | `""` |
| `record` | `recording`（dataset_tools） | Rosbag recording | `false` |
| `record_mode` | `recording`（dataset_tools） | continuous / episodic | `continuous` |
| `record_visualizer` | `rerun_visualizer`（dataset_tools） | none / rerun | `none` |
| `voice_asr_auto_start` | `voice_asr`（voice_asr_service） | Force ASR enabled + continuous | `false` |
| `voice_asr_realtime_pre_roll_seconds` | `voice_asr`（voice_asr_service） | ASR pre-roll | `""` |

## Common Patterns

### Launching the System

```bash
# Standard launch
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Mock backend (no hardware, no Gazebo): runtime target simulation + mock platform
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm \
  use_sim:=true sim_platform:=mock

# Override control mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm \
  control_mode:=moveit_planning

# Mobile base with navigation stage
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_nav_grasp \
  nav_stage:=grasp with_navigation:=true

# With episodic recording
ros2 launch robot_config robot.launch.py control_mode:=teleop \
  record:=true record_mode:=episodic
```

### Adding a New Robot

1. Create YAML: `config/robots/my_robot.yaml`（单顶层键 `robot:`）
2. 需要变体时用 overlay：`base_config: <base-name>`（同目录），列表追加用 `__append__`
3. Define `name`, `joints`, `control_modes`, `peripherals`, `contract`（按需加 `voice_*` / `embodied` / `navigation`）
4. Launch: `ros2 launch robot_config robot.launch.py robot_config:=my_robot`

### Debugging Contracts

契约在 launch 时**内存合成**（`RobotConfig.to_contract()` / `build_contract_from_robot_config_dict()`），不落盘：

```python
# Offline inspection
from robot_config.loader import load_robot_config
rc = load_robot_config("so101_single_arm")
contract = rc.to_contract()
print(contract.rate_hz, [o.key for o in contract.observations])
```

- Launch 日志中配置校验失败会 fail-fast 并给出具体原因（observation source 不在 `peripherals`、inference 引用的模型/pipeline 不存在、executor 类型非法等）。
- `loader.robot_config_digest()` 可用于对比两份配置是否漂移。

## Troubleshooting

### Issue: Config validation fails at launch

**Cause**: SSOT YAML 违反校验规则（`loader.validate_config` 在 launch 内执行）

**Check**:
1. Observation 的 `source` 是否在 `peripherals` 中定义
2. `control_modes.<mode>.inference` 引用的模型/pipeline 是否存在（`inference_config.parse_inference_config`）
3. Executor 类型是否为 `topic` / `benchmark`，controller 名是否出现在 `ros2_control`
4. Overlay 配置的 `base_config` 是否为同目录文件（跨目录、循环会被拒绝）

### Issue: Wrong controllers running

**Cause**: Control mode mismatch（每个 mode 的控制器集合来自 YAML `control_modes.<mode>.controllers`）

**Solution**:
```bash
# For MoveIt / skills
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# For policy inference
ros2 launch robot_config robot.launch.py control_mode:=model_inference
```

参考映射：position controllers（topic executor）用于 teleop / model_inference；trajectory controllers（action executor，经运行时 `motion_server` / `task_dispatch`）用于 moveit_planning；`base_controller` / `base_velocity_controller` 用于 base_navigation。

### Issue: A business node / launch argument is missing

**Cause**: 业务条目由所属包经 entry point 注册，包没有构建或安装时，条目和它的 CLI 参数都不会出现（不报错）。
例如只执行 `--packages-select robot_config` 时，teleop、语音、导航、录制等都不会启动。

**Check**:
1. 用上面的命令列出已注册的 entry point，确认对应条目在列
2. 构建所属包：`./scripts/build.sh -- --packages-up-to robot_config <owning_pkg>`
3. 条目在列但仍未启动：看 launch 日志中 registry 对每个跳过条目输出的原因（启用条件、mock 跳过等）

### Issue: `RegistryError` naming an entry point at launch

**Cause**: 扩展条目解析失败、与已有条目 id / 参数名 / hook config key 冲突，或声明了类型化输出。常见原因是 install
目录中残留了旧布局的 egg-info（例如条目迁移后旧包仍声明同名条目）。

**Solution**: 重新构建报错中提到的包（必要时先删除其 `build/<pkg>`、`install` 中对应的 egg-info）。

### Issue: `audio_io section is not allowed with runtime.provider=...`

**Cause**: 设置了 `runtime.provider` 的部署中音频设备由机器人 runtime 拥有。

**Solution**: 删除 `audio_io` 段，在部署或 runtime profile 中声明 `microphone` / `speaker` 外设，并为启用的语音使用方
声明 `audio_input` / `audio_output` 绑定（见上文「音频归属」）。

### Issue: Inference pipeline fails to start

**Common Errors**:
1. Manifest 加载失败 - bundle 必须是 schema v3（`inference_manifest.load_inference_manifest` 严格校验，含 artifact digest）
2. Backend conformance 失败 - deployment 的 runtime profile 与后端能力不匹配（如 ascend 要求 `target.runtime=acl`、artifacts 格式 `om`；见 `backends/registry.py` 各 validator）
3. Scheduler 会话失败 - session 化路径需先 `/inference/scheduler/ready` 就绪；重复请求会被 `IdempotencyLedger` 拒绝
4. Distributed 模式无视频 - 检查 RTP 通道协商（`VideoStreamNegotiator`）与 keyframe 等待状态（`WAITING_FOR_KEYFRAME`）
