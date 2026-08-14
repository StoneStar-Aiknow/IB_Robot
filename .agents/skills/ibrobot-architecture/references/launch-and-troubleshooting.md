# Launch System and Troubleshooting

## When to Read

- 需要选择正确的 launch builder 来组合启动节点（control/perception/simulation/execution/teleop/recording）
- 需要查阅 `robot.launch.py` 支持的 launch arguments 及默认值
- 第一次启动系统、添加新机器人、调试契约合成时
- 遇到 `ModuleNotFoundError: lerobot`、控制器不匹配、契约合成失败等问题需要排查

## Launch System

### Modular Launch Builders

| Builder | Responsibility |
|---------|---------------|
| `control.py` | ros2_control setup, controller spawning |
| `perception.py` | Camera drivers, TF tree |
| `simulation.py` | Gazebo launch |
| `execution.py` | Inference and dispatch nodes |
| `teleop.py` | Teleoperation nodes |
| `recording.py` | Episode recording |

### Key Launch Arguments

| Argument | Purpose | Default |
|----------|---------|---------|
| `robot_config` | Configuration name | `test_cam` |
| `use_sim` | Enable Gazebo simulation | `false` |
| `control_mode` | Override control mode | (from YAML) |
| `with_inference` | Force enable/disable inference | (auto-detect) |
| `record` | Enable episode recording | `false` |

## Common Patterns

### Launching the System

```bash
# Standard launch
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Override control mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning

# With recording
ros2 launch robot_config robot.launch.py control_mode:=teleop record:=true
```

### Adding a New Robot

1. Create YAML: `config/robots/my_robot.yaml`
2. Define `name`, `joints`, `control_modes`, `models`, `peripherals`
3. Launch: `ros2 launch robot_config robot.launch.py robot_config:=my_robot`

### Debugging Contracts

```bash
# View synthesized contract
cat /tmp/robot_config/contracts/so101_single_arm_teleop.yaml

# Check launch logs
# [robot_config] ✓ Contract synthesis SUCCESS
# [robot_config]   Observations: 2
# [robot_config]   Actions: 6 joints
```

## Troubleshooting

### Issue: ModuleNotFoundError: lerobot

**Cause**: PYTHONPATH not injected properly

**Check**:
1. Look for `[robot_config] PYTHONPATH injection:` in launch logs
2. Verify `AMENT_PREFIX_PATH` includes workspace install directory

### Issue: Wrong controllers running

**Cause**: Control mode mismatch

**Solution**:
```bash
# For MoveIt
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# For ACT inference
ros2 launch robot_config robot.launch.py control_mode:=model_inference
```

### Issue: Contract synthesis fails

**Common Errors**:
1. `KeyError: 'robot'` - Use `robot_config['name']` not `robot_config['robot']['name']`
2. `Observation source not found` - Check `peripherals` matches `observations` sources
3. `Model not found` - Add model to `models:` section in YAML
