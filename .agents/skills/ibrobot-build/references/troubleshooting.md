# Build Troubleshooting

## When to Read

- Build or runtime 命令失败时
- 出现 `ModuleNotFoundError: No module named 'lerobot'` 时
- 出现 `install directory was created with the layout 'merged'` 时
- 控制器无法 spawn 或节点无法互相发现时

## Issue: Controllers fail to spawn / Nodes cannot discover each other

**Root Cause**: ROS_DOMAIN_ID not set, causing DDS discovery to use default domain (0)

**Solution**: Always export ROS_DOMAIN_ID for runtime operations:
```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch ...
```

## Issue: ModuleNotFoundError: No module named 'lerobot'

**Root Cause**: PYTHONPATH not set, .shrc_local not sourced in current shell

**Solution**: Always source .shrc_local:
```bash
source .shrc_local && <your_command>
```

**Wrong** (won't work):
```python
# Bash call 1
source .shrc_local

# Bash call 2
ros2 launch robot_config robot.launch.py  # ← PYTHONPATH and ROS_DOMAIN_ID lost!
```

**Correct**:
```python
# Single Bash call
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py
```

## Issue: Build error - "install directory was created with the layout 'merged'"

**Root Cause**: Workspace uses merged install layout, but a hand-assembled `colcon build` omitted `--merge-install`

**Solution**: Always build through the project script, which applies the correct layout automatically:
```bash
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config
```

If the workspace layout state is already inconsistent, reset it with a clean build:
```bash
source .shrc_local && ./scripts/build.sh --clean
```

## Issue: Build Errors - venv not found

**Root Cause**: Virtual environment doesn't exist

**Solution**: Run setup script first:
```bash
./scripts/setup.sh
```

## Issue: Import errors after build

**Root Cause**: Environment not refreshed after build

**Solution**: Build and refresh in the same command:
```bash
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config && source install/setup.zsh && python3 -c "import lerobot; print('OK')"
```

## Issue: Link-time ABI / undefined symbol errors against ROS libraries

If a colcon link error, `undefined symbol`, or ABI mismatch against a ROS library only
reproduces on one distro (usually openEuler), it is the same class of problem as the
launch-time ROS package version mismatch. Follow the hard rule in AGENTS.md
("跨发行版 ROS 包版本一致性") and the playbook in
[ibrobot-launch/references/troubleshooting.md](../../ibrobot-launch/references/troubleshooting.md)
before adjusting build flags, link order, or source code.
