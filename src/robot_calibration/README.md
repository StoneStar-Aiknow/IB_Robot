# robot_calibration

IB-Robot 的 D435i/MID-360 外参标定工具。`robot_config` 是硬件配置和传感器安装关系的唯一事实来源。

## 现场流程

### 1. 环境准备

在板端和 PC 端执行标定命令前，按 IB-Robot 规范通过 `.shrc_local` 加载环境。

确认传感器网络和 `ROS_DOMAIN_ID` 与当前机器人运行环境一致。采集期间不要同时启动
`robot.launch.py`、底盘遥操作、Nav2 或其他发布速度指令的程序。

### 2. 板端采集

只需执行采集命令。命令会自行启动传感器图，等待 RealSense、MID-360、FAST-LIO
和底盘控制链就绪后再进入第一个 Enter：

```bash
ros2 run robot_calibration calib_capture
```

该等待没有自动超时。命令会每 15 秒列出尚未收到消息的必需 topic；开发板冷启动较慢时继续等待即可，
需要主动放弃时按 `Ctrl+C`。只有八个必需 topic 都实际收到消息后，才会显示 `scene-01` 的录制提示。

原始数据包默认保存到：

```text
~/.ros/ibrobot/calib/raw/<capture-id>.raw.tar
```

保存位置与执行命令时的当前目录无关。采集完成后，命令会打印原始数据包的绝对路径和可直接修改用户、IP 后使用的 `scp` 示例。

传感器图只包含：

- RealSense
- MID-360
- 静态 TF
- FAST-LIO
- 底盘控制器和 `/cmd_vel` bridge

`calib_capture` 会管理本次采集专属的传感器图，并在退出时清理它；不会发布速度指令。需要移动机器人时，另开终端运行
`teleop_twist_keyboard`，移动后停稳再按 Enter 录制。

有图形环境时，命令会自动打开采集 RViz，同时显示：

- `/calib/preview/image`：开发板本地限频到 8 Hz 并经 JPEG 传输的 RealSense 预览
- `/calib/preview/cloud`：开发板本地抽稀后的 MID-360 预览点云

预览节点在开发板上订阅原始图像和点云，只发布低带宽观察数据。采集 bag 仍然只记录原始八个必需 topic。
预览和 RViz 使用当前工作区已有的默认 ROS 2 domain、RMW 和 QoS 环境，不要求配置额外 DDS 文件、domain bridge
或新的环境变量。

按提示依次采集四个约 10 秒的静止场景：

```text
scene-01
scene-02
scene-03
scene-04-test
```

每个场景的操作步骤：

1. 摆放标定板，确保 D435i 和 MID-360 都能看到标定板。
2. 人工移动机器人到当前采集位置。
3. 确认机器人完全静止。
4. 按 Enter 开始录制。
5. 等待约 10 秒，确认当前场景采集完成且画面没有明显遮挡。
6. 按 Enter 进入下一个场景，移动机器人后重复上述步骤。

采集命令会自动发现 D435i 序列号。发现失败时记录 `unavailable`，不会伪造设备身份。
采集异常时按 `Ctrl+C` 退出，未完成的临时数据不能继续用于处理。

### 3. PC 端处理

将 `<capture-id>.raw.tar` 复制到 PC 后执行：

```bash
ros2 run robot_calibration calib_process --input <capture-id>.raw.tar
```

默认使用当前 IB-Robot 工作区中的 `src/fast_calib` 和默认 merged install 路径
`install/lib/fast_calib/fast_calib`。修改 FAST-Calib workspace 时，显式使用 `--workspace <workspace>` 或设置
`FAST_CALIB_WORKSPACE`；不要在文档或代码中固定个人工作区路径。

无需指定 `--output`。默认保存位置为：

```text
~/.ros/ibrobot/calib/process/<capture-id>/
~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar
```

第一个目录保存完整的中间结果和诊断信息，第二个路径是需要传回板端的候选标定包。
命令结束后会打印两个绝对路径，以及复制候选标定包回板端的 `scp` 示例。

查看离线叠加并将候选包发送到开发板：

```bash
xdg-open ~/.ros/ibrobot/calib/process/<capture-id>/test-overlay.png
scp ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar \
  <development-board-host>:~/.ros/ibrobot/calib/candidates/
```

处理流程包括：

1. 校验并解包原始数据。
2. 导出图像、CameraInfo 和点云。
3. 使用每个场景导出的 `CameraInfo` 覆盖 FAST-Calib 模板内参，检测相机与雷达中的标定板。
4. 使用 `scene-01` 至 `scene-03` 计算外参。
5. 使用独立的 `scene-04-test` 测试结果。
6. 从当前 profile 使用的 MID-360 mount 和 `scene-04-test` 导出的 CameraInfo 生成三份候选 YAML，并生成离线叠加图。

重点检查以下输出：

```text
<capture-id>.candidate.tar
base_to_front_camera.candidate.yaml
base_to_mid360.yaml
front_camera_intrinsics.yaml
test-overlay.png
calibration_summary.json
```

`calibration_summary.json` 中的状态应为 `candidate`。候选结果不会写入生产配置。

### 4. 板端实时验证

将 `<capture-id>.candidate.tar` 复制回板端后执行：

```bash
ros2 run robot_calibration calib_validate \
  --input ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar
```

验证命令会像 `calib_capture` 一样自动启动并管理 RealSense、MID-360、FAST-LIO、静态 TF 和底盘控制链，
同时启动实时 RGB/LiDAR 投影和验证 RViz。用户可直接在 RViz 中确认传感器是否正常出图。RViz 显示：

- `/calib/preview/image`：开发板本地限频到 8 Hz 并经 JPEG 传输的 RealSense 预览
- `/calib/preview/cloud`：开发板本地抽稀后的 MID-360 预览点云
- `/calib/overlay`：使用候选外参生成、限频到 5 Hz 并经 JPEG 传输的实时叠加图

检查 LiDAR 投影是否与标定板边缘和孔位一致，并观察不同距离和角度下是否存在固定偏移或旋转误差。
叠加图将 MID-360 最近 3 帧实测点合并显示，并将投影点放大到 2 像素以便观察对应关系。短时累积要求验证期间
机器人和周围场景保持静止，避免运动造成拖影。重复执行 `calib_validate` 时，第二个实例会直接拒绝启动，避免两套传感器图造成画面闪烁。

验证命令只在临时目录中解包候选标定，并使用候选包内的相机外参和 MID-360 mount 生成实时叠加。它不会写入
`~/.ros/ibrobot/calib/current/`，不会批准候选结果，也不会控制底盘。
验证结束时按 `Ctrl+C`。

### 5. 批准标定

确认实时投影正确后执行：

```bash
ros2 run robot_calibration calib_approve \
  --input ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar
```

批准流程会：

1. 检查输入文件必须以 `.candidate.tar` 结尾。
2. 检查相机外参、MID-360 mount 和相机内参三份标定均存在、格式有效且状态为 `candidate`。
3. 保留采集记录中的相机序列号；当前临时流程允许其为 `unavailable`。
4. 将三份标定转为 `approved` 并直接写入当前目录：

```text
~/.ros/ibrobot/calib/current/base_to_front_camera.yaml
~/.ros/ibrobot/calib/current/base_to_mid360.yaml
~/.ros/ibrobot/calib/current/front_camera_intrinsics.yaml
```

`calib_validate` 不会自动执行批准操作。`calib_approve` 也不会查询先前是否运行过实时验证；操作员必须先完成
PC 侧叠加检查，再执行批准。

当前允许 `serial: unavailable` 是为了完成现场链路验证，不构成设备绑定证据。下次加载
`lekiwi_sensor_calib` 配置时，`robot_config` 会读取状态为 `approved` 的 `current` 文件，并将其中
`base_link -> camera_front_optical_frame` 外参转换为现有相机 link/optical TF 链。

### 无图形界面的板端

如果板端没有 `DISPLAY` 或 Wayland，采集和验证命令不会尝试打开本地 RViz，但传感器数据和 `/calib/overlay` 仍会正常发布。

在网络可达、沿用当前工作区默认 ROS 2 环境且有图形环境的 ROS PC 上执行：

```bash
ros2 run robot_calibration calib_view --mode capture
ros2 run robot_calibration calib_view --mode validate
```

`capture` 模式显示低带宽 RealSense 和 MID-360 预览；`validate` 模式额外显示 `/calib/overlay`。

## 标定产物契约

机器人配置可以声明三类标定产物：

```yaml
robot:
  sensor_calibration:
    artifacts:
      base_to_front_camera: ~/.ros/ibrobot/calib/current/base_to_front_camera.yaml
      base_to_mid360: ~/.ros/ibrobot/calib/current/base_to_mid360.yaml
      front_camera_intrinsics: ~/.ros/ibrobot/calib/current/front_camera_intrinsics.yaml
    expected_devices:
      base_to_front_camera: REPLACE_WITH_CAMERA_SERIAL
      base_to_mid360: REPLACE_WITH_LIDAR_SERIAL
      front_camera_intrinsics: REPLACE_WITH_CAMERA_SERIAL
```

检查配置中声明的标定产物：

```bash
ros2 run robot_calibration calib_check <robot-config.yaml>
```

退出码含义：

- `0`：所有声明的产物通过格式、摘要、状态和预期设备检查。
- `1`：产物有效，但尚未全部批准，不能用于生产。
- `2`：配置或产物无效。

每个产物通过其 SHA-256 摘要标识，三个摘要共同构成稳定的标定包身份。
`calib_check` 检查产物合同；`lekiwi_sensor_calib` 启动配置实际消费其中的
`base_to_front_camera` approved 产物。

`config/examples/` 下的文件仅用于展示格式，状态为 `candidate`，不能直接激活到生产环境。

当前自动流程从同一次处理输入生成并批准三份标定产物。`base_to_mid360` 来自当前 robot profile 使用的
权威 mount 配置，`front_camera_intrinsics` 只来自独立 holdout 和离线 overlay 使用的
`scene-04-test/camera_info.yaml`。前三个训练场景仍各自使用其导出的 CameraInfo 进行检测；示例文件不作为测量输入。

## 人工验收要求

- 三个训练场景和独立 `scene-04-test` 均通过数值门限。
- 离线 `test-overlay.png` 中投影方向和尺度正确。
- 实时叠加在不同距离和角度下没有固定平移或旋转偏差。
- 验证期间机器人和场景保持静止，避免最近 3 帧点云累积产生拖影。
- 候选包中的设备身份、相机外参、MID-360 mount 和相机内参均符合现场设备。

## 诊断入口

`calib_offline` 用于开发者回放历史数据和定位检测、计算问题，不是现场用户入口。
离线诊断不会启动传感器、底盘控制器、Nav2 或速度指令发布器。

导入历史采集数据前，工具会校验传输清单、四个 MCAP 文件、四个 rosbag 元数据文件、必需 topic 和 9.5 至 11.0 秒的时长合同：

```bash
ros2 run robot_calibration calib_offline legacy-import <capture-dir> \
  --output ~/.ros/ibrobot/calib/raw \
  --lidar-serial <lidar-serial> --camera-serial <camera-serial>
```

导入过程会创建新的密封副本并记录历史传输清单摘要，不会修改原始目录。
`capture-export` 生成确定且包含完整载荷的归档；`capture-import` 校验所有文件摘要后再原子发布。

将四个密封场景解码为 RGB 图像、CameraInfo、累计点云和单消息 `/cloud_dense_body` rosbag：

```bash
ros2 run robot_calibration calib_offline export <sealed-capture> \
  --output <export-dir>
```

检测器只能使用源码提交和完整补丁状态均符合固定身份的 FAST-Calib 工作区。运行时会复制模板并绑定导出数据，不会原地修改仓库中的模板：

```bash
ros2 run robot_calibration calib_offline detect \
  --workspace <fast-calib-workspace> \
  --templates <scene-template-dir> \
  --exported <export-dir> --output <observation-dir>
```

生成四个 `observation.yaml` 后，使用场景 01 至 03 计算，使用场景 04 独立测试：

```bash
ros2 run robot_calibration calib_offline solve \
  --scene-01 <scene-01-observation.yaml> \
  --scene-02 <scene-02-observation.yaml> \
  --scene-03 <scene-03-observation.yaml> \
  --scene-04-test <scene-04-test-observation.yaml> \
  --output <extrinsic.yaml> --report <report.json>
```

默认训练和测试 RMSE 阈值均为 40 毫米。通过数值门槛只会生成 `candidate`，不会自动批准产物。

历史数据 `calib-19700101-000154` 没有记录 D435i 序列号。回归产物必须保留这一事实，不能使用其他采集的序列号补写。
当前临时批准流程允许该状态，但这种产物不能作为设备强绑定的生产证据。

## FAST-Calib 调优配置

计算器固定使用 `FAST-Calib_Ros2` 提交 `7747dfc6109c04b4bf81d2e3661e41626c8392e1`。
场景模板位于 `config/fast_calib/scenes/`，保存当前安装状态下的调优参数：

| 场景 | `body` ROI：`x`、`y`、`z`，单位米 | 平面阈值 |
| --- | --- | --- |
| `scene-01` | `[-2.0, 2.0]`、`[-5.2, -4.3]`、`[0.10, 1.60]` | `0.01` |
| `scene-02` | `[-2.0, 2.0]`、`[-5.2, -4.3]`、`[0.10, 1.60]` | `0.01` |
| `scene-03` | `[-2.0, 2.0]`、`[-5.2, -4.3]`、`[0.10, 1.60]` | `0.01` |
| `scene-04-test` | `[-2.0, 2.0]`、`[-5.2, -4.3]`、`[0.10, 1.60]` | `0.01` |

必须配套保持的参数：

- `marker_size=0.20 m`：相机检测使用的 ArUco 标记实际边长。
- `delta_width_qr_center=0.55 m`、`delta_height_qr_center=0.35 m`：计算器使用的二维码中心完整间距，不是孔间距。
- `delta_width_circles=0.50 m`、`delta_height_circles=0.40 m`：四个孔中心构成的完整矩形尺寸。
- `circle_radius=0.13 m`：完整平面缺口评分使用的孔半径。
- `prefer_vertical_plane=true`、`plane_eps_angle=0.35`：将标定板视为近似竖直平面。
- `normal_search_radius=0.06 m`、`boundary_search_radius=0.06 m`：法向量和边界搜索半径。
- `observation_only=true`、`lidar_topic=/cloud_dense_body`：使用 TF 归一化后的稠密点云生成观测，不读取原始 Livox 坐标。

`cluster_tolerance`、`min_cluster_size` 和 `circle_fit_*` 保留用于配置来源记录及补丁兼容。
当前实际候选选择使用完整平面评分，不由这些仅用于聚类的参数控制。

## 安装方向与 ROI

权威安装关系定义在 `robot_config/config/hardware/lekiwi_mid360_mount.yaml`：

```yaml
translation_m: [-0.08, 0.0, 0.20]
rpy_deg: [0.0, 0.0, 90.0]
```

当前物理安装对应 `base_link -> body` 偏航角 `+90` 度。同一安装来源派生
`base_link -> livox_frame`、`base_link -> body` 和反向 `body -> base_link`，不存在独立的 FAST-Calib 偏航参数。

`calib_process` 直接加载上述 `robot_config` 文件并将其转换为候选 `base_to_mid360.yaml`；
`config/examples/` 仅说明格式，不参与本次测量或求解。

在当前安装下，MID-360 的 `body -Y` 大致朝向车辆前方 `base_link +X`，因此标定板 ROI 位于 `body y` 负方向。
旧的正 Y、负 X ROI 和 `yaw=180` 度描述的是历史安装方向，不能与当前场景模板混用。

传感器发生移动、旋转或高度变化时：

1. 测量并更新 `robot_config` 中的安装平移和 `rpy_deg`，明确保持 `base_link -> body` 方向。
2. 在新的 `body` 坐标系下重新导出稠密点云，不能仅通过改变坐标符号复用旧 ROI。
3. 检查完整点云并更新每个场景的 `x_min/x_max`、`y_min/y_max` 和 `z_min/z_max`，使标定板位于带有少量实测余量的范围内。
4. 除非物理标定板尺寸发生变化，否则保持竖直平面、标定板几何和完整平面检测参数不变。
5. 重新运行三个训练场景和独立的 `scene-04-test`。
6. 如果无法得到明确的四孔矩形，应先检查 ROI、TF、标定板距离和尺寸，不能直接放宽阈值。
7. 将结果作为新的 `candidate` 版本，传感器安装变化后不能复用旧标定。

计算器不会自动搜索安装偏航角。错误的安装偏航会改变点云坐标系，并导致标定板离开配置的 ROI；工具不会通过静默扩大 ROI 来掩盖安装方向错误。
