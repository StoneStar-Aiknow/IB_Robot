# manipulation_service

`manipulation_service` 提供基于 GraspGen 的 6-DOF 抓取规划能力。该包负责 ROS 2
服务封装、深度图和 mask 转换、GraspGen 调用、抓取结果发布，以及在线/离线
调试产物导出；不负责目标检测、机器人控制、MoveIt 执行、策略推理或数据集转换。

## 入口与接口

节点/工具：

- `grasp_planner_node`：在线抓取规划节点。订阅对齐深度和 CameraInfo，
  调用 `perception_service` 获取目标 mask，运行 pip 安装的 GraspGen，提供
  `~/plan_grasp` 服务，并发布 `~/grasps`。
- `test_graspgen.py`：离线 GraspGen 调试脚本。读取 `grounded_sam2_snapshot`
  生成的数据目录，直接运行 GraspGen，并保存 PLY、JSON 和可选 Open3D 视图。

ROS 接口：

- 服务：`ibrobot_msgs/srv/PlanGrasp`
- 话题：`ibrobot_msgs/msg/GraspCandidateArray`
- 单个抓取结果：`ibrobot_msgs/msg/GraspCandidate`
- 感知依赖：`ibrobot_msgs/srv/DetectSegment`

`GraspCandidate` 只包含机器人无关的抓取候选信息。除了 GraspGen 位姿、置信度和碰撞标记，
节点还会基于分割目标点云估算每个候选沿源夹爪闭合轴方向的 `target_width_m`、
`target_width_quality` 和 `width_axis_camera`。这些字段用于下游机器人执行层做自己的夹爪几何适配；
`manipulation_service` 不读取任何 SO101 配置，也不输出 SO101 专用 offset。

默认在线话题遵循 SO101 wrist RealSense 抓取流水线约定：

- 对齐深度：`/camera/wrist/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 检测服务：`/grounded_sam2/detect_and_segment`

RealSense 顶置相机调试时常用的话题为：

- 对齐深度：`/camera/camera/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/camera/color/camera_info`

## 环境与依赖

当前路径使用 pip 安装的 GraspGen：`manipulation_service` 在同一 Python 进程中导入
`grasp_gen`，并用 CUDA 执行推理。GraspGen 不再放在 `libs/` 下；安装脚本会把
固定上游源码作为 editable pip 依赖放到 workspace venv 的 `src/` 缓存中。

运行前需要满足：

- 当前环境可用 CUDA PyTorch；上游 `GraspGenSampler` 内部会把模型和点云移到 CUDA。
- 已通过 `./scripts/setup.sh --with-grasp` 安装 `grasp_gen` 和 `pointnet2_ops`。
- GraspGen 模型文件在 `models/grasp/`，或通过 `GRASPGEN_MODEL_DIR` 指定。
- 已构建 `manipulation_service` 和 `ibrobot_msgs`。

所有 ROS 调试命令都应在仓库根目录运行，并在同一条命令里完成环境初始化：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && <ros2 command>
```

修改代码或首次运行前，先构建接口和本包：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select ibrobot_msgs manipulation_service
```

## 调试 grasp_planner_node

### 1. 单独运行在线 GraspGen 抓取节点

运行前需先启动相机和 `perception_service` 的 `grounded_sam2_node`。默认使用
SO101 wrist camera 话题：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node
```

默认使用 GraspGen Robotiq 2F-140 作为 source gripper 生成候选，并用同一 source
gripper collision mesh 执行普通场景碰撞过滤和桌面 clearance 过滤。

使用 RealSense 顶置相机，并开启调试输出：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p save_debug_outputs:=true \
  -p debug_output_dir:=outputs/realsense_grasp_debug
```

目标宽度估计参数保持机器人无关，默认适配当前 GraspGen Robotiq 2F-140 配置：

```bash
-p target_width_axis_local:=1.0,0.0,0.0 \
-p target_width_percentile_low:=5.0 \
-p target_width_percentile_high:=95.0 \
-p target_width_min_m:=0.005 \
-p target_width_max_m:=0.14
```

如果更换 GraspGen 模型夹爪，只需把 `target_width_axis_local` 改为该源夹爪局部坐标系下的闭合轴，
不要在本包中加入机器人型号判断。

如果目标只是确认 GraspGen 是否能产生候选，可临时关闭几何过滤：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p save_debug_outputs:=true \
  -p debug_output_dir:=outputs/realsense_grasp_debug \
  -p enable_collision_filter:=false \
  -p enable_tabletop_filter:=false \
  -p require_tabletop_filter:=false
```

### 2. 调用抓取规划服务

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp "{text_prompt: 'banana', confidence_threshold: 0.10, grasp_threshold: 0.5, debug_output_mode: 'default'}"
```

请求字段：

- `text_prompt`：目标名称，会转发给 `DetectSegment`，例如 `banana`。
- `confidence_threshold`：感知检测阈值；大于 `0` 时覆盖 perception 节点默认值。
- `grasp_threshold`：GraspGen discriminator 置信度阈值；大于 `0` 时覆盖节点参数。
- `debug_output_mode`：单次请求调试输出模式。空字符串或 `default` 沿用节点
  `save_debug_outputs` 参数；`none` 不写文件；`diagnostic` 只写 `grasp_result.json`；
  `full` 写 `grasp_result.json`、点云 PLY、gripper mesh/line 和 PNG/HTML 预览。

响应重点看：

- `success` / `message`：是否规划成功和失败原因。
- `diagnostic_details`：规划诊断明细；没有生成抓取点时优先查看
  `failure_stage`、`failure_reason`、mask/depth 点数、GraspGen 原始候选数、
  collision/tabletop 过滤前后数量。
- `debug_output_dir`：本次请求实际写调试文件时返回输出目录；未写文件时为空。
- `grasps.grasps[*].pose_matrix`：每个抓取位姿的 4x4 row-major 矩阵。
- `grasps.grasps[*].confidence`：GraspGen 置信度。
- `grasps.grasps[*].collision_free`：当前过滤配置下是否认为无碰撞。

`manipulation_service` 返回的是机器人无关的 GraspGen 候选，排序主要来自 GraspGen
置信度和几何过滤结果。目标机器人专用的目标夹爪宽度补偿、IK/workspace 过滤和
接触点重对齐属于执行层后处理，当前由
`scripts/test_banana_handeye_pick.py` 完成。该脚本从
`robot_config.robot.grasp_execution` 读取 `target_gripper` 和 `execution_scoring`，
因此 GraspGen 不需要绑定 SO101；新增机器人只需要在自己的 robot_config 中定义目标夹爪几何和评分权重。

常见 `diagnostic_details` 字段：

- `failure_stage` / `failure_reason`：失败阶段和具体原因；成功时通常不存在。
- `detection_confidence`：perception 返回的最佳目标置信度。
- `mask_pixel_count`：对齐到深度图后的目标 mask 像素数；为 `0` 表示 mask 为空。
- `valid_depth_in_mask_count` / `valid_depth_ratio_in_mask`：目标 mask 内有效深度点数和比例；
  为 `0` 时通常是 RGB/depth 未对齐、目标深度空洞、深度编码或尺度异常。
- `object_point_count` / `scene_point_count`：传入 GraspGen/过滤器的目标和场景点数。
- `raw_grasp_count`：GraspGen 原始候选数量；为 `0` 表示模型未产生候选，
  可尝试降低 `grasp_threshold` 或检查目标几何是否适合当前夹爪。
- `collision_filter`：碰撞过滤后/前的候选数量，例如 `0/80` 表示所有候选碰撞。
- `tabletop_plane_found` / `tabletop_best_inlier_ratio`：是否找到桌面平面及最佳内点比例。
- `tabletop_filter`：桌面 clearance 过滤后/前的候选数量。
- `tabletop_auto_tuned` / `tabletop_auto_tune_reason`：低矮目标 adaptive retry
  是否生效，以及 retry 成功或跳过原因。
- `final_grasp_count`：最终返回的抓取数量。

### 3. 订阅抓取结果

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 topic echo /grasp_planner/grasps
```

该话题发布最近一次服务调用的抓取结果，便于检查 frame_id、pose matrix 和置信度。

## grasp_planner_node 参数说明

模型和输入：

- `device`：推理设备，默认 `cuda`。
- `gripper_config`：GraspGen 模型配置，默认 `graspgen_robotiq_2f_140.yml`。
- `model_dir`：GraspGen 模型根目录；空字符串表示使用默认模型目录。
- `depth_topic`：对齐深度图输入。
- `camera_info_topic`：与深度/RGB 对应的相机内参。
- `detect_service`：目标检测分割服务，默认 `/grounded_sam2/detect_and_segment`。
- `depth_scale`：整数深度转米比例，默认 `1000.0`；浮点深度会自动按米处理。

抓取生成：

- `grasp_threshold`：GraspGen discriminator 阈值，默认 `0.5`。
- `num_grasps`：每轮生成的候选数量，默认 `800`。
- `topk_num_grasps`：每轮保留的 top-K 候选，默认 `50`。

输入同步：

- `input_buffer_size`：深度/CameraInfo 缓冲帧数，默认 `30`。
- `sync_max_age_sec`：mask 时间戳与深度/内参允许的最大时间差，默认 `0.20` 秒。

碰撞过滤：

- `enable_collision_filter`：启用 GraspGen 碰撞过滤，默认 `true`。
- `collision_threshold`：碰撞距离阈值，单位米，默认 `0.005`。
- `collision_gripper`：碰撞检测和可视化用 gripper 名称；空字符串表示沿用模型 gripper。

桌面过滤：

- `enable_tabletop_filter`：启用桌面平面过滤，默认 `true`。
- `require_tabletop_filter`：找不到可接受桌面平面时返回空结果，默认 `true`。
- `tabletop_clearance`：夹爪 mesh 到桌面的最小距离，默认 `0.003` 米。
- `tabletop_pregrasp_distance`：沿 GraspGen/source gripper approach 轴后退检查距离，默认 `0.08` 米。
- `tabletop_pregrasp_steps`：pre-grasp 到 final grasp 中间检查步数，默认 `5`。
- `tabletop_ransac_threshold`：桌面 RANSAC 内点距离阈值，默认 `0.006` 米。
- `tabletop_min_inlier_ratio`：接受桌面平面的最小内点比例，默认 `0.15`。
- `tabletop_filter_mode`：桌面过滤模式，默认 `strict`。`adaptive` 会对草莓等低矮目标
  按目标高度自适应降低 clearance 和 pre-grasp 检查距离；`soft` 在低矮目标严格过滤为
  空时，允许保留不低于 hard floor 的风险最低候选，并在诊断中标记 `tabletop_relaxed`。
  桌面过滤始终对 final pose 到 pre-grasp path 的所有采样点取最小 clearance，并以该
  最小值作为硬过滤条件。
- `adaptive_tabletop_low_profile_height`：低矮目标高度阈值，默认 `0.035` 米。
- `adaptive_tabletop_clearance_min` / `adaptive_tabletop_clearance_max`：自适应 clearance
  下限/上限，默认 `0.001` / `0.002` 米。
- `adaptive_tabletop_clearance_height_ratio`：低矮目标 clearance 与目标高度的比例，默认 `0.12`。
- `adaptive_tabletop_pregrasp_min`：自适应 pre-grasp sweep 距离下限，默认 `0.02` 米。
- `adaptive_tabletop_pregrasp_height_ratio`：pre-grasp sweep 距离与目标高度的比例，默认 `2.0`。
- `adaptive_tabletop_hard_floor`：`soft` 模式低矮目标 fallback 允许的最低桌面距离，默认 `-0.002` 米。
- `adaptive_tabletop_auto_tune`：`adaptive` 模式低矮目标被桌面过滤清空时自动 retry，默认 `true`。
- `adaptive_tabletop_retry_clearances`：auto-tune 依次尝试的 clearance 列表，逗号分隔字符串，
  默认 `0.002,0.001` 米；只会尝试低于当前 adaptive clearance 的正值。
- `adaptive_tabletop_retry_pregrasp_height_ratio`：auto-tune retry 的 pre-grasp sweep 距离与目标高度比例，
  默认 `1.5`。
- `adaptive_tabletop_retry_hard_floor`：auto-tune 允许 retry 的最低首轮候选 clearance，默认 `-0.003` 米；
  低于该值视为明显穿桌，不自动放宽。

抓取草莓等贴桌低矮目标时，推荐先用 `adaptive` 模式，不要直接关闭桌面过滤：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p tabletop_filter_mode:=adaptive \
  -p adaptive_tabletop_low_profile_height:=0.035 \
  -p adaptive_tabletop_clearance_min:=0.001 \
  -p adaptive_tabletop_clearance_max:=0.002 \
  -p adaptive_tabletop_pregrasp_min:=0.02
```

`adaptive_tabletop_auto_tune` 默认开启。若首轮 adaptive 因低矮贴桌目标被过滤为空，节点会在
`tabletop_best_candidate_clearance_m` 未明显低于桌面的前提下，自动尝试更小 clearance 和更短
pre-grasp path，并在诊断中记录 `tabletop_auto_tuned`、`tabletop_auto_tune_attempts` 和
`tabletop_auto_tune_reason`。

如果 auto-tune 后仍把所有候选过滤为空，可临时使用 `soft` 做真机前诊断。此时必须检查
`grasp_result.json` 中 `tabletop_relaxed`、`tabletop_best_candidate_clearance_m`、
`tabletop_auto_tune_reason` 和预览图，确认夹爪没有明显穿桌：

```bash
-p tabletop_filter_mode:=soft -p adaptive_tabletop_hard_floor:=-0.002
```

在线调试输出：

- `save_debug_outputs`：默认是否为请求写完整调试产物，默认 `false`；单次请求可用
  `debug_output_mode` 覆盖。
- `debug_output_dir`：输出根目录，默认 `outputs/grasp_pipeline`。
- `debug_max_save_grasps`：保存 gripper mesh/line 的 top grasp 数量，默认 `10`。
- `debug_render_preview`：是否后台生成 PNG/HTML 预览，默认 `true`。
- `debug_render_show_scene`：PNG 中显示整场景点云而非仅目标点云，默认 `false`。
- `debug_render_labels`：是否生成带 index/confidence/depth 标签的 PNG，默认 `true`。
- `debug_render_max_points`：PNG 预览最大采样点数，默认 `60000`。
- `debug_render_width` / `debug_render_height`：PNG 尺寸，默认 `1280` / `720`。
- `debug_render_interactive`：是否生成自包含交互式 HTML，默认 `true`。
- `debug_interactive_show_scene`：HTML 是否包含 scene 点云 trace，默认 `true`。
- `debug_interactive_max_points`：HTML 中 object + scene 最大嵌入点数，默认 `120000`；
  设为 `0` 表示嵌入全部点，文件会很大。

## 在线调试产物

当节点参数 `save_debug_outputs:=true`，或单次请求设置 `debug_output_mode` 为
`diagnostic` / `full` 时，请求会在
`<debug_output_dir>/<timestamp>_<prompt>_<time-ns>/` 下写入调试产物，并在响应的
`debug_output_dir` 字段返回该目录。

`debug_output_mode` 行为：

- `default` 或空字符串：沿用节点 `save_debug_outputs` 参数；参数为 `true` 时等价于
  `full`，为 `false` 时等价于 `none`。
- `none`：不写任何调试文件。
- `diagnostic`：只写 `grasp_result.json`，不生成点云、gripper mesh/line 或预览图。
- `full`：写 `grasp_result.json`、点云 PLY、gripper mesh/line 和 PNG/HTML 预览；如果
  请求在检测或同步阶段提前失败，则只能写诊断 JSON。

完整输出包含：

- `grasp_result.json`：prompt、阈值、相机内参、诊断字段、每个 grasp 的 pose、
  confidence 和点云元数据；即使最终抓取数量为 `0` 也会记录诊断。
- `object_cloud.ply`：目标 mask 对应点云。
- `scene_cloud.ply`：非目标场景点云，用于碰撞上下文。
- `grasp_cloud.ply`：目标 + 场景点云。
- `grasp_grippers.ply`：top grasp 的夹爪 collision mesh。
- `grasp_lines.ply`：top grasp 的夹爪控制点线框。
- `grasp_preview.png`：headless 2D 投影预览，夹爪颜色表示 confidence。
- `grasp_preview_labeled.png`：带 grasp index、confidence 和 depth 标签的 PNG。
- `grasp_preview.html`：自包含交互式 3D 视图，包含 object/scene 点云、
  confidence 着色夹爪线框和 grasp pose hover 信息。
- `grasp_preview_meta.json`：预览渲染状态、采样点数和错误信息。

PNG/HTML 渲染在后台 daemon thread 中执行；service response 不等待图像导出。
因此 `grasp_result.json` 可能先出现，预览文件稍后出现。

## 调试 test_graspgen.py

`test_graspgen.py` 不依赖 ROS service。它读取 `grounded_sam2_snapshot` 生成的
`result.json`、mask、深度和 CameraInfo，直接运行 GraspGen，适合单独验证 GraspGen
模型、过滤器和可视化。

基础命令：

```bash
source .shrc_local && python3 src/manipulation_service/test_graspgen.py \
  --data-dir outputs/grounded_sam2/<timestamp>_banana
```

显示交互式 Open3D 视图和分数标签：

```bash
source .shrc_local && python3 src/manipulation_service/test_graspgen.py \
  --data-dir outputs/grounded_sam2/<timestamp>_banana \
  --show \
  --show-scores
```

只验证 GraspGen 候选生成，临时关闭桌面过滤：

```bash
source .shrc_local && python3 src/manipulation_service/test_graspgen.py \
  --data-dir outputs/grounded_sam2/<timestamp>_banana \
  --no-enable-tabletop-filter \
  --no-require-tabletop-filter
```

常用参数：

- `--data-dir`：`grounded_sam2_snapshot` 输出目录。
- `--show`：打开 Open3D 交互式 3D 视图；无桌面环境时不要加。
- `--show-scores`：在可视化中显示 confidence 标签。
- `--show-scene`：显示完整 scene 点云；默认只显示目标点云，更清晰。
- `--collision-gripper`：碰撞过滤和可视化使用的 gripper 名称；默认沿用模型 gripper。
- `--max-save-grasps`：写入 PLY 的 top grasp 数量，默认 `10`。
- `--num-grasps`：每轮生成候选数，默认 `800`。
- `--topk-num-grasps`：每轮保留 top-K，默认 `50`。
- `--min-grasps`：累计达到该数量后停止，默认 `5`。
- `--max-tries`：最多推理轮数，默认 `4`。
- `--grasp-threshold`：GraspGen discriminator 阈值，默认 `0.5`。
- `--collision-threshold`：碰撞距离阈值，单位米，默认 `0.005`。
- `--depth-scale`：整数深度转米比例，默认 `1000.0`。
- `--enable-tabletop-filter` / `--no-enable-tabletop-filter`：打开/关闭桌面过滤。
- `--require-tabletop-filter` / `--no-require-tabletop-filter`：找不到桌面平面时是否返回空结果。
- `--tabletop-clearance`：夹爪到桌面的最小距离，默认 `0.003` 米。
- `--tabletop-pregrasp-distance`：pre-grasp 后退检查距离，默认 `0.08` 米。
- `--tabletop-pregrasp-steps`：pre-grasp sweep 检查步数，默认 `5`。
- `--tabletop-ransac-threshold`：桌面平面 RANSAC 阈值，默认 `0.006` 米。
- `--tabletop-min-inlier-ratio`：接受桌面平面的最小内点比例，默认 `0.15`。
- `--tabletop-filter-mode`：`strict` / `adaptive` / `soft`，用于验证低矮目标的自适应桌面过滤。
- `--adaptive-tabletop-*`：与在线节点同名的自适应桌面过滤参数。
- `--adaptive-tabletop-auto-tune` / `--no-adaptive-tabletop-auto-tune`：打开/关闭 adaptive retry。
- `--adaptive-tabletop-retry-clearances`：离线 auto-tune retry clearance 列表，例如 `0.002,0.001`。
- `--fx --fy --cx --cy`：手动指定相机内参；仅旧数据目录缺少 CameraInfo 时需要。

输出目录为 `<data-dir>/graspgen_output/`，包含：

- `grasp_result.json`：GraspGen 诊断字段、每个 grasp 的 pose、confidence 和 collision 状态；
  即使最终抓取数量为 `0` 也会写入。
- `object_cloud.ply` / `scene_cloud.ply` / `grasp_cloud.ply`：目标、场景和合并点云。
- `grasp_grippers.ply`：top grasp 的夹爪 mesh。
- `grasp_lines.ply`：top grasp 的夹爪线框。
- `grasp_preview.png`：离线预览截图。

注意：离线预览中的编号仍是 GraspGen 服务返回顺序，不包含 SO101 执行脚本里的
接触点质心重排、动态宽度补偿或 IK/workspace 过滤结果。若要确认最终会抓哪个候选，
以 `scripts/test_banana_handeye_pick.py --detect-only` 输出的 `GRASPGEN_RANK` 和
`GRASPGEN_CANDIDATE_ACCEPT` 日志为准。

## 常见调试路径

### 只调 perception，不调 GraspGen

在 `perception_service` 中运行 `grounded_sam2_node` 和 `grounded_sam2_snapshot`，
确认 `overlay.png`、mask 和点云正确后，再进入本包调 GraspGen。

### 只调 GraspGen，不启动 ROS service

使用已有 `grounded_sam2_snapshot` 输出目录运行 `test_graspgen.py`。这样可以排除
ROS topic 同步、service timeout 和在线 perception 的影响。

### 调完整在线链路

1. 启动相机。
2. 启动 `perception_service` 的 `grounded_sam2_node`。
3. 启动 `grasp_planner_node`，建议先开启 `save_debug_outputs`。
4. 调用 `/grasp_planner/plan_grasp`。
5. 查看输出目录中的 `grasp_preview_labeled.png` 和 `grasp_preview.html`。

## 排障

- `DetectSegment service not available`：perception 节点未启动，或 `detect_service`
  参数不对。
- `No synchronized depth/CameraInfo`：mask 时间戳与深度/内参差距超过
  `sync_max_age_sec`，检查相机话题和时间戳。
- `No grasps generated`：先查看 `diagnostic_details` 或 `grasp_result.json` 中的
  `failure_stage` 和 `failure_reason`，再按阶段处理。
- `failure_stage: point_cloud`：查看 `mask_pixel_count`、`valid_depth_in_mask_count`
  和 `valid_depth_ratio_in_mask`。mask 为空说明 perception 没有有效分割；mask 内
  有效深度为 `0` 时检查 RGB/depth 对齐、RealSense 深度空洞、深度编码和 `depth_scale`。
- `failure_stage: graspgen_inference`：GraspGen 没有原始候选，先降低
  `grasp_threshold`，再检查目标局部几何是否适合当前夹爪。
- `failure_stage: collision_filter`：所有候选被场景碰撞过滤，临时关闭
  `enable_collision_filter` 可确认模型是否本身有候选；同时检查 `scene_cloud.ply`
  是否包含过多目标附近噪点。
- `failure_stage: tabletop_filter`：查看 `tabletop_best_inlier_ratio`、`tabletop_failure_reason`、
  `tabletop_filter`、`tabletop_auto_tuned`、`tabletop_auto_tune_reason`、
  `scene_cloud.ply` 是否包含桌面，必要时调整 `tabletop_ransac_threshold`、
  `tabletop_min_inlier_ratio`、`tabletop_clearance` 或 `tabletop_pregrasp_distance`。
- HTML/PNG 未立刻出现：预览在后台线程生成，等待几秒后查看
  `grasp_preview_meta.json`。
- `ModuleNotFoundError`：未执行 `source .shrc_local`，或 GraspGen 依赖未安装。

## 架构边界

- 机器人和相机拓扑应保留在 `robot_config` YAML 中。
- 本包不拥有 perception 模型推理、机器人控制、MoveIt 轨迹执行、策略推理或数据集转换。
- GraspGen 作为可选重量级运行依赖处理；若后续增加 server backend，ROS service
  合约应保持稳定。
