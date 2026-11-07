# X2 语音、照镜子与抓取的统一配置

## 统一入口

310P 和 Orin 使用同一份机器人配置：

[`aimdk_x2_voice_mirror_grasp.yaml`](../src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml)。

启动参数名为 `robot_config:=aimdk_x2_voice_mirror_grasp`。
这个入口组合语音、手势、照镜子和抓取。
入口复用原来的技能目录 profile。
技能目录不保存安装侧、关节目标或模型服务地址。

本说明使用仓库配置，不需要另写现场 YAML。
下文的命令不会自动提交运动动作。

## 配置继承关系

总入口通过 `base_config` 逐层继承：

```text
aimdk_x2_voice_mirror_grasp.yaml
  -> aimdk_x2_voice_mirror_all.yaml
    -> aimdk_x2_voice_mirror.yaml
      -> aimdk_x2_magicmirror.yaml
        -> aimdk_x2_skills.yaml
          -> aimdk_x2.yaml
```

加载器递归合并父配置，子配置覆盖同名值。
总入口追加抓取模型服务，保留照镜子的视觉服务。
同名技能目录 profile 通过 `embodied.skill_catalog_profile` 引用。
任务姿态文件通过 `task_poses.file` 引用。
这两个文件不属于机器人配置的继承链。
各层职责和真机结果见[测试报告](x2_single_right_hand_pipeline_test_report_20261107_zh.md#2-yaml-继承与引用关系)。

## 单手与双手

修改总配置中的这 2 个字段：

```yaml
robot:
  grasp_execution:
    backend_options:
      installed_hands: [right]
      arm_side: right
```

| 硬件状态 | `installed_hands` | `arm_side` |
| --- | --- | --- |
| 只安装右手，当前默认值 | `[right]` | `right` |
| 只安装左手 | `[left]` | `left` |
| 安装双手，按目标侧选择抓取臂 | `[left, right]` | `auto` |
| 安装双手，只用右侧抓取 | `[left, right]` | `right` |

单手配置拒绝 `arm_side: auto`。
两端必须使用相同的安装声明。
“单手”指拆除一侧灵巧手，双臂仍存在。

总配置还包含以下声明：

```yaml
task_poses:
  file: ../task_poses/x2_example.yaml
  hands: installed
```

配置加载器根据安装声明选择开手动作。
配置加载器保留双臂关节目标和批准值。
配置加载器不根据缺失反馈推断安装侧。
修改手部安装声明后，无需另写任务姿态 YAML。

## 姿态批准与固定姿态库

准备位和休息位继续使用原来的
[`x2_example.yaml`](../src/robot_config/config/task_poses/x2_example.yaml)。
任务姿态的批准默认值仍为 `false`。
真实运动使用本机已核验的关节目标与批准值。
批准值不代表姿态已经满足到位要求。

休息位 `arms_rest` 的左肩侧摆为 `+0.08 rad`。
右肩侧摆为 `-0.08 rad`。
两侧目标距内侧限位均为 `0.141 rad`。
限位余量大于到位容差 `0.1 rad`。
修改目标后，重新编译并同步两端固定姿态库。

310P 使用 `pose_composer.mode: fixed`。
运行时读取离线编译的姿态库，不加载 MuJoCo。
MuJoCo 是工作站使用的物理仿真软件。
默认资产路径为 `commissioning/fixed_poses.json`。
路径以当前工作区的 `WORKSPACE` 为根目录。
仓库忽略生成的资产，避免提交厂商派生几何。

修改安装侧、任务姿态、批准值或关节限制后，重新编译姿态库。
编译器从同一份总配置读取这些值。
运行时拒绝摘要不匹配的旧姿态库。
切换单手或双手无需修改 Python 代码。

前置条件：工作站已配置项目环境与官方 X2 模型。
模型获取入口为 `aimdk_fetch_x2_mjcf`。

1. 在工作站编译候选姿态库。

   ```bash
   source .shrc_local
   ros2 run pose_composer pose_compile_library \
     --robot-config src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml \
     --hand-path-check off \
     --pose arms_down --pose arms_forward --pose arms_up --pose arms_out \
     --pose arms_v --pose flex --pose surrender --pose right_hand_on_head \
     --pose salute --pose thinking --pose right_forward --pose left_forward \
     --output commissioning/fixed_poses.candidate.json
   ```

2. 核对候选姿态库与两端配置。
3. 停止两端旧 pipeline。
4. 将候选库部署为两端的 `commissioning/fixed_poses.json`。

`--hand-path-check off` 只限定离线编译的证明范围。
编译器仍校验手部参考入口、身体路径和关节限制。
这个参数不改变运行时策略。
总配置沿用现场调试策略 `path_validation: {hand: "off", body: "off"}`。
执行器保留入口、端点和执行准入检查，但不检查完整路径。
需要在线检查完整路径时，将 `hand` 和 `body` 都改为 `online`，再重启两端。

现有 `0.1 rad` 手指保持范围可能触发路径拒绝。
拒绝不会自动放宽路径检查。
执行细节见
[固定姿态说明](../src/agent/pose_composer/README.md#fixed-seed-library-on-devices-without-mujoco)。

## 启动前置条件

两端已构建本分支的 `robot_runtime`、`robot_config` 和 `aimdk_grasp`。
310P 还需构建 Agent 栈与 `pose_composer`。
两端已初始化厂商软件开发工具包（SDK）环境。
两端使用相同的通信域。

Orin 的相机驱动已启动。
两端配置、任务姿态和固定姿态库已同步。
同一端只运行一套 pipeline。

310P 还需提供以下模型：

- 总配置声明的 X-ASR 与 ZipVoice。
- 照镜子的 YOLOX 与 PEAR。
- 抓取的 GroundingDINO、SAM2 与 GraspGen-X。

Orin 还需提供 CUDA、cuRobo 和抓取模型描述。
声纹功能默认关闭，无需加载声纹模型。
自我介绍录音位于 `commissioning/audio/self_introduction.wav`。

Agent 模型服务默认地址为 `http://10.42.0.233:8080/v1`。
默认模型为 `qwen3.8-flash-next-iq3_s`。
启动环境通过 `X2_PLANNER_API_KEY` 提供密钥。
仓库配置只保存环境变量名。
地址变化时，使用启动参数 `planner_base_url:=<地址>`。
模型名称变化时，修改总配置的 `embodied.agent.planner.model`。

## 可选文本唤醒

标准唤醒词为“欧拉欧拉”。
总 YAML 的 `wake_phrases` 列出 16 个同音、近音前缀。
这些前缀由“欧拉、欧啦、鸥拉、偶拉”两两组合。
规则只匹配句首，并在转交 Agent 前剥离前缀。
单次“欧拉”、句中提及和“好啦好啦”不触发唤醒。

总配置已启用文本唤醒。
删除 `wake_phrases` 键可恢复直接接收普通语音。
仅启用文本唤醒不需要声纹模型。

总配置沿用厂商处理后的 `external_vad` 和 `audio.speech` 输入。
声纹校验使用独立开关，详见
[语音配置说明](../src/robot_config/README.md#x2-镜子抓取可选欧拉文本唤醒与声纹)。

## 启动两端

以下命令均从本端部署的项目根目录执行。
这些命令启用真实硬件与运动授权。

1. 在 Orin 启动抓取头。

   ```bash
   source .shrc_local
   export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
   ros2 launch aimdk_grasp x2_grasp_head.launch.py \
     robot_config:=aimdk_x2_voice_mirror_grasp
   ```

2. 在 310P 启动主管线。

   ```bash
   source .shrc_local
   export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
   ros2 launch embodied_bringup embodied_pipeline.launch.py \
     robot_config:=aimdk_x2_voice_mirror_grasp \
     control_mode:=model_inference with_moveit:=false \
     use_sim:=false authorize_motion:=true
   ```

`robot_config` 按已安装包查找 YAML。
修改源码后，先通过 `scripts/build.sh` 刷新配置安装。
也支持显式选择源码文件：

```bash
config_path:="$PWD/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
```

`config_path` 优先于 `robot_config`。
使用源码路径时，安装路径仍需包含本分支的软件包。
两端通过同一个总入口加载安装侧与抓取后端。
Orin 的 launch 只启动抓取头，不重复启动主管线。
