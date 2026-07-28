# manipulation_execution

`manipulation_execution` 是抓取能力的闭环执行层。它把一次 `PickObject` action 请求编排为
GraspGen 规划、SO101 目标夹爪几何筛选、IK/FK 接触点补偿、安全 primitive 执行和抓后验证。

## 架构位置

```text
Hermes / robot_mcp
  -> /embodied/execute_skill (pick_object)
  -> skill_library
  -> /manipulation/execute_pick
  -> pick_executor_node
       -> /grasp_planner/plan_grasp
       -> /compute_ik + /compute_fk
       -> /embodied/execute_primitive
       -> /grasp_verifier/verify_grasp
```

Hermes 只看到一个原子技能。内部运动仍逐步经过 `skill_library` 和 `safety_guard`，本包不直接发布
控制器命令，也不直接调用 MoveIt 原始位姿接口。

`pick_executor_node.py` 只管理 ROS action、service client、TF 和 joint-state 生命周期。抓取行为按职责拆分到
`phases/flow.py`、`planning.py`、`preparation.py` 和 `execution.py`；共享状态模型与纯转换函数分别位于
`pick_executor_models.py` 和 `pick_executor_helpers.py`。阶段 mixin 保留原有方法入口，便于隔离测试且不改变
action 行为。

## 公开接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/manipulation/execute_pick` | `ibrobot_msgs/action/PickObject` | 抓取并验证一个运行时文本目标 |

目标字段 `target_query` 是 Grounded-SAM2/GraspGen 的文本查询，例如 `banana`，不是
`named_targets` 中的静态配置键。

反馈阶段包括 `preflight`、`observe`、`planning`、`selecting`、`approach`、`pregrasp`、`descend`、
`close`、`verify_close`、`probe_lift`、`verify_probe`、`lift` 和 `verify_lift`。

只有最终抓取验证成功时 action 才返回 `success=true`。动作执行完成但验证失败或不确定分别返回
`GRASP_VERIFICATION_FAILED` 或 `GRASP_UNCERTAIN`。

## 配置

所有运行参数来自 robot YAML 的 `robot.grasp_execution`。关键配置包括：

- GraspGen、验证器、IK/FK 服务名。
- 相机 topic 和 base/EE frame。
- source gripper 到目标夹爪的几何适配。
- 基于检测目标体积质心的 contact-distance 排序，以及 confidence/top-down 权重。
- approach、lift、速度、候选数量和物理执行重试次数；IK/FK 准备失败不消耗物理重试次数。
- `ik.worker_count` 个隔离 MoveIt worker 并行准备候选；所有 worker 使用同一份 `/joint_states` seed，
  按原候选顺序合并结果，并在每次抓取前校验主 MoveIt 与 worker 0 的 IK 解一致。
- SO101 tabletop 前置检查与监督式脚本相同，使用缓存的 mesh 凸包顶点和批量 NumPy 变换一次处理全部
  候选；固定姿态平移 sweep 只计算两个端点，临界阈值附近再回退到单候选精确检查。
- SO101 实际夹爪 STL 的 approach-to-grasp 桌面间隙，以及 joint5 分支修正后的 FK 接近轴/闭合轴硬校验。
- 候选 FK 残差重排、pregrasp 高位接触点 realign 和最终 IK/FK 接触点补偿阈值。
- IK/FK 准备完成后按候选规划姿态的固定爪前缘包络、目标体积质心距离、FK 接触点 XY/Z 质量和
  置信度做软重排；固定爪包络偏好不会单独硬拒绝候选。
- 非对称单动夹爪先按规划姿态检查固定爪是否位于朝机器人底座的一侧，再按最终 IK 解的 FK 姿态
  复检；最终姿态镜像或缺少目标宽度范围时，在产生运动前拒绝该候选并继续准备其他候选。
- commanded/actual 位姿和接触点 residual 诊断；有 planner 输出目录时写入 `pick_pose_diagnostics.json`。
- 抓取候选使用其 depth capture timestamp 查询 TF，不使用推理完成后的 latest TF；capture/latest/hand-eye
  变换写入 `pick_frame_diagnostics.json`。
- 最终软排序明细写入 `prepared_candidate_ranking.json`，包含固定爪实际/目标间隙、移动爪余量、
  质心距离和综合分数。
- 抓后验证策略。

`robot.embodied.skill_templates.pick_object` 负责把技能入口委托给本执行器：

```yaml
pick_object:
  executor: grasp_pipeline
  required_args: [target_name]
  timeout_sec: 240.0
```

## 安全边界

- 候选选定后，approach、pregrasp、contact realign、最终下降、probe lift 和 final lift 都沿用同一 joint5
  分支，并通过 `move_to_configuration` primitive 执行精确 IK 解；安全层检查完整关节顺序和限位。
- 抓取前检查所有必需服务，缺失时不产生运动。
- 最终 IK 解的 FK 固定爪朝向复检失败时，不执行该候选的 approach/descend/close。
- position-only IK 的 joint5 超出执行门限时，将 seed 翻转 `±π` 以交换固定指/活动指所在侧；最终 FK 接近轴、
  180° 对称闭合轴直线或固定指内侧检查失败时拒绝该候选，不再把 joint5 强制归零或停在 `±2.0` 边界。
- close 验证失败时闭爪撤回后再打开；probe/final retention 验证失败时在抬升位打开并返回观察位，与监督式
  test pipeline 的默认恢复策略一致。
- 桌面平面或 SO101 mesh 不可用时，目标夹爪 tabletop filter fail closed。
- 同一时间只接受一个抓取 goal，并将上游取消请求传给当前 primitive。
- worker pool 只用于无运动的候选 IK/FK 准备；最终接触补偿、分支锁定和全部机器人运动仍使用主 MoveIt。
- `MoveToConfiguration` 当前是同步 ROS service；取消会停止本地等待，但服务端运动不具备 action 级硬取消。

## 测试

```bash
source .shrc_local && source install/setup.bash && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --import-mode=importlib src/manipulation_execution/test
```
