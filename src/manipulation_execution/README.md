# manipulation_execution

`manipulation_execution` 是抓取能力的闭环执行层。它把一次 `PickObject` action 请求编排为
GraspGen 规划、SO101 目标夹爪几何筛选、IK/FK 接触点补偿、安全 primitive 执行和抓后验证。

本包也是抓取行为的唯一实现源。监督式真机测试、批量实验和 Hermes 均发送同一个
`/manipulation/execute_pick` action；`scripts/test_banana_handeye_pick.py` 只保留兼容 CLI，
不再运行自己的候选筛选或物理执行状态机。

旧监督式执行器及其 legacy-only replay benchmark 已删除。所有行为测试直接覆盖正式 phases 和纯几何模块；
`test_pick_single_source.py` 会在 CI 中确保监督式入口始终只是 action 客户端薄包装，且不会重新引入旧执行器。

## 架构位置

```text
Hermes / ibrobot-control / robot-skill
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

当抓取由 Gateway 的 `pick_object` skill 启动时，`PickObject` action goal UUID 会作为不透明的内部
执行授权传入每个 `PrimitiveCommand.execution_token`。`skill_library` 据此让 pipeline primitive 借用
父 skill 的 root lease，避免观察位动作被误判为并发外部请求并返回 `SKILL_BUSY`；直接调用
`PickObject` 时没有对应 Gateway 注册，primitive 仍按外部请求逐个准入。

`pick_executor_node.py` 只管理 ROS action、service client、TF 和 joint-state 生命周期。抓取行为按职责拆分到
`phases/flow.py`、`planning.py`、`preparation.py` 和 `execution.py`；共享状态模型与纯转换函数分别位于
`pick_executor_models.py` 和 `pick_executor_helpers.py`。`pipeline_worker.py` 只提供与机器人无关的动态 worker
队列；SO101 joint5 分支、归一化、闭合轴修正和 IK 重试集中在 `so101_kinematics_guard.py`，
不混入通用 pipeline 工具。阶段 mixin 保留原有方法入口，便于隔离测试且不改变 action 行为。

## 公开接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/manipulation/execute_pick` | `ibrobot_msgs/action/PickObject` | 抓取并验证一个运行时文本目标 |

`PickObject.mode` 支持 `MODE_EXECUTE`、`MODE_PLAN_ONLY` 和 `MODE_OBSERVE_ONLY`。三个模式共享同一
观测、规划和配置入口；plan-only 会完成正式候选筛选与 IK/FK 准备但不执行抓取。测试批次需要在成功后
释放目标时，可通过 goal 的 `release_after_success` 和 `release_drop_height_m` 请求正式 executor 在验证
成功后执行安全下降和开爪。

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
- `ik.worker_count` 个隔离 MoveIt worker 从同一个动态队列准备候选；所有 worker 使用同一份
  `/joint_states` seed，空闲 worker 立即领取下一个候选，结果仍按原候选顺序合并，并在每次抓取前校验
  主 MoveIt 与全部 worker 的 IK 解一致。
- `candidate_selection.selection_attempts` 控制整批候选无安全 IK 解或规划暂时失败时重新获取新帧并规划；
  不可恢复的配置、安全门禁和 worker 一致性错误不会重试。
- SO101 tabletop 前置检查由正式 pipeline 的公共实现完成；相机平面变换到 base 后先将法向规范为朝向
  base +Z 的安全半空间，再使用缓存的 mesh 凸包顶点和批量 NumPy 变换处理全部候选。固定姿态平移 sweep
  只计算两个端点，临界阈值附近再回退到单候选精确检查。
- SO101 实际夹爪 STL 的 approach-to-grasp 桌面间隙，以及 joint5 分支修正后的 FK 接近轴/闭合轴硬校验。
- 候选 FK 残差重排、pregrasp 高位接触点 realign 和最终 IK/FK 接触点补偿阈值。
- IK/FK 准备完成后按候选规划姿态的固定爪前缘包络、目标体积质心距离、FK 接触点 XY/Z 质量和
  置信度做软重排；固定爪包络偏好不会单独硬拒绝候选。
- 非对称单动夹爪先按规划姿态检查固定爪是否位于朝机器人底座的一侧，再按最终 IK 解的 FK 姿态
  复检；最终姿态镜像或缺少目标宽度范围时，在产生运动前拒绝该候选并继续准备其他候选。
- 固定指 robust gap 使用最终 IK/FK 预测接触残差在 approach/descend 前完成硬门禁；下降成功后进入
  commit-to-grasp 状态，`close_gripper` 是下降成功后的第一条动作。低位实测改在闭爪后以 best-effort
  方式记录；诊断异常不影响闭爪和后续验证，也不再回到 pregrasp 或切换候选。
- commanded/actual 位姿和接触点 residual 诊断；有 planner 输出目录时写入 `pick_pose_diagnostics.json`。
- 抓取候选使用其 depth capture timestamp 查询 TF，不使用推理完成后的 latest TF；capture/latest/hand-eye
  变换写入 `pick_frame_diagnostics.json`。
- 最终软排序明细写入 `prepared_candidate_ranking.json`，包含固定爪实际/目标间隙、移动爪余量、
  质心距离和综合分数。
- 抓后验证策略。
- `PickObject.Result.pipeline_timings_json` 返回 `phase_preflight`、`phase_observe`、`phase_planning`、
  `phase_selecting`、`phase_open`、`phase_approach`、`phase_pregrasp`、`phase_descend`、`phase_close`、
  `phase_verify_close`、`phase_probe_lift`、`phase_verify_probe`、`phase_lift`、`phase_verify_lift` 和
  `phase_release` 等正式反馈阶段墙钟；`subphase_contact_realign` 是包含在 approach/pregrasp 阶段内的
  嵌套聚合耗时，`subphase_recovery` 是失败安全恢复的嵌套聚合耗时，二者不能与 phase 字段直接相加。

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
- 固定指间隙不足的候选只在批量候选准备阶段拒绝。到达 pregrasp 并完成在线接触补偿后，不再用第二次
  robust-gap 复核触发撤回或候选 fallback；该结果降级为诊断，执行器继续 `descend` 和 `close_gripper`。
  闭爪后的 TF/pose/robust-gap 检查同样是 best-effort，不能中断 commit-to-grasp。
- position-only IK 的 joint5 超出执行门限时，将 seed 翻转 `±π` 以交换固定指/活动指所在侧；最终 FK 接近轴、
  180° 对称闭合轴直线或固定指内侧检查失败时拒绝该候选，不再把 joint5 强制归零或停在 `±2.0` 边界。
- close 验证失败时闭爪撤回后再打开；probe/final lift 的 IK/FK、运动或 retention 验证失败时在当前位置
  打开并返回观察位。监督式客户端通过同一个 action 获得完全相同的恢复行为。
- 桌面平面或 SO101 mesh 不可用时，目标夹爪 tabletop filter fail closed。
- 同一时间只接受一个抓取 goal，并将上游取消请求传给当前 primitive。
- 隔离 worker pool 同时承担候选准备和执行期接触补偿、分支锁定所需的 IK/FK；单个 worker RPC 超时后会
  移除悬挂请求、隔离该 worker 并切换到其他已验证 worker。主 MoveIt 只负责真实规划和运动，避免不可取消的
  `/compute_ik` 请求阻塞后续恢复运动。
- `MoveToConfiguration` 当前是同步 ROS service；取消会停止本地等待，但服务端运动不具备 action 级硬取消。
  因此该 primitive 返回失败或超时后，`PickObject` 会 fail closed 并终止整个 goal，不会自动切换到下一候选。

## 测试

监督式调用同一正式 action：

```bash
source .shrc_local && source install/setup.bash && \
ros2 run manipulation_execution pick_action_client --prompt banana --mode execute
```

连续真机测试应使用同一个客户端进程，避免为每次抓取新建 DDS participant：

```bash
source .shrc_local && source install/setup.bash && \
ros2 run manipulation_execution pick_action_client \
  --prompt marker \
  --mode execute \
  --release-after-success \
  --release-drop-height-m 0.015 \
  --repeat 5
```

客户端为每个 goal 预先生成 UUID。如果 DDS 压力导致 `SendGoal` response 丢失，客户端会通过原 UUID
查询结果，不会重发抓取动作；`PICK_ACTION_GOAL_RESPONSE_RECOVERY` 表示进入了该安全恢复路径。

只运行正式规划和候选准备：

```bash
source .shrc_local && source install/setup.bash && \
ros2 run manipulation_execution pick_action_client --prompt banana --mode plan_only
```

```bash
source .shrc_local && source install/setup.bash && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --import-mode=importlib src/manipulation_execution/test
```

通用 worker 调度由 `test_pipeline_worker.py` 覆盖；SO101 joint5 运动学约束由
`test_so101_kinematics_guard.py` 覆盖。phases 测试继续验证配置解析、错误码和执行编排。
