# X2 Agent 抓取苹果

本部署使用 `ibrobot_agent` 节点和现有 `pick_object`。
左手抓起苹果后，执行器仍会抬升、保持、放回并回到初始位置。
本次只验收抓取苹果，不增加容器放置或持续握持功能。
2026-11-03，第二次真机请求完成，Agent 和 Gateway 均返回 `SUCCEEDED`。

## 调用链

```text
“抓取苹果”
  -> /agent/request
  -> ibrobot_agent_node
  -> Trae 转发的大模型
  -> pick_object(target_name="an apple")
  -> /embodied/execute_program
  -> Capability Gateway
  -> /manipulation/execute_pick
  -> 原有 X2 抓取执行器
```

模型只生成技能参数。Gateway 仍负责运动授权和执行检查。
Agent 的技能白名单只包含 `pick_object`。
技能目录使用已有的 X2 专用描述。

## 部署位置

| 项目 | 位置 |
| --- | --- |
| 310P worktree | `/root/zf/IB_Robot_agent_grasp` |
| 代码基线 | `dca1def0`，分支 `feat/x2-agent-grasp` |
| 机器人配置 | `aimdk_x2_grasp_310p_agent` |
| 模型 | 复用 `/root/zf/IB_Robot/models` |
| Python 依赖 | 复用主仓库 venv 和系统包 |
| AimDK overlay | `/root/zf/IB_Robot/aimdk/install` |
| 环境入口 | `/root/zf/agent_grasp_env.sh` |
| Agent 记录 | `/root/zf/agent_grasp_state/` |
| Trae 接口 | `http://10.42.0.250:18081/v1` |
| 大模型 | `DeepSeek-V4.1-Flash-1017` |

worktree 使用独立的 `build`、`install` 和 `log`。
worktree 的 venv 引用主仓库已安装的依赖。
源码和构建产物从 worktree 加载。
原有 `/root/zf/run_pipeline.sh` 保持不变。

## 启动和输入

前置条件见[抓取部署说明](aimdk_x2_grasp_skill.md#启动)。
必须先完成 HOME 准备，再启动 Orin 抓取头并等待预热完成。
预热后移动右臂、头部或张开手指，会触发规划器重建。
规划器重建可能耗尽下一次请求的规划时限。

Orin 抓取头继续使用原有 `aimdk_x2_grasp_310p` 配置。
新配置只改变 310P 的 Agent 入口和记录路径。
启动前必须停止原有 310P 管线，避免重复运行。

警告：下面的管线命令授权运动。仅在现场操作员确认安全后执行。

1. 在 310P 的干净终端加载环境。

   ```bash
   env -i HOME=/root USER=root PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash --noprofile --norc
   source /root/zf/agent_grasp_env.sh
   ```

2. 启动 Agent 管线。

   ```bash
   ros2 launch embodied_bringup embodied_pipeline.launch.py \
     robot_config:=aimdk_x2_grasp_310p_agent \
     control_mode:=moveit_planning use_sim:=false authorize_motion:=true
   ```

3. 在另一个 310P 终端加载同一环境。

4. 启动文本界面。

   ```bash
   ros2 run ibrobot_agent ibrobot_agent_chat \
     --channel-id x2_grasp_cli --principal-id local_operator \
     --robot-scope aimdk_x2_grasp
   ```

5. 输入 `抓取苹果`。

Agent 会先调用模型，再提交包含 1 个技能的 Program。
规划成功不代表抓取成功。必须检查 Gateway 的执行结果。
输入 `/stop` 可请求停止；发出停止请求不代表机器人已停止。

管线已经运行时，工作站也能通过 SSH 启动文本界面。
下面的命令只连接现有 Agent，不重复启动管线。

```bash
ssh -tt root@10.42.0.251 \
"env -i HOME=/root USER=root TERM=xterm-256color PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash --noprofile --norc -c 'source /root/zf/agent_grasp_env.sh >/dev/null && exec ros2 run ibrobot_agent ibrobot_agent_chat --channel-id x2_grasp_cli --principal-id local_operator --robot-scope aimdk_x2_grasp'"
```

终端显示 `小智>` 后，输入 `抓取苹果` 并回车。
这个输入会触发真机动作。
当前客户端没有 `--text` 参数。
`/stop` 只请求停止当前终端发出的任务。
`/quit` 退出文本界面。

## 中文目标约定

`target_name` 使用简短的英文物体短语。
Agent 负责从指令提取物体，并保留颜色限定词。
检测入口也支持固定的中文名称转换：

| 中文目标 | 检测短语 |
| --- | --- |
| 苹果 | `an apple` |
| 红苹果、红色苹果 | `a red apple` |
| 青苹果、绿色苹果 | `a green apple` |
| 瓶子、水瓶 | `a bottle` |
| 杯子、水杯 | `a cup` |

检测入口会为英文短语补上 GroundingDINO 所需的句点。
检测入口拒绝空目标和未支持的中文短语。
例如，直接传入 `左边的苹果` 会返回 `INVALID_TARGET`。
执行器不会删除“左边”后继续抓取。
整句 `抓取苹果` 属于 Agent 输入，不属于 `target_name`。

## 软件验证记录

2026-11-03，310P 完成 worktree 构建。
`colcon test` 的定向结果为 97 项，0 错误，0 失败，3 跳过。
跳过项依赖本机未安装的 CUDA 模型。
测试覆盖目录编译、X2 配置、目标转换和 Agent 节点。
本次没有运行完整的导航仿真或 Orin CUDA 测试。

无运动验证使用真实 Agent 节点和 Trae 模型。
输入 `抓取苹果` 后，Agent 生成 `pick_object(target_name="an apple")`。
记录中的 `may_have_submitted` 为 `false`。
`DRY_RUN_ONLY` 只证明规划链路通过，不证明真机抓取成功。

| 记录 | 310P 路径 |
| --- | --- |
| 构建日志 | `/root/zf/x2_agent_grasp_build_cached.log` |
| 定向测试 | `/root/zf/x2_agent_grasp_tests_targeted.log` |
| Agent 规划记录 | `/root/zf/x2_agent_grasp_dryrun.json` |

## 真机启动记录

2026-11-03，真机测试启动了两侧服务。
310P 使用临时 systemd 单元 `x2-agent-grasp`。
启动脚本为 `/root/zf/run_agent_grasp.sh`。
脚本的运动授权参数默认为 `false`。
本次经操作员授权，以 `true` 启动。

运行日志为 `/root/zf/x2_agent_grasp_live_20261103.log`。
手动启动另一套管线前，必须检查这个单元是否仍在运行。

Orin 继续使用 `~/ibrobot_x2grasp/zf_run_head.sh`。
本次日志为 `~/ibrobot_x2grasp/head_agent_grasp_20261103.log`。
两侧模型预热和 Gateway 参数校验已通过。
这些检查不代表真机抓取已完成。

## HOME 准备

现有 Agent 技能白名单只包含 `pick_object`。
初始 HOME 准备仍使用部署目录中的厂商工具。
这个准备动作会移动双臂、头部和双手。
执行前必须确认完整的动作范围。

准备前先查询厂商模式。
如果模式已为 `UPPERBODY_REMOTE_SPLIT` 且状态为 `100`，跳过模式切换。
否则，按部署说明先切换到稳定站立模式 `STAND_DEFAULT`。
控制器接受后，再切换到上肢遥控模式 `UPPERBODY_REMOTE_SPLIT`。
确认模式就绪后，`/root/zf/upper_body_prep.py` 发送 5 s 的准备命令。

双臂目标均为 `[0.4, 0, 0, -1.2, 0, 0, 0]` rad。
头部目标为 `[0, 0]` rad。
左手使用抓取模型的张开预设，右手关节目标均为 0 rad。

准备脚本退出不代表双臂已到达 HOME。
抓取前必须读取关节反馈，检查左臂各关节误差不超过 0.1 rad。

2026-11-03，操作员授权了 HOME 准备动作。
本次直接切换上肢遥控模式时，厂商控制器返回 `code=9`。
随后请求稳定站立模式，控制器同样返回 `code=9`。
返回信息为“当前动作正在播放，无法切换”。
代码将这个错误定义为 `MOTION_BUSY`。

只读查询随后确认，模式已为 `UPPERBODY_REMOTE_SPLIT`，状态为 `100`。
仓库记录了这个固件在重复切换 URS 时返回 `MOTION_BUSY` 的行为。
本次跳过重复切换，执行了 HOME 准备脚本。
脚本在 5 s 内发送了 250 条命令。
关节反馈确认左臂最大 HOME 误差为 0.0404 rad，满足 0.1 rad 的要求。

| 记录 | 310P 路径 |
| --- | --- |
| 上肢遥控模式请求 | `/root/zf/x2_agent_grasp_prepare_20261103.log` |
| 稳定站立模式请求 | `/root/zf/x2_agent_grasp_prepare_sd_20261103.log` |
| HOME 前模式查询 | `/root/zf/x2_agent_grasp_mc_before_home_20261103.log` |
| HOME 准备命令 | `/root/zf/x2_agent_grasp_home_20261103.log` |

## 第一次真机请求

2026-11-03，Agent 收到了 `抓取苹果`。
Trae 模型返回 `pick_object(target_name="an apple")`。
Gateway 接受了请求，并调用了抓取执行器。

| 字段 | 值 |
| --- | --- |
| 请求 ID | `cli-49813f5846a2` |
| 执行 ID | `a6161423-45be-4b70-9ad7-81c4796e277c` |
| 最终状态 | `FAILED` |
| 错误码 | `NO_EXECUTABLE_CANDIDATE` |
| 错误信息 | `TimeoutError: Grasp request deadline exceeded` |
| 已完成技能数 | `0` |
| 执行器耗时 | `34.19 s` |
| 执行记录 | `/root/zf/x2_agent_grasp_apple_first_20261103.json` |

本次在规划阶段终止，没有进入抓取执行阶段。
Orin 日志显示，HOME 改变了右臂和手部位置，触发了规划器重建。
重建耗时计入当前请求的规划时限。
本次验证了 Agent 到抓取执行器的调用，未证明苹果抓取成功。

## HOME 后的无运动验证

本次使用原有 `/root/zf/hunt/plan_only.py` 验证当前姿态。
这个工具将请求模式固定为 `MODE_PLAN_ONLY`。
管线完成观察、感知和规划后返回，不执行抓取轨迹。

2026-11-03，验证返回 `success=True` 和 `full_plan_valid=true`。
检测标签为 `an apple`，检测置信度为 `0.8673`。
完整计划就绪耗时为 4.94 s，小于 10 s 的规划时限。
左手张开误差为 0.01054 rad，小于 0.025 rad 的容差。
本次保持原有时限和碰撞检查。

任务 ID 为 `x2-agent-apple-plan-only-20261103-001`。
日志为 `/root/zf/x2_agent_grasp_plan_only_20261103.log`。
这个结果证明当前场景能生成有效计划，不代表苹果已抓起。

## 第二次真机请求

2026-11-03，操作员发起了新一次 `抓取苹果` 请求。
本次继续使用 `ibrobot_agent` 和 Trae 模型。
模型生成 `pick_object(target_name="an apple")`。
Agent 和 Gateway 均记录了成功结果。

| 字段 | 值 |
| --- | --- |
| 请求 ID | `cli-28ce050481b5` |
| 执行 ID | `09135e3e-07e4-43c5-a131-6f99ae3db112` |
| 最终状态 | `SUCCEEDED` |
| 已完成技能数 | `1/1` |
| 错误码 | 空 |
| 执行器耗时 | `31.39 s` |
| 完成时间 | `2026-11-03 12:42:11`，北京时间 |

执行器保留了抓取、抬升、保持、自动放回和退回流程。
执行结束后，Gateway 返回 `busy=false`，活动任务 ID 为空。
关节反馈确认左臂最大 HOME 误差为 0.0368 rad，满足 0.1 rad 的要求。

| 记录 | 310P 路径 |
| --- | --- |
| Agent 和 Gateway 最终记录 | `/root/zf/x2_agent_grasp_apple_success_20261103.json` |
| 完成后 Gateway 状态 | `/root/zf/x2_agent_grasp_after_success_status_20261103.json` |
| 完成后关节反馈 | `/root/zf/x2_agent_grasp_after_success_joints_20261103.json` |

本次测试遇到了 310P 管理网络间歇中断。
抓取请求在连接恢复并通过状态检查后发出。
测试结束时，310P Agent 管线和 Orin 抓取头保持运行。

## 重复抓取验证

2026-11-03，操作员要求再抓取一次苹果。
Agent 和 Gateway 均返回 `SUCCEEDED`，已完成技能数为 `1/1`。
执行器耗时为 29.53 s，完成时间为北京时间 `12:46:34`。
请求 ID 为 `cli-179cee6fc215`。
执行 ID 为 `d501890e-9b67-4bff-ad65-0a07f857b54d`。

执行后，Gateway 空闲，活动任务 ID 为空。
左臂最大 HOME 误差为 0.0320 rad，满足 0.1 rad 的要求。
执行记录为 `/root/zf/x2_agent_grasp_apple_third_20261103.json`。
本次仍保留自动放回和退回流程。
管理网络中断延迟了结果读取，没有重复提交抓取请求。
