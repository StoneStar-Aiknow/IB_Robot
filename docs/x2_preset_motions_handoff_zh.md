# X2 头顶比心：阶段交付

更新日期：2026-11-10。
状态：软件衔接已实现，完整动作认证待完成。
警告：本轮交付禁止启用机器人比心。

## 分支与基点

本地分支：`feat/x2-preset-motions`。
主线基点：`f446fbceba9b4b33d57cd29318239f67792c3fef`。
合入提交：`8c65f50b45494c3035fa7f7c049e015abce3c659`。
已正式验证的代码：`9793a606080c179602533822ee310122904913f1`。
停止衔接的最终 SHA 见部署报告和本地进度文档。
开发会话不推送、不改写历史、不纳入 AimDK SDK。

用户已批准最多 8 个特性提交，不含 merge。
用户已批准两项 ROS 动作接口依赖声明。
提交身份为 `grangerxsp <xingshiping@huawei.com>`。
人工代码审查与完整动作安全验收仍待完成。

## 配置隔离

`preset_motions.enabled` 控制识别节点。
`preset_motions.motion_enabled` 控制动作衔接。
`aimdk_x2_voice_mirror_all` 只启用识别。
抓取配置显式关闭识别与动作衔接。
`aimdk_x2_summit` 继承抓取配置的关闭值。
本特性没有替换现有技能目录绑定。

动作启动还要求已验证模板与几何产物。
动作启动要求 `stop_bridge_certified: true`。
当前资产不满足这些条件。
模板仍为 `validated: false` 和 `motion_enabled: false`。
运行时头部间距仍为 0.20 m。

同一开关通过 `named_hold_json` 传入 AimDK 运行时。
运行时独立验证模板、固定录制和几何证据。
注册只修改内存中的私有 profile。
原有 profile 文件和已有动作保持不变。
固定录制仍为 `validated: false`。
固定录制只有冻结的视觉造型数据。

STANDBY/MIRRORING 下 ASR 停止。
识别和停止均不依赖 ASR。

## 识别与控制权

匹配器只订阅现有 PEAR、选人和会话输出。
匹配器通过人体正向运动学（FK）计算躯干坐标系。
规则使用肩宽归一化，不修改重定向。
首版只注册 `overhead_heart`。
进入需要持续 0.5 s 且至少 3 帧。
退出使用 0.4 s 迟滞。

模块拒绝过期、乱序、错误演员和旧轮次数据。

普通模板进入、保持和退出保持 MIRRORING/running。
交接先暂停双臂与双手的公共镜像流。
具名动作通过 UUID 租约取得控制权。
模板目标包含双臂 14 维和双手 20 维。
模板不发送腰部和腿部命令。
普通手势覆盖不能改写模板手型。

普通退出验证同一 UUID 的目标哈希与释放结果。
会话从这个完整目标恢复镜像。
`mirror_stop`、长捂和摸头优先于模板保持。
正常停止请求同一所有者执行受检的回中路径。
停止成功证据与恢复镜像证据分别保存。
故障、失联、错误证据和超预算保持 HOLD。

旧匹配结果不能启动新轮次。

## 匹配事件契约

话题：`/preset_motions/matched`。
类型：`std_msgs/msg/String`，内容为 JSON。
每次确认激活只发布 1 个事件。

```json
{
  "schema_version": 1,
  "event_id": "<activation identifier>",
  "template_name": "overhead_heart",
  "score": 0.9,
  "actor_id": 0,
  "source_stamp": {"sec": 0, "nanosec": 0},
  "matched_stamp": {"sec": 0, "nanosec": 0}
}
```

`source_stamp` 是源图像时间。
`matched_stamp` 是确认匹配时间。
`score` 是规则分数，不是标定概率。
事件表示人体模板匹配，不表示机器人动作完成。
拍照特性能够按 `event_id` 去重。
持续匹配状态发布到 `/preset_motions/state`。

跨会话契约确认仍需总控转达。

## 已确认的视觉版本

视觉版本为 `palms_down_v1`。
双手掌面朝下，指尖向头顶中央收拢。
腕部屈曲约 55.95°，距硬限位至少 0.12 rad。
模板保存冻结的 34 维角度。
模板：`src/robot_config/config/preset_motions/overhead_heart.yaml`。
冻结图片与哈希保存在本机设计目录。

模板只新增 `recording_file` 元数据。
原有角度、模型、标志和间距要求完全相同。
旧审计保留原模板的整文件哈希。
元数据等价记录保存在审计目录。
文件名为 `recording_metadata_equivalence.json`。

用户批准的是视觉造型。
15 cm 仅为当时仿真比较的设计条件。
运行时门禁没有改为 15 cm。
旧球覆盖下界 15.1767 cm 不是网格碰撞证明。

连续源网格审计给出头部间距下界 20.781013 cm。
计算覆盖 yaw ±0.349 rad、pitch ±0.35 rad。
计算已扣除 1 μm 数值余量。
runtime 的 yaw ±0.38 rad 超出 URDF 的 yaw 范围。
额外 yaw 区间只有运动学外推结果。
本特性没有改变头部限位或增加头部锁定。

审计目录：`<本机设计目录>/simulation/approved_pose_clearance_audit/`（不在仓库内）。
源网格证据尚未转为运行时认证产物。
传感器外壳、跟踪误差和完整自碰仍待认证。
进入、退出和停止回中路径仍待认证。
因此本轮不能报告完整动作可用。

## 正式板端验证：9793a606

部署目录：`/root/x2-feat-preset-motions-pkg-9793a606`。
部署构建 67 个包，结果为 `DEPLOY: OK`。
校验结果为 FAIL 0、WARN 1。
警告为既有 `robot-clock inactive`。

正式入口为 `colcon test`，域为 84。
测试设置 `ROS_LOCALHOST_ONLY=1`。
测试使用共享锁 `/root/x2-voice-run/x2-voice.lock`。
测试准入确认没有运行中的 `ros2 launch`。

- `manipulation_execution`：1434 通过、1 失败。
- `robot_config`：2119 通过、22 失败、2 错误、31 跳过。
- `aimdk_robot` 定向：40 通过。
- `skill_library` 定向：43 通过。

所有测试进程正常退出，没有退出码 124。
ME 的唯一失败与 f446 基线相同。
失败名为 `test_magicmirror_plan_reaches_mirror_session_through_the_entry_point`。
RC 使用 `--continue-on-collection-errors`。
RC 对照基线为 0fb1 的 43 个失败名。
两边失败名单均已确认非空。

RC 当前减基线为 2 个失败名：

- `test/test_internvl_voice_entry.py::TestNothingElseChanges::test_the_entry_without_the_block_is_the_baseline_file`
- `test/test_internvl_voice_entry.py::TestTheBlock::test_it_is_written_at_the_contract_s_defaults`

RC 基线减当前为 21 个名字。
完整名字和原始 XML 见结果报告。
RC 差异交给总控判断。
这些数字不覆盖本文同时提交的停止衔接修改。
最终 SHA 的复验结果单独记录。

本机报告：`<本机设计目录>/board_validation/9793a606/results/RESULTS.txt`（不在仓库内）。
板端报告：`/root/x2-feat-preset-motions-run/test/acceptance-9793a606-20261110-003912/`。

## 后续验收

1. 复验最终提交的停止衔接。
2. 生成运行时可验证的几何产物。
3. 认证完整碰撞与全部过渡路径。
4. 采集实际 PEAR 正负样本。
5. 获取真机测试许可。

正式识别率和误触发率仍待 PEAR 数据验证。
纯身体规则不能区分全部摸头与比心姿势。
本特性没有启动真机，没有停止其他会话。
