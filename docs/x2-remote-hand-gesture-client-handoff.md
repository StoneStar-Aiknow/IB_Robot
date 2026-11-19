# X2 远端手势客户端交付

本次实现只负责 `.5` 的可选手势接入。
原有照镜子、过肩比心、选人和模板动作保持原逻辑。
有有效结果时使用，没有结果时继续原样运行。
选人和换人不在本次范围，客户端没有新增 ActorState 订阅。

## 版本与分工

- 开发基点：D1 `51bd0e97`。
- 开发分支：`feat/x2-motion-template-remote-client`。
- 公共接口提交：`f77f4a3b`，未推送。
- 最终客户端 SHA 以提交后的交付消息为准。
- `.41` 服务实现与部署由主控转交 Orin 话题。
- 本话题没有部署模型、连接板端或启动机器人。

公共接口见 [服务契约](x2-remote-hand-gesture-contract.md)。
服务复用原 `ModelServiceNode`、manifest 与模型生命周期。
本次不增加第二套模型常驻框架或控制入口。

## 实现边界

手部线程先发布原 HandFrame。
随后只尝试交接图像引用与小型元数据。
裁剪、缩放、服务发送与结果处理都在后台线程。
待处理图像只保留最新 1 帧。
模型请求最多 1 个，超时后仍保留占用直至收尾确认。

请求原子绑定原图时间、ROI 和原有 actor 字段。
actor 只原样传递，原匹配器已有校验继续负责判断。
迟到、重复、错配、错误来源与过期结果禁止计票。
没有输入或推理失败时，不伪造 none 结果。
健康探测也不产生推理票。

来源失效时只清理远端三类投票。
重复健康状态不反复清票，健康心跳不延长正例。
恢复后重新确认，允许进入新一轮模板。
停止不等待远端推理、网络或后台线程退出。
默认不新增图像或逐帧结果保存。

## 配置与启用

出厂 YAML 保持禁用，旧 enabled/path 配置继续按原规则解释。
本地与远端手势源必须唯一选择。
以下仅说明主控审核后的现场配置，不是启动命令。

- `motion_imitation_stream.mock_playback.hands.perception.parameters.gesture_mode` 设为 `remote`。
- `gesture_remote_service` 使用已部署的 `/perception/hand_gesture/classify`。
- `preset_motions.gesture.enabled` 设为 true。
- `preset_motions.gesture.source` 设为 `remote`。
- `source_topic` 与 `gesture_remote_source_topic` 保持一致。

默认服务超时为 0.4 s，源帧年龄最多为 0.8 s。
已有窗口、票数、分数、标签退出与动作时长保持原值。
空 engine 摘要配置会固定首次健康握手的 engine 摘要。
现场也支持显式填写已验证的 engine SHA-256。
远端不可用时禁止自动回退到本地 OM。

新增远端类型缺失或配置错误只禁用手势接入。
本次没有修改 D2、CBF、白名单、拍照或语音逻辑。
没有修改 `.5` 网络、全局依赖或其他进程。

## 验证与未验证项

验证使用本机隔离 ROS 域和回环通信。
模型与图像均使用假服务或合成输入。
最终结果见随候选版本交付的 colcon 报告与 JUnit。
测试同时保留原分支基点的回归对照。

- 基点 manipulation_execution：2148 过，6 跳过。
- 候选 manipulation_execution：2192 过，6 跳过。
- 基点 embodied_common：523 过。
- 候选 embodied_common：548 过。

新增 69 个用例均通过。
线程未处理异常按失败处理，候选未出现这类异常。
原有 1 条数值溢出警告在基点和候选中都存在。

robot_config 全量检查没有通过。
51 个失败或错误中，49 个已在基点复现。
其余 2 个导航仿真启动超时未在基点重复执行。
这部分保留完整报告，禁止把整仓结果写成全绿。

本机测试不代替 Orin 真实模型接口联调。
`.5` 实际资源争用、真实手势效果和真机动作仍待验证。
主控先审核版本，随后协调 Orin 和真机测试。
推送仍由用户执行。
