## IB-Robot 控制与语音策略

- 对真实机器人请求，必须使用 `ibrobot-control` Skill 和 Capability Gateway，不得调用裸 ROS 运动类接口
  （`ros2 topic pub`、`ros2 service call`、`ros2 action`、MoveIt、controller、primitive 或硬件接口）。
- 生成并验证冻结计划后，先展示并 flush exact ordered steps、参数、plan digest、registry identity 和 task ID，
  随后立即调用内部 `confirm-plan` 绑定 exact tuple 并执行；不得询问“确认执行吗”，不得等待用户再次回复。
- `confirm-plan` 是 Gateway 技术绑定，不是用户确认门禁。物理运动仍只能由操作员启动 pipeline 时设置的
  `authorize_motion` 授权；Hermes 不得启动或重启 pipeline、修改 ROS 参数或开启运动授权。
- Gateway 不可用、未授权、计划校验失败、执行超时或状态未知时必须停止，不得自动重试或发起新运动。
- 用户发送“别动”“停止”或“取消”时立即通过受控取消入口处理；取消请求不等于机器人已停止，必须等待
  stop-confirmed terminal result。
- TTS 是系统自动功能，不是机器人 Skill。最终自然语言回复由 `post_llm_call` hook 合成并播放，不得在
  workflow 中加入“播报”“语音”或“发声”步骤。
