# robot_interaction_demo

参数化交互示范包，提供常驻 `interaction_demo_node` 和客户端 `runtime-demo`。
机器人 YAML 只声明启用、超时和逻辑接口；`robot_config` 在 runtime 就绪后根据公开接口描述
解析实际 service/action endpoint，并以只读参数注入本包。

本包不依赖 `robot_config`、`robot_runtime`、AimDK runtime 或 `aimdk_msgs`，也不包含机器人配置、
固定文案或动作配方。它只依赖 `ibrobot_msgs` 的中立协议类型。

```bash
# 已加载环境、SDK overlay 和相同 DDS 域后：
ros2 launch robot_config robot.launch.py robot_config:=aimdk_x2_interaction_demo use_sim:=true
# 另一个终端按请求传业务值：
ros2 run robot_interaction_demo runtime-demo speak \
  --text "大家好，我是灵犀 X2。" --language zh-CN
ros2 run robot_interaction_demo runtime-demo motion \
  --name wave --target right --allow-motion
```

启动只建立 `/interaction_demo/execute`，不会自动发声或运动。节点一次处理一个执行请求，
忙碌时拒绝、不排队。TTS 返回接受结果，不代表播放完成；动作等待公共 action 终态。
CLI 支持 `--service` 和 `--timeout`，不加载 YAML。超时或 Ctrl-C 不取消已发请求。
后续 Agent 应继续使用共享 primitive/skill Gateway，本包仅是操作者参数化接口示范。

完整流程见 [使用指南](../../docs/aimdk_interaction_demo.md)。真机验证待操作者完成。
