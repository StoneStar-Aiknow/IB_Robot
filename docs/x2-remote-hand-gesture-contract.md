# X2 远端手势服务接口 v1

状态：供主控转交 Orin 话题实现和部署。
本接口复用 `inference_service/model_service_node`。
Orin 插件实现 `ModelServicePlugin`，复用已有 manifest 与 runtime。
禁止另写独立模型常驻框架或修改机器人控制接口。

## 接口与职责

- 服务类型：`ibrobot_msgs/srv/ClassifyHandGesture`。
- 服务名：`/perception/hand_gesture/classify`，允许显式配置。
- 请求：原生 Image 与有界 `metadata_json`。
- 响应：有界 `result_json` 与原有标准模型诊断字段。
- JSON 编解码复用 `embodied_common.hand_gesture_contract`。
- Orin 话题负责插件、模型配置、运行时适配与部署。
- 本话题负责公共契约和 `.5` 客户端，不部署 Orin 服务。

服务本身不发布 `/hri/hand/gesture` 或动作命令。
`.5` 在校验来源、人物、会话与时效后发布正式标签。
没有结果时，原镜像和过肩比心继续正常运行。
客户端禁止在超时后把旧请求伪装成已完成。

## infer

`RequestMeta.op` 为 `infer`。
图像固定为 `320 x 320 bgr8`，行跨度 960 字节。
数据长度固定为 307200 字节，禁止重新 JPEG 编码。
header 保留原相机时间，不把元数据塞入 frame_id。
ROI 保留补边前的原图坐标，允许负坐标。

`.5` 已按原规则裁剪、补 114 并线性缩放。
Orin 只做颜色、归一化、布局转换和模型推理。
模型输入为 RGB float32 `[1,3,320,320]`，除以 255。
输出使用已有切头模型及原后处理，不增加 NMS。
8/9 类合并为 heart，2 为 holy，7 为 xsign。
其余有效结果输出 none，分数为有限的 `[0,1]` 数值。

请求 JSON 必须原样回显在 `ResultMeta.request`。
标识、actor、原时间、ROI 和会话 token 禁止重写。
服务端在推理前后都检查本机处理预算。
迟到结果仍能结束请求，但禁止用于客户端投票。
源年龄最终由 `.5` 检查，两台机器的单调时钟不能相减。

## probe 与 retire

`probe` 使用空图，禁止执行模型推理。
响应报告实际就绪状态、进程启动 UUID 和 engine 摘要。
启动 UUID 在一次插件生命周期内不变，重启必须更换。
健康消息不能重复发布旧推理结果。

`retire` 使用空图，并携带 `retire_through` 请求编号。
服务端先拒绝该来源截至这个编号的旧 infer 请求。
服务端确认旧任务不再执行或排队后，才返回 idle。
响应须原样回显本次 retire 请求，并确认退休编号。
无法确认空闲时返回 busy，禁止声称已取消 CUDA 工作。

模型执行和退役确认必须串行。
使用通用模型服务原有的互斥调用组即可。
客户端最多 1 个模型请求和 1 个退役控制请求。
退役控制只用于收尾，不投递第二张新图。
单纯探测到服务在线不能释放旧占用槽。

## 配置与标准诊断

插件通过原有 `bundle_path/deployment/adapter_class` 配置。
标准响应必须保留 `model/inference_time_ms/success/message`。
`model` 来自通用宿主的实际 manifest 和 runtime 状态。
`result_json` 须带本次 server_boot_id 与真实 engine_sha256。
暂用已验证 FP32 engine，不在接入时重新优化或构建模型。

请求与结果 JSON 均不超过 4096 个 UTF-8 字节。
重复键、非有限值、未知字段与错误协议必须拒收。
请求关联信息禁止放到 message 字段中传输。
失败返回标准 success=false，禁止伪造有效 none。
默认不保存逐帧数据，测试排查才临时保存并删除。

## 交付

Orin 话题先交模型服务插件和 manifest 的本地提交。
交付包括 boot 身份、模型摘要、退役语义与本地测试证据。
Orin 部署由主控协调，正式动作联动由主控审核后安排。
本接口不授权启动机器人，也不改变 CBF 或模板播放。
