# Orin 手势模型服务候选

本候选只完成本机开发和隔离测试。
板端部署必须先经过主控审核。
本候选复用 `inference_service/model_service_node`。
本候选没有增加常驻模型宿主。

## 版本与接口

- 分支：`feat/x2-orin-gesture-service`。
- 基点：`f77f4a3b29eca56c41c7f76a4c2dd5d10d749173`。
- 客户端候选：`adaf75adddb145d1c92545b247f3d07074004e70`。
- 服务名：`/perception/hand_gesture/classify`。
- 服务类型：`ibrobot_msgs/srv/ClassifyHandGesture`。
- 插件：`perception_service.hand_gesture_plugin:HandGestureClassifyPlugin`。
- 配置：`config/hand_gesture_model_service.yaml`。
- 打包工具：`package_hand_gesture_bundle`。

插件复用 `embodied_common.hand_gesture_contract` 编解码。
插件原样回显 `RequestMeta`。
插件保留通用宿主的标准模型诊断。
插件只返回检测结果。
插件不发布正式手势或机器人动作。

## 模型与生命周期

模型使用已验证的 FP32 TensorRT engine。
服务启动时只加载一次 engine。
服务启动时禁止构建或优化 engine。
模型工件摘要为：

```text
421c325669b665e3331430f98365fe3a126bce92a4588097e9faeaa89fd7d639
```

打包工具从已有工件生成 v3 manifest。
可审查模板位于 `config/hand_gesture_bundle`。
模板声明已验证的摘要，不包含模型二进制。

打包工具拒绝摘要不符的工件。
打包工具拒绝覆盖已有 bundle。
会话从实际加载的字节计算摘要。
响应中的摘要来自会话，不能用配置伪装。

`TensorRtHandGestureSession` 复用 `ModelSession`。
`ModelRuntimeHandle` 管理会话生命周期。
此处沿用 `EchoServicePlugin` 的显式组装方式。
通用 TensorRT 后端注册不在本次范围。

输入固定为 `320 x 320 bgr8`。
行跨度固定为 `960` 字节。
数据长度固定为 `307200` 字节。
服务端只转换颜色、布局和数值范围。
服务端禁止再次裁剪、缩放或 JPEG 编码。

后处理沿用切头模型的稳定前 300 项选择。
后处理不增加非极大值抑制（NMS）。

类别 `8/9` 合并为 `heart`。
类别 `2` 返回 `holy`。
类别 `7` 返回 `xsign`。
其他有效结果返回 `none`。
失败响应使用 `success=false`。
失败响应不能作为有效的 `none` 结果。

## 安全保护

模型执行最多 `1` 个请求。
服务接收队列深度为 `1`。
同一用户的同一设备只允许一个手势会话。
重复请求和退役请求不能再次执行模型。

来源台账最多保存 `64` 个进程身份。
同一进程的递增代次只保留一个记录。
代次推进后，旧代次请求继续拒收。
台账满时拒绝新增来源，保留退役证据。

推理速率默认不超过 `10 Hz`。
配置只允许下调这个上限。
输入和输出使用固定大小的预分配缓冲区。
模型工件大小不超过 `64 MiB`。
工程日志默认不保存图像或逐帧结果。

可用内存小于 `2 GiB` 时，服务暂停推理准入。
监测温度达到 `80°C` 时，服务暂停推理准入。
监测失败时，服务暂停推理准入。
资源条件恢复后，服务自动恢复准入。
资源检查只读取系统状态。
服务禁止修改调频、功耗或散热策略。

内存和温度检查不是硬件资源隔离。
主控还需安排 CPU、内存和 GPU 共存验收。
正式部署还需限制本服务的进程资源。
资源限额数值须经板端验收确认。
本候选没有修改厂商相机、SLAM 或系统服务。

## 探测、预算与退役

`server_boot_id` 在插件生命周期内固定。
新插件生成新的 `server_boot_id`。
`probe` 只返回状态，禁止执行图像推理。
探测状态不产生推理票。

处理预算从本机回调开始计时。
插件使用本机单调时钟检查预算。
插件在推理前后都检查预算。
插件禁止相减两端的单调时钟。
客户端仍需按原图时间检查源帧年龄。

`retire` 先记录旧请求的拒收水位线。
模型执行和退役确认保持串行。
CUDA 成功同步后，服务才确认模型空闲。
客户端取消等待不能充当 CUDA 完成证据。
同步失败时，服务返回非就绪状态。
完成状态不明时，服务禁止确认退役空闲。

CUDA 故障不能触发全局设备重置。

通用宿主增加两个可选接口。
`request_queue_depth` 默认保持 `10`。
只有本插件将队列深度设为 `1`。
`accepts_unready_request` 默认返回 false。
只有本插件放行非就绪时的探测与退役。

这样模型故障后仍能查询身份和退役状态。
其他插件继续使用原来的就绪门禁。

## 验收范围

本机测试使用假 engine 和假 CUDA 调用。
真实 ROS 服务测试只使用本机回环通信。
测试在独立 ROS 域中执行。
测试禁止访问真实 GPU。
构建和测试限制为 `2` 个 CPU 核。

测试覆盖请求绑定、颜色转换和类别映射。
测试覆盖坏请求、坏输出和处理预算。
测试覆盖并发、台账上限和重复请求。
测试覆盖启动身份、工件摘要和资源准入。
测试覆盖 CUDA 同步失败与部分分配回收。
测试还覆盖客户端取消后的实际退役顺序。

完整结果见独立交付报告和 JUnit 文件。
全包失败项必须与固定基点对照。
假 engine 测试不能证明板端模型性能。
旧旁路的性能不能替代新服务的板端验收。

## 部署命令草稿

警告：本阶段禁止在机器人上执行本节命令。
主控审核候选后，才安排板端部署。
部署只允许写入本任务的独立目录。
部署禁止升级板端依赖或启用自启动。

以下变量必须指向主控审核后的工作树。
打包工具只复用已验证的 engine。

```bash
TREE=/home/agi/x2-orin-gesture-service-REVIEWED_SHA
cd "$TREE"
source .shrc_local
ros2 run perception_service package_hand_gesture_bundle \
  --engine /home/agi/x2-yolo-gesture-orin-20261119/artifacts/yolov10n_gestures_320_fp32.engine \
  --output /home/agi/x2-gesture-service-models/yolov10n_320_fp32
```

第一次服务验证只连接隔离域。
下面的服务进程最多运行 `120 s`。
这个命令不启动机器人动作。

```bash
cd "$TREE"
source .shrc_local
export ROS_DOMAIN_ID=97 ROS_LOCALHOST_ONLY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
timeout --signal=INT --kill-after=5s 120 \
  nice -n 10 ros2 run inference_service model_service_node \
  --ros-args -r __node:=hand_gesture_model_service \
  --params-file "$TREE/install/share/perception_service/config/hand_gesture_model_service.yaml"
```

正式域的两端联调另行安排。
正式域联调先只记录结果。
主控确认原业务正常后，再安排动作联动。
旧独立旁路脚本不能与新服务同时推理。

## 回退

候选未部署时，机器人运行状态保持原样。
板端试验后，只停止本服务的独立进程。
客户端保持无结果降级，原镜像独立运行。
禁止用通配进程终止命令停止厂商业务。
CBF 修复 A 继续独立验收。
