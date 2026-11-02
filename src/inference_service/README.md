# inference_service

`cloud_inference.launch.py` 在 inference/recording 两个分支的节点执行前检查本机受管 DDS
环境。错误按原因、实际状态与建议报告，环境不会被自动改写。这不是跨机通信认证，其他
独立推理入口未自动获得此检查。详见 [DDS 配置工具](../../scripts/dds/README.md)。

`inference_service` 是 IB-Robot 的统一推理运行时。它以一个模型 bundle 和一个命名
deployment 为输入，统一承载 Torch、Ascend、Hisilicon、RKNN 与 HMM 后端。策略模型额外使用稳定的
pipeline ID，并支持单体和边云分布式执行。

运行时从 bundle 中唯一的 `inference_manifest.json` 读取模型类型、语义 tensor、命名 deployment、artifact、
执行顺序和 runtime ABI bindings。Launch 和 robot YAML 通过 deployment 名称选择运行配置。

非 policy bundle 不需要 LeRobot metadata 或 `action` 输出。公共本地执行通过
`ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)` 进入，成功结果统一为 `ModelResult`；
独立 producer 帧通过 `ModelRuntimeHandle.submit_frame(ModelRequest, ExecutionContext)` 返回 future，
与 Dispatch 共用请求登记、取消、失败处理和生命周期排空。帧有独立的有界准入额度，允许 producer 与
Dispatch 重叠执行，统一 diagnostics 的活动计数包含两者。
`ModelRuntimeFactory` 是注册式构造入口，不是请求边界。`ModelRuntimeHandle` 接受 `RuntimeAssembly` 的
所有权转移，负责公开生命周期、准入、deadline、取消、健康状态和关闭等待。`ModelSession` 是 assembly
中的资源，只负责加载、执行和释放 vendor 模型对象、设备 lease、buffer 或 worker。
模型家族的预处理、后处理和 ROS service 形状不属于 backend runtime，由调用方 adapter/plugin 持有。

`IB_TRACE_ENABLED=1` 仅增加 span/flow 观测，不改变 stage 遍历、输入校验位置、取消/超时检查、
异常证据或健康状态。`model_call` 仅在 policy 原有 model request/result stage 边界记录一个聚合
span，不对每个 role/迭代单独埋点。未到达 result stage 的 span 标为 `incomplete`，提前拒绝允许
缺失 span；trace token 仅保存在请求上下文中，不写入业务 frame 或 metadata。

Tracing 不增加设备同步或 tensor/device 遍历。span 是主机调用耗时，不保证 GPU/NPU kernel
已经完成。`backend_latency_ms` 保留原有 request/result stage 间的业务计算，不从 trace 数据派生。

分布式 policy 的实际 LeRobot processors 位于云端，预处理/模型/后处理埋点跟随其真实位置；
edge 的 identity 适配不计作模型预处理。正常与失败结果都在返回边界发送结果 Flow。

manifest fingerprint 是经过验证的 bundle 结构身份，deployment fingerprint 标识所选运行部署。常规 loader
不会为了生成诊断身份扫描大型权重文件；启用推理 scheduler 时，本地 compiled artifact 还必须声明内容
`sha256`，`robot_config` 会在构造 scheduled launch 前流式校验文件内容。

## 核心概念

### 后端完成与释放证据

`runtime_context_defaults` 为当前构造上下文提供 `priority_scheduling` 和 `managed_execution`
默认值。显式参数优先。退出上下文后恢复原值，其他线程不继承这些默认值。

Pipeline 和 Manager 工厂省略优先级参数时，继承构造作用域的默认值。
显式 `False` 仍覆盖作用域默认值。

独立构造的 `AscendOmModelSession` 也读取同一优先级默认值。
现有 monolithic policy 节点按 `robot.runtime_scheduler.enable` 同时设置这两个默认值。
开启调度时，加载的 policy Session 使用 managed 完成等待；关闭时仍按 direct 模式加载。
typed model services 尚未接入这个开关；执行策略接口不代表后续感知调度器已经接线。

`TorchModelSession` 和 `LeRobotTorchModelSession` 共用 `torch_execution_scope()`。
异常保留原有的开始、阶段、恢复和待清理证据。
设备同步只补充完成证据，确定未开始的失败不改写为已开始。
成功推理只在 `managed_execution=True` 时增加完成等待。

CUDA 和 NPU Session 在加载时固定具体设备索引。
后续执行和关闭等待同一设备，不依赖调用线程的当前设备。
LeRobot 的 `resource_domain` 仍按设备类型分组（如 `torch:npu`），多卡共用保守额度，不表示单卡身份。

LeRobot 的预处理和后处理也可能提交设备工作，使用相同的完成工具。
processor 完成不表示 policy 模型状态已推进。
预处理后的请求校验错误保留设备证据，不记录持续的 policy 故障。
managed processor 成功返回前等待完成；direct 正常成功路径不新增等待。

managed CUDA/NPU 成功路径只等待进入回调时的当前流。
回调必须在这个流提交工作，或在返回前将子流汇合到这个流。
接口不支持返回后仍独立运行的子流。
异常、关闭和故障恢复仍使用保守的设备级等待。
`force_completion_on_error` 只控制异常路径，成功等待由 `complete_on_success` 控制。

如果 direct 返回结果时已超时或已取消，Session 先确认设备完成再丢弃结果。
如果完成等待失败，执行结果保持未知。
Session 成功返回后，pipeline 在下一检查点丢弃 direct 结果时也确认设备完成。
当前帧只记录已参与执行且尚需完成等待的 Session；已取得的失败完成证据直接复用，不等待未执行的组件。
多个完成检查合并证据，任何未确认完成的工作都不能被后续成功等待覆盖。

两种模式都在执行异常和关闭时确认设备完成。LeRobot reset 也先等待设备完成。
Session 的公开调用方式保持可用。
SpeechDirection 的 FullSubNet Torch 执行异常也使用统一完成检查。
FullSubNet 关闭和故障后的 reset 先等待设备完成。
等待失败时，FullSubNet 保留资源并允许重试关闭。
设备连续 2 次关闭排空失败后，close 放弃等待并释放引用。
放弃时完成状态未知，日志记录放弃事件。
健康状态的 FullSubNet reset 复用已完成的主机输出证据，不等待其他流。
reset 重建状态后的完成等待只覆盖提交初始化工作的当前 CUDA 流。
FullSubNet 也在构造时固定 CUDA 索引，模型、状态和完成等待使用同一设备。
Torch 执行失败后的同步错误保留在 `completion_error` 诊断中。
Torch 异常边界清除已退栈的局部引用。
Handle 的 direct 和策略调用失败诊断都保存不带异常链和 traceback 的快照，避免长期持有被拒绝交付的请求和结果。
Handle 不缓存抛出的关闭异常，关闭重试后不保留旧调用栈。
ACL 的次要清理错误写入 `details.cleanup_error`，归一化和序列化后仍保留主错误。
D2H 主机缓冲释放失败时，原生指针进入隔离列表，已退栈的输出数组引用从异常帧清除。

ACL、HMM、RKNN 资源在释放成功后才清除所有权。
ACL 保留未释放资源供重试。RKNN/HMM 的重试限制见「关闭失败与恢复限制」。

JPEG codec 自有资源释放后，关闭重试只继续释放 ACL lease，不重新绑定已销毁的 context。

HMM、Ascend、Hisilicon 和 ONNX 的提交前输入错误报告 NOT_STARTED。
ONNX 在原生调用前复用 binding 校验，检查 dtype、rank 和静态维度。
Ascend 输入诊断也属于提交前输入边界。
Ascend 在公共 role 边界合并前序完成证据，Host 编排子类也保留该事实。
ONNX 复用 Session 的多 role 进度合并。
输入字节长度错误报告 role、输入索引、实际长度和预期长度。

RKNN 原生推理成功后的输出校验错误保留 COMPLETED，多 role 聚合也保留这份证据。
RKNN 的空输出检查和数组转换也保留完成证据。
Session 公共入口的执行后校验不抹除完成证据。

ACL 同步执行或共享 stream 同步成功后，输出读取错误保留 COMPLETED。
静态 dataset bank 复用普通 ACL 路径的输入准备与 Host 输出读取。
bank 输入拒绝报告 NOT_STARTED；完成后的输出错误仍要求恢复已推进的状态。
HMM 完成后的输出提取和 NumPy 转换也保留完成证据。
Ascend、Hisilicon 和 ONNX 的 role 输出映射复用 Session 完成边界。
Host 输出错误保留状态修改与恢复要求，但不占用已经空闲的设备额度。
direct Torch 校验失败时复用设备完成等待。

HMM、RKNN、ACL priority stream pool 和语音 ACL runner 接受可选的 `cleanup.defer` 所有者。
传入 `cleanup` 时，外部回滚对象保留构造中途取得的资源。
未传入 `cleanup` 时，构造器先尝试关闭。
如果关闭失败，初始化异常通过 `cleanup_owner` 保留重试入口。
构造回滚失败统一通过 `rollback_errors` 记录清理错误文本。

FullSubNet Torch 在构造设备对象前登记回滚。
ASR 的导入失败保留原始原因和清理所有者。
ASR 节点入口在构造失败后重试这个所有者。
ASR Handle 加载失败、加载后的采样率校验失败及 VAD 初始化失败都保留相应所有者。
ASR 节点的 ROS 接口或定时器初始化失败也保留节点所有者。
构造函数统一移交 `cleanup_owner`，入口和嵌入调用方都能重试关闭。
节点关闭同时覆盖尚未交给 VADModule 的 runtime；只有成功清理才删除引用。
正常退出也保留待清理所有者，并尝试关闭 ROS。

SpeechDirection 入口在节点或 provider 清理抛错时仍执行 ROS shutdown。
SpeechDirection 创建定时器失败时，构造器关闭已经启动的 runtime。
入口的清理范围包含节点和依赖构造，provider 重试未完成的释放。

Hisilicon worker 未退出时 `close()` 返回错误并保留进程，关闭开始后不可再次启动或执行。
执行时 worker 退出保留 `worker_exited` 和未知结果证据，并要求 runtime reload；不能通过重试请求跳过恢复。

Hisilicon 请求写入和响应等待都检查请求期限、取消令牌和协议关闭标记。
取得协议锁前或首字节写入前取消、超时，均报告 NOT_STARTED。
这两种拒绝允许后续请求继续使用 worker。
读写等待使用 poll，支持编号不小于 1024 的管道描述符。
请求通过非阻塞管道分段写入，短写继续发送剩余字节。
部分写入后中断、响应中断或协议错误保留未知结果，并要求 reload。

部分写入中断或响应读取取消、超时后，协议拒绝继续发送请求。
已有进程所有者仍负责清理。
请求退出等待后，Handle close 才能回收 worker。
协议层 close 也能中断 I/O 等待，继续执行 terminate/kill。
无期限且未取消的 Handle 请求仍会等待。

视频编码能力探测把驱动的 FFmpeg 错误视为设备不可用。

### 资源回滚与生命周期预算

`LoadRollback` 逆序释放资源；某项失败后保留该项及其依赖，下一次回滚从失败项继续。
`ModelSession` 保留失败加载的回滚记录，后续 `close()` 可以重试。
Factory 和 Session 共用 `LoadRollback`。
Factory 异常通过 `cleanup_owner` 移交待清理所有者。

感知和 ZipVoice 插件构造失败后先关闭 Handle，仅在清理未完成时移交 provider。
ZipVoice 在 Handle 真正关闭后才标记自身关闭，构造清理失败不会覆盖原加载异常。
`RuntimeProviders.retain_cleanup()` 保留 Factory 和插件构造失败的所有者。
provider 先关闭这些所有者，再关闭准入 provider 和 ACL provider。
`process_resources` 统一列出进程资源，关闭顺序和 Handle 所有权过滤共用这份名单。

`InferencePipelineManager` 在 direct 和 priority 模式下都重试未完成的关闭。
关闭重试是固定行为，不提供关闭重试开关。
Manager 启动失败时通过 `cleanup_owner` 移交待清理所有者。
Pipeline 和 Model Service 节点把关闭失败的所有者交给 provider。
服务退出时，即使节点关闭失败，也会尝试关闭 provider 和 ROS。

模型加载成功后，如果节点创建 ROS 实体失败，构造函数执行相同的关闭路径。
Provider 关闭失败时，外层异常保留 `close_pending=True` 和原始原因链。

`lifecycle_budget` 让嵌套清理共享期限。
`lifecycle_scope` 传递控制调用的期限和取消令牌。

启动检查、控制锁等待、ACL 完成等待和 Hisilicon 进程等待共享剩余预算。
期限耗尽后，关闭操作保留资源供重试。预算不能中断已经进入的不可取消 SDK 调用。
控制调用的取消令牌也约束控制锁和排空等待。

Hisilicon 的 EOF、terminate、kill 和 stderr 线程等待轮询取消令牌。
每段等待不超过 50 ms，取消后保留进程所有权供关闭重试。

无参嵌套控制调用继承外层令牌，并采用更严格的期限。
回滚、provider 和 ACL 所有权锁的等待也使用这个预算。

ACL 在各次独立释放调用之间检查预算，保留尚未释放的资源。
Handle 在加载前、每项资源加载前和发布 READY 前检查启动条件。
HMM 和 RKNN 在后续原生初始化前再次检查启动条件。

多角色请求和 pipeline stage 透传已有 metadata。
`StageTicket` 是可选的阶段身份工具；当前生产调度路径不自动创建或消费它。
派生 `RuntimeContext` 保留 `managed_execution`。

### Handle 执行策略与流边界

`bind_execution_policy` 一次性注入 `RuntimeExecutionPolicy`，返回本地执行授权对象。
私有本地入口和 execute/step 的兼容参数 `_dispatch_key` 接受这个对象。
未携带授权对象的公开 execute、stream、reset、close 使用策略；策略失败后不会直通。
策略可扩展 `submit_frame`。`StageTicket` 是无 scheduler 依赖的不可变交接身份。

基础策略要求 `ready/execute/reset/close/cancel`。
策略必须负责取消尚未进入本地执行的排队请求。
流策略还要求 `open_stream/step/reset_stream/close_stream`。
stream runtime 组装时也检查这 4 个方法，缺少控制方法不能视为清理成功。
策略只能在 Handle 空闲时绑定。
`context_provider` 必须保留请求 ID 和同一取消令牌，只能缩短调用方期限。
原生 Session 必须使用 `managed_execution=True` 加载。
Handle 拒绝把已按 direct 模式加载的 Session 改绑策略。
先绑定再加载时，Handle 也检查这个条件。
阶段执行继续使用 `StagedModelExecutor`，Handle 不另设阶段控制入口。

`drain_managed` 校验 authority，撤销准入并持有控制锁。
drain 使用共享生命周期预算，完成后仍需明确恢复或关闭。
drain 还取得已有 stream 控制锁，避免与 reset/close 同时访问设备。
组合 assembly 的全部 Session 和额外可执行组件都必须支持 drain；缺少任一完成证明时拒绝操作，不归还额度。
drain 展开 Factory 的嵌套 assembly，处理自引用并按身份去重 Session；编排包装器不负责设备完成证明。
assembly 未声明 Session 时，执行器自身必须提供 drain。
独占流 reset 还取得运行时控制锁，防止等待中的 drain 丢失准入屏障。
只有当前调用携带有效 authority 时，Handle 才复用外层准入。
Handle 统一处理策略的同步异常和 Future 异常。缺少完成证据的派发失败保持结果未知。
原始取消和期限异常的默认属性不构成未开始证明。
显式 `operation_started=False` 或 `NOT_STARTED` 证据表示请求未开始。
Session 在真实后端调用前明确提供 `NOT_STARTED`。

Torch callable 内的裸控制异常仍触发设备完成等待。
只有明确的原生未开始证据允许跳过等待。

通用 pipeline 在 StageFrame.evidence 字段汇总执行证据。
后续 role 的未开始不能覆盖整个请求已经执行的事实。
Staged worker 的检查点和错误回调使用同一帧证据；Visual snapshot 不把旧帧证据带入新的 Action 请求。
RequestDirectExecutor 同样保留前序 role 的执行和状态修改证据。原生提交前的输入校验失败报告 NOT_STARTED。
frame 同步提交异常即使尚未返回 Future，也按异常中的启动证据执行隔离；open_stream 保留原生拒绝证据。

未绑定策略的 Handle 使用既有 direct 调用方式。
`ready` 包含策略可用性，`local_ready` 仅表示后端状态。
期限和取消贯穿本地执行与控制操作。

普通请求、frame 和 stream step 共用活动请求身份及取消登记。
正在开流的请求也登记取消令牌，支持公开 `cancel(request_id)`。
绑定策略或支持 frame 的执行器要求活动 `request_id` 唯一。
无策略的同步 direct 请求保留并行使用同一 ID 的兼容行为。
按 ID 取消会通知当前所有匹配的请求，完成一个请求不会丢失其余登记。
Handle 在条件锁外同步调用取消回调。
回调必须及时返回，只发送通知，不等待当前执行或后续 policy/executor 取消钩子。
取消预算不能强制中断任意 Python 回调；阻塞观察者不属于支持范围。

取消先设置本地令牌，再通知 policy，最后调用 executor 的取消钩子。
任一取消钩子抛错不会阻止另一侧收到通知。
executor 的兼容签名在调用前匹配，内部 TypeError 不触发重复调用。

direct 准入等待支持取消，取得额度后再次检查 context。
等待实例或共享域额度的活动请求还检查 Handle 的现有生命周期状态。
close/reset 撤销准入后，这些未进入后端的等待者退出，避免排空等待自身持有的额度。
stream step 复用实例及资源域准入，并遵守 Handle 活动执行上限。
request 和 stream 在退出原生准入范围前归一化异常，共用证据转换规则。
原异常的 `OutcomeEvidence` 优先；提交后的普通取消不构成完成证明。
无状态模型的普通异常也需要完成证据。
policy 及其派发的本地执行采用相同的未知结果判定。
如果原生执行结果未知，准入对象继续持有额度。
成功恢复、完成排空或关闭后，Handle 才归还额度。
`per_stream` 的成功 `close_stream` 必须证明该流原生工作和资源已排空，随后按流 ID 归还额度。
`runtime_exclusive` 的逻辑关流不证明设备完成；运行时级故障的额度也不能由单流恢复归还。
运行时要求恢复时，`reset_stream()` 拒绝调用；使用 `reset()` 完成恢复后才开放准入并归还额度。
Voice ASR 的 VAD 公开 reset 使用运行时级 reset。
VAD 仅在 reset 成功后清空主机音频上下文。

运行时 reset 内的 Host 流清理不归还额度，全部原生 Session 恢复成功后才统一释放。
如果没有独立的原生 reset，活动流通过完整 reset_stream 恢复。
如果独占流已关闭且没有原生 reset，恢复返回 reset_unsupported。
此时额度继续隔离，调用方必须关闭并重建运行时。

独占流正在关闭时发生未知执行，也保留运行时恢复要求。
共享设备域的其他 Handle 也受剩余额度约束。
请求和 stream 的确定未执行准入拒绝都不增加模型故障计数。
frame 准入失败同步抛出 `ExecutionFailure`，已准入请求通过 Future 交付结果。
policy 的 execute、frame、step 和 open_stream 结果交付都检查期限和取消。

这些工作入口在 policy 派发前也检查 context，已失效请求报告 NOT_STARTED。
控制入口保留原有的不可逆关闭语义。

首个 role 尚未执行时取消，Session 报告 NOT_STARTED。
已完成 role 之间取消，多角色编排保留 COMPLETED 证据。
SAM2 在 decoder 输入构造前和最终后处理前检查取消与期限。
Session 和 frame 已完成后发生超时或取消，仍保留 COMPLETED 证据。
Pipeline 保留原始异常类型，并将归一化证据传递到 ROS 边界。
执行结果已知但资源清理未完成时，Handle 也会停止准入。
未指定更大故障范围时，stream 按 state bank 的所属范围停止准入，已知结果保持已知。
有状态步骤已推进状态但结果交付失败时，流必须恢复。
完成证据允许释放设备额度，但不证明调用方的主机状态已经同步。
后端明确要求 `RESET_RUNTIME` 或运行时 `RELOAD` 时，不得降级为流级恢复，即使该流正在关闭。
request 路径同样执行显式恢复要求，包括已知结果的故障；`RELOAD` 不降级为 reset。

对外异常、最后失败快照和 Handle 状态使用同一次恢复判定。
reset 完成后，Handle 先完成恢复收尾，再检查结果交付期限和取消。
后置控制失败保留 COMPLETED，不重新隔离已经恢复的运行时。

流级 `CLOSE_STREAM` 和 `RELOAD` 阻止后续 step，也禁止用 reset_stream 替代。
独占流结果未知时，这两种要求升级为运行时 RELOAD。
reset 明确要求 RELOAD 后，允许重试 reset 的配置也不能绕过要求。

reset 等待中的在途故障会更新恢复状态，等待超时不会重新开放旧的 READY。
单流 reset 等待结束后，在条件锁内重新核验流状态和恢复要求，不覆盖新的 CLOSE_STREAM 或 RELOAD。
本地 reset_stream/close_stream 失败与 step 共用恢复处理，明确的运行时恢复要求关闭整个 Handle 的准入。

同步 execute/step 内重入同一 Handle 的 reset/close/drain_managed 会立即拒绝，不等待自身，也不改变关闭状态。
open_stream 的执行线程也参与这个重入检查。
Handle 的本地和 policy 帧结果共用交付重入检查，避免回调等待自身 worker。
活动 step 重入同流 reset_stream/close_stream 时，公开入口先拒绝，再决定是否派发。
拒绝重入不会关闭流，也不会阻止后续 step。

SpeechDirection 的 Host、FB、SB、VAD 共用一组流状态。
如果音频步骤中断，流适配层要求 reset。
如果底层明确要求 RELOAD，适配层保留 RELOAD 及恢复可用性。
单 role 的 NOT_STARTED 不能覆盖整步已推进的 Host/前序 role 状态。

SpeechDirection 开流时的隐式 reset 继承开流期限和取消令牌。
SpeechDirection 公开 `reset()` 调用运行时级 reset，保留原 stream ID。
生产组装的 stream 层只清理 Host 状态，Handle 统一重置拥有的原生 Session，每个 Session 只重置一次。
公开 `reset_stream()` 不接受内部控制锁开关；单流 reset 仍完整重置该流状态。
语音组合最多使用一个共享资源域；FullSubNet Torch 与 Silero Ascend 混合时仍取得 Ascend 域额度。
多个资源域或不一致的域上限在构造时明确拒绝。

共享域额度覆盖完整音频 hop，包含 Host 和其他后端的处理时间。
关闭 runtime scheduler 时也保留这个安全边界，不承诺逐 role 准入或延迟上限。

Handle 在恢复完成前拒绝下一步输入。
设备完成证据与整步状态恢复要求分别保留。
流 reset 在等待活动 step 前设置控制标记，避免后续输入抢占。

关闭开始后不能重新准入，清理未完成时保留资源供重试。
Handle 拥有整个 assembly，也关闭加载失败时尚未尝试加载的组件。
本地关闭完成后，策略关闭仍需完成确认。
Handle 拒绝帧回调内的 reset/close，外部所有者负责控制操作。
直接关闭 staged executor 时，自身 worker 返回 pending。
ACL 完成回调也拒绝等待同一 worker 的排队任务。

正常输出拷贝使用局部 Host buffer，只有释放失败才进入隔离。
worker 退出等待遵守共享期限。
两个 staged worker 退出后，executor 清空最后一帧快照。
SpeechDirection 的线程 join 分段检查剩余期限和取消令牌。

stream close 先校验参数，再在等锁或派发策略前记录关闭意图。
reset 不能重新开放这个 stream。
open 返回过迟或与 Close 竞争时，Handle 保留 native stream 的清理所有权。
stream 具备所属 handle 身份校验；关闭成功后释放 native state，仅保留最近 128 条关闭记录。
失败或未确认关闭的 stream 继续占有资源，不因历史记录淘汰而丢失。

SpeechDirection 的底层关闭历史也限制为 128 条，旧的已关闭对象仍支持幂等关闭。
SpeechDirection 的 stop、兼容包装和节点销毁保留 pending 清理责任。
节点向 provider 移交 Runtime，让重试先等待 worker 退出。
Runtime 构造失败但 Handle 已创建时，节点仍关闭该 Handle，并移交未完成的清理。
stop 的首个错误不覆盖后续未完成清理；是否需要重试由最终线程和 Handle 状态决定。
SpeechDirection 关闭流时只撤销逻辑身份，开流前才重置独占状态。
最终销毁直接进入后端关闭，不要求设备先恢复可用。

FB、SB 和 VAD 复用外层 step 的取消令牌和期限。
SigLIP2、SAM2 automatic 和 GraspGen 把请求优先级传到各角色的实际 OM 执行。

### Bundle

每个可部署目录必须包含唯一的 `inference_manifest.json`。Policy bundle 还必须包含 LeRobot 语义文件：

```text
policy_bundle/
├── config.json
├── model.safetensors                         # Torch deployment 需要时存在
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # PI0.5 / SmolVLA 按需存在
├── artifacts/
│   ├── ascend/<deployment>/...
│   ├── hisilicon/<deployment>/...
│   ├── rknn/<deployment>/...
│   └── hmm/<deployment>/...
└── inference_manifest.json
```

LeRobot 拥有 `config.json`、processor JSON、processor state、tokenizer 和原生权重。
IB-Robot 只读这些文件，不会添加字段、删字段、重写 device 或创建临时策略目录。
部署信息只属于 `inference_manifest.json`。

### Deployment

一个 manifest 可以为同一策略声明多个命名 deployment，例如 `cpu`、`cuda`、
`rk3588`、`ascend_310p3` 或 `lq50`。Pipeline 选择的是 deployment 名称，而不是后端名。

部署通过显式的 v3 runtime profile 声明后端、target 和运行实例字段：

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 1,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "torch",
    "target": {"runtime": "torch"},
    "profile": {"device": "cpu"}
  }
}
```

编译 deployment 声明 target、artifact、执行顺序和完整 runtime ABI bindings：

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 3,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "rknn",
    "target": {
      "soc": "rk3588",
      "runtime": "rknn-lite2"
    },
    "profile": {
      "target_name": "rk3588",
      "core_mask": 7,
      "device_id": 0
    }
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/generations/<uuid>/policy.rknn",
      "format": "rknn"
    }
  },
  "execution": ["policy"],
  "bindings": {
    "policy": {
      "inputs": [
        {
          "semantic": "observation.state",
          "runtime_name": "observation.state",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 6]
        },
        {
          "semantic": "observation.images.top",
          "runtime_name": "observation.images.top",
          "index": 1,
          "dtype": "float32",
          "shape": [1, 480, 640, 3],
          "layout": "NHWC"
        }
      ],
      "outputs": [
        {
          "semantic": "action",
          "runtime_name": "action",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 100, 6]
        }
      ]
    }
  }
}
```

`execution` 中的每个角色都必须同时存在 artifact 和非空 binding group。图像 binding
必须显式声明 `NCHW` 或 `NHWC`；非图像 tensor 不得仅根据 rank 自动转置。多模块模型
使用匹配的 `internal.*` semantic，或使用 `device_links` 声明 producer、consumer、
device-pointer ownership 和 inference lifetime。

### Pipeline

Pipeline ID 是模型实例和 ROS 路由的稳定标识，必须匹配
`^[a-z][a-z0-9_]{0,62}$`。每个 pipeline 独立拥有：

- 策略 bundle 和命名 deployment
- LeRobot preprocessor / postprocessor
- policy codec 与 binding execution plan
- `ModelRuntimeHandle` 及其独立准入状态和生命周期
- Action、reset、health、action output 和分布式 transport endpoints

`InferencePipeline` 是策略模型 facade，把既有 policy 请求/结果契约适配到
`ModelRuntimeHandle`。Handle 统一生命周期、准入、deadline、cancellation、health 和 diagnostics，并按
`RuntimeAssembly` 的 owned component 顺序加载、反向释放资源。编译模型由
顺序或 staged executor 按 `InferenceStage` 编排，共用 `ComponentModelExecutor` 的组件辅助逻辑；
循环 family 通过 `IterativeStage` 驱动多个 role。
同一个 `ExecutionContext` 把 request ID、deadline 和 cancellation token 传到各 stage 与
`ModelSession` resource；session 持有跨 role 复用的 vendor 设备资源。
两种策略都通过 `ModelSession.execute_role()` 校验角色张量，Ascend 在内部选择隔离 dataset，
继续复用语义绑定和诊断采集。independent 模式通过 PI0.5 policy adapter 验证 request-scoped 的
`vlm` / `action_expert` 双段 ABI：prefix 仅依赖图像和语言，不允许状态、noise/time 或反向链接依赖。
manifest 描述模型 ABI；异步执行与资源隔离能力由加载后的 runtime 验证。
加载后的 Ascend backend 必须提供隔离异步 role 执行及 priority streams；executor 保证不可变发布、
prompt 匹配、快照时效和代际隔离。模型类型或两个角色本身都不足以证明可跨请求复用中间结果。

默认 endpoint：

| 接口 | 默认值 |
| --- | --- |
| 本地节点 | `inference_<pipeline_id>` |
| 云端节点 | `inference_<pipeline_id>_cloud` |
| Action server | `/inference/<pipeline_id>/dispatch` |
| Reset service | `/inference/<pipeline_id>/reset` |
| Health topic | `/inference/<pipeline_id>/health` |
| Action output | `/actions/<pipeline_id>` |
| 分布式 request | `/inference/<pipeline_id>/request` |
| 分布式 result | `/inference/<pipeline_id>/result` |
| 分布式 heartbeat | `/inference/<pipeline_id>/heartbeat` |
| 视频 descriptor | `/inference/<pipeline_id>/video/descriptors` |
| 视频 status | `/inference/<pipeline_id>/video/status` |

## 推理调度控制面

唯一开关是 `robot.runtime_scheduler.enable`，默认 `false`。
SIM/benchmark 始终直通；机器人级开关不会启用一个 `inference.enabled=false` 的控制模式。
调度控制面使用 policy 调度器、FIFO/EDF、profile 及 Open/Dispatch/Close 契约。

- 开关缺失或为 `false`：保持 legacy 路径
  `executor.inference_pipeline -> pipeline /dispatch + /reset`。Launch 只创建
  `pipeline_policy_node` 和 `action_dispatcher_node`；pipeline 不注册 scheduled action server，
  不发布 serving status，也不创建 product session 或 Global Scheduler。
- `runtime_scheduler` 块存在且 `enable: false` 时，可以原样保留完整调度配置；这些字段处于 dormant 状态，不生成
  runtime policy，也不改变 legacy 节点参数、接口、线程数或 backend 执行方式。因此启停调度只需修改
  `robot.runtime_scheduler.enable`。若整个 `runtime_scheduler` 块缺失，scheduled 字段仍按 unknown field 拒绝，以保留拼写检查。
- 开关为 `true`：Launch 创建 pipeline、`global_inference_scheduler_node` 和
  `scheduled_action_dispatcher_node`。产品调用只经过 Global 的 Open、Dispatch、Close endpoint，
  Global 负责逻辑 session、多 pipeline binding、逐请求路由、deadline precheck 和 quarantine；pipeline 负责
  generation fence、实际 admission、执行和 drain。Launch 只为 `required: true` pipeline 注册
  进程退出后的全局 Shutdown；optional pipeline 退出后由 readiness 和路由逻辑将其排除。该路径当前只接受
  `execution_mode: monolithic`。

生产执行路径只支持 schema v3 whole-graph plan。公开 Open 只建立 route-independent 的逻辑 session 和
logical generation，不选择模型、不检查 fallback，也不访问 pipeline。每个 Dispatch 都携带自己的 target
pipeline、fallback chain、priority 和 deadline；候选首次被该 session 选中时，Global 才向对应 pipeline
下发私有 Open/reset 并记录其 pipeline generation。同一个逻辑 session 可按需建立多个 pipeline binding。
Close 停止新 admission，并逐一 drain/reset 实际使用过的全部 binding；从未 Dispatch 的 session 可直接关闭。
下游 Close 明确 `NOT_STARTED` 时，公共 Close 保留该 outcome 与 recoverable 标志，session 留在 `CLOSING`，
不因未开始而隔离 pipeline。可恢复的失败可重试 Close，但不能恢复 Dispatch；重试只处理尚未排空的 binding，
使用新的 operation ID。混合结果中的 `UNKNOWN` 优先隔离，已执行的失败仍返回失败的 `COMPLETED`。
身份校验通过且成功排空的 Close 会回收该 binding 的 reservation 与 operation；仍有 waiter 的
operation 在 waiter 退出后回收，迟到结果不会重新占用已释放的容量。

独立阶段的 producer 可配置 `scheduling.stages.<producer>.max_snapshot_age_ms`，默认 `5000`，必须是正整数；
仅允许配置在 manifest 执行顺序中的第一个角色，terminal 和 sequential 模式拒绝此字段。snapshot 必须匹配
generation 与 prompt，且观测年龄不超过该阈值。Pipeline Node 将 ROS 采样时间相对当前 ROS 时钟的年龄
换算到单调时钟，producer 排队和执行时间均计入年龄；Dispatch 刷新也保留同一采样时间。没有观测时间元数据的
直接 runtime 调用从首次准入计时。刷新后仍过期的观测会失败，不会通过重复计算延长有效期。

### Legacy 生命周期兼容

下列生命周期行为仅在启用 scheduler 时生效：

- reset 跳过明确无状态且不支持 reset 的非 executor 组件，避免无效 reset 报错。
- 有状态 policy 的后端已完成但结果迟到，或已开始后发生失败时，后续 `infer()` 会持续报告未就绪；成功 reset 后才能恢复。

LeRobot Torch session 加载后发现的 reset 能力始终同步到 Handle。
调度启用、关闭或配置缺失时，`pipeline.reset()` 都真正调用 `policy.reset()`。

reset 不再是无害的空操作。drain 或 `policy.reset()` 抛错时，Handle 记录故障。
启用了 `retry_failed_reset` 的 Handle 进入 RESET_REQUIRED，reset 可以重试。
其他 Handle 进入 FAILED，要求 RELOAD。
reset 服务返回 `success=false`，后续请求以 `recovery_required` 拒绝。
reset 服务的调用方必须按失败处理返回值。

`robot.runtime_scheduler.enable=false` 或默认关闭时，保留原有拓扑和同步执行方式。
清理重试独立于调度开关：`PipelineManager.close()` 在两种模式下都重试 `close_pending`。
Torch 异常、reset 和关闭会确认设备完成，direct 成功推理不额外同步。
Handle 的取消和不可逆 stream Close 边界也适用于 direct 路径。
direct 无状态同步调用的普通异常保持完成已知；原生后端显式上报的未知结果仍触发隔离。
设备完成确认、资源安全释放和损坏状态隔离是两种模式的共同保证。
测试同时覆盖 launch 参数和运行时迟到结果、reset 与 Close 行为。

Global 只接受 `InferenceServingStatus.scheduling_capability_schema_version == 1`；零值和未知版本均在
状态准入时拒绝，不按旧版本兼容处理。三个私有 `*PipelineBinding.action` 的字段注释定义身份、重试和
时间契约：`deadline` 必须是正值 UTC Unix 绝对时间，公开接口的默认值由 Global 解析；
`obs_timestamp` 使用 pipeline 的 ROS 时钟，零值表示当前时刻。Dispatch 的 pipeline generation 必须
来自成功 Open，不能为零；清理迟到 Open 的 Close 可用零 generation，但仍必须匹配 binding 和 boot。
Open/Close 不支持 action 取消；Dispatch 取消依赖后端能力。取消被接受或调用超时均不代表设备完成，
只有 Close 的 `success=true` 才证明该 binding 已排空或无需清理，失败的 `COMPLETED` 不能作为排空证明。
私有 Close 以 `(session_id, operation_id)` 区分清理尝试：同 ID 重放，新 ID 经 binding 状态机确认后才排空，
在途 Close 不允许重叠。正常成功必须满足 `drained_generation > closed_pipeline_generation > 0`；
零/零仅允许 generation-zero 的 `cleanup_not_needed`。这些 action/status 是 policy/VLA 私有协议，
感知和语音不承接 `action_chunk`。状态的 deadline-admission 能力来自实际 executor，表示支持该准入方式，
不保证 deadline 内完成；公开优先级只用连续上界表达，不再重复发布 mask 或无消费者的能力指纹。

priority-0 的资源排序和 fallback 取决于配置，默认不启用 profile 准入：

| `global_policy` | `priority_zero_deadline_admission.enable` | 行为 |
| --- | --- | --- |
| `fifo`（默认） | `false`（默认） | 直接下发 target，不建立资源等待队列；不接受非空 fallback chain |
| `fifo` | `true` | 用离线 profile 做完成时间准入，同资源按 FIFO 串行下发；支持 fallback，仅接受 sequential pipeline |
| `edf` | `false` | 同资源等待中的请求按绝对 deadline 排序，同 deadline 按 FIFO；支持 fallback，仅接受 sequential pipeline，不使用 profile 预测完成时间 |

EDF 不抢占已开始的请求，也不保证 deadline 前完成。deadline 驱动（EDF 或 FIFO profile 准入）的
priority-0 是 sequential-only 契约：independent pipeline 把一个请求拆到两个 worker 上重叠执行，
per-request deadline 排序无法评估也无法兑现；robot_config 会在配置加载期拒绝该组合的 target 和
fallback 链，Global 也会按 serving status 能力跳过 independent 候选。FIFO profile 准入按 `[target, *fallback_chain]`
检查匹配身份、输入契约、prompt 大小和有效期的离线测量：已绑定候选使用 `full_infer`，未绑定候选
还需要 `session_open`。p99 估计加 safety margin 必须落在 deadline 内，获得资源下发权时再次检查。
缺失或无效 profile 的候选会被跳过；无可行候选时返回 `no_feasible_deadline/NOT_STARTED`。
启用 scheduler 时，不支持的默认 FIFO/fallback 组合和 deadline 驱动 priority-0 选择 independent pipeline
都会在配置加载时报错。priority 大于 0 只使用 target，不受该 sequential-only 约束。
仅 `NOT_STARTED` 允许继续 fallback；`UNKNOWN` 保持隔离，绝不重试或 fallback。Global ingress 仍保持有界：
Open/Close 各两个执行 context；Dispatch 共四个 context，其中 lower-priority 最多占两个，不能耗尽为 priority-0
保留的进入能力。priority-0 可使用任意空闲 Dispatch context。

profile 文件顶层只包含 `closure_profiles`。Global 消费的 entry scope 为 `global_proxy`。action-generation entry
必须携带 pipeline 发布的 `input_contract_fingerprint` 和标定覆盖的 `prompt_bytes_max`；session-control entry 使用
空 input fingerprint 和 `prompt_bytes_max: 0`。每个 entry 还必须携带匹配的 deployment、hardware、
`profile_compatibility_fingerprint`、work class、closure key、priority、采样时间、样本数、goal acceptance p99.9
和 closure latency p99。Global 选择能够覆盖当前 prompt 的最小 profile bucket；超出覆盖范围时 fail closed。
这些统计量定义 p99 admission SLA，不是绝对完成保证；最终估计仍叠加配置的 safety margin。

priority 大于 0 的请求只路由到 target，不查询 profile，也不尝试 fallback；调用方 deadline 不参与业务准入。
Global 会生成独立的内部 request timeout 并传给 pipeline，用于约束 RPC、取消和故障恢复，避免调用永久挂起。

通用 priority 通过 scheduled request metadata 传给 monolithic backend；`0` 为最高优先级，数值越大优先级越低。
通用 wire 取值为非负 int32，不施加 backend-specific 上限。只有支持多优先级的 backend 才暴露显式的
generic-to-native priority mapping capability；没有该 capability 的 backend 为单优先级，只接受 priority 0。
当前只有 Ascend 暴露多优先级 mapping。
`AscendOmModelSession` 在 scheduled 模式加载时查询 `acl.rt.device_get_stream_priority_range()`，并通过
`acl.rt.create_stream_with_config()` 为硬件支持的每一级优先级创建可复用 stream。Ascend 接受 `[0, 7]`，并将
通用 priority 一一映射到同编号 ACL priority；大于 `7` 的请求在模型执行前直接拒绝。模型通过
`acl.mdl.execute_async()` 下发，并在读取输出或复用 dataset/buffer 前调用
`acl.rt.synchronize_stream()`；同一 backend 实例仍保持单 in-flight。调度模式不设置进程内跨 context 数量上限，
从而允许 ACL 在不同 context 的 stream 之间执行硬件优先级抢占；实际并发能力由 Ascend runtime 和硬件决定。
disabled/legacy 模式不创建 stream，继续调用同步 `acl.mdl.execute()`。
pipeline 的 `action_generation.max_in_flight` 不得超过 backend 声明的 `max_in_flight_per_instance`；没有可用
execution slot 时立即返回 `pipeline_busy/NOT_STARTED`，不在 pipeline 内排队。该容量必须是正整数；pipeline
Dispatch action server 的有界 ingress 容量至少保留两个 context 以处理重复请求，并随更高执行容量扩展。
serving status 仅在 session state 允许且
`current_in_flight < max_in_flight` 时报告 `accepting_requests=true`。后端报告的
serving status 中的 `hardware_priority_levels` 是该 capability 的只读观测值；请求 priority 超出 backend mapping
声明范围时，pipeline 在模型执行前返回不可恢复的 `unsupported_priority/NOT_STARTED`。Global readiness 至少要求
一个通用 priority level；当 launch 配置的默认 priority 大于 0 时，只额外检查默认 target pipeline 在线且支持该
priority，不要求 fallback 或其他 pipeline 支持多优先级。

`hardware_resource_id` 是 backend 实际运行设备的身份，用于 serving readiness 和 Global deadline reservation；
`resource_domain` 只用于进程内并发准入。两者独立：Ascend priority mode 不设置进程内 resource domain gate，
但仍上报 `ascend:<device_id>` 作为硬件身份。
`hardware_profile_fingerprint` 则是离线标定环境的 SHA-256 身份，必须覆盖 SoC、CANN/驱动/固件版本、频率/功耗
模式和其他会影响时延的配置。profile entry 使用独立的 `hardware_fingerprint` 与其匹配，不能再用 `ascend:0`
这类资源互斥 ID 代替标定环境身份。

分布式推理使用 distributed protocol v6 和原 topic，不参与 scheduler、product session 或优先级抢占。
v6 在 `DistributedInferenceResult` 及两个 `DispatchInfer` action 中新增 `execution_horizon` 字段；
这不是 ROS 二进制兼容变更，升级时 edge、cloud 与所有 action client 必须一起重新生成、构建并
协调部署，握手阶段会直接拒绝旧版本 peer（详见
[ibrobot_msgs README](../ibrobot_msgs/README.md)）。
hardware 目标下，`robot.runtime_scheduler.enable=true` 与 `execution_mode: distributed` 的组合会在 `robot_config` 校验阶段被拒绝。

## Robot 配置

推理直接配置在 `control_modes.<mode>.inference.pipelines` 下：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/so101_act
          deployment: rk3588
          execution_mode: monolithic
          request_timeout: 5.0
          default_task: pick up the banana
          runtime_options: {}
```

相对 `model_path` 只相对于环境变量 `WORKSPACE` 解析。`WORKSPACE` 未设置时会失败，
不会回退到当前目录、YAML 目录或源码目录。执行项目命令前应先运行：

```bash
source .shrc_local
```

### AutoHorizon runtime options（native Torch PI0.5）

pipeline 的 `runtime_options` 支持以下 AutoHorizon 选项（默认值即论文复现值）：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `auto_horizon_enabled` | `false` | 开启 action-expert self-attention 采集并估计 `execution_horizon` |
| `auto_horizon_sampling_step` | `3` | 在第几个 denoising step 采集 attention 权重 |
| `auto_horizon_hold_threshold` | `0.3` | soft-pointer 停止增量阈值 |
| `auto_horizon_entropy_quantile` | `0.9` | 低熵（可靠）行的分位数阈值 |
| `auto_horizon_run_length` | `1` | 判定停止所需的连续 hold 长度 |

约束与生效语义：

- 仅支持 native Torch `pi05` 策略和 `predict_action_chunk` action method；在其他
  policy family 或 action method 上启用会在 session 加载/执行阶段 fail closed，
  不会静默忽略。
- 采集器通过 forward hook 在指定 denoising step 对 action expert 各层做
  GPU→CPU 权重拷贝并在 CPU 上估计，属于模型执行路径的一部分；开启后推理时延
  profile 必须重新标定（见下）。
- 找不到 action-expert `self_attn` 模块（例如 lerobot 升级改动模块路径）或
  `auto_horizon_sampling_step` 超过策略 `num_inference_steps` 时，session 加载阶段
  fail closed 报错——显式开启却无法生效属于配置/环境错误；推理过程中的偶发异常
  （如某步权重非有限）仍安全回退为 `execution_horizon=0`，即完整 chunk；
  edge/dispatcher 侧不重算 attention。
- 选项在 session 加载时生效，修改后需重启 pipeline 进程，不支持运行中切换。
- **profile 标定身份**：`robot_config` 把规范化后的有效 runtime options 计入
  `profile_compatibility_fingerprint`。scheduler 启用时，任何影响时延的选项变化
  （含开启/关闭 `auto_horizon_enabled` 或调整采集参数）都会使旧离线 profile 失配并被
  fail-closed 丢弃，priority-0 准入在重新标定前返回 `no_feasible_deadline`。显式写出
  默认值与省略该选项身份相同，不触发重新标定。
- **升级注意**：本版本首次把 runtime options 计入 `profile_compatibility_fingerprint`，
  因此即使不修改任何配置，升级前标定的离线 profile 也会全部失配并被 fail-closed 丢弃；
  启用 scheduler 的部署升级后必须对每个 pipeline 重新标定生成新 profile，否则 priority-0
  准入返回 `no_feasible_deadline`。

分布式 pipeline 中 policy 在 cloud 侧执行，attention 采集也发生在 cloud。独立启动的
cloud 使用 `cloud_inference.launch.py` 的 `runtime_options_json` launch 参数（默认
`{}`），不会继承 edge YAML 的 `runtime_options`：只修改 edge YAML 不会开启 cloud 采集，
系统仍可正常握手但结果始终回退 `execution_horizon=0`。启用时必须向 cloud launch 传入
相同选项（`robot_config_path` 为必填参数，也用于构建视频流契约），例如：

```bash
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=torch-cuda \
    robot_config_path:=/absolute/path/to/robot.yaml \
    runtime_options_json:='{"auto_horizon_enabled": true}'
```

端侧 dispatcher 需同时选择 `dispatch.chunking: auto_horizon` 才会消费分布式结果中的
`execution_horizon`（两段配置缺一不可，见 action_dispatch README「AutoHorizon 集成边界」）。
端侧 bundle 校验要求 `model_path` 指向一个完整可读的 bundle（含 manifest 引用的
state_file；权重文件可省略——推理不发生在端侧）。已在 edge（aarch64）+ cloud（CUDA）
双机拓扑实测：分布式往返约 216 ms（含 cloud 端 pi05 CUDA 推理），截断后的
`execution_horizon` 随请求自适应变化并被端侧 dispatcher 正常消费。

多模型通过多个独立 pipeline 配置：

```yaml
pipelines:
  action_policy:
    model_path: models/so101_act
    deployment: rk3588
    execution_mode: monolithic
  auxiliary_policy:
    model_path: models/auxiliary_smolvla
    deployment: cpu
    execution_mode: monolithic
```

YAML 是默认配置来源。开发调试时可只覆盖一个明确指定的 pipeline：

```bash
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/robot.yaml \
    control_mode:=model_inference \
    inference_pipeline:=policy \
    inference_execution_mode:=distributed
```

`inference_execution_mode` 为空时完全使用 YAML；非空时必须同时提供
`inference_pipeline`，避免多 pipeline 配置被全局误覆盖。

可以在每个 pipeline 的 `transport` mapping 中覆盖 endpoint。不同 pipeline 的节点名、
Action、service 和 topic 必须唯一。Monolithic pipeline 不允许配置 cloud node、request、
result 或 heartbeat overrides。

## 执行模式

### Monolithic

`pipeline_policy_node` 在一个进程内执行：

```text
ROS observations
  -> contract adapter
  -> InferencePipeline policy facade
  -> ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)
  -> SequentialModelExecutor stages
       -> LeRobot preprocessor + policy codec
       -> ModelSession execute / role execution
       -> action decode + LeRobot postprocessor
  -> DispatchInfer result and /actions/<pipeline_id>
```

进入 processor 前，节点会对策略 `input_features` 所需的每个 observation 执行 readiness
检查：缓存中必须已有时间戳不晚于请求时间、且满足 `align.strategy`（`hold` / `asof` / `drop`）
的样本；当 contract 为该 observation 配置了
`max_age_ms > 0` 时，该值作为在线推理额外的最大样本年龄，与 `asof` 对齐使用的 `tol_ms` 分离。
在线年龄使用节点不可被请求方回拨的本地接收时钟计算，request timestamp 仅用于历史样本对齐选择。
缺失、晚于请求时间或过期的样本会返回可恢复的 `observation_not_ready`，不会静默补零后执行模型。调用 pipeline
reset service 会重置 policy、LeRobot preprocessor/postprocessor 并清空 observation 缓存，下一次推理
必须等待新 episode 的输入。推理、reset 和分布式 cancel 都使用 pipeline `request_timeout`
作为协作式 deadline；锁和 admission 等待会按时退出，backend/processor hook 超时会在其返回时被检测。
reset 或 cancel 结果不确定时会使 edge fail closed，避免 cloud 与 edge episode 状态不一致。

完整机器人启动通常由 `robot_config` launch builder 根据 YAML 创建 pipeline。仅评估一个
pipeline 时可直接使用：

```bash
source .shrc_local
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml" \
    model_path:="$WORKSPACE/models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515" \
    deployment:=cpu \
    pipeline_id:=policy \
    action_server:=/inference/policy/dispatch \
    reset_service:=/inference/policy/reset
```

触发一次推理：

```bash
ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'test-001'}"
```

### Distributed

分布式 pipeline 在 edge 的 `pipeline_policy_node` 中保留 observation 采样、机器人状态单位换算、
非图像 tensor 序列化、action `TemporalSmoother` 和最终机器人单位换算。Cloud 的
`pure_inference_node` 组装完整 raw observation，在同一进程中依次运行 LeRobot preprocessor、
selected backend 和 postprocessor，避免 processor state 跨机器分裂。两端在接受请求前必须匹配：

```text
Robot / Edge host                              Compute / Cloud host
┌──────────────────────────────────┐           ┌──────────────────────────────────┐
│ action_dispatch                  │           │ inference_<pipeline_id>_cloud    │
│       │ DispatchInfer            │           │ pure_inference_node              │
│       v                          │           │                                  │
│ inference_<pipeline_id>          │           │ selected deployment and backend  │
│ pipeline_policy_node             │           │                                  │
│ ├─ observation sampling          │           │ ├─ raw observation assembly      │
│ ├─ state/action unit conversion  │           │ ├─ LeRobot pre/postprocessors    │
│ └─ action TemporalSmoother       │           │ └─ selected deployment/backend   │
│       │ request      ▲ result     │           │       ▲ request      │ result    │
└───────┼──────────────┼────────────┘           └───────┼──────────────┼───────────┘
        │              │                                │              │
        └──────────────┴──────── ROS 2 transport ───────┴──────────────┘
          /inference/<pipeline_id>/request
          /inference/<pipeline_id>/result
          /inference/<pipeline_id>/heartbeat
```

每个 pipeline 使用独立的 request、result 和 heartbeat endpoint；两端使用相同的 pipeline
ID、bundle 和 deployment 身份信息。

图像 observation 可继续使用显式 `mode: dds`，或使用 H.264 RTP/UDP 数据面。RTP 模式下，DDS
只承载 descriptor、status、timestamp mapping、request/result 和 heartbeat；H.264 payload 不进入
`VariantsList`。每路相机使用唯一 stream ID、SSRC 和偶数 UDP port，`port + 1` 保留用于端点冲突检查。
Cloud 必须收到匹配 protocol/session generation、contract fingerprint 和 deployment fingerprint 的
全部 descriptor，并等到 keyframe 与 RTP-to-capture timestamp mapping 就绪后才接受请求。

`encoder_backend` 可设为 `software`、`nvidia`、`ascend` 或 `auto`；`nvidia` 当前仅提供 NVENC 编码，
不能用作 decoder。Software backend 使用 PyAV 15 和其 FFmpeg `libx264`/H.264 decoder；启动 probe
缺少 codec 时会 fail closed。NVIDIA backend 会实际打开一个 `h264_nvenc` session，使用 ultra-low-latency、
zero-delay、无 B-frame 和重复 SPS/PPS 的 H.264 配置；RGB/BGR 到 NV12 的转换仍由 FFmpeg 完成，不是
CUDA zero-copy。Ascend backend
仅在选择或自动探测时查找私有 FFmpeg `h264_ascend`，可用
`IBROBOT_ASCEND_FFMPEG` 或 `IBROBOT_ASCEND_FFMPEG_PREFIX` 指定；标准 RPM 安装的
`/usr/bin/ffmpeg-ascend` 也会自动探测，不替换系统 FFmpeg，也不引入
ACL/DVPP Python 依赖。启动日志（"Video stream startup" 行）报告 configured/selected
backend、endpoint、fingerprint、lifecycle 和 readiness。

RPM 安装的 ffmpeg-ascend（`/usr/bin/ffmpeg-ascend`、`/usr/local/bin/ffmpeg-ascend`
及其 dispatch 的 `/usr/local/ffmpeg-ascend-*/bin/ffmpeg` payload）默认以隔离环境启动：
子进程会剔除继承自其他 CANN toolkit 的安装路径变量
（`ASCEND_TOOLKIT_HOME`、`ASCEND_HOME_PATH`、`ASCEND_AICPU_PATH`、`ASCEND_OPP_PATH`、
`ASCEND_NNRT_HOME`、`ASCEND_NNAE_HOME`、`TOOLCHAIN_HOME`），但保留设备运行时变量
（如 `ASCEND_RT_VISIBLE_DEVICES`、`ASCEND_DEVICE_ID`），以保留多 NPU 配置。私有构建默认不隔离，
可用 `IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1` 强制开启、`=0` 关闭 RPM 默认隔离；
不带隔离探测 RPM payload 已知会在首帧挂起。

Ascend DVPP VENC 通道是 per-device 硬件资源，`DeviceVideoStreamManager` 按
观察值 key 排序对 Ascend 流紧凑编号 1..N（≤128），`device_id` 当前固定为 0
（单 NPU 场景）；多 NPU 需在 pipeline/资源层引入 device 分配契约，不在
manager 硬编码。

`auto` 按 `ascend`、`nvidia`、`software` 顺序探测。310B 优先使用 Ascend，本机 NVIDIA GPU 可用时
自动选择 NVENC，其他 Linux 主机回退 software；显式 backend 失败时不会回退。

RTP/UDP 不提供认证、加密或完整性保护，只允许可信机器人网络。链路中断、descriptor 不一致、显式
backend 不可用、时间映射过期或跨相机 skew 超限都会拒绝推理，不会自动回退 DDS。回滚必须在两端
部署匹配的 `mode: dds` contract。当前 rosbag/MCAP 录制仍采集 DDS image topic；RTP-aware recording
和不可信网络安全属于后续工作。

- pipeline ID
- manifest schema version
- bundle digest
- deployment name
- selected deployment fingerprint
- policy input/output summary
- cloud backend `READY` state

Cloud 示例：

```bash
source .shrc_local
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml"
```

单独调试 distributed Edge 时也可直接使用：

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy \
    inference_execution_mode:=distributed
```

本机同时启动 Edge 和 Cloud：

```bash
ros2 launch inference_service local_distributed_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy
```

Edge 可由 robot YAML 中 `execution_mode: distributed` 创建，也可由上述显式 launch override
创建。成功握手后会绑定唯一 session ID 和 generation。Heartbeat 超时、cloud 重启、
fingerprint 改变或 runtime handle 离开 `READY` 都会立即撤销 readiness、拒绝新请求，并使
in-flight request 返回结构化 unavailable 错误。不属于当前 session ID 和 generation 的
response 会被丢弃，连接恢复后必须重新握手。

重新握手不是 stateful runtime 的完整恢复条件。替换已有 distributed session 时，Cloud 会先停止旧
distributed session 的请求准入，并等待其 `ModelRuntimeHandle` 操作结束；stateless runtime 随后可直接创建
新 generation，stateful runtime 则必须先通过 handle 成功 reset，并确认 handle 回到 `READY`，才能发布新
generation。Reset 失败时 Cloud 会持续 fail-closed，后续 heartbeat 不能绕过该恢复屏障。若 stateful
runtime 声明 `resettable: false`，session rollover 无法通过重新握手恢复，必须重启或重建
Cloud runtime 后才能重新提供服务。

## 后端与支持矩阵

Canonical backend 只有以下五个：

| Backend | 用途 |
| --- | --- |
| `torch` | 原生 LeRobot，device 为 `cpu`、`cuda`、`mps` 或 `npu` |
| `ascend` | Ascend ACL 执行 OM artifact |
| `hisilicon` | 面向 `sd3403` target SoC 的 Hisilicon worker runtime |
| `rknn` | RKNNLite 执行 RKNN artifact |
| `hmm` | Houmo TCIM 执行 HMM 多模块 artifact |

原生 Torch policy 默认由 LeRobot factory 构造。仓库自研模型可以在 `torch_models` 注册
稳定的 `(model_type, backend, device)` provider；例如 PI0.5 Ascend310P 使用
`(pi05, torch, npu)`。通用 runtime 不按模型名称维护平台分支，provider 自己负责模型
配置、平台校验和运行时准备。
Provider 可选的 `preprocess_inputs(inputs, config=...)` 在 LeRobot processor 之前执行
模型自己的 CPU 输入优化，仍计入 pipeline 的 preprocess 耗时；通用 session 不包含模型图像规则。
可选的 `configure_processors(preprocessor=..., postprocessor=..., config=...)` 在加载阶段
配置模型特定的 LeRobot steps，例如在上传之前检查语言容量、在 CPU 上处理小动作张量。
Deployment 的 `extra_parameters` 是模型拥有的通用 JSON 参数。原生 Torch session 将其副本
传给可选 `resolve_load_options(config=..., extra_parameters=...)`，返回值用于本次模型构造。
Session 不解释模型字段；没有对应 provider consumer 时拒绝非空参数，且不允许返回值覆盖
严格权重加载/本地加载约束。参数参与部署与性能配置身份，不要求额外参数文件或内容校验值。

Ascend compiled deployment 可通过 manifest `device_links` 把 producer output buffer 直接绑定到 consumer input。
Ascend native runtime 按 `execution` 顺序调度这些 role，只把公开输出以及未声明 device link 的 host-routed 中间
tensor 复制回主机；`AscendOmModelSession` resource 拥有 ACL lease、模型和 device buffer，
`ModelRuntimeHandle` 负责准入、公开生命周期、健康、取消和关闭 drain。

所有本地 policy family 都通过
`InferencePipeline -> ModelRuntimeHandle -> RuntimeAssembly.runtime_executor` 执行。编译 PI0.5 与
SmolVLA 的 family 循环属于共享 executor stage；各 `ModelSession` resource 只负责 vendor artifact 加载与
模型或 role 执行。以下矩阵在启动时
强制校验，不在表中的组合会被拒绝：

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | Runtime | Runtime | Runtime | Runtime | 不支持 |
| Diffusion Policy | Runtime | 不支持 | 不支持 | 不支持 | 不支持 |
| PI0.5 | Runtime | Runtime | 不支持 | 不支持 | Runtime |
| SmolVLA | Runtime | 不支持 | 不支持 | Runtime | Runtime |

`Runtime` 表示通过统一 `ModelRequest/ExecutionContext` 执行，由 handle 负责生命周期、准入、健康、取消和
恢复，由 session resource 负责 vendor 模型资源。
感知 family 的 registry 支持矩阵如下：

| Perception family | `torch` | `ascend` |
| --- | --- | --- |
| RAM++ | 支持 | 支持 |
| SAM2 | 支持（automatic） | 支持（prompt） |
| SigLIP2 | 支持 | 支持 |
| Grounding DINO | 支持（combined） | 支持（raw） |
| GraspGen | 支持（CUDA） | 支持 |
| Dummy Echo | 支持 | 不支持 |

Registry 还要求每个 family/backend 声明 `ConformanceEvidence`；仅列出 family 而没有一致的验证证据
同样会 fail-closed。

### PI0.5 Ascend 行为

优化后的 PI0.5 VLM 在模型内部把多相机图像合并为一个临时 vision batch，并在 handoff 前恢复
camera-major prefix。该优化不改变外部 VLM ABI：runtime bindings、逐相机 observation semantic、
raw image shape 和 ROS camera topic 契约保持不变。

NPU 导出默认对 Gemma text MLP 使用精度保持的 `NPUGeglu`。显式参数
`--fast-gelu-scope vision|vlm-text|ae|all` 可将近似 `NPUFastGelu` 限制到指定模型区域；旧参数
`--fast-gelu` 等价于全局 `all`。近似路径可能降低动作精度，必须针对既有 baseline 验证。

新 Action Expert OM 的 runtime output 名为 `velocity` 或 `v_t`，Manifest 仍将其映射为策略
semantic `action`。策略 runtime assembler 从选中 deployment 的 `denoising_schedule` artifact 读取严格
递减的 timesteps，共享 `IterativeStage` 在 host 侧执行
`x_next = x_t + (next_t - t) * velocity` Euler integration，
最终才返回 action。Exporter 未提供 `--schedule-file` 时，根据 `config.num_inference_steps` 打包
uniform schedule；提供时只接受严格的 `pi05-denoising-schedule-v1` JSON。

`denoising_schedule` 是 versioned、non-execution artifact：不加入 `execution` 或 `bindings`，但
其 artifact path 和 deployment revision 属于 deployment identity，schedule 变化会改变 selected
deployment fingerprint。生产 Runtime 不扫描 bundle 根目录的 `schedule.json`，也不接受 schedule
override。`loss_compare`/tuner 通过隔离的 diagnostic backend factory 注入临时 schedule；
`curvature_log_path` 仅记录诊断数据，最终 schedule 必须安装回 Manifest。

兼容性由 Action Expert runtime output 明确决定。已有 output 为 `action` 且没有 schedule artifact
的 legacy PI0.5 deployment 保留旧的逐步 action-output 行为；velocity deployment 缺少 schedule
则拒绝加载，不会猜测默认值。`hardware_mock` 仍只验证 raw image/topic、joint 和 action 契约，
不需要 PI0.5 或 schedule 专用修改。

原生 Torch Diffusion Policy 按模型 `n_obs_steps` 在 contract 控制频率上采样历史观测，
`predict_action_chunk()` 返回的 nominal chunk 长度取自 `n_action_steps`。历史不足时使用每个
观测流的首帧进行左侧填充；不同频率的传感器仍按相同时间栅格应用各自的 `hold`、`asof`
或 `drop` 对齐策略。

可选 SDK 延迟导入。仅导入 `inference_service` 不要求 ACL、RKNNLite、TCIM、torch NPU 或
Hisilicon worker dependency；只有选择相应 deployment 时才检查依赖。

## 生命周期、健康与能力

`ModelRuntimeHandle` 的公开状态包括 `CREATED`、`LOADING`、`READY`、`RESET_REQUIRED`、
`RESETTING`、`FAILED`、`CLOSING` 和 `CLOSED`。只有 `READY` 接受请求。Handle 在 reset/close 前停止
准入并等待 active execution，统一记录健康与恢复要求；`close()` 幂等，并按 `RuntimeAssembly` 的反向
所有权顺序释放已经加载的 component。

`ModelSession` 是 handle-owned resource，而不是公开生命周期 owner。它持有并释放 context、vendor model
handle、device buffer、worker、tokenizer 或其他模型专属资产，并实现
`execute(ModelRequest, ExecutionContext)` 或 role execution。策略 `InferencePipeline` 将 handle 状态投影为
pipeline diagnostics；distributed `HANDSHAKING` 属于节点协议状态，不是 model resource 生命周期。

Session/deployment capabilities 供 handle 决定：

- 是否 stateful、resettable、thread-safe
- 单实例最大 in-flight 请求数
- 是否支持多个实例
- 是否存在共享 resource domain 及其总限制
- 是否支持 attention 和 cancellation

默认采用保守串行限制。只有 conformance tests 证明重叠调用、输出隔离、故障隔离和确定性
清理后，runtime 才能声明更高并发。不同 handle 有独立准入状态，但共享 accelerator
resource domain 时仍可能被后端串行化。

## Manifest Identity

启动顺序是：严格 JSON/schema 校验，deployment 选择，UUID/revision 与轻量 bundle digest 校验，
路径安全和普通文件校验，LeRobot metadata，binding compatibility，最后构造 session resource、
`RuntimeAssembly` 和 `ModelRuntimeHandle`。
常规 Runtime 不读取 OM、RKNN、HMM 或 safetensors 计算内容 SHA-256；scheduled 本地 compiled deployment
由 `robot_config` 在 launch 构造前额外验证 manifest 中声明的 artifact SHA-256。

`bundle.digest` 算法：

1. 将每个 `bundle.files` path 规范化、去重并排序。
2. 加入 bundle UUID、revision、name 和结构格式域。
3. 使用 canonical UTF-8 JSON 序列化这个小型声明。
4. 对声明 bytes 计算 SHA-256，不读取声明指向的文件。

UUID/revision/digest/fingerprint 用于版本标识和分布式一致性，不提供文件防篡改。正式 artifact 更新
必须经过 packager 并发布新 revision；生产防篡改应使用只读镜像、签名或 verity。

Selected deployment fingerprint 对以下 canonical object 计算 SHA-256：

```json
{
  "format": "ibrobot.deployment-structure-v3",
  "schema_version": 3,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

路径不能是绝对路径、不能包含 parent traversal、不能通过 symlink 逃逸 bundle root，规范化
后也不能重复。

### Identity 错误处理

遇到以下错误时不要手工改 identity：

- `Bundle digest mismatch`
- unsupported schema v1（必须重新导出或重新打包）
- missing or unexpected LeRobot semantic files
- execution role 缺少 artifact 或 bindings
- runtime ABI 与 LeRobot feature shape 不匹配

应重新运行拥有该 artifact 的 exporter 或 packaging workflow。Exporter 负责复制 artifact、
读取 compiler/runtime ABI、生成 bindings、更新 UUID/revision 和轻量结构摘要，并通过生产 loader
重新验证 manifest。Schema v1 和旧版 artifact 不受支持，必须使用当前 exporter 或 packager
重新生成完整 schema-v3 whole-graph bundle。

## Exporter 入口

通用 compiled artifact packaging：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment rk3588 \
    --backend rknn \
    --target-soc rk3588 \
    --target-runtime rknn-lite2 \
    --spec /path/to/compiler-package-spec.json
```

PI0.5 / SmolVLA HMM 使用：

```bash
ros2 run model_utils package-hmm-deployment --help
```

ACT Ascend、ACT RKNN、Hisilicon 和 policy-specific multi-module exporters 位于
`model_utils`。所有工具最终必须调用 shared `inference_manifest` writer；不要手工维护
artifact path、binding 或 digest。

## 验证

从工作区根目录加载项目环境，并禁用外部 pytest plugins。包测试统一收集在 `test/`：

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q src/inference_service
```

只对本次修改的 Python 文件执行 Ruff。项目或 ROS 命令前必须先加载 `.shrc_local`。
## Typed Model Services

`model_service_node` is the family-neutral host for strongly typed model services. Each process loads one schema-v3
bundle、一个命名 deployment 和一个 `ModelServicePlugin`；plugin 负责领域请求/响应映射，构造
`ModelSession`，将其放入 `RuntimeAssembly`，并把 assembly 的所有权转交给 `ModelRuntimeHandle`。
Handle 负责准入、公开生命周期、健康、取消、恢复和清理顺序；session resource 负责 accelerator SDK 和
模型资源。

`robot_config` launches the same host for RAM++, SigLIP2, GraspGen, and ZipVoice TTS. Model packages provide plugins;
they must not implement parallel ROS nodes or duplicate `ModelRuntimeInfo` projection.

When `required=false`, an initialization failure leaves the typed endpoint online so callers receive a not-ready
response and diagnostics. The host does not retry plugin initialization on later requests. After repairing a missing
bundle, dependency, or device, restart the node to recover. Request validation and other request-scoped failures do
not change runtime health; model resource failures are projected through handle health.

### 跨主机部署模型服务（`with_model_services`）

`with_model_services` 是 launch 参数，默认 `true`。它声明于 `perception_models` launch 条目，
控制本机是否启动配置的 `perception_services` 服务节点。

参数为 `false` 时的行为：

- launch 计划不包含 `perception_models` 条目，本机不启动模型服务节点。
- YAML 中的 service 配置不变，本机的调用方照常访问配置的 endpoint。
- 配置加载和 live binding 按「元数据校验」处理 enabled service，见下文。

元数据校验的规则：

- `robot_config` 照常校验 service 的身份、endpoint、deployment 名称和语义契约。
- 校验入口复用 `load_inference_manifest_metadata()`，与分布式 edge 端相同。
- bundle 文件和 deployment artifacts 不在本机检查；托管主机保持完整校验。
- 契约消费者同样按元数据校验。semantic mapping 的参数构造、`load_robot_config()`
  的契约读取（scheduled dispatcher、recording node 等）只消费 endpoint 与身份，
  不加载权重。验证配置的工具（`validate_config.py`）显式要求本机 artifacts。

消费端主机的前置条件：

- 必须持有每个 enabled service 的 bundle 目录和其中的 `inference_manifest.json`。
- 权重和 deployment artifacts 只需要存在于托管模型的主机。
- 两台主机使用同一份 robot YAML；`bundle_path` 在各自主机上解析到实际路径。

示例：

```bash
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/robot.yaml \
    with_model_services:=false
```

### 关闭失败与恢复限制

流关闭失败时，Handle 保留流和配额。`retry_pending_stream_closes()` 可重试待关闭流；
运行时 `reset()` 先重试这些关闭。ASR 在再次开流前执行相同检查。
reset 释放隔离额度前必须完成原生恢复，不能以空 reset 作为完成证明。

RKNN/HMM 原生释放抛出无完成证据的异常时，保留所有权并报告 `close_pending`。
后续关闭不会再次调用同一原生释放函数，也不宣称资源已经释放；这类故障需要进程重启。
只有显式 `NOT_STARTED` 证据允许重试原生释放。成功关闭仍幂等。

字符串流 ID 的关闭幂等历史仅保留最近 128 条；淘汰后返回 `stream_not_found`。
持有原始 `StreamHandle` 对象时，已关闭对象的幂等性不受历史淘汰影响。
该限制也适用于 SpeechDirection 的字符串流 ID。
