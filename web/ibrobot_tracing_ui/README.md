# IB-Robot Trace

面向 `ibrobot_tracing` 离线结果的 Vue 3 工作台。默认从“找到慢请求”开始，使用 **概览、请求分析、实验对比** 三个主入口；拓扑、原始 Event/Span/Flow 和数据质量位于“更多视图”。界面只提供中文，使用真实 trace 文件数据。

## 推荐使用路径

1. 选择 CTF、日志或 `.ibtrace` 格式的 trace 文件并“打开分析”。概览显示核心指标、有效样本数和最慢请求入口。
2. “查看最慢请求”，或在顶部粘贴完整请求 ID。精确查找覆盖整份分析，不受当前列表页限制。
3. 在同一请求上下文切换耗时拆分、时间线和调用层级；点击记录查看摘要，按需展开原始字段。
4. “返回请求列表”保留排序和滚动位置；上一/下一请求按列表排序。总体视图始终明确标记“整份 trace 文件”。
5. 需要时从“更多视图”打开拓扑、原始记录或数据质量。原始表每页100条，支持服务端完整名称/边ID查询。
6. 实验对比选择另一份 trace 文件作为基线，点击“执行比较”；高级统计量和阈值按需展开。
7. 点击左上角 **IB-Robot Trace** 图标或名称返回首页；浏览器后退可回到先前的分析视图。

计时边界、跨主机校准与缺失信息见“trace 文件信息与计时边界”和记录详情。未提供的主机不会被推测成Web服务器地址；记录状态也不是业务成功率。

## 本地运行

在 Ubuntu 上构建或开发 Web UI 需要 Node.js 18 或更高版本。仓库不托管 `dist`，每个部署环境必须
先从 Vue/TypeScript 源码生成生产资源。Web 不属于默认 setup/build，服务端位于
`tools/ibrobot_tracing_web`；Python 依赖准备和启动方式见
[服务端 README](../../tools/ibrobot_tracing_web/README.md)。以下 `npm ci` 是用户显式准备步骤，
构建脚本不会自动安装依赖。

```bash
# 在仓库根目录
source .shrc_local
npm --prefix web/ibrobot_tracing_ui ci
npm --prefix web/ibrobot_tracing_ui run dev
```

Vite 默认监听 `http://localhost:4174`，并将 `/api` 代理到 `http://127.0.0.1:8000`。可在启动前通过 `IBROBOT_TRACING_API` 修改后端地址：

```bash
source .shrc_local && IBROBOT_TRACING_API=http://127.0.0.1:9000 npm --prefix web/ibrobot_tracing_ui run dev
```

验证命令：

```bash
source .shrc_local
npm --prefix web/ibrobot_tracing_ui test
npm --prefix web/ibrobot_tracing_ui run build
```

`dist/`、`node_modules/` 和覆盖率目录均保持忽略，不应提交构建产物。完整生产构建优先从仓库根目录
执行 `source .shrc_local && bash scripts/build_tracing_web.sh`，它会先检查已有依赖，再运行
`npm run build`，并通过统一 `scripts/build.sh` 显式扫描 `src` 和 `tools/ibrobot_tracing_web`，
仅构建 `ibrobot_tracing`、`ibrobot_tracing_web` 两个 ROS 包。

构建后，从仓库根目录运行服务端测试使用新路径（需已有 pytest 和 Web 测试依赖）：

```bash
source .shrc_local
python3 -m pytest tools/ibrobot_tracing_web/test
```

## API 契约

前端使用 `tools/ibrobot_tracing_web` 提供的以下 `/api/v1` 端点。分页响应使用 `{ "items": [...], "total": 0, "offset": 0, "limit": 100, "next_offset": null }`；来源和警告列表使用各自的轻量响应。

| 方法 | 端点 | 用途 |
|---|---|---|
| `GET` | `/api/v1/capabilities` | 发现包括 `baseline_compare` 在内的服务能力和限制 |
| `GET` | `/api/v1/sources` | 发现可读取的 CTF、日志或合并 `.ibtrace` 追踪源 |
| `POST` | `/api/v1/sources/refresh` | 重新扫描受限来源目录 |
| `POST` | `/api/v1/analysis-jobs` | 以 `{ source_id }` 创建加载任务 |
| `GET` | `/api/v1/analysis-jobs/{job_id}` | 轮询 `queued/running/completed/failed/cancelled` 状态 |
| `GET` | `/api/v1/analyses/{id}/summary` | 元数据、阶段矩阵、自定义 Span/事件汇总与覆盖率 |
| `GET` | `/api/v1/analyses/{id}/requests` | 服务端排序/分页与完整 `request_id` 精确查找 |
| `GET` | `/api/v1/analyses/{id}/distribution` | 请求延迟直方图、桶内请求和高延迟请求；指标可省略并由后端选择 |
| `GET` | `/api/v1/analyses/{id}/tracepoints` | 分页埋点定义目录；用于检查器说明匹配 |
| `GET` | `/api/v1/analyses/{id}/graph` | 带处理/流转耗时的拓扑投影；`view` 接受 `nodes`、`components`、`tracepoints` |
| `GET` | `/api/v1/analyses/{id}/timeline` | Event、Span、Flow 的泳道投影 |
| `GET` | `/api/v1/analyses/{id}/call-tree` | 已计算父子、自身耗时和异常父链的调用树 |
| `GET` | `/api/v1/analyses/{id}/span-profile` | 单请求时间几何或跨请求聚合的 Span 墙钟剖析；`max_nodes` 默认 10000 |
| `GET` | `/api/v1/analyses/{id}/critical-path` | 单请求不重叠墙钟归因分区；`request_id` 必填，`max_segments` 默认 1000 |
| `GET` | `/api/v1/analyses/{id}/events` | 原始事件；请求/组件范围、`event_name`、`offset/limit` |
| `GET` | `/api/v1/analyses/{id}/spans` | 配对 Span；请求/组件范围、`name`、`offset/limit` |
| `GET` | `/api/v1/analyses/{id}/flows` | 配对 Flow；请求/组件范围、`edge_id`、`offset/limit` |
| `GET` | `/api/v1/analyses/{id}/warnings` | 分析器警告 |
| `POST` | `/api/v1/comparison-jobs` | 以另一份 trace 的 source ID/version 和当前候选 analysis ID 创建比较任务 |
| `GET` | `/api/v1/comparison-jobs/{job_id}` | 轮询比较任务，不接收 trace 路径 |
| `GET` | `/api/v1/comparisons/{id}` | 读取指标差值、共享分桶和 Span 墙钟差异结果 |
| `DELETE` | `/api/v1/comparisons/{id}` | 删除临时比较结果 |

加载任务响应示例：

```json
{
  "id": "job-01",
  "source_id": "source-01",
  "source_version": "fingerprint-01",
  "status": "running",
  "created_at": "2026-07-15T10:00:00Z",
  "started_at": "2026-07-15T10:00:01Z",
  "finished_at": null,
  "analysis_id": null,
  "error": null,
  "cancel_requested": false,
  "queue_position": null,
  "deduplicated": false
}
```

任务完成时 `status` 为 `completed` 且 `analysis_id` 存在。服务端不提供虚假的阶段百分比，界面只区分排队和执行状态。比较请求固定使用当前已载入的 `candidate_analysis_id`，浏览器没有文件路径输入，也不会在请求或响应中传递基线源路径。

图投影中的节点和边通过 `metric_value_ms` 返回当前指标。选择单个请求时该值表示实际耗时，否则表示 `metric` 指定的聚合值。`tracepoints` 视图中的 `contains` 边是无向结构关系，界面以无箭头的灰色虚线连接组件和埋点；其余数据边保留标签和流向动画。绝对纳秒字段可由后端编码为十进制字符串，时间线使用安全的相对偏移数值。

## 交互

- 页首“使用帮助”可搜索操作、术语与恢复方法，并优先显示当前视图相关说明。帮助搜索与分析过滤互相独立；关闭或Esc返回触发器，保留调查上下文。
- “更多视图”按原始记录、分析与布局分为两组，每组三个入口。耗时摘要优先显示四项指标，计时边界和Span/Flow归因组成可就地展开。
- 拓扑默认折叠，展开偏好保存在本机浏览器；展开后可拖动中间分隔条或用上下方向键调整高度。拓扑自身的搜索、层级和统计控件只影响拓扑。
- `/` 聚焦当前视图过滤框；没有局部过滤框时聚焦完整请求ID查找。`Esc` 优先关闭浮层并返回触发器，其次关闭记录详情，最后清除过滤内容。点击浮层外部也可关闭，过滤条件保留。
- `Alt+1` 到 `Alt+9` 切换下方前九个可用分析视图。服务支持比较时，延迟分布为 `Alt+3`、基线比较为 `Alt+4`、Span 分析为 `Alt+6`、关键路径为 `Alt+7`；服务未声明 `baseline_compare` 时不显示比较标签，后续快捷键自动前移。
- 表格聚焦后可用 `J/K` 或上下方向键移动，回车检查当前行；请求表打开耗时拆分。行内按钮使用原生Enter/Space；调用树展开按钮与拓扑节点也可用键盘操作。
- 标签栏聚焦后可用左右方向键连续切换，焦点保持在选中的页签。
- 检查记录后，所有尺寸都将焦点移交详情标题；关闭后恢复触发点。归因图用一个Tab入口，左右键/Home/End移动，也可使用“查看分段列表”中的原生按钮。
- 窄屏将请求查找/trace 文件信息及局部过滤按需展开；`/`会先展开对应工具再聚焦输入。局部过滤输入、清除和完成在同一行，Enter或“完成”收起工具并保留过滤值。Span图保留至少240px滚动区域，其上方辅助内容可随面板滚动。
- 窄屏或低高度窗口使用紧凑请求栏，保留返回、请求标识及相邻请求；次级导航可滚出阅读区。无匹配提示在可见宽度中显示，宽表/时间线仍可局部横向滚动。没有筛选工具的视图不显示筛选开关。
- 概览的Span汇总行支持点击/回车查看分组统计详情，显示调用次数、状态与分位数，不将聚合组当作单个请求。
- 动画暂停按钮和系统“减少动态效果”偏好均会停止边上的流向示踪。
- 当前视图、请求、组件、拓扑指标/层级、局部过滤、分布指标及 `mode=call-tree|request|aggregate` 保存在URL中。任务导航保留浏览器历史，过滤微调替换当前记录。`source/source_version`用于分析缓存失效后的原 trace 文件恢复；来源已变化时明确要求重新选择，避免混用旧请求。请求列表的分页/排序、原始记录的 `record_offset/record_name` 同样可分享和刷新恢复。旧 `tab=call-tree` 链接继续可用。
- 基线比较保存 `baseline_source`、`statistic`、`relative_threshold`、`absolute_threshold_ms` 和 `comparison_metric`；无效或负数阈值恢复为10%与1ms。首次打开分布视图可由后端选择指标；默认省略 `mode=request`。
- 延迟分布柱可悬停、单击或用 Tab 聚焦，并用回车/空格选择。选中桶和高延迟表中的请求可直接打开精确请求范围的时间线、Span 分析或关键路径；重复打开当前请求不会清除请求范围。
- 打开分析时会一次性分页读取完整埋点目录；检查器按记录身份显示“埋点说明”，没有说明时不显示该区域。
- Span 分析在同一入口提供调用树、请求 Span 区间和聚合 Span 剖析，并固定使用根在上的冰柱方向。冰柱图支持 `+`/`-` 缩放、水平滚动或拖动、`F` 适合宽度和 `0` 重置；聚合图双击节点聚焦子树，并通过面包屑返回父级或全部根。
- 基线比较必须点击“执行比较”，参数变化不会隐式启动昂贵任务。只有选定统计值的相对增幅和绝对增幅都严格超过阈值才标记为差值超阈值；指标表始终同时显示文本状态与颜色。结论旁显示实际样本数并说明实验条件尚未核对，不能把历史 trace 文件的数值差异当成改动导致性能退化的证明。选择指标后再次执行，服务端会用同一组边界生成基线/候选叠加直方图。
- 完成的比较以 `comparison_result` 记录在URL中；刷新只读载入已有结果，不自动重算。结果参数与待执行控件保持分离，参数改变仍提示旧结果；过期或候选不匹配时明确提示并保留恢复入口。
- 结果统计量、双阈值和来源在结论旁完整换行显示。删除缓存结果需就地确认；取消返回触发器，删除失败保留原结果/链接，成功后告知结果并聚焦重新比较入口。原始 trace 文件不受该操作影响。
- 比较直方图中的区间是可聚焦读数，焦点显示基线/候选样本数；请求延迟分布中的柱则是可选择按钮。两者按各自行为提供语义。
- 解析 trace 文件和比较期间都可“停止等待”。这会中止当前页面请求/轮询，后台任务可能继续；迟到响应不会重新打开已退出的分析。trace 文件选择和比较参数保留，可显式重试。
- Instrumented Span Wall Diff 按完整 `component/name/origin` 父链接重建路径，以 `max(baseline, candidate)` 包含墙钟权重分区，并固定使用冰柱方向。双击或 `Shift+Enter` 聚焦子树；点击查看详情。该图包含等待、I/O与异步暂停，不是CPU profile。

## 实现边界

- 本目录只包含前端；对应服务端位于 `tools/ibrobot_tracing_web`。服务未启动时会显示明确的载入错误，不会回退到虚构数据。
- 基线比较直接复用受限 trace source catalog；当前候选 source 会从可选基线中排除。目录中不足两份 trace 时只显示提示，不提供浏览器路径输入。
- 比较轮询绑定分析 ID 和本次请求 generation。再次比较或切换分析会中止旧请求并清空 job、result、loading 与 error，迟到响应不会覆盖新状态。
- 请求、Event、Span和Flow使用真实分页，每页100条，明确显示已返回范围与总数。请求列表排序与精确ID查找由服务端执行；原始记录提供完整名称/边ID查询。局部文字过滤只作用于当前页，界面明确标注。时间线和调用树使用选定请求的完整投影，不受原始表单页限制。
- 图有独立加载/失败状态，切换范围后不会把旧图标记为新请求的数值。相同key进行中的数据页加载会复用；迟到响应不会覆盖新范围。
- 拓扑布局资源或布局计算失败时，显示明确的布局错误与页面恢复入口，不误报“没有拓扑”。页面刷新保留URL中的任务范围，其余视图可继续使用。
- Flow详情直接使用后端提供的发送/接收端来源和校准状态；未提供的信息不补猜。未同步、未校准和时钟不连续等记录状态使用中文说明，原始状态码仍保存在原始字段中。
- trace 文件目录和原始证据读取失败时显示中文上下文、保留状态与明确重试动作；原始错误可展开。数据质量页只解释已知分析器消息格式，并提供准确关联请求入口；未知格式保留原文，不猜测含义。
- 延迟分布按标签首次打开时惰性读取；切换指标会中止旧请求，并按分析和指标缓存结果。桶内请求标识受后端 `bucket_request_limit` 限制，截断状态会在选中桶上明确显示。
- 埋点定义目录会跟随分析切换清空并重新完整加载，不随标签、请求或组件过滤重复请求。
- Span 冰柱图固定采用 **Instrumented Span Wall Time** 语义：数值包含等待、I/O 和异步暂停，不能解释为 CPU 使用率。单请求图按起始偏移和持续时间绘制，聚合图按包含墙钟权重分区；深色底条表示自身或未被子 Span 覆盖的墙钟权重。
- 单请求模式未选择请求时只显示明确提示，不调用要求 `request_id` 的 API。请求、组件或模式变化会中止旧请求；服务端截断和结构诊断会在图上方汇总。
- 关键路径同样要求明确请求，并按完整请求边界比例展示互不重叠的 Flow wait、最深 active Span 和未归因 Gap。它表示包含等待、I/O 与异步暂停的 Instrumented Wall Time，不是 CPU 时间或跨主机调度 DAG；截断时图表单独标出精确的省略尾部。
- 组件筛选保留完整请求时间边界，只对选中组件及关联Flow归因；当前范围的Gap可能包含其他组件已有观测，不能据此认定整份请求缺少观测或发生等待。可就地恢复全部组件归因。
- 耗时拆分详情显示“本段归因时长”和“本段起终点”，不冒充原始记录。可按精确 `source_id` 跨分页打开原始Span/Flow，再返回分段；原始观测时长和状态由真实记录提供。
- 时间线已经展示 Event 点、Span 区间和 Flow 区间。调用树已展示总耗时，并在悬停时给出自身耗时。
- 时间线标尺和条带共用后端 `duration_ns`，文字过滤只隐藏记录；“显示记录间隔”是相邻可见记录起点之差。窄屏横向滚动时标尺与条带同步。
- ELK 布局引擎在首次显示拓扑时按需加载，生产块体积较大但不会进入初始应用包。

## 维护与验收

- 设计约束见 [DESIGN.md](DESIGN.md)，产品目标见 [PRODUCT.md](PRODUCT.md)。
- `tests/keyboard.test.ts` 覆盖原生Enter传播和未加载诊断语义；`profiler.test.ts`覆盖旧响应、图失败、查找、分页及无请求时间线前置。
- `tests/completionContracts.test.ts` 覆盖布局模块失败与Flow来源/缺失耗时的呈现，避免将资源失败或未同步数据解释为正常数值。
- `tests/roundTwoUx.test.ts` 覆盖过滤后的时间域、组件范围说明、汇总详情和浮层取消。颜色角色、组件色板与图表图例统一定义于 `src/theme.css`，支持配置浅色主题；尚未提供深色模式。
- `tests/usabilityContracts.test.ts` 覆盖帮助搜索不改变调查范围、窄屏完成过滤保留条件，以及停止等待后的迟到分析响应。
- 控件边界使用文字/图形适配的语义令牌；系统高对比度下SVG刻度显式使用系统颜色。图例在窄屏保持可见，关键图表刻度及操作说明使用12px层级。投影覆盖解释可用点击、键盘和触控展开。
- UI变更需验证：起始→概览→慢请求→原始证据→返回；精确ID查找；分页/刷新；图失败和分析失效恢复；真实双源对比；320–1440布局及键盘操作。
- 全量统计、单请求区间和跨请求聚合有不同分母，必须保持显式范围。父子Span/并发区间及各段P95不直接相加；缺值不能显示为零；主机信息缺失时不补猜。
