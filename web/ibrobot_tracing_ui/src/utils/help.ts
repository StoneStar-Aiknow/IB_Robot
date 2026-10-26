import type { DetailTab } from "../api/types";

export interface HelpTopic {
  id: string;
  title: string;
  keywords: string;
  views: DetailTab[];
  steps: string[];
  guide?: "capture" | "instrument";
}

export const HELP_TOPICS: HelpTopic[] = [
  { id: "capture", title: "如何抓取 trace", keywords: "抓取 录制 记录 LTTng CTF 开始 停止 trace-root 文件目录 文件上传", views: [], guide: "capture", steps: [
    "在运行业务的机器上记录，停止后将完整 trace 放入 Web 服务配置的目录，再刷新首页列表。已有埋点无需修改代码。",
  ] },
  { id: "instrument", title: "添加自定义埋点", keywords: "Python span event trace_context request_id IB_TRACE_ENABLED 自定义 埋点", views: [], guide: "instrument", steps: [
    "用 Span 测量一段耗时，用 Event 记录瞬间状态；通过同一个 request_id 将记录关联到业务请求。指南提供可运行示例。",
  ] },
  { id: "start", title: "找到一个值得调查的请求", keywords: "开始 trace 文件 慢请求 查找 ID 排序 分页", views: ["summary", "requests"], steps: [
    "打开 trace 文件后，选“查看最慢请求”；也可在顶部查找框粘贴完整请求 ID，查找覆盖整份 trace 文件。",
    "请求列表按耗时排序并真实分页。列表里的文字过滤只作用于当前页，不是完整 ID 查找。",
    "用“返回请求列表”继续原来的排序和位置；上一/下一请求也按列表排序。",
  ] },
  { id: "timing", title: "读懂 Span、Flow 和未归因 Gap", keywords: "CPU 墙钟 耗时 覆盖率 组件 等待 时间线", views: ["critical-path", "timeline", "span-profile"], steps: [
    "Span 是一次有起止的操作记录；Flow 是发送与接收的关联区间，可能包含调度、编解码和传输，不等于纯网络耗时。",
    "耗时拆分将请求墙钟分成互不重叠的归因片段。它包含等待、I/O 和异步暂停，不是 CPU 用时；原始 Span 之间仍可能重叠。",
    "Gap 表示当前范围没有归因。筛选组件后，其他组件已有的观测也可能成为 Gap；用“查看全部组件归因”核对。",
  ] },
  { id: "evidence", title: "从归因片段打开原始证据", keywords: "source_id 原始记录 字段 duration 状态 重复 occurrence", views: ["critical-path", "spans", "flows", "events"], steps: [
    "点击分段或分段列表中的名称，查看“本段归因时长”和本段起终点。",
    "选择“打开原始 Span / Flow”，按精确记录 ID 查看原始时长、状态和字段；它可能比本段更长。再用“返回归因分段”回到原处。",
    "同名调用不是同一条记录。聚合汇总包含符合条件的重复调用；完整/错误/未完成是记录状态，不是业务成功率。缺失值“—”也不是零。",
  ] },
  { id: "statistics", title: "理解 P50、P95 和样本数", keywords: "分位数 延迟分布 均值 最小 最大 投影 统计 预处理 后处理", views: ["summary", "distribution", "comparison"], steps: [
    "P50 是中位耗时；P95描述高位耗时，约95%的样本不超过它。均值是样本平均值，三者不能互相替代。",
    "先看有效样本数和计时边界。样本少时分位数可能不稳定；各阶段可能重叠，各阶段P95不能直接相加。",
    "列表日志汇总与原始记录边界可能有精度差异。需要引用精确边界时，打开请求及原始记录核对。",
  ] },
  { id: "compare", title: "比较两份 trace 文件，并重新打开同一结果", keywords: "实验 基线 候选 阈值 AND 回归 保存 分享 过期", views: ["comparison"], steps: [
    "选择另一份 trace 文件作为基线，当前分析是候选。点击“执行比较”；相对和绝对差值同时严格超过阈值才标记超阈值。",
    "先核对样本数量、硬件、模型、负载与埋点条件。工具只计算观测差值，不证明改动导致性能退化。",
    "结果完成后复制当前地址即可重新打开。修改控件只改变待执行参数；旧结果保留自己的参数。刷新只读取已有结果，过期后需显式重新比较。",
  ] },
  { id: "recovery", title: "无数据、读取失败或结果过期怎么办", keywords: "错误 重试 断网 找不到 空白 缺失 停止等待 取消", views: ["warnings"], steps: [
    "先区分：过滤无匹配可清除过滤；无观测需核对请求、组件范围及数据质量；读取失败可重试，查看具体错误详情。",
    "原始记录找不到或读取失败时，归因详情仍保留。重试原始记录，或重新打开 trace 文件；不要用另一条同名记录代替。",
    "分析过期可在原 trace 文件版本一致时重新分析；版本已变则重新选择 trace 文件。停止等待只退出当前页面等待，后台任务可能继续。",
  ] },
  { id: "keyboard", title: "键盘与小屏操作", keywords: "快捷键 Esc Enter Tab J K 方向键 筛选 缩放", views: ["timeline", "span-profile", "requests"], steps: [
    "/ 聚焦当前过滤，必要时先展开工具；输入完成按Enter或“完成”收起工具继续阅读，过滤条件保留。",
    "Esc先关闭浮层，再关闭记录详情，最后清除过滤。页签用左右方向键；表格用上下/J/K移动、Enter检查；行内按钮支持Enter/Space。",
    "归因图用一个Tab入口，左右/Home/End选择片段。检查后焦点进入详情，关闭回到触发点。宽图表可局部横向滚动，小片段也能从分段列表打开。",
  ] },
];

export function searchHelp(query: string, view: DetailTab): HelpTopic[] {
  const terms = query.trim().toLocaleLowerCase("zh-CN").split(/\s+/).filter(Boolean);
  return HELP_TOPICS.filter((topic) => terms.every((term) =>
    `${topic.title} ${topic.keywords} ${topic.steps.join(" ")}`.toLocaleLowerCase("zh-CN").includes(term)))
    .sort((left, right) => Number(right.views.includes(view)) - Number(left.views.includes(view)));
}
