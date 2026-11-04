<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { RouterLink, useRoute } from "vue-router";
import { guideReturnPath } from "../utils/guide";
import { guideExamples as examples } from "../utils/guideExamples";
import GuideLink from "./GuideLink.vue";
import GuideCode from "./GuideCode.vue";

const route = useRoute();
const capture = computed(() => route.params.topic === "capture");
const back = computed(() => guideReturnPath(route.query.from));
const title = ref<HTMLElement | null>(null);
watch(() => route.fullPath, async () => {
  await nextTick();
  if (capture.value && route.hash === "#manual") {
    const section = document.getElementById("manual") as HTMLDetailsElement | null;
    if (section) {
      section.open = true;
      const heading = section.querySelector("summary");
      heading?.focus();
      heading?.scrollIntoView({ block: "start" });
      return;
    }
  }
  title.value?.focus();
  title.value?.closest(".trace-guide")?.scrollTo(0, 0);
}, { immediate: true });
</script>

<template>
  <article class="trace-guide" aria-labelledby="guide-heading">
    <div class="guide-body">
      <RouterLink class="guide-back" :to="back">{{ back === '/analysis' ? '返回首页' : '返回原分析' }}</RouterLink>
      <h1 id="guide-heading" ref="title" tabindex="-1">{{ capture ? '如何抓取 trace' : '添加自定义埋点' }}</h1>
      <nav class="guide-nav" aria-label="指南主题">
        <GuideLink topic="capture" :aria-current="capture ? 'page' : undefined">如何抓取 trace</GuideLink>
        <GuideLink topic="instrument" :aria-current="!capture ? 'page' : undefined">添加自定义埋点</GuideLink>
      </nav>
      <template v-if="capture">
        <p class="guide-intro">在业务机器上记录，停止后交给 WebUI 分析。已有埋点的流程可以直接抓取，无需先改代码。</p>
        <ol class="guide-steps">
          <li>
            <h2>准备记录环境</h2>
            <p>在<strong>运行业务的机器</strong>上打开终端，进入已初始化、已构建的 IB_Robot 根目录。每个新终端都先加载工作区环境：</p>
            <GuideCode label="检查记录环境" :code="examples.check" />
            <details><summary>缺少 LTTng 或 Python tracing 依赖？</summary><p>在已初始化的工作区补装 tracing 工具后，重新执行上面的检查。若缺少 ibrobot_tracing，请先通过项目构建流程构建该包，再重新加载 .shrc_local。</p><GuideCode label="补装 tracing 工具" :code="examples.setup" /></details>
          </li>
          <li>
            <h2>开始记录，再运行业务</h2>
            <p><strong>已有 robot.launch.py 启动命令：</strong>在原命令末尾附加下面的参数，保留原来的机器人、模型和仿真配置。my_trace 是本次记录名称，每次使用一个新名称。</p>
            <GuideCode label="附加到原 launch 命令的参数（不单独执行）" :code="examples.launch" />
            <p>Launch 会启用业务埋点并管理会话。正常退出 launch（Ctrl+C）后记录停止；从启动日志确认实际输出目录。</p>
            <details id="manual"><summary>独立 Python 程序，或需要手动控制记录</summary>
              <p>此方式与 launch 自动记录二选一。在业务机器的<strong>终端 1</strong>启动会话：</p>
              <GuideCode label="开始手动记录" :code="examples.start" />
              <p>在同一机器的<strong>终端 2</strong>加载环境，用 IB_TRACE_ENABLED=1 启动你的程序。下面运行的是<GuideLink topic="instrument">自定义埋点指南</GuideLink>中的 trace_demo.py：</p>
              <GuideCode label="运行埋点示例" :code="examples.demo" />
              <p>必须在创建 emitter 之前启用该变量。终端 1 的环境不会自动传给终端 2；对已经启动的进程设置变量也不会生效。</p>
              <p>程序结束后回到终端 1，使用开始时相同的会话名停止并释放会话。这个操作保留 trace 数据：</p>
              <GuideCode label="停止手动记录" :code="examples.stop" />
            </details>
          </li>
          <li>
            <h2>确认文件，再放到 Web 主机</h2>
            <p>上面的 my_trace 默认保存在<strong>记录用户</strong>的 <code>~/.ros/tracing/my_trace/</code>。CTF 是整个目录，包含 metadata 和数据流；复制时保留完整目录结构。Topology 文件是可选信息。</p>
            <p>WebUI 读取的是 <strong>Web 服务所在机器</strong>上、由 <code>--trace-root</code> 配置的目录。它不是浏览器电脑的下载目录，页面也不提供文件上传。</p>
            <p>如果业务和 Web 在同一台机器，确认输出位于该服务的 trace-root 内，且服务用户可读；不同机器则在停止记录后复制过去。</p>
            <details><summary>跨机器复制示例</summary><p>在 Ubuntu Web 主机的终端执行；先替换命令中的地址和绝对路径，并确认目标目录存在、目标名称未被占用。trace-root 需要向服务维护者确认，不能从浏览器地址推断。</p><GuideCode label="复制完整 CTF 目录（需替换路径）" :code="examples.copy" /></details>
          </li>
          <li>
            <h2>刷新列表并打开</h2>
            <p>返回首页，点击“刷新 trace 文件列表”，选中刚才的记录，再点击“打开分析”。先看概览，再打开一个请求查看耗时拆分和原始记录。</p>
            <p>使用下方 Python 示例时，查找请求 <code>demo-001</code>，在时间线或 Span 记录中查看 <code>prepare_input</code>。示例没有机器人 dispatch / 首动作边界，因此不产生完整端到端耗时。</p>
            <details><summary>列表里没有文件，或打开后没有业务记录？</summary><ul>
              <li>核对 Web 服务的 trace-root、服务用户读权限、完整 CTF 目录；等待记录停止和复制完成后刷新。</li>
              <li>列表加载失败时先重试连接；不要把读取失败当作空目录。</li>
              <li>没有业务事件：确认实际执行到了埋点、IB_TRACE_ENABLED 在进程启动前启用、Python 能导入 lttngust，并使用 ib_trace.* logger。</li>
              <li>CTF 之外还可读取 IBTRACE1 结构化日志和 .ibtrace 合并文件，普通应用日志不会自动变成可分析的 trace。</li>
            </ul></details>
          </li>
        </ol>
      </template>
      <template v-else>
        <p class="guide-intro">当前自定义埋点支持 Python。用 Span 记录一段代码的耗时，用 Event 记录瞬间状态；已有内置埋点可以直接使用。</p>
        <section>
          <h2>先运行一个最小示例</h2>
          <p>在业务机器的 IB_Robot 根目录保存为 <code>trace_demo.py</code>。这是普通 Python 计算示例，不需要机器人或模型。</p>
          <GuideCode label="trace_demo.py 示例" :code="examples.instrument" />
          <p>先按<GuideLink topic="capture" section="manual">手动记录 Python 示例</GuideLink>的步骤开启会话，再在另一个终端运行：</p>
          <GuideCode label="运行 trace_demo.py" :code="examples.demo" />
          <p>停止会话后，可以先在记录机器验证：</p>
          <GuideCode label="检查示例 Span" :code="examples.inspect" />
          <p>预期包含名称 prepare_input、请求 demo-001 和 origin=user 的 Span，具体耗时由本次运行决定。然后把完整目录交给 WebUI，查找同一个请求，在“时间线”或更多视图中的“Span 记录”查看记录。</p>
        </section>
        <section>
          <h2>接到自己的业务代码</h2>
          <dl class="guide-definitions">
            <dt>同一个请求，使用同一个 request_id</dt><dd>将示例 demo-001 换成业务已有的请求 ID，每次独立请求使用不同 ID。跨节点通过消息传递 ID，并在接收端建立 trace_context；新线程需要显式传播上下文。</dd>
            <dt>Logger 与组件名称</dt><dd>Logger 使用 ib_trace.* 前缀，才能被默认 Python tracing 会话记录。component_id 表示业务模块；每个模块创建并复用 emitter。</dd>
            <dt>Span 测耗时，Event 记状态</dt><dd>用 span 包住真正要测量的代码；可嵌套来细分阶段。Span 默认标记为 user，自定义 Event 显式设置 origin="user"。仅记录数量、shape 等小字段。</dd>
            <dt>在已有请求上下文内添加 Span</dt><dd>如果框架已建立上下文，直接添加 span / event 即可，不要用示例 ID 覆盖真实请求。tracepoint_description 提供埋点说明，便于从记录详情理解用途。</dd>
          </dl>
        </section>
        <section>
          <h2>怎样解释结果</h2>
          <p>Span 是主机侧经过的墙钟时间，包含等待，不等于 CPU 或 GPU/NPU 实际执行时间。没有业务起终点的自定义请求依然可看原始记录，但完整端到端指标可能为空。</p>
          <p>跨组件传递可进一步使用 flow_send / flow_receive；跨主机分析还需要同期时钟校准。完整 API 与高级用法见工作区 <code>src/ibrobot_tracing/README.md</code> 和 <code>src/ibrobot_tracing/docs/trace_merge.md</code>。</p>
        </section>
      </template>
      <footer class="guide-end"><RouterLink :to="back">{{ back === '/analysis' ? '返回首页，选择 trace 文件' : '返回原分析，继续查看' }}</RouterLink></footer>
    </div>
  </article>
</template>
