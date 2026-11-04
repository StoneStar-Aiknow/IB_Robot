/** Commands are displayed as text only. Keep aligned with scripts/tracing. */
export const guideExamples = {
  check: `source .shrc_local
lttng --version
babeltrace2 --version
python3 -c "import lttngust; import ibrobot_tracing"`,
  setup: `source .shrc_local
bash scripts/tracing/setup_tracing.sh`,
  launch: `enable_tracing:=true trace_session_name:=my_trace`,
  start: `source .shrc_local
bash scripts/tracing/start_trace.sh my_trace`,
  demo: `source .shrc_local
IB_TRACE_ENABLED=1 python3 trace_demo.py`,
  stop: `source .shrc_local
bash scripts/tracing/stop_trace.sh my_trace`,
  inspect: `source .shrc_local
python3 -m ibrobot_tracing.cli spans ~/.ros/tracing/my_trace --request-id demo-001`,
  copy: `# 替换业务机器地址、源目录和 Web 服务的 trace-root
scp -r USER@BUSINESS_HOST:/absolute/path/to/my_trace /absolute/web/trace-root/`,
  instrument: `from ibrobot_tracing import get_trace_emitter

trace = get_trace_emitter("ib_trace.demo", component_id="demo")


def process_request(request_id):
    with trace.trace_context(request_id):
        with trace.span(
            "prepare_input",
            tracepoint_description="示例：准备输入数据。",
        ):
            values = [value * 2 for value in range(1000)]
            trace.event("input_ready", origin="user", count=len(values))
    return values


if __name__ == "__main__":
    process_request("demo-001")`,
};
