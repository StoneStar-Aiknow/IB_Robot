#!/usr/bin/env python3
"""IB Robot web control panel.

Serves a mobile-friendly single-page panel on port 8766.
Provides start/stop inference, arm reset, and (sim mode) random banana
placement and AutoTest control via ROS2 service calls.

Usage:
    python3 scripts/model_infer_panel.py [--host 0.0.0.0] [--port 8766]
    python3 scripts/model_infer_panel.py --help
"""

import argparse
import json
import queue
import socket
import threading
from http import server

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty, Trigger

# ─── SSE broadcast ────────────────────────────────────────────────────────────

_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(msg: str, cls: str = "", state: str = "") -> None:
    payload = json.dumps({"msg": msg, "cls": cls, "state": state})
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for d in dead:
            _sse_clients.remove(d)


# ─── HTML page ───────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>IB Robot Panel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#faf8f4;font-family:system-ui,-apple-system,sans-serif;color:#1a1a1a;min-height:100vh}
.wrap{max-width:480px;margin:0 auto;padding:28px 16px}
.tabs{display:flex;border-bottom:1px solid #e8e4de;margin-bottom:32px}
.tab{padding:11px 28px;cursor:pointer;font-size:15px;color:#6b6560;
  border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s}
.tab.on{color:#002fa7;border-bottom-color:#002fa7;font-weight:600}
.ring{display:flex;justify-content:center;margin:8px 0 32px}
.btn-s{width:120px;height:120px;border-radius:50%;background:#002fa7;border:none;cursor:pointer;
  color:#fff;font-size:15px;font-weight:700;letter-spacing:.08em;
  box-shadow:0 4px 24px rgba(0,47,167,.22);transition:transform .1s}
.btn-s:hover{transform:scale(1.04)}
.btn-s.on{animation:glow 2s infinite}
@keyframes glow{
  0%{box-shadow:0 0 0 0 rgba(0,47,167,.5),0 4px 24px rgba(0,47,167,.22)}
  50%{box-shadow:0 0 0 18px rgba(0,47,167,.07),0 4px 40px rgba(0,47,167,.35)}
  100%{box-shadow:0 0 0 0 rgba(0,47,167,.5),0 4px 24px rgba(0,47,167,.22)}}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.btn-a{flex:1;min-width:110px;padding:11px 14px;background:#fff;
  border:1.5px solid #e8e4de;border-radius:10px;cursor:pointer;
  font-size:14px;color:#1a1a1a;transition:border-color .15s,background .15s}
.btn-a:hover{border-color:#002fa7;background:#f0f4ff}
.lhd{font-size:11px;font-weight:600;color:#6b6560;letter-spacing:.09em;
  text-transform:uppercase;margin-bottom:7px}
.log{background:#fff;border:1px solid #e8e4de;border-radius:10px;padding:12px;
  height:220px;overflow-y:auto;font-family:ui-monospace,monospace;font-size:12.5px}
.ll{padding:1.5px 0}
.ok{color:#1a7a47}.er{color:#b91c1c}
.sim-row{display:none}
body.sim .sim-row{display:flex}
@media(max-width:400px){.wrap{padding:16px 10px}.btn-s{width:100px;height:100px}
.tab{padding:10px 18px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="tabs">
    <div class="tab" id="t-sim" onclick="sw('sim')">仿真</div>
    <div class="tab on" id="t-real" onclick="sw('real')">真机</div>
  </div>
  <div class="ring"><button class="btn-s" id="bs" onclick="tog()">START</button></div>
  <div class="row sim-row">
    <button class="btn-a" onclick="act('/api/random_set','随机放置香蕉...')">🎲 Random Set</button>
    <button class="btn-a" onclick="act('/api/scene_reset','重置场景...')">↺ Reset Scene</button>
    <button class="btn-a" onclick="act('/api/autotest','启动 AutoTest...')">▶ AutoTest</button>
  </div>
  <div class="lhd">Log</div>
  <div class="log" id="lg"></div>
</div>
<script>
let on=false;
function sw(m){
  document.getElementById('t-sim').classList.toggle('on',m==='sim');
  document.getElementById('t-real').classList.toggle('on',m==='real');
  document.body.classList.toggle('sim',m==='sim');
  sessionStorage.setItem('m',m);
}
function log(msg,cls){
  const b=document.getElementById('lg');
  const d=document.createElement('div');
  d.className='ll'+(cls?' '+cls:'');
  const ts=new Date().toLocaleTimeString('zh-CN',{hour12:false});
  d.textContent='['+ts+'] '+msg;
  b.appendChild(d);
  if(b.children.length>200)b.removeChild(b.firstChild);
  b.scrollTop=b.scrollHeight;
}
function upd(){
  const b=document.getElementById('bs');
  b.classList.toggle('on',on);
  b.textContent=on?'STOP':'START';
}
function tog(){
  if(!on)act('/api/start','启动推理...');
  else act('/api/stop','停止推理...');
}
async function act(p,m){
  log(m);
  try{
    const r=await fetch(p,{method:'POST'});
    const d=await r.json();
    if(d.ok){
      log(d.msg||'完成','ok');
      if(p==='/api/start'){on=true;upd();}
      if(p==='/api/stop'){on=false;upd();}
    }else{
      log('错误: '+(d.error||'未知'),'er');
    }
  }catch(e){log('请求失败: '+e,'er');}
}
const es=new EventSource('/events');
es.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.msg)log(d.msg,d.cls||'');
  if(d.state){on=(d.state==='running');upd();}
};
es.onerror=()=>setTimeout(()=>location.reload(),3000);
sw(sessionStorage.getItem('m')||'real');
</script>
</body>
</html>"""


# ─── HTTP handler ────────────────────────────────────────────────────────────


class _Handler(server.BaseHTTPRequestHandler):
    node: "PanelNode"  # set on class before server starts

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML.encode())
        elif self.path == "/events":
            self._stream_sse()
        else:
            self.send_error(404)

    def do_POST(self):
        routes = {
            "/api/start": lambda: self.node.request_start(),
            "/api/stop": lambda: self.node.request_stop(),
            "/api/random_set": lambda: self.node.request_sim("random_set"),
            "/api/scene_reset": lambda: self.node.request_sim("scene_reset"),
            "/api/autotest": lambda: self.node.request_sim("autotest"),
        }
        fn = routes.get(self.path)
        if fn is None:
            self.send_error(404)
            return
        result = fn()
        self._send(200, "application/json; charset=utf-8", json.dumps(result).encode())

    def _stream_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue = queue.Queue(maxsize=64)
        with _sse_lock:
            _sse_clients.append(q)
        try:
            while True:
                try:
                    payload = q.get(timeout=15)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # suppress HTTP access log
        return


# ─── ROS2 panel node ──────────────────────────────────────────────────────────


class PanelNode(Node):
    """ROS2 node that bridges HTTP requests to ROS2 service calls."""

    def __init__(
        self,
        arm_topic: str,
        gripper_topic: str,
        rest_arm: list,
        rest_gripper: float | None,
        reset_publish_count: int,
    ):
        super().__init__("model_infer_panel")

        self._rest_arm = list(rest_arm)
        self._rest_gripper = rest_gripper
        self._reset_publish_count = max(1, reset_publish_count)

        # Reentrant: _tick() holds this lock and calls _set_state(), which
        # re-acquires it. A plain Lock would self-deadlock the executor thread
        # ~0.8s after every stop/reset (when the rest-pose countdown ends),
        # which then hangs every later request_start() on the HTTP threads.
        self._lock = threading.RLock()
        self._pending: str | None = None
        self._sim_pending: str | None = None  # "random_set" | "scene_reset" | "autotest"
        self._publish_remaining = 0
        self._stop_pending = False
        self._reset_pending = False
        self._start_pending = False
        self._state = {"busy": False, "state": "idle", "detail": "Ready"}

        self._disp_start = self.create_client(Trigger, "/action_dispatcher/start_evaluate")
        self._disp_stop = self.create_client(Trigger, "/action_dispatcher/stop_evaluate")
        self._disp_reset = self.create_client(Empty, "/action_dispatcher/reset")
        self._disp_status = self.create_client(Trigger, "/action_dispatcher/get_status")

        self._sim_random = self.create_client(Trigger, "/sim/pick_banana/random_set")
        self._sim_reset = self.create_client(Trigger, "/sim/pick_banana/reset")
        self._sim_autotest = self.create_client(Trigger, "/sim/pick_banana/autotest/start")

        self._arm_pub = self.create_publisher(Float64MultiArray, arm_topic, 10)
        self._grip_pub = (
            self.create_publisher(Float64MultiArray, gripper_topic, 10) if rest_gripper is not None else None
        )

        self.create_timer(0.1, self._tick)
        self.create_timer(2.0, self._poll_status)

        self.get_logger().info(f"PanelNode started on {arm_topic}")

    # ── public API (called from HTTP thread) ──────────────────────────────────

    def request_start(self) -> dict:
        with self._lock:
            if self._state["busy"]:
                self.get_logger().info(f"START rejected: busy (state={self._state})")
                return {"ok": False, "error": "busy"}
            self._pending = "start"
            self._state = {"busy": True, "state": "queued", "detail": "Start queued"}
        return {"ok": True, "msg": "Start queued"}

    def request_stop(self) -> dict:
        with self._lock:
            if self._state["busy"]:
                return {"ok": False, "error": "busy"}
            self._pending = "stop"
            self._publish_remaining = self._reset_publish_count
            self._state = {"busy": True, "state": "stopping", "detail": "Stop queued"}
        return {"ok": True, "msg": "Stop queued"}

    def request_sim(self, action: str) -> dict:
        clients = {
            "random_set": self._sim_random,
            "scene_reset": self._sim_reset,
            "autotest": self._sim_autotest,
        }
        if action not in clients:
            return {"ok": False, "error": f"unknown sim action: {action}"}
        if not clients[action].service_is_ready():
            return {"ok": False, "error": f"/sim/pick_banana/{action} not available"}
        # Queue for the ROS timer to dispatch — avoids blocking the HTTP thread
        with self._lock:
            self._sim_pending = action
        label = {"random_set": "随机放置...", "scene_reset": "重置场景...", "autotest": "AutoTest 已排队..."}.get(
            action, action
        )
        return {"ok": True, "msg": label}

    # ── timer callbacks ────────────────────────────────────────────────────────

    def _tick(self):
        do_start = do_stop = False
        sim_action = None
        with self._lock:
            if self._pending == "start":
                self._pending = None
                do_start = True
            elif self._pending == "stop":
                self._pending = None
                do_stop = True
            if self._sim_pending:
                sim_action = self._sim_pending
                self._sim_pending = None
            remaining = self._publish_remaining

        if do_start:
            _broadcast("启动推理...", state="starting")
            self._call_start()
        if do_stop:
            _broadcast("停止推理，手臂回位...", state="stopping")
            self._call_stop()
            self._call_reset()
        if sim_action:
            self._call_sim(sim_action)

        if remaining > 0:
            self._arm_pub.publish(Float64MultiArray(data=self._rest_arm))
            if self._grip_pub and self._rest_gripper is not None:
                self._grip_pub.publish(Float64MultiArray(data=[self._rest_gripper]))
            with self._lock:
                self._publish_remaining -= 1
                if self._publish_remaining <= 0 and not self._stop_pending and not self._reset_pending:
                    self._set_state("stopped", "Arm at rest pose")

    def _poll_status(self):
        with self._lock:
            if self._state["busy"]:
                return
        if not self._disp_status.service_is_ready():
            return
        future = self._disp_status.call_async(Trigger.Request())
        future.add_done_callback(self._on_status)

    def _on_status(self, future):
        try:
            r = future.result()
            msg = (r.message or "").strip().lower()
            state = "running" if msg == "running" else "stopped" if msg == "stopped" else None
            if state:
                with self._lock:
                    if not self._state["busy"]:
                        self._state = {"busy": False, "state": state, "detail": f"Dispatcher: {msg}"}
                        _broadcast("", state=state)
        except Exception:
            pass

    def _call_start(self):
        ready = self._disp_start.service_is_ready()
        self.get_logger().info(f"START -> /action_dispatcher/start_evaluate (service_ready={ready})")
        if not ready:
            self._set_state("error", "start_evaluate unavailable")
            _broadcast("start_evaluate 服务不可用", cls="er", state="error")
            return
        future = self._disp_start.call_async(Trigger.Request())
        with self._lock:
            self._start_pending = True
        future.add_done_callback(self._on_start_done)

    def _call_sim(self, action: str) -> None:
        clients = {
            "random_set": self._sim_random,
            "scene_reset": self._sim_reset,
            "autotest": self._sim_autotest,
        }
        client = clients.get(action)
        if client is None or not client.service_is_ready():
            _broadcast(f"sim/{action}: service not ready", cls="er")
            return
        future = client.call_async(Trigger.Request())

        def _done(f):
            try:
                r = f.result()
                _broadcast(f"sim/{action}: {r.message}", cls="ok" if r.success else "er")
            except Exception as exc:
                _broadcast(f"sim/{action}: {exc}", cls="er")

        future.add_done_callback(_done)

    def _call_stop(self):
        if not self._disp_stop.service_is_ready():
            return
        future = self._disp_stop.call_async(Trigger.Request())
        with self._lock:
            self._stop_pending = True
        future.add_done_callback(self._on_stop_done)

    def _call_reset(self):
        if not self._disp_reset.service_is_ready():
            return
        future = self._disp_reset.call_async(Empty.Request())
        with self._lock:
            self._reset_pending = True
        future.add_done_callback(self._on_reset_done)

    def _on_start_done(self, future):
        try:
            r = future.result()
            state = "running" if r.success else "error"
            detail = r.message or ("Running" if r.success else "Failed to start")
        except Exception as exc:
            state, detail = "error", str(exc)
        self.get_logger().info(f"start_evaluate result: state={state} detail={detail!r}")
        with self._lock:
            self._start_pending = False
        self._set_state(state, detail)
        _broadcast(detail, cls="ok" if state == "running" else "er", state=state)

    def _on_stop_done(self, future):
        with self._lock:
            self._stop_pending = False
            done = self._publish_remaining <= 0 and not self._reset_pending
        if done:
            self._set_state("stopped", "Stopped")
            _broadcast("停止完成", cls="ok", state="stopped")

    def _on_reset_done(self, future):
        with self._lock:
            self._reset_pending = False
            done = self._publish_remaining <= 0 and not self._stop_pending
        if done:
            self._set_state("stopped", "Reset complete")

    def _set_state(self, state: str, detail: str):
        with self._lock:
            busy = state in {"queued", "starting", "stopping"}
            self._state = {"busy": busy, "state": state, "detail": detail}


# ─── Main ────────────────────────────────────────────────────────────────────


def _rest_arm_arg(value: str) -> list[float]:
    try:
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rest arm positions must be comma-separated floats") from exc
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("rest arm positions must contain exactly 5 values")
    return parts


def _local_ip_hint() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IB Robot web control panel")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--arm-topic", default="/arm_position_controller/commands")
    p.add_argument("--gripper-topic", default="/gripper_position_controller/commands")
    p.add_argument(
        "--rest-arm",
        type=_rest_arm_arg,
        default=[0.0, -1.5854, 1.5708, 0.7854, 0.0],
        help="Comma-separated 5-joint rest positions",
    )
    p.add_argument("--rest-gripper", type=float, default=None)
    p.add_argument("--reset-publish-count", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rest_arm = args.rest_arm

    rclpy.init()
    node = PanelNode(
        arm_topic=args.arm_topic,
        gripper_topic=args.gripper_topic,
        rest_arm=rest_arm,
        rest_gripper=args.rest_gripper,
        reset_publish_count=args.reset_publish_count,
    )

    _Handler.node = node  # type: ignore[attr-defined]
    httpd = server.ThreadingHTTPServer((args.host, args.port), _Handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    print(f"Panel: http://localhost:{args.port}")
    if args.host in {"0.0.0.0", "::"}:
        ip_hint = _local_ip_hint()
        if ip_hint:
            print(f"Phone: open http://{ip_hint}:{args.port} on the same network")
        else:
            print(f"Phone: open http://<this-computer-ip>:{args.port} on the same network")
        print("Warning: this panel is reachable from the local network and can start/stop the robot.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
