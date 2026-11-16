from __future__ import annotations

import argparse
import errno
import sys

from .server import WebServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-skill-web", description="Local IB-Robot skill control panel")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8765)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config-name", help="deployed robot_config name (or use ROBOT_CONFIG)")
    selection.add_argument("--config-path", help="explicit robot_config YAML path of the deployed robot")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        parser.error("only 127.0.0.1 and ports 1-65535 are supported")
    try:
        server = WebServer(
            config_name=args.config_name,
            config_path=args.config_path,
            address=args.host,
            port=args.port,
        )
    except (OSError, ValueError) as exc:
        print(
            f"Web 启动失败：无法绑定部署配置（{exc}）。"
            "请传 --config-name/--config-path，或设置 ROBOT_CONFIG/ROBOT_NAME。未启动新的 Web 服务。",
            file=sys.stderr,
            flush=True,
        )
        return 2
    connection = server.state.client.session
    print(
        f"已绑定部署配置：{connection['robot_name'] or connection['robot_id']}（目标 {connection['robot_id']}）。",
        flush=True,
    )
    print("页面不发现、不切换机器人；启动与打开页面都不会执行动作。", flush=True)
    print("仅用于本机界面与受信任的 ROS 网络；运动仍需 Gateway 显式授权。", flush=True)
    print(f"启动成功后访问：http://127.0.0.1:{args.port}/", flush=True)
    print("仅本机访问；会话或页面关闭不会取消已提交任务。", flush=True)
    try:
        with server:
            server.serve_forever()
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(
            f"Web 启动失败：127.0.0.1:{args.port} 端口被占用，或旧连接尚未释放。"
            "请先停止原 Web 服务，或通过 --port 指定其他端口。未启动新的 Web 服务。",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
