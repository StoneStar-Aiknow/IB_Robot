"""Threaded HTTP/HTTPS server for the installed WebPhone interface."""

from __future__ import annotations

import http.server
import json
import logging
import socket
import socketserver
import ssl
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_access_urls(bind_address: str, port: int, path: str = "/web_teleop.html", scheme: str = "http") -> list[str]:
    """Return useful local and LAN URLs for the configured listener."""
    hosts: list[str] = []
    if bind_address not in ("0.0.0.0", "::"):
        hosts.append(bind_address)
    else:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                lan_ip = sock.getsockname()[0]
            if lan_ip and not lan_ip.startswith("127."):
                hosts.append(lan_ip)
        except OSError:
            pass
        hosts.extend(["127.0.0.1", "localhost"])
    return [f"{scheme}://{host}:{port}{path}" for host in dict.fromkeys(hosts)]


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class WebServer:
    """Serve static WebPhone assets and runtime client configuration."""

    def __init__(
        self,
        *,
        web_root: Path,
        bind_address: str,
        port: int,
        use_https: bool,
        cert_file: Path | None,
        key_file: Path | None,
        client_config: dict[str, Any],
    ) -> None:
        self.web_root = web_root.resolve()
        self.bind_address = bind_address
        self.port = port
        self.use_https = use_https
        self.cert_file = cert_file
        self.key_file = key_file
        self.client_config = client_config
        self._server: _ThreadedTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def scheme(self) -> str:
        return "https" if self.use_https else "http"

    def start(self) -> None:
        """Start the server once and fail if the installed page is missing."""
        if self._server is not None:
            raise RuntimeError("WebPhone HTTP server is already running")
        required_assets = ("web_teleop.html", "optical_flow_worker.js", "three.min.js")
        missing_assets = [name for name in required_assets if not (self.web_root / name).is_file()]
        if missing_assets:
            raise FileNotFoundError(f"Installed WebPhone assets are missing: {', '.join(missing_assets)}")

        web_root = self.web_root
        api_config = self.client_config

        class Handler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self) -> None:
                if self.path.split("?", 1)[0] in (
                    "/",
                    "/web_teleop.html",
                    "/optical_flow_worker.js",
                    "/three.min.js",
                ):
                    self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                super().end_headers()

            def do_GET(self) -> None:  # noqa: N802 - inherited HTTP handler API
                if self.path.split("?", 1)[0] == "/api/config":
                    payload = json.dumps(api_config).encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                request_path = self.path.split("?", 1)[0]
                if request_path not in ("/", "/web_teleop.html", "/optical_flow_worker.js", "/three.min.js"):
                    self.send_error(HTTPStatus.NOT_FOUND, "WebPhone asset not found")
                    return
                super().do_GET()

            def translate_path(self, path: str) -> str:
                request_path = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
                if request_path in ("", "web_teleop.html"):
                    return str(web_root / "web_teleop.html")
                if request_path in ("optical_flow_worker.js", "three.min.js"):
                    return str(web_root / request_path)
                return str(web_root / "__not_found__")

            def log_message(self, format_string: str, *args: Any) -> None:
                logger.debug("WebPhone HTTP: " + format_string, *args)

        server = _ThreadedTCPServer((self.bind_address, self.port), Handler)
        if self.use_https:
            if self.cert_file is None or self.key_file is None:
                server.server_close()
                raise ValueError("HTTPS requires both a certificate and private key")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)

        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="webphone-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server idempotently and wait for its thread to exit."""
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
