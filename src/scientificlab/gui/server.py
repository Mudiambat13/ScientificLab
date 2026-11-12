"""Local web server for the ScientificLab graphical interface.

Security model (the solver parses text with SymPy, which uses ``eval``, so the
server must never be reachable by anything but the person using it):

* it binds to the loopback interface only (``127.0.0.1``);
* every API call needs a random per-session token, embedded in the page served
  on ``GET /`` (a page from another website cannot read it);
* the ``Host`` header must be the loopback address and port (blocks DNS
  rebinding) and, if an ``Origin`` header is present, it must be the same origin;
* request bodies are limited to 64 KiB and must be JSON.

Standard library only (``http.server``): no new dependency.
"""

import argparse
import json
import secrets
import threading
import webbrowser
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from scientificlab import __version__
from scientificlab.gui.api import HANDLERS

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024
_TOKEN_MARK = "__SCIENTIFICLAB_TOKEN__"
_VERSION_MARK = "__SCIENTIFICLAB_VERSION__"


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], token: str) -> None:
        super().__init__(address, _Handler)
        self.token = token

    @property
    def allowed_hosts(self) -> set[str]:
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = "ScientificLab"

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default per-request logging."""

    # -- helpers ----------------------------------------------------------
    def _send(
        self, status: int, body: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; img-src data:; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, allow_nan=False).encode("utf-8"))

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.server.allowed_hosts

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:
        """Serve the single-page application."""
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host."})
        elif self.path in ("/", "/index.html"):
            page = (
                resources.files("scientificlab.gui")
                .joinpath("index.html")
                .read_text(encoding="utf-8")
                .replace(_TOKEN_MARK, self.server.token)
                .replace(_VERSION_MARK, __version__)
            )
            self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        """Handle ``POST /api/<name>`` JSON calls."""
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host."})
        origin = self.headers.get("Origin")
        if origin is not None and origin.split("://", 1)[-1] not in (
            self.server.allowed_hosts
        ):
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden origin."})
        supplied = self.headers.get("X-Token", "")
        if not secrets.compare_digest(supplied, self.server.token):
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid token."})
        name = self.path.removeprefix("/api/") if self.path.startswith("/api/") else ""
        handler = HANDLERS.get(name)
        if handler is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send application/json."}
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Body too large."}
            )
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON."})
        try:
            result = handler(payload)
        except Exception:  # unexpected bug: do not leak internals
            return self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal error."}
            )
        status = HTTPStatus.BAD_REQUEST if "error" in result else HTTPStatus.OK
        self._json(status, result)


def create_server(port: int = 0) -> _Server:
    """Create (but do not start) the server on ``127.0.0.1:port``.

    ``port=0`` picks a free port. The token is random per server instance.
    """
    return _Server((HOST, port), secrets.token_urlsafe(32))


def serve(port: int = 0, *, open_browser: bool = True) -> None:
    """Run the interface until interrupted with Ctrl+C."""
    server = create_server(port)
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"ScientificLab {__version__} - interface at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: ``scientificlab-gui [--port N] [--no-browser]``."""
    parser = argparse.ArgumentParser(
        prog="scientificlab-gui", description="ScientificLab local graphical interface"
    )
    parser.add_argument("--port", type=int, default=0, help="port (default: free)")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the browser"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    serve(args.port, open_browser=not args.no_browser)
