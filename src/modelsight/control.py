"""A tiny HTTP endpoint on every rank of a cluster run.

  GET /status                    {"rank": r, "state": "starting|serving|running_bench|done|committed|failed"}
  GET /logs/<name>?offset=<n>    bytes of run_dir/rank-<r>/<name> from byte n

The bench rank reads the other ranks' state from it, and they read the bench
rank's, which is how a failure on one node stops the others. A probe on one node
tails a server log on another through /logs. Only plain file names inside that
rank's directory are served.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

DEFAULT_PORT = 8399
STATES = ("starting", "serving", "running_bench", "done", "committed", "failed")
_LOG_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_log_path(rank_dir: Path, name: str) -> Path | None:
    """The file `name` inside `rank_dir`, or None if the name could escape it."""
    if not _LOG_NAME.fullmatch(name) or name in (".", ".."):
        return None
    root = rank_dir.resolve()
    path = (root / name).resolve()
    if path.parent != root:
        return None
    return path


class ControlServer:
    """Serves this rank's state and log files until stopped."""

    def __init__(self, rank: int, rank_dir: Path, port: int = DEFAULT_PORT,
                 host: str = "0.0.0.0") -> None:  # fmt: skip
        self.rank = rank
        self.rank_dir = rank_dir
        self.state = "starting"
        control = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                url = urlsplit(self.path)
                if url.path == "/status":
                    body = json.dumps({"rank": control.rank, "state": control.state})
                    self._send(200, body.encode(), "application/json")
                    return
                if url.path.startswith("/logs/"):
                    self._log(url.path.removeprefix("/logs/"), parse_qs(url.query))
                    return
                self._send(404, b"not found\n")

            def _log(self, name: str, query: dict) -> None:
                path = safe_log_path(control.rank_dir, name)
                if path is None:
                    self._send(400, b"bad log name\n")
                    return
                try:
                    offset = int(query.get("offset", ["0"])[0])
                except ValueError:
                    self._send(400, b"bad offset\n")
                    return
                if not path.is_file():
                    self._send(404, b"no such log\n")
                    return
                with path.open("rb") as handle:
                    handle.seek(max(0, offset))
                    data = handle.read()
                self._send(200, data, "application/octet-stream")

            def _send(self, code: int, body: bytes,
                      kind: str = "text/plain") -> None:  # fmt: skip
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def set(self, state: str) -> None:
        assert state in STATES, state
        self.state = state

    def start(self) -> ControlServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def peer_state(ip: str, port: int, timeout: float = 3) -> str | None:
    """Another rank's state, or None if it does not answer."""
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/status", timeout=timeout) as response:
            return json.loads(response.read())["state"]
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return None
