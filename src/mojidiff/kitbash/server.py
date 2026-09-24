"""The kitbash tool: parts of two icons, placed and layered as vectors, out as a program.

No model. Every icon in the pilot corpus is served as its canonical 72-box SVG - the
typed codec's own serialization - and the page takes paths from two of them, moves,
scales, flips and layers them on the same box, and hands the composition back. The
server runs it through the project's normalizer, codec and packer and returns the
canonical program with its verdict: how many contours and segments, what the lattice
moved, or why the codec would not take it. Everything is vector; nothing is rendered
on the server.

Bound to one explicit address, the Tailscale IP by default. CPU only.
"""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST = "100.69.189.78"
DEFAULT_PORT = 8789
MAX_BODY = 1 << 20
ICON_CACHE = _REPO_ROOT / "data/processed/kitbash"
PILOT_CONFIG = _REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml"
PAGE = Path(__file__).with_name("index.html")


class Codec:
    def __init__(self) -> None:
        from mojidiff.learning.openmoji_pilot import _selected_codec, load_openmoji_pilot_config

        self.pilot = load_openmoji_pilot_config(PILOT_CONFIG)
        self.codec = _selected_codec(self.pilot)
        self.slots = self.pilot.total_segment_slots
        self.icons: list[dict[str, Any]] = json.loads((ICON_CACHE / "index.json").read_text())
        self.held_out = sorted(
            icon["hexcode"] for icon in self.icons if icon["split"] != "primary/train"
        )

    def canonical(self, svg: str) -> dict[str, Any]:
        """The composition through the codec: canonical SVG and the verdict."""

        from mojidiff.learning.omnisvg import parse_into_codec
        from mojidiff.representation.packed import serialize_packed_svg

        program, info = parse_into_codec(svg.encode(), self.codec, self.slots)
        if program is None:
            return {"ok": False, **info}
        return {
            "ok": True,
            **info,
            "svg": serialize_packed_svg(program, self.codec, self.slots).decode(),
            "capacity": {"paths": int(self.codec.max_paths), "segments": int(self.slots)},
        }


class _Handler(http.server.BaseHTTPRequestHandler):
    codec: Codec
    timeout = 30

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/icons":
            query = parse_qs(url.query).get("q", [""])[0].strip().lower()
            icons = [
                icon
                for icon in self.codec.icons
                if not query
                or query in icon["annotation"].lower()
                or query in icon["hexcode"].lower()
                or query in icon["subgroup"].lower()
            ]
            self._json(200, {"icons": icons[:150], "total": len(icons)})
        elif url.path.startswith("/icon/"):
            name = url.path[len("/icon/") :]
            path = ICON_CACHE / name
            if not name.endswith(".svg") or "/" in name or ".." in name or not path.is_file():
                self._json(404, {"error": "no such icon"})
                return
            self._send(200, path.read_bytes(), "image/svg+xml", "max-age=86400")
        elif url.path == "/palette":
            self._json(200, {"palette": list(self.codec.codec.palette)})
        elif url.path == "/health":
            self._json(200, {"ok": True, "icons": len(self.codec.icons)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/canonical":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            svg = str(payload["svg"])
        except (KeyError, ValueError, TypeError) as error:
            self._json(400, {"error": str(error)[:120]})
            return
        try:
            self._json(200, self.codec.canonical(svg))
        except Exception as error:  # noqa: BLE001 - the verdict is the answer, never a crash
            self._json(200, {"ok": False, "failure": f"{type(error).__name__}: {str(error)[:100]}"})


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str, port: int) -> None:
    codec = Codec()
    handler = type("Handler", (_Handler,), {"codec": codec})
    with _Server((host, port), functools.partial(handler)) as server:
        print(
            f"kitbash on http://{host}:{port} - {len(codec.icons)} icons",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever()
