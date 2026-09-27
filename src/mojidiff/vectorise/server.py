"""Edit pixels, re-vectorise: the render-to-SVG model behind a small web page.

The page shows a held-out OpenMoji icon as pixels. Paint on it, erase, or draw from
scratch; each change is sent here, and the model transcribes the canvas back into a
codec program. The answer is the canonical SVG, the time the model took, and how many
decoder calls it needed. Nothing is looked up: held-out icons are ones the model never
trained on.

Bound to one explicit address, the Tailscale IP by default, like the other tools.
"""

from __future__ import annotations

import base64
import functools
import http.server
import io
import json
import socketserver
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST = "100.69.189.78"
DEFAULT_PORT = 8790
MAX_BODY = 4 << 20
ICON_CACHE = _REPO_ROOT / "data/processed/kitbash"
PILOT_CONFIG = _REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml"
PAGE = Path(__file__).with_name("index.html")


def canvas_to_rgb(png: bytes, size: int) -> np.ndarray:
    """A browser canvas PNG as the model's input: RGB on white, `size` square."""

    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
        if rgba.size != (size, size):
            rgba = rgba.resize((size, size), Image.Resampling.LANCZOS)
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return np.asarray(background.convert("RGB"), dtype=np.uint8)


class Vectoriser:
    """Owns the model, the codec, and one lock so requests decode one at a time."""

    def __init__(self, model: Any, template: Any, device: torch.device) -> None:
        import threading

        self.model = model.eval()
        self.layout = model.layout
        self.template = template
        self.device = device
        self.lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> Vectoriser:
        from mojidiff.learning.autoregressive import SequenceLayout
        from mojidiff.learning.openmoji_pilot import (
            _load_program,
            _selected_codec,
            load_openmoji_pilot_config,
            load_pilot_index,
        )
        from mojidiff.learning.render2svg import ModelConfig, RenderToProgram

        pilot = load_openmoji_pilot_config(PILOT_CONFIG)
        codec = _selected_codec(pilot)
        layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
        by_split, _, _ = load_pilot_index(pilot)
        template = _load_program(by_split["primary/train"][0], pilot, codec)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=device)
        model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device)
        model.load_state_dict(state["model"])
        return cls(model, template, device)

    @property
    def image_size(self) -> int:
        return int(self.model.config.image_size)

    def vectorise(self, rgb: np.ndarray) -> dict[str, Any]:
        from mojidiff.learning.autoregressive import unflatten_program
        from mojidiff.learning.render2svg import DecodeStats, greedy_decode
        from mojidiff.representation.packed import serialize_packed_svg

        images = torch.from_numpy(rgb)[None].to(self.device)
        stats = DecodeStats()
        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                tokens = greedy_decode(self.model, images, stats=stats)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
        program = unflatten_program(tokens[0], self.template, self.layout)
        svg = serialize_packed_svg(program, self.layout.codec, self.layout.total_segment_slots)
        lengths = [int(v) for v in program.path_length if int(v) > 0]
        return {
            "ok": True,
            "svg": svg.decode(),
            "ms": round(elapsed, 1),
            "decoder_calls": stats.model_calls,
            "paths": len(lengths),
            "segments": sum(lengths),
        }

    def icon_png(self, hexcode: str) -> bytes:
        """A held-out icon rendered exactly as the model saw icons in training."""

        from PIL import Image

        from mojidiff.learning.render2svg import render_trusted_rgb

        svg = (ICON_CACHE / f"{hexcode}.svg").read_bytes()
        buffer = io.BytesIO()
        Image.fromarray(render_trusted_rgb(svg, self.image_size)).save(buffer, format="PNG")
        return buffer.getvalue()


def held_out_icons() -> list[dict[str, Any]]:
    icons: list[dict[str, Any]] = json.loads((ICON_CACHE / "index.json").read_text())
    return [icon for icon in icons if icon["split"] != "primary/train"]


class _Handler(http.server.BaseHTTPRequestHandler):
    vectoriser: Vectoriser
    icons: list[dict[str, Any]]
    palette: list[str]
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
            found = [
                icon
                for icon in self.icons
                if not query
                or query in icon["annotation"].lower()
                or query in icon["hexcode"].lower()
            ]
            self._json(200, {"icons": found[:120], "total": len(found)})
        elif url.path.startswith("/render/") and url.path.endswith(".png"):
            hexcode = url.path[len("/render/") : -len(".png")]
            if not any(icon["hexcode"] == hexcode for icon in self.icons):
                self._json(404, {"error": "not a held-out icon"})
                return
            self._send(200, self.vectoriser.icon_png(hexcode), "image/png", "max-age=86400")
        elif url.path == "/config":
            self._json(200, {"size": self.vectoriser.image_size, "palette": self.palette})
        elif url.path == "/health":
            self._json(200, {"ok": True, "icons": len(self.icons)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/vectorise":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            data = str(payload["png"])
            png = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
            rgb = canvas_to_rgb(png, self.vectoriser.image_size)
        except (KeyError, ValueError, TypeError, OSError) as error:
            self._json(400, {"error": str(error)[:120]})
            return
        try:
            self._json(200, self.vectoriser.vectorise(rgb))
        except Exception as error:  # noqa: BLE001 - report, never crash the server
            self._json(200, {"ok": False, "failure": f"{type(error).__name__}: {str(error)[:100]}"})


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str, port: int, checkpoint: Path) -> None:
    vectoriser = Vectoriser.from_checkpoint(checkpoint)
    icons = held_out_icons()
    handler = type(
        "Handler",
        (_Handler,),
        {
            "vectoriser": vectoriser,
            "icons": icons,
            "palette": list(vectoriser.layout.codec.palette),
        },
    )
    with _Server((host, port), functools.partial(handler)) as server:
        print(
            f"vectorise on http://{host}:{port} - {len(icons)} held-out icons, {checkpoint}",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever()
