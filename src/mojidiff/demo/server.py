"""The live re-vectorise demo: a raster edit in, OmniSVG's program out, over the tailnet.

One process loads the released OmniSVG 1.1 4B once and serves a single page. The page
lets a visitor pick an OpenMoji icon (any of the cached 448 px renders), edit it in
pixels - recolour a region, erase with a brush, move a part - or upload a PNG, choose
how many candidates to draw, and submit. Requests run one at a time on the GPU through
a queue; each candidate is drawn under the model's own image sampler, rendered, and
compared with the input in pixels, and the closest wins - the best-of-K decoding that
passed Gate N. The winner comes back as the project's 72-box SVG (the codec's canonical
form when the codec takes it, the scaled raw drawing otherwise) with every candidate
and its error beside it.

Bound to one explicit address, the machine's Tailscale IP by default, never to every
interface; uploads are capped; nothing is written outside the process.
"""

from __future__ import annotations

import base64
import functools
import http.server
import io
import json
import socketserver
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST = "100.69.189.78"
DEFAULT_PORT = 8788
MAX_UPLOAD_BYTES = 3 << 20
MAX_CANDIDATES = 12
MAX_QUEUE = 8
PILOT_CONFIG = _REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml"
RENDER_CACHE = _REPO_ROOT / "data/processed/prior/omnisvg-renders-448"
PAGE = Path(__file__).with_name("index.html")


@dataclass
class Job:
    id: str
    image: Image.Image
    candidates_wanted: int
    seed: int
    created: float = field(default_factory=time.time)
    state: str = "queued"
    started: float | None = None
    finished: float | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def view(self, position: int | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "position": position,
            "candidates_wanted": self.candidates_wanted,
            "candidates": self.candidates,
            "elapsed": round((self.finished or time.time()) - (self.started or time.time()), 1)
            if self.started
            else None,
            "result": self.result,
            "error": self.error,
        }


class Engine:
    """The model, the codec, and one worker thread that drains the queue."""

    def __init__(self, device: str = "cuda") -> None:
        from mojidiff.learning.omnisvg import OmniSVG
        from mojidiff.learning.omnisvg_study import captions
        from mojidiff.learning.openmoji_pilot import (
            _selected_codec,
            load_openmoji_pilot_config,
            load_pilot_index,
        )

        self.pilot = load_openmoji_pilot_config(PILOT_CONFIG)
        self.codec = _selected_codec(self.pilot)
        self.slots = self.pilot.total_segment_slots
        by_split, _, _ = load_pilot_index(self.pilot)
        annotations = captions(self.pilot.raw_root)
        self.icons = [
            {
                "hexcode": row.hexcode,
                "annotation": annotations.get(row.hexcode, row.hexcode),
                "split": row.split,
            }
            for split in ("primary/validation", "primary/test", "primary/train")
            for row in by_split[split]
            if (RENDER_CACHE / f"{row.hexcode}.png").is_file()
        ]
        self.held_out = {row.hexcode for row in by_split["primary/validation"]} | {
            row.hexcode for row in by_split["primary/test"]
        }
        started = time.perf_counter()
        self.model = OmniSVG.load(device)
        self.load_seconds = time.perf_counter() - started
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.worker = threading.Thread(target=self._drain, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ queue
    def submit(self, image: Image.Image, candidates: int, seed: int) -> Job:
        job = Job(uuid.uuid4().hex[:12], image, candidates, seed)
        with self.lock:
            if len(self.queue) >= MAX_QUEUE:
                raise RuntimeError("the queue is full; try again in a few minutes")
            self.jobs[job.id] = job
            self.queue.append(job.id)
            self._forget_old()
        self.wake.set()
        return job

    def view(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            position = self.queue.index(job_id) + 1 if job_id in self.queue else None
            return job.view(position)

    def _forget_old(self) -> None:
        cutoff = time.time() - 3600
        for job_id in [j for j, job in self.jobs.items() if job.finished and job.finished < cutoff]:
            del self.jobs[job_id]

    def _drain(self) -> None:
        while True:
            self.wake.wait(timeout=5)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    job = self.jobs[self.queue.pop(0)]
                try:
                    self._run(job)
                except Exception as error:  # noqa: BLE001 - a failed job is reported, never fatal
                    job.state, job.error = "failed", f"{type(error).__name__}: {str(error)[:120]}"
                    job.finished = time.time()

    # ------------------------------------------------------------------- work
    def _run(self, job: Job) -> None:
        from mojidiff.learning.omnisvg import decode_tokens, parse_into_codec, to_project_svg

        # The study's wall-clock guard uses signals, which only the main thread may set;
        # this worker is a thread, so the steps run unguarded and the candidate count
        # and token budget bound the work instead.
        from mojidiff.learning.omnisvg_study import render_raw
        from mojidiff.representation.packed import serialize_packed_svg

        job.state, job.started = "running", time.time()
        target = np.asarray(job.image.resize((72, 72)), dtype=np.float32)
        for index in range(job.candidates_wanted):
            clock = time.perf_counter()
            tokens = self.model.generate(
                "demo",
                samples=1,
                max_new_tokens=2048,
                seed=job.seed + index,
                style="image",
                image=job.image,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
            )[0]
            svg, info = decode_tokens(
                self.model.svg_tokenizer, self.model.black_color_token, tokens
            )
            candidate: dict[str, Any] = {
                "index": index,
                "tokens": info["tokens"],
                "ended": info["ended"],
                "paths": info.get("paths"),
                "seconds": round(time.perf_counter() - clock, 1),
                "error": None,
                "png": None,
            }
            if svg is not None:
                try:
                    picture = render_raw(svg, 72)
                    candidate["error"] = round(
                        float(
                            np.abs(np.asarray(picture, dtype=np.float32) - target).mean() / 255.0
                        ),
                        4,
                    )
                    candidate["png"] = _png_data_url(render_raw(svg, 144))
                    candidate["svg"] = svg
                except Exception as error:  # noqa: BLE001
                    candidate["failure"] = f"render: {type(error).__name__}: {str(error)[:60]}"
            else:
                candidate["failure"] = info.get("failure", "no drawing")
            job.candidates.append(candidate)
        scored = [c for c in job.candidates if c["error"] is not None]
        if not scored:
            raise RuntimeError("no candidate could be drawn")
        best = min(scored, key=lambda c: c["error"])
        projected, snap = to_project_svg(best["svg"], self.codec.palette)
        try:
            program, parse_info = parse_into_codec(projected, self.codec, self.slots)
        except Exception as error:  # noqa: BLE001
            program, parse_info = None, {"failure": f"parse: {type(error).__name__}"}
        project_svg = (
            serialize_packed_svg(program, self.codec, self.slots).decode()
            if program is not None
            else projected.decode()
        )
        job.result = {
            "best": best["index"],
            "error": best["error"],
            "svg": project_svg,
            "codec_valid": program is not None,
            "codec": parse_info,
            "fills_snapped": snap,
            "png": _png_data_url(_render_svg(project_svg, 288)),
        }
        job.state, job.finished = "done", time.time()
        print(
            json.dumps(
                {
                    "job": job.id,
                    "candidates": job.candidates_wanted,
                    "best_error": best["error"],
                    "codec_valid": program is not None,
                    "seconds": round(job.finished - job.started),
                }
            ),
            flush=True,
        )


def _render_svg(svg: str, size: int) -> Image.Image:
    from mojidiff.learning.omnisvg_study import render_raw

    return render_raw(svg, size)


def _png_data_url(image: Image.Image) -> str:
    payload = io.BytesIO()
    image.save(payload, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode()


def decode_upload(data_url: str) -> Image.Image:
    """A PNG or JPEG data URL, capped in size, as an RGB image on white."""

    if "," not in data_url:
        raise ValueError("expected a data URL")
    header, payload = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("expected an image")
    raw = base64.b64decode(payload, validate=True)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("image too large")
    image = Image.open(io.BytesIO(raw))
    image.load()
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


# ------------------------------------------------------------------------ HTTP


class _Handler(http.server.BaseHTTPRequestHandler):
    engine: Engine
    timeout = 30

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/icons":
            query = parse_qs(url.query).get("q", [""])[0].strip().lower()
            icons = [
                icon
                for icon in self.engine.icons
                if not query
                or query in icon["annotation"].lower()
                or query in icon["hexcode"].lower()
            ]
            self._json(
                200,
                {
                    "icons": icons[:120],
                    "total": len(icons),
                    "held_out": sorted(self.engine.held_out),
                },
            )
        elif url.path.startswith("/render/"):
            name = url.path[len("/render/") :]
            path = RENDER_CACHE / name
            if not name.endswith(".png") or "/" in name or ".." in name or not path.is_file():
                self._json(404, {"error": "no such render"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/palette":
            self._json(200, {"palette": list(self.engine.codec.palette)})
        elif url.path.startswith("/jobs/"):
            view = self.engine.view(url.path[len("/jobs/") :])
            self._json(200 if view else 404, view or {"error": "no such job"})
        elif url.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "queued": len(self.engine.queue),
                    "load_seconds": round(self.engine.load_seconds),
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/jobs":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES + 4096:
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            image = decode_upload(str(payload["image"]))
            candidates = max(1, min(MAX_CANDIDATES, int(payload.get("candidates", 6))))
            seed = int(payload.get("seed", 0)) % 1_000_000
            job = self.engine.submit(image, candidates, seed)
        except (KeyError, ValueError, TypeError) as error:
            self._json(400, {"error": str(error)[:120]})
            return
        except RuntimeError as error:
            self._json(503, {"error": str(error)})
            return
        self._json(202, self.engine.view(job.id))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str, port: int, device: str = "cuda") -> None:
    engine = Engine(device)
    handler = type("Handler", (_Handler,), {"engine": engine})
    with _Server((host, port), functools.partial(handler)) as server:
        print(
            f"re-vectorise demo on http://{host}:{port} - model loaded in "
            f"{engine.load_seconds:.0f} s, {len(engine.icons)} icons",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever()
