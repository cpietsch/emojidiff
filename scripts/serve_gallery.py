#!/usr/bin/env python3
"""Serve the model gallery - every model of this phase, side by side - on this
machine's Tailscale address.

    python scripts/serve_gallery.py [--host 100.69.189.78] [--port 8791]

All twelve checkpoints load at startup (see `mojidiff.gallery.server.MODELS`); a
missing one stops it before it binds. Run it under tmux (`gallery`) for a long-lived
view; stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.gallery.server import DEFAULT_HOST, DEFAULT_PORT, serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
