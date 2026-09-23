#!/usr/bin/env python3
"""Serve the live re-vectorise demo on this machine's Tailscale address.

    python scripts/serve_demo.py

Loads OmniSVG 1.1 4B once (about 40 s) and serves one page. Binds to one explicit
address - the Tailscale IP by default - never to every interface. Run it under tmux
(`mojidiff-demo`) for a long-lived view; stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.demo.server import DEFAULT_HOST, DEFAULT_PORT, serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    serve(args.host, args.port, args.device)


if __name__ == "__main__":
    main()
