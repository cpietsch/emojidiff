#!/usr/bin/env python3
"""Serve the generated research weblog on this machine's Tailscale address.

The site is static and read-only. The server binds to one explicit address - the
machine's Tailscale IP by default - rather than to every interface, so the weblog is
reachable over the tailnet without also being exposed on the container's bridge
network. It serves only files under the output directory.

    python scripts/serve_weblog.py --rebuild

Stop it with Ctrl-C, or run it under tmux for a long-lived view.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

DEFAULT_HOST = "100.69.189.78"
DEFAULT_PORT = 8787

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.weblog.build import build_site  # noqa: E402


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler with a quiet, single-line log."""

    # A client that connects and never sends - a browser's speculative pre-connect, a
    # dropped tailnet link - must not hold a handler thread forever.
    timeout = 10

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} {format % args}\n")


class _Server(socketserver.ThreadingTCPServer):
    """One thread per connection.

    The plain TCPServer is single-threaded, and a single connection that opened and
    sent nothing hung the whole site for everyone else - which is how the weblog went
    dark on 2026-09-21 while its process was alive and listening. Threads make one
    stalled client cost one thread, and the handler timeout reclaims it.
    """

    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT, help="repository root")
    parser.add_argument("--site", type=Path, default=_REPO_ROOT / "site", help="site directory")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--rebuild", action="store_true", help="regenerate the site first")
    args = parser.parse_args()

    if args.rebuild or not (args.site / "index.html").is_file():
        manifest = build_site(args.root.resolve(), args.site.resolve())
        print(f"built {manifest['pages']} pages, {manifest['assets']} assets", file=sys.stderr)

    handler = functools.partial(_Handler, directory=str(args.site.resolve()))
    with _Server((args.host, args.port), handler) as server:
        print(f"serving {args.site} on http://{args.host}:{args.port}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
