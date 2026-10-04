#!/usr/bin/env python3
"""Serve the model gallery - every model of this phase, side by side - on this
machine's Tailscale address.

    python scripts/serve_gallery.py [--host 100.69.189.78] [--port 8791]
        [--registry configs/gallery/models.yaml] [--ratings reports/gallery/ratings.jsonl]
        [--no-ratings]

Every model the registry file lists loads at startup (the built-in
`mojidiff.gallery.server.MODELS` when the default file is absent); a missing checkpoint
or an unknown latent backend stops it before it binds. Ratings from the page append to
the ratings file; `--no-ratings` (the public Hugging Face Space) refuses them and hides
the page's rating controls. Run it under tmux (`gallery`) for a long-lived view; stop
it with Ctrl-C, and restart it to pick up an edited registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.gallery.server import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    RATINGS,
    REGISTRY,
    serve,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=f"models to serve (default {REGISTRY}, or the built-in list if that file is absent)",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=RATINGS,
        help=f"ratings file, appended to (default {RATINGS})",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="record no ratings: POST /rating and GET /ratings answer 503, the page hides them",
    )
    args = parser.parse_args()
    serve(args.host, args.port, args.registry, None if args.no_ratings else args.ratings)


if __name__ == "__main__":
    main()
