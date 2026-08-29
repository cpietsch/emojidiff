"""Minimal resource-limited CairoSVG worker; invoked only by the isolated parent."""

from __future__ import annotations

import argparse
import resource
import sys

import cairosvg


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--max-svg-bytes", type=int, required=True)
    parser.add_argument("--max-png-bytes", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    args = parser.parse_args()
    if not (
        1 <= args.size <= 512
        and 1 <= args.max_svg_bytes <= 4_000_000
        and 1 <= args.max_png_bytes <= 64_000_000
        and 1 <= args.cpu_seconds <= 60
    ):
        return 2
    memory_bytes = 1_500_000_000
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (args.max_png_bytes, args.max_png_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    source = sys.stdin.buffer.read(args.max_svg_bytes + 1)
    if len(source) > args.max_svg_bytes:
        return 3
    try:
        png = cairosvg.svg2png(
            bytestring=source,
            output_width=args.size,
            output_height=args.size,
            unsafe=False,
        )
    except Exception:
        return 4
    if len(png) > args.max_png_bytes:
        return 5
    sys.stdout.buffer.write(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
