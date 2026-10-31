#!/usr/bin/env python3
"""Write the browser port's constants and test fixtures from the Python reference code.

    python scripts/export_web_fixtures.py codec
        web/src/codec.json: codec and layout constants for web/src/{codec,grammar}.mjs.

    python scripts/export_web_fixtures.py fixtures --checkpoint <run>/best.pt \
            [--splits primary/validation] [--out web/test/fixtures] [--workers 16]
        <out>/programs.json.gz: per program, the SHA-256 (128 bits) of its decode walk
            (mask, forced flag, variant-B ids and target at every step) and of the SVG
            serialize_packed_svg writes, and its batch-1 call count. Programs: the corpus
            programs of --splits (plain, in validation, test, train order), the
            checkpoint's CPU greedy decodes of validation icons, and its CPU samples at
            --temperature for the first --sampled icons. The header records the splits and
            the set's counts (web_reference.fixture_counts), which the tests check against.
        <out>/tokens.bin.gz: those programs' tokens, (N, L) in stored positions, as a
            low-byte plane then a high-byte plane.
        <out>/walks.json.gz: the full per-step walk of a subset, masks as indices into a
            table of distinct masks, plus each model decode's recorded greedy_decode call
            schedule (checked against its walk before writing).

Datasets stay out of the repository: the committed set (the default, web/test/fixtures) holds
the validation programs only, and the script refuses to write train or test programs inside
the repository. The set with every split goes to the cache, where
MOJIDIFF_WEB_FIXTURES=<dir> points the Node suite at it:

    python scripts/export_web_fixtures.py fixtures --checkpoint <run>/best.pt \
        --splits all --out /home/dev/.cache/mojidiff/web-fixtures/<run_id>/full

Run it from a checkout that has data/raw (the pilot config's paths are relative), e.g.
/home/dev/workspace/mojidiff; it imports the code of the checkout the script lives in, and
the default --out is that checkout's web/test/fixtures. Decoding runs on the CPU (about
1.1 s per icon, 7.5 min for the default 371 decodes), so the shared RTX 4080 is not used.
The model decodes depend on torch's CPU arithmetic and thread count: regenerate both sets
after any change to the grammar, labels, serializer, checkpoint or torch build, and commit
the validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mojidiff.learning.render2svg import (  # noqa: E402
    ModelConfig,
    RenderToProgram,
    load_corpus,
)
from mojidiff.learning.web_reference import (  # noqa: E402
    COMMITTED_SPLITS,
    FIXTURE_FORMAT,
    FIXTURE_SPLITS,
    capture_greedy,
    check_calls_against_walk,
    check_fixture_destination,
    codec_constants,
    decode_walk,
    fixture_counts,
    mask_table,
    program_entries,
    select_splits,
    token_planes,
    walk_record,
    write_gz,
    write_json_gz,
)

PILOT_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
CODEC_JSON = _REPO_ROOT / "web" / "src" / "codec.json"
FIXTURES = _REPO_ROOT / "web" / "test" / "fixtures"


def write_codec(layout: Any) -> None:
    CODEC_JSON.parent.mkdir(parents=True, exist_ok=True)
    CODEC_JSON.write_text(json.dumps(codec_constants(layout), indent=1) + "\n")
    print(json.dumps({"wrote": str(CODEC_JSON), "bytes": CODEC_JSON.stat().st_size}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("what", choices=("codec", "fixtures"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(COMMITTED_SPLITS),
        choices=(*FIXTURE_SPLITS, "all"),
        help="corpus splits whose programs to include (default: primary/validation)",
    )
    parser.add_argument(
        "--out", type=Path, default=FIXTURES, help=f"fixture directory (default: {FIXTURES})"
    )
    parser.add_argument(
        "--greedy", type=int, default=339, help="validation icons to decode greedily"
    )
    parser.add_argument("--sampled", type=int, default=32, help="validation icons to sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--detail-corpus", type=int, default=16, help="validation programs walked in full"
    )
    parser.add_argument(
        "--detail-greedy", type=int, default=16, help="greedy decodes walked in full"
    )
    parser.add_argument("--detail-sampled", type=int, default=8, help="samples walked in full")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16, help="torch CPU threads for decoding")
    args = parser.parse_args()
    splits = select_splits(args.splits)
    out = args.out.resolve()
    if args.what == "fixtures":
        try:
            check_fixture_destination(out, splits, _REPO_ROOT)
        except ValueError as error:
            parser.error(str(error))

    from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index

    plain, _, layout, pilot, dataset_hash = load_corpus(PILOT_CONFIG, 144)
    if args.what == "codec":
        write_codec(layout)
        return
    if args.checkpoint is None:
        parser.error("fixtures needs --checkpoint")
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, layout.codec)

    items: list[tuple[dict[str, Any], np.ndarray]] = []
    for split in splits:
        data = plain[split]
        for hexcode, tokens in zip(data.hexcodes, data.tokens, strict=True):
            meta: dict[str, Any] = {"source": "corpus", "split": split, "hexcode": hexcode}
            items.append((meta, tokens.numpy().astype(np.int64)))

    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu")
    model = RenderToProgram(layout, ModelConfig(**state["config"])).eval()
    model.load_state_dict(state["model"])
    validation = plain["primary/validation"]
    schedules: list[dict[str, Any]] = []
    decode_seconds: list[float] = []
    plan = [("greedy", i, 0.0) for i in range(min(args.greedy, len(validation.hexcodes)))]
    plan += [
        ("sample", i, args.temperature) for i in range(min(args.sampled, len(validation.hexcodes)))
    ]
    for kind, index, temperature in plan:
        generator = torch.Generator().manual_seed(index) if temperature > 0 else None
        started = time.perf_counter()
        program, calls, model_calls = capture_greedy(
            model, validation.images[index], temperature=temperature, generator=generator
        )
        decode_seconds.append(time.perf_counter() - started)
        walk = decode_walk(program, layout)
        check_calls_against_walk(walk, calls, model_calls)
        meta = {
            "source": kind,
            "split": "primary/validation",
            "hexcode": validation.hexcodes[index],
        }
        if temperature > 0:
            meta.update(temperature=temperature, seed=index)
        meta["recorded_model_calls"] = model_calls
        items.append((meta, program.numpy().astype(np.int64)))
        schedules.append({"program": len(items) - 1, "calls": [[c.start, c.end] for c in calls]})
        if len(schedules) % 25 == 0:
            print(json.dumps({"decoded": len(schedules), "of": len(plan)}), flush=True)

    started = time.perf_counter()
    entries = program_entries(items, template, layout, workers=args.workers)
    walk_seconds = time.perf_counter() - started
    for entry in entries:
        if (
            "recorded_model_calls" in entry
            and entry["recorded_model_calls"] != entry["model_calls"]
        ):
            raise SystemExit(f"call count differs for {entry['hexcode']}")

    def first(source: str, count: int) -> list[int]:
        chosen = [i for i, (meta, _) in enumerate(items) if meta["source"] == source]
        if source == "corpus":
            chosen = [i for i in chosen if items[i][0]["split"] == "primary/validation"]
        return chosen[:count]

    detail = (
        first("corpus", args.detail_corpus)
        + first("greedy", args.detail_greedy)
        + first("sample", args.detail_sampled)
    )
    table: dict[bytes, int] = {}
    walks = []
    for index in detail:
        record = walk_record(decode_walk(torch.from_numpy(items[index][1]), layout), table)
        walks.append({"program": index, **record})
    detailed = set(detail)
    kept = [s for s in schedules if s["program"] in detailed]

    header = {
        "format": FIXTURE_FORMAT,
        "generated_by": "scripts/export_web_fixtures.py",
        "pilot_config": str(PILOT_CONFIG),
        "splits": list(splits),
        "dataset_sha256": dataset_hash,
        "checkpoint": str(checkpoint),
        "run_id": checkpoint.parent.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": state.get("step"),
        "decode": "render2svg.greedy_decode on the CPU, batch 1, float32",
        "torch": torch.__version__,
        "length": layout.length,
        "vocabulary": layout.vocabulary,
        "counts": fixture_counts(entries, len(walks), len(kept)),
    }
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.stack([np.asarray(entry.pop("tokens")) for entry in entries])
    tokens_bytes = write_gz(out / "tokens.bin.gz", token_planes(tokens))
    programs_bytes = write_json_gz(
        out / "programs.json.gz",
        {
            **header,
            "tokens": "tokens.bin.gz: (N, L) low-byte plane, then high-byte plane",
            "programs": entries,
        },
    )
    walks_bytes = write_json_gz(
        out / "walks.json.gz",
        {
            **header,
            "masks": mask_table(table, layout.vocabulary),
            "walks": walks,
            "schedules": kept,
        },
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "splits": list(splits),
                "counts": header["counts"],
                "distinct_masks": len(table),
                "programs_json_gz_bytes": programs_bytes,
                "tokens_bin_gz_bytes": tokens_bytes,
                "walks_json_gz_bytes": walks_bytes,
                "decode_seconds_median": float(np.median(decode_seconds))
                if decode_seconds
                else None,
                "decode_seconds_total": float(np.sum(decode_seconds)),
                "walk_seconds": walk_seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
