"""The browser port (web/) and the Python reference data it is tested against."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from mojidiff.learning.autoregressive import (
    NONE,
    PATH_FIELDS,
    PATH_STRIDE,
    SequenceLayout,
    legal_mask,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import ModelConfig, RenderToProgram, path_major_positions
from mojidiff.learning.web_reference import (
    COMMITTED_SPLITS,
    FIXTURE_SPLITS,
    WEB_FIXTURE_CACHE,
    CapturedCall,
    capture_greedy,
    check_calls_against_walk,
    check_fixture_destination,
    codec_constants,
    decode_walk,
    fixture_counts,
    full_fixture_dir,
    mask_ranges,
    mask_table,
    program_entry,
    program_svg,
    select_splits,
    token_planes,
    walk_digest,
    walk_record,
)
from mojidiff.representation.packed import PackedTensorProgram

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_REPO = Path(__file__).resolve().parents[1]
_WEB = _REPO / "web"
_FIXTURES = _WEB / "test" / "fixtures"
_FIXTURE_FILES = ("programs.json.gz", "tokens.bin.gz", "walks.json.gz")

Fixtures = tuple[dict[str, Any], np.ndarray, dict[str, Any]]


@pytest.fixture(scope="module")
def layout() -> SequenceLayout:
    config = load_openmoji_pilot_config(_CONFIG)
    return SequenceLayout(
        codec=_selected_codec(config), total_segment_slots=config.total_segment_slots
    )


@pytest.fixture(scope="module")
def template(layout: SequenceLayout) -> PackedTensorProgram:
    config = load_openmoji_pilot_config(_CONFIG)
    by_split, _, _ = load_pilot_index(config)
    return _load_program(by_split["primary/train"][0], config, layout.codec)


def _load_fixtures(directory: Path) -> Fixtures:
    meta = json.loads(gzip.decompress((directory / "programs.json.gz").read_bytes()))
    walks = json.loads(gzip.decompress((directory / "walks.json.gz").read_bytes()))
    planes = np.frombuffer(gzip.decompress((directory / "tokens.bin.gz").read_bytes()), np.uint8)
    count, length = len(meta["programs"]), meta["length"]
    assert planes.size == 2 * count * length
    low, high = planes[: count * length], planes[count * length :]
    tokens = (low.astype(np.int64) | (high.astype(np.int64) << 8)).reshape(count, length)
    return meta, tokens, walks


def _full_fixture_dir() -> Path | None:
    """The set with every corpus split for the committed set's checkpoint, if it was written."""

    meta = json.loads(gzip.decompress((_FIXTURES / "programs.json.gz").read_bytes()))
    directory = full_fixture_dir(meta["run_id"])
    return directory if all((directory / name).is_file() for name in _FIXTURE_FILES) else None


@pytest.fixture(scope="module")
def committed() -> Fixtures:
    return _load_fixtures(_FIXTURES)


@pytest.fixture(scope="module")
def full() -> Fixtures:
    directory = _full_fixture_dir()
    if directory is None:
        pytest.skip(f"no full fixture set under {WEB_FIXTURE_CACHE} (export_web_fixtures.py)")
    return _load_fixtures(directory)


@pytest.fixture(params=["committed", "full"])
def fixture_set(request: pytest.FixtureRequest) -> Fixtures:
    value: Fixtures = request.getfixturevalue(request.param)
    return value


def test_codec_json_is_what_the_exporter_writes(layout: SequenceLayout) -> None:
    written = json.loads((_WEB / "src" / "codec.json").read_text())
    assert written == json.loads(json.dumps(codec_constants(layout)))
    assert written["vocabulary"] == 418 and written["length"] == 1376


def test_token_planes_and_mask_ranges_round_trip() -> None:
    tokens = np.array([[0, 1, 300, 65535], [417, 2, 0, 256]])
    packed = token_planes(tokens)
    low, high = np.frombuffer(packed[:8], np.uint8), np.frombuffer(packed[8:], np.uint8)
    assert np.array_equal((low.astype(np.int64) | high.astype(np.int64) << 8).reshape(2, 4), tokens)
    with pytest.raises(ValueError):
        token_planes(np.array([[70000]]))
    mask = np.zeros(10, dtype=bool)
    mask[[0, 2, 3, 4, 9]] = True
    assert mask_ranges(mask) == [[0, 1], [2, 5], [9, 10]]
    table = {np.packbits(mask).tobytes(): 0}
    assert mask_table(table, 10) == [[[0, 1], [2, 5], [9, 10]]]


def test_split_selection_and_the_repository_guard(tmp_path: Path) -> None:
    assert select_splits(["primary/validation"]) == COMMITTED_SPLITS
    assert select_splits(["primary/train", "primary/validation", "primary/train"]) == (
        "primary/validation",
        "primary/train",
    )
    assert select_splits(["all"]) == select_splits(list(reversed(FIXTURE_SPLITS)))
    assert select_splits(["all"]) == FIXTURE_SPLITS
    for names in ([], ["primary/extra"]):
        with pytest.raises(ValueError):
            select_splits(names)

    check_fixture_destination(_FIXTURES, COMMITTED_SPLITS, _REPO)
    check_fixture_destination(tmp_path / "full", FIXTURE_SPLITS, _REPO)
    for out in (_FIXTURES, _REPO / "web" / "other", _REPO / "web" / ".." / "tmp-fixtures"):
        with pytest.raises(ValueError, match="primary/train programs stay out"):
            check_fixture_destination(out, ("primary/validation", "primary/train"), _REPO)
    assert full_fixture_dir("some-run") == WEB_FIXTURE_CACHE / "some-run" / "full"


def test_committed_fixtures_hold_the_validation_split_only(committed: Fixtures) -> None:
    """Datasets stay out of the repository: the committed set holds the validation programs
    (every validation icon, the icons the model decodes come from) and no train or test
    program, and its recorded counts are what it holds."""

    meta, tokens, walks = committed
    programs = meta["programs"]
    assert meta["splits"] == list(COMMITTED_SPLITS)
    assert {program["split"] for program in programs} == set(COMMITTED_SPLITS)
    corpus = [p["hexcode"] for p in programs if p["source"] == "corpus"]
    greedy = [p["hexcode"] for p in programs if p["source"] == "greedy"]
    assert len(corpus) == len(set(corpus)) and corpus == greedy
    expected = fixture_counts(programs, len(walks["walks"]), len(walks["schedules"]))
    assert meta["counts"] == walks["counts"] == expected
    assert expected["corpus_splits"] == {"primary/validation": len(corpus)}
    assert tokens.shape == (len(programs), meta["length"])


def test_full_fixtures_extend_the_committed_ones(committed: Fixtures, full: Fixtures) -> None:
    """The cache's full set is the committed set plus the test and train programs: the same
    checkpoint and dataset, and the same entries, tokens, walks and schedules for every
    validation program and decode."""

    (meta, tokens, walks), (full_meta, full_tokens, full_walks) = committed, full
    keys = ("dataset_sha256", "checkpoint_sha256", "run_id", "torch", "length", "vocabulary")
    assert {k: full_meta[k] for k in keys} == {k: meta[k] for k in keys}
    assert full_meta["splits"] == list(FIXTURE_SPLITS)
    programs = full_meta["programs"]
    counts = fixture_counts(programs, len(full_walks["walks"]), len(full_walks["schedules"]))
    assert full_meta["counts"] == full_walks["counts"] == counts
    assert set(counts["corpus_splits"]) == set(FIXTURE_SPLITS)
    shared = [i for i, p in enumerate(programs) if p["split"] in COMMITTED_SPLITS]
    assert [programs[i] for i in shared] == meta["programs"]
    assert np.array_equal(full_tokens[shared], tokens)
    index = {full_index: i for i, full_index in enumerate(shared)}
    assert full_walks["masks"] == walks["masks"]
    for name in ("walks", "schedules"):
        moved = [{**w, "program": index[w["program"]]} for w in full_walks[name]]
        assert moved == walks[name], name


def test_decode_walk_refuses_a_program_outside_the_grammar(
    committed: Fixtures, layout: SequenceLayout
) -> None:
    _, tokens, _ = committed
    program = torch.from_numpy(tokens[0].copy())
    program[0] = 0  # first path absent, yet its fields and segments are not PAD
    with pytest.raises(ValueError, match="not legal"):
        decode_walk(program, layout)


def _tiny_model(layout: SequenceLayout, order: str = "path") -> RenderToProgram:
    torch.manual_seed(0)
    config = ModelConfig(
        image_size=16,
        d_model=32,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        feedforward=64,
        metric=True,
        fourier=4,
        order=order,
    )
    return RenderToProgram(layout, config).eval()


@pytest.mark.parametrize("temperature", [0.0, 1.0], ids=["greedy", "sampled"])
def test_captured_greedy_calls_are_the_walks_schedule(
    layout: SequenceLayout, temperature: float
) -> None:
    """On a random model, every greedy_decode call is a free step of the walk, fed with
    the walk's ids rows for every step queued since the previous call."""

    model = _tiny_model(layout)
    image = torch.randint(
        0, 256, (16, 16, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(1)
    )
    generator = torch.Generator().manual_seed(2) if temperature > 0 else None
    program, calls, model_calls = capture_greedy(
        model, image, temperature=temperature, generator=generator
    )
    assert "step_inputs" not in model.__dict__
    walk = decode_walk(program, layout)
    check_calls_against_walk(walk, calls, model_calls)
    assert model_calls == walk.model_calls > 0
    assert any(call.end - call.start > 1 for call in calls)  # forced steps were queued
    broken = [CapturedCall(c.start, c.end, c.ids, c.target + 1) for c in calls]
    with pytest.raises(ValueError, match="target"):
        check_calls_against_walk(walk, broken, model_calls)
    with pytest.raises(ValueError, match="calls recorded"):
        check_calls_against_walk(walk, calls[:-1], model_calls)


def test_capture_needs_a_metric_path_order_model(layout: SequenceLayout) -> None:
    with pytest.raises(ValueError, match="path-order"):
        capture_greedy(
            _tiny_model(layout, order="layout"), torch.zeros((16, 16, 3), dtype=torch.uint8)
        )


def test_fixtures_match_the_python_reference(
    fixture_set: Fixtures, layout: SequenceLayout, template: PackedTensorProgram
) -> None:
    """A spread of programs of the committed set (and of the cache's full set, when it was
    written) re-derived from today's Python: a change to the grammar, labels or serializer
    must come with regenerated fixtures."""

    meta, tokens, walks = fixture_set
    programs = meta["programs"]
    assert meta["length"] == layout.length and meta["vocabulary"] == layout.vocabulary
    assert {p["source"] for p in programs} == {"corpus", "greedy", "sample"}
    chosen: list[int] = []
    for source in ("corpus", "greedy", "sample"):
        indices = [i for i, p in enumerate(programs) if p["source"] == source]
        chosen += indices[:: max(len(indices) // 4, 1)][:4]
    for index in chosen:
        expected = programs[index]
        entry = program_entry({}, torch.from_numpy(tokens[index]), template, layout)
        for key in ("walk_sha256", "model_calls", "svg_sha256", "svg_bytes"):
            assert entry[key] == expected[key], (index, key)
        assert expected.get("recorded_model_calls", entry["model_calls"]) == entry["model_calls"]

    reference = walks["walks"][0]
    walk = decode_walk(torch.from_numpy(tokens[reference["program"]]), layout)
    table: dict[bytes, int] = {}
    record = walk_record(walk, table)
    masks = mask_table(table, layout.vocabulary)
    assert [masks[i] for i in record.pop("mask")] == [walks["masks"][i] for i in reference["mask"]]
    assert record == {k: v for k, v in reference.items() if k not in ("program", "mask")}
    assert walk_digest(walk) == programs[reference["program"]]["walk_sha256"]


def _random_legal_program(
    layout: SequenceLayout, rng: np.random.Generator, short_paths: bool
) -> Tensor:
    """A program with a uniformly random legal token at every step of the path-major walk.

    With `short_paths` every path takes 1-3 segments, so all paths are active and the
    layers run out and repeat; otherwise path lengths are uniform, and programs often take
    every segment slot."""

    decoded = torch.zeros(layout.length, dtype=torch.long)
    for position in path_major_positions(decoded, layout):
        legal = np.flatnonzero(legal_mask(position, decoded, layout).numpy())
        if short_paths and layout.path_length_index(position) is not None:
            small = legal[(legal >= 1) & (legal <= 3)]
            legal = small if small.size else legal
        decoded[position] = int(rng.choice(legal))
    return decoded


def test_random_legal_programs_match_python(
    layout: SequenceLayout, template: PackedTensorProgram, tmp_path: Path
) -> None:
    """Programs the committed fixtures never reach, checked live against Node.

    No fixture program has a stroke opacity below 1, and the fixtures' SVG check never
    runs the JavaScript that writes one. Random legal programs reach it, all paths active
    and every segment slot taken; Python walks and serialises each, and
    web/test/crosscheck.mjs must give the same walk digest, call count and SVG text.
    """

    node = _node()
    if node is None:
        pytest.skip("Node.js 20 or newer is not available (set MOJIDIFF_NODE)")
    programs = [
        _random_legal_program(layout, np.random.default_rng(seed), short_paths=seed % 2 == 1)
        for seed in range(24)
    ]
    fields = np.stack(
        [p[: layout.path_positions].numpy().reshape(-1, PATH_STRIDE) for p in programs]
    )
    at = {name: PATH_FIELDS.index(name) for name in PATH_FIELDS}
    one = layout.codec.opacities.index(1.0) + 2
    active = fields[..., at["path_length"]] > 0
    stroked = active & (fields[..., at["stroke"]] != NONE)
    assert (stroked & (fields[..., at["stroke_opacity"]] != one)).any()
    assert (stroked & (fields[..., at["dash_pattern"]] != NONE)).any()
    assert (active & (fields[..., at["opacity"]] != one)).any()
    assert active.all(axis=1).any()
    assert (fields[..., at["path_length"]].sum(axis=1) == layout.total_segment_slots).any()
    assert (active[:, 1:] & (fields[:, 1:, at["layer"]] == fields[:, :-1, at["layer"]])).any()

    source = tmp_path / "programs.json"
    source.write_text(json.dumps([p.tolist() for p in programs]))
    result = subprocess.run(
        [node, "test/crosscheck.mjs", str(source)],
        cwd=_WEB,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    javascript = json.loads(result.stdout)
    assert len(javascript) == len(programs)
    for seed, (program, js) in enumerate(zip(programs, javascript, strict=True)):
        walk = decode_walk(program, layout)
        svg = program_svg(program, template, layout)
        assert svg is not None, seed
        assert js["walk_sha256"] == walk_digest(walk), seed
        assert js["model_calls"] == walk.model_calls, seed
        assert js["svg"] == svg, seed


def _node() -> str | None:
    node = os.environ.get("MOJIDIFF_NODE") or shutil.which("node")
    if node is None:
        return None
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    match = re.match(r"v(\d+)\.", version.stdout.strip())
    return node if match and int(match.group(1)) >= 20 else None


def _node_env(fixtures: Path | None) -> dict[str, str]:
    """The environment for Node: MOJIDIFF_WEB_FIXTURES set to `fixtures`, or unset."""

    env = {k: v for k, v in os.environ.items() if k != "MOJIDIFF_WEB_FIXTURES"}
    if fixtures is not None:
        env["MOJIDIFF_WEB_FIXTURES"] = str(fixtures)
    return env


def _run_node_suite(node: str, fixtures: Path | None) -> str:
    files = sorted(str(p.relative_to(_WEB)) for p in (_WEB / "test").glob("*.test.mjs"))
    assert len(files) >= 3
    result = subprocess.run(
        [node, "--test", *files],
        cwd=_WEB,
        env=_node_env(fixtures),
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
    assert re.search(r"^# fail 0$", result.stdout, re.MULTILINE), result.stdout[-3000:]
    return result.stdout


def _suite_read(stdout: str, directory: Path, programs: int) -> bool:
    """Whether the suite's diagnostics show it read `programs` programs from `directory`."""

    pattern = rf"^# fixtures {re.escape(str(directory))}/?: {programs} programs "
    return re.search(pattern, stdout, re.MULTILINE) is not None


def test_node_suite_passes(committed: Fixtures) -> None:
    """web/test/*.test.mjs: the JavaScript grammar, decode loop and codec against the
    committed fixtures (MOJIDIFF_WEB_FIXTURES unset)."""

    node = _node()
    if node is None:
        pytest.skip("Node.js 20 or newer is not available (set MOJIDIFF_NODE)")
    stdout = _run_node_suite(node, None)
    assert _suite_read(stdout, _FIXTURES, committed[0]["counts"]["programs"]), stdout[-3000:]


def test_node_suite_passes_on_the_full_fixtures(full: Fixtures) -> None:
    """The same suite on the cache's full set: every corpus program of every split, at
    every position, as the committed fixtures held before they were cut to validation."""

    node = _node()
    if node is None:
        pytest.skip("Node.js 20 or newer is not available (set MOJIDIFF_NODE)")
    directory = _full_fixture_dir()
    assert directory is not None
    stdout = _run_node_suite(node, directory)
    assert _suite_read(stdout, directory, full[0]["counts"]["programs"]), stdout[-3000:]


def test_node_fixture_directory_follows_the_environment(tmp_path: Path) -> None:
    """helpers.mjs reads MOJIDIFF_WEB_FIXTURES when set, and refuses a path that names no
    directory instead of falling back to the committed set."""

    node = _node()
    if node is None:
        pytest.skip("Node.js 20 or newer is not available (set MOJIDIFF_NODE)")
    script = (
        "const h = await import('./test/helpers.mjs'); const f = h.fixtures(); "
        "console.log(JSON.stringify({dir: f.dir, programs: f.meta.programs.length}));"
    )

    def run(fixtures: Path | None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [node, "--input-type=module", "-e", script],
            cwd=_WEB,
            env=_node_env(fixtures),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    copy = tmp_path / "copy"
    shutil.copytree(_FIXTURES, copy)
    meta = json.loads(gzip.decompress((copy / "programs.json.gz").read_bytes()))
    for fixtures, expected in ((None, _FIXTURES), (copy, copy)):
        result = run(fixtures)
        assert result.returncode == 0, result.stderr[-2000:]
        read = json.loads(result.stdout)
        assert Path(read["dir"]) == expected and read["programs"] == meta["counts"]["programs"]
    result = run(tmp_path / "missing")
    assert result.returncode != 0 and "is not a directory" in result.stderr
