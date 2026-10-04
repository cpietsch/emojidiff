"""The weblog must be a faithful view over the committed record, not a second one."""

from __future__ import annotations

import ipaddress
import json
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from mojidiff.weblog.build import PAGES, build_site
from mojidiff.weblog.charts import Series, metric_chart
from mojidiff.weblog.markdown import render_markdown

_ROOT = Path(__file__).resolve().parent.parent
# 100.64.0.0/10 is the shared address space Tailscale hands out.
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}


class _Nesting(HTMLParser):
    """Enough of a parser to catch an unbalanced tag in generated markup."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            while self.stack and self.stack.pop() != tag:
                pass
        else:
            self.stack.pop()


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build_site(_ROOT, out)
    return out


def test_every_generated_page_is_well_formed(site: Path) -> None:
    pages = sorted(site.rglob("*.html"))
    assert pages, "the build produced no pages"
    for page in pages:
        checker = _Nesting()
        checker.feed(page.read_text())
        assert not checker.errors, f"{page.name}: {checker.errors[:3]}"
        assert not checker.stack, f"{page.name}: unclosed {checker.stack[:3]}"


def test_the_site_covers_every_registered_run(site: Path) -> None:
    registered: set[str] = set()
    for line in (_ROOT / "state" / "runs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("run_id"), str):
            registered.add(row["run_id"])
        # A batch row names several runs at once under `run_ids`.
        registered.update(item for item in row.get("run_ids", []) if isinstance(item, str))
    registered.update(path.parent.name for path in (_ROOT / "runs").glob("*/run.yaml"))
    built = {path.stem for path in (site / "run").glob("*.html")}
    # A run that exists only in the append-only registry still gets a page, and a run
    # directory with no registry row still gets one. Dropping either would let the
    # weblog show a tidier history than the one actually recorded.
    assert registered <= built


def test_failed_and_falsified_runs_stay_visible(site: Path) -> None:
    index = (site / "runs.html").read_text()
    states = {
        json.loads(line).get("state")
        for line in (_ROOT / "state" / "runs.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert "failed" in states, "fixture assumption: the registry records failures"
    assert "badge-failed" in index


def test_generated_markup_escapes_hostile_source_text(tmp_path: Path) -> None:
    rendered = render_markdown(
        "A <script>alert(1)</script> paragraph with [bad](javascript:alert(1)) and **strong** text."
    )
    assert "<script>" not in rendered
    assert "javascript:" not in rendered.split("href=")[0] or "href=" not in rendered
    assert "<strong>strong</strong>" in rendered


def test_markdown_subset_renders_the_structures_the_repository_uses() -> None:
    rendered = render_markdown(
        "# Title\n\n"
        "Body text with `code`.\n\n"
        "| a | b |\n| --- | ---: |\n| 1 | 2 |\n\n"
        "- first\n- second\n\n"
        "> quoted\n"
    )
    for fragment in ("<h1>", "<table>", "<th>a</th>", "<td>1</td>", "<ul>", "<blockquote>"):
        assert fragment in rendered, fragment


def test_a_chart_ships_a_table_view_and_a_legend_for_multiple_series() -> None:
    chart = metric_chart(
        "Held-out accuracy",
        [
            Series("aggregate", ((0.0, 0.1), (60.0, 0.2))),
            Series("changed", ((0.0, 0.05), (60.0, 0.09))),
        ],
        x_label="optimizer step",
        y_label="accuracy",
        y_zero=True,
    )
    # Identity is never carried by colour alone: a legend names each series and the
    # table view repeats the exact numbers.
    assert "legend" in chart and "aggregate" in chart
    assert "Table view" in chart and "0.0900" in chart


def test_a_single_series_chart_omits_the_legend_box() -> None:
    chart = metric_chart(
        "Held-out loss",
        [Series("held-out loss", ((0.0, 9.0), (60.0, 7.5)))],
        x_label="optimizer step",
        y_label="loss",
    )
    assert 'class="legend"' not in chart
    assert "Table view" in chart


def test_a_chart_refuses_more_series_than_the_validated_palette_supports() -> None:
    with pytest.raises(ValueError, match="at most three series"):
        metric_chart(
            "too many",
            [Series(f"s{index}", ((0.0, 0.0), (1.0, 1.0))) for index in range(4)],
            x_label="step",
            y_label="value",
        )


def test_a_registry_only_run_still_shows_its_evidence(tmp_path: Path) -> None:
    """A run with no run.yaml must still reach its summary, metrics and renders.

    Gate I's runs are registered in `state/runs.jsonl` alone. Before this, such a run
    rendered as a bare row with nothing attached and a caption pointing at a run.yaml
    that does not exist - a page that looked like evidence and carried none, which is
    the one failure mode this site cannot have.
    """

    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    (root / "reports" / "demo").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "configs").mkdir()
    (root / "configs" / "demo.yaml").write_text("report_root: reports/demo\n")
    (root / "reports" / "demo" / "summary.json").write_text('{"held_out_nll_per_free_token": 2.5}')
    (root / "reports" / "demo" / "metrics.jsonl").write_text(
        '{"step": 1, "held_out_nll": 3.0, "marginal_nll": 4.0}\n'
        '{"step": 2, "held_out_nll": 2.5, "marginal_nll": 4.0}\n'
    )
    (root / "reports" / "findings.md").write_text("# Findings\n")
    (root / "state" / "CURRENT.md").write_text("# State\n")
    (root / "state" / "gates.yaml").write_text("gates: []\n")
    (root / "state" / "runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "demo-run",
                "state": "completed",
                "timestamp": "2026-01-01T00:00:00Z",
                "config": "configs/demo.yaml",
                "selected_step": 2,
            }
        )
        + "\n"
    )

    build_site(root, tmp_path / "site")
    page = (tmp_path / "site" / "run" / "demo-run.html").read_text()
    assert "held_out_nll_per_free_token" in page
    assert "position-marginal floor" in page, "the floor must be plotted beside the model"
    assert "runs/demo-run/run.yaml" not in page, "it must not cite a file that is absent"
    assert "append-only registry" in page


def test_a_superseded_attempt_keeps_its_own_evidence(tmp_path: Path) -> None:
    """A falsified attempt's artifacts move aside; its page must show them, not the rerun's.

    Gate L's first overfit attempt was falsified by a harness defect, its report
    directory renamed, and the fixed rerun registered against the same config. Without
    this the attempt's page would resolve the config's report_root and present the
    rerun's numbers as its own.
    """

    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    (root / "reports" / "demo").mkdir(parents=True)
    (root / "reports" / "demo-attempt1").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "configs").mkdir()
    (root / "configs" / "demo.yaml").write_text("report_root: reports/demo\n")
    (root / "reports" / "demo" / "summary.json").write_text('{"verdict": "rerun-numbers"}')
    (root / "reports" / "demo-attempt1" / "summary.json").write_text(
        '{"verdict": "attempt-numbers"}'
    )
    (root / "reports" / "findings.md").write_text("# Findings\n")
    (root / "state" / "CURRENT.md").write_text("# State\n")
    (root / "state" / "gates.yaml").write_text("gates: []\n")
    (root / "state" / "runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "demo-attempt",
                "state": "completed",
                "timestamp": "2026-01-01T00:00:00Z",
                "config": "configs/demo.yaml",
                "artifacts": {"report_root": "reports/demo-attempt1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "run_id": "demo-rerun",
                "state": "completed",
                "timestamp": "2026-01-02T00:00:00Z",
                "config": "configs/demo.yaml",
            }
        )
        + "\n"
    )

    build_site(root, tmp_path / "site")
    attempt = (tmp_path / "site" / "run" / "demo-attempt.html").read_text()
    rerun = (tmp_path / "site" / "run" / "demo-rerun.html").read_text()
    assert "attempt-numbers" in attempt and "rerun-numbers" not in attempt
    assert "rerun-numbers" in rerun and "attempt-numbers" not in rerun


class _References(HTMLParser):
    """Collect every href and src a page asks the browser to follow or load."""

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.refs.extend(value for key, value in attrs if key in {"href", "src"} and value)


def _private_host(host: str) -> bool:
    if host in {"localhost", "0.0.0.0"} or host.endswith(".ts.net"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address in _TAILNET


def test_the_pages_build_works_under_a_sub_path(tmp_path: Path) -> None:
    """GitHub Pages serves the site at /emojidiff/, to readers outside the tailnet.

    A root-absolute link would resolve to cpietsch.github.io/ and miss the site, and a
    link to a tailnet address or localhost is dead for every public reader.
    """

    site = tmp_path / "site"
    build_site(_ROOT, site, PAGES)
    pages = sorted(site.rglob("*.html"))
    assert pages
    for page in pages:
        parser = _References()
        parser.feed(page.read_text())
        for ref in parser.refs:
            parts = urlsplit(ref)
            where = f"{page.relative_to(site)}: {ref}"
            if parts.scheme or parts.netloc:
                assert parts.scheme in {"http", "https", "mailto"}, where
                assert not _private_host(parts.hostname or ""), where
                continue
            assert not ref.startswith("/"), f"root-absolute link breaks under a sub-path: {where}"
            if not parts.path:
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            # Committed prose may cite a repository path the site does not mirror; any
            # link that lands inside the site, which is every link the build generates,
            # must name a file the build wrote.
            if site.resolve() in target.parents:
                assert target.is_file(), f"dangling link: {where}"

    index = (site / "index.html").read_text()
    assert "Tailscale address only" not in index
    for url in (
        "https://huggingface.co/chrispie/mojidiff-checkpoints",
        "https://openmoji.org/",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/licenses/by/4.0/",
    ):
        assert f"href='{url}'" in index, url
    # The public copy names both artwork sources and says gtc was only where it began.
    assert "All emojis designed by OpenMoji" in index and "Twemoji" in index
    assert "only a starting point and an inspiration" in index


def test_the_local_build_keeps_its_tailnet_footer(site: Path) -> None:
    index = (site / "index.html").read_text()
    assert "Tailscale address only" in index
    assert "huggingface.co/spaces" not in index
    assert "starting point" not in index


def test_the_front_page_carries_the_newest_decisions(site: Path) -> None:
    """The newest entry of the trail must be reachable without reading it end to end.

    The trail is written oldest-first and is now many screens long, so the entry that
    says what the project currently believes was the hardest thing on the site to find.
    """

    index = (site / "index.html").read_text()
    findings = (_ROOT / "reports" / "findings.md").read_text()
    titles = [line[3:].strip() for line in findings.splitlines() if line.startswith("## ")]
    assert titles, "the trail has no entries to surface"
    assert escape(titles[-1]) in index, "the newest decision is not on the front page"
    assert escape(titles[0]) not in index, "the oldest decision should not lead the page"
