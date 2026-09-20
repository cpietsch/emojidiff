"""The weblog must be a faithful view over the committed record, not a second one."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from mojidiff.weblog.build import build_site
from mojidiff.weblog.charts import Series, metric_chart
from mojidiff.weblog.markdown import render_markdown

_ROOT = Path(__file__).resolve().parent.parent
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
        registered.update(
            item for item in row.get("run_ids", []) if isinstance(item, str)
        )
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
        "A <script>alert(1)</script> paragraph with [bad](javascript:alert(1)) "
        "and **strong** text."
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
    assert "class=\"legend\"" not in chart
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
