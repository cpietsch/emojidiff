"""Generate the static research weblog from the committed evidence.

The weblog is a *view* over the research record, never a second source of truth.
Every page is derived from files that are already committed and already reviewed:

* `state/CURRENT.md`   - the session handoff and current hypothesis;
* `state/gates.yaml`   - the evidence-gate board;
* `state/runs.jsonl`   - the append-only state-transition log;
* `runs/*/run.yaml`    - per-run hypotheses, predeclared criteria, and outcomes;
* `runs/*/result.md`   - the written result of a run;
* `reports/findings.md`- the dated decision trail;
* `reports/**/*.png|svg` - the rendered visual evidence.

Nothing here invents a status, a threshold, or a conclusion.  Where a run's record
is silent the page says so rather than guessing, and failed or superseded runs are
rendered exactly like successful ones - a weblog that quietly dropped them would
misrepresent the research.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import yaml

from mojidiff.weblog.charts import Series, metric_chart
from mojidiff.weblog.markdown import render_inline, render_markdown

STATE_ORDER = ("planned", "staged", "running", "completed", "failed", "cancelled")
_IMAGE_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class Publication:
    """Where a built site is served, which decides what its footer may say and link.

    The pages themselves are identical for every target and use only relative links,
    so the same build works at a host root (the tailnet server) and under a sub-path
    (GitHub Pages serves the project site at `/emojidiff/`).
    """

    note: str
    remarks: tuple[str, ...] = ()
    links: tuple[tuple[str, str], ...] = ()


LOCAL = Publication(
    note="Generated from the committed research record. Served on the machine's "
    "Tailscale address only."
)

# The public copy points at the public source and checkpoints rather than at the tailnet services, which
# a reader outside the tailnet cannot reach.
PAGES = Publication(
    note="Generated from the committed research record and published from the main branch.",
    remarks=(
        "Code, text and checkpoints are CC BY-SA 4.0. All emojis designed by OpenMoji, "
        "the open-source emoji and icon project, CC BY-SA 4.0; the models learn from and "
        "render adaptations of them. The v8 checkpoint was also trained on adapted Twemoji "
        "graphics, CC BY 4.0.",
        "The project began from a handoff package for gtc, a Codex control-plane machine. "
        "That was only a starting point and an inspiration; MojiDiff has since evolved into "
        "something else, and early entries that mention gtc are kept as history.",
    ),
    links=(
        ("Source", "https://github.com/cpietsch/emojidiff"),
        ("Checkpoints", "https://huggingface.co/chrispie/mojidiff-checkpoints"),
        ("OpenMoji", "https://openmoji.org/"),
        ("Twemoji", "https://github.com/jdecked/twemoji"),
        ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    ),
)

PUBLICATIONS = {"local": LOCAL, "pages": PAGES}


class WeblogError(RuntimeError):
    """The weblog could not be generated from the committed evidence."""


@dataclass
class RunPage:
    run_id: str
    record: dict[str, Any]
    transitions: list[dict[str, Any]] = field(default_factory=list)
    result_markdown: str | None = None
    summary: dict[str, Any] | None = None
    validation: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)

    @property
    def state(self) -> str:
        if self.transitions:
            return str(self.transitions[-1].get("state", "unknown"))
        return str(self.record.get("state", "unknown"))

    @property
    def timestamp(self) -> str:
        for key in ("completed_at", "failed_at", "staged_at", "planned_at"):
            value = self.record.get(key)
            if isinstance(value, str):
                return value
        if self.transitions:
            return str(self.transitions[-1].get("timestamp", ""))
        return ""

    @property
    def href(self) -> str:
        return f"run/{self.run_id}.html"


def build_site(root: Path, out: Path, publication: Publication = LOCAL) -> dict[str, Any]:
    """Write the complete site under `out` and return a build manifest."""

    runs = _collect_runs(root)
    gates, headline = _load_gates(root)
    findings = _read_text(root / "reports" / "findings.md")
    current = _read_text(root / "state" / "CURRENT.md")
    gallery = _collect_gallery(root)

    if out.exists():
        shutil.rmtree(out)
    (out / "run").mkdir(parents=True)
    (out / "assets").mkdir(parents=True)
    for name in ("style.css", "chart.js"):
        _write(out / name, _asset(name))

    copied = _copy_assets(root, out, gallery, runs)
    _write(
        out / "index.html",
        _index_page(runs, gates, headline or _headline(current), gallery, findings, publication),
    )
    _write(out / "runs.html", _runs_page(runs, publication))
    _write(
        out / "findings.html",
        _document_page("Decision trail", findings, "findings", publication),
    )
    _write(
        out / "state.html",
        _document_page("Current research state", current, "state", publication),
    )
    _write(out / "gallery.html", _gallery_page(gallery, publication))
    for run in runs:
        _write(out / "run" / f"{run.run_id}.html", _run_page(run, root, publication))
    manifest = {
        "runs": len(runs),
        "pages": 5 + len(runs),
        "assets": copied,
        "gates": len(gates),
    }
    _write(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --------------------------------------------------------------------------- input


def _collect_runs(root: Path) -> list[RunPage]:
    transitions: dict[str, list[dict[str, Any]]] = {}
    registry = root / "state" / "runs.jsonl"
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Most rows name one run. A batch row names several under `run_ids`, and
            # that transition belongs on every run it covers.
            named = row.get("run_id")
            batch = row.get("run_ids")
            targets = [named] if isinstance(named, str) else []
            if isinstance(batch, list):
                targets.extend(item for item in batch if isinstance(item, str))
            for run_id in targets:
                transitions.setdefault(run_id, []).append(row)

    pages: list[RunPage] = []
    seen: set[str] = set()
    for path in sorted((root / "runs").glob("*/run.yaml")):
        record = yaml.safe_load(path.read_text()) or {}
        run_id = str(record.get("run_id") or path.parent.name)
        seen.add(run_id)
        page = RunPage(run_id=run_id, record=record, transitions=transitions.get(run_id, []))
        page.result_markdown = _read_text(path.parent / "result.md")
        report_root = _report_root(root, record)
        for directory in (path.parent, report_root):
            if directory is None or not directory.is_dir():
                continue
            if page.summary is None:
                page.summary = _read_json(directory / "summary.json")
            if not page.validation:
                page.validation = _read_jsonl(directory / "validation.jsonl")
            if not page.metrics:
                page.metrics = _read_jsonl(directory / "metrics.jsonl")
            page.images.extend(
                item.relative_to(root)
                for item in sorted(directory.rglob("*"))
                if item.suffix.lower() in _IMAGE_SUFFIXES and item.is_file()
            )
        page.images = sorted(set(page.images))
        pages.append(page)

    # A run may exist only in the append-only registry, for instance one recorded
    # retroactively. It still belongs on the timeline, and if its rows name a config the
    # page can reach that config's report root - which is where the summary, the metric
    # trace and any renders live. Without this such a run shows as a bare row with no
    # evidence attached, which is the opposite of what this site is for.
    for run_id, rows in transitions.items():
        if run_id in seen:
            continue
        merged: dict[str, Any] = {"run_id": run_id}
        for row in rows:
            for key in ("config", "outputs", "artifacts"):
                if key in row:
                    merged[key] = row[key]
        page = RunPage(run_id=run_id, record=merged, transitions=rows)
        report_root = _report_root(root, merged)
        if report_root is not None:
            page.summary = _read_json(report_root / "summary.json")
            page.metrics = _read_jsonl(report_root / "metrics.jsonl")
            page.images = sorted(
                item.relative_to(root)
                for item in report_root.rglob("*")
                if item.suffix.lower() in _IMAGE_SUFFIXES and item.is_file()
            )
        pages.append(page)
    pages.sort(key=lambda page: (page.timestamp, page.run_id))
    return pages


def _report_root(root: Path, record: dict[str, Any]) -> Path | None:
    # A superseded attempt keeps its artifacts under a directory of its own, named in
    # its registry row, and must not be shown the directory its config now points at.
    for block in (record.get("outputs"), record.get("artifacts")):
        if not isinstance(block, dict):
            continue
        for key in ("report_root", "report", "compact_report"):
            value = block.get(key)
            if isinstance(value, str):
                candidate = root / value
                if candidate.is_dir():
                    return candidate
    config = record.get("config")
    path = config.get("path") if isinstance(config, dict) else config
    if isinstance(path, str) and (root / path).is_file():
        try:
            parsed = yaml.safe_load((root / path).read_text()) or {}
        except yaml.YAMLError:
            return None
        value = parsed.get("report_root") if isinstance(parsed, dict) else None
        if isinstance(value, str) and (root / value).is_dir():
            return root / value
    return None


def _collect_gallery(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    reports = root / "reports"
    if not reports.is_dir():
        return groups
    for item in sorted(reports.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        groups.setdefault(str(item.parent.relative_to(root)), []).append(item.relative_to(root))
    return groups


def _load_gates(root: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Return the gate board and the operator's explicit one-line headline."""

    path = root / "state" / "gates.yaml"
    if not path.is_file():
        return [], None
    parsed = yaml.safe_load(path.read_text()) or {}
    gates = parsed.get("gates") if isinstance(parsed, dict) else None
    if not isinstance(gates, list):
        raise WeblogError("state/gates.yaml must contain a `gates` list")
    headline = parsed.get("headline") if isinstance(parsed, dict) else None
    return (
        [gate for gate in gates if isinstance(gate, dict)],
        headline.strip() if isinstance(headline, str) else None,
    )


def _copy_assets(root: Path, out: Path, gallery: dict[str, list[Path]], runs: list[RunPage]) -> int:
    wanted = {item for items in gallery.values() for item in items}
    wanted.update(image for run in runs for image in run.images)
    for relative in sorted(wanted):
        target = out / "assets" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return len(wanted)


def _read_text(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    parsed = json.loads(path.read_text())
    return parsed if isinstance(parsed, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


# -------------------------------------------------------------------------- output


def _latest_decisions(findings: str | None, count: int = 3) -> str:
    """The newest entries of the decision trail, on the page people actually open.

    The trail is written oldest-first and is now long enough that its newest entry -
    the one that says what the project currently believes - is several screens down a
    separate page. The point of this site is that someone can see where the research
    stands without reading it end to end.
    """

    if not findings:
        return ""
    entries: list[tuple[str, list[str]]] = []
    for line in findings.splitlines():
        if line.startswith("## "):
            entries.append((line[3:].strip(), []))
        elif entries:
            entries[-1][1].append(line)
    if not entries:
        return ""
    cards = []
    for title, body in entries[-count:][::-1]:
        summary = next(
            (paragraph for paragraph in " ".join(body).split("  ") if paragraph.strip()),
            "",
        ).strip()
        cards.append(
            f'<article class="decision"><h3>{escape(title)}</h3>'
            f"<p>{render_inline(summary[:420])}</p></article>"
        )
    return (
        "<section><h2>Latest decisions</h2>"
        "<p class='note'>Newest first, verbatim openings from "
        "<code>reports/findings.md</code>.</p>"
        f"<div class='decision-grid'>{''.join(cards)}</div>"
        '<p><a class="more" href="findings.html">Full decision trail &rarr;</a></p>'
        "</section>"
    )


def _index_page(
    runs: list[RunPage],
    gates: list[dict[str, Any]],
    headline: str,
    gallery: dict[str, list[Path]],
    findings: str | None = None,
    publication: Publication = LOCAL,
) -> str:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.state] = counts.get(run.state, 0) + 1
    tiles = "".join(
        f'<div class="tile"><span class="tile-value">{counts.get(state, 0)}</span>'
        f'<span class="tile-label state-{state}">{state}</span></div>'
        for state in STATE_ORDER
        if counts.get(state)
    )
    images = sum(len(items) for items in gallery.values())
    tiles += (
        f'<div class="tile"><span class="tile-value">{images}</span>'
        f'<span class="tile-label">rendered artifacts</span></div>'
    )

    recent = [run for run in runs if run.timestamp][-12:][::-1]
    return _shell(
        "Overview",
        "index",
        f"""
<section class="hero">
  <p class="eyebrow">MojiDiff &middot; categorical denoising over typed SVG programs</p>
  <h1>Research weblog</h1>
  <p class="lede">{render_inline(headline)}</p>
  <div class="tiles">{tiles}</div>
</section>
{_latest_decisions(findings)}
{_gate_board(gates)}
<section>
  <h2>Recent experiments</h2>
  <p class="note">Newest first. Failed and superseded runs are kept and shown; a run
  whose predeclared criterion was falsified is a result, not a mistake.</p>
  {_run_table(recent)}
  <p><a class="more" href="runs.html">All {len(runs)} runs &rarr;</a></p>
</section>
<section>
  <h2>Visual evidence</h2>
  <p class="note">Renders the claims are checkable against: corpus contact sheets,
  codec worst cases, and paired corruption/prediction trajectories.</p>
  {_image_strip(gallery)}
  <p><a class="more" href="gallery.html">Full gallery &rarr;</a></p>
</section>
""",
        publication=publication,
    )


def _headline(current: str | None) -> str:
    """The newest paragraph of the current hypothesis, as the operator wrote it.

    The section is written oldest-first, so the last paragraph is the current state
    of the research; leading with the first one would headline Gate B forever.
    """

    if not current:
        return "No current-state handoff is committed yet."
    lines = current.splitlines()
    try:
        start = lines.index("## Current hypothesis and evidence") + 1
    except ValueError:
        return "No current hypothesis section is committed yet."
    paragraphs: list[list[str]] = []
    current_paragraph: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if not line.strip():
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
            continue
        current_paragraph.append(line.strip())
    if current_paragraph:
        paragraphs.append(current_paragraph)
    if not paragraphs:
        return "No current hypothesis section is committed yet."
    return " ".join(paragraphs[-1])


def _gate_board(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return ""
    cards = []
    for gate in gates:
        status = str(gate.get("status", "unknown"))
        evidence = gate.get("evidence")
        cards.append(
            f'<article class="gate gate-{escape(status)}">'
            f'<header><span class="gate-id">{escape(str(gate.get("id", "?")))}</span>'
            f'<span class="badge badge-{escape(status)}">{escape(status)}</span></header>'
            f"<h3>{escape(str(gate.get('title', '')))}</h3>"
            f"<p>{escape(str(gate.get('question', '')))}</p>"
            + (f'<p class="evidence">{escape(str(evidence))}</p>' if evidence else "")
            + "</article>"
        )
    return (
        "<section><h2>Evidence gates</h2>"
        "<p class='note'>The project advances from cheap tests to expensive ones. A gate "
        "opens only when the preceding evidence justifies it.</p>"
        f"<div class='gate-grid'>{''.join(cards)}</div></section>"
    )


def _runs_page(runs: list[RunPage], publication: Publication = LOCAL) -> str:
    return _shell(
        "Experiments",
        "runs",
        f"""
<section>
  <h1>Experiments</h1>
  <p class="lede">Every registered run, newest first, with the state recorded in the
  append-only registry. Run identity is a slug plus short code, config, and data
  hashes - seeds alone are not run identities.</p>
  {_run_table(runs[::-1])}
</section>
""",
        publication=publication,
    )


def _run_table(runs: list[RunPage]) -> str:
    rows = []
    for run in runs:
        outcome = _outcome_badge(run)
        rows.append(
            f"<tr><td class='mono'><a href='{escape(_relative(run.href))}'>"
            f"{escape(run.run_id)}</a></td>"
            f"<td><span class='badge badge-{escape(run.state)}'>{escape(run.state)}</span></td>"
            f"<td>{outcome}</td>"
            f"<td class='mono nowrap'>{escape(run.timestamp[:16].replace('T', ' '))}</td></tr>"
        )
    return (
        "<div class='table-scroll'><table class='runs'><thead><tr><th>run</th><th>state</th>"
        f"<th>predeclared outcome</th><th>recorded</th></tr></thead><tbody>{''.join(rows)}"
        "</tbody></table></div>"
    )


def _relative(href: str) -> str:
    return href


def _outcome_badge(run: RunPage) -> str:
    outcome = run.record.get("predeclared_outcome")
    if isinstance(outcome, dict) and isinstance(outcome.get("overall"), str):
        value = str(outcome["overall"])
        return f"<span class='badge badge-{escape(value)}'>{escape(value)}</span>"
    if run.record.get("predeclared_criteria"):
        return "<span class='badge badge-planned'>declared</span>"
    return "<span class='muted'>&mdash;</span>"


def _run_page(run: RunPage, root: Path, publication: Publication = LOCAL) -> str:
    sections = [
        f"<section class='run-head'><p class='eyebrow'><a href='../runs.html'>"
        f"&larr; experiments</a></p><h1 class='mono'>{escape(run.run_id)}</h1>"
        f"<p><span class='badge badge-{escape(run.state)}'>{escape(run.state)}</span> "
        f"{_outcome_badge(run)}</p></section>"
    ]
    hypothesis = run.record.get("hypothesis")
    if isinstance(hypothesis, str):
        sections.append(
            f"<section><h2>Hypothesis</h2><blockquote>{escape(hypothesis)}</blockquote></section>"
        )
    for key, title in (
        ("expected_information_gain", "Why run it"),
        ("scope_limits", "Scope limits"),
    ):
        value = run.record.get(key)
        if isinstance(value, str):
            sections.append(f"<section><h2>{title}</h2><p>{escape(value)}</p></section>")

    charts = _run_charts(run)
    if charts:
        sections.append(f"<section><h2>Measured behaviour</h2>{charts}</section>")
    # The result itself. Some runs carry their whole outcome here - predeclared
    # criteria, which of them passed, the timings - and without this section the page
    # showed a chart and a state transition and never said what happened.
    if run.summary:
        sections.append(
            "<section><h2>Measured result</h2><p class='note'>Verbatim from the run's "
            f"<code>summary.json</code>.</p>{_tree(run.summary)}</section>"
        )
    if run.images:
        sections.append(
            "<section><h2>Visual output</h2>"
            f"{_image_grid(run.images, prefix='../assets/')}</section>"
        )
    if run.result_markdown:
        sections.append(
            f"<section class='prose'><h2>Written result</h2>"
            f"{render_markdown(_strip_leading_heading(run.result_markdown))}</section>"
        )
    if run.transitions:
        sections.append(f"<section><h2>State transitions</h2>{_transitions(run)}</section>")
    # Most runs carry a committed run.yaml; a registry-only run does not, and saying
    # otherwise would point the reader at a file that is not there.
    if (root / "runs" / run.run_id / "run.yaml").is_file():
        provenance = (
            f"Verbatim from <code>runs/{escape(run.run_id)}/run.yaml</code>, "
            "the record committed before launch."
        )
    else:
        provenance = (
            "This run has no <code>run.yaml</code>. What follows is the identity and "
            "configuration carried by its rows in <code>state/runs.jsonl</code>, the "
            "append-only registry."
        )
    sections.append(
        f"<section><h2>Run record</h2><p class='note'>{provenance}</p>{_tree(run.record)}</section>"
    )
    return _shell(run.run_id, "runs", "".join(sections), depth=1, publication=publication)


def _strip_leading_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:])
    return markdown


def _run_charts(run: RunPage) -> str:
    charts: list[str] = []
    selected = None
    summary = run.summary or {}
    selection = summary.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("selected_step"), int):
        selected = [("selected", float(selection["selected_step"]))]
    elif isinstance(summary.get("selected_step"), int):
        selected = [("selected", float(summary["selected_step"]))]

    if run.validation:
        steps = [float(row["step"]) for row in run.validation if "step" in row]
        losses = [
            (step, float(row["loss"]))
            for step, row in zip(steps, run.validation, strict=True)
            if "loss" in row
        ]
        if losses:
            charts.append(
                metric_chart(
                    "Held-out loss",
                    [Series("held-out loss", tuple(losses))],
                    x_label="optimizer step",
                    y_label="loss",
                    markers=selected,
                )
            )
        accuracy_series = []
        for label, key in (
            ("aggregate", "accuracy"),
            ("changed fields", "changed_accuracy"),
            ("retained fields", "retained_accuracy"),
        ):
            points = tuple(
                (step, float(row[key]))
                for step, row in zip(steps, run.validation, strict=True)
                if isinstance(row.get(key), (int, float))
            )
            if points:
                accuracy_series.append(Series(label, points))
        if accuracy_series:
            charts.append(
                metric_chart(
                    "Held-out token accuracy",
                    accuracy_series,
                    x_label="optimizer step",
                    y_label="accuracy",
                    y_zero=True,
                    markers=selected,
                )
            )
    # Gate I reports likelihood rather than a denoising score, and its floor belongs on
    # the same axis as the model. Plotted apart, a model tracking the marginal policy
    # looks like a model that is learning.
    if run.metrics and any("held_out_nll" in row for row in run.metrics):
        likelihood = []
        for label, key in (
            ("model", "held_out_nll"),
            ("position-marginal floor", "marginal_nll"),
        ):
            points = tuple(
                (float(row["step"]), float(row[key]))
                for row in run.metrics
                if "step" in row and isinstance(row.get(key), (int, float))
            )
            if points:
                likelihood.append(Series(label, points))
        if likelihood:
            charts.append(
                metric_chart(
                    "Held-out negative log likelihood per free token",
                    likelihood,
                    x_label="optimizer step",
                    y_label="nats per token",
                    y_zero=True,
                    markers=selected,
                )
            )
    # Gate L's trace: accuracy over the masked positions the grammar leaves free, with
    # the position-marginal policy's accuracy on the same positions beside it.
    if run.metrics and any("masked_token_accuracy" in row for row in run.metrics):
        points = tuple(
            (float(row["step"]), float(row["masked_token_accuracy"]))
            for row in run.metrics
            if "step" in row and isinstance(row.get("masked_token_accuracy"), (int, float))
        )
        series = [Series("model", points)] if points else []
        floor = summary.get("marginal_masked_token_accuracy")
        if points and isinstance(floor, (int, float)):
            series.append(
                Series(
                    "position-marginal floor",
                    tuple((step, float(floor)) for step, _ in points),
                )
            )
        if series:
            charts.append(
                metric_chart(
                    "Masked-token accuracy over unforced positions",
                    series,
                    x_label="optimizer step",
                    y_label="accuracy",
                    y_zero=True,
                    markers=selected,
                )
            )
    if run.metrics and any("free_token_accuracy" in row for row in run.metrics):
        points = tuple(
            (float(row["step"]), float(row["free_token_accuracy"]))
            for row in run.metrics
            if "step" in row and "free_token_accuracy" in row
        )
        if points:
            charts.append(
                metric_chart(
                    "Next-token accuracy over unforced positions",
                    [Series("free-token accuracy", points)],
                    x_label="optimizer step",
                    y_label="accuracy",
                    y_zero=True,
                )
            )
    if run.metrics and any("train_token_accuracy" in row for row in run.metrics):
        points = tuple(
            (float(row["step"]), float(row["train_token_accuracy"]))
            for row in run.metrics
            if "step" in row and "train_token_accuracy" in row
        )
        if points:
            charts.append(
                metric_chart(
                    "Training token accuracy (per batch)",
                    [Series("train accuracy", points)],
                    x_label="optimizer step",
                    y_label="accuracy",
                    y_zero=True,
                    markers=selected,
                )
            )
    return "".join(chart for chart in charts if chart)


def _transitions(run: RunPage) -> str:
    items = []
    for row in run.transitions:
        state = str(row.get("state", "unknown"))
        detail = {
            key: value for key, value in row.items() if key not in {"run_id", "state", "timestamp"}
        }
        summary = str(detail.get("reason") or detail.get("result") or "") if detail else ""
        items.append(
            f"<li><span class='badge badge-{escape(state)}'>{escape(state)}</span>"
            f"<span class='mono nowrap'>{escape(str(row.get('timestamp', '')))}</span>"
            + (f"<span class='detail'>{escape(summary)}</span>" if summary else "")
            + "</li>"
        )
    return f"<ol class='transitions'>{''.join(items)}</ol>"


def _tree(value: Any, level: int = 0) -> str:
    if isinstance(value, dict):
        rows = "".join(
            f"<div class='kv'><div class='k'>{escape(str(key))}</div>"
            f"<div class='v'>{_tree(item, level + 1)}</div></div>"
            for key, item in value.items()
        )
        return f"<div class='tree'>{rows}</div>"
    if isinstance(value, list):
        # A long list is almost always per-sample detail - 32 digests, 128 coverage
        # figures - and printing it whole buries the numbers that carry the result.
        # The full list stays in the committed artifact, which this only summarises.
        head, rest = value[:6], len(value) - 6
        if all(not isinstance(item, (dict, list)) for item in value):
            text = ", ".join(str(item) for item in head)
            return escape(text if rest <= 0 else f"{text} … and {rest} more")
        rendered = "".join(f"<div class='li'>{_tree(item, level + 1)}</div>" for item in head)
        if rest > 0:
            rendered += f"<div class='li note'>… and {rest} more</div>"
        return rendered
    if isinstance(value, bool):
        return f"<span class='bool bool-{str(value).lower()}'>{str(value).lower()}</span>"
    text = str(value)
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return f"<code class='hash' title='{escape(text)}'>{escape(text[:16])}&hellip;</code>"
    return escape(text)


def _document_page(
    title: str, markdown: str | None, active: str, publication: Publication = LOCAL
) -> str:
    if markdown is None:
        body = "<p class='note'>Not committed yet.</p>"
    else:
        body = render_markdown(_strip_leading_heading(markdown))
    return _shell(
        title,
        active,
        f"<section class='prose'><h1>{escape(title)}</h1>{body}</section>",
        publication=publication,
    )


def _gallery_page(gallery: dict[str, list[Path]], publication: Publication = LOCAL) -> str:
    groups = "".join(
        f"<section><h2 class='mono'>{escape(name)}</h2>{_image_grid(items)}</section>"
        for name, items in sorted(gallery.items())
    )
    return _shell(
        "Gallery",
        "gallery",
        "<section><h1>Visual evidence</h1><p class='lede'>Every render committed under "
        "<code>reports/</code>, grouped by the study that produced it.</p></section>" + groups,
        publication=publication,
    )


def _image_grid(images: list[Path], prefix: str = "assets/") -> str:
    cards = "".join(
        f"<figure class='shot'><a href='{escape(prefix + item.as_posix())}'>"
        f"<img loading='lazy' src='{escape(prefix + item.as_posix())}' "
        f"alt='{escape(item.stem)}'></a>"
        f"<figcaption class='mono'>{escape(item.name)}</figcaption></figure>"
        for item in images
    )
    return f"<div class='shots'>{cards}</div>"


def _image_strip(gallery: dict[str, list[Path]]) -> str:
    picked: list[Path] = []
    for _, items in sorted(gallery.items()):
        picked.append(items[0])
    return _image_grid(picked[:8])


def _shell(
    title: str, active: str, body: str, depth: int = 0, publication: Publication = LOCAL
) -> str:
    up = "../" * depth
    notes = "".join(f"<p>{escape(text)}</p>" for text in (publication.note, *publication.remarks))
    links = " &middot; ".join(
        f"<a href='{escape(href, quote=True)}'>{escape(label)}</a>"
        for label, href in publication.links
    )
    nav = "".join(
        f"<a class='{'on' if key == active else ''}' href='{up}{href}'>{label}</a>"
        for key, href, label in (
            ("index", "index.html", "Overview"),
            ("runs", "runs.html", "Experiments"),
            ("findings", "findings.html", "Decisions"),
            ("gallery", "gallery.html", "Gallery"),
            ("state", "state.html", "Current state"),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{escape(title)} &middot; MojiDiff research weblog</title>
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<header class="topbar"><a class="brand" href="{up}index.html">MojiDiff</a>
<nav>{nav}</nav></header>
<main>{body}</main>
<footer>{notes}{f"<p>{links}</p>" if links else ""}</footer>
<script src="{up}chart.js"></script>
</body>
</html>
"""


def _asset(name: str) -> str:
    """Read one of the site's static assets, which live beside this module."""

    return (Path(__file__).parent / "assets" / name).read_text()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--out", type=Path, default=Path("site"), help="output directory")
    parser.add_argument(
        "--publication",
        choices=sorted(PUBLICATIONS),
        default="local",
        help="where the site is served: the tailnet machine, or the public GitHub Pages copy",
    )
    args = parser.parse_args()
    manifest = build_site(args.root.resolve(), args.out.resolve(), PUBLICATIONS[args.publication])
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
