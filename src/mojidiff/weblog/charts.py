"""Inline SVG line charts for the research weblog.

Every chart here plots change over time - a metric against the optimizer step - so
the form is always a line chart.  Two rules from the project's own reporting
discipline are enforced structurally rather than by care:

* **One axis per chart.**  Loss and accuracy have different scales and never share
  a plot; `metric_chart` takes one y-scale and the caller splits measures across
  charts.  A dual-axis chart would let either curve be made to look dominant.
* **Nothing is encoded by colour alone.**  Two or more series always carry a
  legend, and every chart ships a table view of the exact numbers, so a reader who
  cannot separate the hues still gets the values.

Colours are the validated categorical slots 1-3 (blue, orange, aqua), which clear
the colour-vision-deficiency separation gates as an adjacent set in both themes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape

SERIES_SLOTS = ("series-1", "series-2", "series-3")
"""Categorical slots, assigned in fixed order and never cycled."""

_WIDTH = 760.0
_HEIGHT = 340.0
_PAD_LEFT = 68.0
_PAD_RIGHT = 18.0
_PAD_TOP = 18.0
_PAD_BOTTOM = 46.0


@dataclass(frozen=True)
class Series:
    """One named curve: matched x and y samples."""

    label: str
    points: tuple[tuple[float, float], ...]


def metric_chart(
    title: str,
    series: list[Series],
    *,
    x_label: str,
    y_label: str,
    y_zero: bool = False,
    value_format: str = ".4f",
    markers: list[tuple[str, float]] | None = None,
) -> str:
    """Render one line chart plus its legend, hover layer, and table view."""

    if not series or any(not item.points for item in series):
        return ""
    if len(series) > len(SERIES_SLOTS):
        raise ValueError("metric_chart takes at most three series; facet instead")

    xs = [x for item in series for x, _ in item.points]
    ys = [y for item in series for _, y in item.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = (0.0 if y_zero else min(ys)), max(ys)
    if y_max == y_min:
        y_max = y_min + 1.0
    if x_max == x_min:
        x_max = x_min + 1.0
    y_min, y_max, y_ticks = _nice_scale(y_min, y_max, zero_based=y_zero)

    def px(value: float) -> float:
        span = _WIDTH - _PAD_LEFT - _PAD_RIGHT
        return _PAD_LEFT + (value - x_min) / (x_max - x_min) * span

    def py(value: float) -> float:
        span = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
        return _HEIGHT - _PAD_BOTTOM - (value - y_min) / (y_max - y_min) * span

    parts: list[str] = []
    for tick in y_ticks:
        y = py(tick)
        parts.append(
            f'<line class="grid" x1="{_PAD_LEFT:.1f}" y1="{y:.1f}" '
            f'x2="{_WIDTH - _PAD_RIGHT:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick tick-y" x="{_PAD_LEFT - 10:.1f}" y="{y + 4:.1f}">'
            f"{_trim(tick)}</text>"
        )
    for tick in _x_ticks(x_min, x_max):
        x = px(tick)
        parts.append(
            f'<text class="tick tick-x" x="{x:.1f}" y="{_HEIGHT - _PAD_BOTTOM + 20:.1f}">'
            f"{_trim(tick)}</text>"
        )
    parts.append(
        f'<line class="axis" x1="{_PAD_LEFT:.1f}" y1="{_HEIGHT - _PAD_BOTTOM:.1f}" '
        f'x2="{_WIDTH - _PAD_RIGHT:.1f}" y2="{_HEIGHT - _PAD_BOTTOM:.1f}"/>'
    )

    for marker_label, marker_x in markers or ():
        x = px(marker_x)
        parts.append(
            f'<line class="marker" x1="{x:.1f}" y1="{_PAD_TOP:.1f}" '
            f'x2="{x:.1f}" y2="{_HEIGHT - _PAD_BOTTOM:.1f}"/>'
        )
        anchor = "end" if x > _WIDTH * 0.66 else "start"
        offset = -6 if anchor == "end" else 6
        parts.append(
            f'<text class="marker-label" x="{x + offset:.1f}" y="{_PAD_TOP + 12:.1f}" '
            f'text-anchor="{anchor}">{escape(marker_label)}</text>'
        )

    for index, item in enumerate(series):
        slot = SERIES_SLOTS[index]
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in item.points)
        parts.append(f'<polyline class="line {slot}" points="{path}"/>')
        last_x, last_y = item.points[-1]
        parts.append(
            f'<circle class="end {slot}" cx="{px(last_x):.1f}" cy="{py(last_y):.1f}" r="4"/>'
        )

    hover = {
        "x": [x for x, _ in series[0].points],
        "left": _PAD_LEFT,
        "right": _WIDTH - _PAD_RIGHT,
        "top": _PAD_TOP,
        "bottom": _HEIGHT - _PAD_BOTTOM,
        "format": value_format,
        "xLabel": x_label,
        "series": [
            {"label": item.label, "values": [y for _, y in item.points]} for item in series
        ],
        "px": [px(x) for x, _ in series[0].points],
    }
    parts.append(
        f'<line class="crosshair" x1="0" y1="{_PAD_TOP:.1f}" x2="0" '
        f'y2="{_HEIGHT - _PAD_BOTTOM:.1f}" hidden/>'
    )
    parts.append(
        f'<rect class="hit" x="{_PAD_LEFT:.1f}" y="{_PAD_TOP:.1f}" '
        f'width="{_WIDTH - _PAD_LEFT - _PAD_RIGHT:.1f}" '
        f'height="{_HEIGHT - _PAD_TOP - _PAD_BOTTOM:.1f}"/>'
    )

    legend = ""
    if len(series) > 1:
        chips = "".join(
            f'<span class="chip"><i class="{SERIES_SLOTS[index]}"></i>'
            f"{escape(item.label)}</span>"
            for index, item in enumerate(series)
        )
        legend = f'<div class="legend">{chips}</div>'

    return (
        f'<figure class="chart" data-hover=\'{escape(json.dumps(hover), quote=True)}\'>'
        f"<figcaption>{escape(title)}</figcaption>"
        f"{legend}"
        f'<div class="plot"><svg viewBox="0 0 {_WIDTH:.0f} {_HEIGHT:.0f}" '
        f'role="img" aria-label="{escape(title)}" preserveAspectRatio="xMidYMid meet">'
        f'<text class="axis-label axis-label-y" x="14" y="{_HEIGHT / 2:.1f}" '
        f'transform="rotate(-90 14 {_HEIGHT / 2:.1f})">{escape(y_label)}</text>'
        f'<text class="axis-label" x="{(_PAD_LEFT + _WIDTH - _PAD_RIGHT) / 2:.1f}" '
        f'y="{_HEIGHT - 8:.1f}">{escape(x_label)}</text>'
        f'{"".join(parts)}</svg>'
        f'<div class="tooltip" hidden></div></div>'
        f"{_table(series, x_label, value_format)}"
        f"</figure>"
    )


def _table(series: list[Series], x_label: str, value_format: str) -> str:
    """The table view: the same numbers, for readers colour cannot serve."""

    head = "".join(f"<th>{escape(item.label)}</th>" for item in series)
    rows: list[str] = []
    for index, (x, _) in enumerate(series[0].points):
        cells = "".join(
            f"<td>{format(item.points[index][1], value_format)}</td>"
            if index < len(item.points)
            else "<td>-</td>"
            for item in series
        )
        rows.append(f"<tr><td>{_trim(x)}</td>{cells}</tr>")
    return (
        '<details class="chart-table"><summary>Table view</summary>'
        f'<div class="table-scroll"><table><thead><tr><th>{escape(x_label)}</th>{head}</tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def _nice_scale(low: float, high: float, *, zero_based: bool) -> tuple[float, float, list[float]]:
    span = high - low
    step = _nice_step(span / 4.0)
    start = 0.0 if zero_based else _floor_to(low, step)
    end = _ceil_to(high, step)
    ticks: list[float] = []
    value = start
    while value <= end + step / 2:
        ticks.append(round(value, 10))
        value += step
    return start, ticks[-1], ticks


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exponent = 10.0 ** _floor_log10(raw)
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= candidate * exponent:
            return candidate * exponent
    return 10.0 * exponent


def _floor_log10(value: float) -> int:
    exponent = 0
    while value < 1.0:
        value *= 10.0
        exponent -= 1
    while value >= 10.0:
        value /= 10.0
        exponent += 1
    return exponent


def _floor_to(value: float, step: float) -> float:
    return step * (value // step)


def _ceil_to(value: float, step: float) -> float:
    return -_floor_to(-value, step)


def _x_ticks(low: float, high: float) -> list[float]:
    step = _nice_step((high - low) / 5.0)
    ticks: list[float] = []
    value = _ceil_to(low, step)
    while value <= high:
        ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


def _trim(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"
