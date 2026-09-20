"""A deliberately small Markdown subset renderer for the research weblog.

The weblog turns committed research prose - `reports/findings.md`, per-run
`result.md`, and `state/CURRENT.md` - into HTML.  A full Markdown implementation is
not a declared dependency of this project, and pulling one in to render text the
repository itself authors would widen the dependency surface for no scientific gain.

The supported subset is exactly what that prose uses: ATX headings, fenced code,
pipe tables, unordered and ordered lists, blockquotes, paragraphs, and the inline
spans `**strong**`, `*emphasis*`, `` `code` `` and `[text](url)`.  Anything outside
the subset degrades to escaped literal text rather than being silently dropped.

Every source byte is HTML-escaped before any markup is applied, so the renderer
cannot emit markup that the source did not ask for.
"""

from __future__ import annotations

import re
from html import escape

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_UNORDERED = re.compile(r"^\s*[-*]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^>\s?(.*)$")
_FENCE = re.compile(r"^\s*```\s*([A-Za-z0-9_+-]*)\s*$")

_CODE_SPAN = re.compile(r"`([^`]+)`")
_STRONG = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*")
_EMPHASIS = re.compile(r"(?<![*\w])\*(\S(?:.*?\S)?)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_ALLOWED_SCHEMES = frozenset({"http", "https", "mailto"})


def render_markdown(source: str) -> str:
    """Render the supported Markdown subset to HTML."""

    lines = source.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        fence = _FENCE.match(line)
        if fence:
            index, block = _collect_fence(lines, index)
            out.append(block)
            continue
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue
        if _is_table_start(lines, index):
            index, block = _collect_table(lines, index)
            out.append(block)
            continue
        if _UNORDERED.match(line) or _ORDERED.match(line):
            index, block = _collect_list(lines, index)
            out.append(block)
            continue
        if _BLOCKQUOTE.match(line):
            index, block = _collect_blockquote(lines, index)
            out.append(block)
            continue
        index, block = _collect_paragraph(lines, index)
        out.append(block)
    return "\n".join(out)


def _collect_fence(lines: list[str], index: int) -> tuple[int, str]:
    body: list[str] = []
    index += 1
    while index < len(lines) and not _FENCE.match(lines[index]):
        body.append(lines[index])
        index += 1
    return index + 1, f"<pre><code>{escape('\n'.join(body))}</code></pre>"


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        "|" in lines[index]
        and index + 1 < len(lines)
        and _TABLE_DIVIDER.match(lines[index + 1]) is not None
    )


def _collect_table(lines: list[str], index: int) -> tuple[int, str]:
    header = _cells(lines[index])
    index += 2
    rows: list[list[str]] = []
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        rows.append(_cells(lines[index]))
        index += 1
    head = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return index, f'<div class="table-scroll">{table}</div>'


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _collect_list(lines: list[str], index: int) -> tuple[int, str]:
    ordered = _ORDERED.match(lines[index]) is not None
    pattern = _ORDERED if ordered else _UNORDERED
    items: list[str] = []
    while index < len(lines):
        match = pattern.match(lines[index])
        if match:
            items.append(match.group(1).strip())
            index += 1
            continue
        # A wrapped continuation line belongs to the item above it.
        if items and lines[index].strip() and lines[index].startswith((" ", "\t")):
            items[-1] = f"{items[-1]} {lines[index].strip()}"
            index += 1
            continue
        break
    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{_inline(item)}</li>" for item in items)
    return index, f"<{tag}>{body}</{tag}>"


def _collect_blockquote(lines: list[str], index: int) -> tuple[int, str]:
    body: list[str] = []
    while index < len(lines):
        match = _BLOCKQUOTE.match(lines[index])
        if not match:
            break
        body.append(match.group(1))
        index += 1
    return index, f"<blockquote>{_inline(' '.join(part.strip() for part in body))}</blockquote>"


def _collect_paragraph(lines: list[str], index: int) -> tuple[int, str]:
    body: list[str] = []
    while index < len(lines) and lines[index].strip():
        if (
            _HEADING.match(lines[index])
            or _FENCE.match(lines[index])
            or _UNORDERED.match(lines[index])
            or _ORDERED.match(lines[index])
            or _BLOCKQUOTE.match(lines[index])
            or _is_table_start(lines, index)
        ):
            break
        body.append(lines[index].strip())
        index += 1
    return index, f"<p>{_inline(' '.join(body))}</p>"


def _inline(text: str) -> str:
    """Escape, then apply the supported inline spans."""

    result = escape(text, quote=False)
    result = _CODE_SPAN.sub(lambda m: f"<code>{m.group(1)}</code>", result)
    result = _STRONG.sub(lambda m: f"<strong>{m.group(1)}</strong>", result)
    result = _EMPHASIS.sub(lambda m: f"<em>{m.group(1)}</em>", result)
    result = _LINK.sub(_link, result)
    return result


def _link(match: re.Match[str]) -> str:
    """Emit a link only for a relative target or an explicitly allowed scheme.

    A bare `scheme:` prefix is the dangerous case - `javascript:` and `data:` both
    parse as ordinary Markdown links - so anything with a scheme outside the
    allowlist degrades to the escaped literal source text.
    """

    href = match.group(2)
    scheme = _SCHEME.match(href)
    if scheme is not None and scheme.group(1).lower() not in _ALLOWED_SCHEMES:
        return escape(match.group(0), quote=False)
    return f'<a href="{escape(href, quote=True)}">{match.group(1)}</a>'
