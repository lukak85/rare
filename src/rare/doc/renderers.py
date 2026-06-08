"""HTML and Markdown renderers for GlasanaDocument."""

from __future__ import annotations

from typing import Optional

from .schema import (
    AbandonItem,
    AdvertisementItem,
    AnyDocItem,
    AuthorItem,
    BylineItem,
    CaptionItem,
    DatelineItem,
    DeckItem,
    DocItem,
    DropcapItem,
    EditNoteItem,
    FigBylineItem,
    FigureItem,
    FootnoteItem,
    GlasanaDocument,
    HeadlineItem,
    KickerItem,
    LiteraryItem,
    LiteratureItem,
    MarginNoteItem,
    OrderedListItem,
    ParagraphItem,
    QuestionItem,
    QuoteItem,
    SectionItem,
    SubheadItem,
    SubsubheadItem,
    TableData,
    TableItem,
    TOCItem,
    TranslatorItem,
    UnorderedListItem,
)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def to_markdown_pages(doc: GlasanaDocument, raw: bool = False) -> dict[int, str]:
    """Render the document to one markdown string per page, keyed by page_no.

    Splits `body_order` by each item's `provenance.page_no`, then renders each
    page's slice through `to_markdown`. Used by the OmniDocBench VLM track,
    whose end2end evaluator matches predictions per page (`<stem>_<page>.md`)
    rather than per document. Pages present in `doc.pages` but with no body
    items still get an entry (empty string) so prediction files stay aligned
    with the ground-truth page set.
    """
    by_page: dict[int, list[str]] = {p: [] for p in doc.pages}
    for iid in doc.body_order:
        item = doc.items.get(iid)
        if item is None:
            continue
        by_page.setdefault(item.provenance.page_no, []).append(iid)
    return {
        page_no: to_markdown(doc.model_copy(update={"body_order": iids}), raw=raw)
        for page_no, iids in by_page.items()
    }


def to_markdown(doc: GlasanaDocument, raw: bool = False) -> str:
    """Render body content as GitHub-Flavored Markdown.

    With `raw=True`, emit each body item's text verbatim (joined by blank
    lines) without any label-derived markup. This mirrors specialized parsers
    that already produce their own markdown per block (e.g. MinerU's
    `images_to_markdown`, which concatenates each block's `content`), so
    OmniDocBench scores the model's own text rather than text we re-marked-up.
    """
    if raw:
        parts = [
            item.text
            for item in doc.iter_body()
            if getattr(item, "text", "").strip()
        ]
        return "\n\n".join(parts)

    lines: list[str] = []
    seen: set[str] = set()  # item_ids already emitted (e.g. captions inside figures)

    for item in doc.iter_body():
        if item.item_id in seen:
            continue

        if isinstance(item, AbandonItem):
            continue

        elif isinstance(item, HeadlineItem):
            lines.append(f"# {item.text}\n")

        elif isinstance(item, KickerItem):
            lines.append(f"*{item.text}*\n")

        elif isinstance(item, DeckItem):
            lines.append(f"**{item.text}**\n")

        elif isinstance(item, SubheadItem):
            lines.append(f"## {item.text}\n")

        elif isinstance(item, SubsubheadItem):
            lines.append(f"### {item.text}\n")

        elif isinstance(item, (AuthorItem, BylineItem, TranslatorItem)):
            lines.append(f"*{item.text}*\n")

        elif isinstance(item, ParagraphItem):
            lines.append(f"{item.text}\n")

        elif isinstance(item, QuoteItem):
            lines.append(f"> {item.text}\n")

        elif isinstance(item, DropcapItem):
            lines.append(f"{item.text}")  # drop-cap letter; next item continues the paragraph

        elif isinstance(item, FigureItem):
            alt = item.alt_text or "Figure"
            src = item.image_path or ""
            lines.append(f"![{alt}]({src})\n")
            # Emit caption and figbyline inline
            for sub in doc.iter_body():
                if isinstance(sub, (CaptionItem, FigBylineItem)) and sub.figure_id == item.item_id:
                    lines.append(f"*{sub.text}*\n")
                    seen.add(sub.item_id)

        elif isinstance(item, (CaptionItem, FigBylineItem)):
            # Only reaches here if orphaned (no matching FigureItem)
            lines.append(f"*{item.text}*\n")

        elif isinstance(item, TableItem):
            if item.table_data:
                lines.extend(_table_to_markdown(item.table_data))
            else:
                lines.append(f"```\n{item.raw_text}\n```\n")

        elif isinstance(item, (OrderedListItem, UnorderedListItem)):
            if item.items:
                for li in sorted(item.items, key=lambda x: x.index):
                    bullet = f"{li.index + 1}." if isinstance(item, OrderedListItem) else "-"
                    lines.append(f"{bullet} {li.text}")
                lines.append("")
            else:
                lines.append(item.raw_text)

        elif isinstance(item, FootnoteItem):
            lines.append(f"[^note]: {item.text}\n")

        elif isinstance(item, TOCItem):
            lines.append(f"*Contents:* {item.text}\n")

        elif isinstance(item, (DatelineItem, SectionItem, EditNoteItem, MarginNoteItem)):
            lines.append(f"_{item.text}_\n")

        elif isinstance(item, (LiteraryItem, LiteratureItem)):
            lines.append(f"{item.text}\n")

        elif isinstance(item, QuestionItem):
            lines.append(f"**Q:** {item.text}\n")

        elif isinstance(item, AdvertisementItem):
            pass  # omit ads from Markdown

        elif hasattr(item, "text"):
            lines.append(f"{item.text}\n")

    return "\n".join(lines)


def _table_to_markdown(td: TableData) -> list[str]:
    grid = [[""] * td.num_cols for _ in range(td.num_rows)]
    for cell in td.cells:
        grid[cell.row][cell.col] = cell.text
    lines = []
    lines.append("| " + " | ".join(grid[0]) + " |")
    lines.append("| " + " | ".join(["---"] * td.num_cols) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def to_html(doc: GlasanaDocument, wrap_articles: bool = True, css_path: str = "glasana.css") -> str:
    """Render body content as semantic HTML5.

    Walks body_order directly so items always appear in reading order.
    <article> tags open/close as article_id changes — orphaned items
    (no article_id) are wrapped in their own anonymous <article> at the
    correct position in the flow, not dumped at the end.
    """
    title = doc.source_pdf or "Glasana"
    parts = [
        "<!DOCTYPE html>",
        "<html lang='sl'>",
        "<head>",
        f'  <meta charset="UTF-8"/>',
        f'  <meta name="viewport" content="width=device-width, initial-scale=1"/>',
        f'  <title>{title}</title>',
        f'  <link rel="stylesheet" href="{css_path}"/>',
        "</head>",
        "<body>",
        '<div class="magazine">',
    ]
    seen: set[str] = set()

    if wrap_articles:
        current_art_id: Optional[str] = "##sentinel##"  # forces first open

        for item in doc.iter_body():
            if item.article_id != current_art_id:
                # Close previous article if one was open
                if current_art_id != "##sentinel##":
                    parts += ["</div>", "</article>"]
                current_art_id = item.article_id
                art = doc.articles.get(current_art_id) if current_art_id else None
                art_id_attr = f' id="{current_art_id}"' if current_art_id else ""
                parts += [f"<article{art_id_attr}>", '<div class="article-body">']

            parts.extend(_item_to_html(item, doc, seen))

        if current_art_id != "##sentinel##":
            parts += ["</div>", "</article>"]
    else:
        parts.append('<div class="article-body">')
        for item in doc.iter_body():
            parts.extend(_item_to_html(item, doc, seen))
        parts.append("</div>")

    parts += ["</div>", "</body>", "</html>"]
    return "\n".join(parts)


def _prov_attrs(item: DocItem) -> str:
    b = item.provenance.bbox
    return (
        f'data-page="{item.provenance.page_no}" '
        f'data-bbox="{b["x1"]},{b["y1"]},{b["x2"]},{b["y2"]}"'
    )


def _item_to_html(
    item: AnyDocItem,
    doc: GlasanaDocument,
    seen: set[str],
) -> list[str]:
    if item.item_id in seen:
        return []
    seen.add(item.item_id)
    p = _prov_attrs(item)

    if isinstance(item, AbandonItem):
        return []

    elif isinstance(item, HeadlineItem):
        return [f"<h1 {p}>{item.text}</h1>"]

    elif isinstance(item, KickerItem):
        return [f'<p class="kicker" {p}>{item.text}</p>']

    elif isinstance(item, DeckItem):
        return [f'<p class="deck" {p}>{item.text}</p>']

    elif isinstance(item, SubheadItem):
        return [f"<h2 {p}>{item.text}</h2>"]

    elif isinstance(item, SubsubheadItem):
        return [f"<h3 {p}>{item.text}</h3>"]

    elif isinstance(item, (AuthorItem, BylineItem)):
        css = item.category.value.lower()
        return [f'<address class="{css}" {p}>{item.text}</address>']

    elif isinstance(item, TranslatorItem):
        return [f'<p class="translator" {p}>{item.text}</p>']

    elif isinstance(item, ParagraphItem):
        return [f"<p {p}>{item.text}</p>"]

    elif isinstance(item, QuoteItem):
        return [f"<blockquote {p}>{item.text}</blockquote>"]

    elif isinstance(item, DropcapItem):
        return [f'<span class="dropcap" {p}>{item.text}</span>']

    elif isinstance(item, FigureItem):
        src = item.image_path or ""
        alt = item.alt_text or "Figure"
        html = [f'<figure id="{item.item_id}" {p}>', f'  <img src="{src}" alt="{alt}"/>']
        for iid in doc.body_order:
            sub = doc.items.get(iid)
            if sub and isinstance(sub, CaptionItem) and sub.figure_id == item.item_id:
                html.append(f"  <figcaption>{sub.text}</figcaption>")
                seen.add(sub.item_id)
            elif sub and isinstance(sub, FigBylineItem) and sub.figure_id == item.item_id:
                html.append(f'  <cite class="fig-byline">{sub.text}</cite>')
                seen.add(sub.item_id)
        html.append("</figure>")
        return html

    elif isinstance(item, (CaptionItem, FigBylineItem)):
        css = item.category.value.lower()
        return [f'<p class="{css}" {p}>{item.text}</p>']

    elif isinstance(item, TableItem):
        if item.table_data:
            return _tabledata_to_html(item.table_data, p)
        return [f'<pre class="table-fallback" {p}>{item.raw_text}</pre>']

    elif isinstance(item, OrderedListItem):
        if item.items:
            rows = [f"<ol {p}>"]
            rows += [f"  <li>{li.text}</li>" for li in sorted(item.items, key=lambda x: x.index)]
            rows += ["</ol>"]
            return rows
        return [f"<pre {p}>{item.raw_text}</pre>"]

    elif isinstance(item, UnorderedListItem):
        if item.items:
            rows = [f"<ul {p}>"]
            rows += [f"  <li>{li.text}</li>" for li in sorted(item.items, key=lambda x: x.index)]
            rows += ["</ul>"]
            return rows
        return [f"<pre {p}>{item.raw_text}</pre>"]

    elif isinstance(item, FootnoteItem):
        return [f'<aside class="footnote" {p}>{item.text}</aside>']

    elif isinstance(item, TOCItem):
        return [f'<nav class="toc" {p}>{item.text}</nav>']

    elif isinstance(item, QuestionItem):
        return [f'<p class="question" {p}>{item.text}</p>']

    elif isinstance(item, AdvertisementItem):
        return [f'<aside class="advertisement" {p}></aside>']

    elif hasattr(item, "text"):
        css = item.category.value.lower()
        return [f'<p class="{css}" {p}>{item.text}</p>']

    return []


def _tabledata_to_html(td: TableData, prov: str) -> list[str]:
    grid = [[""] * td.num_cols for _ in range(td.num_rows)]
    header = [[False] * td.num_cols for _ in range(td.num_rows)]
    for cell in td.cells:
        grid[cell.row][cell.col] = cell.text
        header[cell.row][cell.col] = cell.is_header
    html = [f"<table {prov}>"]
    for r, row in enumerate(grid):
        html.append("  <tr>")
        for c, text in enumerate(row):
            tag = "th" if header[r][c] else "td"
            html.append(f"    <{tag}>{text}</{tag}>")
        html.append("  </tr>")
    html.append("</table>")
    return html
