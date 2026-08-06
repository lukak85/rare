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


def to_markdown(
    doc: GlasanaDocument, raw: bool = False, by_article: bool = False
) -> str:
    """Render body content as GitHub-Flavored Markdown.

    With `raw=True`, emit each body item's text verbatim (joined by blank
    lines) without any label-derived markup. This mirrors specialized parsers
    that already produce their own markdown per block (e.g. MinerU's
    `images_to_markdown`, which concatenates each block's `content`), so
    OmniDocBench scores the model's own text rather than text we re-marked-up.

    With `by_article=True`, group the output one article at a time, each under
    its title and page span, instead of a single flat reading-order stream.
    The default stays flat so text-similarity scores stay comparable with
    earlier runs.
    """
    if by_article and not raw:
        return _to_markdown_by_article(doc)

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


def _to_markdown_by_article(doc: GlasanaDocument) -> str:
    """One section per article, each headed by its title and page span."""
    chunks: list[str] = []
    for article_id, item_ids in _article_blocks(doc):
        article = doc.articles.get(article_id) if article_id else None

        heading: list[str] = []
        if article is not None:
            title = article.title.strip() or "(untitled)"
            heading.append(f"## {title}")
            # The article heading already shows the title; emitting the
            # HeadlineItem too would print it twice in a row.
            if item_ids:
                first = doc.items.get(item_ids[0])
                if (
                    isinstance(first, HeadlineItem)
                    and first.text.strip() == article.title.strip()
                ):
                    item_ids = item_ids[1:]
            meta = []
            if article.page_nos:
                pages = ", ".join(str(p) for p in article.page_nos)
                meta.append(f"pages {pages}")
            if article.section:
                meta.append(article.section)
            if article.continued:
                meta.append("continued")
            if meta:
                heading.append(f"*{' — '.join(meta)}*")
            heading.append("")

        body = to_markdown(doc.model_copy(update={"body_order": item_ids}))
        chunks.append("\n".join(heading + [body]).strip())

    return "\n\n".join(chunk for chunk in chunks if chunk) + "\n"


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

def _article_blocks(doc: GlasanaDocument) -> list[tuple[Optional[str], list[str]]]:
    """Split body_order into (article_id, item_ids) blocks, in document order.

    Grouping is driven by `Article.item_ids`, so every item of an article is
    emitted together even when another article's items are interleaved with it
    in reading order — which happens whenever a piece continues after a jump.
    Items belonging to no article keep their place in the flow and are grouped
    into anonymous runs rather than being dumped at the end.
    """
    position = {iid: i for i, iid in enumerate(doc.body_order)}
    body = set(doc.body_order)
    claimed: set[str] = set()

    blocks: list[tuple[int, Optional[str], list[str]]] = []
    for article in doc.articles.values():
        item_ids = [iid for iid in article.item_ids if iid in body]
        if not item_ids:
            continue
        item_ids.sort(key=lambda iid: position[iid])
        claimed.update(item_ids)
        blocks.append((position[item_ids[0]], article.article_id, item_ids))

    # Unclaimed items form anonymous blocks, one per contiguous run.
    run: list[str] = []
    for iid in doc.body_order:
        if iid in claimed:
            if run:
                blocks.append((position[run[0]], None, run))
                run = []
            continue
        run.append(iid)
    if run:
        blocks.append((position[run[0]], None, run))

    blocks.sort(key=lambda b: b[0])
    return [(article_id, item_ids) for _, article_id, item_ids in blocks]


def to_html(doc: GlasanaDocument, wrap_articles: bool = True, css_path: str = "glasana.css") -> str:
    """Render body content as semantic HTML5.

    With `wrap_articles`, each Article becomes one <article> element carrying
    its title and page span; items belonging to no article are wrapped in an
    anonymous <article> at the right point in the flow. Without it, body items
    are emitted in plain reading order.
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
        for article_id, item_ids in _article_blocks(doc):
            article = doc.articles.get(article_id) if article_id else None
            parts.append(f"<article{_article_attrs(article)}>")
            if article is not None and article.title.strip():
                parts.append(
                    f'<h1 class="article-title">{article.title}</h1>'
                )
            parts.append('<div class="article-body">')
            for iid in item_ids:
                item = doc.items.get(iid)
                if item is not None:
                    parts.extend(_item_to_html(item, doc, seen))
            parts += ["</div>", "</article>"]
    else:
        parts.append('<div class="article-body">')
        for item in doc.iter_body():
            parts.extend(_item_to_html(item, doc, seen))
        parts.append("</div>")

    parts += ["</div>", "</body>", "</html>"]
    return "\n".join(parts)


def _article_attrs(article) -> str:
    """id / section / page-span attributes for an <article> element."""
    if article is None:
        return ""
    attrs = [f'id="{article.article_id}"']
    if article.page_nos:
        attrs.append(
            'data-pages="{}"'.format(",".join(str(p) for p in article.page_nos))
        )
    if article.section:
        attrs.append(f'data-section="{article.section}"')
    if article.continued:
        attrs.append('data-continued="true"')
    return " " + " ".join(attrs)


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
