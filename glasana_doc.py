"""Intermediate document representation for Glasana magazine layout parsing.

Bridges layout detection output (COCO JSON + reading-order graph + extracted text)
and final render formats (HTML, Markdown).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Iterable, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RegionCategory(str, Enum):
    """All region categories used in the Glasana annotation scheme.

    Source of truth: GLASANA_COLOR_MAP in utils/displayutils.py.
    """
    # Page furniture
    HEADER         = "Header"
    FOOTER         = "Footer"
    PAGE_NUM       = "PageNum"
    SECTION        = "Section"
    DATELINE       = "Dateline"
    EDIT_NOTE      = "EditNote"
    MARGIN_NOTE    = "MarginNote"
    # Article structure
    HEADLINE       = "Headline"
    KICKER         = "Kicker"
    DECK           = "Deck"
    SUBHEAD        = "Subhead"
    SUBSUBHEAD     = "Subsubhead"
    AUTHOR         = "Author"
    BYLINE         = "Byline"
    TRANSLATOR     = "Translator"
    # Content
    PARAGRAPH      = "Paragraph"
    QUOTE          = "Quote"
    DROPCAP        = "Dropcap"
    # Figures
    FIGURE         = "Figure"
    CAPTION        = "Caption"
    FIG_BYLINE     = "FigByline"
    # Structured
    TABLE          = "Table"
    ORDERED_LIST   = "OrderedList"
    UNORDERED_LIST = "UnorderedList"
    # Reference
    FOOTNOTE       = "Footnote"
    TOC            = "TOC"
    LITERARY       = "Literary"
    LITERATURE     = "Literature"
    # Other
    ADVERTISEMENT  = "Advertisement"
    QUESTION       = "Question"
    ABANDON        = "Abandon"


class ContentLayer(str, Enum):
    BODY      = "body"
    FURNITURE = "furniture"


FURNITURE_CATEGORIES: frozenset[RegionCategory] = frozenset({
    RegionCategory.HEADER,
    RegionCategory.FOOTER,
    RegionCategory.PAGE_NUM,
    RegionCategory.SECTION,
    RegionCategory.DATELINE,
    RegionCategory.EDIT_NOTE,
    RegionCategory.MARGIN_NOTE,
    RegionCategory.ADVERTISEMENT,
    RegionCategory.ABANDON,
})


# ---------------------------------------------------------------------------
# BBox — frozen dataclass (pure value object, not Pydantic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates.

    Convention: (x1, y1) = top-left, (x2, y2) = bottom-right, origin top-left.
    Matches lp.Rectangle and pdfplumber (after coordinate-flip).
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @classmethod
    def from_coco(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Convert from COCO [x, y, w, h] format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_norm_1000(
        cls,
        coords: list[float],
        page_width: float,
        page_height: float,
    ) -> BBox:
        """Convert from connections.json normalized 0-1000 format [x0,y0,x1,y1]."""
        x0, y0, x1, y1 = coords
        return cls(
            x1=x0 / 1000.0 * page_width,
            y1=y0 / 1000.0 * page_height,
            x2=x1 / 1000.0 * page_width,
            y2=y1 / 1000.0 * page_height,
        )

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


# ---------------------------------------------------------------------------
# Provenance & PageInfo
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Links a content item back to its source location in the original document."""
    page_no: int
    bbox: dict                          # BBox.to_dict() result
    detection_score: Optional[float] = None
    source_region_id: Optional[str] = None

    @classmethod
    def from_bbox(
        cls,
        page_no: int,
        bbox: BBox,
        detection_score: Optional[float] = None,
        source_region_id: Optional[str] = None,
    ) -> Provenance:
        return cls(
            page_no=page_no,
            bbox=bbox.to_dict(),
            detection_score=detection_score,
            source_region_id=source_region_id,
        )

    def get_bbox(self) -> BBox:
        return BBox(**self.bbox)


class PageInfo(BaseModel):
    page_no: int
    width: float
    height: float
    source_file: str = ""


# ---------------------------------------------------------------------------
# DocItem base + TextItem
# ---------------------------------------------------------------------------

class DocItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: RegionCategory
    content_layer: ContentLayer = ContentLayer.BODY
    provenance: Provenance
    article_id: Optional[str] = None
    reading_order: Optional[int] = None


class TextItem(DocItem):
    text: str = ""


# ---------------------------------------------------------------------------
# All text-bearing region types
# ---------------------------------------------------------------------------

class HeadlineItem(TextItem):
    category: Literal[RegionCategory.HEADLINE] = RegionCategory.HEADLINE

class KickerItem(TextItem):
    category: Literal[RegionCategory.KICKER] = RegionCategory.KICKER

class DeckItem(TextItem):
    category: Literal[RegionCategory.DECK] = RegionCategory.DECK

class SubheadItem(TextItem):
    category: Literal[RegionCategory.SUBHEAD] = RegionCategory.SUBHEAD

class SubsubheadItem(TextItem):
    category: Literal[RegionCategory.SUBSUBHEAD] = RegionCategory.SUBSUBHEAD

class AuthorItem(TextItem):
    category: Literal[RegionCategory.AUTHOR] = RegionCategory.AUTHOR

class BylineItem(TextItem):
    category: Literal[RegionCategory.BYLINE] = RegionCategory.BYLINE

class TranslatorItem(TextItem):
    category: Literal[RegionCategory.TRANSLATOR] = RegionCategory.TRANSLATOR

class ParagraphItem(TextItem):
    category: Literal[RegionCategory.PARAGRAPH] = RegionCategory.PARAGRAPH

class QuoteItem(TextItem):
    category: Literal[RegionCategory.QUOTE] = RegionCategory.QUOTE

class DropcapItem(TextItem):
    category: Literal[RegionCategory.DROPCAP] = RegionCategory.DROPCAP

class FootnoteItem(TextItem):
    category: Literal[RegionCategory.FOOTNOTE] = RegionCategory.FOOTNOTE

class TOCItem(TextItem):
    category: Literal[RegionCategory.TOC] = RegionCategory.TOC

class LiteraryItem(TextItem):
    category: Literal[RegionCategory.LITERARY] = RegionCategory.LITERARY

class LiteratureItem(TextItem):
    category: Literal[RegionCategory.LITERATURE] = RegionCategory.LITERATURE

class QuestionItem(TextItem):
    category: Literal[RegionCategory.QUESTION] = RegionCategory.QUESTION

class DatelineItem(TextItem):
    category: Literal[RegionCategory.DATELINE] = RegionCategory.DATELINE

class SectionItem(TextItem):
    category: Literal[RegionCategory.SECTION] = RegionCategory.SECTION

class EditNoteItem(TextItem):
    category: Literal[RegionCategory.EDIT_NOTE] = RegionCategory.EDIT_NOTE

class MarginNoteItem(TextItem):
    category: Literal[RegionCategory.MARGIN_NOTE] = RegionCategory.MARGIN_NOTE

class AdvertisementItem(TextItem):
    category: Literal[RegionCategory.ADVERTISEMENT] = RegionCategory.ADVERTISEMENT
    content_layer: ContentLayer = ContentLayer.FURNITURE

class AbandonItem(TextItem):
    category: Literal[RegionCategory.ABANDON] = RegionCategory.ABANDON
    content_layer: ContentLayer = ContentLayer.FURNITURE

# Page furniture
class HeaderItem(TextItem):
    category: Literal[RegionCategory.HEADER] = RegionCategory.HEADER
    content_layer: ContentLayer = ContentLayer.FURNITURE

class FooterItem(TextItem):
    category: Literal[RegionCategory.FOOTER] = RegionCategory.FOOTER
    content_layer: ContentLayer = ContentLayer.FURNITURE

class PageNumItem(TextItem):
    category: Literal[RegionCategory.PAGE_NUM] = RegionCategory.PAGE_NUM
    content_layer: ContentLayer = ContentLayer.FURNITURE


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class FigureItem(DocItem):
    category: Literal[RegionCategory.FIGURE] = RegionCategory.FIGURE
    image_path: Optional[str] = None
    alt_text: str = ""


class CaptionItem(TextItem):
    category: Literal[RegionCategory.CAPTION] = RegionCategory.CAPTION
    figure_id: Optional[str] = None  # item_id of the associated FigureItem


class FigBylineItem(TextItem):
    category: Literal[RegionCategory.FIG_BYLINE] = RegionCategory.FIG_BYLINE
    figure_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    is_header: bool = False


class TableData(BaseModel):
    num_rows: int
    num_cols: int
    cells: list[TableCell]

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None


class TableItem(DocItem):
    category: Literal[RegionCategory.TABLE] = RegionCategory.TABLE
    table_data: Optional[TableData] = None  # None when structure extraction not run
    raw_text: str = ""                      # fallback: all OCR text concatenated


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

class ListItem(BaseModel):
    """A single item within an ordered or unordered list. Not a DocItem (no provenance)."""
    index: int
    text: str
    sub_items: list[ListItem] = Field(default_factory=list)


ListItem.model_rebuild()  # resolve forward ref


class OrderedListItem(DocItem):
    category: Literal[RegionCategory.ORDERED_LIST] = RegionCategory.ORDERED_LIST
    items: list[ListItem] = Field(default_factory=list)
    raw_text: str = ""


class UnorderedListItem(DocItem):
    category: Literal[RegionCategory.UNORDERED_LIST] = RegionCategory.UNORDERED_LIST
    items: list[ListItem] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Article grouping
# ---------------------------------------------------------------------------

class Article(BaseModel):
    """A logical grouping of DocItems belonging to the same editorial piece."""
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    item_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discriminated union for all item types
# ---------------------------------------------------------------------------

AnyDocItem = Annotated[
    Union[
        HeadlineItem, KickerItem, DeckItem, SubheadItem, SubsubheadItem,
        AuthorItem, BylineItem, TranslatorItem,
        ParagraphItem, QuoteItem, DropcapItem,
        FigureItem, CaptionItem, FigBylineItem,
        TableItem, OrderedListItem, UnorderedListItem,
        FootnoteItem, TOCItem, LiteraryItem, LiteratureItem, QuestionItem,
        DatelineItem, SectionItem, EditNoteItem, MarginNoteItem,
        AdvertisementItem, AbandonItem,
        HeaderItem, FooterItem, PageNumItem,
    ],
    Field(discriminator="category"),
]


# ---------------------------------------------------------------------------
# Top-level GlasanaDocument
# ---------------------------------------------------------------------------

class GlasanaDocument(BaseModel):
    """Top-level intermediate document representation.

    Hybrid design (inspired by Docling):
    - Flat dict  ``items``      — efficient lookup by item_id
    - Ordered list ``body_order`` — reading order for body items (furniture excluded)
    - Article index ``articles`` — logical groupings of items
    - Page index  ``pages``     — per-page image metadata
    """
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_pdf: str = ""
    items: dict[str, AnyDocItem] = Field(default_factory=dict)
    body_order: list[str] = Field(default_factory=list)
    articles: dict[str, Article] = Field(default_factory=dict)
    pages: dict[int, PageInfo] = Field(default_factory=dict)

    def get_item(self, item_id: str) -> Optional[AnyDocItem]:
        return self.items.get(item_id)

    def iter_body(self) -> Iterable[AnyDocItem]:
        """Yield body items in reading order."""
        for iid in self.body_order:
            item = self.items.get(iid)
            if item is not None:
                yield item

    def iter_furniture(self) -> Iterable[AnyDocItem]:
        """Yield furniture items (not in body_order)."""
        body_set = set(self.body_order)
        for iid, item in self.items.items():
            if iid not in body_set:
                yield item

    def iter_article(self, article_id: str) -> Iterable[AnyDocItem]:
        """Yield items of a specific article in reading order."""
        art = self.articles.get(article_id)
        if art is None:
            return
        for iid in art.item_ids:
            item = self.items.get(iid)
            if item is not None:
                yield item

    def add_item(self, item: AnyDocItem, to_body: bool = True) -> str:
        """Register an item. Furniture items are auto-excluded from body_order."""
        self.items[item.item_id] = item
        if to_body and item.content_layer == ContentLayer.BODY:
            self.body_order.append(item.item_id)
        return item.item_id

    def add_article(self, article: Article) -> str:
        self.articles[article.article_id] = article
        return article.article_id


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def to_markdown(doc: GlasanaDocument) -> str:
    """Render body content as GitHub-Flavored Markdown."""
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
