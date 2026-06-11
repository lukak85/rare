"""Intermediate document representation for Glasana magazine layout parsing.

Bridges layout detection output (COCO JSON + reading-order graph + extracted text)
and final render formats (HTML, Markdown).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Union
try:
    from typing import Annotated, Literal  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated, Literal  # Python 3.7 / 3.8

from pydantic import BaseModel, Field
from typing import List, Dict


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
    # cells: list[TableCell]
    cells: List[TableCell]

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
    # sub_items: list[ListItem] = Field(default_factory=list)
    sub_items: List[ListItem] = Field(default_factory=list)


ListItem.model_rebuild()  # resolve forward ref


class OrderedListItem(DocItem):
    category: Literal[RegionCategory.ORDERED_LIST] = RegionCategory.ORDERED_LIST
    # items: list[ListItem] = Field(default_factory=list)
    items: List[ListItem] = Field(default_factory=list)
    raw_text: str = ""


class UnorderedListItem(DocItem):
    category: Literal[RegionCategory.UNORDERED_LIST] = RegionCategory.UNORDERED_LIST
    # items: list[ListItem] = Field(default_factory=list)
    items: List[ListItem] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Article grouping
# ---------------------------------------------------------------------------

class Article(BaseModel):
    """A logical grouping of DocItems belonging to the same editorial piece."""
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    # item_ids: list[str] = Field(default_factory=list)
    item_ids: List[str] = Field(default_factory=list)


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
    # items: dict[str, AnyDocItem] = Field(default_factory=dict)
    items: Dict[str, AnyDocItem] = Field(default_factory=dict)
    # body_order: list[str] = Field(default_factory=list)
    body_order: List[str] = Field(default_factory=list)
    # articles: dict[str, Article] = Field(default_factory=dict)
    articles: Dict[str, Article] = Field(default_factory=dict)
    # pages: dict[int, PageInfo] = Field(default_factory=dict)
    pages: Dict[int, PageInfo] = Field(default_factory=dict)

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
# Label → DocItem class lookup
# ---------------------------------------------------------------------------
# Maps the string label produced by layout detectors (RegionCategory values)
# to the concrete DocItem subclass used during document assembly.

LABEL_TO_CLASS: dict[str, type] = {
    "Header":         HeaderItem,
    "Footer":         FooterItem,
    "PageNum":        PageNumItem,
    "Section":        SectionItem,
    "Dateline":       DatelineItem,
    "EditNote":       EditNoteItem,
    "MarginNote":     MarginNoteItem,
    "Headline":       HeadlineItem,
    "Kicker":         KickerItem,
    "Deck":           DeckItem,
    "Subhead":        SubheadItem,
    "Subsubhead":     SubsubheadItem,
    "Author":         AuthorItem,
    "Byline":         BylineItem,
    "Translator":     TranslatorItem,
    "Paragraph":      ParagraphItem,
    "Quote":          QuoteItem,
    "Dropcap":        DropcapItem,
    "Figure":         FigureItem,
    "Caption":        CaptionItem,
    "FigByline":      FigBylineItem,
    "Table":          TableItem,
    "OrderedList":    OrderedListItem,
    "UnorderedList":  UnorderedListItem,
    "Footnote":       FootnoteItem,
    "TOC":            TOCItem,
    "Literary":       LiteraryItem,
    "Literature":     LiteratureItem,
    "Advertisement": AdvertisementItem,
    "Question":       QuestionItem,
    "Abandon":        AbandonItem,
}

