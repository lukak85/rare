"""Group a document's items into articles running headline to headline.

The whole issue is laid out in reading order and cut in exactly two places:
* at a **Headline** — it opens a new article;
* where the **running header** changes to a different section.
"""

from __future__ import annotations

import logging
from typing import Optional

from rare.doc.schema import Article, ContentLayer, GlasanaDocument, RegionCategory
from rare.link.articles import _squash, header_at, header_tokens, page_headers, same_section
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

# A header whose letters, run together, are shorter than this is a fragment of
# the facing page's mirrored header ("E H"), not a section name.
_MIN_HEADER_LETTERS = 4

_LEADS_ALWAYS = frozenset({RegionCategory.KICKER})

# Placed by position, not read in sequence: a figure in the other section's part
# of the page says nothing about where the text is.
_VISUALS = frozenset({RegionCategory.FIGURE, RegionCategory.CAPTION, RegionCategory.FIG_BYLINE})
_SENTENCE_END = tuple('.!?…:;»"”)')


def runs_on(previous: Optional[str], text: Optional[str]) -> bool:
    """Whether `text` continues the sentence `previous` left unfinished.

    "…je z izjemo" followed by "ta stara glasbena zvrst…" is one text flowing
    from one column into the next, whatever header either column sits under.
    """
    previous = (previous or "").rstrip()
    first = next((c for c in (text or "") if c.isalpha()), "")
    return bool(previous) and not previous.endswith(_SENTENCE_END) and first.islower()


def readable_header(text: Optional[str]) -> bool:
    """Whether a running header names a section: a content word, or enough letters.
    """
    return bool(header_tokens(text)) or len(_squash(text)) >= _MIN_HEADER_LETTERS
_LEADS_IF_SAME_ARTICLE = frozenset({
    RegionCategory.FIGURE,
    RegionCategory.CAPTION,
    RegionCategory.FIG_BYLINE,
})


def segment_items(
    flow: list[tuple[str, int, Optional[str], RegionCategory, Optional[str], Optional[str]]],
    config: LinkConfig,
) -> list[list[str]]:
    pieces: list[list[tuple[str, RegionCategory, Optional[str], int]]] = []
    current_header: Optional[str] = None
    last_text: Optional[str] = None
    page = None

    # `header` is the running header over this item — a page can carry several
    # (see `rare.link.articles.header_at`) — or None where none is readable.
    # `text` is the item's text, to tell a new piece from one running on.
    for item_id, page_no, header, category, source_id, text in flow:
        cut = not pieces
        same_page = page_no == page
        page = page_no

        if header and readable_header(header) and category not in _VISUALS:
            if current_header is None:
                cut = True
            elif not same_section(
                current_header,
                header,
                config.section_change_max_similarity,
                config.section_change_min_char_similarity,
            ) and not (same_page and runs_on(last_text, text)):
                # Across a page turn a new header is a new section; within a
                # page it may only be the next column of the same text.
                cut = True
            current_header = header
        if text and category not in _VISUALS:
            last_text = text

        entry = (item_id, category, source_id, page_no)
        if category == RegionCategory.HEADLINE and pieces:
            # Carry the lead-in (kicker, the headline's own figure) across.
            lead = []
            last = pieces[-1]
            while last and (
                last[-1][1] in _LEADS_ALWAYS
                or (last[-1][1] in _LEADS_IF_SAME_ARTICLE and last[-1][2] == source_id)
            ):
                lead.insert(0, last.pop())
            if not last:
                # The piece was nothing but this lead-in; whatever opened it
                # (a header change, the start of the issue) now opens this one.
                pieces.pop()
                cut = True
            # Before its first Headline a piece is only a lead-in to it — the
            # opening of a new section just above it — so the Headline joins.
            # Only on the page the piece began on: a Headline opening a new
            # page starts its own article, not the tail of the last page's.
            if (
                cut
                or any(c == RegionCategory.HEADLINE for _, c, _, _ in pieces[-1])
                or pieces[-1][0][3] != page_no
            ):
                pieces.append(lead)
            else:
                pieces[-1].extend(lead)
        elif cut:
            pieces.append([])
        pieces[-1].append(entry)

    return [[item_id for item_id, _, _, _ in piece] for piece in pieces if piece]


def segment_articles(doc: GlasanaDocument, config: LinkConfig) -> int:
    """Regroup every body item of `doc` into headline/header articles. Returns the count.

    An article keeps the id of the article its headline had, where there was
    one, so links recorded against it stay valid; `articles.rebuild` should run
    afterwards to refresh page spans and sections.
    """
    zones = page_headers(doc, config.section_header_max_words, config.section_header_top_frac)
    flow = [
        (
            item.item_id,
            item.provenance.page_no,
            header_at(zones.get(item.provenance.page_no), item.provenance.bbox),
            item.category,
            item.article_id,
            getattr(item, "text", None),
        )
        for item in (doc.items.get(iid) for iid in doc.body_order)
        if item is not None and item.content_layer != ContentLayer.FURNITURE
    ]
    old = doc.articles
    doc.articles = {}

    for item_ids in segment_items(flow, config):
        headline = next(
            (
                doc.items[iid]
                for iid in item_ids
                if doc.items[iid].category == RegionCategory.HEADLINE
            ),
            None,
        )
        article = Article(title=" ".join((headline.text or "").split()) if headline else "")
        if headline is not None and headline.article_id in old and headline.article_id not in doc.articles:
            article.article_id = headline.article_id
        doc.add_article(article)
        article.item_ids = list(item_ids)
        for iid in item_ids:
            doc.items[iid].article_id = article.article_id

    logger.debug("segmented %d articles into %d", len(old), len(doc.articles))
    return len(doc.articles)
