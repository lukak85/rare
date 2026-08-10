"""Cut a column of independent short pieces into one article per piece.

The seed rule in `rare.parse.assemble` opens an article at every Headline and
closes it at the next one. Magazine columns break that assumption: a page of
record reviews, readers' letters or news briefs carries a single Headline for
the whole column ("PISMA", "IMPROVIZIRANA GLASBA") and sets each individual
piece under a Subhead. Everything after the first piece is swallowed, so a
column of fifteen news briefs arrives as one 45-item article.

A Subhead is not a reliable boundary on its own — a feature article uses them
for internal section breaks ("NALOGA:", twice, inside "KAKO POSLUŠATI GLASBO").
What separates the two is density: a column opens a piece every few items, a
feature every twenty. This pass splits only where the headings come thick
enough, with the thresholds relaxed for sections known to run as columns.

Each piece records an "article-split" Link back to the column, and carries
`Article.split_from` so `rare.link.crosspage` cannot merge the column back
together on the strength of the shared running header and entities.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, Optional

from rare.doc.schema import (
    Article,
    ContentLayer,
    GlasanaDocument,
    Link,
    RegionCategory,
    TextItem,
)
from rare.link.articles import header_tokens, running_headers, same_section
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

# Headings that can open a piece inside a column. Headline is excluded: it
# already opens an article under the seed rule.
HEADING_CATEGORIES: frozenset[RegionCategory] = frozenset({
    RegionCategory.SUBHEAD,
    RegionCategory.SUBSUBHEAD,
})


# "2. Rebusa", "(3) …" — a numbered clue or list item, not a piece title.
_ENUMERATED = re.compile(r"^\(?\d{1,2}\s*[.)]")


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so 'Plošče' matches 'plosce'."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def split_group(article: Article) -> str:
    """The column an article belongs to — itself, when it was never split."""
    return article.split_from or article.article_id


def _is_column_section(section: Optional[str], keywords: Iterable[str]) -> bool:
    """True when the running header names a section that runs as a column.

    Substring rather than equality: the scans mirror the facing page's header
    through the paper, so the section name arrives with noise attached to it
    ("IAHHCIO ODMEVI", "3PTOZI IZDAJE").
    """
    folded = _fold(section or "")
    if not folded:
        return False
    return any(_fold(keyword) in folded for keyword in keywords)


def _titles_a_piece(text: str, config: LinkConfig) -> bool:
    """Whether a heading reads like the title of a self-contained piece.

    A Subhead can also be internal scaffolding: a quiz introduces each quoted
    passage with a lead-in ending in a colon, a crossword numbers its clues.
    Those head a paragraph, not a piece, and splitting on them shreds one item
    into a dozen fragments.
    """
    text = (text or "").strip()
    if not text:
        return False
    if config.split_skip_colon_headings and text.endswith(":"):
        return False
    if config.split_skip_enumerated_headings and _ENUMERATED.match(text):
        return False
    return True


def _piece_starts(
    doc: GlasanaDocument, article: Article, config: LinkConfig
) -> list[int]:
    """Indices into `article.item_ids` where a new piece begins.

    A Subhead immediately followed by a Subsubhead is one piece's heading, not
    two pieces, so only the first index of each run of headings is returned.
    """
    starts: list[int] = []
    previous_was_heading = False
    for index, item_id in enumerate(article.item_ids):
        item = doc.items.get(item_id)
        is_heading = (
            item is not None
            and item.category in HEADING_CATEGORIES
            and _titles_a_piece(getattr(item, "text", ""), config)
        )
        if is_heading and not previous_was_heading:
            starts.append(index)
        previous_was_heading = is_heading
    return starts


def _body_count(doc: GlasanaDocument, article: Article) -> int:
    """How many of the article's items are content rather than page chrome."""
    total = 0
    for item_id in article.item_ids:
        item = doc.items.get(item_id)
        if item is not None and item.content_layer == ContentLayer.BODY:
            total += 1
    return total


def _decide(
    doc: GlasanaDocument,
    article: Article,
    starts: list[int],
    config: LinkConfig,
) -> tuple[bool, str, float]:
    """Whether `article` is a column. Returns (split?, method, density)."""
    if not starts:
        return False, "", 0.0

    in_column_section = _is_column_section(article.section, config.column_sections)
    if in_column_section:
        min_headings = config.column_section_min_headings
        max_items = config.column_section_max_items_per_heading
        method = "section+density"
    else:
        min_headings = config.split_min_headings
        max_items = config.split_max_items_per_heading
        method = "density"

    if len(starts) < min_headings:
        return False, method, 0.0

    density = _body_count(doc, article) / len(starts)
    return density <= max_items, method, density


def _piece_title(doc: GlasanaDocument, item_id: str) -> str:
    """The title a piece takes from the item it opens on.

    Only a heading gives a title. A piece cut at a change of section usually
    opens mid-flow on a paragraph, and that paragraph's first words are not a
    title — such a piece stays untitled, exactly as a jump continuation does.
    """
    item = doc.items.get(item_id)
    if isinstance(item, TextItem) and item.category in HEADING_CATEGORIES:
        return " ".join((item.text or "").split())
    return ""


def split_article(
    doc: GlasanaDocument,
    article: Article,
    starts: list[int],
    method: str = "density",
    density: float = 0.0,
) -> list[Article]:
    """Open a new article at each start index. Returns the pieces created.

    Whatever precedes the first heading — the column's own Headline and any
    standfirst — stays on `article`, which keeps its identity as the column.
    When the column had nothing but headings, it ends up empty and the next
    `articles.rebuild` drops it.
    """
    group = split_group(article)
    pieces: list[Article] = []

    for position, start in enumerate(starts):
        is_last = position + 1 == len(starts)
        end = len(article.item_ids) if is_last else starts[position + 1]
        piece_ids = article.item_ids[start:end]
        if not piece_ids:
            continue

        piece = Article(
            title=_piece_title(doc, piece_ids[0]),
            item_ids=list(piece_ids),
            section=article.section,
            split_from=group,
        )
        doc.add_article(piece)
        for item_id in piece_ids:
            item = doc.items.get(item_id)
            if item is not None:
                item.article_id = piece.article_id

        doc.add_link(
            Link(
                kind="article-split",
                from_id=article.article_id,
                to_id=piece.article_id,
                score=round(density, 4),
                method=method,
                evidence=[piece.title] if piece.title else [],
            )
        )
        pieces.append(piece)

    article.item_ids = article.item_ids[: starts[0]]
    return pieces


def _article_pages(doc: GlasanaDocument, article: Article) -> list[int]:
    return sorted(
        {
            doc.items[item_id].provenance.page_no
            for item_id in article.item_ids
            if item_id in doc.items
        }
    )


def _section_boundaries(
    doc: GlasanaDocument,
    article: Article,
    headers: dict[int, str],
    config: LinkConfig,
) -> list[int]:
    """Indices into `article.item_ids` where the running header changes section.

    Pages whose header was never detected, or whose header survives cleaning as
    nothing usable, are passed over rather than treated as a change — the
    comparison carries forward the last header actually read.
    """
    pages = _article_pages(doc, article)
    if len(pages) < 2:
        return []

    changed_at: list[int] = []
    previous: Optional[str] = None
    for page_no in pages:
        header = headers.get(page_no, "")
        if not header_tokens(header):
            continue
        if previous is not None and not same_section(
            previous,
            header,
            config.section_change_max_similarity,
            config.section_change_min_char_similarity,
        ):
            changed_at.append(page_no)
        previous = header

    starts: set[int] = set()
    for page_no in changed_at:
        for index, item_id in enumerate(article.item_ids):
            item = doc.items.get(item_id)
            if item is not None and item.provenance.page_no >= page_no:
                # Index 0 would leave the article empty and buy nothing: the
                # whole of it already sits in the new section.
                if index > 0:
                    starts.add(index)
                break

    # Drop a boundary that would shear off only a stray item or two — that is
    # an item on the wrong side of a page break, not a second piece.
    ordered = sorted(starts)
    kept: list[int] = []
    for position, start in enumerate(ordered):
        is_last = position + 1 == len(ordered)
        end = len(article.item_ids) if is_last else ordered[position + 1]
        if end - start >= config.section_change_min_piece_items:
            kept.append(start)
    return kept


def split_section_changes(doc: GlasanaDocument, config: LinkConfig) -> int:
    """Cut every article that runs across a change of section. Returns pieces.

    The seed rule keeps appending to the open article until the next Headline,
    so a piece whose continuation was never given one swallows whatever follows
    it — including the start of the next section. The running header says where
    that happened: the magazine changed section, so the article ended.

    Needs `articles.rebuild` to have run, and should be followed by another, to
    give the pieces their page spans and their own section.
    """
    if not config.split_section_changes:
        return 0

    headers = running_headers(doc)
    created = 0
    for article in list(doc.iter_articles()):
        starts = _section_boundaries(doc, article, headers, config)
        if not starts:
            continue
        pieces = split_article(doc, article, starts, method="section-change")
        if pieces:
            created += len(pieces)
            logger.debug(
                "split %r across a section change into %d pieces",
                article.title,
                len(pieces) + 1,
            )

    logger.info("split %d pieces off at a change of section", created)
    return created


def split_columns(doc: GlasanaDocument, config: LinkConfig) -> int:
    """Split every column in `doc` into its pieces. Returns pieces created.

    Item membership is repointed here; `articles.rebuild` should run afterwards
    to give each new piece its page span and to drop a column left empty.
    """
    if not config.split_columns:
        return 0

    created = 0
    columns = 0
    for article in list(doc.iter_articles()):
        starts = _piece_starts(doc, article, config)
        should_split, method, density = _decide(doc, article, starts, config)
        if not should_split:
            continue
        pieces = split_article(doc, article, starts, method, density)
        if pieces:
            columns += 1
            created += len(pieces)
            logger.debug(
                "split %r (%d items, %.1f per heading) into %d pieces",
                article.title,
                len(article.item_ids) + sum(len(p.item_ids) for p in pieces),
                density,
                len(pieces),
            )

    logger.info("split %d columns into %d pieces", columns, created)
    return created