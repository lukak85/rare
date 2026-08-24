"""Whole-document linking: entities, captions, articles, continuations.

The parse tracks build a document page by page and can only see one page at a
time. This package runs afterwards, with the finished document in view, and
fills in the relationships that need that wider context:

1. `entities.annotate`        — named entities on every text item
2. `figure_matching.link_captions` — caption/photo-credit -> figure (geometry)
3. `articles.rebuild`         — articles made authoritative, ordered, dated
4. `split.split_section_changes` — an article stops where its section does
5. `split.split_columns`      — a column of short pieces becomes one each
6. `crosspage.merge_continuations` — a piece split across pages becomes one
7. `figure_link.link_figures` — figure + its caption -> article (geometry+NER)
8. `entities.cross_link`      — items sharing a rare entity, across articles
9. `classify.classify_articles` — editorial genre per finished article

Order matters: figures are placed after the merge so the article they are
given is the final one, and the merge runs after `rebuild` so it can use page
spans and section headers. The column split sits between the two — it needs
the section header `rebuild` computes, and running it before the merge keeps
the two passes from undoing each other's work.

Every inference is also recorded in `doc.links` with its method, score and
evidence, so nothing the linker decides is invisible.
"""

from __future__ import annotations

import logging
from typing import Optional

from rare.doc.schema import GlasanaDocument
from rare.link import (
    articles,
    classify,
    crosspage,
    entities,
    figure_link,
    figure_matching,
    split,
)
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

__all__ = ["link_document", "LinkConfig"]


def link_document(
    doc: GlasanaDocument,
    ner=None,
    classifier=None,
    config: dict | LinkConfig | None = None,
) -> GlasanaDocument:
    """Run every linking pass over `doc`, in place. Returns the same document.

    `ner` and `classifier` may both be None — the passes that need them then
    contribute nothing and the geometric ones still run, so a missing model
    degrades the result rather than failing the parse.
    """
    cfg = config if isinstance(config, LinkConfig) else LinkConfig.from_dict(config)

    entities.annotate(doc, ner)

    linked_captions = figure_matching.link_captions(doc, cfg)

    index = entities.EntityIndex(doc, cfg)
    articles.rebuild(doc, index, cfg)

    # Two ways an article holds more than one piece, coarse before fine: it ran
    # on past the end of its section, and it is a column of short pieces. Both
    # need a `rebuild` afterwards, to give the new pieces their page spans and
    # their own section, and to drop an article left with nothing of its own.
    sections = split.split_section_changes(doc, cfg)
    if sections:
        articles.rebuild(doc, index, cfg)

    pieces = split.split_columns(doc, cfg)
    if pieces:
        articles.rebuild(doc, index, cfg)

    # Rebuild the index once articles are settled: entity rarity is measured
    # per article, so it is only meaningful after grouping is cleaned up.
    index = entities.EntityIndex(doc, cfg)
    merged = crosspage.merge_continuations(doc, index, cfg)

    # Articles are settled, so the article a figure is given here is the one it
    # will render under. `rebuild` afterwards folds the moved visuals into
    # their new article's item list and page span.
    placed = figure_link.link_figures(doc, index, cfg)
    if placed:
        articles.rebuild(doc, index, cfg)

    entities.cross_link(doc, index, cfg)

    # Last: articles are final here, so a piece that spanned two pages is
    # classified once, from its whole text.
    classified = classify.classify_articles(doc, classifier, cfg)

    logger.info(
        "linked %d captions to figures, split %d pieces at a section change "
        "and %d out of columns, merged %d continuations, placed %d figure "
        "groups in articles, classified %d articles, %d articles remain",
        linked_captions,
        sections,
        pieces,
        merged,
        placed,
        classified,
        len(doc.articles),
    )
    return doc
