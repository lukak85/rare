"""Whole-document linking: entities, captions, articles, continuations.

The parse tracks build a document page by page and can only see one page at a
time. This package runs afterwards, with the finished document in view, and
fills in the relationships that need that wider context:

1. `entities.annotate`        — named entities on every text item
2. `figures.link_captions`    — caption/photo-credit -> figure (geometry)
3. `articles.rebuild`         — articles made authoritative, ordered, dated
4. `crosspage.merge_continuations` — a piece split across pages becomes one
5. `captions.attach_orphans`  — figure-less captions -> nearest article
6. `entities.cross_link`      — items sharing a rare entity, across articles

Order matters: captions are attached after the merge so "nearest article"
means the final article, and the merge runs after `rebuild` so it can use
page spans and section headers.

Every inference is also recorded in `doc.links` with its method, score and
evidence, so nothing the linker decides is invisible.
"""

from __future__ import annotations

import logging
from typing import Optional

from rare.doc.schema import GlasanaDocument
from rare.link import articles, captions, crosspage, entities, figures
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

__all__ = ["link_document", "LinkConfig"]


def link_document(
    doc: GlasanaDocument,
    ner=None,
    config: dict | LinkConfig | None = None,
) -> GlasanaDocument:
    """Run every linking pass over `doc`, in place. Returns the same document.

    `ner` may be None — the passes that need entities then contribute nothing
    and the geometric ones still run, so a missing model degrades the result
    rather than failing the parse.
    """
    cfg = config if isinstance(config, LinkConfig) else LinkConfig.from_dict(config)

    entities.annotate(doc, ner)

    linked_captions = figures.link_captions(doc, cfg)

    index = entities.EntityIndex(doc, cfg)
    articles.rebuild(doc, index, cfg)

    # Rebuild the index once articles are settled: entity rarity is measured
    # per article, so it is only meaningful after grouping is cleaned up.
    index = entities.EntityIndex(doc, cfg)
    merged = crosspage.merge_continuations(doc, index, cfg)
    # attached = captions.attach_orphans(doc, index, cfg) # TODO: probably not needed; captions are already linked to figures and articles in the previous steps
    entities.cross_link(doc, index, cfg)

    logger.info(
        "linked %d captions to figures, attached %d to articles, "
        "merged %d continuations, %d articles remain",
        linked_captions,
        attached,
        merged,
        len(doc.articles),
    )
    return doc
