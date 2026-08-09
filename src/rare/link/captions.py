"""Attach captions that belong to no figure to the nearest article.

A caption with no figure on its page is not noise — it is text about something,
and leaving it stranded means it renders detached from any article. The rule:
if a caption is connected to nothing on the page, connect it to the closest
article, preferring the one it shares named entities with.
"""

from __future__ import annotations

from collections import defaultdict

from rare.doc.schema import CaptionItem, FigBylineItem, GlasanaDocument, Link
from rare.link._geometry import box_of, gap_between, page_size
from rare.link.articles import refresh
from rare.link.config import LinkConfig
from rare.link.entities import EntityIndex


def _articles_on_page(doc: GlasanaDocument, page_no: int) -> dict[str, list[str]]:
    """article_id -> its item_ids that appear on this page."""
    by_article: dict[str, list[str]] = defaultdict(list)
    for item in doc.items.values():
        if item.article_id and item.provenance.page_no == page_no:
            by_article[item.article_id].append(item.item_id)
    return by_article


def _best_by_entities(
    doc: GlasanaDocument,
    caption,
    candidates: dict[str, list[str]],
    index: EntityIndex,
) -> tuple[float, str | None, list[str]]:
    caption_keys = index.keys_of(caption.item_id)
    if not caption_keys:
        return 0.0, None, []

    best_score, best_id, best_evidence = 0.0, None, []
    for article_id in sorted(candidates):
        article = doc.articles.get(article_id)
        if article is None:
            continue
        # Exclude the caption itself: it is already a member of whichever
        # article it currently sits in, and counting its own entities towards
        # that article would just re-elect the status quo.
        others = [iid for iid in article.item_ids if iid != caption.item_id]
        score, shared = index.similarity(
            caption_keys, index.keys_of_items(others)
        )
        if score > best_score:
            best_score, best_id, best_evidence = score, article_id, shared[:3]
    return best_score, best_id, best_evidence


def _best_by_geometry(
    doc: GlasanaDocument,
    caption,
    candidates: dict[str, list[str]],
    config: LinkConfig,
) -> tuple[float, str | None]:
    caption_box = box_of(caption)
    _, page_height = page_size(doc, caption.provenance.page_no)
    limit = config.orphan_max_distance_frac * page_height

    best_distance, best_id = None, None
    for article_id in sorted(candidates):
        for item_id in candidates[article_id]:
            item = doc.items.get(item_id)
            if item is None or item.item_id == caption.item_id:
                continue
            distance = gap_between(caption_box, box_of(item))
            if best_distance is None or distance < best_distance:
                best_distance, best_id = distance, article_id

    if best_id is None or best_distance is None or best_distance > limit:
        return 0.0, None
    return 1.0 - (best_distance / limit if limit else 0.0), best_id


def attach_orphans(
    doc: GlasanaDocument, index: EntityIndex, config: LinkConfig
) -> int:
    """Give every figure-less caption an article. Returns how many moved."""
    orphans = [
        item
        for item in doc.items.values()
        if isinstance(item, (CaptionItem, FigBylineItem)) and not item.figure_id
    ]

    attached = 0
    touched: set[str] = set()
    for caption in sorted(orphans, key=lambda i: i.item_id):
        page_no = caption.provenance.page_no
        # The caption's current article stays in the running: it is often
        # already right, and it should only lose to a strictly better match.
        candidates = _articles_on_page(doc, page_no)

        score, article_id, evidence = _best_by_entities(
            doc, caption, candidates, index
        )
        method = "ner"
        if article_id is None:
            score, article_id = _best_by_geometry(doc, caption, candidates, config)
            method, evidence = "geometry", []
        if article_id is None:
            # Nothing on this page to attach to; whatever the assembler
            # already decided is better than orphaning it.
            continue

        previous = caption.article_id
        if previous == article_id:
            continue

        caption.article_id = article_id
        doc.add_link(
            Link(
                kind="caption-to-article",
                from_id=caption.item_id,
                to_id=article_id,
                score=round(score, 4),
                method=method,
                evidence=evidence,
            )
        )
        touched.update(a for a in (previous, article_id) if a)
        attached += 1

    if attached:
        position = {iid: i for i, iid in enumerate(doc.body_order)}
        for article_id in touched:
            article = doc.articles.get(article_id)
            if article is None:
                continue
            article.item_ids = sorted(
                {
                    item.item_id
                    for item in doc.items.values()
                    if item.article_id == article_id
                },
                key=lambda iid: (position.get(iid, len(position)), iid),
            )
            refresh(doc, article, index)

    return attached
