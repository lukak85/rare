"""Figure -> article: which piece on the page does this visual illustrate?

`figure_matching` has already tied each caption and photo credit to its figure.
This pass takes that group as one unit — a figure never ends up in a different
article from its own caption — and gives it to the article it belongs to, using
two signals and nothing else:

* **proximity** — how close the group sits to the article's own elements on the
  page, with elements above it (or, side by side, to its left) preferred: a
  figure illustrates the text it follows, so the piece running down the column
  above it is the likelier owner than the one starting below.
* **entities** — how much of the caption's text names the same people, places
  and works as the article does, scored on rarity by `EntityIndex`. This is the
  same NER evidence `crosspage` merges continuations on.

Either signal can carry a group on its own: a caption too far from any article
still goes where its names point, and a photo credit with no usable text goes
where the geometry points. The decision is recorded as a "figure-to-article"
Link with the score and the shared entity keys that drove it.

Coordinates are page-image pixels with a top-left origin, so "below" is a
larger y — the same convention as `figure_matching`.
"""

from __future__ import annotations

from collections import defaultdict

from rare.doc.schema import (
    CaptionItem,
    ContentLayer,
    FigBylineItem,
    FigureItem,
    GlasanaDocument,
    Link,
)
from rare.link._geometry import Box, box_of, centre, page_size
from rare.link.config import LinkConfig
from rare.link.entities import EntityIndex

# The regions this pass places. They are also the regions it never treats as an
# article's own content: a figure's nearest neighbour is usually another
# figure's caption, and anchoring to that would just chain visuals together.
VISUAL_TYPES = (FigureItem, CaptionItem, FigBylineItem)


def _group_visuals(doc: GlasanaDocument) -> list[list]:
    """Every figure with its caption and credit; every stray caption alone."""
    groups: dict[str, list] = {}
    strays: list[list] = []

    for item in doc.items.values():
        if isinstance(item, FigureItem):
            groups[item.item_id] = [item]

    for item in doc.items.values():
        if isinstance(item, (CaptionItem, FigBylineItem)):
            if item.figure_id in groups:
                groups[item.figure_id].append(item)
            else:
                strays.append([item])

    ordered = sorted(groups.values(), key=lambda g: g[0].item_id)
    return ordered + sorted(strays, key=lambda g: g[0].item_id)


def _union_box(items) -> Box:
    boxes = [box_of(item) for item in items]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _directional_distance(
    group: Box, item: Box, page_w: float, page_h: float, config: LinkConfig
) -> float:
    """Edge distance from a group to one article element, sided.

    The two axes are normalised by the page and summed, so the result is a
    fraction of the page and comparable between pages of different sizes. An
    element below the group, or beside it to the right, is then pushed away by
    its penalty — near enough still wins, but only clearly.
    """
    dx = max(0.0, max(group[0], item[0]) - min(group[2], item[2])) / page_w
    dy = max(0.0, max(group[1], item[1]) - min(group[3], item[3])) / page_h

    group_cx, group_cy = centre(group)
    item_cx, item_cy = centre(item)

    penalty = 1.0
    if item_cy > group_cy:
        penalty *= config.figure_link_below_penalty
    if item_cx > group_cx:
        penalty *= config.figure_link_side_penalty
    return (dx + dy) * penalty


def _candidates(doc: GlasanaDocument, page_no: int) -> dict[str, list]:
    """article_id -> the article's non-visual body items on this page."""
    by_article: dict[str, list] = defaultdict(list)
    for item in doc.items.values():
        if (
            item.article_id
            and item.provenance.page_no == page_no
            and item.content_layer == ContentLayer.BODY
            and not isinstance(item, VISUAL_TYPES)
        ):
            by_article[item.article_id].append(item)
    return by_article


def _proximity(
    group_box: Box, items, page_w: float, page_h: float, config: LinkConfig
) -> float:
    """0…1, falling to 0 at `figure_link_max_distance_frac` and beyond."""
    distances = [
        _directional_distance(group_box, box_of(item), page_w, page_h, config)
        for item in items
    ]
    if not distances:
        return 0.0
    distance = (
        sum(distances) / len(distances)
        if config.figure_link_use_mean
        else min(distances)
    )
    limit = config.figure_link_max_distance_frac
    if limit <= 0 or distance >= limit:
        return 0.0
    return 1.0 - distance / limit


def link_figures(
    doc: GlasanaDocument, index: EntityIndex, config: LinkConfig
) -> int:
    """Give every figure group the article it illustrates. Returns how many."""

    position = {iid: i for i, iid in enumerate(doc.body_order)}
    linked = 0

    for group in _group_visuals(doc):
        page_no = group[0].provenance.page_no
        candidates = _candidates(doc, page_no)
        if not candidates:
            # Nothing on this page to belong to; whatever the assembler
            # decided is better than orphaning the group.
            continue

        page_w, page_h = page_size(doc, page_no)
        group_box = _union_box(group)
        group_ids = {item.item_id for item in group}
        group_keys = index.keys_of_items(group_ids)

        def opens_at(article_id: str) -> tuple[int, str]:
            article = doc.articles.get(article_id)
            item_ids = article.item_ids if article else ()
            first = min(
                (position[iid] for iid in item_ids if iid in position),
                default=len(position),
            )
            return first, article_id

        best = None
        for article_id in sorted(candidates, key=opens_at):
            article = doc.articles.get(article_id)
            if article is None:
                continue

            proximity = _proximity(
                group_box, candidates[article_id], page_w, page_h, config
            )
            # The group's own items are excluded from the article's side:
            # counting a caption's entities towards the article it already
            # sits in would only re-elect the status quo.
            others = [iid for iid in article.item_ids if iid not in group_ids]
            entities, shared = index.similarity(
                group_keys, index.keys_of_items(others)
            )

            score = (
                config.figure_link_geometry_weight * proximity
                + config.figure_link_ner_weight * entities
            )
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, article_id, proximity, entities, shared[:3])

        if best is None:
            continue

        score, article_id, proximity, entities, evidence = best
        if proximity > 0 and entities > 0:
            method = "geometry+ner"
        elif entities > 0:
            method = "ner"
        else:
            method = "geometry"

        for item in group:
            item.article_id = article_id
        doc.add_link(
            Link(
                kind="figure-to-article",
                from_id=group[0].item_id,
                to_id=article_id,
                score=round(score, 4),
                method=method,
                evidence=evidence,
            )
        )
        linked += 1

    return linked
