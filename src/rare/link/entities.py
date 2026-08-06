"""Entity annotation, rarity indexing, and entity-overlap links.

Two entities matching is only evidence if the entity is *rare* in this
document. A music magazine mentions "Ljubljana" and "Glasbena mladina" on
nearly every page, so an unweighted overlap connects everything to everything.
`EntityIndex` therefore scores every key by inverse document frequency and
drops the ubiquitous ones outright.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

from rare.doc.schema import (
    AuthorItem,
    BylineItem,
    FigBylineItem,
    GlasanaDocument,
    Link,
    TextItem,
    TranslatorItem,
)
from rare.link.config import LinkConfig
from rare.models.ner.normalize import keys_for

logger = logging.getLogger(__name__)

# Regions that are, by construction, almost pure person names — used to seed
# gazetteer-style backends with this document's own cast list.
NAME_ITEM_TYPES = (BylineItem, AuthorItem, TranslatorItem, FigBylineItem)


def annotate(doc: GlasanaDocument, ner) -> None:
    """Run `ner` over the document's text items and store the entities."""
    if ner is None:
        return

    items = [
        item
        for item in doc.items.values()
        if isinstance(item, TextItem) and (item.text or "").strip()
    ]
    if not items:
        return

    try:
        batches = ner.extract([item.text for item in items])
    except Exception:
        logger.exception("NER backend failed; continuing without entities")
        return

    for item, entities in zip(items, batches):
        item.entities = list(entities or [])


class EntityIndex:
    """Maps normalised entity keys to the items and articles that mention them."""

    def __init__(self, doc: GlasanaDocument, config: LinkConfig):
        self.config = config
        self.key_to_items: dict[str, set[str]] = {}
        self.item_to_keys: dict[str, set[str]] = {}
        self.key_to_articles: dict[str, set[str]] = {}

        for item in doc.items.values():
            keys: set[str] = set()
            for entity in getattr(item, "entities", []) or []:
                for key in keys_for(entity.text, entity.label):
                    if len(key) >= config.min_key_length:
                        keys.add(key)
            if not keys:
                continue
            self.item_to_keys[item.item_id] = keys
            for key in keys:
                self.key_to_items.setdefault(key, set()).add(item.item_id)
                if item.article_id:
                    self.key_to_articles.setdefault(key, set()).add(item.article_id)

        self.n_articles = max(1, len(doc.articles))
        # Floor of 2: an entity shared by exactly two articles is the most
        # informative case there is, and must stay usable however few articles
        # the document has. Above that the share-based gate takes over.
        self._max_df = max(2.0, config.max_doc_frequency * self.n_articles)

    def weight(self, key: str) -> float:
        """Inverse-document-frequency weight; 0 for keys too common to matter."""
        df = len(self.key_to_articles.get(key, ()))
        if df == 0 or df > self._max_df:
            return 0.0
        return math.log(self.n_articles / df) + 1.0

    def keys_of(self, item_id: str) -> set[str]:
        return self.item_to_keys.get(item_id, set())

    def keys_of_items(self, item_ids: Iterable[str]) -> set[str]:
        keys: set[str] = set()
        for item_id in item_ids:
            keys |= self.item_to_keys.get(item_id, set())
        return keys

    def similarity(self, a: set[str], b: set[str]) -> tuple[float, list[str]]:
        """Rarity-weighted cosine similarity between two key sets.

        Returns the score and the shared keys that drove it, so a merge can
        record what convinced it.
        """
        shared = [k for k in (a & b) if self.weight(k) > 0]
        if not shared:
            return 0.0, []
        num = sum(self.weight(k) for k in shared)
        norm_a = sum(self.weight(k) for k in a)
        norm_b = sum(self.weight(k) for k in b)
        if norm_a <= 0 or norm_b <= 0:
            return 0.0, []
        score = num / math.sqrt(norm_a * norm_b)
        shared.sort(key=self.weight, reverse=True)
        return min(1.0, score), shared


def cross_link(
    doc: GlasanaDocument, index: EntityIndex, config: LinkConfig
) -> None:
    """Emit entity-overlap links between items in *different* articles.

    Within an article the grouping already says the items belong together, so
    only cross-article mentions add information. Common keys are skipped and
    fan-out is capped, otherwise this is quadratic on a 500-item document.
    """
    emitted: dict[str, int] = {}

    for key, item_ids in sorted(index.key_to_items.items()):
        weight = index.weight(key)
        if weight <= 0 or len(item_ids) < 2:
            continue

        ordered = sorted(item_ids)
        for i, source in enumerate(ordered):
            source_item = doc.items.get(source)
            if source_item is None:
                continue
            for target in ordered[i + 1 :]:
                target_item = doc.items.get(target)
                if target_item is None:
                    continue
                if (
                    source_item.article_id
                    and source_item.article_id == target_item.article_id
                ):
                    continue
                if (
                    emitted.get(source, 0) >= config.max_entity_links_per_item
                    or emitted.get(target, 0) >= config.max_entity_links_per_item
                ):
                    continue
                doc.add_link(
                    Link(
                        kind="entity-overlap",
                        from_id=source,
                        to_id=target,
                        score=round(weight, 4),
                        method="ner",
                        evidence=[key],
                    )
                )
                emitted[source] = emitted.get(source, 0) + 1
                emitted[target] = emitted.get(target, 0) + 1
