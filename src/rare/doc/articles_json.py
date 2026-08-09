"""Flatten a document into a self-contained per-article view.

`{stem}_doc.json` is normalised — items in one dict, order in another, article
membership in a third — which is right for the pipeline but means any consumer
wanting "the articles in this issue" has to join three structures and know the
`body_order` convention.

`{stem}_articles.json` is the denormalised counterpart: one entry per article
with its items inlined in reading order, ready to render to HTML/Markdown or
feed to anything downstream without touching the rest of the document.
"""

from __future__ import annotations

from typing import Any

from rare.doc.renderers import _article_blocks
from rare.doc.schema import CaptionItem, FigBylineItem, FigureItem, GlasanaDocument


def _item_payload(item) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "item_id": item.item_id,
        "category": item.category.value,
        "page_no": item.provenance.page_no,
        "bbox": item.provenance.bbox,
    }
    text = getattr(item, "text", None)
    if text:
        payload["text"] = text
    if isinstance(item, FigureItem):
        payload["image_path"] = item.image_path
        if item.alt_text:
            payload["alt_text"] = item.alt_text
    if isinstance(item, (CaptionItem, FigBylineItem)) and item.figure_id:
        payload["figure_id"] = item.figure_id
    if item.entities:
        payload["entities"] = [
            {"text": e.text, "label": e.label, "key": e.key}
            for e in item.entities
        ]
    return payload


def to_articles(doc: GlasanaDocument) -> dict[str, Any]:
    """Build the flattened per-article structure for `doc`."""
    articles: list[dict[str, Any]] = []

    for article_id, item_ids in _article_blocks(doc):
        article = doc.articles.get(article_id) if article_id else None
        items = [doc.items[iid] for iid in item_ids if iid in doc.items]

        entry: dict[str, Any] = {
            "article_id": article_id,
            "title": article.title if article is not None else "",
            "section": article.section if article is not None else None,
            "page_nos": (
                article.page_nos
                if article is not None
                else sorted({i.provenance.page_no for i in items})
            ),
            "genre": article.genre if article is not None else None,
            "continued": bool(article.continued) if article is not None else False,
            "entity_keys": article.entity_keys if article is not None else [],
            "items": [_item_payload(item) for item in items],
        }
        articles.append(entry)

    return {
        "source_pdf": doc.source_pdf,
        "doc_id": doc.doc_id,
        "articles": articles,
        # Links between articles/items that the grouping itself cannot express.
        "links": [
            link.model_dump()
            for link in doc.links
            if link.kind != "entity-overlap"
        ],
    }
