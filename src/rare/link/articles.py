"""Make `doc.articles` authoritative, complete and ordered.

The assembler's seed rule ("a Headline opens an article") leaves three
problems this pass cleans up:

* furniture — Header/PageNum/Abandon — is appended to `Article.item_ids` even
  though it is excluded from `body_order`, so an article's item list mixes
  content with page chrome;
* `item_ids` is in page-assembly order, which is not necessarily `body_order`;
* nothing records which pages an article covers or what section it sits in.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from rare.doc.schema import (
    Article,
    ContentLayer,
    GlasanaDocument,
    HeaderItem,
    SectionItem,
)
from rare.link.config import LinkConfig
from rare.link.entities import EntityIndex

# How many of an article's rarest entity keys to surface in the JSON.
TOP_ENTITY_KEYS = 10


_MIXED_CASE = re.compile(r"[a-zčšžćđ][A-ZČŠŽĆĐ]")


def clean_header(text: str) -> str:
    """Strip mirrored bleed-through from a running header.

    These scans print the facing page's header through the paper, and OCR reads
    it back-to-front interleaved with the real one: "OHVaLLNSCKOH KOMENTIRAMO",
    "nraaH dso a V OSPREDJU". The reversed text lands as tokens with lowercase
    letters immediately followed by uppercase ones, which no real word here has,
    and the genuine section names are set in capitals. Dropping everything else
    recovers the section name often enough to be useful for display and for the
    loose matching in `rare.link.crosspage`.
    """
    tokens = [
        token
        for token in (text or "").split()
        if token.isupper() and not _MIXED_CASE.search(token)
    ]
    return " ".join(tokens) or (text or "").strip()


def _running_headers(doc: GlasanaDocument) -> dict[int, str]:
    """The section header printed on each page, if any."""
    headers: dict[int, list[str]] = defaultdict(list)
    for item in doc.items.values():
        if isinstance(item, (HeaderItem, SectionItem)):
            text = clean_header(item.text or "")
            if text:
                headers[item.provenance.page_no].append(text)
    # Longest wins: what survives cleaning is usually the real section name.
    return {
        page_no: max(texts, key=len) for page_no, texts in headers.items()
    }


def refresh(
    doc: GlasanaDocument,
    article: Article,
    index: Optional[EntityIndex] = None,
    headers: Optional[dict[int, str]] = None,
) -> None:
    """Recompute an article's derived fields from its current item list."""
    pages = sorted(
        {
            doc.items[iid].provenance.page_no
            for iid in article.item_ids
            if iid in doc.items
        }
    )
    article.page_nos = pages

    if headers:
        for page_no in pages:
            if page_no in headers:
                article.section = headers[page_no]
                break

    if index is not None:
        keys = index.keys_of_items(article.item_ids)
        ranked = sorted(
            (k for k in keys if index.weight(k) > 0),
            key=lambda k: (-index.weight(k), k),
        )
        article.entity_keys = ranked[:TOP_ENTITY_KEYS]


def rebuild(
    doc: GlasanaDocument,
    index: Optional[EntityIndex] = None,
    config: Optional[LinkConfig] = None,
) -> None:
    """Rebuild every article's item list from the items themselves.

    Items are the source of truth: whatever `item.article_id` says now decides
    membership, so earlier passes can move an item simply by repointing it.
    """
    position = {iid: i for i, iid in enumerate(doc.body_order)}
    headers = _running_headers(doc)

    members: dict[str, list[str]] = defaultdict(list)
    for item in doc.items.values():
        if not item.article_id:
            continue
        # Page furniture is not part of any article's content. Its section
        # header is captured on Article.section instead.
        if item.content_layer == ContentLayer.FURNITURE:
            item.article_id = None
            continue
        if item.article_id not in doc.articles:
            item.article_id = None
            continue
        members[item.article_id].append(item.item_id)

    for article_id, article in list(doc.articles.items()):
        item_ids = members.get(article_id, [])
        if not item_ids:
            del doc.articles[article_id]
            continue
        article.item_ids = sorted(
            item_ids, key=lambda iid: (position.get(iid, len(position)), iid)
        )
        refresh(doc, article, index, headers)
