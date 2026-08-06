"""Merge an article with its continuation elsewhere in the issue.

A piece that runs over a page break becomes two articles under the seed rule,
because the continuation either carries its own (jump) headline or none at all.
This pass folds the continuation back into the original, recording what
convinced it in an "article-continues" Link so the decision stays auditable.

Signals, none of which is trusted alone:

* a near-identical title — a jump headline, or the same headline detected twice;
* the continuation having no headline of its own;
* a shared running header (matched loosely: these scans mirror bleed-through
  from the facing page into the header, so equality never holds);
* rare named entities in common;
* a sentence running straight across the seam.
"""

from __future__ import annotations

import difflib
import re

from rare.doc.schema import Article, GlasanaDocument, Link, TextItem
from rare.link.articles import refresh
from rare.link.config import LinkConfig
from rare.link.entities import EntityIndex
from rare.parse.merge import _ends_sentence, _is_continuation

_WORD = re.compile(r"\w+", re.UNICODE)


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall((text or "").casefold()))


def _title_similarity(a: str, b: str) -> float:
    left, right = _normalise(a), _normalise(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _header_similarity(a: str | None, b: str | None) -> float:
    """Token-overlap between two running headers.

    Token-set rather than string similarity: the real section name survives
    the mirrored noise as a token, while the noise itself does not repeat.
    """
    left = {t for t in _WORD.findall((a or "").casefold()) if len(t) > 2}
    right = {t for t in _WORD.findall((b or "").casefold()) if len(t) > 2}
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _text_items(doc: GlasanaDocument, article: Article) -> list[TextItem]:
    return [
        doc.items[iid]
        for iid in article.item_ids
        if iid in doc.items
        and isinstance(doc.items[iid], TextItem)
        and (doc.items[iid].text or "").strip()
    ]


def _seam_continues(doc: GlasanaDocument, first: Article, second: Article) -> bool:
    """True when the last sentence of `first` runs into `second`."""
    left = _text_items(doc, first)
    right = _text_items(doc, second)
    if not left or not right:
        return False
    return not _ends_sentence(left[-1].text) and _is_continuation(right[0].text)


def _score(
    doc: GlasanaDocument,
    first: Article,
    second: Article,
    index: EntityIndex,
    config: LinkConfig,
) -> tuple[float, str, list[str]]:
    score = 0.0
    methods: list[str] = []
    evidence: list[str] = []

    title_sim = _title_similarity(first.title, second.title)
    if title_sim >= config.title_similarity_threshold:
        score += config.title_weight
        methods.append("title")
        evidence.append(second.title or first.title)
    elif not (second.title or "").strip():
        score += config.untitled_weight
        methods.append("untitled")

    header_sim = _header_similarity(first.section, second.section)
    if header_sim > 0:
        score += config.header_weight * header_sim
        methods.append("header")
        if second.section:
            evidence.append(second.section)

    entity_sim, shared = index.similarity(
        index.keys_of_items(first.item_ids), index.keys_of_items(second.item_ids)
    )
    if entity_sim > 0:
        score += config.entity_weight * entity_sim
        methods.append("ner")
        evidence.extend(shared[:3])

    if _seam_continues(doc, first, second):
        score += config.seam_weight
        methods.append("seam")

    return score, "+".join(methods), evidence


def _absorb(doc: GlasanaDocument, first: Article, second: Article) -> None:
    """Move every item of `second` into `first` and drop `second`."""
    for item_id in second.item_ids:
        item = doc.items.get(item_id)
        if item is not None:
            item.article_id = first.article_id

    position = {iid: i for i, iid in enumerate(doc.body_order)}
    merged = set(first.item_ids) | set(second.item_ids)
    first.item_ids = sorted(
        merged, key=lambda iid: (position.get(iid, len(position)), iid)
    )
    if not (first.title or "").strip():
        first.title = second.title
    first.continued = True
    doc.articles.pop(second.article_id, None)


def merge_continuations(
    doc: GlasanaDocument, index: EntityIndex, config: LinkConfig
) -> int:
    """Fold continuations into their parent article. Returns merges applied.

    Runs to a fixed point so a piece split three ways collapses to one.
    """
    merges = 0

    while True:
        articles = list(doc.iter_articles())
        merged_this_pass = False

        for first, second in zip(articles, articles[1:]):
            if first.article_id not in doc.articles:
                continue
            if second.article_id not in doc.articles:
                continue
            if not first.page_nos or not second.page_nos:
                continue
            # Only a piece that resumes on the same or the next page.
            if not (
                0 <= second.page_nos[0] - first.page_nos[-1] <= config.max_page_gap
            ):
                continue

            score, method, evidence = _score(doc, first, second, index, config)
            if score < config.continuation_min_score:
                continue

            doc.add_link(
                Link(
                    kind="article-continues",
                    from_id=first.article_id,
                    to_id=first.article_id,
                    score=round(score, 4),
                    method=method,
                    evidence=[e for e in evidence if e][:5],
                )
            )
            _absorb(doc, first, second)
            refresh(doc, first, index)
            merges += 1
            merged_this_pass = True
            break

        if not merged_this_pass:
            return merges
