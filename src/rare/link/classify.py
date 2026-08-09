"""Assign an editorial genre to every article.

This runs last, once the articles are final: a piece split across two pages is
one article by then, so it is classified once, from its whole text, instead of
twice from two halves that each read like a fragment.

A `ClassificationBackend` takes plain text and returns a label. Generative
backends (GaMS) answer in free-form Slovenian, so the reply is matched back
against the backend's own `classes` list where it exposes one; an answer that
matches nothing leaves `Article.genre` as None rather than inventing a label.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from rare.doc.schema import Article, GlasanaDocument, TextItem
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

# Categories that are page furniture or fragments, never editorial pieces.
_MIN_CHARS = 40


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so 'Recenzija' matches 'recenzija'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def article_text(doc: GlasanaDocument, article: Article, max_chars: int) -> str:
    """The article's title and body as one string, truncated to `max_chars`.

    Truncation is a cost guard: a generative classifier is charged by the token
    and the genre of a magazine piece is clear from its opening. The cut lands
    on an item boundary so the classifier never sees a half sentence.
    """
    parts: list[str] = []
    if article.title.strip():
        parts.append(article.title.strip())

    budget = max_chars - len(parts[0]) if parts else max_chars
    for item_id in article.item_ids:
        item = doc.items.get(item_id)
        if not isinstance(item, TextItem):
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        if budget - len(text) < 0 and parts:
            break
        parts.append(text)
        budget -= len(text)
    return "\n\n".join(parts)


def _match_label(reply: str, classes: list[str] | None) -> str | None:
    """Map a backend's reply onto one of its declared classes.

    Discriminative backends return the label itself and match trivially.
    Generative ones wrap it in a sentence ("Besedilo spada v kategorijo
    intervju."), so the longest class name occurring in the reply wins —
    longest first, or "informativni členek" would lose to a bare substring.
    """
    reply = (reply or "").strip()
    if not reply:
        return None
    if not classes:
        return reply

    folded = _fold(reply)
    for label in sorted(classes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(_fold(label))}\b", folded):
            return label
    return None


def classify_articles(
    doc: GlasanaDocument,
    classifier,
    config: LinkConfig,
) -> int:
    """Set `Article.genre` on every article with enough text. Returns the count.

    Best-effort, like every other pass here: a backend that raises on one
    article is logged and skipped, so one bad piece of text cannot lose the
    genres of the whole issue — nor the parse itself.
    """
    if classifier is None:
        return 0

    classes = list(getattr(classifier, "classes", []) or []) or None
    classified = 0

    for article in doc.articles.values():
        text = article_text(doc, article, config.classify_max_chars)
        if len(text) < _MIN_CHARS:
            continue
        try:
            label = _match_label(classifier.classify(text), classes)
        except Exception as exc:  # noqa: BLE001 — one article must not fail the parse
            logger.warning("classification failed for %s: %s", article.article_id, exc)
            continue
        if label:
            article.genre = label
            classified += 1

    return classified