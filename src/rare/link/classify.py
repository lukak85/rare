"""Assign an editorial genre to every article.

This runs last, once the articles are final: a piece split across two pages is
one article by then, so it is classified once, from its whole text, instead of
twice from two halves that each read like a fragment.

A `ClassificationBackend` takes plain text and returns a label. Generative
backends (GaMS) answer in free-form Slovenian, so the reply is matched back
against the backend's own `classes` list where it exposes one; an answer that
matches nothing leaves `Article.genre` as None rather than inventing a label.

The running header is used twice over. A magazine sorts its own pieces by
genre and prints the sort order at the top of the page — "Telegrami" is news,
"Plošče" reviews, "Pisma" letters — so the section goes to the classifier as
context, and stands in for its answer when it gives none. That fallback also
reaches the pieces a column was split into, which are often a heading and one
paragraph: too little text to classify, but their section is not in doubt.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from rare.doc.schema import Article, GlasanaDocument, TextItem
from rare.link.articles import header_tokens
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

# Categories that are page furniture or fragments, never editorial pieces.
_MIN_CHARS = 40

# The genre a Glasbena Mladina section runs. Keys are folded and matched as
# substrings, because the running header arrives with the facing page mirrored
# through it ("IAHHCIO ODMEVI", "3PTOZI IZDAJE"); longest key wins. Values must
# name a class the backend declares, or the fallback declines to guess.
SECTION_TO_GENRE: dict[str, str] = {
    # news and briefs
    "telegrami":         "novice",
    "odmevi":            "novice",
    "od vsepovsod":      "novice",
    "novice":            "novice",
    # records
    "mine iz tujih":     "recenzija",
    "plosce":            "recenzija",
    "izdaje":            "recenzija",
    "recenzij":          "recenzija",
    # readers
    "pisma":             "pisma",
    # puzzles
    "krizanka":          "kviz",
    "uganka":            "kviz",
    "kviz":              "kviz",
    # editorial
    "beseda urednistva": "članek",
    "uvodnik":           "članek",
    "reportaza":         "članek",
    # talk
    "intervju":          "intervju",
    "pogovor":           "intervju",
    # front matter
    "iz vsebine":        "kazalo",
    "kazalo":            "kazalo",
    "naslovnica":        "naslovnica",
    "oglas":             "reklama",
    "reklama":           "reklama",
}


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so 'Recenzija' matches 'recenzija'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def genre_for_section(
    section: str | None, classes: list[str] | None = None
) -> str | None:
    """The genre a section runs, or None when the header names none we know.

    `classes` is the backend's own vocabulary: a section whose genre the
    backend cannot express yields None rather than a label nothing else in the
    document uses. Matching is folded on both sides, so the map can be written
    without diacritics and still match "Plošče".
    """
    folded = _fold(section or "")
    if not folded:
        return None

    folded_classes = {_fold(c): c for c in classes} if classes else None
    for keyword in sorted(SECTION_TO_GENRE, key=len, reverse=True):
        if keyword not in folded:
            continue
        genre = SECTION_TO_GENRE[keyword]
        if folded_classes is None:
            return genre
        return folded_classes.get(_fold(genre))
    return None


def article_text(
    doc: GlasanaDocument,
    article: Article,
    max_chars: int,
    include_section: bool = False,
) -> str:
    """The article's title and body as one string, truncated to `max_chars`.

    Truncation is a cost guard: a generative classifier is charged by the token
    and the genre of a magazine piece is clear from its opening. The cut lands
    on an item boundary so the classifier never sees a half sentence.

    With `include_section`, the running header is named first. A header too
    damaged to yield a single content word is left out — it would cost tokens
    and say nothing.
    """
    parts: list[str] = []
    if include_section and header_tokens(article.section):
        parts.append(f"Rubrika: {(article.section or '').strip()}")
    if article.title.strip():
        parts.append(article.title.strip())

    budget = max_chars - sum(len(part) for part in parts)
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
    """Set `Article.genre` on every article it can. Returns the count.

    Best-effort, like every other pass here: a backend that raises on one
    article is logged and skipped, so one bad piece of text cannot lose the
    genres of the whole issue — nor the parse itself. With no backend at all,
    the section fallback still runs, so a document parsed without a classifier
    comes out with the genres its running headers give away.
    """
    fallback = config.classify_section_fallback
    if classifier is None and not fallback:
        return 0

    classes = list(getattr(classifier, "classes", []) or []) or None
    classified = 0
    from_section = 0

    for article in doc.articles.values():
        label = None

        if classifier is not None:
            text = article_text(
                doc,
                article,
                config.classify_max_chars,
                config.classify_include_section,
            )
            # Too short to read a genre off: the fallback below may still know.
            if len(text) >= _MIN_CHARS:
                try:
                    label = _match_label(classifier.classify(text), classes)
                except Exception as exc:  # noqa: BLE001 — one article must not fail the parse
                    logger.warning(
                        "classification failed for %s: %s", article.article_id, exc
                    )

        if not label and fallback:
            label = genre_for_section(article.section, classes)
            if label:
                from_section += 1

        if label:
            article.genre = label
            classified += 1

    if from_section:
        logger.info("took the genre from the section header for %d articles", from_section)
    return classified