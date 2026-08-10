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

import difflib
import re
import unicodedata
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

# A running header is a couple of words. Anything longer is a caption or a
# standfirst the detector labelled Header by mistake.
DEFAULT_HEADER_MAX_WORDS = 8


_MIXED_CASE = re.compile(r"[a-zčšžćđ][A-ZČŠŽĆĐ]")
_WORD = re.compile(r"\w+", re.UNICODE)


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


def header_tokens(text: str | None) -> set[str]:
    """The content words of a running header, for loose comparison.

    Short tokens are dropped: what survives `clean_header` still carries stray
    two-letter fragments of the mirrored text, and those collide by accident.
    """
    return {t for t in _WORD.findall((text or "").casefold()) if len(t) > 2}


def header_similarity(a: str | None, b: str | None) -> float:
    """Token overlap between two running headers, 0.0 when either is unusable.

    Token-set rather than string similarity: the real section name survives the
    mirrored noise as a token, while the noise itself does not repeat, so
    "IAHHCIO ODMEVI" and "ODMEVI" score 1.0 where equality scores nothing.

    Callers that need to tell "different sections" from "no header to compare"
    must check `header_tokens` themselves — both cases return 0.0 here.
    """
    left, right = header_tokens(a), header_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _squash(text: str | None) -> str:
    """A header reduced to bare letters, for comparison as a character run."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped)


def header_char_similarity(a: str | None, b: str | None) -> float:
    """Similarity of two headers as character runs, ignoring word boundaries.

    The token comparison fails when OCR mangles the words themselves rather
    than adding noise around them: "(PRED)USM ERJENE STRANI" comes back as
    "CPRED)USMEHJENE STRAHI", which shares no whole word with it but is plainly
    the same header. Character overlap sees that where tokens cannot.
    """
    left, right = _squash(a), _squash(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def same_section(
    a: str | None,
    b: str | None,
    min_token_similarity: float,
    min_char_similarity: float,
) -> bool:
    """Whether two running headers name the same section of the magazine.

    Either measure alone is enough. Tokens catch a header printed with the
    facing page's mirrored through it ("IAHHCIO ODMEVI" against "ODMEVI",
    which share no characters in order); characters catch one whose words came
    back misread. A header too damaged for both is treated as a change only by
    callers that first check there was a header to read at all.
    """
    if header_similarity(a, b) >= min_token_similarity:
        return True
    return header_char_similarity(a, b) >= min_char_similarity


def is_running_header(text: str | None, max_words: int) -> bool:
    """Whether `text` is short enough to be a running header rather than prose.

    Captions and standfirsts get labelled Header often enough to matter, and
    one of them standing in for a section name invents a section change on
    every page it appears.
    """
    words = _WORD.findall(text or "")
    return bool(words) and len(words) <= max_words


def running_headers(
    doc: GlasanaDocument, max_words: int = DEFAULT_HEADER_MAX_WORDS
) -> dict[int, str]:
    """The section header printed on each page, if any."""
    headers: dict[int, list[str]] = defaultdict(list)
    for item in doc.items.values():
        if isinstance(item, (HeaderItem, SectionItem)):
            text = clean_header(item.text or "")
            if text and is_running_header(text, max_words):
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
    headers = running_headers(
        doc,
        config.section_header_max_words if config else DEFAULT_HEADER_MAX_WORDS,
    )

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
