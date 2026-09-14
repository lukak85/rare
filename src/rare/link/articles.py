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
# …and it is printed in the top of the page (see LinkConfig.section_header_top_frac).
DEFAULT_HEADER_TOP_FRAC = 0.2


_MIXED_CASE = re.compile(r"[a-zčšžćđ][A-ZČŠŽĆĐ]")
_WORD = re.compile(r"\w+", re.UNICODE)
# "Hat", "(Hat": a capital then only lowercase letters.
_TITLE_CASE = re.compile(r"^\W*[A-ZČŠŽĆĐ][a-zčšžćđ]+\W*$")


def clean_header(text: str, keep_tail: bool = True) -> str:
    """Strip mirrored bleed-through from a running header.

    The bleed lands before or between the capitals of the real section name
    ("vpasa sirasoa POSKUS ESEJA", "H im i zi hnih MINE IZ TUJIH"), so lowercase
    there is dropped. Lowercase after the last capital token is kept: that is a
    mixed-case header ("GM novice", "AFRIKA s prve roke"), not noise — unless
    it is title case ("(XI, (Hat"), which no header tail here is.
    `keep_tail=False` keeps the capitals alone, for a Header region that is
    really an announcement with a section name set in capitals inside it.
    """
    tokens = (text or "").split()
    is_caps = [token.isupper() and not _MIXED_CASE.search(token) for token in tokens]
    if not any(is_caps):
        return (text or "").strip()
    last_caps = max(i for i, caps in enumerate(is_caps) if caps)
    kept = [
        token
        for i, token in enumerate(tokens)
        if is_caps[i]
        or (keep_tail and i > last_caps and not _MIXED_CASE.search(token)
            and not _TITLE_CASE.match(token))
    ]
    return " ".join(kept)


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
    """
    words = _WORD.findall(text or "")
    count = 0
    in_fragment_run = False
    for word in words:
        fragment = len(word) <= 2
        if not (fragment and in_fragment_run):
            count += 1
        in_fragment_run = fragment
    return bool(words) and count <= max_words


def _header_text(item, max_words: int) -> Optional[str]:
    """The section name a Header/Section item carries, or None when it names none."""
    text = clean_header(item.text or "")
    if text and not is_running_header(text, max_words):
        # A mixed-case tail made it prose-length; the capitals alone
        # may still name the section.
        text = clean_header(item.text or "", keep_tail=False)
    return text if text and is_running_header(text, max_words) else None


def running_headers(
    doc: GlasanaDocument, max_words: int = DEFAULT_HEADER_MAX_WORDS
) -> dict[int, str]:
    """One section header per page, if any: the longest of the page's headers.

    A page can carry several (`page_headers`); this is for the passes that
    only ask what section a page belongs to as a whole.
    """
    headers: dict[int, list[str]] = defaultdict(list)
    for item in doc.items.values():
        if isinstance(item, (HeaderItem, SectionItem)):
            text = _header_text(item, max_words)
            if text:
                headers[item.provenance.page_no].append(text)
    # Longest wins: what survives cleaning is usually the real section name.
    return {
        page_no: max(texts, key=len) for page_no, texts in headers.items()
    }


# (x1, y1, text) of one header on a page, in page coordinates.
PageHeader = tuple[float, float, str]


def page_headers(
    doc: GlasanaDocument,
    max_words: int = DEFAULT_HEADER_MAX_WORDS,
    top_frac: float = DEFAULT_HEADER_TOP_FRAC,
) -> dict[int, list[PageHeader]]:
    """Every section header on each page, with where it is printed.

    A page is often split between sections — "Odmevi" over its left part and
    "Telegrami" over the right column — so one header per page cannot say which
    section an item is in. `header_at` can, from these positions.

    Headers below the top `top_frac` of the page are dropped wherever the page
    has one inside it: down there a Header is usually a caption, a title or a
    photo credit, and it would carve the page into sections that are not.
    """
    headers: dict[int, list[PageHeader]] = defaultdict(list)
    sections: dict[int, list[PageHeader]] = defaultdict(list)
    for item in doc.items.values():
        if isinstance(item, (HeaderItem, SectionItem)):
            text = _header_text(item, max_words)
            if text:
                box = item.provenance.bbox
                found = headers if isinstance(item, HeaderItem) else sections
                found[item.provenance.page_no].append((box["x1"], box["y1"], text))
    # A Section label sits inside the text — a record title, a column's name —
    # so its position marks out no part of the page. It stands in only on a
    # page with no Header, and then, as the page's one header, covers all of it.
    for page_no, found in sections.items():
        if page_no not in headers:
            headers[page_no] = [max(found, key=lambda h: len(h[2]))]
    for page_no, found in headers.items():
        page = doc.pages.get(page_no)
        if page is None or not page.height:
            continue
        top = [h for h in found if h[1] <= top_frac * page.height]
        if top:
            headers[page_no] = top
    return {page_no: sorted(found) for page_no, found in headers.items()}


def header_at(headers: Optional[list[PageHeader]], bbox: dict) -> Optional[str]:
    """The header governing an item at `bbox`, among its page's `headers`.

    A lone header covers the whole page. With several, a header covers what is
    below it and to its right, up to the next header across — so an item takes,
    among the headers printed above its middle, the nearest one to its left;
    failing that the leftmost of them. An item above every header takes the
    nearest to its left of all of them, on the same terms.
    """
    if not headers:
        return None
    if len(headers) == 1:
        return headers[0][2]
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2
    candidates = [h for h in headers if h[1] <= cy] or list(headers)
    to_the_left = [h for h in candidates if h[0] <= cx]
    if to_the_left:
        return max(to_the_left, key=lambda h: h[0])[2]
    return min(candidates, key=lambda h: h[0])[2]


def refresh(
    doc: GlasanaDocument,
    article: Article,
    index: Optional[EntityIndex] = None,
    headers: Optional[dict[int, str]] = None,
    zones: Optional[dict[int, list[PageHeader]]] = None,
) -> None:
    """Recompute an article's derived fields from its current item list.

    With `zones` (`page_headers`), the section is the header over the article's
    first item that has one; with only `headers`, that of its first page.
    """
    pages = sorted(
        {
            doc.items[iid].provenance.page_no
            for iid in article.item_ids
            if iid in doc.items
        }
    )
    article.page_nos = pages

    if zones:
        for iid in article.item_ids:
            item = doc.items.get(iid)
            section = item and header_at(zones.get(item.provenance.page_no), item.provenance.bbox)
            if section:
                article.section = section
                break
    elif headers:
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
    zones = page_headers(
        doc,
        config.section_header_max_words if config else DEFAULT_HEADER_MAX_WORDS,
        config.section_header_top_frac if config else DEFAULT_HEADER_TOP_FRAC,
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
        refresh(doc, article, index, zones=zones)
