"""Does a region's text look like text, or did the scan's OCR layer fail here?

The corpus PDFs are scans carrying an invisible upstream OCR layer. Where that
OCR failed, `rare.parse.text` extracts nothing — there are no glyphs under the
box. Those regions are easy to find. The expensive failures are the ones that
come back *non-empty and wrong*: the headline JIŘÍ KYLIÁN arrives as "W Z7",
which passes every emptiness test there is and then ships.

This module scores a region's text so both the audit script and the OCR
fallback can ask the same question. Three things get a region flagged, on top
of `empty`, and a region can trip more than one:

``junk``      most of its tokens are not words — no vowel in them, letters and
              digits mixed together, or a long run of bare single letters where
              a recogniser read letterspaced display type one glyph at a time.
              "W Z7" scores 2 junk tokens out of 2.
``sparse``    far less text than a box that shape holds. A one-line region is
              measured by its aspect ratio, so type size drops out and a short
              header set large is not mistaken for a broken one; everything
              multi-line falls back to characters per unit area.
``alien``     too many characters from outside the Slovene alphabet — the "□"
              class of failure.

Every threshold here was calibrated against the 19757 hand-drawn regions in
``datasets/glasbena_mladina``; `DEFAULT_FILL_MEDIANS` is measured from them.
The tests are deliberately blunt, because both callers use them to decide what
gets *looked at again*, not what gets believed: a false positive costs one
re-read, a false negative ships a wrong headline nobody ever sees.
"""

from __future__ import annotations

import statistics
import unicodedata
from collections import defaultdict
from typing import Iterable, Mapping, Optional, Sequence

# A box in page pixels (or points — anything, as long as `page_size` is in the
# same units), (x0, y0, x1, y1), top-left origin.
Box = tuple[float, float, float, float]

REASONS = ("empty", "junk", "sparse", "alien")

# Regions that hold no text by construction. A Figure with no text is a Figure,
# not a failure.
NON_TEXT_LABELS = frozenset({"Figure", "Form", "Abandon", "Advertisement"})

# A Dropcap is one letter and a PageNum is one number; neither can pass a test
# that asks whether its tokens look like words.
NO_JUNK_CHECK_LABELS = frozenset({"Dropcap", "PageNum"})

# A Dropcap holds exactly one letter however big its box is — the box is sized
# to the display capital, not to a quantity of text — so "less text than a box
# that shape holds" means nothing here. Left in, it flagged every correctly
# extracted drop cap in the corpus.
NO_FILL_CHECK_LABELS = frozenset({"Dropcap"})

# Labels whose whole content is somebody's signature. These are initials by
# convention — "pbč", "jh", "kp." — so the vowel test, which is what catches a
# scanner's leavings everywhere else, condemns the correct ones.
SIGNATURE_LABELS = frozenset({
    "Author", "Byline", "CaptionByline", "FigByline", "Translator",
})

# Regions that are one line of type by construction — they get the aspect-ratio
# fill test, everything else gets characters-per-area. Wider than
# `rare.parse.ocr.SINGLE_LINE_LABELS`, which drives Tesseract's page
# segmentation mode: a display Headline often *is* set over two lines, so it
# must not be forced through psm 7, but its fill still reads correctly off the
# aspect ratio because `ABSOLUTE_FILL_FLOOR` only asks for one line's worth.
ONE_LINE_LABELS = frozenset({
    "Author", "Byline", "Dateline", "FigByline", "Footer", "Header", "Headline",
    "Kicker", "PageNum", "Section", "Subhead", "Subsubhead", "Translator",
})

# Average glyph advance as a fraction of the type size. Only used to turn a
# box's aspect ratio into "about this many characters fit", so it cancels out
# of the ratio against the label median — it just has to be sane.
CHAR_ASPECT = 0.6

# Slovene has syllabic r ("vrt", "smrt", "trg"), so a token built round an r and
# no other vowel is a real word. Leaving r out of this set flags several hundred
# perfectly good ones.
VOWELS = frozenset("aeiouräáàâeéèêiíìîoóòôuúùû")
# One-letter words that genuinely occur: the prepositions and the conjunction.
ONE_LETTER_WORDS = frozenset("aikosuvz")
# Everything a Slovene text is allowed to be made of, for the `alien` test.
ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyzčšžćđ")

# A run of bare single letters is letterspaced display type read letter by
# letter ("0 GL A S N A D E S K A"), which is the commonest failure both OCR
# backends are asked about. Judged token by token it scores *clean*, because
# "a", "s", "v" and "z" are real words and "GL" is a plausible abbreviation —
# so it is judged as a run instead, once there are enough tokens for the shape
# to mean anything.
SINGLETON_RUN_RATIO = 0.5   # share of single-letter tokens that makes it a run
SINGLETON_RUN_MIN = 4       # tokens below which the shape proves nothing

JUNK_RATIO = 0.5            # tokens that must be junk before the region is
ALIEN_RATIO = 0.15          # characters that must be alien before the region is
SPARSE_RATIO = 0.35         # share of the label's median fill below which it is
ABSOLUTE_FILL_FLOOR = 0.7   # a one-line region must fill this much of one line
MIN_SPARSE_SAMPLES = 8      # labels with fewer regions than this get no fill test


# --- token tests ------------------------------------------------------------

def tokens_of(text: str) -> list[str]:
    return [t for t in text.split() if t]


# Marks that join two whole words into one token. A token built round one of
# these is judged by its parts: "25/št." and "068/45-367" are not failures, and
# counting them as such flagged every dateline on the masthead.
JOINERS = "/-–—"


def _core_of(token: str) -> str:
    """The token with its punctuation taken off — what a word test judges."""
    return "".join(
        ch for ch in token
        if not unicodedata.category(ch).startswith("P") and ch not in "«»„“”"
    )


def is_junk_token(token: str, *, signature: bool = False) -> bool:
    """True when a token cannot plausibly be a word.

    `signature` relaxes the test for a region that holds a byline rather than
    prose. Punctuation-only tokens answer True here, but `junk_ratio` never
    asks — see there.
    """
    parts = [p for p in _split_on(token, JOINERS) if p]
    if len(parts) > 1:
        # Junk only if every part is: one good half redeems the token.
        return all(is_junk_token(part, signature=signature) for part in parts)

    # An initial — one letter and a full stop — is a word in every label there
    # is: "L. DIMIKAROVSKI" is a subhead, not a failed scan.
    if len(token) == 2 and token[1] == "." and token[0].isalpha():
        return False

    core = _core_of(token)
    if not core:
        return True                                   # punctuation on its own
    if core.isdigit():
        return False                                  # page numbers, years
    if any(ch.isdigit() for ch in core) and any(ch.isalpha() for ch in core):
        return True                                   # "Z7", "1n", "l971"
    lowered = core.casefold()
    # A byline signed "pbč" or "jh" has no vowel in it and is not a failure.
    if signature and len(core) <= 4 and core.isalpha():
        return False
    # One-letter tokens are decided by the word list, not by the vowel test —
    # the prepositions "s", "v", "z" and "k" have no vowel in them and are the
    # commonest words on the page.
    if len(core) == 1:
        return lowered not in ONE_LETTER_WORDS
    # A short run of capitals is an abbreviation, not a failure: CD, GM, LP,
    # RTV, and the initials bylines are signed with. Two letters is the floor
    # deliberately — a lone capital stays junk, which is what "W Z7" is made of.
    if len(core) <= 3 and core.isupper():
        return False
    if not any(ch in VOWELS for ch in lowered):
        return True                                   # "trst" is fine, "tst" is not
    return False


def _split_on(token: str, separators: str) -> list[str]:
    out, current = [], []
    for ch in token:
        if ch in separators:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    out.append("".join(current))
    return out


def has_word_core(token: str) -> bool:
    """Does the token contain anything a word could be made of?"""
    return any(ch.isalnum() for ch in token)


def junk_ratio(text: str, label: str = "") -> float:
    """Share of the *word-shaped* tokens that cannot be words.

    Bare punctuation is left out of both halves of the fraction rather than
    counted as junk: an asterisk or a dash is a real typographic mark, and in a
    two-token region like "Velenje *" counting it as a failure is enough on its
    own to condemn the region. Text made of nothing else is still junk, which
    is what the empty-denominator case means.
    """
    tokens = [t for t in tokens_of(text) if has_word_core(t)]
    if not tokens:
        return 1.0 if text.strip() else 0.0

    singletons = [t for t in tokens if len(_core_of(t)) == 1]
    letterspaced = (
        len(tokens) >= SINGLETON_RUN_MIN
        and len(singletons) / len(tokens) > SINGLETON_RUN_RATIO
    )

    signature = label in SIGNATURE_LABELS
    junk = sum(
        True if (letterspaced and t in singletons)
        else is_junk_token(t, signature=signature)
        for t in tokens
    )
    return junk / len(tokens)


def alien_ratio(text: str) -> float:
    """Share of letters that are not in the Slovene alphabet."""
    letters = [ch for ch in text.casefold() if ch.isalpha()]
    if not letters:
        return 0.0
    # Foreign names carry real diacritics (Kylián, Dvořák), so a letter that
    # decomposes to a Slovene one is not alien; "□" and "¢" are.
    alien = 0
    for ch in letters:
        base = unicodedata.normalize("NFD", ch)[0]
        if ch not in ALPHABET and base not in ALPHABET:
            alien += 1
    return alien / len(letters)


# --- fill -------------------------------------------------------------------

def fill_of(
    text: str, label: str, box: Box, page_w: float, page_h: float
) -> float:
    """How full of text the box is. Units are arbitrary — only the ratio
    against the label's median is used, plus the absolute floor for one-liners.

    Two measures, because one does not fit both shapes of region:

    * A one-line region is measured by its *aspect ratio*: a box w/h wide holds
      roughly ``w / h / CHAR_ASPECT`` characters whatever the type size, so a
      short header set large ("FONOGRAF") scores the same as a long one set
      small. Characters-per-area does not have that property — it reads big
      display type as sparse, and 40% of the Headers that flagged were nothing
      worse than short.
    * Anything multi-line falls back to characters per unit area, where the
      line count makes aspect ratio meaningless. The area is a *fraction of the
      page*, so the score does not move with the render DPI.

    Boxes taller than they are wide are turned on their side first: rotated
    captions and spine credits are set up the side of a photograph, and their
    aspect ratio means the same thing only once it is the right way up.
    """
    x0, y0, x1, y1 = box
    width = max(1e-9, x1 - x0)
    height = max(1e-9, y1 - y0)

    if label in ONE_LINE_LABELS:
        if height > width:
            width, height = height, width
        expected = max(1.0, width / height / CHAR_ASPECT)
        return len(text) / expected

    area = (width / max(1e-9, page_w)) * (height / max(1e-9, page_h))
    return len(text) / max(1e-9, area)


def median_fills(
    rows: Iterable[Mapping],
    *,
    min_samples: int = MIN_SPARSE_SAMPLES,
) -> dict[str, float]:
    """Median fill per label, over the regions that have text.

    Each row is a mapping with ``text``, ``label``, ``box`` and ``page_size``.
    Empty regions are excluded for the obvious reason: they are what the median
    is being used to find.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row["text"]:
            continue
        page_w, page_h = row["page_size"]
        grouped[row["label"]].append(
            fill_of(row["text"], row["label"], row["box"], page_w, page_h)
        )
    return {
        label: statistics.median(values)
        for label, values in grouped.items()
        if len(values) >= min_samples
    }


# Measured over the 19757 hand-drawn regions of the Glasbena Mladina GT with
# `median_fills`, i.e. what a *working* region of each label scores. Regenerate
# with `python -m rare.parse.quality <omnidocbench.json> <annotations.json>`
# after changing `fill_of`, or the sparse test silently drifts.
DEFAULT_FILL_MEDIANS: dict[str, float] = {
    'Author': 1.7571,
    'Byline': 1.3524,
    'Caption': 9688.6390,
    'Dateline': 1.4111,
    'Deck': 4693.5391,
    'Dropcap': 2460.0557,
    'EditNote': 11586.0496,
    'FigByline': 1.6614,
    'Footer': 1.3984,
    'Footnote': 11326.7673,
    'Header': 1.2255,
    'Headline': 3.3700,
    'Kicker': 1.5138,
    'Literary': 8446.5295,
    'Literature': 7126.0731,
    'MarginNote': 9651.9890,
    'OrderedList': 9502.6770,
    'PageNum': 0.9665,
    'Paragraph': 11840.0451,
    'Question': 9774.1309,
    'Quote': 10114.9428,
    'Section': 61.5733,
    'Subhead': 2.9639,
    'Subsubhead': 1.3345,
    'TOC': 3590.1853,
    'Translator': 1.6637,
    'UnorderedList': 9159.9038,
}


# --- the verdict ------------------------------------------------------------

def assess(
    text: str,
    label: str,
    box: Box,
    page_w: float,
    page_h: float,
    medians: Optional[Mapping[str, float]] = None,
) -> tuple[list[str], float]:
    """Return the region's reasons and its fill ratio (0 when untestable)."""
    text = (text or "").strip()
    if not text:
        return ["empty"], 0.0

    if medians is None:
        medians = DEFAULT_FILL_MEDIANS

    reasons: list[str] = []

    if label not in NO_JUNK_CHECK_LABELS and junk_ratio(text, label) >= JUNK_RATIO:
        reasons.append("junk")

    if alien_ratio(text) >= ALIEN_RATIO:
        reasons.append("alien")

    fill = fill_of(text, label, box, page_w, page_h)
    median = 0.0 if label in NO_FILL_CHECK_LABELS else medians.get(label, 0.0)
    ratio = fill / median if median else 0.0
    # Two conditions, because either alone is wrong. The median catches "less
    # than its label normally holds", but Headlines and Subheads normally run
    # to two or three lines, so a perfectly good one-line headline sits at a
    # third of the median and every one of them came back flagged. The absolute
    # floor says the text does not fill even one line of a box that tall, which
    # a complete one-line headline always does.
    absolute = fill / ABSOLUTE_FILL_FLOOR if label in ONE_LINE_LABELS else ratio
    if median and ratio < SPARSE_RATIO and absolute < 1.0:
        reasons.append("sparse")

    return reasons, ratio


def is_suspect(
    text: str,
    label: str,
    box: Box,
    page_w: float,
    page_h: float,
    *,
    reasons: Sequence[str] = REASONS,
    medians: Optional[Mapping[str, float]] = None,
) -> list[str]:
    """The subset of `reasons` this region trips. Empty list means it is fine."""
    found, _ = assess(text, label, box, page_w, page_h, medians)
    wanted = frozenset(reasons)
    return [reason for reason in found if reason in wanted]


def box_from_norm_1000(
    bbox_norm_1000: Sequence[float], page_w: float, page_h: float
) -> Box:
    """0–1000 normalised region coordinates → page pixels.

    `fill_of` needs the box's true aspect ratio, and 0–1000 space has thrown it
    away by normalising each axis independently.
    """
    x0, y0, x1, y1 = bbox_norm_1000
    return (
        x0 / 1000.0 * page_w,
        y0 / 1000.0 * page_h,
        x1 / 1000.0 * page_w,
        y1 / 1000.0 * page_h,
    )


# --- regenerating the medians ----------------------------------------------

def _main(argv: Sequence[str]) -> int:
    """Print a fresh `DEFAULT_FILL_MEDIANS` from an OmniDocBench export."""
    import json
    from pathlib import Path

    if len(argv) != 2:
        print(__doc__)
        print("usage: python -m rare.parse.quality "
              "<omnidocbench.json> <annotations.json>")
        return 2

    odb = json.loads(Path(argv[0]).read_text())
    coco = json.loads(Path(argv[1]).read_text())
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    labels = {a["id"]: categories[a["category_id"]] for a in coco["annotations"]}

    rows = []
    for page in odb:
        info = page["page_info"]
        page_size = (info["width"], info["height"])
        for det in page["layout_dets"]:
            label = labels.get(det["anno_id"], det.get("category_type", ""))
            if label in NON_TEXT_LABELS or det.get("ignore"):
                continue
            xs, ys = det["poly"][0::2], det["poly"][1::2]
            rows.append({
                "text": (det.get("text") or "").strip(),
                "label": label,
                "box": (min(xs), min(ys), max(xs), max(ys)),
                "page_size": page_size,
            })

    print("DEFAULT_FILL_MEDIANS: dict[str, float] = {")
    for label, value in sorted(median_fills(rows).items()):
        print(f"    {label!r}: {value:.4f},")
    print("}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
