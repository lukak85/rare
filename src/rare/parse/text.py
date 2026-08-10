"""Per-region text extraction via pdfplumber.

Lifted from build_doc._extract_text_for_page so the parse pipeline and any
future evaluation flows share one implementation.

Extraction is deliberately *not* done by cropping the page once per region.
``Page.crop``/``Page.within_bbox`` keep every glyph whose bbox merely
*intersects* the rectangle, and they clip the retained glyph boxes to it. Both
behaviours corrupt the text of adjacent regions:

* a neighbouring line that pokes into the box by a fraction of a point is
  pulled in wholesale — a Byline sitting 0.5pt inside a Paragraph's box ends up
  appended to the paragraph;
* clipping collapses the vertical distance between that foreign line and the
  region's own first line, so ``extract_text``'s line clustering (default
  ``y_tolerance=3``) merges them into a single line and sorts the result by
  ``x0`` — interleaving the two strings character by character
  (``"estacij." + "BREDA"`` → ``"eBstRacEijD.A"``).

So instead the page's words are extracted *once*, with their true unclipped
geometry, and each word is then assigned to exactly one region. A word can
never appear in two regions, and lines are rebuilt from real page coordinates.
"""

from __future__ import annotations

import logging
from typing import Hashable, Iterable, Mapping, Sequence

import pdfplumber

from rare.parse.clean import STRUCTURED_LABELS, normalize_text

logger = logging.getLogger(__name__)

# Word-building tolerances handed to pdfplumber. Deliberately tighter than
# pdfplumber's defaults (3.0/3.0): the magazine scans set text densely, and a
# loose y_tolerance is what lets two adjacent baselines merge into one line.
X_TOLERANCE = 1.5
Y_TOLERANCE = 2.0
# Two words belong to the same visual line if their tops differ by less than
# this (PDF points).
LINE_TOLERANCE = 2.0
# A word whose centre lies outside every region is still claimed by a region
# that covers at least this fraction of its area.
MIN_OVERLAP_RATIO = 0.5

# A box is (x0, top, x1, bottom) in PDF points, top-left origin — the same
# convention pdfplumber uses for word "top"/"bottom" and for crop rectangles.
Box = tuple[float, float, float, float]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _word_box(word: Mapping) -> Box:
    return (word["x0"], word["top"], word["x1"], word["bottom"])


def assign_words_to_boxes(
    words: Iterable[Mapping],
    boxes: Sequence[tuple[Hashable, Box]],
    *,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> tuple[dict[Hashable, list[Mapping]], list[Mapping]]:
    """Assign each word to at most one box.

    A word goes to the box containing its centre point. When several boxes
    contain it — the nested case, e.g. a Byline drawn inside a Paragraph — the
    *smallest* box wins, so the more specific region keeps its own text.

    A word whose centre lies in no box (regions rarely tile the page exactly)
    falls back to the box covering the largest share of it, provided that share
    reaches `min_overlap_ratio`.

    Returns ``({box_key: [word, ...]}, [unassigned_word, ...])``.
    """
    assigned: dict[Hashable, list[Mapping]] = {key: [] for key, _ in boxes}
    unassigned: list[Mapping] = []

    for word in words:
        wbox = _word_box(word)
        cx = (wbox[0] + wbox[2]) / 2.0
        cy = (wbox[1] + wbox[3]) / 2.0

        containing = [
            (_area(box), key)
            for key, box in boxes
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
        ]
        if containing:
            assigned[min(containing)[1]].append(word)
            continue

        warea = _area(wbox)
        if warea > 0 and boxes:
            overlap, key = max(
                (_intersection_area(wbox, box), key) for key, box in boxes
            )
            if overlap >= min_overlap_ratio * warea:
                assigned[key].append(word)
                continue

        unassigned.append(word)

    return assigned, unassigned


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def split_into_columns(words: Sequence[Mapping]) -> list[list[Mapping]]:
    """Partition a region's words into text columns, left to right.

    Layout detectors regularly emit one wide box over a two-column block. Left-
    and right-column lines then share a baseline to within a fraction of a
    point, so clustering by ``top`` alone would merge them into one line and
    x-sort them — splicing the two columns together word by word.

    A gutter is found by projecting every word's x-range onto the x axis and
    looking for a strip the *whole region* leaves uncovered. Projecting over
    all lines rather than each line in isolation is what makes this reliable:
    one line's wide inter-word gap is covered by the forty lines around it,
    while a real gutter is empty down the entire column.
    """
    if len(words) < 2:
        return [list(words)]

    # Scale the gutter to the type size, floored so that ordinary inter-word
    # spacing (~2-4pt at this body size) can never be mistaken for a gutter.
    min_gutter = max(4.0, 0.6 * _median([w["bottom"] - w["top"] for w in words]))

    spans: list[list[float]] = []
    for x0, x1 in sorted((w["x0"], w["x1"]) for w in words):
        if spans and x0 - spans[-1][1] <= min_gutter:
            spans[-1][1] = max(spans[-1][1], x1)
        else:
            spans.append([x0, x1])

    if len(spans) < 2:
        return [list(words)]

    columns: list[list[Mapping]] = [[] for _ in spans]
    for word in words:
        cx = (word["x0"] + word["x1"]) / 2.0
        # Spans are disjoint and ordered; take the first one covering the word.
        for index, (left, right) in enumerate(spans):
            if left <= cx <= right:
                columns[index].append(word)
                break
        else:
            columns[0].append(word)

    return [column for column in columns if column]


def _cluster_by_top(
    words: Sequence[Mapping], line_tolerance: float
) -> list[list[Mapping]]:
    """Group a column's words into lines by the top of their glyph boxes."""
    lines: list[list[Mapping]] = []
    line_top: float | None = None
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if line_top is None or word["top"] - line_top > line_tolerance:
            lines.append([])
            line_top = word["top"]
        lines[-1].append(word)
    return lines


def _merge_shared_baselines(
    lines: Sequence[Sequence[Mapping]], line_tolerance: float
) -> list[list[Mapping]]:
    """Rejoin lines that `_cluster_by_top` split across a single baseline.

    A line setting two type sizes — a bold lead-in, a run-in head — puts all
    its words on one baseline, but their tops sit an ascender apart, which at
    display sizes exceeds `line_tolerance`. Clustering by ``top`` then splits
    one visual line in two and emits the larger type first, reversing the
    halves whenever the larger type is not the leftmost: the running header
    "od tam **in tod**..." came back as "in tod od tam".

    Merging on ``bottom`` afterwards, rather than clustering on it throughout,
    is what keeps drop caps intact. A capital spanning three lines shares its
    top with the first of them but bottoms out level with the last, so
    clustering by ``bottom`` would strand it mid-paragraph; here its bottom
    lies nowhere near its neighbours' and no merge fires.
    """
    merged: list[list[Mapping]] = []
    for line in lines:
        baseline = _median([w["bottom"] for w in line])
        if (
            merged
            and abs(_median([w["bottom"] for w in merged[-1]]) - baseline)
            <= line_tolerance
        ):
            merged[-1].extend(line)
        else:
            merged.append(list(line))
    return merged


def words_to_text(
    words: Sequence[Mapping], *, line_tolerance: float = LINE_TOLERANCE
) -> str:
    """Join a region's words back into text, one line per visual line.

    Columns are read left to right, each top to bottom. Lines are separated by
    ``\\n`` rather than spaces because `normalize_text` de-hyphenates and
    unwraps *across* newlines; collapsing them here would leave hyphenated
    line-breaks welded into the middle of words.
    """
    if not words:
        return ""

    lines: list[list[Mapping]] = []
    for column in split_into_columns(words):
        lines.extend(
            _merge_shared_baselines(
                _cluster_by_top(column, line_tolerance), line_tolerance
            )
        )

    return "\n".join(
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))
        for line in lines
    )


def extract_text_for_boxes(
    pdf_page: pdfplumber.page.Page,
    boxes: Sequence[tuple[Hashable, Box]],
) -> dict[Hashable, str]:
    """Return ``{box_key: text}`` for boxes given in PDF points, top-origin.

    The shared core: extracts the page's words once, partitions them across the
    boxes, and rebuilds each box's text. Returned text is raw (no
    `normalize_text`) so callers can decide per region.
    """
    words = pdf_page.extract_words(
        x_tolerance=X_TOLERANCE,
        y_tolerance=Y_TOLERANCE,
        keep_blank_chars=False,
    )
    assigned, unassigned = assign_words_to_boxes(words, boxes)

    if unassigned:
        # Words covered by no region mean the layout detector missed something —
        # worth surfacing, but not an error: the old crop-based code silently
        # produced nothing for them too.
        logger.debug(
            "page %s: %d/%d words fall outside every region",
            getattr(pdf_page, "page_number", "?"),
            len(unassigned),
            len(words),
        )

    return {key: words_to_text(ws) for key, ws in assigned.items()}


def extract_text_for_page(
    pdf: pdfplumber.PDF,
    page_no: int,
    regions: list[dict],
    img_width: float,
    img_height: float,
) -> dict[str, str]:
    """Return {region_id: text} for all regions on a page.

    Each region must carry a `bbox_norm_1000` field — [x0, y0, x1, y1] in
    0–1000 normalised image space (top-left origin). The function scales
    those coordinates to PDF point space (which may differ from the rendered
    image dimensions) before extracting.
    """
    pdf_page = pdf.pages[page_no]
    pw, ph = pdf_page.width, pdf_page.height

    boxes: list[tuple[Hashable, Box]] = []
    for region in regions:
        x0n, y0n, x1n, y1n = region["bbox_norm_1000"]
        boxes.append(
            (
                region["region_id"],
                (
                    x0n / 1000.0 * pw,
                    y0n / 1000.0 * ph,
                    x1n / 1000.0 * pw,
                    y1n / 1000.0 * ph,
                ),
            )
        )

    try:
        raw_by_id = extract_text_for_boxes(pdf_page, boxes)
    except Exception:
        logger.exception("text extraction failed on page %d", page_no)
        return {region["region_id"]: "" for region in regions}

    results: dict[str, str] = {}
    for region in regions:
        raw = raw_by_id.get(region["region_id"], "").strip()
        # Strip line-wrap newlines / de-hyphenate prose; leave structured
        # regions (tables, lists) untouched, where newlines carry meaning.
        if region.get("label") in STRUCTURED_LABELS:
            results[region["region_id"]] = raw
        else:
            results[region["region_id"]] = normalize_text(raw)

    return results
