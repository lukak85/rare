"""Merge paragraph regions that flow as one logical paragraph.

A single logical paragraph is frequently split into two physically separate
regions by page geometry: a column break, a page break, or a figure/table
inserted mid-paragraph. The layout detector emits these as two `Paragraph`
regions; this pass re-joins them.

Operates on the same structures the pipeline already builds: a list of region
dicts in reading order (each with `region_id`, `label`, `bbox_norm_1000`) and a
`{region_id: text}` map. It is intentionally string/geometry-light — the strong
signal here is linguistic (does the first block end mid-sentence and the second
begin as a continuation), corroborated by category (don't merge across a
heading; do skip over floats).
"""

from __future__ import annotations

# Categories whose text can flow across a break and be merged.
MERGEABLE_LABELS: frozenset[str] = frozenset({"Paragraph"})

# Floats that may interrupt a paragraph mid-flow. The merge test skips *over*
# these so a figure inserted between two halves doesn't block the join, but they
# are kept in the output, in place.
SKIP_LABELS: frozenset[str] = frozenset(
    {"Figure", "Caption", "FigByline", "Table"}
)

# A first block ending in one of these is a paragraph that finished naturally —
# do not merge. Trailing closing quotes/brackets are stripped before the test.
_SENTENCE_END = ("。", ".", "!", "?", "…", "!", "?")
_TRAILING_CLOSERS = "\"”’'»)]"


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    return stripped.endswith(_SENTENCE_END)


def _is_continuation(text: str) -> bool:
    """True when `text` reads like the tail of an interrupted sentence."""
    head = text.lstrip()
    if not head:
        return False
    first = head[0]
    return first.islower() or first.isdigit() or first in ",;)–-"


def should_merge(prev_text: str, cur_text: str) -> bool:
    """Decide whether `cur_text` continues the paragraph in `prev_text`."""
    if not prev_text.strip() or not cur_text.strip():
        return False
    if _ends_sentence(prev_text):
        return False
    return _is_continuation(cur_text)


def join_texts(prev_text: str, cur_text: str) -> str:
    """Concatenate two paragraph halves, de-hyphenating at the seam."""
    prev = prev_text.rstrip()
    cur = cur_text.lstrip()
    if prev.endswith("-") and cur[:1].islower():
        # word split across the break: "informa-" + "tion" -> "information"
        return prev[:-1] + cur
    return prev + " " + cur


def _union(a: list[float], b: list[float]) -> list[float]:
    """Bounding box covering both regions, [x0, y0, x1, y1]."""
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def merge_flowing_paragraphs(
    regions: list[dict],
    texts: dict[str, str],
    *,
    mergeable: frozenset[str] = MERGEABLE_LABELS,
    skip: frozenset[str] = SKIP_LABELS,
) -> tuple[list[dict], dict[str, str]]:
    """Join consecutive paragraph regions that form one logical paragraph.

    `regions` must be in reading order. Returns a new `(regions, texts)` pair:
    when two paragraphs merge, the second is absorbed into the first (text
    concatenated, bbox unioned) and dropped from the region list. Float regions
    (figures, tables, captions) are passed through untouched but do not break
    the flow, so a paragraph interrupted by a figure still re-joins.
    """
    out_regions: list[dict] = []
    out_texts: dict[str, str] = {}
    open_idx: int | None = None  # index in out_regions of the paragraph still open

    for region in regions:
        label = region["label"]
        rid = region["region_id"]
        text = (texts.get(rid) or "").strip()

        if label in mergeable and open_idx is not None:
            prev = out_regions[open_idx]
            prev_text = out_texts[prev["region_id"]]
            if should_merge(prev_text, text):
                out_texts[prev["region_id"]] = join_texts(prev_text, text)
                prev["bbox_norm_1000"] = _union(
                    prev["bbox_norm_1000"], region["bbox_norm_1000"]
                )
                continue  # region absorbed; open paragraph stays open

        out_regions.append(region)
        out_texts[rid] = text

        if label in mergeable:
            open_idx = len(out_regions) - 1
        elif label in skip:
            pass  # float interrupts visually but not logically; keep flow open
        else:
            open_idx = None  # heading / other prose ends the paragraph

    return out_regions, out_texts