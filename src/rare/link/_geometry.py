"""Box helpers for the linking passes.

Deliberately overlap-ratio based rather than IoU: a caption is a thin strip and
a figure is a large block, so their IoU is tiny even when the caption plainly
belongs to the figure. `rare.evaluate._matching.iou` is the right tool for
detection matching and the wrong one here.
"""

from __future__ import annotations

from typing import Optional

# (x1, y1, x2, y2) in page-image pixels, top-left origin.
Box = tuple[float, float, float, float]


def box_of(item) -> Box:
    b = item.provenance.bbox
    return float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])


def width(box: Box) -> float:
    return max(0.0, box[2] - box[0])


def height(box: Box) -> float:
    return max(0.0, box[3] - box[1])


def centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def h_overlap_ratio(a: Box, b: Box) -> float:
    """Shared width as a fraction of the narrower box.

    Using the narrower box as the denominator is what lets a short caption
    under a wide figure still score 1.0.
    """
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrower = min(width(a), width(b))
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def vertical_gap(a: Box, b: Box) -> float:
    """Vertical distance between two boxes; 0 when they overlap."""
    if a[3] <= b[1]:
        return b[1] - a[3]
    if b[3] <= a[1]:
        return a[1] - b[3]
    return 0.0


def is_below(caption: Box, figure: Box) -> bool:
    """True when the caption's centre sits below the figure's centre."""
    return centre(caption)[1] > centre(figure)[1]


def gap_between(a: Box, b: Box) -> float:
    """Edge-to-edge distance, 0 when the boxes intersect."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def page_size(doc, page_no: int) -> tuple[float, float]:
    page = doc.pages.get(page_no)
    if page is None:
        return 1000.0, 1000.0
    return float(page.width), float(page.height)


def item_page(item) -> Optional[int]:
    return item.provenance.page_no
