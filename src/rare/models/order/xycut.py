"""XY-Cut reading-order backend, backed by PaddleX's vendored `xycut_enhanced`.

The vendored algorithm operates on `LayoutBlock` / `LayoutRegion` domain
objects (see src/rare/models/order/xycut_enhanced/). This module is the
adapter: it converts a layoutparser `Layout` into those objects, runs
`xycut_enhanced`, and returns the resulting permutation of indices into
the original layout — the shape `ReadingOrderBackend` requires.
"""

from __future__ import annotations

import warnings

from rare.models.registry import register
from rare.models.order.builtin import TopBottomBackend
from rare.models.order.xycut_enhanced import (
    LayoutBlock,
    LayoutRegion,
    xycut_enhanced,
)

# Glasana-labelled blocks (see layout-parser .../doclayout_yolo/catalog.py)
# emit one of five labels. Map each into PaddleX's BLOCK_LABEL_MAP buckets,
# which drive xycut_enhanced's header/footer/title/vision/text dispatch.
# Unknown labels fall through to "text" (the neutral, position-sorted bucket).
_LABEL_MAP: dict[str, str] = {
    "headline":   "paragraph_title",
    "paragraph":  "text",
    "admonition": "text",
    "figure":     "image",
    "caption":    "figure_title",
}


def _estimate_line_height(page_height: int) -> int:
    # The algorithm uses `region.text_line_height // 2` as a floor-division
    # denominator, so the region mean must be >= 2. Without OCR we don't know
    # the real line height; this estimate (~75 body lines per page) is large
    # enough to avoid div-by-zero on any realistic page size.
    return max(8, page_height // 75)


def _line_metrics_from_ocr(block_bbox, ocr_lines):
    """Per-block (text_line_height, text_line_width, num_of_lines) from OCR lines.

    `ocr_lines` are [x0, y0, x1, y1] *line* boxes in the same pixel space as the
    layout (each box is one text line, so no span-grouping is needed). A line is
    attributed to this block when its centre falls inside `block_bbox`.

    Mirrors PaddleX's LayoutBlock.group_boxes_into_lines, which sets
    text_line_height / text_line_width to the mean over the block's lines.
    Returns None when no line falls inside the block (caller should fall back to
    the page-height estimate).
    """
    bx0, by0, bx1, by1 = block_bbox
    heights, widths = [], []
    for lx0, ly0, lx1, ly1 in ocr_lines:
        cx, cy = (lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            heights.append(ly1 - ly0)
            widths.append(lx1 - lx0)
    if not heights:
        return None
    return (
        sum(heights) / len(heights),
        sum(widths) / len(widths),
        len(heights),
    )


@register("order", "paddlex-xy-cut")
class XYCutBackend:
    """Recursive XY-Cut with PaddleX's title/header/footer/vision heuristics."""

    def order(
        self, layout, *, image=None, page_no=None, pdf_stem=None, ocr_lines=None
    ) -> list[int]:
        """Return a reading-order permutation of `layout` indices.

        `ocr_lines`, when supplied, is a list of [x0, y0, x1, y1] text-line boxes
        in the same pixel space as the layout. Each block then gets real
        text_line_height / width / num_of_lines from the lines inside it; blocks
        with no lines (and the whole page when `ocr_lines` is None) fall back to
        a page-height estimate.
        """
        n = len(layout)
        if n == 0:
            return []
        if n == 1:
            return [0]

        if image is not None:
            page_w, page_h = image.size
            page_bbox = [0, 0, int(page_w), int(page_h)]
        else:
            xs = [b.coordinates[0] for b in layout] + [b.coordinates[2] for b in layout]
            ys = [b.coordinates[1] for b in layout] + [b.coordinates[3] for b in layout]
            page_bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        fallback_h = _estimate_line_height(page_bbox[3] - page_bbox[1])

        blocks: list[LayoutBlock] = []
        kept_orig: list[int] = []          # original layout index for each kept block
        for orig_i, block in enumerate(layout):
            x1, y1, x2, y2 = (int(v) for v in block.coordinates)
            # Drop sub-2px slivers: shrink_overlapping_boxes (frozen) collapses them
            # to a 1px box, which produces a zero-width projection interval and the
            # "zero-size array to reduction" crash. Re-added in original order below.
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            bbox = [x1, y1, x2, y2]
            label = _LABEL_MAP.get(getattr(block, "type", None) or "", "text")
            lb = LayoutBlock(label=label, bbox=bbox)
            # ... (metrics / text_line_height / num_of_lines / seg coords, unchanged) ...
            blocks.append(lb)
            kept_orig.append(orig_i)

        region = LayoutRegion(page_bbox, blocks)
        try:
            ordered = xycut_enhanced(region)
        except Exception as exc:
            warnings.warn(
                f"XY-Cut failed ({exc!r}); falling back to top-to-bottom ordering.",
                RuntimeWarning,
                stacklevel=2,
            )
            return TopBottomBackend().order(
                layout, image=image, page_no=page_no, pdf_stem=pdf_stem
            )

        seen: set[int] = set()
        indices: list[int] = []
        for b in ordered:
            idx = getattr(b, "index", None)
            if isinstance(idx, int) and 0 <= idx < len(blocks):
                orig_i = kept_orig[idx]
                if orig_i not in seen:
                    seen.add(orig_i)
                    indices.append(orig_i)
        for i in range(n):           # re-append any dropped/missing boxes in original order
            if i not in seen:
                indices.append(i)
        return indices