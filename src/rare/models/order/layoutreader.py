# RaRe LayoutReader reading-order backend.
#
# The code in this file is original to RaRe (Apache-2.0), but it is a client of
# two CC BY-NC-SA 4.0 artefacts by Hantian Pang:
#   * the inference helpers vendored in `layoutreader_helpers/helpers.py`
#     (from https://github.com/FreeOCR-AI/layoutreader), and
#   * the `hantian/layoutreader` LayoutLMv3 checkpoint it loads at runtime
#     (https://huggingface.co/hantian/layoutreader).
# Running this backend therefore inherits the NonCommercial restriction; see
# NOTICE and licenses/LICENSE-LAYOUTREADER.

import warnings
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

import fitz  # PyMuPDF
import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import LayoutLMv3ForTokenClassification
from transformers.utils import logging

import layoutparser as lp

from rare.models.order.builtin import TopBottomBackend
from rare.models.order.layoutreader_helpers.helpers import prepare_inputs, boxes2inputs, parse_logits, MAX_LEN
from rare.models.registry import register

ORDER_EXCLUDE = frozenset({
    "Header", "Footer", "PageNum", "Footnote", "MarginNote", "Abandon",
    "Advertisement", "Figure", "Caption", "Form", "FigByline",
})


def normalize_bbox(bbox, page_width, page_height, scale=1000):
    """Normalize a bbox from PDF-point coords to the 0-`scale` range the
    LayoutReader model expects. Values are clamped so rounding / slightly
    off-page boxes can never produce an index outside the embedding table.
    bbox: [left, top, right, bottom] in PDF points.
    """
    left, top, right, bottom = bbox

    def n(v, size):
        return min(scale, max(0, int(v / size * scale)))

    return [
        n(left, page_width),
        n(top, page_height),
        n(right, page_width),
        n(bottom, page_height),
    ]


def words_to_lines(words):
    """Group PyMuPDF words into text lines.

    LayoutReader is trained on line/span-level boxes (ReadingBank), NOT on
    individual words. Feeding one box per word both blows past the model's
    sequence limit and is out-of-distribution. Each word tuple from
    page.get_text("words") is:
        (x0, y0, x1, y1, "word", block_no, line_no, word_no)

    Returns line dicts with a union bbox (PDF points) and joined text, given a
    stable top-to-bottom / left-to-right starting order.
    """
    groups = defaultdict(list)
    for w in words:
        groups[(w[5], w[6])].append(w)  # (block_no, line_no)

    lines = []
    for (block_no, line_no), ws in groups.items():
        ws.sort(key=lambda w: w[0])  # left-to-right within the line
        x0 = min(w[0] for w in ws)
        y0 = min(w[1] for w in ws)
        x1 = max(w[2] for w in ws)
        y1 = max(w[3] for w in ws)
        lines.append(
            {
                "bbox": (x0, y0, x1, y1),
                "text": " ".join(w[4] for w in ws),
                "block": block_no,
                "line": line_no,
            }
        )

    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return lines


DEFAULT_CATEGORY_ID=1
DEFAULT_CATEGORIES={
    0: "Abandon",
    1: "Advertisement",
    2: "Author",
    3: "Byline",
    4: "Caption",
    5: "CaptionByline",
    6: "Dateline",
    7: "Deck",
    8: "Dropcap",
    9: "EditNote",
    10: "FigByline",
    11: "Figure",
    12: "Footer",
    13: "Footnote",
    14: "Form",
    15: "Header",
    16: "Headline",
    17: "Kicker",
    18: "Literary",
    19: "Literature",
    20: "MarginNote",
    21: "OrderedList",
    22: "PageNum",
    23: "Paragraph",
    24: "Question",
    25: "Quote",
    26: "Section",
    27: "Subhead",
    28: "Subsubhead",
    29: "TOC",
    30: "Translator",
    31: "UnorderedList"
}

def overlap_ratio(line_box, ann_box, metric="line"):
    """Overlap between a line box and an annotation box, both [x0, y0, x1, y1]
    in the SAME coordinate space.

    metric="line": intersection / line area. This is the meaningful criterion
        for "is this line inside this region": a line is small relative to the
        paragraph that contains it, so true IoU tops out near
        line_area / ann_area and can essentially never reach 0.5.
    metric="iou":  classic intersection over union, for when you really want it.
    """
    ax0, ay0, ax1, ay1 = line_box
    bx0, by0, bx1, by1 = ann_box

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    line_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if metric == "iou":
        ann_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = line_area + ann_area - inter
        return inter / union if union > 0 else 0.0

    return inter / line_area if line_area > 0 else 0.0


def annotation_mean_ranks(
        layout, lines_px, rank, cat_name=None, exclude=ORDER_EXCLUDE,
        metric="line", thresh=0.5, categories=DEFAULT_CATEGORIES
):
    """Mean LayoutReader reading-order position per annotation region.

    anns:     COCO annotation dicts (bbox in image pixels, COCO xywh).
    lines_px: line boxes as [x0, y0, x1, y1] in image pixels, index-aligned
              with `rank`.
    rank:     rank[i] = reading-order position of line i (from parse_logits).
    cat_name: {category_id: name}, used to apply `exclude`.
    exclude:  category names that carry no position in the reading flow.

    A line counts toward an annotation when its overlap with that annotation's
    box is >= `thresh`. Lines can be assigned to more than one annotation if the
    boxes overlap; annotations with no qualifying line get mean_rank None.

    Returns a list of dicts, one per annotation, in the input order.
    """
    name_to_id = {v: k for k, v in categories.items()}
    cat_name = cat_name or {}
    results = []
    for idx, block in enumerate(layout):
        name = block.type
        x_min, y_min, x_max, y_max = block.coordinates
        ann_box = (x_min, y_min, x_max, y_max)

        member_ranks = [
            rank[i]
            for i, lb in enumerate(lines_px)
            if overlap_ratio(lb, ann_box, metric) >= thresh
        ]

        results.append(
            {
                "index": idx,          # 0-based index into `layout`
                "category": name,
                "box": ann_box,
                "in_flow": name not in exclude,
                "n_lines": len(member_ranks),
                "line_ranks": sorted(member_ranks),
                "mean_rank": (
                    sum(member_ranks) / len(member_ranks) if member_ranks else None
                ),
                "median_rank": (
                    median(member_ranks) if member_ranks else None
                ),
            }
        )

    return results


def order_regions(results):
    """Rank regions by their mean line rank and assign `pred_order_id`.

    Ordering, in three tiers:
      1. In-flow text regions that matched lines -> sorted by mean_rank. This is
         the actual prediction.
      2. In-flow regions that matched no line (empty or image-only text
         regions) -> appended, top-to-bottom / left-to-right.
      3. Excluded regions (figures, page furniture, adverts) -> appended last,
         same geometric fallback.

    Every region still gets an id, so the visualization stays complete; only
    tier 1 carries information from the model. Mutates and returns `results`.
    """
    def reading_fallback(r):
        x0, y0, _, _ = r["box"]
        return y0, x0

    tier1 = sorted(
        (r for r in results if r["in_flow"] and r["median_rank"] is not None),
        key=lambda r: r["median_rank"],
    )
    tier2 = sorted(
        (r for r in results if r["in_flow"] and r["median_rank"] is None),
        key=reading_fallback,
    )
    tier3 = sorted((r for r in results if not r["in_flow"]), key=reading_fallback)

    predicted = {id(r) for r in tier1}
    ordered = tier1 + tier2 + tier3
    for pos, r in enumerate(ordered):
        r["pred_order_id"] = pos
        r["predicted"] = id(r) in predicted

    return ordered


@register("order", "layoutreader")
class LayoutReaderBackend:
    """LayoutLMv3 reading order, aggregated from PDF text lines onto regions."""

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        self.model_id: str = cfg.get("model_id", "hantian/layoutreader")
        self.device: Optional[str] = cfg.get("device")
        self.granularity: str = cfg.get("granularity", "auto")
        self.overlap_metric: str = cfg.get("overlap_metric", "line")
        self.overlap_thresh: float = float(cfg.get("overlap_thresh", 0.5))
        self.exclude = frozenset(cfg.get("exclude", ORDER_EXCLUDE))
        self.pdf_root: Optional[Path] = (
            Path(cfg["pdf_root"]) if cfg.get("pdf_root") else None
        )
        if self.granularity not in {"auto", "line", "region"}:
            raise ValueError(
                f"granularity must be auto|line|region, got {self.granularity!r}"
            )

        self._model = None
        self._pdfs: dict[str, object] = {}   # stem -> pdfplumber.PDF | None
        logging.set_verbosity_error()

    def _get_model(self):
        if self._model is None:
            self._model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
            self._model.eval()
        return self._model


    # -- backend contract -------------------------------------------------

    def order(
        self, layout, *, image=None, page_no=None, pdf_stem=None, img_path=None, pdf_root=None
    ) -> list[int]:
        """Return a reading-order permutation of `layout` indices.

        Same contract as the XY-Cut backend: `result[k]` is the index into
        `layout` of the block that comes k-th in reading order.
        """
        n = len(layout)
        if n == 0:
            return []
        if n == 1:
            return [0]

        pdf_name = (
            f"{pdf_root}" if str(pdf_root)[-1] == "/" else f"{pdf_root}/"
            f"{pdf_stem}.pdf"
        )
        doc = fitz.open(pdf_name)
        page = doc[page_no]

        img = Image.open(img_path) if img_path else image
        W, H = img.size

        page_w, page_h = page.rect.width, page.rect.height

        words = page.get_text("words")
        lines = words_to_lines(words)

        boxes_pts = [l["bbox"] for l in lines]  # for drawing
        boxes = [normalize_bbox(b, page_w, page_h) for b in boxes_pts]  # for model

        max_boxes = MAX_LEN - 2
        if len(boxes) > max_boxes:
            lines = lines[:max_boxes]
            boxes_pts = boxes_pts[:max_boxes]
            boxes = boxes[:max_boxes]

        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, self._get_model())

        with torch.no_grad():
            logits = self._get_model()(**inputs).logits.cpu().squeeze(0)

        # orders[k] = index of the box that is k-th in reading order.
        orders = parse_logits(logits, len(boxes))

        # rank[i] = reading position of box i (used as the drawn label).
        rank = [0] * len(boxes)
        for pos, box_idx in enumerate(orders):
            rank[box_idx] = pos

        sx, sy = W / page_w, H / page_h
        boxes_px = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy) for x0, y0, x1, y1 in boxes_pts]

        region_ranks = annotation_mean_ranks(
            layout, boxes_px, rank, exclude=self.exclude,
            metric=self.overlap_metric, thresh=self.overlap_thresh,
        )
        kept_orig = [r["index"] for r in region_ranks]
        ordered = order_regions(region_ranks)

        # Permutation of original layout indices, in reading order.
        seen: set[int] = set()
        indices: list[int] = []
        for r in ordered:
            idx = r["index"]
            if isinstance(idx, int) and 0 <= idx < len(region_ranks):
                orig_i = kept_orig[idx]
                if orig_i not in seen:
                    seen.add(orig_i)
                    indices.append(orig_i)
        for i in range(n):  # re-append any dropped/missing boxes in original order
            if i not in seen:
                indices.append(i)
        return indices
