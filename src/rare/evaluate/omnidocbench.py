"""COCO → OmniDocBench JSON converter (layout structure + reading order only).

We do not emit `text`, `latex`, `html`, `line_with_spans`, `merge_list`,
`attribute`, or `extra.relation`; those belong to OmniDocBench's end-to-end
track. The shape we produce is the per-page object documented at
https://github.com/opendatalab/OmniDocBench and verified against
`demo_data/omnidocbench_demo/OmniDocBench_demo.json`:

    {
      "layout_dets": [
        {"category_type": "...", "poly": [x1,y1, x2,y1, x2,y2, x1,y2],
         "ignore": bool, "order": int|None, "anno_id": int}
      ],
      "page_info": {"page_no": int, "height": int, "width": int,
                     "image_path": str},
      "extra": {"relation": []}
    }

A full file is a JSON array of these objects.

This module is intentionally pure: it consumes COCO-shaped dicts (the same
shape produced by `rare.utils.conversionutils.layout_parser_to_coco` and by
our annotated datasets like `annotations_with_order.json`) and returns
plain dicts. No layoutparser, no pycocotools, no I/O.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

from typing import List

# A `text_source` callable returns the `text` for one layout_det:
#   (image_path, poly, img_w, img_h) -> str
# See `rare.evaluate.pdf_text.PdfTextSource` for the PDF-backed implementation.
# TextSource = Callable[[str, list[float], int, int], str] TODO: does not work with Python<3.9
TextSource = Callable[[str, List[float], int, int], str]


# ---------------------------------------------------------------------------
# Default category map: source COCO `name` → OmniDocBench `category_type`.
# Keys match the names in `datasets/glasbena_mladina/annotations_with_order.json`
# and the `CATEGORIES` list in `src/rare/utils/conversionutils.py`.
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_MAP: dict[str, str] = {
    # OmniDocBench title-class
    "Headline":      "title",
    "Section":       "title",
    "Subhead":       "title",
    "Subsubhead":    "title",
    "Kicker":        "title",
    # text bodies
    "Deck":          "text",
    "Paragraph":     "text",
    "Quote":         "text",
    "Literary":      "text",
    "Literature":    "text",
    "Dropcap":       "text",
    "Byline":        "text",
    "FigByline":     "text",
    "Author":        "text",
    "Translator":    "text",
    "Dateline":      "text",
    "EditNote":      "text",
    "Question":      "text",
    "OrderedList":   "text",
    "UnorderedList": "text",
    "TOC":           "text",
    # figures + captions
    "Figure":        "figure",
    "Caption":       "figure_caption",
    # page furniture
    "Header":        "abandon",
    "Footer":        "abandon",
    "PageNum":       "abandon",
    "Footnote":      "abandon",
    "MarginNote":    "abandon",
    # ignored
    "Abandon":       "abandon",
    "Advertisement": "abandon",
}

UNKNOWN_FALLBACK = "text_block"

# Stub-text scheme: a unique token per layout_det so OmniDocBench's quick_match
# can align predicted markdown paragraphs to GT boxes without real OCR text.
# Both GT and (IoU-relabeled) predictions emit the same token for the same box.
STUB_TEXT_PREFIX  = "__B"
STUB_TEXT_SUFFIX  = "__"
UNMATCHED_PREFIX  = "__UNMATCHED_"

# Module-level set used to warn once per unknown source name.
_warned_unknown: set[str] = set()


def _stub_text(anno_id: int) -> str:
    """Unique stub token for a GT layout_det, used as `text` in stub-text mode."""
    return f"{STUB_TEXT_PREFIX}{anno_id}{STUB_TEXT_SUFFIX}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_to_poly(bbox: list[float]) -> list[float]:
    """COCO `[x, y, w, h]` → OmniDocBench `poly` (8 floats, TL/TR/BR/BL)."""
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _bbox_to_xyxy(bbox: list[float]) -> list[float]:
    """COCO `[x, y, w, h]` → axis-aligned `[x_min, y_min, x_max, y_max]`."""
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def _detection_image_name(file_name: str) -> str:
    """Strip the extension from a COCO `file_name` for the simple-format
    `image_name` field.

    OmniDocBench's `DetectionDatasetSimpleFormat` loader keys predictions by
    `pred["image_name"] + ".jpg"` and matches that against the GT
    `page_info.image_path` (which equals the COCO `file_name`). So `image_name`
    must be the file name *without* its extension; the loader re-appends a
    hard-coded `.jpg`. Our rendered pages are `.jpg`, so this round-trips. Any
    directory component is preserved, matching the GT path verbatim.
    """
    return str(Path(file_name).with_suffix(""))


def _page_no_from_filename(file_name: str) -> int:
    """Mirror the `<stem>_<page>.<ext>` convention used in
    `src/rare/evaluate/datasets.py:182-188`. Returns 0 when the suffix is
    missing or not an int."""
    stem = Path(file_name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return 0
    try:
        return int(parts[1])
    except ValueError:
        return 0


def _resolve_category(name: str, category_map: dict[str, str]) -> str:
    """Look up `name` in `category_map`; fall back to `UNKNOWN_FALLBACK` with
    a once-per-name warning."""
    if name in category_map:
        return category_map[name]
    if name not in _warned_unknown:
        _warned_unknown.add(name)
        warnings.warn(
            f"OmniDocBench converter: unknown source category {name!r}; "
            f"falling back to {UNKNOWN_FALLBACK!r}.",
            stacklevel=3,
        )
    return UNKNOWN_FALLBACK


def _resolve_map(override: Optional[dict[str, str]]) -> dict[str, str]:
    """Merge a user override on top of the default. Override wins; missing
    keys keep their default. Pass `None` to use the default unchanged."""
    if not override:
        return DEFAULT_CATEGORY_MAP
    merged = dict(DEFAULT_CATEGORY_MAP)
    merged.update(override)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coco_page_to_omnidocbench(
    image_info: dict,
    annotations: list[dict],
    categories: list[dict],
    category_map: Optional[dict[str, str]] = None,
    text_stub: bool = False,
    text_source: Optional[TextSource] = None,
) -> dict:
    """Convert one COCO image + its annotations into a single OmniDocBench
    page object.

    Args:
        image_info: COCO `images[i]` dict (`id`, `file_name`, `width`,
            `height`).
        annotations: List of COCO `annotations` dicts already filtered to
            this image (callers usually do `[a for a in anns if
            a["image_id"] == image_info["id"]]`).
        categories: COCO `categories` list — used to resolve `category_id` →
            source name → OmniDocBench `category_type`.
        category_map: Optional override applied on top of
            `DEFAULT_CATEGORY_MAP`. Pass `None` to use the default.
        text_stub: When True, attach `text: "__B<anno_id>__"` to every emitted
            layout_det. Needed by OmniDocBench's end2end pipeline, which reads
            `item["text"]` directly for all text-bearing categories (see
            `src/core/matching/match.py:582` upstream). Stub tokens let
            quick_match align GT boxes to predicted markdown paragraphs by
            exact text match. Ignored when `text_source` is supplied.
        text_source: Optional callable
            `(image_path, poly, img_w, img_h) -> str`. When supplied, takes
            precedence over `text_stub` for the `text` field. An empty return
            falls through to `""` so figure-only / image-only regions become
            quick_match-ignorable rather than mis-aligned. The pipeline-side
            implementation `rare.evaluate.pdf_text.PdfTextSource` extracts
            text from the rendered PDF.
    """
    cmap = _resolve_map(category_map)
    name_by_id = {c["id"]: c["name"] for c in categories}
    img_w = int(image_info["width"])
    img_h = int(image_info["height"])
    file_name = image_info["file_name"]

    layout_dets: list[dict] = []
    for ann in annotations:
        src_name = name_by_id.get(ann["category_id"], "")
        # category_type = _resolve_category(src_name, cmap) if src_name else UNKNOWN_FALLBACK
        category_type = src_name # TODO: For now, keep the category name from the original dataset and do the mapping in YAML
        order = ann.get("order_id")
        anno_id = int(ann["id"])
        poly = _bbox_to_poly(ann["bbox"])
        det: dict = {
            "category_type": category_type,
            "poly":          poly,
            "ignore":        bool(ann.get("ignore", 0)),
            "order":         order if order is not None else None,
            "anno_id":       anno_id,
        }
        if text_source is not None:
            try:
                det["text"] = text_source(file_name, poly, img_w, img_h) or ""
            except Exception:
                det["text"] = ""
        elif text_stub:
            det["text"] = _stub_text(anno_id)
        layout_dets.append(det)

    return {
        "layout_dets": layout_dets,
        "page_info": {
            "page_no":        _page_no_from_filename(file_name),
            "height":         img_h,
            "width":          img_w,
            "image_path":     file_name,
            # OmniDocBench's `pipeline_eval.__init__` indexes this directly
            # (`page['page_info']['page_attribute']`) and `_build_page_attribute_labels`
            # iterates its items to slice results by attribute. Empty dict →
            # one aggregate group ("ALL"), which is what we want without
            # per-attribute splits.
            "page_attribute": {},
        },
        "extra": {"relation": []},
    }


def coco_to_omnidocbench(
    coco_doc: dict,
    category_map: Optional[dict[str, str]] = None,
    text_stub: bool = False,
    text_source: Optional[TextSource] = None,
) -> list[dict]:
    """Convert a full COCO document (`{images, categories, annotations}`) into
    the OmniDocBench list-of-pages shape. Annotations are grouped by
    `image_id` and pages are emitted in the order of `coco_doc["images"]`.
    """
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco_doc.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    categories = coco_doc.get("categories", [])
    return [
        coco_page_to_omnidocbench(
            img, anns_by_image.get(img["id"], []), categories,
            category_map, text_stub=text_stub, text_source=text_source,
        )
        for img in coco_doc.get("images", [])
    ]


def coco_to_detection_prediction(
    coco_doc: dict,
    category_map: Optional[dict[str, str]] = None,
    default_score: float = 1.0,
) -> dict:
    """Convert a COCO *prediction* document into OmniDocBench's
    `detection_dataset_simple_format` prediction JSON (the layout-detection
    counterpart to `coco_to_omnidocbench`).

    The emitted shape — verified against the pinned eval image's loader
    `dataset/detection_dataset.py::DetectionDatasetSimpleFormat` — is::

        {
          "results": [
            {"image_name": "<file_name without ext>",
             "bbox": [x_min, y_min, x_max, y_max],
             "category_id": int,
             "score": float},
            ...
          ],
          "categories": {"<id>": "<category_type>", ...}
        }

    Unlike `coco_to_omnidocbench`, predictions are *flat* (one entry per box,
    not grouped per page) and use axis-aligned `bbox` rather than `poly`.

    Category names: source COCO names are mapped to OmniDocBench
    `category_type` via `DEFAULT_CATEGORY_MAP` (+ `category_map` override),
    exactly as the GT converter does — so GT and predictions share one
    vocabulary and a single eval config can reuse `gt_cat_mapping` as
    `pred_cat_mapping`. Note the loader iterates `pred_cat_mapping`'s keys and
    looks each up in this `categories` map, so every key your config's
    `pred_cat_mapping` references must be a `category_type` that actually
    appears here (otherwise the eval raises `KeyError`).

    Args:
        coco_doc: COCO `{images, categories, annotations}`. Each annotation may
            carry a `score`; missing scores default to `default_score`.
        category_map: Optional override merged on top of `DEFAULT_CATEGORY_MAP`.
        default_score: Confidence assigned to annotations without a `score`.
    """
    cmap = _resolve_map(category_map)
    name_by_id = {c["id"]: c["name"] for c in coco_doc.get("categories", [])}
    file_by_image = {img["id"]: img["file_name"] for img in coco_doc.get("images", [])}

    def category_type(ann: dict) -> str:
        src_name = name_by_id.get(ann["category_id"], "")
        return _resolve_category(src_name, cmap) if src_name else UNKNOWN_FALLBACK

    anns = coco_doc.get("annotations", [])
    # Deterministic ids: sort the distinct category_types so output does not
    # depend on annotation order.
    id_by_type = {t: i for i, t in enumerate(sorted({category_type(a) for a in anns}))}

    results = [
        {
            "image_name": _detection_image_name(file_by_image[ann["image_id"]]),
            "bbox":       _bbox_to_xyxy(ann["bbox"]),
            "category_id": id_by_type[category_type(ann)],
            "score":      float(ann.get("score", default_score)),
        }
        for ann in anns
    ]

    return {
        "results": results,
        "categories": {str(i): t for t, i in id_by_type.items()},
    }


def merge_prediction_pages(
    per_image_dicts: Iterable[dict],
    label_map: dict[int, str],
    default_score: float = 1.0,
) -> dict:
    """Concatenate per-image COCO prediction dicts (as emitted by
    `rare.utils.conversionutils.layout_parser_to_coco`) into OmniDocBench's
    `detection_dataset_simple_format` prediction JSON::

        {"results": [{"image_name", "bbox", "category_id", "score"}, ...],
         "categories": {"<id>": "<source name>", ...}}

    Each annotation keeps its *original* COCO `category_id` (which indexes
    `label_map`), and `categories` is `label_map` verbatim. This mirrors the GT
    converter (which keeps source category names and defers all taxonomy
    mapping to the YAML `pred_cat_mapping`/`gt_cat_mapping`), so a single
    `categories` dict stays consistent across every page.

    Routing through the per-page re-indexing of `coco_to_detection_prediction`
    is wrong here: it renumbers ids per page from the sorted distinct types on
    that page, which neither matches `label_map` nor stays stable across pages.
    """
    results: list[dict] = []
    for doc in per_image_dicts:
        file_by_image = {img["id"]: img["file_name"]
                         for img in doc.get("images", [])}
        for ann in doc.get("annotations", []):
            results.append({
                "image_name":  _detection_image_name(file_by_image[ann["image_id"]]),
                "bbox":        _bbox_to_xyxy(ann["bbox"]),
                "category_id": ann["category_id"],
                "score":       float(ann.get("score", default_score)),
            })
    return {
        "results": results,
        "categories": {str(k): v for k, v in label_map.items()},
    }


# ---------------------------------------------------------------------------
# Stub-text plumbing: IoU relabel + per-page markdown emission
# ---------------------------------------------------------------------------

def _poly_to_bbox(poly: list[float]) -> tuple[float, float, float, float]:
    """8-coord axis-aligned poly → (x1, y1, x2, y2)."""
    xs = poly[0::2]
    ys = poly[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def _iou(a: list[float], b: list[float]) -> float:
    """IoU between two OmniDocBench polys (treated as axis-aligned bboxes)."""
    ax1, ay1, ax2, ay2 = _poly_to_bbox(a)
    bx1, by1, bx2, by2 = _poly_to_bbox(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def relabel_predictions_to_gt(
    pred_pages: list[dict],
    gt_pages: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """In-place rewrite of each pred layout_det's `text` to its IoU-matched
    GT layout_det's `text`, keyed by `page_info.image_path`. Unmatched pred
    boxes get a unique `__UNMATCHED_<pred_anno_id>__` sentinel so quick_match
    has something to anchor against (and they show up as ordering errors).

    Greedy 1-1 matching, highest IoU first, mirroring
    `rare.evaluate._matching.match_by_iou`. Returns `pred_pages` for chaining.

    Requires GT to have been built with `text_stub=True`.
    """
    gt_by_path = {p["page_info"]["image_path"]: p for p in gt_pages}
    for pred_page in pred_pages:
        path = pred_page["page_info"]["image_path"]
        gt_page = gt_by_path.get(path)
        gt_dets = gt_page["layout_dets"] if gt_page else []
        pred_dets = pred_page["layout_dets"]

        # Score all pairs above threshold, then greedily match.
        scored: list[tuple[float, int, int]] = []
        for i, p in enumerate(pred_dets):
            for j, g in enumerate(gt_dets):
                s = _iou(p["poly"], g["poly"])
                if s >= iou_threshold:
                    scored.append((s, i, j))
        scored.sort(reverse=True)
        used_p, used_g = set(), set()
        pred_to_gt: dict[int, int] = {}
        for _, i, j in scored:
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            pred_to_gt[i] = j

        for i, det in enumerate(pred_dets):
            if i in pred_to_gt:
                det["text"] = gt_dets[pred_to_gt[i]]["text"]
            else:
                det["text"] = f"{UNMATCHED_PREFIX}{det['anno_id']}{STUB_TEXT_SUFFIX}"
    return pred_pages


def emit_stub_markdown(
    pages: list[dict],
    out_dir: str | Path,
    paragraph_sep: str = "\n\n",
) -> list[Path]:
    """Write one `<image_stem>.md` per page under `out_dir`, with each
    layout_det's `text` on its own paragraph in `order` order. OmniDocBench's
    end2end loader splits markdown on double newlines, so each token becomes
    its own paragraph for quick_match to pick up.

    Skips layout_dets with `ignore=True` or `order is None` (page furniture
    that has no place in the reading-order stream). Returns the list of
    written `.md` paths for convenience.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for page in pages:
        ordered = [d for d in page["layout_dets"]
                   if not d.get("ignore", False) and d.get("order") is not None
                   and d.get("text")]
        ordered.sort(key=lambda d: d["order"])
        tokens = [d["text"] for d in ordered]
        stem = Path(page["page_info"]["image_path"]).stem
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(paragraph_sep.join(tokens))
        written.append(md_path)
    return written


def load_category_map(path: str | Path) -> dict[str, str]:
    """Load a JSON file of `{source_name: omnidocbench_category_type}` for use
    as the `category_map` override. Caller passes the result to the converter
    functions; we don't merge here so the caller can compose multiple files.
    """
    import json
    return json.loads(Path(path).read_text())
