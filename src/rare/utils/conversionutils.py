"""Utilities for converting layout-parser output to COCO format."""

# Mapping from DocLayout-YOLO PubLayNet labels to our category IDs
DOCLAYOUT_YOLO_PUBLAY_TO_OUR_LABEL_MAP = {
    "title": 5,        # -> Headline
    "plain text": 10,  # -> Paragraph
    "table": 13,       # -> Figure
    "figure": 13,      # -> Figure
    "table_caption": 14,  # -> Caption
    "figure_caption": 14, # -> Caption
}

# Full category list used in our annotation scheme
CATEGORIES = [
    {"id": 0,  "name": "Header",        "supercategory": None},
    {"id": 1,  "name": "Footer",        "supercategory": None},
    {"id": 2,  "name": "PageNum",       "supercategory": None},
    {"id": 3,  "name": "Section",       "supercategory": None},
    {"id": 4,  "name": "Kicker",        "supercategory": None},
    {"id": 5,  "name": "Headline",      "supercategory": None},
    {"id": 6,  "name": "Deck",          "supercategory": None},
    {"id": 7,  "name": "Subhead",       "supercategory": None},
    {"id": 8,  "name": "Byline",        "supercategory": None},
    {"id": 9,  "name": "Dropcap",       "supercategory": None},
    {"id": 10, "name": "Paragraph",     "supercategory": None},
    {"id": 11, "name": "Quote",         "supercategory": None},
    {"id": 12, "name": "Footnote",      "supercategory": None},
    {"id": 13, "name": "Figure",        "supercategory": None},
    {"id": 14, "name": "Caption",       "supercategory": None},
    {"id": 15, "name": "Advertisement", "supercategory": None},
    {"id": 16, "name": "Dateline",      "supercategory": None},
    {"id": 17, "name": "EditNote",      "supercategory": None},
    {"id": 18, "name": "MarginNote",    "supercategory": None},
    {"id": 19, "name": "UnorderedList", "supercategory": None},
    {"id": 20, "name": "OrderedList",   "supercategory": None},
    {"id": 21, "name": "Byline",        "supercategory": None},
    {"id": 22, "name": "Translator",    "supercategory": None},
    {"id": 23, "name": "TOC",           "supercategory": None},
    {"id": 24, "name": "Literary",      "supercategory": None},
]

# Default category ID for labels not found in the mapping
DEFAULT_CATEGORY_ID = 1


def layout_parser_to_coco(
    layout,
    img_info,
    categories,
    predicted_order=None,
):
    """Convert a layoutparser Layout to COCO annotation format.

    Args:
        layout: A layoutparser Layout (list of TextBlocks).
        img_info: Dict with image metadata (id, file_name, width, height).
        categories: COCO categories dict (unused — we use our own CATEGORIES).
        category_mapping: Dict mapping model label names to our category IDs.
        predicted_order: Optional list[int] — a permutation of layout indices
            such that layout[predicted_order[k]] is the k-th region in reading
            order. When provided, each annotation gets an `order_id` field
            with its 0-based reading-order rank; consumers sort by order_id
            to walk the page in reading order.

    Returns:
        A COCO-format dict with 'images', 'annotations', and 'categories'.
    """
    rank_by_layout_idx: dict[int, int] = {}
    if predicted_order is not None:
        for rank, layout_idx in enumerate(predicted_order):
            rank_by_layout_idx[layout_idx] = rank

    annotations = []

    name_to_id = {v: k for k, v in categories.items()}
    for idx, block in enumerate(layout, start=1):
        category_id = name_to_id.get(block.type, DEFAULT_CATEGORY_ID)
        x_min, y_min, x_max, y_max = block.coordinates
        width = x_max - x_min
        height = y_max - y_min

        ann = {
            "id": idx,
            "image_id": img_info["id"],
            "category_id": int(category_id),
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": False,
            "score": float(block.score) if hasattr(block, "score") else 1.0,
        }
        if predicted_order is not None:
            ann["order_id"] = rank_by_layout_idx.get(idx - 1, -1)
        annotations.append(ann)

    coco_categories = [
        {
            "id": name,
            "name": value
        } for name, value in categories.items()
    ]

    return {
        "images": [{
            "id": img_info["id"],
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
        }],
        "annotations": annotations,
        "categories": coco_categories,
    }


def scale_coco_annotations(annotations, src_wh, dst_wh):
    """Rescale COCO annotation boxes from one render size to another.

    Use when boxes were produced against an image of size `src_wh` but you want
    to draw them on a render of the same page at a different size `dst_wh`
    (e.g. a COCO file emitted at high DPI displayed over a low-DPI image). Boxes
    are stored as absolute pixels, so the only correction is a uniform scale.

    The x-axis and y-axis are scaled by their own dimension ratios rather than a
    single DPI scalar: rasterizing rounds points->pixels independently per axis,
    so `dst_w/src_w` and `dst_h/src_h` can differ by a fraction of a pixel, and
    using each absorbs that rounding exactly.

    Args:
        annotations: list of COCO annotation dicts (`bbox = [x, y, w, h]`).
        src_wh: (width, height) the annotations were created against.
        dst_wh: (width, height) of the image you want to draw on.

    Returns:
        A new list of annotation dicts with `bbox`/`area` scaled; all other
        fields (`category_id`, `order_id`, `score`, ...) are copied unchanged.
    """
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    sx = dst_w / src_w
    sy = dst_h / src_h

    scaled = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        new_ann = dict(ann)
        new_ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        if "area" in ann:
            new_ann["area"] = ann["area"] * sx * sy
        scaled.append(new_ann)
    return scaled


def rescale_coco_to(coco, target_wh):
    """Rescale a whole COCO dict's boxes to new per-image sizes.

    Args:
        coco: a COCO dict (`images`, `annotations`, `categories`).
        target_wh: maps `image_id -> (target_width, target_height)`.

    Returns:
        A new COCO dict with each image's `width`/`height` set to its target and
        all of its annotations scaled accordingly (see `scale_coco_annotations`).
        Images absent from `target_wh` are passed through unscaled.
    """
    src_wh = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    out_images = []
    for img in coco["images"]:
        if img["id"] in target_wh:
            tw, th = target_wh[img["id"]]
            out_images.append({**img, "width": tw, "height": th})
        else:
            out_images.append(dict(img))

    out_annotations = []
    by_image: dict = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    for image_id, anns in by_image.items():
        if image_id in target_wh:
            out_annotations.extend(
                scale_coco_annotations(anns, src_wh[image_id], target_wh[image_id])
            )
        else:
            out_annotations.extend(dict(a) for a in anns)

    return {**coco, "images": out_images, "annotations": out_annotations}
