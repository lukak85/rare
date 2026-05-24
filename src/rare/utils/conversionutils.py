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
DEFAULT_CATEGORY_ID = 4  # Kicker


def layout_parser_to_coco(
    layout,
    img_info,
    categories,
    category_mapping=DOCLAYOUT_YOLO_PUBLAY_TO_OUR_LABEL_MAP,
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

    for idx, block in enumerate(layout, start=1):
        category_id = category_mapping.get(block.type, DEFAULT_CATEGORY_ID)
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

    return {
        "images": [{
            "id": img_info["id"],
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
        }],
        "annotations": annotations,
        "categories": CATEGORIES,
    }
