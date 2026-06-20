"""Evaluation metrics for document layout analysis."""

import re
import string
import sys
import unicodedata
from collections import Counter

from typing import List, Optional, Dict

from layoutparser.elements import Layout

# All Unicode punctuation characters plus standard ASCII punctuation
PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)


def normalize_answer(text):
    """Normalize text for comparison: lowercase, strip articles, punctuation, extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s):
        return "".join(ch for ch in s if ch not in PUNCT)

    def collapse_whitespace(s):
        return " ".join(token for token in s.split() if token.strip())

    return collapse_whitespace(remove_articles(remove_punctuation(text.lower())))


def f1_score(prediction, ground_truth):
    """Compute token-level F1 score between two text strings.

    Based on the evaluation approach from evaluate_mlqa.py (UniLM).
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _map_label(name: str, mapping: Optional[dict]) -> str:
    """Translate a category name through `mapping`, leaving unmapped names
    unchanged (fall-through). Used to bridge two taxonomies into a shared space."""
    if mapping is None:
        return name
    return mapping.get(name, name)


def mean_average_precision(
    predictions: Layout,
    ground_truths: Layout,
    pred_category_map: Optional[dict] = None,
    gt_category_map: Optional[dict] = None,
    class_agnostic: bool = False,
    class_metrics: bool = False,
) -> Dict[str, float]:
    """Compute mean average precision (mAP) for bounding box detections.

    mAP is computed per class: a prediction only matches a ground-truth box when
    they share the same label. If the detector was trained on a different dataset
    (e.g. DocLayout-YOLO on DocLayNet) its predicted category names will not match
    your ground-truth category names, so every box becomes a false positive /
    false negative and mAP collapses to ~0 even with perfect localization.

    To compare across taxonomies, normalize *both* sides into one shared label
    space (here: OmniDocBench `category_type`) via the two maps, or ignore
    categories entirely with `class_agnostic`.

    Args:
        predictions: Layout elements with .block (x_1, y_1, width, height),
            .type (category name) and .score.
        ground_truths: Ground-truth layout elements with .block and .type.
        pred_category_map: Optional dict translating each *prediction* category
            name into the shared label space, e.g. DocLayNet
            {"Text": "text_block", "Picture": "figure", "Section-header": "title"}.
            Names absent from the dict pass through unchanged. Ignored when
            class_agnostic is True.
        gt_category_map: Optional dict translating each *ground-truth* category
            name into the same shared space (e.g. your source COCO names ->
            OmniDocBench category_type). Names absent pass through unchanged.
        class_agnostic: If True, collapse every box (GT and pred) to a single
            class. Measures pure localization quality, independent of taxonomy.
        class_metrics: If True, also return per-class AP (map_per_class /
            mar_100_per_class) plus a `classes_names` list aligned to it, so you
            can see which of your categories the model handles well.

    Returns:
        Dict of mAP metrics from torchmetrics.
    """
    import torch
    from torchvision.ops import box_convert
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    if class_agnostic:
        target_label_names = ["object"] * len(ground_truths)
        pred_label_names = ["object"] * len(predictions)
    else:
        target_label_names = [_map_label(t.type, gt_category_map) for t in ground_truths]
        pred_label_names = [_map_label(p.type, pred_category_map) for p in predictions]

    # Build a single consistent string -> int label map over the union.
    classes = sorted(set(target_label_names) | set(pred_label_names))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    target_boxes = [
        [t.block.x_1, t.block.y_1, t.block.width, t.block.height]
        for t in ground_truths
    ]
    pred_boxes = [
        [p.block.x_1, p.block.y_1, p.block.width, p.block.height]
        for p in predictions
    ]
    pred_scores = [p.score for p in predictions]

    target_classes = [class_to_idx[name] for name in target_label_names]
    pred_classes = [class_to_idx[name] for name in pred_label_names]

    t_boxes = torch.tensor(target_boxes, dtype=torch.float32).reshape(-1, 4)
    p_boxes = torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4)

    t_boxes_xyxy = box_convert(t_boxes, in_fmt='xywh', out_fmt='xyxy')
    p_boxes_xyxy = box_convert(p_boxes, in_fmt='xywh', out_fmt='xyxy')

    # Wrap in single-image format expected by torchmetrics
    preds = [{
        "boxes": p_boxes_xyxy,
        "scores": torch.tensor(pred_scores, dtype=torch.float32),
        "labels": torch.tensor(pred_classes, dtype=torch.int64),
    }]
    targets = [{
        "boxes": t_boxes_xyxy,
        "labels": torch.tensor(target_classes, dtype=torch.int64),
    }]

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=class_metrics)
    metric.update(preds, targets)
    result = metric.compute()
    if class_metrics:
        # Map the per-class rows back to your category names for readability.
        result["classes_names"] = classes
    return result


def kendall_tau(a: List[int], b: List[int]) -> float: # TODO: a: list[int], b: list[int], but doesn't work with Python<3.9
    """Kendall's tau-b on two rank sequences of equal length. O(n^2)."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
                continue
            if db == 0:
                ties_b += 1
                continue
            if (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    total_a = concordant + discordant + ties_a
    total_b = concordant + discordant + ties_b
    denom = (total_a * total_b) ** 0.5
    return (concordant - discordant) / denom if denom > 0 else 0.0
