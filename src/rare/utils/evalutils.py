"""Evaluation metrics for document layout analysis."""

import re
import string
import sys
import unicodedata
from collections import Counter

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


def mean_average_precision(predictions, ground_truths):
    """Compute mean average precision (mAP) for bounding box detections.

    Args:
        predictions: List of layout elements with .block attributes (x_1, y_1, width, height).
        ground_truths: List of ground-truth layout elements with .block attributes.

    Returns:
        Dict of mAP metrics from torchmetrics.
    """
    import torch
    from torchvision.ops import box_convert
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    target_boxes = [
        [t.block.x_1, t.block.y_1, t.block.width, t.block.height]
        for t in ground_truths
    ]

    pred_boxes = [
        [p.block.x_1, p.block.y_1, p.block.width, p.block.height]
        for p in predictions
    ]

    t_boxes = torch.tensor(target_boxes, dtype=torch.float32)
    p_boxes = torch.tensor(pred_boxes, dtype=torch.float32)

    t_boxes_xyxy = box_convert(t_boxes, in_fmt='xywh', out_fmt='xyxy')
    p_boxes_xyxy = box_convert(p_boxes, in_fmt='xywh', out_fmt='xyxy')

    # Wrap in single-image format expected by torchmetrics
    preds = [{
        "boxes": torch.tensor(p_boxes_xyxy, dtype=torch.float32),
        "scores": torch.ones(len(p_boxes_xyxy)),
        "labels": torch.ones(len(p_boxes_xyxy), dtype=torch.long),
    }]
    targets = [{
        "boxes": torch.tensor(t_boxes_xyxy, dtype=torch.float32),
        "labels": torch.ones(len(t_boxes_xyxy), dtype=torch.long),
    }]

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    return metric.compute()


def kendall_tau(a: list[int], b: list[int]) -> float:
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
