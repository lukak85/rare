"""IoU-based matching between predicted and ground-truth bounding boxes."""

from __future__ import annotations


def iou(a, b) -> float:
    """IoU between two boxes, each given as an lp.TextBlock-like object with
    `.coordinates` returning (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a.coordinates
    bx1, by1, bx2, by2 = b.coordinates
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(
    predicted, ground_truth, iou_threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Greedy 1-1 matching of predicted boxes to ground-truth boxes by IoU.

    Returns a list of (pred_idx, gt_idx) pairs above the IoU threshold.
    A predicted/gt box is matched at most once. Order: highest-IoU first.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            score = iou(p, g)
            if score >= iou_threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)

    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j))
    return matched
