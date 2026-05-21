"""Pipeline-track metrics: layout mAP + reading-order Kendall tau."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rare.evaluate._matching import match_by_iou
from rare.utils.evalutils import mean_average_precision, kendall_tau

if TYPE_CHECKING:
    import layoutparser as lp


LAYOUT_METRICS = {"map", "map_50", "map_75"}
ORDER_METRICS = {"kendall_tau", "matched_pairs"}
METRICS = LAYOUT_METRICS | ORDER_METRICS


def score_layout(predicted: "lp.Layout", ground: "lp.Layout") -> dict[str, float]:
    """Compute mAP / mAP@50 / mAP@75 between predicted and ground layouts.

    All predictions are scored as a single (class-agnostic) label, matching
    the existing `mean_average_precision` helper.
    """
    if len(predicted) == 0 or len(ground) == 0:
        return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

    result = mean_average_precision(predicted, ground)
    return {
        "map":    float(result["map"].item()),
        "map_50": float(result["map_50"].item()),
        "map_75": float(result["map_75"].item()),
    }


def score_order(
    predicted: "lp.Layout",
    predicted_order: list[int],
    ground: "lp.Layout",
    ground_order: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Kendall tau between predicted and ground reading order, after matching
    predicted boxes to ground boxes by IoU.

    `predicted_order` and `ground_order` are permutations over `predicted` and
    `ground` respectively (i.e. `predicted[predicted_order[k]]` is the k-th
    region in reading order).
    """
    matched = match_by_iou(predicted, ground, iou_threshold=iou_threshold)
    if len(matched) < 2:
        return {"kendall_tau": 0.0, "matched_pairs": float(len(matched))}

    pred_rank = {pi: r for r, pi in enumerate(predicted_order)}
    ground_rank = {gi: r for r, gi in enumerate(ground_order)}

    pred_ranks = [pred_rank.get(pi, 0) for pi, _ in matched]
    ground_ranks = [ground_rank.get(gi, 0) for _, gi in matched]

    return {
        "kendall_tau":   kendall_tau(pred_ranks, ground_ranks), # TODO - check
        "edit_distance": None, # TODO - implementation of normalized Levenshtein distance
        "matched_pairs": float(len(matched)),
    }


def aggregate(
    per_image_scores: list[dict],
) -> dict[str, float]:
    """Mean of each known metric across images."""
    if not per_image_scores:
        return {}
    out: dict[str, float] = {}
    for k in METRICS:
        vals = [s[k] for s in per_image_scores if k in s]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
