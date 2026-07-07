"""Pipeline-track metrics: layout mAP + reading-order Kendall tau."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rare.evaluate._matching import match_by_iou
from rare.utils.evalutils import (
    mean_average_precision,
    kendall_tau,
    normalized_edit_distance,
)
from rare.evaluate.omnidocbench import _resolve_map

if TYPE_CHECKING:
    import layoutparser as lp


LAYOUT_METRICS = {"map", "map_50", "map_75", "map_cat", "map_50_cat", "map_75_cat"}
ORDER_METRICS = {"kendall_tau", "edit_distance", "matched_pairs"}
# Reading order scored on ground-truth boxes directly (detection quality removed).
ORDER_GT_METRICS = {"edit_distance_gt", "kendall_tau_gt", "order_regions_gt"}
METRICS = LAYOUT_METRICS | ORDER_METRICS | ORDER_GT_METRICS

# Region categories that carry no position in the linear reading flow and are
# dropped before scoring reading order. A region is excluded when *either* its
# raw label (as listed below, e.g. GlasbenaMladina COCO names) *or* its
# OmniDocBench `category_type` (after translation through the category map) is
# in this set — so you may list raw labels for fine-grained control, or a whole
# category_type (e.g. "abandon", "figure") to drop a group at once.
DEFAULT_ORDER_EXCLUDE = frozenset({
    "Header", "Footer", "PageNum", "Footnote", "MarginNote", "Abandon", "Advertisement",
    "Figure", "Caption", "Form", "FigByL", "FigByline"
})


def score_layout(
    predicted: "lp.Layout",
    ground: "lp.Layout",
    pred_category_map: Optional[dict[str, str]] = None,
    gt_category_map: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Compute mAP between predicted and ground layouts, twice:

    - `map` / `map_50` / `map_75`: **class-agnostic** — every box collapses to a
      single label. Pure localization quality, independent of taxonomy. This is
      the honest cross-dataset headline: a detector trained on DocLayNet can be
      compared to one trained on your data without either being penalised for
      naming a region differently.
    - `map_cat` / `map_50_cat` / `map_75_cat`: **category-aware** — both sides
      are translated into a shared label space (OmniDocBench `category_type`)
      via `pred_category_map` (model vocabulary) and `gt_category_map` (your
      source vocabulary), then scored per class. Tells you how well the model
      serves *your* schema. Only emitted when at least one map is supplied;
      otherwise it would just duplicate the same-vocabulary case.
    """
    if len(predicted) == 0 or len(ground) == 0:
        zero = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}
        if pred_category_map or gt_category_map:
            zero.update({"map_cat": 0.0, "map_50_cat": 0.0, "map_75_cat": 0.0})
        return zero

    agnostic = mean_average_precision(predicted, ground, class_agnostic=True)
    out = {
        "map":    float(agnostic["map"].item()),
        "map_50": float(agnostic["map_50"].item()),
        "map_75": float(agnostic["map_75"].item()),
    }

    if pred_category_map or gt_category_map:
        mapped = mean_average_precision(
            predicted, ground,
            pred_category_map=pred_category_map,
            gt_category_map=gt_category_map,
            class_metrics = True
        )
        out.update({
            "map_cat":    float(mapped["map"].item()),
            "map_50_cat": float(mapped["map_50"].item()),
            "map_75_cat": float(mapped["map_75"].item()),
        })

    return out


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
        "edit_distance": normalized_edit_distance(pred_ranks, ground_ranks), # TODO - implementation of normalized Levenshtein distance
        "matched_pairs": float(len(matched)),
    }


def score_order_gt(
    ground: "lp.Layout",
    ground_order: list[int],
    predicted_order: list[int],
    gt_category_map: Optional[dict[str, str]] = None,
    exclude_categories: Optional[set[str]] = None,
) -> dict[str, float]:
    """Reading-order quality scored on the *ground-truth* boxes directly.

    Unlike `score_order`, this feeds the ground-truth boxes (not the detector's
    output) to the reading-order model, so no IoU matching is needed and the
    result is not confounded by detection quality — it isolates the ordering
    model. Both `ground_order` and `predicted_order` are permutations over the
    same `ground` boxes (`ground[order[k]]` is the k-th region), so the metric
    is the normalized edit distance between the two index sequences.

    Regions that don't belong to the linear reading flow (page numbers, figures,
    figure captions, headers/footers, adverts, forms, …) are dropped from *both*
    sequences before scoring. A region is excluded when *either* its raw label
    *or* its OmniDocBench `category_type` (via `gt_category_map`, defaulting to
    the OmniDocBench map) is in `exclude_categories` — so raw GlasbenaMladina
    names like "PageNum"/"Figure" and whole category_types like "abandon"/
    "figure" both work. Defaults to `DEFAULT_ORDER_EXCLUDE`.

    Returns:
        edit_distance_gt: normalized Levenshtein distance in [0, 1]; 0 is a
            perfect ordering.
        kendall_tau_gt: Kendall's tau on the surviving regions' ranks (for
            parity with `score_order`).
        order_regions_gt: number of regions scored after exclusion.
    """
    exclude = DEFAULT_ORDER_EXCLUDE if exclude_categories is None else exclude_categories
    cmap = _resolve_map(gt_category_map)
    excluded = {
        i for i, block in enumerate(ground)
        if block.type in exclude or cmap.get(block.type, block.type) in exclude
    }
    gt_seq = [i for i in ground_order if i not in excluded]
    pred_seq = [i for i in predicted_order if i not in excluded]

    if len(gt_seq) < 2:
        return {
            "edit_distance_gt": 0.0,
            "kendall_tau_gt":   1.0,
            "order_regions_gt": float(len(gt_seq)),
            "ground_order": gt_seq,
            "predicted_order_gt": pred_seq,
        }

    gt_rank = {gi: r for r, gi in enumerate(gt_seq)}
    pred_rank = {pi: r for r, pi in enumerate(pred_seq)}
    common = [i for i in gt_seq if i in pred_rank]
    gt_ranks = [gt_rank[i] for i in common]
    pred_ranks = [pred_rank[i] for i in common]

    return {
        "edit_distance_gt": normalized_edit_distance(pred_seq, gt_seq),
        "kendall_tau_gt":   kendall_tau(pred_ranks, gt_ranks),
        "order_regions_gt": float(len(gt_seq)),
        "ground_order": gt_seq,
        "predicted_order_gt": pred_seq,
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
