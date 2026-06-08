"""VLM-track metrics: text F1 + Levenshtein-like edit distance."""

from __future__ import annotations

from difflib import SequenceMatcher

from rare.utils.evalutils import f1_score, normalize_answer


def _edit_ratio(a: str, b: str) -> float:
    """SequenceMatcher.ratio() on normalised text — a string-similarity score
    in [0, 1] where 1.0 is identical. Avoids an extra Levenshtein dependency.
    """
    return SequenceMatcher(None, normalize_answer(a), normalize_answer(b)).ratio()


METRICS = {"f1", "edit_ratio"}


def score_text(predicted_md: str, gold_md: str) -> dict[str, float]:
    return {
        "f1":         float(f1_score(predicted_md, gold_md)),
        "edit_ratio": _edit_ratio(predicted_md, gold_md),
    }


def aggregate(per_doc_scores: list[dict]) -> dict[str, float]:
    if not per_doc_scores:
        return {}
    out: dict[str, float] = {}
    for k in METRICS:
        vals = [s[k] for s in per_doc_scores if k in s]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
