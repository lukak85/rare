from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from rare.doc.schema import (
    CaptionItem,
    FigBylineItem,
    GlasanaDocument,
)
from rare.evaluate.ground import load_documents, split_stem_page
from rare.link import entities, figure_link, figure_matching
from rare.link._geometry import box_of
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

# The regions this module scores. Taken from the pass under test rather than
# restated, so the two can never drift apart.
VISUAL_TYPES = figure_link.VISUAL_TYPES

# Every visual ends up under exactly one of these. The `skipped_` ones are
# reported but kept out of the accuracy — the linker was never given a fair
# question in those cases, and scoring them either way would be a fiction.
OUTCOMES = (
    "correct",                    # came back to the article it started in
    "wrong_article",              # placed in a different one
    "no_article",                 # the linker declined to place it at all
    "skipped_no_anchor",          # nothing precedes it in reading order
    "skipped_anchor_no_article",  # the block it follows is in no article
    "skipped_gt_article_empty",   # its article is visuals-only: unreachable
)

# The config variants scored side by side. `full` is whatever the caller
# configured; the rest isolate one signal each, and `nearest` is the baseline
# every heuristic here has to beat to justify itself.
VARIANTS: dict[str, dict] = {
    "full": {},
    "geometry": {"figure_link_ner_weight": 0.0},
    "ner": {"figure_link_geometry_weight": 0.0},
    "nearest": {
        "figure_link_geometry_weight": 1.0,
        "figure_link_ner_weight": 0.0,
        "figure_link_below_penalty": 1.0,
        "figure_link_side_penalty": 1.0,
    },
    # Distance to an article's *mean* item rather than its nearest. The default
    # (nearest) lets an article that wraps around the page beat the short piece
    # directly above a photo, because one of its many columns is always a few
    # pixels away; the mean asks which article actually surrounds the figure.
    "mean": {"figure_link_use_mean": True},
}


# ---------------------------------------------------------------------------
# Ground truth: the article a visual interrupts
# ---------------------------------------------------------------------------

def ground_articles(doc: GlasanaDocument) -> dict[str, tuple[bool, Optional[str]]]:
    """`{item_id: (had_anchor, article_id)}` for every visual, from reading order.

    Walks `body_order` — which is already the annotated order, page by page,
    with furniture excluded — and remembers the last non-visual block. That
    block is the visual's *anchor*, and its article is the answer.

    `had_anchor` separates the two ways the answer can be missing: nothing
    precedes the visual at all (a photo opening the issue), or the block it
    follows is itself in no article. Both are unscoreable, for different
    reasons, and the summary says which.

    Deliberately not `visual.article_id`: a document parsed with linking on has
    already had `link_figures` move its visuals, so the value stored on a
    figure is the heuristic's own answer rather than the ground truth. The
    anchor rule recovers the reading-order base from either kind of document.
    """
    truth: dict[str, tuple[bool, Optional[str]]] = {}
    anchor: Optional[object] = None

    for item in doc.iter_body():
        if isinstance(item, VISUAL_TYPES):
            truth[item.item_id] = (
                anchor is not None,
                anchor.article_id if anchor is not None else None,
            )
        else:
            anchor = item
    return truth


# ---------------------------------------------------------------------------
# Hold the visuals out
# ---------------------------------------------------------------------------

def hold_out_visuals(doc: GlasanaDocument) -> set[str]:
    """Undo every placement decision about visuals. Returns the ids held out.

    Their article, their caption pairing and the links recording both are
    removed; everything else — the article partition this is scored against —
    is left untouched. `articles.rebuild` is deliberately *not* called: it
    deletes articles left with no items, which is exactly the ground truth an
    all-visual article carries.
    """
    held: set[str] = set()

    for item in doc.items.values():
        if not isinstance(item, VISUAL_TYPES):
            continue
        held.add(item.item_id)
        item.article_id = None
        if isinstance(item, (CaptionItem, FigBylineItem)):
            item.figure_id = None

    for article in doc.articles.values():
        article.item_ids = [i for i in article.item_ids if i not in held]

    doc.links = [
        link
        for link in doc.links
        if not (
            link.kind in ("caption-of", "figure-to-article")
            and (link.from_id in held or link.to_id in held)
        )
    ]
    return held


def relink(doc: GlasanaDocument, config: LinkConfig) -> dict[str, str]:
    """Re-run the two passes that place visuals. Returns `{item_id: method}`.

    The same order `rare.link.link_document` uses, with everything that would
    reshape the articles left out: the partition stays at ground truth, so a
    miss is attributable to these two passes and nothing else. The entity index
    is built after the hold-out, so a caption's own names never count towards
    the article it used to sit in.
    """
    figure_matching.link_captions(doc, config)
    index = entities.EntityIndex(doc, config)
    figure_link.link_figures(doc, index, config)

    # `link_figures` records its reasoning against the group's leader — the
    # figure, or a stray caption standing alone. Spread it back over the group
    # the same way the pass formed it, so every visual carries the method that
    # decided its article.
    method_of_leader = {
        link.from_id: link.method
        for link in doc.links
        if link.kind == "figure-to-article"
    }

    methods: dict[str, str] = {}
    for item in doc.items.values():
        if not isinstance(item, VISUAL_TYPES):
            continue
        leader = item.item_id
        if isinstance(item, (CaptionItem, FigBylineItem)) and item.figure_id:
            leader = item.figure_id
        method = method_of_leader.get(leader)
        if method:
            methods[item.item_id] = method
    return methods


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def summarise(outcomes: Counter) -> dict:
    """Outcome counts plus the rates derived from them."""
    scored = outcomes["correct"] + outcomes["wrong_article"] + outcomes["no_article"]
    skipped = sum(v for k, v in outcomes.items() if k.startswith("skipped"))
    placed = outcomes["correct"] + outcomes["wrong_article"]
    return {
        **{name: outcomes.get(name, 0) for name in OUTCOMES},
        "visuals": scored + skipped,
        "scored": scored,
        "skipped": skipped,
        "accuracy": outcomes["correct"] / scored if scored else 0.0,
        # How much of the corpus the accuracy above actually speaks for.
        "coverage": scored / (scored + skipped) if (scored + skipped) else 0.0,
        # How often the linker committed to an answer at all.
        "assignment_rate": placed / scored if scored else 0.0,
    }


def score_document(
    doc: GlasanaDocument,
    config: LinkConfig,
    page_types: Optional[dict[tuple[str, int], str]] = None,
) -> tuple[Counter, dict[str, Counter], dict[str, Counter], list[dict]]:
    """Hold the visuals out of one document, put them back, and score the result.
    """
    stem = Path(doc.source_pdf).stem or doc.source_pdf

    truth = ground_articles(doc)
    # An article the hold-out empties has no items left to be found by.
    non_visual_members = {
        item.article_id
        for item in doc.items.values()
        if item.article_id and not isinstance(item, VISUAL_TYPES)
    }

    hold_out_visuals(doc)
    methods = relink(doc, config)

    outcomes: Counter = Counter()
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_method: dict[str, Counter] = defaultdict(Counter)
    cases: list[dict] = []

    def title_of(article_id: Optional[str]) -> str:
        article = doc.articles.get(article_id) if article_id else None
        return article.title if article else ""

    for item_id, (had_anchor, expected) in truth.items():
        item = doc.items[item_id]
        actual = item.article_id

        if not had_anchor:
            outcome = "skipped_no_anchor"
        elif expected is None:
            outcome = "skipped_anchor_no_article"
        elif expected not in non_visual_members:
            outcome = "skipped_gt_article_empty"
        elif actual is None:
            outcome = "no_article"
        elif actual == expected:
            outcome = "correct"
        else:
            outcome = "wrong_article"

        method = methods.get(item_id, "none")
        label = item.category.value
        outcomes[outcome] += 1
        by_label[label][outcome] += 1
        if not outcome.startswith("skipped"):
            by_method[method][outcome] += 1

        page_no = item.provenance.page_no
        cases.append({
            "pdf_stem": stem,
            "page_no": page_no,
            "page_type": (page_types or {}).get((stem, page_no)),
            "label": label,
            "outcome": outcome,
            "method": method,
            "put_in": title_of(actual),
            "should_be": title_of(expected),
            "bbox": list(box_of(item)),
        })

    return outcomes, by_label, by_method, cases


def evaluate_documents(
    docs: Iterable[GlasanaDocument],
    config: LinkConfig,
    page_types: Optional[dict[tuple[str, int], str]] = None,
) -> tuple[dict, list[dict]]:
    """Score every document under one config; returns (summary, cases)."""
    totals: Counter = Counter()
    label_totals: dict[str, Counter] = defaultdict(Counter)
    method_totals: dict[str, Counter] = defaultdict(Counter)
    page_type_totals: dict[str, Counter] = defaultdict(Counter)
    by_document: dict[str, Counter] = {}
    all_cases: list[dict] = []

    for doc in docs:
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        outcomes, by_label, by_method, cases = score_document(
            doc, config, page_types
        )
        totals.update(outcomes)
        by_document[stem] = outcomes
        for label, counter in by_label.items():
            label_totals[label].update(counter)
        for method, counter in by_method.items():
            method_totals[method].update(counter)
        for case in cases:
            page_type_totals[case["page_type"] or "unknown"][case["outcome"]] += 1
        all_cases.extend(cases)

    overall = summarise(totals)

    summary = {
        "overall": overall,
        "by_label": {k: summarise(v) for k, v in sorted(label_totals.items())},
        "by_method": {k: summarise(v) for k, v in sorted(method_totals.items())},
        "by_page_type": {k: summarise(v) for k, v in sorted(page_type_totals.items())},
        "by_document": {k: summarise(v) for k, v in sorted(by_document.items())},
        "documents": len(by_document),
    }
    return summary, all_cases


# ---------------------------------------------------------------------------
# Page types, for the breakdown
# ---------------------------------------------------------------------------

def load_page_types(coco_path: str | Path | None) -> dict[tuple[str, int], str]:
    """`{(stem, page_no): page_type}` from a COCO file's `images` list.

    Only the annotated page attribute is wanted here, so the (large)
    `annotations` array is never touched beyond being parsed.
    """
    if coco_path is None:
        return {}
    raw = json.loads(Path(coco_path).read_text())
    types: dict[tuple[str, int], str] = {}
    for image in raw.get("images", []):
        page_type = image.get("page_type")
        if page_type:
            types[split_stem_page(image["file_name"])] = page_type
    return types


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_figure_link(
    run_dir: str | Path,
    docs_dir: str | Path,
    coco_path: str | Path | None = None,
    config: dict | LinkConfig | None = None,
    variants: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    dataset_name: str = "",
) -> dict:
    """Score figure → article placement for every variant, and write the results.

    Each variant gets its own copy of every document, because scoring mutates
    it. Writes `attachment_summary.json`, `attachment_cases.jsonl` (one row per
    visual, from the `full` variant) and the shared `report.md`/`scores.csv`.
    """
    from rare.evaluate.report import write_report

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    base = config if isinstance(config, LinkConfig) else LinkConfig.from_dict(config)
    names = list(variants) if variants else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(
            f"unknown variant(s) {unknown}; choose from {sorted(VARIANTS)}"
        )

    page_types = load_page_types(coco_path)
    loaded = list(load_documents(docs_dir))
    if limit:
        loaded = loaded[:limit]
    if not loaded:
        raise ValueError(f"no *_doc.json documents found under {docs_dir}")

    summaries: dict[str, dict] = {}
    cases: list[dict] = []

    for name in names:
        variant_config = replace(base, **VARIANTS[name])
        # Scoring mutates a document, so each variant works on its own copy.
        copies = (doc.model_copy(deep=True) for doc in loaded)
        summary, variant_cases = evaluate_documents(copies, variant_config, page_types)
        summaries[name] = summary
        if name == names[0]:
            cases = variant_cases
        logger.info(
            "variant %s: accuracy %.4f over %d visuals",
            name, summary["overall"]["accuracy"], summary["overall"]["visuals"],
        )

    headline = summaries[names[0]]
    result = {
        **headline,
        "source": str(docs_dir),
        "variants": {name: summaries[name]["overall"] for name in names},
        "variant_detail": summaries,
    }

    (run_dir / "attachment_summary.json").write_text(json.dumps(result, indent=2))
    with open(run_dir / "attachment_cases.jsonl", "w") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    write_report(
        run_dir,
        track="figure-link",
        dataset_name=dataset_name or Path(docs_dir).name,
        aggregates={name: summaries[name]["overall"] for name in names},
        per_image_rows=cases,
    )
    return result
