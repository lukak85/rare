"""Figure/caption → article attachment, scored against the annotated reading order.

The Glasbena Mladina annotations carry no article grouping of their own, but
they do carry a per-page reading order in which a Figure, Caption or FigByline
is placed immediately after the piece it illustrates. That gives one local
ground-truth rule, and it is the only one used here:

    a visual belongs to the same article as the last body block before it in
    reading order (skipping the other visuals in its own run).

That block is the visual's *anchor*. Scoring is therefore agnostic about where
articles begin and end — it never has to reconstruct ground-truth articles,
only to ask whether two annotated regions ended up together.

Two numbers come out of it, because the rule on its own can be satisfied by
doing nothing:

* `attachment_accuracy` — is the visual in the same predicted article as its
  anchor? A parser that dropped every boundary and put the whole magazine into
  one article scores 1.0 here.
* `separation` — is the visual kept *out* of the articles it must not be in?
  Ground truth for "must not" comes from Headlines: on a page, everything after
  a Headline is a different piece from everything before it, so a visual and a
  body block on opposite sides of one form a pair that has to end up in two
  different articles. A parser that gave every block its own article scores 1.0
  here, and 0.0 on attachment.

`attachment_score` is their harmonic mean, so only a parser that gets both
right wins. `attachment_recall` is the end-to-end view of the first number:
correct attachments over every visual the annotation places, so a figure the
detector never found counts against the total instead of quietly leaving the
denominator.

A third, narrower metric comes from the same ordering: a Caption or FigByline
that directly follows a Figure belongs to that figure, which is exactly what
`rare.link.figures` decides from geometry — reported as
`caption_figure_accuracy`.

Both tracks the module supports feed the same scorer:

* **oracle** — build documents from the ground-truth boxes and the ground-truth
  order (`build_ground_documents`) and run the linker over them. Detection and
  reading order are perfect, so the numbers are about `rare.link` alone.
* **end-to-end** — score the `*_doc.json` of a real parse, whose items are
  matched back to the annotations by IoU.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from rare.doc.schema import GlasanaDocument
from rare.evaluate._matching import match_by_iou

logger = logging.getLogger(__name__)

# Page chrome: never part of an article, and annotated at the tail of the page
# order rather than in the flow, so it must not be able to anchor a visual.
FURNITURE_LABELS = frozenset({"Header", "Footer", "PageNum", "Abandon"})

# The regions whose article membership this module scores. They are also the
# regions skipped when walking back to an anchor: a caption's neighbour is its
# figure, not the block the pair as a whole hangs off.
VISUAL_LABELS = frozenset({"Figure", "Caption", "FigByline"})

# Labels that open a new piece, used to derive the "must not be together"
# pairs. Only Headline is safe: Subhead and Section also occur inside a piece.
BOUNDARY_LABELS = frozenset({"Headline"})


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundRegion:
    ann_id: int
    label: str
    page_no: int
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2, annotation pixels
    order_id: int
    run: int                                   # index of the Headline-delimited run


@dataclass
class GroundPage:
    image_id: int
    pdf_stem: str
    page_no: int
    width: float
    height: float
    page_type: Optional[str]
    regions: list[GroundRegion]                # every annotation, in reading order


@dataclass(frozen=True)
class GroundAttachment:
    """One annotated visual and what the reading order says it belongs with."""
    ann_id: int
    label: str
    page_no: int
    page_type: Optional[str]
    bbox: tuple[float, float, float, float]
    anchor_ann_id: Optional[int]               # None when nothing precedes it
    anchor_label: Optional[str]
    anchor_page_no: Optional[int]
    anchor_cross_page: bool
    figure_ann_id: Optional[int]               # for Caption/FigByline only
    foreign_ann_ids: tuple[int, ...]           # same page, other side of a Headline


@dataclass
class GroundDoc:
    pdf_stem: str
    pages: dict[int, GroundPage] = field(default_factory=dict)
    attachments: list[GroundAttachment] = field(default_factory=list)


def split_stem_page(file_name: str) -> tuple[str, int]:
    """"<stem>_<page>.jpg" → (stem, page_no); (full stem, 0) when it doesn't fit."""
    name = Path(file_name).name
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1].rsplit(".", 1)[0])
        except ValueError:
            pass
    return Path(name).stem, 0


def _page_regions(anns: list[dict], categories: dict[int, str], page_no: int) -> list[GroundRegion]:
    """A page's annotations as `GroundRegion`s in reading order, runs assigned.

    Annotations without an `order_id` fall to the end in their original order —
    the same convention `rare.evaluate.datasets` uses.
    """
    with_order = [a for a in anns if a.get("order_id") is not None]
    without = [a for a in anns if a.get("order_id") is None]
    ordered = sorted(with_order, key=lambda a: a["order_id"]) + without

    regions: list[GroundRegion] = []
    run = 0
    for position, ann in enumerate(ordered):
        label = categories.get(ann["category_id"], "")
        if label in BOUNDARY_LABELS:
            run += 1
        x, y, w, h = ann["bbox"]
        regions.append(GroundRegion(
            ann_id=ann["id"],
            label=label,
            page_no=page_no,
            bbox=(x, y, x + w, y + h),
            order_id=ann.get("order_id", position),
            run=run,
        ))
    return regions


def _anchor_of(body: list[GroundRegion], index: int) -> Optional[GroundRegion]:
    """The nearest body block before `body[index]` that is not itself a visual."""
    for j in range(index - 1, -1, -1):
        if body[j].label not in VISUAL_LABELS:
            return body[j]
    return None


def _figure_of(body: list[GroundRegion], index: int) -> Optional[GroundRegion]:
    """The Figure a Caption/FigByline at `body[index]` follows, if any.

    Walks back over other captions — a figure may carry both a caption and a
    photo credit — but stops at the first body block, since a caption separated
    from every figure by running text is not attributable from order alone.
    """
    for j in range(index - 1, -1, -1):
        if body[j].label == "Figure":
            return body[j]
        if body[j].label not in VISUAL_LABELS:
            return None
    return None


def load_ground(
    coco_path: str | Path,
    cross_page_anchor: bool = True,
) -> dict[str, GroundDoc]:
    """Derive per-document attachment ground truth from a COCO file with `order_id`.

    Returns `{pdf_stem: GroundDoc}`. When `cross_page_anchor` is true a visual
    that opens a page — 12% of them, mostly a photo at the top of a page whose
    article started on the one before — is anchored to the last body block of
    the previous page, the document-level reading order being the concatenation
    of the page-level ones. With it false, those visuals are reported as
    `anchor_missing` instead and left out of the accuracy denominator.
    """
    raw = json.loads(Path(coco_path).read_text())
    categories = {c["id"]: c["name"] for c in raw["categories"]}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in raw["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    docs: dict[str, GroundDoc] = {}
    for info in raw["images"]:
        stem, page_no = split_stem_page(info["file_name"])
        doc = docs.setdefault(stem, GroundDoc(pdf_stem=stem))
        doc.pages[page_no] = GroundPage(
            image_id=info["id"],
            pdf_stem=stem,
            page_no=page_no,
            width=float(info["width"]),
            height=float(info["height"]),
            page_type=info.get("page_type"),
            regions=_page_regions(anns_by_image.get(info["id"], []), categories, page_no),
        )

    for doc in docs.values():
        _attachments(doc, cross_page_anchor=cross_page_anchor)
    return docs


def _attachments(doc: GroundDoc, cross_page_anchor: bool) -> None:
    """Fill `doc.attachments` from its pages' reading orders."""
    previous_tail: Optional[GroundRegion] = None   # last body block of the previous page

    for page_no in sorted(doc.pages):
        page = doc.pages[page_no]
        body = [r for r in page.regions if r.label not in FURNITURE_LABELS]

        for index, region in enumerate(body):
            if region.label not in VISUAL_LABELS:
                continue

            anchor = _anchor_of(body, index)
            cross_page = False
            if anchor is None and cross_page_anchor:
                anchor = previous_tail
                cross_page = anchor is not None

            figure = _figure_of(body, index) if region.label != "Figure" else None
            foreign = tuple(
                r.ann_id for r in body
                if r.run != region.run and r.label not in VISUAL_LABELS
            )

            doc.attachments.append(GroundAttachment(
                ann_id=region.ann_id,
                label=region.label,
                page_no=page_no,
                page_type=page.page_type,
                bbox=region.bbox,
                anchor_ann_id=anchor.ann_id if anchor else None,
                anchor_label=anchor.label if anchor else None,
                anchor_page_no=anchor.page_no if anchor else None,
                anchor_cross_page=cross_page,
                figure_ann_id=figure.ann_id if figure else None,
                foreign_ann_ids=foreign,
            ))

        tail = [r for r in body if r.label not in VISUAL_LABELS]
        if tail:
            previous_tail = tail[-1]


# ---------------------------------------------------------------------------
# Matching predictions to annotations
# ---------------------------------------------------------------------------

class _Box:
    """Minimal stand-in for `lp.TextBlock` — `match_by_iou` only reads `.coordinates`."""
    __slots__ = ("coordinates",)

    def __init__(self, coordinates: tuple[float, float, float, float]) -> None:
        self.coordinates = coordinates


def match_items(
    doc: GlasanaDocument,
    ground: GroundDoc,
    iou_threshold: float = 0.5,
) -> dict[int, str]:
    """Map each annotation id to the id of the document item that covers it.

    Ground boxes are in annotation-image pixels and document items in rendered
    page pixels, so every page is rescaled by its own width/height ratio before
    matching (the two rasterisations are not always the same aspect ratio to
    the pixel). Matching is greedy 1-1 by IoU, as everywhere else in the eval.
    """
    matches: dict[int, str] = {}

    items_by_page: dict[int, list] = defaultdict(list)
    for item in doc.items.values():
        items_by_page[item.provenance.page_no].append(item)

    for page_no, page in ground.pages.items():
        doc_page = doc.pages.get(page_no)
        items = items_by_page.get(page_no)
        if doc_page is None or not items or not page.regions:
            continue

        sx = doc_page.width / page.width if page.width else 1.0
        sy = doc_page.height / page.height if page.height else 1.0

        predicted = []
        for item in items:
            box = item.provenance.get_bbox()
            predicted.append(_Box((box.x1, box.y1, box.x2, box.y2)))
        truth = [
            _Box((r.bbox[0] * sx, r.bbox[1] * sy, r.bbox[2] * sx, r.bbox[3] * sy))
            for r in page.regions
        ]

        for pred_index, gt_index in match_by_iou(predicted, truth, iou_threshold):
            matches[page.regions[gt_index].ann_id] = items[pred_index].item_id

    return matches


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    """Raw counts; every rate reported is derived from these."""
    visuals: int = 0
    visual_unmatched: int = 0
    anchor_missing: int = 0
    anchor_unmatched: int = 0
    anchor_no_article: int = 0
    correct: int = 0
    wrong: int = 0
    no_article: int = 0
    cross_page_anchors: int = 0
    separation_pairs: int = 0
    separation_correct: int = 0
    caption_figure_scored: int = 0
    caption_figure_correct: int = 0

    def add(self, other: "Tally") -> None:
        for key, value in vars(other).items():
            setattr(self, key, getattr(self, key) + value)

    @property
    def scored(self) -> int:
        """Visuals where both sides were found, so the linker could be judged."""
        return self.correct + self.wrong + self.no_article

    @property
    def answerable(self) -> int:
        """Visuals the annotation actually places — the honest recall denominator.

        A visual with no anchor at all (the first body region of the document,
        or of a page when cross-page anchoring is off) states nothing about any
        article, so it is not something a parser can get wrong.
        """
        return self.visuals - self.anchor_missing

    def rates(self) -> dict[str, float]:
        accuracy = self.correct / self.scored if self.scored else 0.0
        recall = self.correct / self.answerable if self.answerable else 0.0
        separation = (
            self.separation_correct / self.separation_pairs
            if self.separation_pairs else 0.0
        )
        both = accuracy + separation
        out = {
            "attachment_accuracy": accuracy,
            "attachment_recall": recall,
            "separation": separation,
            "attachment_score": (2 * accuracy * separation / both) if both else 0.0,
        }
        if self.caption_figure_scored:
            out["caption_figure_accuracy"] = (
                self.caption_figure_correct / self.caption_figure_scored
            )
        return out

    def as_dict(self) -> dict[str, float]:
        return {
            **vars(self),
            "scored": self.scored,
            "answerable": self.answerable,
            **self.rates(),
        }


def score_document(
    doc: GlasanaDocument,
    ground: GroundDoc,
    iou_threshold: float = 0.5,
) -> tuple[Tally, dict[str, Tally], dict[str, Tally], list[dict]]:
    """Score one document. Returns (totals, by label, by page type, per-visual cases).

    The cases are one row per annotated visual, carrying enough context —
    predicted article titles, the anchor's label and page — to look an error up
    in the rendered HTML without re-running anything.
    """
    matches = match_items(doc, ground, iou_threshold)
    totals = Tally()
    by_label: dict[str, Tally] = defaultdict(Tally)
    by_page_type: dict[str, Tally] = defaultdict(Tally)
    cases: list[dict] = []

    def article_of(ann_id: Optional[int]) -> Optional[str]:
        item_id = matches.get(ann_id) if ann_id is not None else None
        item = doc.items.get(item_id) if item_id else None
        return item.article_id if item else None

    def title_of(article_id: Optional[str]) -> str:
        article = doc.articles.get(article_id) if article_id else None
        return article.title if article else ""

    for attachment in ground.attachments:
        tallies = [totals, by_label[attachment.label], by_page_type[attachment.page_type or "?"]]
        for tally in tallies:
            tally.visuals += 1
            if attachment.anchor_cross_page:
                tally.cross_page_anchors += 1

        visual_item_id = matches.get(attachment.ann_id)
        visual_article = article_of(attachment.ann_id)
        anchor_article = article_of(attachment.anchor_ann_id)

        if visual_item_id is None:
            status = "visual_unmatched"
        elif attachment.anchor_ann_id is None:
            status = "anchor_missing"
        elif matches.get(attachment.anchor_ann_id) is None:
            status = "anchor_unmatched"
        elif anchor_article is None:
            status = "anchor_no_article"
        elif visual_article is None:
            status = "no_article"
        elif visual_article == anchor_article:
            status = "correct"
        else:
            status = "wrong"

        for tally in tallies:
            setattr(tally, status, getattr(tally, status) + 1)

        # Separation: only meaningful once the visual has an article at all.
        pairs = failures = 0
        if visual_article is not None:
            for foreign_id in attachment.foreign_ann_ids:
                foreign_article = article_of(foreign_id)
                if foreign_article is None:
                    continue
                pairs += 1
                failures += int(foreign_article == visual_article)
            for tally in tallies:
                tally.separation_pairs += pairs
                tally.separation_correct += pairs - failures

        # Caption → figure, the geometric link `rare.link.figures` makes.
        caption_figure = None
        if attachment.figure_ann_id is not None and visual_item_id is not None:
            figure_item_id = matches.get(attachment.figure_ann_id)
            if figure_item_id is not None:
                predicted_figure = getattr(doc.items[visual_item_id], "figure_id", None)
                caption_figure = predicted_figure == figure_item_id
                for tally in tallies:
                    tally.caption_figure_scored += 1
                    tally.caption_figure_correct += int(caption_figure)

        cases.append({
            "pdf_stem": ground.pdf_stem,
            "page_no": attachment.page_no,
            "page_type": attachment.page_type,
            "label": attachment.label,
            "status": status,
            "bbox": list(attachment.bbox),
            "anchor_label": attachment.anchor_label,
            "anchor_page_no": attachment.anchor_page_no,
            "anchor_cross_page": attachment.anchor_cross_page,
            "predicted_article": visual_article,
            "predicted_article_title": title_of(visual_article),
            "anchor_article": anchor_article,
            "anchor_article_title": title_of(anchor_article),
            "separation_pairs": pairs,
            "separation_failures": failures,
            "caption_figure_correct": caption_figure,
        })

    return totals, dict(by_label), dict(by_page_type), cases


def evaluate_documents(
    docs: Iterable[GlasanaDocument],
    ground: dict[str, GroundDoc],
    iou_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    """Score every document against its ground truth; returns (summary, cases).

    Documents are paired with ground truth by `source_pdf`; one with no entry
    there is skipped, and a warning names it — silently scoring nothing would
    look like a perfect run.
    """
    totals = Tally()
    by_label: dict[str, Tally] = defaultdict(Tally)
    by_page_type: dict[str, Tally] = defaultdict(Tally)
    by_document: dict[str, Tally] = {}
    all_cases: list[dict] = []

    for doc in docs:
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        doc_ground = ground.get(stem) or ground.get(doc.source_pdf)
        if doc_ground is None:
            logger.warning("no ground truth for document %r; skipped", doc.source_pdf)
            continue

        doc_totals, doc_labels, doc_types, cases = score_document(
            doc, doc_ground, iou_threshold
        )
        totals.add(doc_totals)
        for label, tally in doc_labels.items():
            by_label[label].add(tally)
        for page_type, tally in doc_types.items():
            by_page_type[page_type].add(tally)
        by_document[stem] = doc_totals
        all_cases.extend(cases)

    summary = {
        "overall": totals.as_dict(),
        "by_label": {k: v.as_dict() for k, v in sorted(by_label.items())},
        "by_page_type": {k: v.as_dict() for k, v in sorted(by_page_type.items())},
        "by_document": {k: v.as_dict() for k, v in sorted(by_document.items())},
        "documents": len(by_document),
    }
    return summary, all_cases


# ---------------------------------------------------------------------------
# Oracle documents: ground-truth boxes + ground-truth order, linked
# ---------------------------------------------------------------------------

def build_ground_documents(
    ground: dict[str, GroundDoc],
    pdfs_dir: str | Path | None = None,
    linker: Optional[Callable[[GlasanaDocument], object]] = None,
    stems: Optional[Iterable[str]] = None,
) -> Iterator[GlasanaDocument]:
    """Assemble one document per stem from the annotations, then link it.

    This is `rare.parse.coco.parse_coco` with everything the metric does not
    need taken out: no page rendering, no figure crops, no HTML/Markdown. Text
    still comes from `<pdfs_dir>/<stem>.pdf` when it is there, because the
    splitting and continuation passes read it; without it the geometric passes
    still run and the numbers are a floor rather than a fair reading.
    """
    import pdfplumber

    from rare.doc.schema import PageInfo
    from rare.parse.assemble import assemble_page
    from rare.parse.text import extract_text_for_page

    pdfs_dir = Path(pdfs_dir) if pdfs_dir else None
    wanted = set(stems) if stems is not None else None

    for stem in sorted(ground):
        if wanted is not None and stem not in wanted:
            continue
        doc_ground = ground[stem]
        doc = GlasanaDocument(source_pdf=stem)
        current_article = None

        pdf_path = pdfs_dir / f"{stem}.pdf" if pdfs_dir else None
        if pdf_path is not None and not pdf_path.exists():
            logger.warning(
                "no PDF at %s; %s is assembled without text, so the linking passes "
                "that read it contribute nothing", pdf_path, stem,
            )
        pdf = pdfplumber.open(pdf_path) if pdf_path and pdf_path.exists() else None
        try:
            for page_no in sorted(doc_ground.pages):
                page = doc_ground.pages[page_no]
                doc.pages[page_no] = PageInfo(
                    page_no=page_no,
                    width=page.width,
                    height=page.height,
                    source_file=f"{stem}_{page_no}.jpg",
                )
                regions = [
                    {
                        "region_id": str(uuid.uuid4()),
                        "label": r.label,
                        "bbox_norm_1000": [
                            r.bbox[0] / page.width * 1000.0,
                            r.bbox[1] / page.height * 1000.0,
                            r.bbox[2] / page.width * 1000.0,
                            r.bbox[3] / page.height * 1000.0,
                        ],
                        "score": 1.0,
                    }
                    for r in page.regions
                ]
                texts = (
                    extract_text_for_page(pdf, page_no, regions, page.width, page.height)
                    if pdf is not None and page_no < len(pdf.pages)
                    else {}
                )
                current_article = assemble_page(
                    doc,
                    page_no=page_no,
                    regions=regions,
                    texts=texts,
                    img_w=page.width,
                    img_h=page.height,
                    figures_dir=Path("."),      # unused: no page image is passed
                    current_article=current_article,
                    page_image=None,
                )
        finally:
            if pdf is not None:
                pdf.close()

        if linker is not None:
            linker(doc)
        yield doc


def load_documents(docs_dir: str | Path) -> Iterator[GlasanaDocument]:
    """Every `*_doc.json` under `docs_dir`, as parsed documents.

    Hidden directories are skipped and a stem is taken once: output trees keep
    superseded runs beside the current one (`.old/`, dated snapshots), and
    scoring an archived copy of a document a second time would silently double
    its weight in the aggregate.
    """
    seen: set[str] = set()
    for path in sorted(Path(docs_dir).rglob("*_doc.json")):
        if any(part.startswith(".") for part in path.parts):
            continue
        doc = GlasanaDocument.model_validate_json(path.read_text())
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        if stem in seen:
            logger.warning("document %r already scored; skipping %s", stem, path)
            continue
        seen.add(stem)
        yield doc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_figure_link(
    coco_path: str | Path,
    run_dir: str | Path,
    docs_dir: str | Path | None = None,
    pdfs_dir: str | Path | None = None,
    linker: Optional[Callable[[GlasanaDocument], object]] = None,
    limit: Optional[int] = None,
    iou_threshold: float = 0.5,
    cross_page_anchor: bool = True,
    dataset_name: str = "",
) -> dict:
    """Score figure/caption → article attachment and write the results.

    With `docs_dir` the already-parsed documents under it are scored (end to
    end: detection, order and linking all count). Without it, documents are
    built from the ground-truth layout and order and linked with `linker`,
    isolating `rare.link`.

    Writes `attachment_summary.json`, `attachment_cases.jsonl` (one row per
    annotated visual) and the shared `report.md`/`scores.csv` into `run_dir`.
    """
    from rare.evaluate.report import write_report

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ground = load_ground(coco_path, cross_page_anchor=cross_page_anchor)

    if docs_dir is not None:
        docs: Iterable[GlasanaDocument] = load_documents(docs_dir)
        source = str(docs_dir)
    else:
        stems = sorted(ground)[:limit] if limit else None
        docs = build_ground_documents(ground, pdfs_dir=pdfs_dir, linker=linker, stems=stems)
        source = "ground-truth layout"

    if limit and docs_dir is not None:
        docs = list(docs)[:limit]

    summary, cases = evaluate_documents(docs, ground, iou_threshold=iou_threshold)
    summary["source"] = source
    summary["iou_threshold"] = iou_threshold
    summary["cross_page_anchor"] = cross_page_anchor

    (run_dir / "attachment_summary.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / "attachment_cases.jsonl", "w") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")

    write_report(
        run_dir,
        track="figure-link",
        dataset_name=dataset_name or Path(coco_path).stem,
        aggregates={source: summary["overall"]},
        per_image_rows=cases,
    )
    return summary