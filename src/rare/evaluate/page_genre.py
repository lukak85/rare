"""Page type (annotated) vs article genre (predicted) — a first, deliberately blunt check.

Every annotated image carries a `page_type` ("NewsPage", "InterviewPage", ...)
and `rare.link.classify` gives every article a `genre` ("novice", "intervju",
...). The two vocabularies describe different things, so this module compares
them through one editable table, `PAGE_TYPE_TO_GENRE`, and reports where they
disagree.

The comparison is approximate by construction, and knowing how is the point:

* **a page can hold pieces of several genres** — a news page whose last column
  is a record review is not a mistake by either side. Scoring is therefore
  reported twice: `accuracy_dominant` takes the article holding most of the
  page and `accuracy_any` asks only whether *some* article on the page has the
  expected genre. The truth for a mixed page is between them.
* **some page types are not genres at all.** "SpecialPage" says how the page is
  laid out, "BackPage" where it sits in the issue; neither implies what the
  piece on it is. They map to None below and are left out of the totals rather
  than counted as failures. So are pages with no `page_type` in the annotation.
* **one page type can legitimately run two genres.** A value may therefore be a
  list, and any of the listed genres counts as correct.

The table is the part meant to change. Override it from a JSON file of
`{page_type: genre | [genres] | null}` (null = do not score this page type)
without touching the code:

    rare evaluate --track page-genre --dataset glasbena_mladina \
        --docs-dir outputs/parsed --page-type-map my_map.json

The confusion matrix in the summary — page type against the genre actually
predicted — is what to read when deciding how the table should change.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from rare.doc.schema import ContentLayer, GlasanaDocument
from rare.evaluate.figure_link import load_documents, split_stem_page

logger = logging.getLogger(__name__)

# Genre expected of an article on a page of each annotated type. A list means
# any of them is right; None means the page type says nothing about genre and
# is left out of the score entirely.
PAGE_TYPE_TO_GENRE: dict[str, Union[str, list[str], None]] = {
    "ArticlePage":   "članek",
    "NewsPage":      "novice",
    "InterviewPage": "intervju",
    "RecordsPage":   "recenzija",
    "LettersPage":   "pisma",
    "QuizPage":      "kviz",
    "EventsPage":    "dogodki",
    "ImagesPage":    "slike",
    "TOCPage":       "kazalo",
    "AdvertPage":    "reklama",
    "CoverPage":     "naslovnica",

    # Not scored — a layout or a position in the issue, not an editorial genre.
    "SpecialPage":   None,   # "laid out differently", says nothing about genre
    "BackPage":      None,   # the last page; in practice ads, a quiz or a photo
}


def resolve_map(path: str | Path | None) -> dict[str, Union[str, list[str], None]]:
    """`PAGE_TYPE_TO_GENRE` with a JSON file's entries merged on top."""
    if path is None:
        return dict(PAGE_TYPE_TO_GENRE)
    override = json.loads(Path(path).read_text())
    return {**PAGE_TYPE_TO_GENRE, **override}


def expected_genres(
    page_type: Optional[str],
    mapping: dict[str, Union[str, list[str], None]],
) -> Optional[set[str]]:
    """The genres acceptable on this page type, or None when it is not scored."""
    if page_type is None or page_type not in mapping:
        return None
    value = mapping[page_type]
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


def load_page_types(coco_path: str | Path) -> dict[str, dict[int, Optional[str]]]:
    """`{pdf_stem: {page_no: page_type}}` from a COCO file's `images`."""
    raw = json.loads(Path(coco_path).read_text())
    out: dict[str, dict[int, Optional[str]]] = defaultdict(dict)
    for info in raw["images"]:
        stem, page_no = split_stem_page(info["file_name"])
        out[stem][page_no] = info.get("page_type")
    return dict(out)


@dataclass
class Tally:
    pages: int = 0                    # pages of a scored type
    pages_unclassified: int = 0       # ... with no article carrying a genre
    dominant_correct: int = 0
    any_correct: int = 0
    articles: int = 0                 # (page, article) pairs with a genre
    articles_correct: int = 0
    pages_skipped: int = 0            # page type absent, unmapped or mapped to None
    confusion: Counter = field(default_factory=Counter)   # (page_type, genre) → n

    def add(self, other: "Tally") -> None:
        for key, value in vars(other).items():
            if key == "confusion":
                self.confusion.update(value)
            else:
                setattr(self, key, getattr(self, key) + value)

    def rates(self) -> dict[str, float]:
        return {
            "accuracy_dominant": self.dominant_correct / self.pages if self.pages else 0.0,
            "accuracy_any": self.any_correct / self.pages if self.pages else 0.0,
            "article_accuracy": (
                self.articles_correct / self.articles if self.articles else 0.0
            ),
            "genre_coverage": (
                (self.pages - self.pages_unclassified) / self.pages if self.pages else 0.0
            ),
        }

    def as_dict(self) -> dict:
        out = {k: v for k, v in vars(self).items() if k != "confusion"}
        out.update(self.rates())
        return out


def page_articles(doc: GlasanaDocument, page_no: int) -> list[tuple[str, int]]:
    """`(article_id, body items on this page)`, most of the page first.

    Furniture is not counted: a running header and a page number say nothing
    about which piece owns the page.
    """
    counts: Counter = Counter()
    for item in doc.items.values():
        if item.provenance.page_no != page_no or not item.article_id:
            continue
        if item.content_layer == ContentLayer.FURNITURE:
            continue
        counts[item.article_id] += 1
    return counts.most_common()


def score_document(
    doc: GlasanaDocument,
    page_types: dict[int, Optional[str]],
    mapping: dict[str, Union[str, list[str], None]],
) -> tuple[Tally, list[dict]]:
    """Compare each page's annotated type with the genres predicted on it."""
    tally = Tally()
    rows: list[dict] = []

    for page_no in sorted(page_types):
        page_type = page_types[page_no]
        wanted = expected_genres(page_type, mapping)
        if wanted is None:
            tally.pages_skipped += 1
            continue

        ranked = page_articles(doc, page_no)
        genres = [
            (doc.articles[article_id].genre, count)
            for article_id, count in ranked
            if article_id in doc.articles and doc.articles[article_id].genre
        ]

        dominant = genres[0][0] if genres else None
        matched = {genre for genre, _ in genres if genre in wanted}

        tally.pages += 1
        tally.confusion[(page_type, dominant or "∅")] += 1
        if not genres:
            tally.pages_unclassified += 1
        if dominant in wanted:
            tally.dominant_correct += 1
        if matched:
            tally.any_correct += 1
        for genre, _ in genres:
            tally.articles += 1
            tally.articles_correct += int(genre in wanted)

        rows.append({
            "pdf_stem": doc.source_pdf,
            "page_no": page_no,
            "page_type": page_type,
            "expected": sorted(wanted),
            "dominant_genre": dominant,
            "page_genres": [genre for genre, _ in genres],
            "dominant_correct": dominant in wanted,
            "any_correct": bool(matched),
            "articles_on_page": len(ranked),
        })

    return tally, rows


def evaluate_documents(
    docs: Iterable[GlasanaDocument],
    page_types: dict[str, dict[int, Optional[str]]],
    mapping: Optional[dict[str, Union[str, list[str], None]]] = None,
) -> tuple[dict, list[dict]]:
    """Score every document against the page types of its stem."""
    mapping = mapping or dict(PAGE_TYPE_TO_GENRE)
    totals = Tally()
    by_page_type: dict[str, Tally] = defaultdict(Tally)
    by_document: dict[str, Tally] = {}
    all_rows: list[dict] = []

    for doc in docs:
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        pages = page_types.get(stem) or page_types.get(doc.source_pdf)
        if pages is None:
            logger.warning("no annotated page types for %r; skipped", doc.source_pdf)
            continue

        doc_tally, rows = score_document(doc, pages, mapping)
        totals.add(doc_tally)
        by_document[stem] = doc_tally
        all_rows.extend(rows)

        for row in rows:
            single = Tally()
            single.pages = 1
            single.pages_unclassified = int(not row["page_genres"])
            single.dominant_correct = int(row["dominant_correct"])
            single.any_correct = int(row["any_correct"])
            single.articles = len(row["page_genres"])
            single.articles_correct = sum(
                1 for genre in row["page_genres"] if genre in set(row["expected"])
            )
            by_page_type[row["page_type"]].add(single)

    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for (page_type, genre), count in sorted(totals.confusion.items()):
        confusion[page_type][genre] = count

    summary = {
        "overall": totals.as_dict(),
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "by_page_type": {k: v.as_dict() for k, v in sorted(by_page_type.items())},
        "by_document": {k: v.as_dict() for k, v in sorted(by_document.items())},
        "scored_page_types": sorted(
            k for k in mapping if expected_genres(k, mapping) is not None
        ),
        "ignored_page_types": sorted(k for k in mapping if mapping[k] is None),
        "documents": len(by_document),
    }
    return summary, all_rows


def run_page_genre(
    coco_path: str | Path,
    run_dir: str | Path,
    docs_dir: str | Path | None = None,
    pdfs_dir: str | Path | None = None,
    linker=None,
    page_type_map: str | Path | None = None,
    limit: Optional[int] = None,
    dataset_name: str = "",
) -> dict:
    """Score genres against page types and write the results into `run_dir`.

    With `docs_dir`, the parsed documents under it are scored. Without it,
    documents are rebuilt from the ground-truth layout and reading order and
    linked with `linker` — with no classification backend that still yields the
    genres `rare.link.classify` reads off the running headers, which is the
    cheap way to try a change to `PAGE_TYPE_TO_GENRE`.

    Writes `page_genre_summary.json` (totals, confusion matrix, per page type
    and per document) and `page_genre_pages.jsonl` (one row per scored page),
    plus the shared `report.md` / `scores.csv`.
    """
    from rare.evaluate.figure_link import build_ground_documents, load_ground
    from rare.evaluate.report import write_report

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    mapping = resolve_map(page_type_map)
    page_types = load_page_types(coco_path)

    if docs_dir is not None:
        docs = list(load_documents(docs_dir))
        source = str(docs_dir)
    else:
        ground = load_ground(coco_path)
        stems = sorted(ground)[:limit] if limit else None
        docs = list(build_ground_documents(
            ground, pdfs_dir=pdfs_dir, linker=linker, stems=stems
        ))
        source = "ground-truth layout"
    if limit:
        docs = docs[:limit]

    summary, rows = evaluate_documents(docs, page_types, mapping)
    summary["source"] = source

    (run_dir / "page_genre_summary.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / "page_genre_pages.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_report(
        run_dir,
        track="page-genre",
        dataset_name=dataset_name or Path(coco_path).stem,
        aggregates={source: summary["overall"]},
        per_image_rows=rows,
    )
    return summary