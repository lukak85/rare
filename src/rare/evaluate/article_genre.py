"""Article genre (predicted) vs article genre (ground truth), on frozen articles.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional, Union

from rare.evaluate.ground import load_documents, split_stem_page
from rare.link.classify import compose_article_text, predict_genre
from rare.link.config import LinkConfig

logger = logging.getLogger(__name__)

DEFAULT_GT_DIR = Path("outputs/parsed/gt/articles_fixed")

# Genre of an article on a page of each annotated type. A list means the page
# type allows several (the article is then left for a manual label); None means
# the page type says nothing about genre. Override from a JSON file of
# `{page_type: genre | [genres] | null}` with `--page-type-map`.
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

    # A layout or a position in the issue, not an editorial genre.
    "SpecialPage":   None,   # "laid out differently", says nothing about genre
    "BackPage":      None,   # the last page; no text, just image
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
    """The genres a page type allows, or None when it names none."""
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


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def genre_from_page_types(
    page_types: dict[int, Optional[str]],
    mapping: dict[str, Union[str, list[str], None]],
) -> tuple[Optional[str], Optional[str]]:
    """`(genre, None)` when the pages agree on one, else `(None, reason)`."""
    genres: set[str] = set()
    for page_type in page_types.values():
        wanted = expected_genres(page_type, mapping)
        if wanted is None:
            continue
        if len(wanted) > 1:
            return None, "ambiguous page type"
        genres |= wanted

    if not genres:
        return None, "no page type names a genre"
    if len(genres) > 1:
        return None, "page types disagree"
    return next(iter(genres)), None


def build_ground_truth(
    docs_dir: str | Path,
    coco_path: str | Path,
    out_dir: str | Path = DEFAULT_GT_DIR,
    page_type_map: str | Path | None = None,
    force: bool = False,
) -> Counter:
    """Write `{stem}_articles.json` with genres into `out_dir`. Returns tallies.

    An existing file is left alone unless `force`; with it, the page-type
    genres are recomputed but every article labelled "manual" keeps its label.
    Articles left without a genre are listed in `out_dir/needs_genre.txt`.
    """
    from rare.doc.articles_json import to_articles

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_map(page_type_map)
    page_types = load_page_types(coco_path)
    tally: Counter = Counter()

    for doc in load_documents(docs_dir):
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        path = out_dir / f"{stem}_articles.json"
        if path.exists() and not force:
            tally["files kept"] += 1
            continue
        if stem not in page_types:
            logger.warning("no annotated page types for %r; skipped", stem)
            continue

        manual = {}
        if path.exists():
            manual = {
                a["article_id"]: a["genre"]
                for a in json.loads(path.read_text())["articles"]
                if a.get("article_id") and a.get("genre_source") == "manual"
            }

        payload = to_articles(doc)
        for article in payload["articles"]:
            types = {p: page_types[stem].get(p) for p in article["page_nos"]}
            article["page_types"] = types
            if not article["article_id"]:
                article.update(genre=None, genre_source=None)
                continue

            tally["articles"] += 1
            if article["article_id"] in manual:
                article.update(genre=manual[article["article_id"]], genre_source="manual")
                tally["manual"] += 1
                continue

            genre, reason = genre_from_page_types(types, mapping)
            article.update(genre=genre, genre_source="page_type" if genre else None)
            tally["page_type" if genre else f"null: {reason}"] += 1

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tally["files written"] += 1

    tally["needs manual genre"] = write_needs_genre(out_dir, mapping)
    return tally


def load_ground_truth(gt_dir: str | Path) -> Iterator[tuple[str, dict]]:
    """`(stem, articles payload)` for every `*_articles.json` in `gt_dir`."""
    for path in sorted(Path(gt_dir).glob("*_articles.json")):
        yield path.name[: -len("_articles.json")], json.loads(path.read_text())


NEEDS_GENRE_FILE = "needs_genre.txt"
_TEXT_PREVIEW = 400


def _preview(article: dict) -> str:
    text = " ".join(
        (item.get("text") or "").strip() for item in article["items"] if item.get("text")
    )
    text = " ".join(text.split())
    return text[:_TEXT_PREVIEW] + ("…" if len(text) > _TEXT_PREVIEW else "")


def write_needs_genre(
    gt_dir: str | Path, mapping: dict[str, Union[str, list[str], None]]
) -> int:
    """List every labelled-null article in `gt_dir/needs_genre.txt`. Returns the count.

    One block per article, ending in an empty `genre:` line to fill in;
    `apply_manual_genres` reads the filled file back.
    """
    blocks: list[str] = []
    for stem, payload in load_ground_truth(gt_dir):
        for article in payload["articles"]:
            if not article.get("article_id") or article.get("genre"):
                continue
            _, reason = genre_from_page_types(article.get("page_types") or {}, mapping)
            blocks.append("\n".join([
                f"file: {stem}_articles.json",
                f"article_id: {article['article_id']}",
                f"pages: {', '.join(str(p) for p in article['page_nos'])}",
                "page_types: " + ", ".join(
                    f"{p}={t}" for p, t in (article.get("page_types") or {}).items()
                ),
                f"reason: {reason}",
                f"section: {' '.join((article.get('section') or '').split())}",
                f"title: {' '.join((article.get('title') or '').split())}",
                f"text: {_preview(article)}",
                "genre: ",
            ]))

    header = (
        f"# Articles with no genre from their page types ({len(blocks)}).\n"
        f"# Fill in each `genre:` line with one of: {', '.join(known_genres(mapping))}\n"
        "# then: python scripts/classification/build_article_genre_gt.py --apply-manual\n"
    )
    (Path(gt_dir) / NEEDS_GENRE_FILE).write_text(header + "\n" + "\n\n".join(blocks) + "\n")
    return len(blocks)


def known_genres(mapping: dict[str, Union[str, list[str], None]]) -> list[str]:
    """Every genre the page-type map can produce, sorted."""
    return sorted({
        genre
        for value in mapping.values() if value
        for genre in ([value] if isinstance(value, str) else value)
    })


def apply_manual_genres(
    out_dir: str | Path = DEFAULT_GT_DIR,
    txt_path: str | Path | None = None,
    page_type_map: str | Path | None = None,
) -> Counter:
    """Write the `genre:` lines filled in `needs_genre.txt` into the JSON files.

    A genre outside the map's vocabulary is refused (and logged), so a typo
    cannot become a class of its own in the scores.
    """
    known = set(known_genres(resolve_map(page_type_map)))
    out_dir = Path(out_dir)
    txt_path = Path(txt_path) if txt_path else out_dir / NEEDS_GENRE_FILE
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    for block in txt_path.read_text().split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        if not (fields.get("genre") and fields.get("file") and fields.get("article_id")):
            continue
        if fields["genre"] not in known:
            logger.warning(
                "%s %s: %r is not a known genre; skipped",
                fields["file"], fields["article_id"], fields["genre"],
            )
            continue
        labels[fields["file"]][fields["article_id"]] = fields["genre"]

    tally: Counter = Counter()
    for file_name, by_id in labels.items():
        path = out_dir / file_name
        payload = json.loads(path.read_text())
        for article in payload["articles"]:
            if article.get("article_id") in by_id:
                article.update(genre=by_id.pop(article["article_id"]), genre_source="manual")
                tally["applied"] += 1
        tally["unknown article_id"] += len(by_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return tally


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _per_genre(confusion: Counter) -> dict[str, dict]:
    """Precision / recall / F1 per true genre, from `(true, predicted)` counts."""
    genres = sorted({true for true, _ in confusion})
    out = {}
    for genre in genres:
        tp = confusion[(genre, genre)]
        support = sum(n for (true, _), n in confusion.items() if true == genre)
        predicted = sum(n for (_, pred), n in confusion.items() if pred == genre)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        out[genre] = {
            "support": support, "precision": precision, "recall": recall,
            "f1": _f1(precision, recall),
        }
    return out


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _averages(per_genre: dict[str, dict], correct: int, predicted: int, scored: int) -> dict:
    n = len(per_genre)
    macro = {
        f"macro_{key}": sum(g[key] for g in per_genre.values()) / n if n else 0.0
        for key in ("precision", "recall", "f1")
    }
    micro_precision = correct / predicted if predicted else 0.0
    micro_recall = correct / scored if scored else 0.0
    return {
        **macro,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall),
    }


ARTICLE_IDS_FILE = "article_ids.json"


def select_articles(
    labelled: list[tuple[str, dict]],
    sample: Optional[int] = None,
    seed: int = 0,
    ids_path: str | Path | None = None,
) -> list[tuple[str, dict]]:
    """The `(stem, article)` pairs to score, in their order in `labelled`.

    An existing `ids_path` decides: exactly the articles it lists. IDs it names
    that are no longer in the ground truth — a re-parse gives articles new ids —
    are warned about rather than silently dropped from the comparison. Without
    one, `sample` draws that many at random with `seed`, and writes the draw to
    `ids_path` when given, so the next model is scored on the same articles.
    """
    if ids_path is not None and Path(ids_path).exists():
        wanted = [a["article_id"] for a in json.loads(Path(ids_path).read_text())["articles"]]
        by_id = {article["article_id"]: (stem, article) for stem, article in labelled}
        missing = [aid for aid in wanted if aid not in by_id]
        if missing:
            logger.warning(
                "%d of the %d articles in %s are not labelled in the ground truth "
                "(re-parsed, or unlabelled since); scoring the other %d",
                len(missing), len(wanted), ids_path, len(wanted) - len(missing),
            )
        keep = {aid for aid in wanted if aid in by_id}
        return [pair for pair in labelled if pair[1]["article_id"] in keep]

    if sample is None or sample >= len(labelled):
        chosen = list(labelled)
    else:
        picked = set(random.Random(seed).sample(range(len(labelled)), sample))
        chosen = [pair for n, pair in enumerate(labelled) if n in picked]
    if ids_path is not None:
        write_article_ids(ids_path, chosen, None, sample, seed)
    return chosen


def write_article_ids(
    path: str | Path,
    chosen: list[tuple[str, dict]],
    gt_dir: str | Path | None,
    sample: Optional[int],
    seed: int,
) -> None:
    """Record which articles were scored, in the form `select_articles` reads back."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "source": str(gt_dir) if gt_dir is not None else None,
        "sample": sample,
        "seed": seed,
        "articles": [
            {"pdf_stem": stem, "article_id": article["article_id"], "true_genre": article["genre"]}
            for stem, article in chosen
        ],
    }, ensure_ascii=False, indent=2))


def run_article_genre(
    gt_dir: str | Path,
    run_dir: str | Path,
    classifier=None,
    config: dict | LinkConfig | None = None,
    limit: Optional[int] = None,
    dataset_name: str = "",
    sample: Optional[int] = None,
    seed: int = 0,
    ids_path: str | Path | None = None,
) -> dict:
    """Classify the labelled articles in `gt_dir` and score them. Returns the summary.

    Which articles: those listed in `ids_path` when that file exists — so every
    model is scored on the same articles and their macro and micro F1 compare —
    else `sample` of them drawn at random with `seed` (written to `ids_path`
    when one is given), else all. The articles scored are always written to
    `article_ids.json` in the run directory too, which `ids_path` accepts.

    Writes `article_genre_summary.json` (totals, per genre, confusion matrix),
    `article_genre_articles.jsonl` (one row per scored article, with the
    backend's raw reply), plus the shared `report.md` / `scores.csv`.
    """
    from rare.evaluate.report import write_report

    cfg = config if isinstance(config, LinkConfig) else LinkConfig.from_dict(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    confusion: Counter = Counter()
    sources: Counter = Counter()
    unlabelled = documents = 0

    labelled: list[tuple[str, dict]] = []
    for stem, payload in load_ground_truth(gt_dir):
        if limit and documents >= limit:
            break
        documents += 1
        for article in payload["articles"]:
            if not article.get("article_id"):
                continue
            if article.get("genre"):
                labelled.append((stem, article))
            else:
                unlabelled += 1

    chosen = select_articles(labelled, sample, seed, ids_path)
    write_article_ids(run_dir / ARTICLE_IDS_FILE, chosen, gt_dir, sample, seed)

    for stem, article in chosen:
        truth = article["genre"]
        text = compose_article_text(
            article.get("section"),
            [item.get("text", "") for item in article["items"]],
            cfg.classify_max_chars,
            cfg.classify_include_section,
        )
        prediction = predict_genre(
            text if classifier is not None else "",
            article.get("section"),
            classifier,
            cfg,
            article["article_id"],
        )
        confusion[(truth, prediction.label or "∅")] += 1
        sources[prediction.source or "none"] += 1
        rows.append({
            "pdf_stem": stem,
            "article_id": article["article_id"],
            "title": article.get("title", ""),
            "page_nos": article["page_nos"],
            "genre_source": article.get("genre_source"),
            "true_genre": truth,
            "predicted_genre": prediction.label,
            "prediction_source": prediction.source,
            "correct": prediction.label == truth,
            "reply": prediction.reply,
        })

    scored = len(rows)
    predicted = sum(1 for row in rows if row["predicted_genre"])
    correct = sum(1 for row in rows if row["correct"])
    per_genre = _per_genre(confusion)
    overall = {
        "articles": scored,
        "predicted": predicted,
        "correct": correct,
        "unlabelled": unlabelled,
        "accuracy": correct / scored if scored else 0.0,
        "accuracy_on_predicted": correct / predicted if predicted else 0.0,
        "coverage": predicted / scored if scored else 0.0,
        **_averages(per_genre, correct, predicted, scored),
    }

    matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (truth, pred), count in sorted(confusion.items()):
        matrix[truth][pred] = count
    model = getattr(classifier, "name", None) or "section headers only"
    summary = {
        "model": model,
        "source": str(gt_dir),
        "documents": documents,
        "sample": {"size": sample, "seed": seed, "ids": str(ids_path) if ids_path else None},
        "overall": overall,
        "prediction_sources": dict(sources),
        "per_genre": per_genre,
        "confusion": {k: dict(v) for k, v in sorted(matrix.items())},
    }

    (run_dir / "article_genre_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    with open(run_dir / "article_genre_articles.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_report(
        run_dir,
        track="article-genre",
        dataset_name=dataset_name,
        aggregates={model: overall},
        per_image_rows=rows,
    )
    return summary
