"""Evaluation runner for both pipeline and VLM tracks.

Design decision: one layout backend (or one VLM) per invocation. Mixing
multiple layoutparser backends inside one Python process is fragile because
`LAYOUTPARSER_BACKEND` only takes effect at `import layoutparser` time.

To compare N models, run `rare evaluate` N times against the same `--run-id`.
Each run drops a `per_model/<name>.json` file into the run directory; the
report is regenerated from whatever per-model files are present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from rare.evaluate.datasets import EvalDataset
from rare.evaluate.pipeline_eval import (
    aggregate as pipeline_aggregate,
    score_layout,
    score_order,
)
from rare.evaluate.report import write_report
#from rare.evaluate.vlm_eval import aggregate as vlm_aggregate, score_text
from rare.doc.renderers import to_markdown
from rare.utils.conversionutils import layout_parser_to_coco
from rare.utils.fileutils import save_coco_to_json


def _open_image(path):
    from PIL import Image
    return Image.open(path)


# ---------------------------------------------------------------------------
# Pipeline track
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset: EvalDataset,
    layout,
    order,
    run_dir: Path,
    limit: Optional[int] = None,
    save_coco: bool = True,
) -> dict:
    """Run one (layout, order) combo over `dataset`, write per-model results."""
    model_name = f"{layout.name}__{order.name}"
    per_image: list[dict] = []
    coco_predictions: list[dict] = []

    samples = list(dataset.iter_samples())
    if limit:
        samples = samples[:limit]

    for sample in samples:
        image = _open_image(sample.image_path)
        predicted = layout.detect(image)
        predicted_order = order.order(
            predicted,
            image=image,
            page_no=sample.page_no,
            pdf_stem=sample.pdf_stem,
        )

        row: dict = {
            "model":     model_name,
            "image_id":  sample.image_id,
            "pdf_stem":  sample.pdf_stem,
            "page_no":   sample.page_no,
            "file_name": sample.image_path.name,
        }
        row.update(score_layout(predicted, sample.ground_layout))
        if sample.ground_order is not None:
            row.update(score_order(
                predicted, predicted_order, sample.ground_layout, sample.ground_order
            ))
        per_image.append(row)

        if save_coco:
            categories = {1: {"id": 1, "name": "Region"}}
            image_info = {
                "id":        sample.image_id,
                "file_name": sample.image_path.name,
                "width":     sample.width,
                "height":    sample.height,
            }
            coco_predictions.append(layout_parser_to_coco(predicted, image_info, categories))

    aggregates = pipeline_aggregate(per_image)
    _write_per_model(
        run_dir=run_dir,
        model_name=model_name,
        track="pipeline",
        dataset_name=dataset.name,
        aggregates=aggregates,
        per_image=per_image,
    )
    if save_coco and coco_predictions:
        coco_dir = run_dir / "per_model" / f"{model_name}_coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        for sample, payload in zip(samples, coco_predictions):
            save_coco_to_json(payload, str(coco_dir / f"{sample.image_id}.json"))
    _regenerate_report(run_dir, track="pipeline", dataset_name=dataset.name)
    return aggregates


# ---------------------------------------------------------------------------
# VLM track
# ---------------------------------------------------------------------------

def run_vlm(
    dataset: EvalDataset,
    vlm,
    run_dir: Path,
    pdfs_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    """Run one VLM over the dataset's PDFs, scoring against gold markdown."""
    model_name = vlm.name
    per_doc: list[dict] = []

    # Resolve PDFs: use `pdfs_dir` if given, else assume `<dataset_root>/pdfs/`.
    # If neither yields a match for a stem, skip that document.
    pdfs_root = pdfs_dir
    pdf_stems = list(dataset.gold_markdown.keys())
    if limit:
        pdf_stems = pdf_stems[:limit]

    for pdf_stem in pdf_stems:
        gold_md = dataset.gold_markdown.get(pdf_stem)
        if gold_md is None:
            continue
        pdf_path = _resolve_pdf(pdfs_root, pdf_stem)
        if pdf_path is None:
            print(f"[warn] no PDF found for {pdf_stem}, skipping")
            continue

        doc = vlm.parse_pdf(pdf_path)
        predicted_md = to_markdown(doc)
        row = {
            "model":    model_name,
            "pdf_stem": pdf_stem,
        }
        row.update(score_text(predicted_md, gold_md))
        per_doc.append(row)

        out = run_dir / "per_model" / f"{model_name}__{pdf_stem}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(predicted_md)

    aggregates = vlm_aggregate(per_doc)
    _write_per_model(
        run_dir=run_dir,
        model_name=model_name,
        track="vlm",
        dataset_name=dataset.name,
        aggregates=aggregates,
        per_image=per_doc,
    )
    _regenerate_report(run_dir, track="vlm", dataset_name=dataset.name)
    return aggregates


def _resolve_pdf(pdfs_root: Optional[Path], stem: str) -> Optional[Path]:
    for root in [pdfs_root, Path("dataset/pdfs"), Path("data/glasbena_mladina/pdfs")]:
        if root is None:
            continue
        cand = root / f"{stem}.pdf"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Per-model dump + report regeneration
# ---------------------------------------------------------------------------

def _write_per_model(
    run_dir: Path,
    model_name: str,
    track: str,
    dataset_name: str,
    aggregates: dict[str, float],
    per_image: list[dict],
) -> None:
    per_model_dir = run_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    (per_model_dir / f"{model_name}.json").write_text(json.dumps({
        "model":      model_name,
        "track":      track,
        "dataset":    dataset_name,
        "aggregates": aggregates,
        "per_image":  per_image,
    }, indent=2))


def _regenerate_report(run_dir: Path, track: str, dataset_name: str) -> None:
    aggregates: dict[str, dict[str, float]] = {}
    per_image: list[dict] = []
    per_model_dir = run_dir / "per_model"
    if not per_model_dir.exists():
        return
    for fp in sorted(per_model_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if data.get("track") != track:
            continue
        aggregates[data["model"]] = data.get("aggregates", {})
        per_image.extend(data.get("per_image", []))
    write_report(run_dir, track, dataset_name, aggregates, per_image)
