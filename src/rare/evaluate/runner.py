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
from rare.evaluate.omnidocbench import (
    coco_to_omnidocbench,
    emit_stub_markdown,
    merge_prediction_pages,
    relabel_predictions_to_gt,
)
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
    emit_omnidocbench: bool = True,
    category_map: Optional[dict[str, str]] = None,
    pdfs_dir: Optional[Path] = None,
) -> dict:
    """Run one (layout, order) combo over `dataset`, write per-model results.

    When `emit_omnidocbench` is true, also writes:
      - `<run_dir>/omnidocbench/gt.json` — one OmniDocBench page list.
      - `<run_dir>/omnidocbench/<model>_pred.json` — same shape, predictions.
      - `<run_dir>/omnidocbench/markdown_pred_<model>/<image_stem>.md` — one
        markdown file per page in predicted reading order; this is what
        `scripts/omnidocbench/run.sh` mounts at `data_md/predictions`.

    The `text` field per layout_det depends on `pdfs_dir`:
      - If `pdfs_dir` resolves and a PDF for the page exists, real text is
        extracted via `pdfplumber` and used (the "OCR-equivalent" path —
        OmniDocBench's `reading_order` and `text_block` Edit_dist become
        meaningful). Empty extracts (figures, ads) stay empty so quick_match
        ignores them.
      - Otherwise we fall back to stub tokens (`__B<anno_id>__` on GT;
        IoU-matched GT tokens or `__UNMATCHED_<id>__` on predictions). This
        still unblocks OmniDocBench but only measures box ordering.

    `category_map` is an optional override merged on top of
    `omnidocbench.DEFAULT_CATEGORY_MAP`.
    """
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
            "model":         model_name,
            "image_id":      sample.image_id,
            "pdf_stem":      sample.pdf_stem,
            "page_no":       sample.page_no,
            "file_name":     sample.image_path.name,
            "predicted_order": list(predicted_order),
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
            coco_predictions.append(layout_parser_to_coco(
                predicted, image_info, categories,
                predicted_order=predicted_order,
            ))

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

    if emit_omnidocbench:
        from rare.evaluate.pdf_text import PdfTextSource

        odb_dir = run_dir / "omnidocbench"
        odb_dir.mkdir(parents=True, exist_ok=True)

        # Pick the text source: real PDF text when a PDF directory resolves,
        # stub tokens otherwise. The PDF source needs cleanup of file handles,
        # so own it for the duration of the emit.
        pdf_text_source = None
        pdf_root = _resolve_pdfs_dir(pdfs_dir, dataset)
        if pdf_root is not None:
            pdf_text_source = PdfTextSource(pdf_root)
        use_stub = pdf_text_source is None

        try:
            # Ground truth: convert the source COCO file (whole dataset, not
            # just the scored slice) once per run. Idempotent — overwriting
            # is fine since the GT does not depend on the model.
            gt_pages: list[dict] = []
            if dataset.coco_path is not None:
                gt_doc = json.loads(Path(dataset.coco_path).read_text())
                gt_pages = coco_to_omnidocbench(
                    gt_doc, category_map,
                    text_stub=use_stub, text_source=pdf_text_source,
                )
                (odb_dir / "gt.json").write_text(json.dumps(gt_pages, indent=2))
            # Predictions: one combined JSON per model. With real PDF text
            # each pred box's own crop is queried; the IoU-to-GT relabel is
            # only needed in stub mode (where tokens must match exactly).
            if coco_predictions:
                pred_pages = merge_prediction_pages(
                    coco_predictions, category_map,
                    text_stub=use_stub, text_source=pdf_text_source,
                )
                if use_stub and gt_pages:
                    relabel_predictions_to_gt(pred_pages, gt_pages)
                (odb_dir / f"{model_name}_pred.json").write_text(
                    json.dumps(pred_pages, indent=2)
                )
                # Per-page markdown for OmniDocBench's `data_md/predictions` mount.
                emit_stub_markdown(pred_pages, odb_dir / f"markdown_pred_{model_name}")
        finally:
            if pdf_text_source is not None:
                pdf_text_source.close()

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


def _resolve_pdfs_dir(
    explicit: Optional[Path], dataset: EvalDataset
) -> Optional[Path]:
    """Pick a directory of PDFs for the OmniDocBench text-extraction path.

    Order: explicit `--pdfs-dir`, then `<dataset coco parent>/pdfs`,
    `<dataset coco parent>/PDF` (DocLayNet convention), then the legacy
    fallbacks shared with the VLM track. Returns None when nothing resolves;
    the caller then drops back to stub-text mode.
    """
    candidates: list[Optional[Path]] = [explicit]
    if dataset.coco_path is not None:
        parent = Path(dataset.coco_path).parent
        candidates += [parent / "pdfs", parent / "PDF"]
    candidates += [Path("dataset/pdfs"), Path("data/glasbena_mladina/pdfs")]
    for root in candidates:
        if root is None:
            continue
        if root.exists() and root.is_dir():
            return root
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
        # Only per-model result dicts contribute to the report. Skip anything
        # else (e.g. a stale list-shaped export accidentally dropped here).
        if not isinstance(data, dict) or data.get("track") != track:
            continue
        aggregates[data["model"]] = data.get("aggregates", {})
        per_image.extend(data.get("per_image", []))
    write_report(run_dir, track, dataset_name, aggregates, per_image)
