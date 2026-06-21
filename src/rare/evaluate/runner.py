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
from rare.evaluate.vlm_eval import aggregate as vlm_aggregate, score_text
from rare.evaluate.omnidocbench import (
    coco_to_omnidocbench,
    emit_stub_markdown,
    merge_prediction_pages,
    relabel_predictions_to_gt,
    _resolve_map,
)
from rare.doc.renderers import to_markdown, to_markdown_pages
from rare.utils.conversionutils import layout_parser_to_coco
from rare.utils.fileutils import save_coco_to_json


def _open_image(path):
    from PIL import Image
    return Image.open(path)


def _write_omnidocbench_gt(
    dataset: EvalDataset,
    odb_dir: Path,
    category_map: Optional[dict[str, str]],
    text_stub: bool,
    text_source,
) -> tuple[Optional[Path], list[dict]]:
    """Build `<odb_dir>/gt.json` and return `(gt_path, gt_pages)`. Shared by the
    pipeline and VLM tracks so both score against an identically-built ground
    truth. The GT does not depend on the model, so writing is idempotent across
    runs.

    Two sources, in priority order:
      1. `dataset.omnidocbench_path` — a native OmniDocBench GT JSON (already in
         the target shape, carrying real `text`/`latex`/`html`). Copied through
         verbatim; the COCO conversion and `text_source` are not used.
      2. `dataset.coco_path` — a COCO file converted to OmniDocBench shape.

    Returns `(None, [])` when the dataset exposes neither.
    """
    odb_dir.mkdir(parents=True, exist_ok=True)
    if dataset.omnidocbench_path is not None:
        gt_pages = json.loads(Path(dataset.omnidocbench_path).read_text())
        gt_path = odb_dir / "gt.json"
        gt_path.write_text(json.dumps(gt_pages, indent=2))
        return gt_path, gt_pages
    if dataset.coco_path is None:
        return None, []
    gt_doc = json.loads(Path(dataset.coco_path).read_text())
    gt_pages = coco_to_omnidocbench(
        gt_doc, category_map, text_stub=text_stub, text_source=text_source,
    )
    gt_path = odb_dir / "gt.json"
    gt_path.write_text(json.dumps(gt_pages, indent=2))
    return gt_path, gt_pages


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
    run_omnidocbench: bool = False,
    omnidocbench_image: Optional[str] = None,
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

    # Category-aware mAP needs both taxonomies translated into the shared
    # OmniDocBench space. GT (source COCO names) is resolved via the same map the
    # OmniDocBench export uses. Predictions use the backend's own map; when the
    # backend declares none, its predictions already share the GT vocabulary
    # (e.g. DocLayout-YOLO's "Glasana" label_map), so we reuse the GT map.
    gt_category_map = _resolve_map(category_map)
    pred_category_map = getattr(layout, "pred_category_map", None) or gt_category_map

    samples = list(dataset.iter_samples())
    if limit:
        samples = samples[:limit]

    for sample in samples:
        image = _open_image(sample.image_path)
        predicted = layout.detect(sample.image_path)
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
        row.update(score_layout(
            predicted, sample.ground_layout,
            pred_category_map=pred_category_map,
            gt_category_map=gt_category_map,
        ))
        if sample.ground_order is not None:
            row.update(score_order(
                predicted, predicted_order, sample.ground_layout, sample.ground_order
            ))
        per_image.append(row)

        if save_coco:
            categories = layout.label_map
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

        gt_path: Optional[Path] = None
        markdown_dir: Optional[Path] = None
        try:
            if False: # TODO: add back
                gt_path, gt_pages = _write_omnidocbench_gt(
                    dataset, odb_dir, category_map,
                    text_stub=use_stub, text_source=pdf_text_source,
                )
            # Predictions: one combined JSON per model. With real PDF text
            # each pred box's own crop is queried; the IoU-to-GT relabel is
            # only needed in stub mode (where tokens must match exactly).
            if coco_predictions:
                pred_pages = merge_prediction_pages(
                    coco_predictions, category_map,
                    text_stub=use_stub, text_source=pdf_text_source,
                )
                # if use_stub and gt_pages:
                #     relabel_predictions_to_gt(pred_pages, gt_pages)
                (odb_dir / f"{model_name}_pred.json").write_text(
                    json.dumps(pred_pages, indent=2)
                )
                # Per-page markdown for OmniDocBench's `data_md/predictions` mount.
                markdown_dir = odb_dir / f"markdown_pred_{model_name}"
                # emit_stub_markdown(pred_pages, markdown_dir) # TODO: currently out of scope for pipeline track
        finally:
            if pdf_text_source is not None:
                pdf_text_source.close()

        # Approach C: run OmniDocBench's pinned container against the artifacts
        # we just emitted, and fold the Edit-distance numbers into `aggregates`
        # so they appear as columns in report.md. Per-model result dir keeps
        # accumulated models from clobbering each other's `predictions_*` files.
        if run_omnidocbench and gt_path is not None and markdown_dir is not None:
            from rare.evaluate.omnidocbench_docker import run_eval, DEFAULT_LAYOUT_IMAGE

            odb_metrics = run_eval(
                gt_path=gt_path,
                pred_md_dir=markdown_dir,
                result_dir=odb_dir / f"results_{model_name}",
                image=omnidocbench_image or DEFAULT_LAYOUT_IMAGE,
                type='detection',
                gt_cat_mapping=_convert_to_string(gt_category_map),
                pred_cat_mapping=_convert_to_string(pred_category_map)
            )
            aggregates.update(odb_metrics)

    _write_per_model(
        run_dir=run_dir,
        model_name=model_name,
        track="pipeline",
        dataset_name=dataset.name,
        aggregates=aggregates,
        per_image=per_image,
    )
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
    images_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    run_omnidocbench: bool = False,
    omnidocbench_image: Optional[str] = None,
    category_map: Optional[dict[str, str]] = None,
) -> dict:
    """Run one VLM over the dataset's PDFs, scoring against gold markdown.

    When `run_omnidocbench` is set, also emit OmniDocBench artifacts and run the
    pinned container (see `run_pipeline` for the same flow on the pipeline
    track). The VLM emits *real* text, so the ground truth must be built with
    real PDF text too — stub-token GT would never match and every page would
    score the max Edit distance. We therefore require a resolvable PDF directory
    and skip the container (with a warning) when none is found.
    """
    model_name = vlm.name
    per_doc: list[dict] = []
    parsed_docs: list[tuple[str, object]] = []  # (pdf_stem, GlasanaDocument)

    # Specialized parsers (e.g. MinerU) emit their own markdown per block; we
    # then score that verbatim rather than re-applying label-derived markup.
    # Chat VLMs leave this False and go through the structured renderer.
    raw_markdown = bool(getattr(vlm, "raw_markdown", False))

    # Resolve PDFs: use `pdfs_dir` if given, else assume `<dataset_root>/pdfs/`.
    # If neither yields a match for a stem, skip that document.
    pdfs_root = pdfs_dir
    pdf_stems = list(dataset.ground_markdown.keys())
    if limit:
        pdf_stems = pdf_stems[:limit]

    for pdf_stem in pdf_stems:
        ground_md = dataset.ground_markdown.get(pdf_stem)
        if ground_md is None:
            continue
        pdf_path = _resolve_pdf(pdfs_root, pdf_stem)
        if pdf_path is None:
            print(f"[warn] no PDF found for {pdf_stem}, skipping")
            continue

        doc = vlm.parse_pdf(pdf_path)
        parsed_docs.append((pdf_stem, doc))
        predicted_md = to_markdown(doc, raw=raw_markdown)
        row = {
            "model":    model_name,
            "pdf_stem": pdf_stem,
        }
        row.update(score_text(predicted_md, ground_md))
        per_doc.append(row)

        out = run_dir / "per_model" / f"{model_name}__{pdf_stem}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(predicted_md)

    aggregates = vlm_aggregate(per_doc)

    #if run_omnidocbench and parsed_docs:
    if run_omnidocbench:
        out_md_dir = f"outputs/rare/omnidocbench/{model_name}"
        parsed_docs = vlm.to_markdown(
            image_dir=images_dir,
            pdf_dir=pdfs_dir,
            out_md_dir=out_md_dir,
            skip_existing=True
        )
        aggregates.update(_run_vlm_omnidocbench(
            dataset, run_dir, model_name, parsed_docs,
            pdfs_dir=pdfs_dir,
            out_md_dir=out_md_dir,
            category_map=category_map,
            omnidocbench_image=omnidocbench_image,
            raw_markdown=raw_markdown,
        ))

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

def _convert_to_string(mapping: dict[str, str]):
    """
    Takes in a dictionary of class mappings and outputs a YAML-ready string.

    Args:
        mapping: dictionary of class mappings

    Returns:
        string: YAML-ready string
    """
    yaml_str = ""
    for key, value in mapping.items():
        yaml_str += f"""
        {key}: {value}\n
        """
    return yaml_str


def _run_vlm_omnidocbench(
    dataset: EvalDataset,
    run_dir: Path,
    model_name: str,
    parsed_docs: list[tuple[str, object]],
    pdfs_dir: Optional[Path],
    out_md_dir: Path,
    category_map: Optional[dict[str, str]],
    omnidocbench_image: Optional[str],
    raw_markdown: bool = False,
) -> dict[str, float]:
    """Emit OmniDocBench artifacts for parsed VLM docs and run the container.

    Returns the parsed Edit-distance metrics (empty on any skip/failure). The
    per-page prediction files are named `<pdf_stem>_<page>.md`, matching the
    GT page stems (`<pdf_stem>_<page>.jpg`) produced from the dataset COCO.
    """
    from rare.evaluate.omnidocbench_docker import run_eval, DEFAULT_IMAGE
    from rare.evaluate.pdf_text import PdfTextSource

    odb_dir = run_dir / "omnidocbench"
    odb_dir.mkdir(parents=True, exist_ok=True)

    if dataset.omnidocbench_path is not None:
        # Native OmniDocBench GT already carries real text/latex/html, so no PDF
        # extraction is needed — pass the file through verbatim.
        gt_path, _ = _write_omnidocbench_gt(
            dataset, odb_dir, category_map, text_stub=False, text_source=None,
        )
    else:
        # Real-text GT is mandatory here (see run_vlm docstring). Bail loudly
        # rather than silently producing all-1.0 scores against stub tokens.
        pdf_root = _resolve_pdfs_dir(pdfs_dir, dataset)
        if pdf_root is None:
            print("[omnidocbench] VLM track needs a resolvable --pdfs-dir to build "
                  "real-text ground truth; skipping container eval.")
            return {}

        pdf_text_source = PdfTextSource(pdf_root)
        try:
            gt_path, _ = _write_omnidocbench_gt(
                dataset, odb_dir, category_map,
                text_stub=False, text_source=pdf_text_source,
            )
        finally:
            pdf_text_source.close()
    if gt_path is None:
        print("[omnidocbench] dataset has no ground-truth source; skipping.")
        return {}

    # One markdown file per page in OmniDocBench's expected flat layout.
    """
    markdown_dir = odb_dir / f"markdown_pred_{model_name}"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for pdf_stem, doc in parsed_docs:
        for page_no, page_md in to_markdown_pages(doc, raw=raw_markdown).items():
            (markdown_dir / f"{pdf_stem}_{page_no}.md").write_text(page_md)
    """

    return run_eval(
        gt_path=gt_path,
        pred_md_dir=Path(out_md_dir),
        result_dir=odb_dir / f"results_{model_name}",
        image=omnidocbench_image or DEFAULT_IMAGE,
    )


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
