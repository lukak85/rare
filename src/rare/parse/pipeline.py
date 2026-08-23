"""End-to-end pipeline parse: PDF → layout → order → assemble → render."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from rare.doc.renderers import to_markdown
from rare.doc.schema import (
    Article,
    GlasanaDocument,
    PageInfo,
    relabel_to_glasbena_mladina,
)
from rare.parse.assemble import assemble_page
from rare.parse.io import write_outputs
from rare.parse.merge import merge_flowing_paragraphs
from rare.parse.ocr import DEFAULT_OCR_LABELS, fill_failed_regions
from rare.parse.pdf import render_pages
from rare.parse.text import extract_text_for_page
from rare.utils.conversionutils import layout_parser_to_coco
from rare.utils.fileutils import save_coco_to_json

if TYPE_CHECKING:
    from rare.models.base import LayoutBackend, ReadingOrderBackend


def _bbox_to_norm_1000(block, img_w: int, img_h: int) -> list[float]:
    """Convert an lp.TextBlock (pixel coords) to [x0,y0,x1,y1] in 0-1000 space."""
    x1, y1, x2, y2 = block.coordinates
    return [
        x1 / img_w * 1000.0,
        y1 / img_h * 1000.0,
        x2 / img_w * 1000.0,
        y2 / img_h * 1000.0,
    ]


def parse_pdf(
    pdf_path: str | Path,
    layout: "LayoutBackend",
    order: "ReadingOrderBackend",
    output_dir: str | Path = "outputs/parsed",
    dpi: int = 200,
    per_page: bool = False,
    save_coco: bool = False,
    emit_omnidocbench: bool = False,
    linker=None,
    ocr=None,
    ocr_labels=None,
    ocr_retry=None,
) -> Path:
    """Run layout detection, reading-order, text extraction, and assembly on a PDF.

    Writes `{stem}_doc.json`, `{stem}.md`, `{stem}.html`, and `figures/` to
    `output_dir/<pdf_stem>/`. When `per_page` is True (default), also writes one
    Markdown file per page under `pages/` as `{stem}_{page_no}.md`. Returns the
    output directory.

    If `layout` advertises a `source_taxonomy` (e.g. VGT reporting "D4LA"), each
    detected label is translated into the Glasbena vocabulary via
    `relabel_to_glasbena` before assembly, so foreign-trained detectors produce
    the correct GlasanaDocument item types.

    When `save_coco` is True, also writes `{stem}_coco.json` — one joined COCO
    file carrying the layout detector's raw boxes/labels plus an `order_id` per
    annotation from the reading-order backend (same export the evaluate pipeline
    produces).

    When `emit_omnidocbench` is True, also writes one Markdown file per page
    under `omnidocbench/` as `{stem}_{page_no}.md` — the flat `<image_stem>.md`
    layout OmniDocBench's end2end evaluator mounts at `data_md/predictions`.
    These are rendered from the regions *as detected*, before
    `merge_flowing_paragraphs` re-joins paragraphs split across columns or
    pages, so the export scores the DLA model's own segmentation rather than
    our post-processing. The regular `{stem}.md` / `pages/` outputs stay merged.

    `linker`, when given, is called once with the finished document before it is
    written — see `rare.link.link_document`. It runs here rather than inside
    `write_outputs` so the throw-away single-page documents built for the
    OmniDocBench export are left untouched.

    `ocr`, when given, re-reads regions whose text the PDF's text layer does not
    carry — see `rare.parse.ocr`. `ocr_labels` restricts which labels are
    eligible (default: Header only), and `ocr_retry` names the
    `rare.parse.quality` reasons (`junk`, `sparse`, `alien`) that also earn a
    region a second reading, so text that is present but wrong is not skipped
    for being non-empty. It runs before the OmniDocBench export and before the
    paragraph merge, so every downstream consumer sees one set of texts
    regardless of where each one came from.
    """
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem
    out_dir = Path(output_dir) / pdf_stem
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_taxonomy = getattr(layout, "source_taxonomy", None)

    doc = GlasanaDocument(source_pdf=pdf_stem)
    current_article: Article | None = None
    coco_predictions: list[dict] = []

    page_images = render_pages(pdf_path, dpi=dpi)

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page_image in enumerate(page_images):
            img_w, img_h = page_image.size
            doc.pages[page_no] = PageInfo(
                page_no=page_no,
                width=img_w,
                height=img_h,
                source_file=f"{pdf_stem}_{page_no}.jpg",
            )

            detected_layout = layout.detect(page_image)
            order_indices = order.order(
                detected_layout,
                image=page_image,
                page_no=page_no,
                pdf_stem=pdf_stem,
                pdf_root=str(pdf_path.parent),
            )

            if save_coco:
                image_info = {
                    "id":        page_no,
                    "file_name": f"{pdf_stem}_{page_no}.jpg",
                    "width":     img_w,
                    "height":    img_h,
                }
                coco_predictions.append(layout_parser_to_coco(
                    detected_layout, image_info, layout.label_map,
                    predicted_order=order_indices,
                ))

            # Build regions dicts in reading order
            regions: list[dict] = []
            for idx in order_indices:
                block = detected_layout[idx]
                regions.append({
                    "region_id": str(uuid.uuid4()),
                    "label": relabel_to_glasbena_mladina(block.type or "Paragraph", source_taxonomy),
                    "bbox_norm_1000": _bbox_to_norm_1000(block, img_w, img_h),
                    "score": getattr(block, "score", None),
                })

            texts = extract_text_for_page(pdf, page_no, regions, img_w, img_h)

            if ocr is not None:
                fill_failed_regions(
                    regions, texts,
                    recognizer=ocr,
                    pdf_path=pdf_path,
                    page_no=page_no,
                    page_w=img_w,
                    page_h=img_h,
                    labels=ocr_labels or DEFAULT_OCR_LABELS,
                    retry=ocr_retry or (),
                    page_image=page_image,
                    page_image_dpi=dpi,
                )

            # OmniDocBench scores the detector's own segmentation, so this
            # export is taken before the paragraph merge below.
            if emit_omnidocbench:
                _write_omnidocbench_page(
                    out_dir / "omnidocbench",
                    pdf_stem=pdf_stem,
                    page_no=page_no,
                    regions=regions,
                    texts=texts,
                    img_w=img_w,
                    img_h=img_h,
                )

            # Re-join paragraphs split across columns, page geometry, or floats
            # (e.g. a figure inserted mid-paragraph) into one logical paragraph.
            regions, texts = merge_flowing_paragraphs(regions, texts)

            current_article = assemble_page(
                doc,
                page_no=page_no,
                regions=regions,
                texts=texts,
                img_w=img_w,
                img_h=img_h,
                figures_dir=figures_dir,
                current_article=current_article,
                page_image=page_image,
            )

    if save_coco and coco_predictions:
        coco_data = _join_coco_pages(coco_predictions)
        save_coco_to_json(coco_data, str(out_dir / f"{pdf_stem}_coco.json"))

    if linker is not None:
        linker(doc)

    return write_outputs(doc, output_dir, per_page=per_page)


def _write_omnidocbench_page(
    odb_dir: Path,
    pdf_stem: str,
    page_no: int,
    regions: list[dict],
    texts: dict[str, str],
    img_w: int,
    img_h: int,
) -> Path:
    """Render one page's regions to `<odb_dir>/{pdf_stem}_{page_no}.md`.

    Assembles a throw-away single-page document so the markdown goes through
    the same renderer the VLM track is scored with. `page_image` is left None:
    figure crops already belong to the main output and are not re-written here
    (figures render as an image with an empty src, which the evaluator's
    markdown parser drops).
    """
    page_doc = GlasanaDocument(source_pdf=pdf_stem)
    page_doc.pages[page_no] = PageInfo(
        page_no=page_no,
        width=img_w,
        height=img_h,
        source_file=f"{pdf_stem}_{page_no}.jpg",
    )
    assemble_page(
        page_doc,
        page_no=page_no,
        regions=regions,
        texts=texts,
        img_w=img_w,
        img_h=img_h,
        figures_dir=odb_dir,
        current_article=None,
    )

    odb_dir.mkdir(parents=True, exist_ok=True)
    md_path = odb_dir / f"{pdf_stem}_{page_no}.md"
    md_path.write_text(to_markdown(page_doc))
    return md_path


def _join_coco_pages(pages: list[dict]) -> dict:
    """Merge per-page COCO dicts into one, reassigning globally unique annotation
    IDs. Categories are taken from the first page (all pages share one detector,
    hence one taxonomy)."""
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for page in pages:
        images.extend(page["images"])
        for ann in page["annotations"]:
            ann = dict(ann)
            ann["id"] = annotation_id
            annotation_id += 1
            annotations.append(ann)
    return {
        "images": images,
        "annotations": annotations,
        "categories": pages[0]["categories"],
    }
