"""End-to-end pipeline parse: PDF → layout → order → assemble → render."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from rare.doc.schema import (
    Article,
    GlasanaDocument,
    PageInfo,
    relabel_to_glasbena_mladina,
)
from rare.parse.assemble import assemble_page
from rare.parse.io import write_outputs
from rare.parse.pdf import render_pages
from rare.parse.text import extract_text_for_page

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
    """
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem
    out_dir = Path(output_dir) / pdf_stem
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_taxonomy = getattr(layout, "source_taxonomy", None)

    doc = GlasanaDocument(source_pdf=pdf_stem)
    current_article: Article | None = None

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
            )

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

    return write_outputs(doc, output_dir, per_page=per_page)
