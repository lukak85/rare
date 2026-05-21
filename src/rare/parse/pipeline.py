"""End-to-end pipeline parse: PDF → layout → order → assemble → render."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from rare.doc.schema import (
    AnyDocItem,
    Article,
    BBox,
    FigureItem,
    GlasanaDocument,
    HeadlineItem,
    LABEL_TO_CLASS,
    PageInfo,
    ParagraphItem,
    Provenance,
)
from rare.parse.figures import crop_and_save_figure
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
) -> Path:
    """Run layout detection, reading-order, text extraction, and assembly on a PDF.

    Writes `{stem}_doc.json`, `{stem}.md`, `{stem}.html`, and `figures/` to
    `output_dir/<pdf_stem>/`. Returns the output directory.
    """
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem
    out_dir = Path(output_dir) / pdf_stem
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    "label": block.type or "Paragraph",
                    "bbox_norm_1000": _bbox_to_norm_1000(block, img_w, img_h),
                    "score": getattr(block, "score", None),
                })

            texts = extract_text_for_page(pdf, page_no, regions, img_w, img_h)

            for reading_pos, region in enumerate(regions):
                label = region["label"]
                bbox = BBox.from_norm_1000(region["bbox_norm_1000"], img_w, img_h)
                prov = Provenance.from_bbox(
                    page_no=page_no,
                    bbox=bbox,
                    detection_score=region.get("score"),
                    source_region_id=region["region_id"],
                )
                text = texts.get(region["region_id"], "")
                item_cls = LABEL_TO_CLASS.get(label, ParagraphItem)

                kwargs = dict(provenance=prov, reading_order=reading_pos)
                if item_cls is FigureItem:
                    fig_path = figures_dir / f"p{page_no}_{region['region_id']}.jpg"
                    crop_and_save_figure(page_image, region["bbox_norm_1000"], fig_path)
                    item: AnyDocItem = FigureItem(image_path=str(fig_path), **kwargs)
                else:
                    item = item_cls(text=text, **kwargs)

                if isinstance(item, HeadlineItem):
                    current_article = Article(title=text)
                    doc.add_article(current_article)

                if current_article is not None:
                    item.article_id = current_article.article_id
                    current_article.item_ids.append(item.item_id)

                doc.add_item(item)

    return write_outputs(doc, output_dir)
