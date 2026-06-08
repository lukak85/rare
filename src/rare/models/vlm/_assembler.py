"""Convert a VLMDocument into a full GlasanaDocument.

Reuses the same Article-grouping logic as the pipeline track (Headline starts
a new Article; subsequent items belong to it until the next Headline) so the
two tracks produce structurally comparable outputs.
"""

from __future__ import annotations

from pathlib import Path

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
from rare.models.vlm._vlm_schema import VLMDocument


def _empty_bbox() -> BBox:
    return BBox(x1=0, y1=0, x2=0, y2=0)


def assemble_document(
    vlm_doc: VLMDocument,
    pdf_stem: str,
    figures_dir: Path | None = None,
) -> GlasanaDocument:
    """Build a GlasanaDocument from a VLMDocument."""
    doc = GlasanaDocument(source_pdf=pdf_stem)
    current_article: Article | None = None

    for vlm_page in sorted(vlm_doc.pages, key=lambda p: p.page_no):
        w = vlm_page.width or 1000
        h = vlm_page.height or 1000
        doc.pages[vlm_page.page_no] = PageInfo(
            page_no=vlm_page.page_no,
            width=w,
            height=h,
            source_file=f"{pdf_stem}_{vlm_page.page_no}.jpg",
        )

        for reading_pos, region in enumerate(vlm_page.regions):
            if region.bbox_norm_1000 is not None:
                bbox = BBox.from_norm_1000(region.bbox_norm_1000, w, h)
            else:
                bbox = _empty_bbox()
            prov = Provenance.from_bbox(
                page_no=vlm_page.page_no,
                bbox=bbox,
                detection_score=region.detection_score,
            )
            item_cls = LABEL_TO_CLASS.get(region.label, ParagraphItem)

            kwargs = dict(provenance=prov, reading_order=reading_pos)
            if item_cls is FigureItem:
                item: AnyDocItem = FigureItem(
                    image_path=region.image_path, **kwargs
                )
            else:
                item = item_cls(text=region.text, **kwargs)

            if isinstance(item, HeadlineItem):
                current_article = Article(title=region.text)
                doc.add_article(current_article)

            if current_article is not None:
                item.article_id = current_article.article_id
                current_article.item_ids.append(item.item_id)

            doc.add_item(item)

    return doc
