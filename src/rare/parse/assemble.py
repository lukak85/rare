"""Shared per-page assembly: regions + text → DocItems on a GlasanaDocument.

Both the end-to-end pipeline (PDF → layout → order) and the COCO track
(annotations → order) funnel through `assemble_page`, so the
region→item→article logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rare.doc.schema import (
    AnyDocItem,
    Article,
    BBox,
    FigureItem,
    GlasanaDocument,
    HeadlineItem,
    LABEL_TO_CLASS,
    ParagraphItem,
    Provenance,
)
from rare.parse.figures import crop_and_save_figure


def attach_to_article(
    doc: GlasanaDocument,
    item: AnyDocItem,
    text: str,
    current_article: Optional[Article],
) -> Optional[Article]:
    """Register `item` on `doc`, grouped under the article it belongs to.

    The seed heuristic: a Headline opens a new article and everything after it
    belongs to that article until the next Headline. It is deliberately crude —
    `rare.link.articles` revisits the result with the whole document in view,
    where jump headlines and page-spanning pieces can actually be resolved.

    Shared by the pipeline/COCO assembler and the VLM assembler so the rule
    exists once. Returns the (possibly new) current article.
    """
    if isinstance(item, HeadlineItem):
        current_article = Article(title=text)
        doc.add_article(current_article)

    if current_article is not None:
        item.article_id = current_article.article_id
        current_article.item_ids.append(item.item_id)

    doc.add_item(item)
    return current_article


def assemble_page(
    doc: GlasanaDocument,
    page_no: int,
    regions: list[dict],
    texts: dict[str, str],
    img_w: int,
    img_h: int,
    figures_dir: Path,
    current_article: Optional[Article],
    page_image=None,
) -> Optional[Article]:
    """Turn one page's ordered `regions` into DocItems, registered on `doc`.

    `regions` must already be in reading order; each carries `region_id`,
    `label`, `bbox_norm_1000` ([x0,y0,x1,y1] in 0–1000 space) and an optional
    `score`. `texts` maps region_id → extracted text. `page_image` is used to
    crop figures when present; when None, figures keep an empty `image_path`.

    Returns the (possibly updated) `current_article` so the caller can carry
    article continuity across pages.
    """
    for reading_pos, region in enumerate(regions):
        label = region["label"]
        bbox = BBox.from_norm_1000(region["bbox_norm_1000"], img_w, img_h)
        prov = Provenance.from_bbox(
            page_no=page_no,
            bbox=bbox,
            detection_score=region.get("score"),
            source_region_id=region["region_id"],
            # Set by `rare.parse.ocr` when it re-read a region.
            text_source=region.get("text_source", "pdf"),
            ocr_confidence=region.get("ocr_confidence"),
            text_before_ocr=region.get("text_before_ocr"),
            text_flags=region.get("text_flags"),
        )
        text = texts.get(region["region_id"], "")
        item_cls = LABEL_TO_CLASS.get(label, ParagraphItem)

        kwargs = dict(provenance=prov, reading_order=reading_pos)
        if item_cls is FigureItem:
            image_path = ""
            if page_image is not None:
                fig_path = figures_dir / f"p{page_no}_{region['region_id']}.jpg"
                crop_and_save_figure(page_image, region["bbox_norm_1000"], fig_path)
                image_path = str(fig_path)
            item: AnyDocItem = FigureItem(image_path=image_path, **kwargs)
        else:
            item = item_cls(text=text, **kwargs)

        current_article = attach_to_article(doc, item, text, current_article)

    return current_article