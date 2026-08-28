"""Shared region → OmniDocBench per-page Markdown rendering.

Both `rare parse --emit-omnidocbench` (`rare.parse.pipeline`) and
`rare evaluate --track pipeline` (`rare.evaluate.runner`) need to turn a
detector's boxes plus a reading order into the flat `<image_stem>.md` layout
OmniDocBench's `end2end_dataset` loader expects. Keeping both on this one
renderer is what makes the pipeline track's Edit_dist comparable to the VLM
track's: the markdown goes through `rare.doc.renderers.to_markdown` either way.

Lives in its own module so `rare.evaluate.runner` can import it without pulling
in `rare.parse.pipeline`'s pdfplumber/`render_pages` dependencies.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Optional

from rare.doc.renderers import to_markdown
from rare.doc.schema import GlasanaDocument, PageInfo, relabel_to_glasbena_mladina
from rare.parse.assemble import assemble_page


def bbox_to_norm_1000(block, img_w: int, img_h: int) -> list[float]:
    """Convert an lp.TextBlock (pixel coords) to [x0,y0,x1,y1] in 0-1000 space."""
    x1, y1, x2, y2 = block.coordinates
    return [
        x1 / img_w * 1000.0,
        y1 / img_h * 1000.0,
        x2 / img_w * 1000.0,
        y2 / img_h * 1000.0,
    ]


def regions_from_layout(
    detected_layout,
    order_indices: Iterable[int],
    img_w: int,
    img_h: int,
    source_taxonomy: Optional[str] = None,
) -> list[dict]:
    """Build the region dicts the assembler consumes, in reading order.

    `order_indices` indexes into `detected_layout`; labels are translated out of
    the backend's own taxonomy (`source_taxonomy`, e.g. VGT reporting "D4LA")
    into the Glasbena Mladina vocabulary the schema is built around.
    """
    regions: list[dict] = []
    for idx in order_indices:
        block = detected_layout[idx]
        regions.append({
            "region_id": str(uuid.uuid4()),
            "label": relabel_to_glasbena_mladina(block.type or "Paragraph", source_taxonomy),
            "bbox_norm_1000": bbox_to_norm_1000(block, img_w, img_h),
            "score": getattr(block, "score", None),
        })
    return regions


def write_omnidocbench_page(
    odb_dir: Path,
    out_stem: str,
    page_no: int,
    regions: list[dict],
    texts: dict[str, str],
    img_w: int,
    img_h: int,
) -> Path:
    """Render one page's regions to `<odb_dir>/<out_stem>.md`.

    `out_stem` must match the stem of the corresponding ground-truth page image
    (OmniDocBench resolves predictions as `basename(image_path)[:-4] + ".md"`),
    so callers pass the GT image's own stem rather than reconstructing one.

    Assembles a throw-away single-page document so the markdown goes through the
    same renderer the VLM track is scored with. `page_image` is left None:
    figure crops already belong to the main output and are not re-written here
    (figures render as an image with an empty src, which the evaluator's
    markdown parser drops).
    """
    page_doc = GlasanaDocument(source_pdf=out_stem)
    page_doc.pages[page_no] = PageInfo(
        page_no=page_no,
        width=img_w,
        height=img_h,
        source_file=f"{out_stem}.jpg",
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
    md_path = odb_dir / f"{out_stem}.md"
    md_path.write_text(to_markdown(page_doc))
    return md_path
