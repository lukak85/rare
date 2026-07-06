"""Parse a COCO layout JSON (ground truth or any prediction) into rendered docs.

Given a COCO annotations file describing region boxes + categories (and,
optionally, per-annotation reading order via `order_id`), this rebuilds a
`GlasanaDocument` and writes Markdown / HTML / JSON — the same outputs as the
end-to-end pipeline, but skipping layout detection entirely.

Text is filled per-box from a matching source PDF when one resolves under
`pdfs_dir` (`<stem>.pdf`); otherwise regions keep empty text (structure-only).
Figure crops are taken from the page image when an image (from `images_dir`)
or a rendered PDF page is available.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pdfplumber
from pycocotools.coco import COCO
from tqdm import tqdm

from rare.doc.schema import Article, GlasanaDocument, PageInfo
from rare.parse.assemble import assemble_page
from rare.parse.io import write_outputs
from rare.parse.pdf import render_page
from rare.parse.text import extract_text_for_page

if TYPE_CHECKING:
    from rare.models.base import ReadingOrderBackend


def _split_stem_page(file_name: str) -> tuple[str, int]:
    """Split a "<stem>_<page>.jpg" file name into (pdf_stem, page_no).

    Falls back to (full-stem, 0) when the trailing token is not an int —
    matching the convention in `rare.evaluate.datasets`.
    """
    name = Path(file_name).name
    stem_parts = name.rsplit("_", 1)
    if len(stem_parts) == 2:
        try:
            return stem_parts[0], int(stem_parts[1].rsplit(".", 1)[0])
        except ValueError:
            pass
    return Path(name).stem, 0


def _bbox_to_norm_1000(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """COCO [x, y, w, h] (pixels) → [x0, y0, x1, y1] in 0–1000 space."""
    x, y, w, h = bbox
    return [
        x / img_w * 1000.0,
        y / img_h * 1000.0,
        (x + w) / img_w * 1000.0,
        (y + h) / img_h * 1000.0,
    ]


def _order_regions(
    regions: list[dict],
    anns: list[dict],
    order: Optional["ReadingOrderBackend"],
    page_image,
    page_no: int,
    pdf_stem: str,
) -> list[dict]:
    """Return `regions` in reading order.

    Precedence: per-annotation `order_id` when every annotation carries it;
    else the supplied reading-order backend; else top-to-bottom, left-to-right.
    """
    if anns and all("order_id" in a for a in anns):
        return [regions[i] for i in sorted(range(len(anns)), key=lambda i: anns[i]["order_id"])]

    if order is not None:
        import layoutparser as lp

        layout = lp.Layout()
        for r in regions:
            x0, y0, x1, y1 = r["bbox_norm_1000"]
            layout.append(lp.TextBlock(block=lp.Rectangle(x0, y0, x1, y1), type=r["label"]))
        indices = order.order(layout, image=page_image, page_no=page_no, pdf_stem=pdf_stem)
        return [regions[i] for i in indices]

    return sorted(regions, key=lambda r: (r["bbox_norm_1000"][1], r["bbox_norm_1000"][0]))


def _load_page_image(images_dir: Optional[Path], file_name: str, pdf_path: Optional[Path],
                     page_no: int, dpi: int):
    """Best-effort page image for figure cropping: image file, else PDF render, else None."""
    if images_dir is not None:
        cand = images_dir / Path(file_name).name
        if cand.exists():
            from PIL import Image
            return Image.open(cand)
    if pdf_path is not None:
        try:
            return render_page(pdf_path, page_no, dpi=dpi)
        except Exception:
            return None
    return None


def parse_coco(
    coco_path: str | Path,
    pdf_path: str | Path | None = None,
    images_dir: str | Path | None = None,
    pdfs_dir: str | Path | None = None,
    order: Optional["ReadingOrderBackend"] = None,
    output_dir: str | Path = "outputs/parsed",
    dpi: int = 200,
    emit_omnidocbench: bool = False,
    category_map: Optional[dict[str, str]] = None,
) -> list[Path]:
    """Render every document described by a COCO file to HTML / MD / JSON.

    Images are grouped into documents by `<stem>` (parsed from each file name);
    pages within a document are processed in `page_no` order. Returns the list
    of per-document output directories.

    Text source per document: the explicit `pdf_path` (when given) wins, else
    `<pdfs_dir>/<stem>.pdf`. When `pdf_path` is given, only the matching
    document is rendered.

    When `emit_omnidocbench` is True, also writes `<output_dir>/omnidocbench.json`
    — the OmniDocBench per-page list for every processed document, with each
    region's `text` filled from the same PDF crops (so VLM markdown can be
    scored against it end-to-end). `category_map` overrides the default
    source-name → OmniDocBench `category_type` map.
    """
    coco = COCO(str(coco_path))
    images_dir = Path(images_dir) if images_dir else None
    pdfs_dir = Path(pdfs_dir) if pdfs_dir else None
    pdf_path = Path(pdf_path) if pdf_path else None
    only_stem = pdf_path.stem if pdf_path is not None else None

    # Group COCO image entries by source-document stem.
    by_stem: dict[str, list[tuple[int, int]]] = {}  # stem → [(page_no, image_id)]
    for image_id, info in coco.imgs.items():
        stem, page_no = _split_stem_page(info["file_name"])
        if only_stem is not None and stem != only_stem:
            continue
        by_stem.setdefault(stem, []).append((page_no, image_id))

    out_dirs: list[Path] = []
    for stem, pages in sorted(by_stem.items()):
        pages.sort()
        doc = GlasanaDocument(source_pdf=stem)
        current_article: Optional[Article] = None

        # Resolve the PDF for this stem: explicit --pdf wins, else <pdfs_dir>/<stem>.pdf.
        stem_pdf = pdf_path
        if stem_pdf is None and pdfs_dir is not None:
            cand = pdfs_dir / f"{stem}.pdf"
            stem_pdf = cand if cand.exists() else None

        figures_dir = Path(output_dir) / stem / "figures"
        pdf = pdfplumber.open(stem_pdf) if stem_pdf is not None else None
        try:
            for page_no, image_id in tqdm(pages):
                info = coco.imgs[image_id]
                img_w, img_h = info["width"], info["height"]
                doc.pages[page_no] = PageInfo(
                    page_no=page_no,
                    width=img_w,
                    height=img_h,
                    source_file=info["file_name"],
                )

                anns = coco.loadAnns(coco.getAnnIds([image_id]))
                regions = [
                    {
                        "region_id": str(uuid.uuid4()),
                        "label": coco.cats[a["category_id"]]["name"],
                        "bbox_norm_1000": _bbox_to_norm_1000(a["bbox"], img_w, img_h),
                        "score": a.get("score"),
                    }
                    for a in anns
                ]

                page_image = _load_page_image(images_dir, info["file_name"], stem_pdf, page_no, dpi)
                regions = _order_regions(regions, anns, order, page_image, page_no, stem)

                if pdf is not None and page_no < len(pdf.pages):
                    texts = extract_text_for_page(pdf, page_no, regions, img_w, img_h)
                else:
                    texts = {}

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
        finally:
            if pdf is not None:
                pdf.close()

        out_dirs.append(write_outputs(doc, output_dir))

    if emit_omnidocbench:
        _emit_omnidocbench(
            coco_path, set(by_stem), pdf_path, pdfs_dir, category_map, Path(output_dir)
        )

    return out_dirs


def _emit_omnidocbench(
    coco_path: str | Path,
    stems: set[str],
    pdf_path: Optional[Path],
    pdfs_dir: Optional[Path],
    category_map: Optional[dict[str, str]],
    output_dir: Path,
) -> Path:
    """Write `<output_dir>/omnidocbench.json` for the processed `stems`.

    Region `text` is filled per box from the PDF via `PdfTextSource`, using
    `pdfs_dir` when given, else the directory holding the explicit `--pdf`.
    """
    from rare.evaluate.omnidocbench import (
        DEFAULT_PAGE_ATTRIBUTE_FIELDS,
        coco_to_omnidocbench,
    )
    from rare.evaluate.pdf_text import PdfTextSource

    raw = json.loads(Path(coco_path).read_text())
    keep_ids = {
        img["id"] for img in raw.get("images", [])
        if _split_stem_page(img["file_name"])[0] in stems
    }
    filtered = {
        "images":      [i for i in raw.get("images", []) if i["id"] in keep_ids],
        "categories":  raw.get("categories", []),
        "annotations": [a for a in raw.get("annotations", []) if a["image_id"] in keep_ids],
    }

    text_dir = pdfs_dir if pdfs_dir is not None else (pdf_path.parent if pdf_path else None)
    text_source = PdfTextSource(text_dir) if text_dir is not None else None
    try:
        pages = coco_to_omnidocbench(
            filtered, category_map, text_source=text_source,
            page_attribute_fields=DEFAULT_PAGE_ATTRIBUTE_FIELDS,
        )
    finally:
        if text_source is not None:
            text_source.close()

    out_path = output_dir / "omnidocbench.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False))
    return out_path