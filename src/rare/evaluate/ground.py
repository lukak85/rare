"""Loading the annotated ground truth, and the documents scored against it.

Shared by every document-level evaluation track. Two ways to get a
`GlasanaDocument` to score live here:

* `load_documents` reads the `*_doc.json` files a parse already wrote — the
  end-to-end reading, where detection, order and linking all count;
* `build_ground_documents` assembles one from the annotations themselves —
  `rare.parse.coco.parse_coco` with everything the metrics do not need taken
  out, which isolates whatever pass the caller then runs over it.

Both yield the same type, so a track can take either without knowing which.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from rare.doc.schema import GlasanaDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The annotations, as read off disk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundRegion:
    ann_id: int
    label: str
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2, annotation pixels


@dataclass
class GroundPage:
    page_no: int
    width: float                               # the size the boxes were drawn at
    height: float
    page_type: Optional[str]
    regions: list[GroundRegion]                # every annotation, in reading order


@dataclass
class GroundDoc:
    pdf_stem: str
    pages: dict[int, GroundPage] = field(default_factory=dict)


def split_stem_page(file_name: str) -> tuple[str, int]:
    """"<stem>_<page>.jpg" → (stem, page_no); (full stem, 0) when it doesn't fit."""
    name = Path(file_name).name
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1].rsplit(".", 1)[0])
        except ValueError:
            pass
    return Path(name).stem, 0


def load_ground(coco_path: str | Path) -> dict[str, GroundDoc]:
    """Read a COCO file with `order_id` into `{pdf_stem: GroundDoc}`.

    Annotations without an `order_id` fall to the end of their page in their
    original order — the same convention `rare.evaluate.datasets` uses.
    """
    raw = json.loads(Path(coco_path).read_text())
    label_of = {c["id"]: c["name"] for c in raw["categories"]}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in raw["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    docs: dict[str, GroundDoc] = {}
    for image in raw["images"]:
        stem, page_no = split_stem_page(image["file_name"])
        anns = anns_by_image.get(image["id"], [])
        anns.sort(key=lambda a: (a.get("order_id") is None, a.get("order_id", 0)))

        doc = docs.setdefault(stem, GroundDoc(pdf_stem=stem))
        doc.pages[page_no] = GroundPage(
            page_no=page_no,
            width=float(image["width"]),
            height=float(image["height"]),
            page_type=image.get("page_type"),
            regions=[
                GroundRegion(
                    ann_id=a["id"],
                    label=label_of[a["category_id"]],
                    bbox=(a["bbox"][0], a["bbox"][1],
                          a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]),
                )
                for a in anns
            ],
        )
    return docs


# ---------------------------------------------------------------------------
# Documents: from a real parse, or assembled from the annotations
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str | Path) -> Iterator[GlasanaDocument]:
    """Every `*_doc.json` under `docs_dir`, as parsed documents.

    Hidden directories are skipped and a stem is taken once: output trees keep
    superseded runs beside the current one (`.old/`, dated snapshots), and
    scoring an archived copy of a document a second time would silently double
    its weight in the aggregate.
    """
    seen: set[str] = set()
    for path in sorted(Path(docs_dir).rglob("*_doc.json")):
        if any(part.startswith(".") for part in path.parts):
            continue
        doc = GlasanaDocument.model_validate_json(path.read_text())
        stem = Path(doc.source_pdf).stem or doc.source_pdf
        if stem in seen:
            logger.warning("document %r already scored; skipping %s", stem, path)
            continue
        seen.add(stem)
        yield doc


def build_ground_documents(
    ground: dict[str, GroundDoc],
    pdfs_dir: str | Path | None = None,
    linker: Optional[Callable[[GlasanaDocument], object]] = None,
    stems: Optional[Iterable[str]] = None,
) -> Iterator[GlasanaDocument]:
    """Assemble one document per stem from the annotations, then link it.

    This is `rare.parse.coco.parse_coco` with everything the metric does not
    need taken out: no page rendering, no figure crops, no HTML/Markdown. Text
    still comes from `<pdfs_dir>/<stem>.pdf` when it is there, because the
    splitting and continuation passes read it; without it the geometric passes
    still run and the numbers are a floor rather than a fair reading.
    """
    import pdfplumber

    from rare.doc.schema import PageInfo
    from rare.parse.assemble import assemble_page
    from rare.parse.text import extract_text_for_page

    pdfs_dir = Path(pdfs_dir) if pdfs_dir else None
    wanted = set(stems) if stems is not None else None

    for stem in sorted(ground):
        if wanted is not None and stem not in wanted:
            continue
        doc_ground = ground[stem]
        doc = GlasanaDocument(source_pdf=stem)
        current_article = None

        pdf_path = pdfs_dir / f"{stem}.pdf" if pdfs_dir else None
        if pdf_path is not None and not pdf_path.exists():
            logger.warning(
                "no PDF at %s; %s is assembled without text, so the linking passes "
                "that read it contribute nothing", pdf_path, stem,
            )
        pdf = pdfplumber.open(pdf_path) if pdf_path and pdf_path.exists() else None
        try:
            for page_no in sorted(doc_ground.pages):
                page = doc_ground.pages[page_no]
                doc.pages[page_no] = PageInfo(
                    page_no=page_no,
                    width=page.width,
                    height=page.height,
                    source_file=f"{stem}_{page_no}.jpg",
                )
                regions = [
                    {
                        "region_id": str(uuid.uuid4()),
                        "label": r.label,
                        "bbox_norm_1000": [
                            r.bbox[0] / page.width * 1000.0,
                            r.bbox[1] / page.height * 1000.0,
                            r.bbox[2] / page.width * 1000.0,
                            r.bbox[3] / page.height * 1000.0,
                        ],
                        "score": 1.0,
                    }
                    for r in page.regions
                ]
                texts = (
                    extract_text_for_page(pdf, page_no, regions, page.width, page.height)
                    if pdf is not None and page_no < len(pdf.pages)
                    else {}
                )
                current_article = assemble_page(
                    doc,
                    page_no=page_no,
                    regions=regions,
                    texts=texts,
                    img_w=page.width,
                    img_h=page.height,
                    figures_dir=Path("."),      # unused: no page image is passed
                    current_article=current_article,
                    page_image=None,
                )
        finally:
            if pdf is not None:
                pdf.close()

        if linker is not None:
            linker(doc)
        yield doc
