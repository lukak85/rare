from __future__ import annotations

import json
import os
import re
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from rare.doc.schema import (
    Article,
    FigureItem,
    GlasanaDocument,
    PageInfo,
    TableCell,
    TableData,
    TableItem,
)
from rare.models.registry import register
from rare.models.vlm.prompts import YOUTU_LABEL_MAP
from rare.parse.assemble import assemble_page
from rare.utils.fileutils import split_stem_page

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "youtu")
class YoutuBackend:

    name = "youtu"

    def __init__(self, config: dict | None = None):
        self._converter = None  # built lazily on first use
        cfg = config or {}
        # Read rather than required: a document whose pages are already parsed
        # is rebuilt from the JSON on disk, and never reaches the model.
        self.model_path = cfg.get("model_path")
        self.angle_correct_model_path = cfg.get("angle_correct_model_path")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force everything on the same device
        self.verbose = True
        # parse_pdf: DPI the PDF is rendered at, and where the parser's own
        # JSON/Markdown/preview files are kept. Defaulting the work directory
        # to a path rather than a temp dir is deliberate — a Youtu run is slow
        # enough that being able to re-render the document from JSON, without
        # re-running the model, is worth more than tidiness.
        self.dpi = int(cfg.get("dpi", 200))
        self.work_dir = cfg.get("work_dir", "outputs/youtu")

    def _get_converter(self):
        if self._converter is None:
            if not self.model_path:
                raise RuntimeError(
                    "youtu: no model_path in the backend config, and a page "
                    "still has to be parsed. Pass --config configs/youtu/config.json."
                )
            from youtu_hf_parser import YoutuOCRParserHF
            self._converter = YoutuOCRParserHF(
                model_path=self.model_path,                    # Path to downloaded model weights
                enable_angle_correct=True,                # Set to False to disable angle correction
                angle_correct_model_path=self.angle_correct_model_path  # If None, model will auto-download to default path; if custom path, manually download https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth to specified location
            )
        return self._converter

    @staticmethod
    def _load_image_paths(image_dir: str | Path) -> list[str]:
        """Recursively collect supported images under `image_dir`, returning
        parallel lists of absolute paths and RGB PIL images."""
        image_paths: list[str] = []
        for root, _dirs, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
        image_paths.sort()
        print(f"found {len(image_paths)} image files.")
        return image_paths

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        image_paths = self._load_image_paths(image_dir)

        for image_path in tqdm(image_paths):
            out_md = Path(out_md_dir) / f"{Path(image_path).stem}.md"   # xx_1.jpg -> xx_1.md
            if skip_existing and out_md.exists():
                print(f"skipping {out_md}")
                continue
            try:
                self._get_converter().parse_file(
                    input_path=image_path,     # Input document path
                    output_dir=out_md_dir      # Output directory for results
                )
            except Exception as e:
                if self.verbose:
                    print(e)
                try:
                    import re
                    lower_res_image_path = re.sub(r"/eval_\d+dpi/", "/eval/", image_path)
                    print(f"Using {lower_res_image_path}")
                    self._get_converter().parse_file(
                        input_path=lower_res_image_path,     # Input document path
                        output_dir=out_md_dir      # Output directory for results
                    )
                except Exception as e2:
                    if self.verbose:
                        print(e2)
                    print("Skipping " + image_path)
                    pass

        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Render `pdf_path`, parse every page, and rebuild a GlasanaDocument.

        Page images are written to `<work_dir>/images/<stem>_<page>.jpg` and the
        parser's output to `<work_dir>/<stem>/`, both kept afterwards. That is
        what makes this the only Youtu entry point needed: a page whose JSON is
        already there is not parsed again, so re-running the same command on a
        finished document rebuilds it from disk — no GPU, no model weights in
        the config, and not even a PDF render, since the page images are kept
        too. Pages the parser fails on are skipped with a warning rather than
        aborting the document.
        """
        from rare.parse.pdf import page_count, render_pages

        pdf_path = Path(pdf_path)
        stem = pdf_path.stem
        work_dir = Path(self.work_dir) / stem
        images_dir = Path(self.work_dir) / "images"
        work_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # A page needs the model when its JSON is missing, and needs a render
        # when its image is missing — the crops in `document_from_pages` come
        # off that image, so a re-render still wants it on disk.
        todo = [
            page_no
            for page_no in range(page_count(pdf_path))
            if not (work_dir / f"{stem}_{page_no}.json").exists()
            or not (images_dir / f"{stem}_{page_no}.jpg").exists()
        ]
        if todo:
            page_images = render_pages(pdf_path, dpi=self.dpi)
            # Loaded once, before the loop: a missing model path or missing
            # weights is the whole run's problem, not one page's, and inside
            # the loop it would come out as a per-page warning and a document
            # quietly short of its pages.
            unparsed = [p for p in todo if not (work_dir / f"{stem}_{p}.json").exists()]
            converter = self._get_converter() if unparsed else None
            for page_no in tqdm(todo):
                image_path = images_dir / f"{stem}_{page_no}.jpg"
                if not image_path.exists():
                    page_images[page_no].convert("RGB").save(image_path)
                if page_no not in unparsed:
                    continue
                try:
                    converter.parse_file(
                        input_path=str(image_path), output_dir=str(work_dir)
                    )
                except Exception as exc:
                    print(f"[warn] Youtu failed on page {page_no} of {stem}: {exc}")

        pages = collect_pages(work_dir).get(stem, [])
        if not pages:
            print(f"[warn] no Youtu page JSON under {work_dir}")
        return document_from_pages(
            stem, pages, images_dir=images_dir, figures_dir=work_dir / "figures"
        )

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Youtu prints the heading level it inferred into a Title region's own text
# ("## KAM JE IZGINILA TIŠINA?"). We read the level off it, then strip it —
# the renderer re-applies markup from the label, so leaving it would emit
# "# ## KAM JE ...".
_MD_HEADING = re.compile(r"^\s*(#{1,6})\s+")

# Depth (1-based, from the hierarchy tree) → heading label. Anything deeper
# than the map stops at Subsubhead, which is as far as the schema goes.
_DEPTH_TO_HEADING = {1: "Headline", 2: "Subhead"}

# Longest figure text kept as alt text; Youtu's figure OCR is often a caption
# printed inside the picture, but it can also be a page of stray glyphs.
_MAX_ALT_TEXT = 200

# Structure markers Youtu leaves in recognized text ("<start>0<content>...").
# They mean nothing outside its own decoding and read as markup once the text
# lands in an HTML attribute.
_YOUTU_MARKER = re.compile(r"<[a-z_]+>\d*")


# ---------------------------------------------------------------------------
# Reading Youtu's output files
# ---------------------------------------------------------------------------

def collect_pages(youtu_dir: str | Path) -> dict[str, list[tuple[int, Path]]]:
    """Group `<stem>_<page>.json` files under `youtu_dir` by document stem.

    Returns `{pdf_stem: [(page_no, json_path), ...]}` with pages sorted. The
    `_hierarchy.json` siblings are skipped here and picked up per page.
    """
    by_stem: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(Path(youtu_dir).glob("*.json")):
        if path.name.endswith("_hierarchy.json"):
            continue
        stem, page_no = split_stem_page(path.name)
        by_stem.setdefault(stem, []).append((page_no, path))
    for pages in by_stem.values():
        pages.sort()
    return by_stem


def _load_hierarchy_depths(json_path: Path) -> dict[int, int]:
    """Flatten `<stem>_<page>_hierarchy.json` into `{flat index: depth}`.

    Depth is 1-based (roots are 1). Regions Youtu left out of the tree — page
    furniture, mostly, plus the occasional stray — simply get no entry, and the
    caller falls back to the label map's default. Returns `{}` when the file is
    missing or unreadable, which is the same thing as "no depth information".
    """
    hierarchy_path = json_path.with_name(f"{json_path.stem}_hierarchy.json")
    try:
        tree = json.loads(hierarchy_path.read_text())
    except (OSError, ValueError):
        return {}

    depths: dict[int, int] = {}
    stack = [(node, 1) for node in reversed(tree.get("nodes") or [])]
    while stack:
        node, depth = stack.pop()
        node_id = node.get("id")
        # First depth seen wins. `build_hierarchy_json` already breaks cycles
        # when it writes the file, so this is only here to stop a region that
        # somehow appears twice from looping the walk.
        if isinstance(node_id, int) and node_id not in depths:
            depths[node_id] = depth
        for child in reversed(node.get("children") or []):
            stack.append((child, depth + 1))
    return depths


def _page_size(json_path: Path, image_path: Optional[Path], items: list[dict]) -> tuple[int, int]:
    """Pixel size of the frame this page's bboxes live in.

    `_layout.png` first: Youtu writes it from the very image it parsed, so it
    stays right even when a run read some pages at one resolution and some at
    another. Then the page image the caller pointed us at, then the boxes' own
    extent, which at least keeps everything on the page.
    """
    from PIL import Image

    for candidate in (json_path.with_name(f"{json_path.stem}_layout.png"), image_path):
        if candidate is not None and candidate.exists():
            try:
                with Image.open(candidate) as img:
                    return img.size
            except OSError:
                pass

    width = height = 1
    for item in items:
        bbox = item.get("bbox") or []
        if bbox:
            width = max(width, int(max(bbox[0::2])))
            height = max(height, int(max(bbox[1::2])))
    return width, height


def _find_page_image(images_dir: Optional[Path], stem: str, page_no: int) -> Optional[Path]:
    """Locate `<stem>_<page>.<ext>` under `images_dir`, or None."""
    if images_dir is None:
        return None
    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}_{page_no}{suffix}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

def _bbox_to_norm_1000(bbox: Iterable[float], img_w: int, img_h: int) -> list[float]:
    """Youtu bbox → [x0, y0, x1, y1] in 0–1000 space.

    Takes the axis-aligned hull of the interleaved x/y pairs, so it reads both
    the 8-point quad `parse_file` writes and the plain 4-point rectangle the
    parser carries internally. The hull is clipped: an inverse rotation on a
    skewed page can leave a corner outside the image.
    """
    coords = list(bbox)
    xs, ys = coords[0::2], coords[1::2]
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]

    def clamp(value: float) -> float:
        return max(0.0, min(1000.0, value))

    return [
        clamp(min(xs) / img_w * 1000.0),
        clamp(min(ys) / img_h * 1000.0),
        clamp(max(xs) / img_w * 1000.0),
        clamp(max(ys) / img_h * 1000.0),
    ]


def _heading_label(text: str, depth: Optional[int]) -> tuple[str, str]:
    """Resolve a Youtu `Title` into a heading label, and strip its markup.

    Youtu's own heading level wins when it printed one, because it is a
    judgement about this region; the hierarchy depth is the fallback, and a
    Title with neither is a Headline (the label map's default). Returns
    `(label, text)`.
    """
    match = _MD_HEADING.match(text)
    if match:
        level = len(match.group(1))
        text = text[match.end():]
    elif depth is not None:
        level = depth
    else:
        level = 1
    label = _DEPTH_TO_HEADING.get(level, "Subsubhead" if level > 2 else "Headline")
    return label, text.strip()


def page_to_regions(
    items: list[dict],
    depths: dict[int, int],
    img_w: int,
    img_h: int,
) -> tuple[list[dict], dict[str, str]]:
    """One page's flat Youtu region list → (`regions`, `texts`) for `assemble_page`.

    Regions come back in the input order, which is Youtu's reading order.
    `texts` is keyed by the generated `region_id`, as the assembler expects.
    """
    regions: list[dict] = []
    texts: dict[str, str] = {}

    for index, item in enumerate(items):
        youtu_type = item.get("type") or "Text"
        label = YOUTU_LABEL_MAP.get(youtu_type, "Paragraph")
        text = (item.get("content") or "").strip()

        if youtu_type == "Title":
            label, text = _heading_label(text, depths.get(index))

        region_id = str(uuid.uuid4())
        regions.append({
            "region_id": region_id,
            "label": label,
            "bbox_norm_1000": _bbox_to_norm_1000(item.get("bbox") or [], img_w, img_h),
        })
        texts[region_id] = text

    return regions, texts


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class _TableHTMLParser(HTMLParser):
    """Collect `<tr>/<td>` cells, honouring rowspan/colspan.

    Youtu recognizes a table as OTSL and converts it to HTML before writing it
    into the region's `content` (`youtu_parsing_utils.table_utils.
    convert_table_ostl_to_html`), so this is the only shape we need to read.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[TableCell] = []
        self._occupied: set[tuple[int, int]] = set()
        self._row = -1
        self._col = 0
        self._cell: Optional[TableCell] = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._close_cell()
            self._row += 1
            self._col = 0
        elif tag in ("td", "th"):
            self._close_cell()
            attributes = dict(attrs)
            while (self._row, self._col) in self._occupied:
                self._col += 1
            row_span = _positive_int(attributes.get("rowspan"))
            col_span = _positive_int(attributes.get("colspan"))
            self._cell = TableCell(
                row=max(self._row, 0),
                col=self._col,
                row_span=row_span,
                col_span=col_span,
                is_header=(tag == "th"),
            )
            for r in range(self._cell.row, self._cell.row + row_span):
                for c in range(self._cell.col, self._cell.col + col_span):
                    self._occupied.add((r, c))
            self._col += col_span

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close_cell()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._close_cell()

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        self._cell.text = " ".join("".join(self._text).split())
        self.cells.append(self._cell)
        self._cell = None
        self._text = []


def _alt_text(text: str) -> str:
    """Youtu's figure text → a one-line alt string.

    The renderers drop `alt_text` straight into `alt="..."` and `![...]()`
    without escaping, so the quotes and decoder markers that come out of figure
    OCR are taken out here rather than left to break the attribute.
    """
    cleaned = _YOUTU_MARKER.sub(" ", text).replace('"', "'")
    return " ".join(cleaned.split())[:_MAX_ALT_TEXT].strip()


def _positive_int(value: Optional[str]) -> int:
    try:
        return max(1, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


def html_table_to_data(html: str) -> Optional[TableData]:
    """Youtu's `<table>` HTML → `TableData`, or None when it holds no cells.

    None is the signal to keep the HTML in the item's `raw_text` instead, which
    is what `TableItem` falls back to.
    """
    if not html or "<t" not in html:
        return None
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return None
    if not parser.cells:
        return None
    return TableData(
        num_rows=max(c.row + c.row_span for c in parser.cells),
        num_cols=max(c.col + c.col_span for c in parser.cells),
        cells=parser.cells,
    )


# ---------------------------------------------------------------------------
# Post-assembly fixups
# ---------------------------------------------------------------------------

def _apply_region_content(doc: GlasanaDocument, regions: list[dict], texts: dict[str, str]) -> None:
    """Move region text into the fields `assemble_page` cannot fill.

    `assemble_page` builds every non-figure item as `item_cls(text=...)`, which
    is right for the text categories and wrong for the two that keep their
    content elsewhere: a FigureItem has `alt_text`, and a TableItem has
    `table_data` / `raw_text`. Both are matched back up by
    `provenance.source_region_id`, so this stays independent of item ordering.
    """
    wanted = {region["region_id"] for region in regions}
    by_region = {
        item.provenance.source_region_id: item
        for item in doc.items.values()
        if item.provenance.source_region_id in wanted
    }

    for region in regions:
        item = by_region.get(region["region_id"])
        text = texts.get(region["region_id"], "")
        if item is None or not text:
            continue
        if isinstance(item, FigureItem):
            item.alt_text = _alt_text(text)
        elif isinstance(item, TableItem):
            item.table_data = html_table_to_data(text)
            item.raw_text = text


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def document_from_pages(
    stem: str,
    pages: list[tuple[int, Path]],
    images_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> GlasanaDocument:
    """Build one `GlasanaDocument` from a document's Youtu page JSONs.

    `pages` is `[(page_no, json_path), ...]`; `collect_pages` produces it.
    Figures are cropped from `<images_dir>/<stem>_<page>.<ext>` when both that
    image and `figures_dir` are available, and keep an empty `image_path`
    otherwise — the crops are the one thing the JSON cannot supply.

    Article grouping is the same crude seed the other tracks use (a Headline
    opens an article, via `rare.parse.assemble.attach_to_article`); run
    `rare.link.link_document` over the result to resolve it properly.
    """
    doc = GlasanaDocument(source_pdf=stem)
    current_article: Optional[Article] = None

    for page_no, json_path in pages:
        try:
            items = json.loads(json_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"[warn] cannot read {json_path}: {exc}")
            continue
        if not isinstance(items, list):
            print(f"[warn] {json_path} is not a Youtu page result (expected a list)")
            continue

        image_path = _find_page_image(images_dir, stem, page_no)
        img_w, img_h = _page_size(json_path, image_path, items)
        doc.pages[page_no] = PageInfo(
            page_no=page_no,
            width=img_w,
            height=img_h,
            source_file=image_path.name if image_path else f"{stem}_{page_no}.jpg",
        )

        regions, texts = page_to_regions(
            items, _load_hierarchy_depths(json_path), img_w, img_h
        )

        page_image = None
        if image_path is not None and figures_dir is not None:
            from PIL import Image

            try:
                page_image = Image.open(image_path)
            except OSError as exc:
                print(f"[warn] cannot open {image_path}: {exc}")

        current_article = assemble_page(
            doc,
            page_no=page_no,
            regions=regions,
            texts=texts,
            img_w=img_w,
            img_h=img_h,
            figures_dir=figures_dir or Path("figures"),
            current_article=current_article,
            page_image=page_image,
        )
        _apply_region_content(doc, regions, texts)

    return doc
