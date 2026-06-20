"""Dataset loaders for evaluation.

Each loader returns an `EvalDataset` with per-page samples carrying ground
layouts and (optionally) ground reading orders, plus per-PDF ground markdown
for the VLM track.

Importatn note: in Glasbena Mladina's ground annotations, `connections.json` region IDs do NOT
match `annotations.json` COCO IDs. We match images by `file_name` and
regions by IoU (see `_matching.match_by_iou`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from pycocotools.coco import COCO

from rare.evaluate._matching import match_by_iou

if TYPE_CHECKING:
    import layoutparser as lp


@dataclass
class EvalSample:
    image_path: Path
    pdf_stem: str
    page_no: int
    image_id: int
    width: int
    height: int
    ground_layout: "lp.Layout"
    ground_order: Optional[list[int]] = None  # permutation over ground_layout


@dataclass
class EvalDataset:
    name: str
    samples: list[EvalSample]
    ground_markdown: dict[str, str] = field(default_factory=dict)  # pdf_stem → md text
    coco_path: Optional[Path] = None  # source COCO annotations file (for OmniDocBench export)
    omnidocbench_path: Optional[Path] = None  # native OmniDocBench GT JSON (used as gt.json verbatim)

    def iter_samples(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def by_pdf(self) -> dict[str, list[EvalSample]]:
        out: dict[str, list[EvalSample]] = {}
        for s in self.samples:
            out.setdefault(s.pdf_stem, []).append(s)
        for v in out.values():
            v.sort(key=lambda s: s.page_no)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coco_to_layout(coco: COCO, image_id: int) -> "lp.Layout":
    """Convert all COCO annotations for one image into an lp.Layout."""
    import layoutparser as lp
    layout = lp.Layout()
    for ann in coco.loadAnns(coco.getAnnIds([image_id])):
        x, y, w, h = ann["bbox"]
        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, x + w, y + h),
                type=coco.cats[ann["category_id"]]["name"],
                id=ann["id"],
                score=ann.get("score", 1.0),
            )
        )
    return layout


def _poly_to_rect(poly: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (x0, y0, x1, y1) of an OmniDocBench `poly`.

    OmniDocBench polys are 8-number quads [x1,y1, x2,y2, x3,y3, x4,y4]; we take
    the min/max over the x and y components so rotated/irregular quads still
    yield a usable lp.Rectangle.
    """
    xs = poly[0::2]
    ys = poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _omnidocbench_to_layout(layout_dets: list[dict]) -> "lp.Layout":
    """Convert one OmniDocBench page's `layout_dets` into an lp.Layout.

    Each block's `id` is its index within the page so that a reading-order
    permutation (from the per-det `order` field) indexes the layout directly.
    """
    import layoutparser as lp
    layout = lp.Layout()
    for idx, det in enumerate(layout_dets):
        x0, y0, x1, y1 = _poly_to_rect(det["poly"])
        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x0, y0, x1, y1),
                type=det["category_type"],
                id=idx,
                score=det.get("score", 1.0),
            )
        )
    return layout


def _omnidocbench_order(layout_dets: list[dict]) -> Optional[list[int]]:
    """Reading-order permutation over layout indices from per-det `order` fields.

    Returns indices sorted by `order` (so layout[result[k]] is the k-th region),
    or None if no det carries an `order`. Dets without an `order` are appended at
    the end in their original index order.
    """
    have_order = [i for i, d in enumerate(layout_dets) if d.get("order") is not None]
    if not have_order:
        return None
    tail = [i for i, d in enumerate(layout_dets) if d.get("order") is None]
    return sorted(have_order, key=lambda i: layout_dets[i]["order"]) + tail


def _order_id_to_order(coco: COCO, image_id: int) -> Optional[list[int]]:
    """Build a reading-order permutation from per-annotation `order_id` fields.

    Returns a list of `ground_layout` indices in reading order (so that
    ground_layout[result[k]] is the k-th region), or None if the annotations
    don't carry `order_id`. The layout index order matches `_coco_to_layout`,
    since both iterate `coco.getAnnIds([image_id])` / `loadAnns` identically.

    This is the precomputed alternative to `_connections_to_order`: the IoU
    match against the LabelStudio export has already been baked into `order_id`
    by scripts/join_annotations.py, so no matching happens at load time.
    """
    anns = coco.loadAnns(coco.getAnnIds([image_id]))
    if not anns or any("order_id" not in a for a in anns):
        return None
    return sorted(range(len(anns)), key=lambda i: anns[i]["order_id"])


def _connections_to_order(
    entry: dict,
    ground_layout: "lp.Layout",
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.3,
) -> Optional[list[int]]:
    import layoutparser as lp
    """Translate a connections.json entry's reading order into a permutation
    over `ground_layout` indices.

    Strategy: build a temporary Layout from the connections regions (in
    layoutreader-tgt order), then match each to ground_layout by IoU. The
    resulting list of matched ground indices is the ground reading order.
    Unmatched ground regions are appended at the end in their original order.
    """
    regions = entry.get("regions", [])
    tgt_index = entry.get("layoutreader", {}).get("text", {}).get("tgt_index")
    if not regions or not tgt_index:
        return None

    ordered_regions = [regions[i] for i in tgt_index if i < len(regions)]
    conn_layout = lp.Layout()
    for r in ordered_regions:
        x0n, y0n, x1n, y1n = r["bbox_norm_1000"]
        conn_layout.append(
            lp.TextBlock(
                block=lp.Rectangle(
                    x0n / 1000.0 * img_w,
                    y0n / 1000.0 * img_h,
                    x1n / 1000.0 * img_w,
                    y1n / 1000.0 * img_h,
                ),
                type=r.get("label", ""),
            )
        )

    matched = dict(match_by_iou(conn_layout, ground_layout, iou_threshold=iou_threshold))
    ground_indices_in_order = [matched[i] for i in range(len(conn_layout)) if i in matched]
    matched_set = set(ground_indices_in_order)
    tail = [j for j in range(len(ground_layout)) if j not in matched_set]
    return ground_indices_in_order + tail


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_glasbena_mladina(
    root: str | Path = "datasets/glasbena_mladina",
    images_dir: str | Path | None = None,
    pdfs_dir: str | Path | None = None,
    ground_markdown_dir: str | Path | None = None,
    annotations_file: str | Path | None = None,
    omnidocbench_file: str | Path | None = None,
) -> EvalDataset:
    """Load the Glasbena Mladina annotated dataset.

    Defaults to the existing layout (`dataset/annotations.json`,
    `dataset/connections.json`, images alongside). Pass `images_dir` if your
    image files live elsewhere.

    Reading order is taken from per-annotation `order_id` fields when present
    (the precomputed join from scripts/join_annotations.py); otherwise it falls
    back to IoU-matching `connections.json` at load time. `annotations_file`
    overrides which COCO file to load; by default `annotations_with_order.json`
    is preferred over `annotations.json` when it exists in `root`.

    `omnidocbench_file` points at a pre-built native OmniDocBench GT JSON; the
    runner then uses it as `gt.json` verbatim (no COCO→OmniDocBench conversion
    or runtime PDF text extraction), exactly like the `omnidocbench` dataset.
    When omitted, the conventional `<root>/omnidocbench/omnidocbench.json` is
    auto-detected and used if present.
    """
    root = Path(root)
    if annotations_file is not None:
        ann_path = Path(annotations_file)
        if not ann_path.is_absolute():
            ann_path = root / ann_path
    else:
        enriched = root / "annotations_with_order.json"
        ann_path = enriched if enriched.exists() else root / "annotations.json"
    conn_path = root / "connections.json"
    coco = COCO(str(ann_path))

    # Native OmniDocBench GT (optional): explicit path, else the conventional
    # location. Used verbatim by the runner when present.
    if omnidocbench_file is not None:
        odb_path = Path(omnidocbench_file)
        if not odb_path.is_absolute():
            odb_path = root / odb_path
    else:
        odb_path = root / "omnidocbench" / "omnidocbench.json"
    odb_path = odb_path if odb_path.exists() else None

    connections = []
    if conn_path.exists():
        connections = json.loads(conn_path.read_text())
    conn_by_filename = {Path(e["image"]).name: e for e in connections}

    img_root = Path(images_dir) if images_dir else root
    pdf_root = Path(pdfs_dir) if pdfs_dir else root # TODO
    samples: list[EvalSample] = []
    for image_id, info in coco.imgs.items():
        file_name = info["file_name"]
        # Extract pdf_stem and page_no from filename "<stem>_<page>.jpg"
        stem_parts = file_name.rsplit("_", 1)
        pdf_stem = stem_parts[0] if len(stem_parts) == 2 else file_name
        try:
            page_no = int(stem_parts[1].rsplit(".", 1)[0])
        except (IndexError, ValueError):
            page_no = 0

        ground_layout = _coco_to_layout(coco, image_id)
        # Prefer the precomputed permutation in `order_id`; fall back to the
        # connections.json IoU match only when annotations lack order_id.
        ground_order = _order_id_to_order(coco, image_id)
        if ground_order is None:
            conn_entry = conn_by_filename.get(file_name)
            ground_order = (
                _connections_to_order(conn_entry, ground_layout, info["width"], info["height"])
                if conn_entry
                else None
            )

        samples.append(EvalSample(
            image_path=img_root / file_name,
            pdf_stem=pdf_stem,
            page_no=page_no,
            image_id=image_id,
            width=info["width"],
            height=info["height"],
            ground_layout=ground_layout,
            ground_order=ground_order,
        ))

    # Optional ground markdown for the VLM track
    ground_md: dict[str, str] = {}
    md_dir = Path(ground_markdown_dir) if ground_markdown_dir else (root / "ground")
    if md_dir.exists():
        for f in md_dir.glob("*.md"):
            ground_md[f.stem] = f.read_text()

    return EvalDataset(
        name="glasbena_mladina",
        samples=samples,
        ground_markdown=ground_md,
        coco_path=ann_path,
        omnidocbench_path=odb_path,
    )


def load_doclaynet(
    root: str | Path = "datasets/doclaynet",
    split: str = "val",
) -> EvalDataset:
    """Load the DocLayNet COCO split. Looks for `COCO/<split>.json` and
    `PNG/` images under `root`.
    """
    root = Path(root)
    ann_path = root / "COCO" / f"{split}.json"
    if not ann_path.exists():
        # Fallbacks
        candidates = list((root / "COCO").glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No COCO JSON found under {root / 'COCO'}")
        ann_path = candidates[0]

    coco = COCO(str(ann_path))
    img_root = root / "PNG"
    samples: list[EvalSample] = []
    for image_id, info in coco.imgs.items():
        samples.append(EvalSample(
            image_path=img_root / info["file_name"],
            pdf_stem=Path(info["file_name"]).stem,
            page_no=0,
            image_id=image_id,
            width=info["width"],
            height=info["height"],
            ground_layout=_coco_to_layout(coco, image_id),
            ground_order=None,  # DocLayNet has no reading-order labels
        ))
    return EvalDataset(name=f"doclaynet/{ann_path.stem}", samples=samples, coco_path=ann_path)

def load_publaynet(
    root: str | Path = "datasets/publaynet",
    split: str = "val",
) -> EvalDataset:
    """Load the PubLayNet COCO split. Looks for `<split>.json` and
    `PNG/` images under `root`.
    """
    root = Path(root)
    ann_path = root / f"{split}.json"
    if not ann_path.exists():
        # Fallbacks
        candidates = list(root.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No COCO JSON found under {root}")
        ann_path = candidates[0]

    coco = COCO(str(ann_path))
    img_root = root / "samples"
    samples: list[EvalSample] = []
    for image_id, info in coco.imgs.items():
        samples.append(EvalSample(
            image_path=img_root / info["file_name"],
            pdf_stem=Path(info["file_name"]).stem,
            page_no=0,
            image_id=image_id,
            width=info["width"],
            height=info["height"],
            ground_layout=_coco_to_layout(coco, image_id),
            ground_order=None,  # DocLayNet has no reading-order labels
        ))
    return EvalDataset(name=f"publaynet/{ann_path.stem}", samples=samples, coco_path=ann_path)


def load_omnidocbench(
    root: str | Path = "OmniDocBench-data",
    annotations_file: str | Path | None = None,
    images_dir: str | Path | None = None,
    drop_ignore: bool = False,
) -> EvalDataset:
    """Load the OmniDocBench dataset in its native JSON format.

    Unlike the COCO-based loaders this reads OmniDocBench's own per-page schema
    (`OmniDocBench.json`): a JSON array where each entry has `layout_dets`
    (quad `poly` + `category_type` + `order` + `text`/`latex`/`html`) and a
    `page_info` block (`page_no`, `width`, `height`, `image_path`). Each entry
    is one page; the image stem is used as `pdf_stem`.

    `category_type` values are kept verbatim (they already match the
    OmniDocBench vocabulary produced by `rare.evaluate.omnidocbench`), and the
    per-det `order` field becomes `ground_order`. Set `drop_ignore=True` to skip
    dets flagged `ignore` (e.g. mask regions). `coco_path` is left None since
    the ground truth is already in OmniDocBench shape.
    """
    root = Path(root)
    ann_path = Path(annotations_file) if annotations_file else root / "OmniDocBench.json"
    if not ann_path.is_absolute():
        ann_path = ann_path if ann_path.exists() else root / ann_path
    entries = json.loads(Path(ann_path).read_text())

    img_root = Path(images_dir) if images_dir else root / "images"
    samples: list[EvalSample] = []
    for image_id, entry in enumerate(entries):
        page_info = entry["page_info"]
        dets = entry["layout_dets"]
        if drop_ignore:
            dets = [d for d in dets if not d.get("ignore")]

        file_name = Path(page_info["image_path"]).name
        samples.append(EvalSample(
            image_path=img_root / file_name,
            pdf_stem=Path(file_name).stem,
            page_no=page_info.get("page_no", 0),
            image_id=image_id,
            width=page_info["width"],
            height=page_info["height"],
            ground_layout=_omnidocbench_to_layout(dets),
            ground_order=_omnidocbench_order(dets),
        ))

    return EvalDataset(
        name="omnidocbench",
        samples=samples,
        omnidocbench_path=Path(ann_path),
    )


DATASETS = {
    "glasbena_mladina": load_glasbena_mladina,
    "doclaynet":        load_doclaynet,
    "publaynet":        load_publaynet,
    "omnidocbench":     load_omnidocbench,
}


def load(name: str, **kwargs) -> EvalDataset:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)
