# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/PaddleOCR_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""PaddleOCR (PP-StructureV3) backend.

PP-StructureV3 (https://github.com/PaddlePaddle/PaddleOCR) is a local document
pipeline (layout + OCR + table). For each page it produces a result whose
`save_to_markdown` writes the page's Markdown. That formatter output is the
faithful prediction OmniDocBench compares, so — like Docling/dots.ocr — we treat
it as a black box: wrap each page's Markdown in a single region. Two entry points
share one pipeline:

- `to_markdown(pdf_dir, image_dir, out_md_dir, ...)` — faithful port of
  OmniDocBench's `PaddleOCR_img2md.py`: a folder of page images -> one
  `<stem>.md` per image. Used with the standalone `normalize_pred.py` + `run.sh`
  path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract. PP-StructureV3 rasterizes the PDF itself (one result per page); we
  wrap each page's Markdown in one region. Because `raw_markdown = True`, the
  runner joins region text verbatim, reproducing PP-StructureV3's own Markdown.

Install on the inference box::

    pip install -U paddleocr paddlepaddle   # (or paddlepaddle-gpu)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from paddleocr import PPStructureV3
from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


@register("vlm", "paddleocr")
class PaddleOCRBackend:
    """Runs PP-StructureV3 and exposes its Markdown for evaluation."""

    name = "paddleocr"
    # parse_pdf wraps PP-StructureV3's own per-page Markdown in one region per
    # page, so the runner scores it verbatim (raw join).
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._pipeline_kwargs = dict(cfg.get("pipeline_kwargs") or {})
        # Predict-time flags; defaults mirror PaddleOCR_img2md.py.
        self.predict_flags = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            **(cfg.get("predict_flags") or {}),
        }
        self._pipeline = None  # built lazily on first use

    def _get_pipeline(self) -> PPStructureV3:
        if self._pipeline is None:
            self._pipeline = PPStructureV3(**self._pipeline_kwargs)
        return self._pipeline

    # --- inference ---------------------------------------------------------

    @staticmethod
    def _result_to_markdown(res) -> str:
        """Get one result's Markdown via `save_to_markdown` into a temp dir, then
        read it back — byte-faithful to the reference and robust to whether
        PaddleX writes a flat `<base>.md` or a per-page bundle subfolder."""
        with tempfile.TemporaryDirectory(prefix="rare-paddle-") as td:
            base = os.path.join(td, "page")
            res.save_to_markdown(base, pretty=False)
            direct = Path(base + ".md")
            if direct.exists():
                return direct.read_text(encoding="utf-8")
            mds = sorted(Path(td).rglob("*.md"))
            return mds[0].read_text(encoding="utf-8") if mds else ""

    def _predict_markdowns(self, source: str | Path) -> list[str]:
        """Run PP-StructureV3 on a PDF or image path; return per-page Markdown in
        order (one result per page)."""
        results = self._get_pipeline().predict(str(source), **self.predict_flags)
        return [self._result_to_markdown(res) for res in results]

    # --- faithful reference port -------------------------------------------

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        """Convert every page image under `image_dir` and write `<stem>.md` into
        `out_md_dir` (one per image), matching `PaddleOCR_img2md.py`. `pdf_dir` is
        unused (PaddleOCR scores per page image) but kept for signature parity."""
        os.makedirs(out_md_dir, exist_ok=True)
        paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        todo = [
            p for p in paths
            if not (skip_existing and (Path(out_md_dir) / f"{p.stem}.md").exists())
        ]
        print(f"found {len(paths)} images; processing {len(todo)}.")
        for path in tqdm(todo, desc="paddleocr"):
            mds = self._predict_markdowns(path)
            (Path(out_md_dir) / f"{path.stem}.md").write_text(
                "\n\n".join(m for m in mds if m), encoding="utf-8"
            )
        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Run PP-StructureV3 over `pdf_path` (it rasterizes the PDF itself) and
        wrap each page's Markdown in a single region for the runner's verbatim
        (raw) scoring."""
        pdf_path = Path(pdf_path)
        pages = [
            VLMPage(page_no=page_no, regions=[VLMRegion(label="Paragraph", text=md)])
            for page_no, md in enumerate(self._predict_markdowns(pdf_path))
        ]
        return assemble_document(VLMDocument(pages=pages), pdf_path.stem)