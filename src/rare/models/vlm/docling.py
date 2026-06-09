# Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench),
# tools/model_infer/docling_img2md.py. Copyright (c) 2024 OpenDataLab and the
# OmniDocBench authors. Licensed under the Apache License, Version 2.0; see
# licenses/LICENSE-OMNIDOCBENCH and the NOTICE file.
"""Docling backend.

Docling (https://github.com/docling-project/docling) parses a whole PDF into a
structured DoclingDocument and exports Markdown via `export_to_markdown()`.
Unlike MinerU's per-block VLM, Docling is a black-box document converter — we
treat its Markdown as the faithful output and do NOT re-derive layout regions
from it. Two entry points share one converter:

- `pdfs_to_markdown(...)` — faithful port of OmniDocBench's `docling_img2md.py`:
  a folder of PDFs -> one `<stem>.md` per PDF (whole-document Markdown). Used
  with the standalone `normalize_pred.py` (`--split-on`) + `run.sh` path.
- `parse_pdf(pdf) -> GlasanaDocument` — the `evaluate/runner.py::run_vlm`
  contract. Splits Docling's Markdown per page via `export_to_markdown`'s
  `page_break_placeholder`, then wraps each page's Markdown in a single region.
  Because `raw_markdown = True`, the runner joins region text verbatim, so the
  per-page prediction reproduces Docling's own Markdown exactly for OmniDocBench.

Install Docling separately on the inference box::

    pip install -U docling
"""

from __future__ import annotations

import os
from pathlib import Path
import traceback

from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion

# Default page-break marker injected into Docling's Markdown so we can split it
# back into per-page chunks. Mirrors the placeholder `normalize_pred.py` uses.
_DEFAULT_PAGE_BREAK = "<!-- PAGE -->"


@register("vlm", "docling")
class DoclingBackend:
    """Runs Docling over PDFs and exposes its Markdown for evaluation."""

    name = "docling"
    # parse_pdf wraps Docling's own per-page Markdown in one region per page, so
    # the runner scores it verbatim (raw join) rather than re-applying markup.
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.page_break_marker = cfg.get("page_break_marker", _DEFAULT_PAGE_BREAK)
        self._converter = None  # built lazily on first use

    def _get_converter(self):
        """Build the DocumentConverter on first use (lazy import keeps
        registration cheap on boxes without `docling` installed)."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    # --- faithful reference port -------------------------------------------

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        for img_name in tqdm(os.listdir(pdf_dir)):
            if not img_name.endswith('.pdf'):
                continue

            img_name = img_name.strip()

            save_result_path = os.path.join(out_md_dir, img_name[:-4] + '.md')

            if os.path.exists(save_result_path):
                continue

            img_path_tmp = os.path.join(pdf_dir, img_name)
            try:
                result = self._get_converter().convert(img_path_tmp)
                result_md = result.document.export_to_markdown(page_break_placeholder="<!-- PAGE -->")
            except Exception:
                print(traceback.format_exc())
                continue

            with open(save_result_path, 'w', encoding='utf-8') as output_file:
                output_file.write(result_md)

            from .helpers.normalize_pred import split_single_markdown

            text_pages = split_single_markdown(Path(save_result_path), self.page_break_marker)

            for stem, content in sorted(text_pages.items()):
                (Path(out_md_dir) / f"{stem}.md").write_text(content)

            # Remove the all-including markdown
            os.remove(save_result_path)


        """Convert every PDF under `pdf_dir` and write `<stem>.md` (whole-document
        Markdown) into `out_md_dir`, matching `docling_img2md.py`."""
        """
        os.makedirs(out_md_dir, exist_ok=True)
        pdfs = sorted(p for p in Path(pdf_dir).iterdir() if p.suffix.lower() == ".pdf")
        print(f"found {len(pdfs)} PDF files.")
        converter = self._get_converter()
        for pdf in tqdm(pdfs, desc="docling"):
            out_path = Path(out_md_dir) / f"{pdf.stem}.md"
            if skip_existing and out_path.exists():
                continue
            try:
                result = converter.convert(str(pdf))
                out_path.write_text(
                    result.document.export_to_markdown(), encoding="utf-8"
                )
            except Exception as exc:  # mirror the reference's continue-on-failure
                print(f"failed {pdf.name}: {exc}")
        """
        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def _markdown_pages(self, pdf_path: str | Path) -> list[str]:
        """Return Docling's Markdown split into one string per page.

        Docling inserts `page_break_marker` between pages; splitting on it keeps
        page indices aligned with the document (empty chunks are preserved so a
        blank page still produces an aligned, empty prediction)."""
        result = self._get_converter().convert(str(pdf_path))
        full_md = result.document.export_to_markdown(
            page_break_placeholder=self.page_break_marker
        )
        return [chunk.strip("\n") for chunk in full_md.split(self.page_break_marker)]

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Convert `pdf_path` with Docling and wrap each page's Markdown in a
        single region. The runner renders this verbatim (raw_markdown) into the
        per-page Markdown OmniDocBench scores."""
        pdf_path = Path(pdf_path)
        pages = [
            VLMPage(page_no=page_no, regions=[VLMRegion(label="Paragraph", text=md)])
            for page_no, md in enumerate(self._markdown_pages(pdf_path))
        ]
        return assemble_document(VLMDocument(pages=pages), pdf_path.stem)