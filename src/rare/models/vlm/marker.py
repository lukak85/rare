from __future__ import annotations

import os
import re
from pathlib import Path
import traceback

from tqdm import tqdm

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register
from rare.models.vlm._assembler import assemble_document
from rare.models.vlm._vlm_schema import VLMDocument, VLMPage, VLMRegion

# Default page-break marker injected into Docling's Markdown so we can split it
# back into per-page chunks. Mirrors the placeholder `normalize_pred.py` uses.
PAGE_MARKER = re.compile(r"\n\n\{(\d+)\}-{48}\n\n")


@register("vlm", "marker")
class MarkerBackend:
    """Runs Docling over PDFs and exposes its Markdown for evaluation."""

    name = "docling"
    # parse_pdf wraps Docling's own per-page Markdown in one region per page, so
    # the runner scores it verbatim (raw join) rather than re-applying markup.
    raw_markdown = True

    def __init__(self, config: dict | None = None):
        if config is None:
            self.config=ConfigParser({
                "output_format": "markdown",
                "paginate_output": True
            }).generate_config_dict()
        else:
            self.config=ConfigParser(config).generate_config_dict()
        self._converter = None  # built lazily on first use

    def _get_converter(self):
        if self._converter is None:
            self._converter = PdfConverter(
                config=self.config,
                artifact_dict=create_model_dict(),
            )
        return self._converter

    # --- faithful reference port -------------------------------------------

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        for pdf_name in tqdm(os.listdir(pdf_dir)):
            rendered = self._get_converter()(os.path.join(pdf_dir, pdf_name))
            text, _, images = text_from_rendered(rendered)

            parts = PAGE_MARKER.split(text)

            chunks = parts[0::2]
            pages = [c.strip() for c in chunks if c.strip()]

            os.makedirs(out_md_dir, exist_ok=True)
            for i, page_md in enumerate(pages):
                path = os.path.join(out_md_dir, f"{pdf_name.split('.')[0]}_{i}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(page_md + "\n")

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