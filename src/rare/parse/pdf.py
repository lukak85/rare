"""PDF → page image rendering."""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL.Image import Image


def render_pages(pdf_path: str | Path, dpi: int = 200) -> list[Image]:
    """Render all pages of a PDF as PIL images at the given DPI."""
    return convert_from_path(str(pdf_path), dpi=dpi)


def render_page(pdf_path: str | Path, page_no: int, dpi: int = 200) -> Image:
    """Render a single page (0-indexed) as a PIL image."""
    pages = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_no + 1, last_page=page_no + 1
    )
    return pages[0]
