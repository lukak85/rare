"""PDF-backed text extraction for OmniDocBench converter `text` fields.

Wraps `rare.parse.text.extract_text_for_page` so the OmniDocBench converter
can pull real text per layout_det's bbox without re-implementing the
image-pixel → PDF-point coordinate scaling. PDFs are opened lazily and
cached by stem; call `.close()` (or use as a context manager) to release the
file handles when the run finishes.

Designed to be a drop-in `Callable[[image_path, poly, img_w, img_h], str]`
that the converter can call per layout_det.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pdfplumber

from rare.parse.text import extract_text_for_page


class PdfTextSource:
    """Resolve `<image_path>` → PDF page → text for an axis-aligned poly.

    Image paths are assumed to follow the `<stem>_<page>.<ext>` convention used
    everywhere else in the codebase (see `src/rare/evaluate/datasets.py:182-188`
    and `src/rare/evaluate/omnidocbench.py:_page_no_from_filename`). PDFs are
    expected at `<pdfs_dir>/<stem>.pdf`. Missing PDFs return `""` for every
    box on that page; the converter then leaves `text` empty and lets
    OmniDocBench's `quick_match` ignore those boxes.
    """

    def __init__(self, pdfs_dir: Path):
        self.pdfs_dir = Path(pdfs_dir)
        self._cache: dict[str, Optional[pdfplumber.PDF]] = {}

    def _open(self, stem: str) -> Optional[pdfplumber.PDF]:
        if stem in self._cache:
            return self._cache[stem]
        pdf_path = self.pdfs_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            self._cache[stem] = None
            return None
        try:
            pdf = pdfplumber.open(str(pdf_path))
        except Exception:
            pdf = None
        self._cache[stem] = pdf
        return pdf

    def __call__(self, image_path: str, poly: list[float],
                 img_w: int, img_h: int) -> str:
        name = Path(image_path).stem
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            return ""
        stem = parts[0]
        try:
            page_no = int(parts[1])
        except ValueError:
            return ""

        pdf = self._open(stem)
        if pdf is None or page_no >= len(pdf.pages):
            return ""

        # poly is 8 axis-aligned coords; convert to `bbox_norm_1000` (the shape
        # `extract_text_for_page` expects).
        xs = poly[0::2]
        ys = poly[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bbox_norm = [
            x0 / img_w * 1000.0,
            y0 / img_h * 1000.0,
            x1 / img_w * 1000.0,
            y1 / img_h * 1000.0,
        ]
        result = extract_text_for_page(
            pdf, page_no,
            [{"region_id": "x", "bbox_norm_1000": bbox_norm}],
            img_w, img_h,
        )
        return result.get("x", "")

    def close(self) -> None:
        for pdf in self._cache.values():
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
        self._cache.clear()

    def __enter__(self) -> "PdfTextSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
