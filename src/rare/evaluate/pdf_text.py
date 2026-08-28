"""PDF-backed text extraction for OmniDocBench converter `text` fields.

Wraps `rare.parse.text.extract_text_for_page` so the OmniDocBench converter
can pull real text per layout_det's bbox without re-implementing the
image-pixel → PDF-point coordinate scaling. PDFs are opened lazily and
cached by stem; call `.close()` (or use as a context manager) to release the
file handles when the run finishes.

Designed to be a drop-in `Callable[[image_path, poly, img_w, img_h], str]`
that the converter can call per layout_det.

**OCR is applied to predictions only.** With `ocr` given, `texts_for_regions`
— the call that fills predicted regions — runs `rare.parse.ocr` over the
regions the text layer failed on, exactly as the parse pipeline does. The
per-det `__call__`, which is what builds the OmniDocBench *ground truth*, never
does. That asymmetry is the whole point of being able to evaluate the effect:
one source feeding both sides would compare OCR against itself and move
Edit_dist by nothing. If the ground truth should carry corrected text too, hand
it in with `--omnidocbench-ground` — `examples/manual/audit/run.py --apply`
produces exactly that file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pdfplumber

from rare.parse.text import extract_text_for_page

logger = logging.getLogger(__name__)


class PdfTextSource:
    """Resolve `<image_path>` → PDF page → text for an axis-aligned poly.

    Image paths are assumed to follow the `<stem>_<page>.<ext>` convention used
    everywhere else in the codebase (see `src/rare/evaluate/datasets.py:182-188`
    and `src/rare/evaluate/omnidocbench.py:_page_no_from_filename`). PDFs are
    expected at `<pdfs_dir>/<stem>.pdf`. Missing PDFs return `""` for every
    box on that page; the converter then leaves `text` empty and lets
    OmniDocBench's `quick_match` ignore those boxes.
    """

    def __init__(
        self,
        pdfs_dir: Path,
        *,
        ocr=None,
        ocr_labels: Iterable[str] = (),
        ocr_retry: Iterable[str] = (),
    ):
        self.pdfs_dir = Path(pdfs_dir)
        self._cache: dict[str, Optional[pdfplumber.PDF]] = {}
        self.ocr = ocr
        self.ocr_labels = frozenset(ocr_labels)
        self.ocr_retry = tuple(ocr_retry)
        # One rendered page at a time. The converter walks pages in order, so a
        # single slot is all the reuse there is to get — and a whole dataset of
        # 400-DPI pages held at once would cost several GB.
        self._page_image_key: Optional[tuple[str, int]] = None
        self._page_image = None
        self.ocr_filled = 0

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

    @staticmethod
    def _split_stem_page(image_path: str) -> Optional[tuple[str, int]]:
        """`<stem>_<page>.<ext>` -> `(stem, page_no)`, or None if it doesn't fit."""
        name = Path(image_path).stem
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            return None
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return None

    def _resolve_page(self, image_path: str):
        """`(pdf, page_no)` for an image path, or `(None, -1)` when unresolvable."""
        split = self._split_stem_page(image_path)
        if split is None:
            return None, -1
        stem, page_no = split
        pdf = self._open(stem)
        if pdf is None or page_no >= len(pdf.pages):
            return None, -1
        return pdf, page_no

    def __call__(self, image_path: str, poly: list[float],
                 img_w: int, img_h: int) -> str:
        pdf, page_no = self._resolve_page(image_path)
        if pdf is None:
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

    def texts_for_regions(self, image_path: str, regions: list[dict],
                          img_w: int, img_h: int) -> dict[str, str]:
        """Text for a whole page of *predicted* regions, OCR included.

        This is the prediction side, and the only place OCR runs — see the
        module docstring for why the ground-truth path is left alone.
        """
        pdf, page_no = self._resolve_page(image_path)
        if pdf is None:
            return {r["region_id"]: "" for r in regions}
        texts = extract_text_for_page(pdf, page_no, regions, img_w, img_h)
        self._ocr_fill(image_path, page_no, regions, texts, img_w, img_h)
        return texts

    def _ocr_fill(self, image_path: str, page_no: int, regions: list[dict],
                  texts: dict[str, str], img_w: int, img_h: int) -> None:
        """Re-read the regions the text layer failed on. Mutates `texts`."""
        if self.ocr is None:
            return
        split = self._split_stem_page(image_path)
        if split is None:
            return
        pdf_path = self.pdfs_dir / f"{split[0]}.pdf"
        if not pdf_path.exists():
            return

        from rare.parse.ocr import fill_failed_regions

        key = (split[0], page_no)
        if key != self._page_image_key:
            # Dropped before rendering the next one, so peak memory is one page.
            self._page_image_key, self._page_image = key, None
            from rare.parse.pdf import render_page
            try:
                self._page_image = render_page(pdf_path, page_no, dpi=self.ocr.dpi)
            except Exception:
                logger.exception("could not render %s page %d for OCR",
                                 pdf_path.name, page_no)

        try:
            self.ocr_filled += fill_failed_regions(
                regions, texts,
                recognizer=self.ocr,
                pdf_path=pdf_path,
                page_no=page_no,
                page_w=img_w,
                page_h=img_h,
                labels=self.ocr_labels,
                retry=self.ocr_retry,
                page_image=self._page_image,
                page_image_dpi=self.ocr.dpi if self._page_image is not None else None,
            )
        except Exception:
            logger.exception("OCR failed on %s page %d", pdf_path.name, page_no)

    def close(self) -> None:
        self._page_image_key, self._page_image = None, None
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
