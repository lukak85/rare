"""Per-region text extraction via pdfplumber.

Lifted from build_doc._extract_text_for_page so the parse pipeline and any
future evaluation flows share one implementation.
"""

from __future__ import annotations

import pdfplumber


def extract_text_for_page(
    pdf: pdfplumber.PDF,
    page_no: int,
    regions: list[dict],
    img_width: float,
    img_height: float,
) -> dict[str, str]:
    """Return {region_id: text} for all regions on a page.

    Each region must carry a `bbox_norm_1000` field — [x0, y0, x1, y1] in
    0–1000 normalised image space (top-left origin). The function scales
    those coordinates to PDF point space (which may differ from the rendered
    image dimensions) before cropping.
    """
    pdf_page = pdf.pages[page_no]
    pw, ph = pdf_page.width, pdf_page.height
    results: dict[str, str] = {}

    for region in regions:
        x0n, y0n, x1n, y1n = region["bbox_norm_1000"]
        x0 = x0n / 1000.0 * pw
        x1 = x1n / 1000.0 * pw
        y0_pdf = y0n / 1000.0 * ph
        y1_pdf = y1n / 1000.0 * ph
        try:
            cropped = pdf_page.crop((x0, y0_pdf, x1, y1_pdf), strict=False)
            results[region["region_id"]] = (cropped.extract_text() or "").strip()
        except Exception:
            results[region["region_id"]] = ""

    return results
