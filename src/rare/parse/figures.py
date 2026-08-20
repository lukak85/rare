"""Crop figure regions from rendered page images."""

from __future__ import annotations

from pathlib import Path

from PIL.Image import Image


def crop_region(
    page_image: Image,
    bbox_norm_1000: list[float],
    padding: int = 4,
) -> Image:
    """Crop one region from a rendered page image.

    bbox_norm_1000 is [x0, y0, x1, y1] in 0–1000 normalised image space
    (top-left origin). Because the box is normalised, the page image may be
    rendered at any DPI — which is what lets the OCR fallback re-render a page
    at a higher resolution and crop the very same region.
    """
    w, h = page_image.size
    x0 = max(0, int(bbox_norm_1000[0] / 1000 * w) - padding)
    y0 = max(0, int(bbox_norm_1000[1] / 1000 * h) - padding)
    x1 = min(w, int(bbox_norm_1000[2] / 1000 * w) + padding)
    y1 = min(h, int(bbox_norm_1000[3] / 1000 * h) + padding)
    return page_image.crop((x0, y0, x1, y1))


def crop_and_save_figure(
    page_image: Image,
    bbox_norm_1000: list[float],
    out_path: Path,
    padding: int = 4,
) -> str:
    """Crop a figure region from a rendered page image and save it.

    Returns the saved path as a string.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_region(page_image, bbox_norm_1000, padding).save(out_path)
    return str(out_path)
