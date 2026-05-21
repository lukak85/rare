"""Crop figure regions from rendered page images."""

from __future__ import annotations

from pathlib import Path

from PIL.Image import Image


def crop_and_save_figure(
    page_image: Image,
    bbox_norm_1000: list[float],
    out_path: Path,
    padding: int = 4,
) -> str:
    """Crop a figure region from a rendered page image and save it.

    bbox_norm_1000 is [x0, y0, x1, y1] in 0–1000 normalised image space
    (top-left origin). Returns the saved path as a string.
    """
    w, h = page_image.size
    x0 = max(0, int(bbox_norm_1000[0] / 1000 * w) - padding)
    y0 = max(0, int(bbox_norm_1000[1] / 1000 * h) - padding)
    x1 = min(w, int(bbox_norm_1000[2] / 1000 * w) + padding)
    y1 = min(h, int(bbox_norm_1000[3] / 1000 * h) + padding)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_image.crop((x0, y0, x1, y1)).save(out_path)
    return str(out_path)
